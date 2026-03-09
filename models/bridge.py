"""
Bridge modules for multimodal training.

Role:
- Define a small, explicit boundary from visual features to LM prefix tokens.
- Keep the first implementation very simple (MLP bridge) with clear extension points.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BridgeConfig:
    bridge_type: str = "mlp"
    num_visual_tokens: int = 8
    lm_hidden_size: int = 768
    bridge_hidden_dim: int = 1024
    input_feature_mode: str = "auto"  # auto | global | token
    token_reduce: str = "adaptive_pool"  # adaptive_pool | mean_expand | all
    learned_init_std: float = 0.02
    add_2d_pos_emb: bool = False


class MLPVisualBridge(nn.Module):
    """
    Supports visual feature inputs:
    - [B, Dv]      (global embedding)
    - [B, Nv, Dv]  (token sequence)
    Outputs:
    - [B, K, D_lm]
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.requires_visual_features = True
        self.cfg = cfg
        h = int(cfg.bridge_hidden_dim)
        self._h = h
        self._input_dim: int | None = None
        # Build only needed branch on first use with observed Dv.
        self._global_proj: nn.Module | None = None
        self._token_proj: nn.Module | None = None
        self._pos_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}

    @staticmethod
    def _build_1d_sincos(length: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if dim <= 0:
            return torch.zeros((length, 0), device=device, dtype=dtype)
        half = dim // 2
        if half == 0:
            return torch.zeros((length, dim), device=device, dtype=dtype)
        omega = torch.arange(half, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / max(1, half)))
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        out = torch.cat([torch.sin(pos * omega), torch.cos(pos * omega)], dim=1)
        if out.shape[1] < dim:
            out = F.pad(out, (0, dim - out.shape[1]))
        elif out.shape[1] > dim:
            out = out[:, :dim]
        return out.to(dtype=dtype)

    def _token_pos_emb(self, nv: int, dv: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(nv), int(dv), str(device), str(dtype))
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached

        side = int(math.isqrt(int(nv)))
        if side * side == int(nv):
            h, w = side, side
        else:
            h, w = 1, int(nv)

        dy = dv // 2
        dx = dv - dy
        ey = self._build_1d_sincos(h, dy, device=device, dtype=dtype)
        ex = self._build_1d_sincos(w, dx, device=device, dtype=dtype)
        gy = torch.arange(h, device=device, dtype=torch.long).unsqueeze(1).expand(h, w).reshape(-1)
        gx = torch.arange(w, device=device, dtype=torch.long).unsqueeze(0).expand(h, w).reshape(-1)
        emb = torch.cat([ey[gy], ex[gx]], dim=1)
        if emb.shape[0] != int(nv):
            emb = emb[: int(nv)]
        self._pos_cache[key] = emb
        return emb

    def _ensure_input_dim(self, dv: int) -> None:
        if self._input_dim is not None:
            if int(dv) != int(self._input_dim):
                raise ValueError(f"Bridge input dim changed from {self._input_dim} to {dv}")
        else:
            self._input_dim = int(dv)

    def _ensure_global_built(self, dv: int, device: torch.device, dtype: torch.dtype) -> None:
        self._ensure_input_dim(dv)
        if self._global_proj is not None:
            return
        assert self._input_dim is not None
        k = int(self.cfg.num_visual_tokens)
        d_lm = int(self.cfg.lm_hidden_size)
        self._global_proj = nn.Sequential(
            nn.Linear(self._input_dim, self._h),
            nn.GELU(),
            nn.Linear(self._h, k * d_lm),
        ).to(device=device, dtype=dtype)

    def _ensure_token_built(self, dv: int, device: torch.device, dtype: torch.dtype) -> None:
        self._ensure_input_dim(dv)
        if self._token_proj is not None:
            return
        assert self._input_dim is not None
        d_lm = int(self.cfg.lm_hidden_size)
        self._token_proj = nn.Sequential(
            nn.Linear(self._input_dim, self._h),
            nn.GELU(),
            nn.Linear(self._h, d_lm),
        ).to(device=device, dtype=dtype)

    def _ensure_built(self, dv: int, device: torch.device, dtype: torch.dtype) -> None:
        # Backward-compat shim for callers that expect full materialization.
        self._ensure_global_built(dv, device, dtype)
        self._ensure_token_built(dv, device, dtype)

    def _from_global(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        k = int(self.cfg.num_visual_tokens)
        d_lm = int(self.cfg.lm_hidden_size)
        self._ensure_global_built(int(x.shape[-1]), x.device, x.dtype)
        assert self._global_proj is not None
        out = self._global_proj(x)
        return out.view(b, k, d_lm)

    def _from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        k = int(self.cfg.num_visual_tokens)
        if bool(self.cfg.add_2d_pos_emb):
            x = x + self._token_pos_emb(int(x.shape[1]), int(x.shape[2]), device=x.device, dtype=x.dtype).unsqueeze(0)
        self._ensure_token_built(int(x.shape[-1]), x.device, x.dtype)
        assert self._token_proj is not None
        x = self._token_proj(x)  # [B, Nv, D_lm]
        nv = int(x.shape[1])

        # Hotpath: if token count already matches desired prefix count, no reduction needed.
        if nv == k:
            return x

        if self.cfg.token_reduce == "all":
            raise ValueError(
                f"token_reduce='all' requires Nv == K, but got Nv={nv}, K={k}. "
                "Set --num_visual_tokens to the visual token count or switch reduce mode."
            )
        if self.cfg.token_reduce == "mean_expand":
            return x.mean(dim=1, keepdim=True).expand(-1, k, -1)
        # Default: smooth down/up-sample token count to K.
        return F.adaptive_avg_pool1d(x.transpose(1, 2), output_size=k).transpose(1, 2)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim == 2:
            if self.cfg.input_feature_mode not in ("auto", "global"):
                raise ValueError(
                    f"Bridge expected token features for mode={self.cfg.input_feature_mode}, got shape={tuple(visual_features.shape)}"
                )
            return self._from_global(visual_features)

        if visual_features.ndim == 3:
            if self.cfg.input_feature_mode not in ("auto", "token"):
                raise ValueError(
                    f"Bridge expected global features for mode={self.cfg.input_feature_mode}, got shape={tuple(visual_features.shape)}"
                )
            return self._from_tokens(visual_features)

        raise ValueError(
            f"Unsupported visual feature shape {tuple(visual_features.shape)}. Expected [B,D] or [B,N,D]."
        )


class LearnedVisualTokensBridge(nn.Module):
    """
    Image-agnostic baseline: learn K visual prefix tokens directly.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.requires_visual_features = False
        self.cfg = cfg
        k = int(cfg.num_visual_tokens)
        d_lm = int(cfg.lm_hidden_size)
        std = float(cfg.learned_init_std)
        self._tokens = nn.Parameter(torch.randn(1, k, d_lm) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        return self._tokens.expand(b, -1, -1)


def build_bridge(cfg: BridgeConfig) -> nn.Module:
    if cfg.bridge_type == "mlp":
        return MLPVisualBridge(cfg)
    if cfg.bridge_type == "learned_tokens":
        return LearnedVisualTokensBridge(cfg)
    raise ValueError(f"Unsupported bridge_type={cfg.bridge_type}. Supported: mlp, learned_tokens")
