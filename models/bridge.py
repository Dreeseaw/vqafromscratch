"""
Bridge modules for multimodal training.

Role:
- Define a small, explicit boundary from visual features to LM prefix tokens.
- Keep the first implementation very simple (MLP bridge) with clear extension points.
"""
from __future__ import annotations

from dataclasses import dataclass

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
        self.cfg = cfg
        h = int(cfg.bridge_hidden_dim)
        self._h = h
        self._input_dim: int | None = None
        # Build on first forward with observed Dv to avoid UninitializedParameter issues.
        self._global_proj: nn.Module | None = None
        self._token_proj: nn.Module | None = None

    def _ensure_built(self, dv: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._input_dim is not None:
            if int(dv) != int(self._input_dim):
                raise ValueError(f"Bridge input dim changed from {self._input_dim} to {dv}")
            return
        self._input_dim = int(dv)
        k = int(self.cfg.num_visual_tokens)
        d_lm = int(self.cfg.lm_hidden_size)
        self._global_proj = nn.Sequential(
            nn.Linear(self._input_dim, self._h),
            nn.GELU(),
            nn.Linear(self._h, k * d_lm),
        ).to(device=device, dtype=dtype)
        self._token_proj = nn.Sequential(
            nn.Linear(self._input_dim, self._h),
            nn.GELU(),
            nn.Linear(self._h, d_lm),
        ).to(device=device, dtype=dtype)

    def _from_global(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        k = int(self.cfg.num_visual_tokens)
        d_lm = int(self.cfg.lm_hidden_size)
        self._ensure_built(int(x.shape[-1]), x.device, x.dtype)
        assert self._global_proj is not None
        out = self._global_proj(x)
        return out.view(b, k, d_lm)

    def _from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        k = int(self.cfg.num_visual_tokens)
        self._ensure_built(int(x.shape[-1]), x.device, x.dtype)
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


def build_bridge(cfg: BridgeConfig) -> nn.Module:
    if cfg.bridge_type != "mlp":
        raise ValueError(f"Unsupported bridge_type={cfg.bridge_type}. Supported: mlp")
    return MLPVisualBridge(cfg)
