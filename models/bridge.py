"""
Bridge modules for multimodal training.

Role:
- Define the visual-to-language interface for multimodal prefix tuning.
- Keep simple baselines (`mlp`, `learned_tokens`) and provide richer reducers
  (`learned_query`, `perceiver_resampler`, `qformer_lite`, `hybrid_const_image`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    bridge_num_heads: int = 8
    bridge_attn_dropout: float = 0.0
    bridge_query_depth: int = 2
    bridge_refine_layers: int = 0
    bridge_pre_mixer_type: str = "none"  # none | self_attn | conv1d
    bridge_pre_mixer_layers: int = 1
    bridge_pre_mixer_kernel_size: int = 3
    bridge_hybrid_alpha_mode: str = "scalar"  # scalar | token
    bridge_hybrid_alpha_init: float = 0.5
    bridge_hybrid_image_bridge_type: str = "learned_query"
    bridge_question_conditioning: bool = False
    bridge_qcond_scale: float = 0.5
    bridge_token_selector_type: str = "none"  # none | topk
    bridge_token_select_k: int = 0


def _resolve_num_heads(dim: int, requested: int) -> int:
    d = max(1, int(dim))
    h = max(1, min(int(requested), d))
    while h > 1 and (d % h) != 0:
        h -= 1
    return h


def _as_token_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected [B,D] or [B,N,D], got shape={tuple(x.shape)}")


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


def _build_token_2d_pos_emb(
    nv: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    side = int(math.isqrt(int(nv)))
    if side * side == int(nv):
        h, w = side, side
    else:
        h, w = 1, int(nv)

    dy = dim // 2
    dx = dim - dy
    ey = _build_1d_sincos(h, dy, device=device, dtype=dtype)
    ex = _build_1d_sincos(w, dx, device=device, dtype=dtype)
    gy = torch.arange(h, device=device, dtype=torch.long).unsqueeze(1).expand(h, w).reshape(-1)
    gx = torch.arange(w, device=device, dtype=torch.long).unsqueeze(0).expand(h, w).reshape(-1)
    emb = torch.cat([ey[gy], ex[gx]], dim=1)
    if emb.shape[0] != int(nv):
        emb = emb[: int(nv)]
    return emb


class _SelfAttnFFNBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, attn_dropout: float, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(int(dim))
        self.self_attn = nn.MultiheadAttention(
            int(dim),
            num_heads=_resolve_num_heads(int(dim), int(num_heads)),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(int(dim))
        h = max(4, int(dim) * int(mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(int(dim), h),
            nn.GELU(),
            nn.Linear(h, int(dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        attn_out, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class _CrossAttnFFNBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, attn_dropout: float, mlp_ratio: int = 4):
        super().__init__()
        self.ln_q = nn.LayerNorm(int(dim))
        self.ln_kv = nn.LayerNorm(int(dim))
        self.cross_attn = nn.MultiheadAttention(
            int(dim),
            num_heads=_resolve_num_heads(int(dim), int(num_heads)),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(int(dim))
        h = max(4, int(dim) * int(mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(int(dim), h),
            nn.GELU(),
            nn.Linear(h, int(dim)),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qq = self.ln_q(q)
        kvn = self.ln_kv(kv)
        attn_out, _ = self.cross_attn(qq, kvn, kvn, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.ln2(q))
        return q


class _TokenConvMixerBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        k = max(1, int(kernel_size))
        if (k % 2) == 0:
            k += 1
        self.ln = nn.LayerNorm(int(dim))
        self.dw = nn.Conv1d(int(dim), int(dim), kernel_size=k, padding=k // 2, groups=int(dim), bias=False)
        self.pw = nn.Conv1d(int(dim), int(dim), kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x).transpose(1, 2)
        y = self.dw(y)
        y = F.gelu(y)
        y = self.pw(y).transpose(1, 2)
        return x + y


class _SpatialMixer(nn.Module):
    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        mixer_type = str(cfg.bridge_pre_mixer_type)
        layers = max(0, int(cfg.bridge_pre_mixer_layers))
        d = int(cfg.lm_hidden_size)
        if mixer_type == "none" or layers == 0:
            self.blocks = nn.ModuleList()
            return
        if mixer_type == "self_attn":
            self.blocks = nn.ModuleList(
                [
                    _SelfAttnFFNBlock(
                        d,
                        num_heads=int(cfg.bridge_num_heads),
                        attn_dropout=float(cfg.bridge_attn_dropout),
                    )
                    for _ in range(layers)
                ]
            )
            return
        if mixer_type == "conv1d":
            self.blocks = nn.ModuleList(
                [_TokenConvMixerBlock(d, kernel_size=int(cfg.bridge_pre_mixer_kernel_size)) for _ in range(layers)]
            )
            return
        raise ValueError(
            f"Unsupported bridge_pre_mixer_type={mixer_type}. "
            "Supported: none, self_attn, conv1d"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class _QueryBridgeBase(nn.Module):
    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.cfg = cfg
        self.requires_visual_features = True
        self._pos_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}
        self.visual_proj = nn.LazyLinear(int(cfg.lm_hidden_size))
        self.spatial_mixer = _SpatialMixer(cfg)
        selector_type = str(getattr(cfg, "bridge_token_selector_type", "none"))
        selector_k = int(getattr(cfg, "bridge_token_select_k", 0))
        self._selector_type = selector_type
        self._selector_k = max(0, selector_k)
        if selector_type == "none" or self._selector_k <= 0:
            self.token_selector = None
        elif selector_type == "topk":
            d = int(cfg.lm_hidden_size)
            self.token_selector = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, 1),
            )
        else:
            raise ValueError(
                f"Unsupported bridge_token_selector_type={selector_type}. Supported: none, topk"
            )

    def _token_pos_emb(self, nv: int, d_model: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(nv), int(d_model), str(device), str(dtype))
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached
        emb = _build_token_2d_pos_emb(int(nv), int(d_model), device=device, dtype=dtype)
        self._pos_cache[key] = emb
        return emb

    def _prepare_visual_tokens(self, visual_features: torch.Tensor) -> torch.Tensor:
        x = _as_token_sequence(visual_features)
        x = self.visual_proj(x)
        if bool(self.cfg.add_2d_pos_emb):
            x = x + self._token_pos_emb(
                int(x.shape[1]),
                int(x.shape[2]),
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(0)
        x = self.spatial_mixer(x)
        x = self._maybe_select_tokens(x)
        return x

    def _maybe_select_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_selector is None:
            return x
        n = int(x.shape[1])
        k = min(max(1, int(self._selector_k)), n)
        if k >= n:
            return x
        scores = self.token_selector(x).squeeze(-1)  # [B, N]
        top_vals, top_idx = torch.topk(scores, k=k, dim=1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, int(x.shape[-1]))
        selected = torch.gather(x, dim=1, index=gather_idx)
        # Gate selected tokens by learned confidence while keeping hard top-k selection.
        gate = torch.sigmoid(top_vals).unsqueeze(-1)
        return selected * gate


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

    def _token_pos_emb(self, nv: int, dv: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(nv), int(dv), str(device), str(dtype))
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached
        emb = _build_token_2d_pos_emb(int(nv), int(dv), device=device, dtype=dtype)
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


class LearnedQueryCrossAttentionBridge(_QueryBridgeBase):
    """
    Main candidate: learned query tokens cross-attend to visual tokens.
    Optional spatial mixer pre-compression and optional query refinement blocks.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg)
        d_lm = int(cfg.lm_hidden_size)
        k = int(cfg.num_visual_tokens)
        std = float(cfg.learned_init_std)
        self.query_tokens = nn.Parameter(torch.randn(1, k, d_lm) * std)
        self.cross = _CrossAttnFFNBlock(
            d_lm,
            num_heads=int(cfg.bridge_num_heads),
            attn_dropout=float(cfg.bridge_attn_dropout),
        )
        self.refine = nn.ModuleList(
            [
                _SelfAttnFFNBlock(
                    d_lm,
                    num_heads=int(cfg.bridge_num_heads),
                    attn_dropout=float(cfg.bridge_attn_dropout),
                )
                for _ in range(max(0, int(cfg.bridge_refine_layers)))
            ]
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        visual_tokens = self._prepare_visual_tokens(visual_features)
        q = self.query_tokens.expand(int(visual_tokens.shape[0]), -1, -1)
        q = self.cross(q, visual_tokens)
        for blk in self.refine:
            q = blk(q)
        return q


class PerceiverResamplerBridge(_QueryBridgeBase):
    """
    Perceiver-style latent resampler: latents repeatedly cross-attend to visual tokens.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg)
        d_lm = int(cfg.lm_hidden_size)
        k = int(cfg.num_visual_tokens)
        std = float(cfg.learned_init_std)
        rounds = max(1, int(cfg.bridge_query_depth))
        self.latents = nn.Parameter(torch.randn(1, k, d_lm) * std)
        self.cross_blocks = nn.ModuleList(
            [
                _CrossAttnFFNBlock(
                    d_lm,
                    num_heads=int(cfg.bridge_num_heads),
                    attn_dropout=float(cfg.bridge_attn_dropout),
                )
                for _ in range(rounds)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                _SelfAttnFFNBlock(
                    d_lm,
                    num_heads=int(cfg.bridge_num_heads),
                    attn_dropout=float(cfg.bridge_attn_dropout),
                )
                for _ in range(rounds)
            ]
        )
        self.supports_question_context = bool(getattr(cfg, "bridge_question_conditioning", False))
        self.qcond_scale = float(getattr(cfg, "bridge_qcond_scale", 0.5))
        if self.supports_question_context:
            self.qcond_ln = nn.LayerNorm(d_lm)
            self.qcond_proj = nn.Linear(d_lm, 2 * d_lm)
        else:
            self.qcond_ln = None
            self.qcond_proj = None

    def _apply_question_conditioning(
        self,
        latents: torch.Tensor,
        question_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.supports_question_context or question_context is None:
            return latents
        if question_context.ndim != 2:
            raise ValueError(
                f"Expected question_context [B,D], got {tuple(question_context.shape)}"
            )
        assert self.qcond_ln is not None and self.qcond_proj is not None
        qctx = self.qcond_ln(question_context)
        gamma_beta = self.qcond_proj(qctx)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        s = max(0.0, float(self.qcond_scale))
        scale = 1.0 + s * torch.tanh(gamma).unsqueeze(1)
        shift = s * beta.unsqueeze(1)
        return latents * scale + shift

    def forward(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_tokens = self._prepare_visual_tokens(visual_features)
        latents = self.latents.expand(int(visual_tokens.shape[0]), -1, -1)
        latents = self._apply_question_conditioning(latents, question_context)
        for cross_blk, self_blk in zip(self.cross_blocks, self.self_blocks):
            latents = cross_blk(latents, visual_tokens)
            latents = self_blk(latents)
        return latents


class _QFormerLiteBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, attn_dropout: float):
        super().__init__()
        self.ln_self = nn.LayerNorm(int(dim))
        self.self_attn = nn.MultiheadAttention(
            int(dim),
            num_heads=_resolve_num_heads(int(dim), int(num_heads)),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.ln_cross_q = nn.LayerNorm(int(dim))
        self.ln_cross_kv = nn.LayerNorm(int(dim))
        self.cross_attn = nn.MultiheadAttention(
            int(dim),
            num_heads=_resolve_num_heads(int(dim), int(num_heads)),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.ln_ff = nn.LayerNorm(int(dim))
        h = max(4, int(dim) * 4)
        self.ffn = nn.Sequential(
            nn.Linear(int(dim), h),
            nn.GELU(),
            nn.Linear(h, int(dim)),
        )

    def forward(self, q: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        qs = self.ln_self(q)
        self_out, _ = self.self_attn(qs, qs, qs, need_weights=False)
        q = q + self_out
        qq = self.ln_cross_q(q)
        kv = self.ln_cross_kv(visual_tokens)
        cross_out, _ = self.cross_attn(qq, kv, kv, need_weights=False)
        q = q + cross_out
        q = q + self.ffn(self.ln_ff(q))
        return q


class QFormerLiteBridge(_QueryBridgeBase):
    """
    Q-former-lite: alternating self-attn on queries and cross-attn into visual tokens.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg)
        d_lm = int(cfg.lm_hidden_size)
        k = int(cfg.num_visual_tokens)
        std = float(cfg.learned_init_std)
        depth = max(1, int(cfg.bridge_query_depth))
        self.query_tokens = nn.Parameter(torch.randn(1, k, d_lm) * std)
        self.blocks = nn.ModuleList(
            [
                _QFormerLiteBlock(
                    d_lm,
                    num_heads=int(cfg.bridge_num_heads),
                    attn_dropout=float(cfg.bridge_attn_dropout),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        visual_tokens = self._prepare_visual_tokens(visual_features)
        q = self.query_tokens.expand(int(visual_tokens.shape[0]), -1, -1)
        for blk in self.blocks:
            q = blk(q, visual_tokens)
        return q


class HybridConstantImageBridge(nn.Module):
    """
    Hybrid of stable learned prefix and image-conditioned prefix.

    prefix = alpha * learned_prefix + (1 - alpha) * image_prefix
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.requires_visual_features = True
        self.cfg = cfg
        k = int(cfg.num_visual_tokens)
        d_lm = int(cfg.lm_hidden_size)
        std = float(cfg.learned_init_std)
        self.learned_prefix = nn.Parameter(torch.randn(1, k, d_lm) * std)
        alpha_mode = str(cfg.bridge_hybrid_alpha_mode)
        if alpha_mode == "scalar":
            shape = (1, 1, 1)
        elif alpha_mode == "token":
            shape = (1, k, 1)
        else:
            raise ValueError(
                f"Unsupported bridge_hybrid_alpha_mode={alpha_mode}. Supported: scalar, token"
            )
        a0 = min(1.0 - 1e-4, max(1e-4, float(cfg.bridge_hybrid_alpha_init)))
        logit = math.log(a0 / (1.0 - a0))
        self.alpha_logit = nn.Parameter(torch.full(shape, logit))

        image_type = str(cfg.bridge_hybrid_image_bridge_type)
        if image_type in ("learned_tokens", "hybrid_const_image"):
            raise ValueError(
                "bridge_hybrid_image_bridge_type must be image-conditioned "
                "(e.g., mlp, learned_query, perceiver_resampler, qformer_lite)."
            )
        sub_cfg = replace(cfg, bridge_type=image_type)
        self.image_bridge = build_bridge(sub_cfg)
        if not bool(getattr(self.image_bridge, "requires_visual_features", True)):
            raise ValueError("Hybrid image branch must require visual features.")
        self.supports_question_context = bool(getattr(self.image_bridge, "supports_question_context", False))

    def forward(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.supports_question_context:
            img = self.image_bridge(visual_features, question_context=question_context)
        else:
            img = self.image_bridge(visual_features)
        b = int(img.shape[0])
        learned = self.learned_prefix.expand(b, -1, -1)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * learned + (1.0 - alpha) * img


def build_bridge(cfg: BridgeConfig) -> nn.Module:
    bt = str(cfg.bridge_type)
    if bt == "mlp":
        return MLPVisualBridge(cfg)
    if bt == "learned_tokens":
        return LearnedVisualTokensBridge(cfg)
    if bt in ("learned_query", "query_cross_attn"):
        return LearnedQueryCrossAttentionBridge(cfg)
    if bt == "perceiver_resampler":
        return PerceiverResamplerBridge(cfg)
    if bt == "qformer_lite":
        return QFormerLiteBridge(cfg)
    if bt == "hybrid_const_image":
        return HybridConstantImageBridge(cfg)
    raise ValueError(
        f"Unsupported bridge_type={cfg.bridge_type}. "
        "Supported: mlp, learned_tokens, learned_query, perceiver_resampler, qformer_lite, hybrid_const_image"
    )
