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
    bridge_query_bank_mode: str = "learned"  # learned | question_mix | question_hidden_mean | question_hidden_attn | question_hidden_mean_multi | question_hidden_hybrid
    bridge_qquery_basis_count: int = 4
    bridge_qquery_scale: float = 1.0
    bridge_qquery_multi_count: int = 1
    bridge_qquery_hybrid_gate_init: float = 0.5
    bridge_query_role_specialization: bool = False
    bridge_question_context_mode: str = "all_text"  # all_text | prompt_only | question_only
    bridge_iterative_qquery_steps: int = 1
    bridge_iterative_qquery_residual_scale: float = 1.0
    bridge_token_selector_type: str = "none"  # none | topk | qtopk | qadaptive
    bridge_token_select_k: int = 0
    bridge_token_select_k_min: int = 0
    bridge_num_roles: int = 4
    bridge_evidence_topk: int = 0
    semantic_bottleneck: bool = False
    semantic_tokens: int = 16
    semantic_latent_dim: int = 256
    semantic_recon_loss_weight: float = 0.1
    semantic_consistency_loss_weight: float = 0.1
    semantic_token_schedule: str = ""


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


def _as_multiscale_token_pair(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return _as_token_sequence(x[0]), _as_token_sequence(x[1])
    raise ValueError("Expected a 2-tensor multiscale feature tuple/list.")


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

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        qq = self.ln_q(q)
        kvn = self.ln_kv(kv)
        attn_out, attn_weights = self.cross_attn(
            qq,
            kvn,
            kvn,
            need_weights=bool(return_attn),
            average_attn_weights=False,
        )
        q = q + attn_out
        q = q + self.ffn(self.ln2(q))
        if not bool(return_attn):
            return q
        if attn_weights is None:
            raise RuntimeError("Requested cross-attention weights but none were returned.")
        return q, attn_weights


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
        self._selector_k_min = max(0, int(getattr(cfg, "bridge_token_select_k_min", 0)))
        self.supports_question_context = selector_type in ("qtopk", "qadaptive")
        if selector_type == "none" or self._selector_k <= 0:
            self.token_selector = None
            self.token_selector_qproj = None
            self.token_budget = None
        elif selector_type == "topk":
            d = int(cfg.lm_hidden_size)
            self.token_selector = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, 1),
            )
            self.token_selector_qproj = None
            self.token_budget = None
        elif selector_type in ("qtopk", "qadaptive"):
            d = int(cfg.lm_hidden_size)
            self.token_selector = nn.Sequential(
                nn.LayerNorm(d),
                nn.GELU(),
                nn.Linear(d, 1),
            )
            self.token_selector_qproj = nn.Linear(d, d)
            self.token_budget = (
                nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, 1),
                )
                if selector_type == "qadaptive"
                else None
            )
        else:
            raise ValueError(
                "Unsupported bridge_token_selector_type="
                f"{selector_type}. Supported: none, topk, qtopk, qadaptive"
            )

    def _token_pos_emb(self, nv: int, d_model: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(nv), int(d_model), str(device), str(dtype))
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached
        emb = _build_token_2d_pos_emb(int(nv), int(d_model), device=device, dtype=dtype)
        self._pos_cache[key] = emb
        return emb

    def _prepare_visual_tokens(
        self,
        visual_features: torch.Tensor,
        *,
        question_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        x = self._maybe_select_tokens(x, question_context=question_context)
        return x

    def _maybe_select_tokens(
        self,
        x: torch.Tensor,
        *,
        question_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.token_selector is None:
            return x
        b = int(x.shape[0])
        n = int(x.shape[1])
        max_k = min(max(1, int(self._selector_k)), n)
        if max_k >= n:
            return x

        if self._selector_type == "topk":
            scores = self.token_selector(x).squeeze(-1)
            kept_k = torch.full((b,), max_k, device=x.device, dtype=torch.long)
        else:
            if question_context is None:
                raise ValueError(
                    f"bridge_token_selector_type={self._selector_type} requires question_context, but got None."
                )
            assert self.token_selector_qproj is not None
            q_bias = self.token_selector_qproj(question_context).unsqueeze(1)
            scores = self.token_selector(torch.tanh(x + q_bias)).squeeze(-1)
            if self._selector_type == "qadaptive" and self.token_budget is not None:
                min_k = min(max_k, max(1, int(self._selector_k_min) or max_k // 2))
                if min_k >= max_k:
                    kept_k = torch.full((b,), max_k, device=x.device, dtype=torch.long)
                else:
                    alpha = torch.sigmoid(self.token_budget(question_context)).squeeze(-1)
                    kept_k = torch.round(min_k + alpha * float(max_k - min_k)).to(torch.long)
                    kept_k = kept_k.clamp(min=min_k, max=max_k)
            else:
                kept_k = torch.full((b,), max_k, device=x.device, dtype=torch.long)

        top_vals, top_idx = torch.topk(scores, k=max_k, dim=1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, int(x.shape[-1]))
        selected = torch.gather(x, dim=1, index=gather_idx)
        # Gate selected tokens by learned confidence while keeping hard top-k selection.
        gate = torch.sigmoid(top_vals).unsqueeze(-1)
        if self._selector_type == "qadaptive":
            rank = torch.arange(max_k, device=x.device, dtype=torch.long).unsqueeze(0)
            active = (rank < kept_k.unsqueeze(1)).unsqueeze(-1)
            gate = gate * active.to(dtype=gate.dtype)
        return selected * gate


class _PerceiverCore(nn.Module):
    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        d_lm = int(cfg.lm_hidden_size)
        k = int(cfg.num_visual_tokens)
        std = float(cfg.learned_init_std)
        rounds = max(1, int(cfg.bridge_query_depth))
        query_bank_mode = str(getattr(cfg, "bridge_query_bank_mode", "learned"))
        role_specialization = bool(getattr(cfg, "bridge_query_role_specialization", False))
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
        self.qquery_mode = query_bank_mode
        self.uses_question_queries = query_bank_mode != "learned"
        self.uses_question_conditioning = bool(getattr(cfg, "bridge_question_conditioning", False))
        self.uses_question_tokens = query_bank_mode in (
            "question_hidden_mean",
            "question_hidden_attn",
            "question_hidden_mean_multi",
            "question_hidden_hybrid",
        )
        if query_bank_mode not in (
            "learned",
            "question_mix",
            "question_hidden_mean",
            "question_hidden_attn",
            "question_hidden_mean_multi",
            "question_hidden_hybrid",
        ):
            raise ValueError(
                "Unsupported bridge_query_bank_mode="
                f"{query_bank_mode}. Supported: learned, question_mix, question_hidden_mean, "
                "question_hidden_attn, question_hidden_mean_multi, question_hidden_hybrid"
            )
        self.supports_question_context = self.uses_question_conditioning or query_bank_mode in (
            "question_mix",
            "question_hidden_mean",
            "question_hidden_mean_multi",
            "question_hidden_hybrid",
        )
        self.qcond_scale = float(getattr(cfg, "bridge_qcond_scale", 0.5))
        self.qquery_scale = float(getattr(cfg, "bridge_qquery_scale", 1.0))
        self.iterative_steps = max(1, int(getattr(cfg, "bridge_iterative_qquery_steps", 1)))
        self.iterative_residual_scale = float(getattr(cfg, "bridge_iterative_qquery_residual_scale", 1.0))
        if self.uses_question_conditioning:
            self.qcond_ln = nn.LayerNorm(d_lm)
            self.qcond_proj = nn.Linear(d_lm, 2 * d_lm)
        else:
            self.qcond_ln = None
            self.qcond_proj = None
        if query_bank_mode == "question_mix":
            basis_count = max(1, int(getattr(cfg, "bridge_qquery_basis_count", 4)))
            self.qquery_ln = nn.LayerNorm(d_lm)
            self.qquery_mix = nn.Linear(d_lm, basis_count)
            self.qquery_basis = nn.Parameter(torch.randn(basis_count, k, d_lm) * std)
            self.qquery_hidden_ln = None
            self.qquery_hidden_proj = None
            self.qquery_token_ln = None
            self.qquery_token_attn = None
        elif query_bank_mode == "question_hidden_mean":
            self.qquery_ln = None
            self.qquery_mix = None
            self.qquery_basis = None
            self.qquery_hidden_ln = nn.LayerNorm(d_lm)
            self.qquery_hidden_proj = nn.Linear(d_lm, k * d_lm)
            self.qquery_token_ln = None
            self.qquery_token_attn = None
        elif query_bank_mode == "question_hidden_attn":
            self.qquery_ln = None
            self.qquery_mix = None
            self.qquery_basis = None
            self.qquery_hidden_ln = None
            self.qquery_hidden_proj = None
            self.qquery_multi_count = 1
            self.qquery_multi_ctx_ln = None
            self.qquery_multi_ctx_proj = None
            self.qquery_group_ids = None
            self.qquery_token_ln = nn.LayerNorm(d_lm)
            self.qquery_token_attn = nn.MultiheadAttention(
                d_lm,
                num_heads=_resolve_num_heads(d_lm, int(cfg.bridge_num_heads)),
                dropout=float(cfg.bridge_attn_dropout),
                batch_first=True,
            )
            self.qquery_hybrid_gate_logit = None
        elif query_bank_mode == "question_hidden_mean_multi":
            self.qquery_ln = None
            self.qquery_mix = None
            self.qquery_basis = None
            self.qquery_hidden_ln = None
            self.qquery_hidden_proj = None
            self.qquery_multi_count = max(2, int(getattr(cfg, "bridge_qquery_multi_count", 4)))
            self.qquery_multi_ctx_ln = nn.LayerNorm(d_lm)
            self.qquery_multi_ctx_proj = nn.Linear(d_lm, self.qquery_multi_count * d_lm)
            self.qquery_group_ids = torch.arange(k, dtype=torch.long) % self.qquery_multi_count
            self.register_buffer("qquery_group_ids_buf", self.qquery_group_ids, persistent=False)
            self.qquery_token_ln = None
            self.qquery_token_attn = None
            self.qquery_hybrid_gate_logit = None
        elif query_bank_mode == "question_hidden_hybrid":
            self.qquery_ln = None
            self.qquery_mix = None
            self.qquery_basis = None
            self.qquery_hidden_ln = nn.LayerNorm(d_lm)
            self.qquery_hidden_proj = nn.Linear(d_lm, k * d_lm)
            self.qquery_multi_count = 1
            self.qquery_multi_ctx_ln = None
            self.qquery_multi_ctx_proj = None
            self.qquery_group_ids = None
            self.qquery_token_ln = nn.LayerNorm(d_lm)
            self.qquery_token_attn = nn.MultiheadAttention(
                d_lm,
                num_heads=_resolve_num_heads(d_lm, int(cfg.bridge_num_heads)),
                dropout=float(cfg.bridge_attn_dropout),
                batch_first=True,
            )
            gate = min(1.0 - 1e-4, max(1e-4, float(getattr(cfg, "bridge_qquery_hybrid_gate_init", 0.5))))
            self.qquery_hybrid_gate_logit = nn.Parameter(torch.full((1, 1, d_lm), math.log(gate / (1.0 - gate))))
        else:
            self.qquery_ln = None
            self.qquery_mix = None
            self.qquery_basis = None
            self.qquery_hidden_ln = None
            self.qquery_hidden_proj = None
            self.qquery_multi_count = 1
            self.qquery_multi_ctx_ln = None
            self.qquery_multi_ctx_proj = None
            self.qquery_group_ids = None
            self.qquery_token_ln = None
            self.qquery_token_attn = None
            self.qquery_hybrid_gate_logit = None
        if self.iterative_steps > 1:
            self.iter_ln = nn.LayerNorm(2 * d_lm)
            self.iter_proj = nn.Linear(2 * d_lm, d_lm)
        else:
            self.iter_ln = None
            self.iter_proj = None
        if role_specialization:
            role_count = max(1, int(getattr(cfg, "bridge_num_roles", 4)))
            self.role_embeddings = nn.Parameter(torch.randn(role_count, d_lm) * std)
            role_ids = torch.arange(k, dtype=torch.long) % role_count
            self.register_buffer("role_ids", role_ids, persistent=False)
        else:
            self.role_embeddings = None
            self.role_ids = None

    @staticmethod
    def _masked_mean(question_tokens: torch.Tensor, question_token_mask: torch.Tensor | None) -> torch.Tensor:
        if question_token_mask is None:
            return question_tokens.mean(dim=1)
        valid = question_token_mask.unsqueeze(-1).to(dtype=question_tokens.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (question_tokens * valid).sum(dim=1) / denom

    def _apply_role_specialization(self, latents: torch.Tensor) -> torch.Tensor:
        if self.role_embeddings is None or self.role_ids is None:
            return latents
        return latents + self.role_embeddings[self.role_ids].unsqueeze(0)

    def _apply_question_queries(
        self,
        latents: torch.Tensor,
        question_context: torch.Tensor | None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.uses_question_queries:
            return latents
        if self.qquery_mode == "question_mix" and question_context is None:
            return latents
        if self.qquery_mode == "question_hidden_mean" and question_context is None and question_tokens is None:
            return latents
        if self.qquery_mode == "question_hidden_mean_multi" and question_context is None and question_tokens is None:
            return latents
        if self.qquery_mode == "question_hidden_hybrid" and question_context is None and question_tokens is None:
            return latents
        if self.qquery_mode == "question_hidden_attn" and question_tokens is None:
            return latents
        if self.qquery_mode == "question_mix":
            assert question_context is not None
            assert self.qquery_ln is not None and self.qquery_mix is not None and self.qquery_basis is not None
            weights = torch.softmax(self.qquery_mix(self.qquery_ln(question_context)), dim=-1)
            delta = torch.einsum("br,rkd->bkd", weights, self.qquery_basis)
            return latents + float(self.qquery_scale) * delta
        if self.qquery_mode == "question_hidden_mean":
            pooled = question_context
            if question_tokens is not None:
                pooled = self._masked_mean(question_tokens, question_token_mask)
            assert pooled is not None
            assert self.qquery_hidden_ln is not None and self.qquery_hidden_proj is not None
            delta = self.qquery_hidden_proj(self.qquery_hidden_ln(pooled)).view_as(latents)
            return latents + float(self.qquery_scale) * torch.tanh(delta)
        if self.qquery_mode == "question_hidden_mean_multi":
            pooled = question_context
            if question_tokens is not None:
                pooled = self._masked_mean(question_tokens, question_token_mask)
            assert pooled is not None
            assert self.qquery_multi_ctx_ln is not None and self.qquery_multi_ctx_proj is not None
            group_ctx = self.qquery_multi_ctx_proj(self.qquery_multi_ctx_ln(pooled))
            group_ctx = torch.tanh(group_ctx.view(int(latents.shape[0]), self.qquery_multi_count, int(latents.shape[-1])))
            group_ids = getattr(self, "qquery_group_ids_buf", None)
            if group_ids is None:
                raise RuntimeError("question_hidden_mean_multi missing qquery_group_ids_buf")
            delta = group_ctx[:, group_ids.to(device=latents.device), :]
            return latents + float(self.qquery_scale) * delta
        if self.qquery_mode == "question_hidden_attn":
            if question_tokens is None:
                return latents
            assert self.qquery_token_ln is not None and self.qquery_token_attn is not None
            tok = self.qquery_token_ln(question_tokens)
            key_padding_mask = None
            if question_token_mask is not None:
                key_padding_mask = ~question_token_mask
            delta, _ = self.qquery_token_attn(
                self.qquery_token_ln(latents),
                tok,
                tok,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            return latents + float(self.qquery_scale) * delta
        if self.qquery_mode == "question_hidden_hybrid":
            pooled = question_context
            if question_tokens is not None:
                pooled = self._masked_mean(question_tokens, question_token_mask)
            mean_delta = 0.0
            attn_delta = 0.0
            if pooled is not None:
                assert self.qquery_hidden_ln is not None and self.qquery_hidden_proj is not None
                mean_delta = torch.tanh(self.qquery_hidden_proj(self.qquery_hidden_ln(pooled)).view_as(latents))
            if question_tokens is not None:
                assert self.qquery_token_ln is not None and self.qquery_token_attn is not None
                tok = self.qquery_token_ln(question_tokens)
                key_padding_mask = None if question_token_mask is None else ~question_token_mask
                attn_delta, _ = self.qquery_token_attn(
                    self.qquery_token_ln(latents),
                    tok,
                    tok,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
            gate = torch.sigmoid(self.qquery_hybrid_gate_logit) if self.qquery_hybrid_gate_logit is not None else 0.5
            delta = gate * mean_delta + (1.0 - gate) * attn_delta
            return latents + float(self.qquery_scale) * delta
        return latents

    def _apply_question_conditioning(
        self,
        latents: torch.Tensor,
        question_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.uses_question_conditioning or question_context is None:
            return latents
        if question_context.ndim != 2:
            raise ValueError(f"Expected question_context [B,D], got {tuple(question_context.shape)}")
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
        visual_tokens: torch.Tensor,
        *,
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if latents is None:
            latents = self.latents.expand(int(visual_tokens.shape[0]), -1, -1)
        latents = self._apply_role_specialization(latents)
        attn_maps: list[torch.Tensor] = []
        for step_idx in range(self.iterative_steps):
            latents = self._apply_question_queries(
                latents,
                question_context,
                question_tokens=question_tokens,
                question_token_mask=question_token_mask,
            )
            latents = self._apply_question_conditioning(latents, question_context)
            for cross_blk, self_blk in zip(self.cross_blocks, self.self_blocks):
                if bool(return_attn):
                    latents, attn = cross_blk(latents, visual_tokens, return_attn=True)
                    attn_maps.append(attn)
                else:
                    latents = cross_blk(latents, visual_tokens)
                latents = self_blk(latents)
            if step_idx + 1 < self.iterative_steps:
                summary = latents.mean(dim=1)
                if question_context is None:
                    qctx = summary
                else:
                    qctx = question_context
                fuse = torch.cat([summary, qctx], dim=-1)
                assert self.iter_ln is not None and self.iter_proj is not None
                delta = torch.tanh(self.iter_proj(self.iter_ln(fuse))).unsqueeze(1)
                latents = latents + float(self.iterative_residual_scale) * delta
        if not bool(return_attn):
            return latents
        return latents, attn_maps


class SemanticBottleneck(nn.Module):
    """
    Late semantic compression over post-perceiver evidence latents.

    Shapes:
    - evidence_latents: [B, K, D]
    - semantic_latents: [B, M, Z]
    - exported_tokens: [B, M, D]
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        d_model = int(cfg.lm_hidden_size)
        self.target_token_count = max(1, int(cfg.num_visual_tokens))
        self.semantic_tokens = max(1, min(int(getattr(cfg, "semantic_tokens", 16)), self.target_token_count))
        self.semantic_latent_dim = max(1, min(int(getattr(cfg, "semantic_latent_dim", 256)), d_model))
        std = float(cfg.learned_init_std)
        self.semantic_queries = nn.Parameter(torch.randn(1, self.semantic_tokens, d_model) * std)
        self.compress = _CrossAttnFFNBlock(
            d_model,
            num_heads=int(cfg.bridge_num_heads),
            attn_dropout=float(cfg.bridge_attn_dropout),
        )
        self.to_semantic = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.semantic_latent_dim),
        )
        self.to_export = nn.Sequential(
            nn.LayerNorm(self.semantic_latent_dim),
            nn.Linear(self.semantic_latent_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.recon_ln = nn.LayerNorm(d_model)
        # Linear decoder over token axis: [B, M, D] -> [B, K_target, D]
        self.recon_token_proj = nn.Linear(self.semantic_tokens, self.target_token_count, bias=True)

    def _target_tokens(self, evidence_latents: torch.Tensor, target_evidence_latents: torch.Tensor | None) -> torch.Tensor:
        target = target_evidence_latents if target_evidence_latents is not None else evidence_latents
        if int(target.shape[1]) != self.target_token_count:
            target = F.adaptive_avg_pool1d(
                target.transpose(1, 2),
                self.target_token_count,
            ).transpose(1, 2)
        return target.detach()

    def forward(
        self,
        evidence_latents: torch.Tensor,
        *,
        target_evidence_latents: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        # evidence_latents: [B, K, D] -> semantic_slots: [B, M, D]
        compress_out = self.compress(
            self.semantic_queries.expand(int(evidence_latents.shape[0]), -1, -1),
            evidence_latents,
            return_attn=return_attn,
        )
        if bool(return_attn):
            semantic_slots, semantic_attn = compress_out
        else:
            semantic_slots = compress_out
            semantic_attn = None
        # semantic_latents: [B, M, Z] -> exported_tokens: [B, M, D]
        semantic_latents = self.to_semantic(semantic_slots)
        exported_tokens = self.to_export(semantic_latents)
        # reconstruction tokens: [B, M, D] -> [B, K_target, D]
        recon_tokens = self.recon_token_proj(self.recon_ln(exported_tokens).transpose(1, 2)).transpose(1, 2)
        target_tokens = self._target_tokens(evidence_latents, target_evidence_latents)
        recon_loss = F.mse_loss(recon_tokens, target_tokens)
        consistency_loss = 1.0 - F.cosine_similarity(
            recon_tokens.float(),
            target_tokens.float(),
            dim=-1,
        ).mean()
        aux = {
            "semantic_recon_loss": recon_loss,
            "semantic_consistency_loss": consistency_loss,
            "semantic_token_count": evidence_latents.new_tensor(float(self.semantic_tokens)),
            "semantic_latent_dim": evidence_latents.new_tensor(float(self.semantic_latent_dim)),
            "semantic_target_token_count": evidence_latents.new_tensor(float(self.target_token_count)),
            "semantic_bottleneck_enabled": evidence_latents.new_tensor(1.0),
        }
        if semantic_attn is not None:
            attn = semantic_attn.float().clamp_min(1e-8)
            entropy = (-(attn * attn.log()).sum(dim=-1)).mean()
            aux["compression_mean_attn_entropy"] = entropy
        if not bool(return_attn):
            return exported_tokens, aux
        if semantic_attn is None:
            raise RuntimeError("Requested semantic attention weights but none were returned.")
        return exported_tokens, aux, semantic_attn


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
        self.last_aux_info: dict[str, torch.Tensor] = {
            "semantic_bottleneck_enabled": torch.tensor(0.0),
        }
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

    def forward(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        visual_tokens = self._prepare_visual_tokens(visual_features, question_context=question_context)
        q = self.query_tokens.expand(int(visual_tokens.shape[0]), -1, -1)
        attn_maps: list[torch.Tensor] = []
        if bool(return_attn):
            q, attn = self.cross(q, visual_tokens, return_attn=True)
            attn_maps.append(attn)
        else:
            q = self.cross(q, visual_tokens)
        for blk in self.refine:
            q = blk(q)
        if not bool(return_attn):
            return q
        return q, attn_maps


class PerceiverResamplerBridge(_QueryBridgeBase):
    """
    Perceiver-style latent resampler: latents repeatedly cross-attend to visual tokens.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg)
        self.core = _PerceiverCore(cfg)
        self.supports_question_context = bool(self.supports_question_context or self.core.supports_question_context)
        self.supports_question_tokens = bool(getattr(self.core, "uses_question_tokens", False))
        self.semantic_bottleneck = SemanticBottleneck(cfg) if bool(getattr(cfg, "semantic_bottleneck", False)) else None
        self.semantic_bottleneck_enabled = self.semantic_bottleneck is not None
        self.eval_bypass_compression = False
        self.semantic_recon_loss_weight = float(getattr(cfg, "semantic_recon_loss_weight", 0.0))
        self.semantic_consistency_loss_weight = float(getattr(cfg, "semantic_consistency_loss_weight", 0.0))
        self.last_aux_info: dict[str, torch.Tensor] = {
            "semantic_bottleneck_enabled": torch.tensor(1.0 if self.semantic_bottleneck_enabled else 0.0),
        }

    def forward_evidence(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        visual_tokens = self._prepare_visual_tokens(visual_features, question_context=question_context)
        return self.core(
            visual_tokens,
            question_context=question_context,
            question_tokens=question_tokens,
            question_token_mask=question_token_mask,
            return_attn=return_attn,
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
        semantic_target_latents: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        core_out = self.forward_evidence(
            visual_features,
            question_context=question_context,
            question_tokens=question_tokens,
            question_token_mask=question_token_mask,
            return_attn=return_attn,
        )
        if bool(return_attn):
            evidence_latents, attn_maps = core_out
        else:
            evidence_latents = core_out
            attn_maps = None
        perceiver_final_attn = None
        if attn_maps:
            perceiver_final_attn = attn_maps[-1]
        if self.semantic_bottleneck is None or bool(getattr(self, "eval_bypass_compression", False)):
            self.last_aux_info = {
                "semantic_bottleneck_enabled": evidence_latents.new_tensor(0.0),
                "perceiver_final_attn": perceiver_final_attn if perceiver_final_attn is not None else evidence_latents.new_zeros((int(evidence_latents.shape[0]), 1, int(evidence_latents.shape[1]), 1)),
                "compression_bypassed": evidence_latents.new_tensor(1.0 if bool(getattr(self, "eval_bypass_compression", False)) else 0.0),
            }
            if not bool(return_attn):
                return evidence_latents
            return evidence_latents, attn_maps if attn_maps is not None else []
        semantic_out = self.semantic_bottleneck(
            evidence_latents,
            target_evidence_latents=semantic_target_latents,
            return_attn=return_attn,
        )
        if bool(return_attn):
            semantic_tokens, aux, semantic_attn = semantic_out
            if attn_maps is None:
                attn_maps = []
            attn_maps = list(attn_maps) + [semantic_attn]
        else:
            semantic_tokens, aux = semantic_out
        if perceiver_final_attn is not None:
            aux["perceiver_final_attn"] = perceiver_final_attn
        if bool(return_attn):
            aux["semantic_attn"] = semantic_attn
        aux["compression_bypassed"] = evidence_latents.new_tensor(0.0)
        self.last_aux_info = aux
        if not bool(return_attn):
            return semantic_tokens
        return semantic_tokens, attn_maps if attn_maps is not None else []


class MultiScalePerceiverBridge(nn.Module):
    """
    Perceiver over fused early/late visual tokens.
    Expected input: (encoder_tokens, posterior_mu_tokens).
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.cfg = cfg
        self.requires_visual_features = True
        self._pos_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}
        d_lm = int(cfg.lm_hidden_size)
        self.low_proj = nn.LazyLinear(d_lm)
        self.high_proj = nn.LazyLinear(d_lm)
        self.low_log_scale = nn.Parameter(torch.zeros(1, 1, d_lm))
        self.high_log_scale = nn.Parameter(torch.zeros(1, 1, d_lm))
        self.spatial_mixer = _SpatialMixer(cfg)
        self.core = _PerceiverCore(cfg)
        self.supports_question_context = bool(self.core.supports_question_context)
        self.supports_question_tokens = bool(getattr(self.core, "uses_question_tokens", False))
        self.semantic_bottleneck = SemanticBottleneck(cfg) if bool(getattr(cfg, "semantic_bottleneck", False)) else None
        self.semantic_bottleneck_enabled = self.semantic_bottleneck is not None
        self.eval_bypass_compression = False
        self.semantic_recon_loss_weight = float(getattr(cfg, "semantic_recon_loss_weight", 0.0))
        self.semantic_consistency_loss_weight = float(getattr(cfg, "semantic_consistency_loss_weight", 0.0))
        self.last_aux_info: dict[str, torch.Tensor] = {
            "semantic_bottleneck_enabled": torch.tensor(1.0 if self.semantic_bottleneck_enabled else 0.0),
        }

    def _token_pos_emb(self, nv: int, d_model: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(nv), int(d_model), str(device), str(dtype))
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached
        emb = _build_token_2d_pos_emb(int(nv), int(d_model), device=device, dtype=dtype)
        self._pos_cache[key] = emb
        return emb

    def _prepare_tokens(
        self,
        x: torch.Tensor,
        proj: nn.Module,
        scale_log: torch.Tensor,
    ) -> torch.Tensor:
        y = proj(_as_token_sequence(x))
        if bool(self.cfg.add_2d_pos_emb):
            y = y + self._token_pos_emb(int(y.shape[1]), int(y.shape[2]), device=y.device, dtype=y.dtype).unsqueeze(0)
        return y * torch.exp(scale_log).clamp(min=0.25, max=4.0)

    def forward(
        self,
        visual_features: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
        semantic_target_latents: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        low, high = _as_multiscale_token_pair(visual_features)
        low_t = self._prepare_tokens(low, self.low_proj, self.low_log_scale)
        high_t = self._prepare_tokens(high, self.high_proj, self.high_log_scale)
        visual_tokens = self.spatial_mixer(torch.cat([low_t, high_t], dim=1))
        core_out = self.core(
            visual_tokens,
            question_context=question_context,
            question_tokens=question_tokens,
            question_token_mask=question_token_mask,
            return_attn=return_attn,
        )
        if bool(return_attn):
            evidence_latents, attn_maps = core_out
        else:
            evidence_latents = core_out
            attn_maps = None
        perceiver_final_attn = None
        if attn_maps:
            perceiver_final_attn = attn_maps[-1]
        if self.semantic_bottleneck is None or bool(getattr(self, "eval_bypass_compression", False)):
            self.last_aux_info = {
                "semantic_bottleneck_enabled": evidence_latents.new_tensor(0.0),
                "perceiver_final_attn": perceiver_final_attn if perceiver_final_attn is not None else evidence_latents.new_zeros((int(evidence_latents.shape[0]), 1, int(evidence_latents.shape[1]), 1)),
                "compression_bypassed": evidence_latents.new_tensor(1.0 if bool(getattr(self, "eval_bypass_compression", False)) else 0.0),
            }
            if not bool(return_attn):
                return evidence_latents
            return evidence_latents, attn_maps if attn_maps is not None else []
        semantic_out = self.semantic_bottleneck(
            evidence_latents,
            target_evidence_latents=semantic_target_latents,
            return_attn=return_attn,
        )
        if bool(return_attn):
            semantic_tokens, aux, semantic_attn = semantic_out
            if attn_maps is None:
                attn_maps = []
            attn_maps = list(attn_maps) + [semantic_attn]
        else:
            semantic_tokens, aux = semantic_out
        if perceiver_final_attn is not None:
            aux["perceiver_final_attn"] = perceiver_final_attn
        if bool(return_attn):
            aux["semantic_attn"] = semantic_attn
        aux["compression_bypassed"] = evidence_latents.new_tensor(0.0)
        self.last_aux_info = aux
        if not bool(return_attn):
            return semantic_tokens
        return semantic_tokens, attn_maps if attn_maps is not None else []


class StructuredRolesBridge(_QueryBridgeBase):
    """
    Query bridge with fixed semantic role groups.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg)
        d_lm = int(cfg.lm_hidden_size)
        k = int(cfg.num_visual_tokens)
        std = float(cfg.learned_init_std)
        role_count = max(1, int(getattr(cfg, "bridge_num_roles", 4)))
        self.query_tokens = nn.Parameter(torch.randn(1, k, d_lm) * std)
        self.role_embeddings = nn.Parameter(torch.randn(role_count, d_lm) * std)
        role_ids = torch.arange(k, dtype=torch.long) % role_count
        self.register_buffer("role_ids", role_ids, persistent=False)
        depth = max(1, int(cfg.bridge_query_depth))
        self.cross_blocks = nn.ModuleList(
            [
                _CrossAttnFFNBlock(
                    d_lm,
                    num_heads=int(cfg.bridge_num_heads),
                    attn_dropout=float(cfg.bridge_attn_dropout),
                )
                for _ in range(depth)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                _SelfAttnFFNBlock(
                    d_lm,
                    num_heads=int(cfg.bridge_num_heads),
                    attn_dropout=float(cfg.bridge_attn_dropout),
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        visual_tokens = self._prepare_visual_tokens(visual_features, question_context=question_context)
        role_emb = self.role_embeddings[self.role_ids].unsqueeze(0)
        q = self.query_tokens.expand(int(visual_tokens.shape[0]), -1, -1) + role_emb
        attn_maps: list[torch.Tensor] = []
        for cross_blk, self_blk in zip(self.cross_blocks, self.self_blocks):
            if bool(return_attn):
                q, attn = cross_blk(q, visual_tokens, return_attn=True)
                attn_maps.append(attn)
            else:
                q = cross_blk(q, visual_tokens)
            q = self_blk(q)
        if not bool(return_attn):
            return q
        return q, attn_maps


class EvidenceSparseBridge(_QueryBridgeBase):
    """
    One global summary token plus sparse evidence queries over top-scoring visual tokens.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__(cfg)
        d_lm = int(cfg.lm_hidden_size)
        k = int(cfg.num_visual_tokens)
        std = float(cfg.learned_init_std)
        evidence_queries = max(0, k - 1)
        self.evidence_queries = nn.Parameter(torch.randn(1, evidence_queries, d_lm) * std)
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(d_lm),
            nn.Linear(d_lm, d_lm),
        )
        self.evidence_scorer = nn.Sequential(
            nn.LayerNorm(d_lm),
            nn.Linear(d_lm, 1),
        )
        topk = int(getattr(cfg, "bridge_evidence_topk", 0))
        self.evidence_topk = max(1, topk) if topk > 0 else max(8, min(32, 2 * max(1, evidence_queries)))
        depth = max(1, int(cfg.bridge_query_depth))
        self.cross_blocks = nn.ModuleList(
            [
                _CrossAttnFFNBlock(
                    d_lm,
                    num_heads=int(cfg.bridge_num_heads),
                    attn_dropout=float(cfg.bridge_attn_dropout),
                )
                for _ in range(depth)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                _SelfAttnFFNBlock(
                    d_lm,
                    num_heads=int(cfg.bridge_num_heads),
                    attn_dropout=float(cfg.bridge_attn_dropout),
                )
                for _ in range(depth)
            ]
        )

    def _select_evidence_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        n = int(visual_tokens.shape[1])
        k = min(self.evidence_topk, n)
        if k >= n:
            return visual_tokens
        scores = self.evidence_scorer(visual_tokens).squeeze(-1)
        top_vals, top_idx = torch.topk(scores, k=k, dim=1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, int(visual_tokens.shape[-1]))
        selected = torch.gather(visual_tokens, dim=1, index=gather_idx)
        return selected * torch.sigmoid(top_vals).unsqueeze(-1)

    def forward(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_tokens = self._prepare_visual_tokens(visual_features, question_context=question_context)
        summary = self.summary_proj(visual_tokens.mean(dim=1, keepdim=True))
        if int(self.evidence_queries.shape[1]) == 0:
            return summary
        evidence_tokens = self._select_evidence_tokens(visual_tokens)
        q = self.evidence_queries.expand(int(visual_tokens.shape[0]), -1, -1)
        for cross_blk, self_blk in zip(self.cross_blocks, self.self_blocks):
            q = cross_blk(q, evidence_tokens)
            q = self_blk(q)
        return torch.cat([summary, q], dim=1)


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

    def forward(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_tokens = self._prepare_visual_tokens(visual_features, question_context=question_context)
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
        self.supports_question_tokens = bool(getattr(self.image_bridge, "supports_question_tokens", False))

    def forward(
        self,
        visual_features: torch.Tensor,
        question_context: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        question_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.supports_question_context or self.supports_question_tokens:
            img = self.image_bridge(
                visual_features,
                question_context=question_context,
                question_tokens=question_tokens,
                question_token_mask=question_token_mask,
            )
        else:
            img = self.image_bridge(visual_features)
        b = int(img.shape[0])
        learned = self.learned_prefix.expand(b, -1, -1)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * learned + (1.0 - alpha) * img


def build_bridge(cfg: BridgeConfig) -> nn.Module:
    bt = str(cfg.bridge_type)
    if bool(getattr(cfg, "semantic_bottleneck", False)) and bt not in ("perceiver_resampler", "multiscale_perceiver"):
        raise ValueError(
            "semantic_bottleneck is currently supported only for perceiver-based bridges: "
            "perceiver_resampler, multiscale_perceiver"
        )
    if bt == "mlp":
        return MLPVisualBridge(cfg)
    if bt == "learned_tokens":
        return LearnedVisualTokensBridge(cfg)
    if bt in ("learned_query", "query_cross_attn"):
        return LearnedQueryCrossAttentionBridge(cfg)
    if bt == "perceiver_resampler":
        return PerceiverResamplerBridge(cfg)
    if bt == "multiscale_perceiver":
        return MultiScalePerceiverBridge(cfg)
    if bt == "qformer_lite":
        return QFormerLiteBridge(cfg)
    if bt == "structured_roles":
        return StructuredRolesBridge(cfg)
    if bt == "evidence_sparse":
        return EvidenceSparseBridge(cfg)
    if bt == "hybrid_const_image":
        return HybridConstantImageBridge(cfg)
    raise ValueError(
        f"Unsupported bridge_type={cfg.bridge_type}. "
        "Supported: mlp, learned_tokens, learned_query, perceiver_resampler, multiscale_perceiver, "
        "qformer_lite, structured_roles, evidence_sparse, hybrid_const_image"
    )
