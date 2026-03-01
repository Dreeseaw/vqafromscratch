"""
language modeling components of vqa
"""
import sys
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import yaml

"""
not even a baseline, just to test myself
"""
class RNN(nn.Module):
    def __init__(self, vocab_size: int = 64, embed: int = 128):
        super().__init__()
        self._e = embed
        self._in = nn.Linear(vocab_size, embed)
        self._h  = nn.Linear(embed*2, embed)
        self._h_act = nn.ReLU(inplace=True)
        self._out = nn.Linear(embed, vocab_size)

    def forward(self, seq: torch.Tensor):
        """
        seq: [B, L, V]
        """
        (B, L, V) = seq.shape
        h_cur = torch.zeros(B, self._e)
        for i in range(L):
            in_tokens = seq[:, i, :]  # [B, E]
            h_cur = self._h_act(self._h(torch.concat([self._in(in_tokens), h_cur], dim=1)))
        return self._out(h_cur)

"""
Transformer-based approaches
"""


def _masked_mean_std(values: torch.Tensor, keep_mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    vals = values.float()
    if keep_mask is not None:
        mask = keep_mask.to(device=vals.device, dtype=torch.bool)
        while mask.dim() < vals.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(vals)
        selected = vals[mask]
    else:
        selected = vals.reshape(-1)
    if selected.numel() == 0:
        return float("nan"), float("nan")
    mean = float(selected.mean().item())
    std = float(selected.std(unbiased=False).item())
    return mean, std


def _masked_rms(values: torch.Tensor, keep_mask: Optional[torch.Tensor] = None) -> float:
    vals = values.detach().float()
    if keep_mask is not None:
        mask = keep_mask.to(device=vals.device, dtype=torch.bool)
        while mask.dim() < vals.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(vals)
        selected = vals[mask]
    else:
        selected = vals.reshape(-1)
    if selected.numel() == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean(selected * selected)).item())


def _capture_norm_rms(
    debug_capture: Optional[Dict[str, Any]],
    prefix: str,
    pre: torch.Tensor,
    post: torch.Tensor,
    keep_mask: Optional[torch.Tensor],
) -> None:
    if debug_capture is None:
        return
    debug_capture[f"{prefix}_pre_rms"] = _masked_rms(pre, keep_mask)
    debug_capture[f"{prefix}_post_rms"] = _masked_rms(post, keep_mask)


def _mean_attn_entropy(
    masked_scores: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    probs = torch.softmax(masked_scores, dim=-1)
    probs = torch.where(valid, probs, torch.zeros_like(probs))
    row_valid = valid.any(dim=-1)  # [B,H,Q]
    probs_safe = probs.clamp_min(1e-12)
    row_entropy = -(probs_safe * probs_safe.log()).sum(dim=-1)
    if row_valid.shape != row_entropy.shape:
        row_valid = row_valid.expand_as(row_entropy)
    if not bool(row_valid.any().item()):
        return float("nan"), probs
    return float(row_entropy[row_valid].mean().item()), probs


def _build_attn_masks(
    pad_mask: Optional[torch.Tensor],
    q_len: int,
    k_len: int,
    device: torch.device,
    is_causal: bool,
):
    key_pad_mask = None
    if pad_mask is not None:
        key_pad_mask = pad_mask[:, None, None, :]

    if key_pad_mask is not None and not key_pad_mask.any():
        key_pad_mask = None

    sdpa_causal = bool(is_causal and key_pad_mask is None)
    sdpa_mask = None
    blocked = None

    causal = None
    if is_causal:
        causal = torch.triu(
            torch.ones(q_len, k_len, device=device, dtype=torch.bool),
            diagonal=1,
        )[None, None, :, :]

    if key_pad_mask is not None:
        if is_causal:
            blocked = key_pad_mask | causal
            sdpa_mask = ~blocked
        else:
            blocked = key_pad_mask.expand(-1, 1, q_len, -1)
            sdpa_mask = ~key_pad_mask
    elif causal is not None:
        blocked = causal

    return key_pad_mask, sdpa_causal, sdpa_mask, blocked


def _apply_layerscale(x: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
    if scale is None:
        return x
    if scale.dtype != x.dtype:
        scale = scale.to(dtype=x.dtype)
    return x * scale


def _rms_norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    inv_rms = torch.rsqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + eps)
    return x * inv_rms.to(dtype=x.dtype)


def clamp_residual(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    with torch.no_grad():
        norms = x.norm(dim=-1, keepdim=True)
        scale = (max_norm / (norms + 1e-8)).clamp(max=1.0)
    return x.mul_(scale)


def cap_vector_norm(
    x: torch.Tensor,
    max_norm: float,
    *,
    keep_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    mode: str = "token",
) -> torch.Tensor:
    """
    Project vectors onto an L2 ball with radius max_norm.

    x: [B,S,E] or [B,S,H,D]
    keep_mask: optional [B,S] bool; when provided, only kept rows are capped.
    mode:
      - token: norm over last dim
      - token_head: norm over head dim vectors (same as token for [B,S,H,D], fallback to token for [B,S,E])
      - token_global: for [B,S,H,D], norm over H and D jointly; for [B,S,E], same as token
    """
    if float(max_norm) <= 0.0:
        return x
    if mode not in ("token", "token_head", "token_global"):
        raise ValueError(f"Unsupported cap mode: {mode}")
    if x.ndim not in (3, 4):
        raise ValueError(f"cap_vector_norm expects rank 3 or 4 tensor, got shape={tuple(x.shape)}")

    if mode == "token_global" and x.ndim == 4:
        reduce_dims = (-2, -1)
    else:
        reduce_dims = (-1,)

    x_f = x.float()
    norms = torch.linalg.vector_norm(x_f, ord=2, dim=reduce_dims, keepdim=True)
    scales = (float(max_norm) / (norms + float(eps))).clamp(max=1.0)

    if keep_mask is not None:
        if keep_mask.ndim != 2 or keep_mask.shape[0] != x.shape[0] or keep_mask.shape[1] != x.shape[1]:
            raise ValueError(
                f"keep_mask must be [B,S] matching x[:2]; got mask={tuple(keep_mask.shape)} x={tuple(x.shape)}"
            )
        mask = keep_mask.to(device=x.device, dtype=torch.bool)
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        scales = torch.where(mask, scales, torch.ones_like(scales))

    return x * scales.to(dtype=x.dtype)


class LMConfig:
    """
    Store all LM configurables - including arch & training params, all in one place
    """
    def __init__(
        self, 
        vocab_size: int = 16384, 
        embed_size: int = 768,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        layers: int = 5,
        max_seq_len: int = 2048,
        tie_embeds: bool = False,
        causal_lm: bool = True,
        activation_checkpointing: bool = False,
        attn_impl: str = "sdpa",
        sdp_backend: str = "auto",
        cosine_attn: bool = False,
        v_rmsnorm: bool = False,
        layerscale: bool = False,
        layerscale_init: float = 1e-5,
        dropout: float = 0.1,
        resid_max_norm: Optional[float] = None,
        cap_attn_out_norm: float = 0.0,
        cap_mlp_out_norm: float = 0.0,
        cap_out_mode: str = "token",
        cap_keep_masked: bool = True,
        logit_softcap: float = 0.0,
        config_file: str = None, 
    ):
        # load defaults/ params first
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.tie_embeds = tie_embeds
        self.causal_lm = causal_lm
        self.activation_checkpointing = activation_checkpointing
        self.attn_impl = attn_impl
        self.sdp_backend = sdp_backend
        self.cosine_attn = cosine_attn
        self.v_rmsnorm = v_rmsnorm
        self.layerscale = layerscale
        self.layerscale_init = layerscale_init
        self.dropout = dropout
        self.resid_max_norm = resid_max_norm
        self.cap_attn_out_norm = cap_attn_out_norm
        self.cap_mlp_out_norm = cap_mlp_out_norm
        self.cap_out_mode = cap_out_mode
        self.cap_keep_masked = cap_keep_masked
        self.logit_softcap = logit_softcap
        self._config_file = config_file

        # if given, overwrite loaded params from yaml
        if self._config_file:
            with open(self._config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        if self.embed_size % self.num_heads != 0:
            raise ValueError(
                f"embed_size ({self.embed_size}) must be divisible by num_heads ({self.num_heads})."
            )
        head_dim = self.embed_size // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even for RoPE (embed_size/num_heads)."
            )
        if self.attn_impl not in ("sdpa", "eager"):
            raise ValueError("attn_impl must be 'sdpa' or 'eager'.")
        if self.sdp_backend not in ("auto", "flash", "mem_efficient", "math"):
            raise ValueError("sdp_backend must be one of: auto, flash, mem_efficient, math.")
        if float(self.layerscale_init) < 0.0:
            raise ValueError("layerscale_init must be >= 0.")
        self.dropout = float(self.dropout)
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if self.resid_max_norm is not None:
            self.resid_max_norm = float(self.resid_max_norm)
        self.cap_attn_out_norm = float(self.cap_attn_out_norm)
        self.cap_mlp_out_norm = float(self.cap_mlp_out_norm)
        if self.cap_attn_out_norm < 0.0:
            raise ValueError("cap_attn_out_norm must be >= 0.0.")
        if self.cap_mlp_out_norm < 0.0:
            raise ValueError("cap_mlp_out_norm must be >= 0.0.")
        self.cap_out_mode = str(self.cap_out_mode)
        if self.cap_out_mode not in ("token", "token_head", "token_global"):
            raise ValueError("cap_out_mode must be one of: token, token_head, token_global.")
        self.cap_keep_masked = bool(self.cap_keep_masked)
        self.logit_softcap = float(self.logit_softcap)
        if self.logit_softcap < 0.0:
            raise ValueError("logit_softcap must be >= 0.0.")


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got {dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_len)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cos = freqs.cos()[None, :, :]
        sin = freqs.sin()[None, :, :]
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _get_cos_sin(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        if seq_len > self.cos.size(1):
            raise ValueError(f"RoPE max_len={self.cos.size(1)} is smaller than seq_len={seq_len}.")
        cos = self.cos[:, :seq_len].to(device=device, dtype=dtype)
        sin = self.sin[:, :seq_len].to(device=device, dtype=dtype)
        return cos, sin

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Broadcast cos/sin to match x shape for arbitrary leading dims.
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return x_rot.flatten(-2)

    def apply(self, q: torch.Tensor, k: torch.Tensor):
        cos_q, sin_q = self._get_cos_sin(q.size(1), q.dtype, q.device)
        cos_k, sin_k = self._get_cos_sin(k.size(1), k.dtype, k.device)
        q = self._apply_rotary(q, cos_q, sin_q)
        k = self._apply_rotary(k, cos_k, sin_k)
        return q, k


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: LMConfig, idx=-1):
        super().__init__()

        # MHA sublayer
        E = config.embed_size
        self._num_heads = config.num_heads
        self._head_dim = E // config.num_heads
        self._in_proj = nn.Linear(E, 3 * E)
        self._out_proj = nn.Linear(E, E)
        self._scalar = 1.0 / math.sqrt(self._head_dim)
        self._attn_impl = config.attn_impl
        self._sdp_backend = config.sdp_backend
        self._cosine_attn = bool(getattr(config, "cosine_attn", False))
        self._qk_norm_eps = 1e-6
        self._rmsnorm_eps = 1e-6
        self._layerscale = bool(getattr(config, "layerscale", False))
        self._layerscale_init = float(getattr(config, "layerscale_init", 1e-5))
        self._dropout_p = float(getattr(config, "dropout", 0.1))
        resid_max_norm = getattr(config, "resid_max_norm", None)
        self._resid_max_norm = math.sqrt(E) if resid_max_norm is None else float(resid_max_norm)
        self._cap_attn_out_norm = float(getattr(config, "cap_attn_out_norm", 0.0))
        self._cap_mlp_out_norm = float(getattr(config, "cap_mlp_out_norm", 0.0))
        self._cap_out_mode = str(getattr(config, "cap_out_mode", "token"))
        self._cap_keep_masked = bool(getattr(config, "cap_keep_masked", True))
        self._ls_attn = None
        self._ls_mlp = None
        if self._layerscale:
            self._ls_attn = nn.Parameter(torch.full((E,), self._layerscale_init))
            self._ls_mlp = nn.Parameter(torch.full((E,), self._layerscale_init))
        self._resid_dropout = nn.Dropout(self._dropout_p)
        self._mlp_dropout = nn.Dropout(self._dropout_p)

        # MLP sublayer
        self._mlp = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size*config.mlp_ratio),
            nn.GELU(),
            nn.Linear(config.embed_size*config.mlp_ratio, config.embed_size),
        )

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask=None,
        q_pad_mask=None,
        is_causal=False,
        debug_capture: Optional[Dict[str, Any]] = None,
        capture_attn_scores: bool = False,
        entropy_capture: Optional[Dict[str, Any]] = None,
    ):
        """
            q, k, v: [B, H, S, D]
            pad_mask: [B, S] (True = mask)
        """
        q_len = q.size(-2)
        k_len = k.size(-2)
        key_pad_mask, sdpa_causal, sdpa_mask, blocked = _build_attn_masks(
            pad_mask=pad_mask,
            q_len=q_len,
            k_len=k_len,
            device=q.device,
            is_causal=is_causal,
        )
        q_attn, k_attn = q, k
        score_scale = self._scalar
        if self._cosine_attn:
            q_attn = F.normalize(q_attn, p=2.0, dim=-1, eps=self._qk_norm_eps)
            k_attn = F.normalize(k_attn, p=2.0, dim=-1, eps=self._qk_norm_eps)
            score_scale = 1.0

        raw_scores = None
        valid = None
        masked_scores = None
        if capture_attn_scores or entropy_capture is not None:
            raw_scores = torch.matmul(q_attn.float(), k_attn.float().transpose(-2, -1)) * score_scale
            if blocked is not None:
                blocked_for_scores = blocked
                if blocked_for_scores.size(0) == 1 and raw_scores.size(0) > 1:
                    blocked_for_scores = blocked_for_scores.expand(raw_scores.size(0), -1, -1, -1)
                valid = ~blocked_for_scores
                masked_scores = raw_scores.masked_fill(blocked_for_scores, torch.finfo(raw_scores.dtype).min)
            else:
                valid = torch.ones_like(raw_scores, dtype=torch.bool)
                masked_scores = raw_scores

        if debug_capture is not None:
            q_norm = q.float().norm(dim=-1)
            k_norm = k.float().norm(dim=-1)
            v_norm = v.float().norm(dim=-1)
            q_keep = None if q_pad_mask is None else ~q_pad_mask
            k_keep = None if pad_mask is None else ~pad_mask
            q_mean, q_std = _masked_mean_std(q_norm, q_keep)
            k_mean, k_std = _masked_mean_std(k_norm, k_keep)
            v_mean, v_std = _masked_mean_std(v_norm, k_keep)
            debug_capture["q_mag_mean"] = q_mean
            debug_capture["q_mag_std"] = q_std
            debug_capture["k_mag_mean"] = k_mean
            debug_capture["k_mag_std"] = k_std
            debug_capture["v_mag_mean"] = v_mean
            debug_capture["v_mag_std"] = v_std

            if raw_scores is not None and valid is not None:
                valid_for_scores = valid if valid.shape == raw_scores.shape else valid.expand_as(raw_scores)
                valid_scores = raw_scores[valid_for_scores]
                if valid_scores.numel() > 0:
                    debug_capture["attn_score_mean"] = float(valid_scores.mean().item())
                    debug_capture["attn_score_std"] = float(valid_scores.std(unbiased=False).item())
                    debug_capture["attn_score_min"] = float(valid_scores.min().item())
                    debug_capture["attn_score_max"] = float(valid_scores.max().item())
                    debug_capture["attn_presoftmax_std"] = float(valid_scores.std(unbiased=False).item())
                    debug_capture["attn_presoftmax_max"] = float(valid_scores.max().item())
                    debug_capture["attn_presoftmax_p99"] = float(torch.quantile(valid_scores, 0.99).item())
                else:
                    debug_capture["attn_score_mean"] = float("nan")
                    debug_capture["attn_score_std"] = float("nan")
                    debug_capture["attn_score_min"] = float("nan")
                    debug_capture["attn_score_max"] = float("nan")
                    debug_capture["attn_presoftmax_std"] = float("nan")
                    debug_capture["attn_presoftmax_max"] = float("nan")
                    debug_capture["attn_presoftmax_p99"] = float("nan")

            if capture_attn_scores:
                _, probs = _mean_attn_entropy(masked_scores=masked_scores, valid=valid)
                valid_row = valid.any(dim=-1, keepdim=True)
                probs = torch.where(valid_row, probs, torch.zeros_like(probs))
                if probs.size(0) > 0:
                    debug_capture["attn_prob_grid"] = probs[0].mean(dim=0).detach().cpu()
                score0 = raw_scores[0]
                valid0 = valid[0].expand_as(score0)
                score0 = score0.masked_fill(~valid0, float("nan"))
                score_grid = torch.nanmean(score0, dim=0)
                score_grid = torch.nan_to_num(score_grid, nan=0.0, posinf=0.0, neginf=0.0)
                debug_capture["attn_score_grid"] = score_grid.detach().cpu()

        if entropy_capture is not None:
            attn_entropy, _ = _mean_attn_entropy(masked_scores=masked_scores, valid=valid)
            entropy_capture["attn_entropy"] = attn_entropy

        if self._attn_impl == "sdpa":
            if q.device.type == "cuda" and self._sdp_backend != "auto":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=(self._sdp_backend == "flash"),
                    enable_mem_efficient=(self._sdp_backend == "mem_efficient"),
                    enable_math=(self._sdp_backend == "math"),
                ):
                    return F.scaled_dot_product_attention(
                        q_attn,
                        k_attn,
                        v,
                        attn_mask=sdpa_mask,
                        is_causal=sdpa_causal,
                        scale=score_scale,
                        dropout_p=(self._dropout_p if self.training else 0.0),
                    )
            return F.scaled_dot_product_attention(
                q_attn,
                k_attn,
                v,
                attn_mask=sdpa_mask,
                is_causal=sdpa_causal,
                scale=score_scale,
                dropout_p=(self._dropout_p if self.training else 0.0),
            )
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * score_scale
        if key_pad_mask is not None:
            scores = scores.masked_fill(key_pad_mask, torch.finfo(scores.dtype).min)
        if is_causal:
            causal = torch.triu(
                torch.ones(q_len, k_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )[None, None, :, :]
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self._dropout_p, training=self.training)
        return torch.matmul(probs, v)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask=None,
        rope=None,
        is_causal: bool = False,
        debug_capture: Optional[Dict[str, Any]] = None,
        capture_attn_scores: bool = False,
        entropy_capture: Optional[Dict[str, Any]] = None,
    ):
        """
            x: [B, S, E] tensor
        """
        B, S, E = x.shape
        x_residual = x
        x_pre_norm = x
        x = _rms_norm_last_dim(x, eps=self._rmsnorm_eps)
        _capture_norm_rms(
            debug_capture=debug_capture,
            prefix="norm_attn",
            pre=x_pre_norm,
            post=x,
            keep_mask=(None if pad_mask is None else ~pad_mask),
        )
        qkv = self._in_proj(x)
        wq, wk, wv = qkv.chunk(3, dim=-1)
        wq = wq.view(B, S, self._num_heads, self._head_dim)
        wk = wk.view(B, S, self._num_heads, self._head_dim)
        wv = wv.view(B, S, self._num_heads, self._head_dim)
        if rope is not None:
            wq, wk = rope.apply(wq, wk)
        wq = wq.transpose(1, 2)
        wk = wk.transpose(1, 2)
        wv = wv.transpose(1, 2)
        h = self._attn(
            wq,
            wk,
            wv,
            pad_mask=pad_mask,
            q_pad_mask=pad_mask,
            is_causal=is_causal,
            debug_capture=debug_capture,
            capture_attn_scores=capture_attn_scores,
            entropy_capture=entropy_capture,
        )
        h = h.transpose(1, 2).contiguous().view(B, S, E)
        attn_out = _apply_layerscale(self._out_proj(h), self._ls_attn)
        attn_keep_mask = (~pad_mask) if (pad_mask is not None and self._cap_keep_masked) else None
        attn_out = cap_vector_norm(
            attn_out,
            self._cap_attn_out_norm,
            keep_mask=attn_keep_mask,
            mode=self._cap_out_mode,
        )
        x = x_residual + self._resid_dropout(attn_out)
        if self._resid_max_norm > 0.0:
            x = clamp_residual(x, self._resid_max_norm)

        x_residual = x
        x_pre_norm = x
        x = _rms_norm_last_dim(x, eps=self._rmsnorm_eps)
        _capture_norm_rms(
            debug_capture=debug_capture,
            prefix="norm_mlp",
            pre=x_pre_norm,
            post=x,
            keep_mask=(None if pad_mask is None else ~pad_mask),
        )
        mlp_out = _apply_layerscale(self._mlp(x), self._ls_mlp)
        mlp_keep_mask = (~pad_mask) if (pad_mask is not None and self._cap_keep_masked) else None
        mlp_out = cap_vector_norm(
            mlp_out,
            self._cap_mlp_out_norm,
            keep_mask=mlp_keep_mask,
            mode=self._cap_out_mode,
        )
        x = x_residual + self._mlp_dropout(mlp_out)
        if self._resid_max_norm > 0.0:
            x = clamp_residual(x, self._resid_max_norm)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config: LMConfig, idx=-1):
        # allow different layer configs in same model via idx

        super().__init__()
        # CSA, MHA, & MLP thirds
        E = config.embed_size
        self._num_heads = config.num_heads
        self._head_dim = E // config.num_heads
        self._attn_impl = config.attn_impl
        self._sdp_backend = config.sdp_backend
        self._cosine_attn = bool(getattr(config, "cosine_attn", False))
        self._qk_norm_eps = 1e-6
        self._rmsnorm_eps = 1e-6
        self._layerscale = bool(getattr(config, "layerscale", False))
        self._layerscale_init = float(getattr(config, "layerscale_init", 1e-5))
        self._dropout_p = float(getattr(config, "dropout", 0.1))
        resid_max_norm = getattr(config, "resid_max_norm", None)
        self._resid_max_norm = math.sqrt(E) if resid_max_norm is None else float(resid_max_norm)
        self._cap_attn_out_norm = float(getattr(config, "cap_attn_out_norm", 0.0))
        self._cap_mlp_out_norm = float(getattr(config, "cap_mlp_out_norm", 0.0))
        self._cap_out_mode = str(getattr(config, "cap_out_mode", "token"))
        self._cap_keep_masked = bool(getattr(config, "cap_keep_masked", True))
        self._ls_self_attn = None
        self._ls_cross_attn = None
        self._ls_mlp = None
        if self._layerscale:
            self._ls_self_attn = nn.Parameter(torch.full((E,), self._layerscale_init))
            self._ls_cross_attn = nn.Parameter(torch.full((E,), self._layerscale_init))
            self._ls_mlp = nn.Parameter(torch.full((E,), self._layerscale_init))
        self._resid_dropout = nn.Dropout(self._dropout_p)
        self._mlp_dropout = nn.Dropout(self._dropout_p)

        # CSA sublayer
        self._self_in_proj = nn.Linear(E, 3 * E)
        self._self_out_proj = nn.Linear(E, E)
        self._scalar = 1.0 / math.sqrt(self._head_dim)

        # Enc-MHA sublayer
        self._cross_q = nn.Linear(E, E)
        self._cross_kv = nn.Linear(E, 2 * E)
        self._cross_out_proj = nn.Linear(E, E)

        # MLP sublayer
        self._mlp = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size*config.mlp_ratio),
            nn.GELU(),
            nn.Linear(config.embed_size*config.mlp_ratio, config.embed_size),
        )

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask=None,
        q_pad_mask=None,
        is_causal=False,
        debug_capture: Optional[Dict[str, Any]] = None,
        capture_attn_scores: bool = False,
        entropy_capture: Optional[Dict[str, Any]] = None,
    ):
        q_len = q.size(-2)
        k_len = k.size(-2)
        key_pad_mask, sdpa_causal, sdpa_mask, blocked = _build_attn_masks(
            pad_mask=pad_mask,
            q_len=q_len,
            k_len=k_len,
            device=q.device,
            is_causal=is_causal,
        )
        q_attn, k_attn = q, k
        score_scale = self._scalar
        if self._cosine_attn:
            q_attn = F.normalize(q_attn, p=2.0, dim=-1, eps=self._qk_norm_eps)
            k_attn = F.normalize(k_attn, p=2.0, dim=-1, eps=self._qk_norm_eps)
            score_scale = 1.0

        raw_scores = None
        valid = None
        masked_scores = None
        if capture_attn_scores or entropy_capture is not None:
            raw_scores = torch.matmul(q_attn.float(), k_attn.float().transpose(-2, -1)) * score_scale
            if blocked is not None:
                blocked_for_scores = blocked
                if blocked_for_scores.size(0) == 1 and raw_scores.size(0) > 1:
                    blocked_for_scores = blocked_for_scores.expand(raw_scores.size(0), -1, -1, -1)
                valid = ~blocked_for_scores
                masked_scores = raw_scores.masked_fill(blocked_for_scores, torch.finfo(raw_scores.dtype).min)
            else:
                valid = torch.ones_like(raw_scores, dtype=torch.bool)
                masked_scores = raw_scores

        if debug_capture is not None:
            q_norm = q.float().norm(dim=-1)
            k_norm = k.float().norm(dim=-1)
            v_norm = v.float().norm(dim=-1)
            q_keep = None if q_pad_mask is None else ~q_pad_mask
            k_keep = None if pad_mask is None else ~pad_mask
            q_mean, q_std = _masked_mean_std(q_norm, q_keep)
            k_mean, k_std = _masked_mean_std(k_norm, k_keep)
            v_mean, v_std = _masked_mean_std(v_norm, k_keep)
            debug_capture["q_mag_mean"] = q_mean
            debug_capture["q_mag_std"] = q_std
            debug_capture["k_mag_mean"] = k_mean
            debug_capture["k_mag_std"] = k_std
            debug_capture["v_mag_mean"] = v_mean
            debug_capture["v_mag_std"] = v_std

            if raw_scores is not None and valid is not None:
                valid_for_scores = valid if valid.shape == raw_scores.shape else valid.expand_as(raw_scores)
                valid_scores = raw_scores[valid_for_scores]
                if valid_scores.numel() > 0:
                    debug_capture["attn_score_mean"] = float(valid_scores.mean().item())
                    debug_capture["attn_score_std"] = float(valid_scores.std(unbiased=False).item())
                    debug_capture["attn_score_min"] = float(valid_scores.min().item())
                    debug_capture["attn_score_max"] = float(valid_scores.max().item())
                    debug_capture["attn_presoftmax_std"] = float(valid_scores.std(unbiased=False).item())
                    debug_capture["attn_presoftmax_max"] = float(valid_scores.max().item())
                    debug_capture["attn_presoftmax_p99"] = float(torch.quantile(valid_scores, 0.99).item())
                else:
                    debug_capture["attn_score_mean"] = float("nan")
                    debug_capture["attn_score_std"] = float("nan")
                    debug_capture["attn_score_min"] = float("nan")
                    debug_capture["attn_score_max"] = float("nan")
                    debug_capture["attn_presoftmax_std"] = float("nan")
                    debug_capture["attn_presoftmax_max"] = float("nan")
                    debug_capture["attn_presoftmax_p99"] = float("nan")

            if capture_attn_scores:
                _, probs = _mean_attn_entropy(masked_scores=masked_scores, valid=valid)
                valid_row = valid.any(dim=-1, keepdim=True)
                probs = torch.where(valid_row, probs, torch.zeros_like(probs))
                if probs.size(0) > 0:
                    debug_capture["attn_prob_grid"] = probs[0].mean(dim=0).detach().cpu()
                score0 = raw_scores[0]
                valid0 = valid[0].expand_as(score0)
                score0 = score0.masked_fill(~valid0, float("nan"))
                score_grid = torch.nanmean(score0, dim=0)
                score_grid = torch.nan_to_num(score_grid, nan=0.0, posinf=0.0, neginf=0.0)
                debug_capture["attn_score_grid"] = score_grid.detach().cpu()

        if entropy_capture is not None:
            attn_entropy, _ = _mean_attn_entropy(masked_scores=masked_scores, valid=valid)
            entropy_capture["attn_entropy"] = attn_entropy

        if self._attn_impl == "sdpa":
            if q.device.type == "cuda" and self._sdp_backend != "auto":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=(self._sdp_backend == "flash"),
                    enable_mem_efficient=(self._sdp_backend == "mem_efficient"),
                    enable_math=(self._sdp_backend == "math"),
                ):
                    return F.scaled_dot_product_attention(
                        q_attn,
                        k_attn,
                        v,
                        attn_mask=sdpa_mask,
                        is_causal=sdpa_causal,
                        scale=score_scale,
                        dropout_p=(self._dropout_p if self.training else 0.0),
                    )
            return F.scaled_dot_product_attention(
                q_attn,
                k_attn,
                v,
                attn_mask=sdpa_mask,
                is_causal=sdpa_causal,
                scale=score_scale,
                dropout_p=(self._dropout_p if self.training else 0.0),
            )
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * score_scale
        if key_pad_mask is not None:
            scores = scores.masked_fill(key_pad_mask, torch.finfo(scores.dtype).min)
        if is_causal:
            causal = torch.triu(
                torch.ones(q_len, k_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )[None, None, :, :]
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self._dropout_p, training=self.training)
        return torch.matmul(probs, v)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        self_pad_mask=None,
        enc_pad_mask=None,
        rope=None,
        cross_is_causal: bool = False,
        self_debug_capture: Optional[Dict[str, Any]] = None,
        cross_debug_capture: Optional[Dict[str, Any]] = None,
        capture_self_attn_scores: bool = False,
        capture_cross_attn_scores: bool = False,
        self_entropy_capture: Optional[Dict[str, Any]] = None,
        cross_entropy_capture: Optional[Dict[str, Any]] = None,
    ):
        """
            x: [B, S, E] tensor
            kv: [B, S, E] (cache for that layer only)
        """
        (B, S, E) = x.shape
        x_residual = x
        x_pre_norm = x
        x = _rms_norm_last_dim(x, eps=self._rmsnorm_eps)
        _capture_norm_rms(
            debug_capture=self_debug_capture,
            prefix="norm_self_attn",
            pre=x_pre_norm,
            post=x,
            keep_mask=(None if self_pad_mask is None else ~self_pad_mask),
        )
        qkv = self._self_in_proj(x)
        wq, wk, wv = qkv.chunk(3, dim=-1)
        wq = wq.view(B, S, self._num_heads, self._head_dim)
        wk = wk.view(B, S, self._num_heads, self._head_dim)
        wv = wv.view(B, S, self._num_heads, self._head_dim)
        if rope is not None:
            wq, wk = rope.apply(wq, wk)
        wq = wq.transpose(1, 2)
        wk = wk.transpose(1, 2)
        wv = wv.transpose(1, 2)
        h1 = self._attn(
            wq,
            wk,
            wv,
            pad_mask=self_pad_mask,
            q_pad_mask=self_pad_mask,
            is_causal=True,
            debug_capture=self_debug_capture,
            capture_attn_scores=capture_self_attn_scores,
            entropy_capture=self_entropy_capture,
        )
        h1 = h1.transpose(1, 2).contiguous().view(B, S, E)
        self_attn_out = _apply_layerscale(self._self_out_proj(h1), self._ls_self_attn)
        self_keep_mask = (~self_pad_mask) if (self_pad_mask is not None and self._cap_keep_masked) else None
        self_attn_out = cap_vector_norm(
            self_attn_out,
            self._cap_attn_out_norm,
            keep_mask=self_keep_mask,
            mode=self._cap_out_mode,
        )
        x = x_residual + self._resid_dropout(self_attn_out)
        if self._resid_max_norm > 0.0:
            x = clamp_residual(x, self._resid_max_norm)

        # Cross attention third
        x_residual = x
        x_pre_norm = x
        kv_pre_norm = kv
        x = _rms_norm_last_dim(x, eps=self._rmsnorm_eps)
        kv = _rms_norm_last_dim(kv, eps=self._rmsnorm_eps)
        _capture_norm_rms(
            debug_capture=cross_debug_capture,
            prefix="norm_cross_x",
            pre=x_pre_norm,
            post=x,
            keep_mask=(None if self_pad_mask is None else ~self_pad_mask),
        )
        _capture_norm_rms(
            debug_capture=cross_debug_capture,
            prefix="norm_cross_kv",
            pre=kv_pre_norm,
            post=kv,
            keep_mask=(None if enc_pad_mask is None else ~enc_pad_mask),
        )

        wq2 = self._cross_q(x)
        wk2, wv2 = self._cross_kv(kv).chunk(2, dim=-1)
        wq2 = wq2.view(B, S, self._num_heads, self._head_dim)
        wk2 = wk2.view(B, kv.size(1), self._num_heads, self._head_dim)
        wv2 = wv2.view(B, kv.size(1), self._num_heads, self._head_dim)
        if rope is not None:
            wq2, wk2 = rope.apply(wq2, wk2)
        wq2 = wq2.transpose(1, 2)
        wk2 = wk2.transpose(1, 2)
        wv2 = wv2.transpose(1, 2)
        h2 = self._attn(
            wq2,
            wk2,
            wv2,
            pad_mask=enc_pad_mask,
            q_pad_mask=self_pad_mask,
            is_causal=cross_is_causal,
            debug_capture=cross_debug_capture,
            capture_attn_scores=capture_cross_attn_scores,
            entropy_capture=cross_entropy_capture,
        )
        h2 = h2.transpose(1, 2).contiguous().view(B, S, E)
        cross_attn_out = _apply_layerscale(self._cross_out_proj(h2), self._ls_cross_attn)
        cross_attn_out = cap_vector_norm(
            cross_attn_out,
            self._cap_attn_out_norm,
            keep_mask=self_keep_mask,
            mode=self._cap_out_mode,
        )
        x = x_residual + self._resid_dropout(cross_attn_out)
        if self._resid_max_norm > 0.0:
            x = clamp_residual(x, self._resid_max_norm)

        x_residual = x
        x_pre_norm = x
        x = _rms_norm_last_dim(x, eps=self._rmsnorm_eps)
        _capture_norm_rms(
            debug_capture=cross_debug_capture,
            prefix="norm_mlp",
            pre=x_pre_norm,
            post=x,
            keep_mask=(None if self_pad_mask is None else ~self_pad_mask),
        )
        mlp_out = _apply_layerscale(self._mlp(x), self._ls_mlp)
        mlp_out = cap_vector_norm(
            mlp_out,
            self._cap_mlp_out_norm,
            keep_mask=self_keep_mask,
            mode=self._cap_out_mode,
        )
        x = x_residual + self._mlp_dropout(mlp_out)
        if self._resid_max_norm > 0.0:
            x = clamp_residual(x, self._resid_max_norm)
        return x


class TransformerV1(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self._config = config
        self._logit_softcap = float(getattr(config, "logit_softcap", 0.0))
        # self._embed = nn.Linear(config.vocab_size, config.embed_size, bias=False)
        self._embed = nn.Embedding(config.vocab_size, config.embed_size)
        self._embed_dropout = nn.Dropout(float(getattr(config, "dropout", 0.1)))
        self._rope = RotaryEmbedding(config.embed_size // config.num_heads, max_len=config.max_seq_len)
        self._enc_blocks = nn.ModuleList([TransformerEncoderBlock(config, i) for i in range(config.layers)])
        self._dec_blocks = nn.ModuleList([TransformerDecoderBlock(config, i) for i in range(config.layers)])
        self._unembed = nn.Linear(config.embed_size, config.vocab_size, bias=False)
        if config.tie_embeds:
            # nn.Embedding defaults to N(0, 1); tied LM heads need much smaller scale.
            with torch.no_grad():
                self._embed.weight.mul_(config.embed_size ** -0.5)
            self._unembed.weight = self._embed.weight

    def _lm_head(self, h: torch.Tensor) -> torch.Tensor:
        """
            h: [B, E] tensor (embedded values)

            out: [B, E] tensor (logits)
        """
        logits = self._unembed(h)
        if self._logit_softcap > 0.0:
            cap = self._logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return logits

    def _encode(
        self,
        x: torch.Tensor,
        pad_mask=None,
        is_causal: bool = False,
        debug_state: Optional[Dict[str, Any]] = None,
        metric_state: Optional[Dict[str, Any]] = None,
        debug_layers: Optional[set] = None,
        debug_score_layers: Optional[set] = None,
    ) -> torch.Tensor:
        """
            x: [B, S, E] tensor

            returns kv: [B, L, S, E] tensor  
        """
        kv = list()
        for i, block in enumerate(self._enc_blocks):
            layer_debug = None
            layer_metrics = None
            capture_scores = bool(debug_score_layers is not None and i in debug_score_layers)
            if debug_state is not None:
                include_debug = debug_layers is None or i in debug_layers
                if include_debug:
                    layer_debug = {"layer": i}
                    debug_state["encoder_layers"].append(layer_debug)
            if metric_state is not None:
                layer_metrics = {"layer": i}
                metric_state["encoder_layers"].append(layer_metrics)
            if (
                self.training
                and getattr(self._config, "activation_checkpointing", False)
                and x.requires_grad
                and debug_state is None
                and metric_state is None
            ):
                x = checkpoint(
                    lambda x_in: block(x_in, pad_mask=pad_mask, rope=self._rope, is_causal=is_causal),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    pad_mask=pad_mask,
                    rope=self._rope,
                    is_causal=is_causal,
                    debug_capture=layer_debug,
                    capture_attn_scores=capture_scores,
                    entropy_capture=layer_metrics,
                )
            if debug_state is not None:
                if i == 0:
                    debug_state["hidden"]["enc_l0"] = x.detach()
                if i == len(self._enc_blocks) - 1:
                    debug_state["hidden"]["enc_last"] = x.detach()
            kv.append(x)
        return torch.stack(kv, dim=1)
        
    def _decode(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        self_pad_mask=None,
        enc_pad_mask=None,
        cross_is_causal: bool = False,
        debug_state: Optional[Dict[str, Any]] = None,
        metric_state: Optional[Dict[str, Any]] = None,
        debug_layers: Optional[set] = None,
        debug_score_layers: Optional[set] = None,
    ) -> torch.Tensor:
        """
            x: [B, S, E] tensor
            kv: [B, L, S, S] tensor

            where L = layers (each layer needs kv cache from sister encoder layer)
            returns: [B, S, E] tensor
        """
        for i, block in enumerate(self._dec_blocks):
            kv_i = kv[:, i]
            self_layer_debug = None
            cross_layer_debug = None
            self_layer_metrics = None
            cross_layer_metrics = None
            capture_scores = bool(debug_score_layers is not None and i in debug_score_layers)
            if debug_state is not None:
                include_debug = debug_layers is None or i in debug_layers
                if include_debug:
                    self_layer_debug = {"layer": i}
                    cross_layer_debug = {"layer": i}
                    debug_state["decoder_self_layers"].append(self_layer_debug)
                    debug_state["decoder_cross_layers"].append(cross_layer_debug)
            if metric_state is not None:
                self_layer_metrics = {"layer": i}
                cross_layer_metrics = {"layer": i}
                metric_state["decoder_self_layers"].append(self_layer_metrics)
                metric_state["decoder_cross_layers"].append(cross_layer_metrics)
            if (
                self.training
                and getattr(self._config, "activation_checkpointing", False)
                and x.requires_grad
                and debug_state is None
                and metric_state is None
            ):
                x = checkpoint(
                    lambda x_in, kv_in: block(
                        x_in,
                        kv_in,
                        self_pad_mask=self_pad_mask,
                        enc_pad_mask=enc_pad_mask,
                        rope=self._rope,
                        cross_is_causal=cross_is_causal,
                    ),
                    x,
                    kv_i,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    kv_i,
                    self_pad_mask=self_pad_mask,
                    enc_pad_mask=enc_pad_mask,
                    rope=self._rope,
                    cross_is_causal=cross_is_causal,
                    self_debug_capture=self_layer_debug,
                    cross_debug_capture=cross_layer_debug,
                    capture_self_attn_scores=capture_scores,
                    capture_cross_attn_scores=capture_scores,
                    self_entropy_capture=self_layer_metrics,
                    cross_entropy_capture=cross_layer_metrics,
                )
            if debug_state is not None:
                if i == 0:
                    debug_state["hidden"]["dec_l0"] = x.detach()
                if i == len(self._dec_blocks) - 1:
                    debug_state["hidden"]["dec_last"] = x.detach()
        return x

    def forward(
        self,
        seq: torch.Tensor,
        pad_mask=None,
        return_debug: bool = False,
        return_attn_entropy: bool = False,
        debug_layers: Optional[set] = None,
        debug_score_layers: Optional[set] = None,
    ):
        """
            seq: [B, S] tensor
        """
        (B, S) = seq.shape
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=seq.device, dtype=torch.bool)
        causal_lm = bool(getattr(self._config, "causal_lm", True))
        embeds = self._embed_dropout(self._embed(seq))
        debug_state = None
        metric_state = None
        debug_layers_set = None if debug_layers is None else {int(x) for x in debug_layers}
        debug_score_layers_set = None if debug_score_layers is None else {int(x) for x in debug_score_layers}
        if return_debug:
            if debug_layers_set is None:
                debug_layers_set = set(range(len(self._enc_blocks)))
            if debug_score_layers_set is None:
                last_layer = max(0, len(self._enc_blocks) - 1)
                debug_score_layers_set = {0, last_layer}
            debug_state = {
                "encoder_layers": [],
                "decoder_self_layers": [],
                "decoder_cross_layers": [],
                "hidden": {
                    "embed": embeds.detach(),
                },
            }
        if return_attn_entropy:
            metric_state = {
                "encoder_layers": [],
                "decoder_self_layers": [],
                "decoder_cross_layers": [],
            }

        kv_cache = self._encode(
            embeds,
            pad_mask=pad_mask,
            is_causal=causal_lm,
            debug_state=debug_state,
            metric_state=metric_state,
            debug_layers=debug_layers_set,
            debug_score_layers=debug_score_layers_set,
        )
        out = self._decode(
            embeds,
            kv_cache,
            self_pad_mask=pad_mask,
            enc_pad_mask=pad_mask,
            cross_is_causal=causal_lm,
            debug_state=debug_state,
            metric_state=metric_state,
            debug_layers=debug_layers_set,
            debug_score_layers=debug_score_layers_set,
        )
        logits = self._lm_head(out)
        if return_debug and return_attn_entropy:
            return logits, debug_state, metric_state
        if return_debug:
            return logits, debug_state
        if return_attn_entropy:
            return logits, metric_state
        return logits


if __name__=="__main__":
    # for testing sizes + overfit testing

    B, S, V, E, L = 10, 1024, 16278, 768, 5
    lmc = LMConfig(vocab_size=V, embed_size=E, layers=L)
    t = TransformerV1(lmc)

    total_params, total_bytes = 0, 0
    for name, param in t.named_parameters():
        total_params += param.numel()
        total_bytes += param.nelement() * param.element_size()
    enc_params, enc_bytes = 0, 0
    for name, param in t._enc_blocks.named_parameters():
        enc_params += param.numel()
        enc_bytes += param.nelement() * param.element_size()
    dec_params, dec_bytes = 0, 0
    for name, param in t._dec_blocks.named_parameters():
        dec_params += param.numel()
        dec_bytes += param.nelement() * param.element_size()
    param_size_mb = total_bytes / (1024**2)
    enc_param_size_mb = enc_bytes / (1024**2)
    dec_param_size_mb = dec_bytes / (1024**2)
    latent_param_size_mb = param_size_mb - (enc_param_size_mb + dec_param_size_mb)
    print(f"Total: {total_params:,}")
    print(f"Total size (MB): {param_size_mb:.4f}")
    print(f"Encoder size (MB): {enc_param_size_mb:.4f}")
    print(f"Decoder size (MB): {dec_param_size_mb:.4f}")
    print(f"Emb/other size (MB): {latent_param_size_mb:.4f}")

    # overfit run
    t.train()
    opt = torch.optim.Adam(t.parameters(), lr=0.001, weight_decay=0.0001)

    batch = torch.randint(0, V, (B, S))
    for _ in range(100):
        inputs  = batch[:, :-1]
        targets = batch[:, 1:]

        predictions = t(inputs)

        loss = F.cross_entropy(predictions.transpose(1, 2), targets)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.detach())
