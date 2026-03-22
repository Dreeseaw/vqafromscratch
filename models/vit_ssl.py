from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (int(x.shape[0]),) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x * random_tensor / keep_prob


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(dim), int(hidden_dim))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_dim), int(dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // int(num_heads)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(int(dim), int(dim) * 3)
        self.attn_drop = nn.Dropout(float(attn_dropout))
        self.proj = nn.Linear(int(dim), int(dim))
        self.proj_drop = nn.Dropout(float(proj_dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(round(float(dim) * float(mlp_ratio)))
        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = Attention(int(dim), int(num_heads), attn_dropout=float(attn_dropout), proj_dropout=float(dropout))
        self.drop_path1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(int(dim))
        self.mlp = MLP(int(dim), hidden_dim, dropout=float(dropout))
        self.drop_path2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


@dataclass
class ViTSSLConfig:
    image_size: int = 224
    patch_size: int = 16
    dim: int = 192
    depth: int = 12
    heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path: float = 0.05
    proj_dim: int = 256


class ViTSSLBackbone(nn.Module):
    """
    Small ViT backbone for SSL pretraining and later multimodal use.

    Design choices:
    - patch tokens are the primary output
    - no CLS token is required for forward or downstream fusion
    - pooled features are derived from mean-pooled patch tokens
    """

    def __init__(self, cfg: ViTSSLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_size = int(cfg.image_size)
        self.patch_size = int(cfg.patch_size)
        self.embed_dim = int(cfg.dim)
        self.num_patches_base = (self.image_size // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_base, self.embed_dim))
        self.pos_drop = nn.Dropout(float(cfg.dropout))

        dpr = torch.linspace(0.0, float(cfg.drop_path), steps=int(cfg.depth)).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.embed_dim,
                    num_heads=int(cfg.heads),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                    attn_dropout=float(cfg.attn_dropout),
                    drop_path=float(dpr[i]),
                )
                for i in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        _trunc_normal_(self.pos_embed, std=0.02)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                _trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Conv2d):
                _trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)

    def interpolate_pos_encoding(self, h_tokens: int, w_tokens: int) -> torch.Tensor:
        if int(h_tokens * w_tokens) == int(self.pos_embed.shape[1]):
            return self.pos_embed
        base_hw = int(math.isqrt(int(self.pos_embed.shape[1])))
        pos = self.pos_embed.reshape(1, base_hw, base_hw, self.embed_dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(int(h_tokens), int(w_tokens)), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, int(h_tokens * w_tokens), self.embed_dim)
        return pos

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        h_tokens, w_tokens = int(x.shape[-2]), int(x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        x = x + self.interpolate_pos_encoding(h_tokens, w_tokens).to(dtype=x.dtype, device=x.device)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_pooled(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(images).mean(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(images)

    def get_config(self) -> Dict[str, Any]:
        return asdict(self.cfg)


class DINOProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ) -> None:
        super().__init__()
        layers = []
        cur_dim = int(in_dim)
        depth = max(1, int(nlayers))
        for _ in range(depth - 1):
            layers.append(nn.Linear(cur_dim, int(hidden_dim)))
            layers.append(nn.GELU())
            cur_dim = int(hidden_dim)
        layers.append(nn.Linear(cur_dim, int(bottleneck_dim)))
        self.mlp = nn.Sequential(*layers)
        self.last_norm = nn.functional.normalize
        self.last_layer = nn.utils.weight_norm(nn.Linear(int(bottleneck_dim), int(out_dim), bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                _trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.last_norm(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOStudentTeacher(nn.Module):
    def __init__(
        self,
        backbone_cfg: ViTSSLConfig,
        *,
        out_dim: int = 4096,
        head_hidden_dim: int = 1024,
        head_bottleneck_dim: int = 256,
        head_layers: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = ViTSSLBackbone(backbone_cfg)
        self.head = DINOProjectionHead(
            int(backbone_cfg.dim),
            out_dim=int(out_dim),
            hidden_dim=int(head_hidden_dim),
            bottleneck_dim=int(head_bottleneck_dim),
            nlayers=int(head_layers),
        )

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_tokens(images)

    def forward_pooled(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_pooled(images)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        patch_tokens = self.backbone.forward_tokens(images)
        pooled = patch_tokens.mean(dim=1)
        logits = self.head(pooled)
        return {
            "tokens": patch_tokens,
            "pooled": pooled,
            "logits": logits,
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "backbone": self.backbone.get_config(),
            "head_out_dim": int(self.head.last_layer.out_features),
        }


class DINOCheckpointBackbone(nn.Module):
    """
    Frozen ViTSSL backbone loader for MM bridge experiments.

    Inputs are expected to already be ImageNet-normalized tensors, matching the
    VQAv2 data pipeline. The wrapper only resizes spatially when needed.
    """

    def __init__(self, checkpoint_path: str, *, device: Optional[str] = None) -> None:
        super().__init__()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_data = dict(payload.get("model_config") or {})
        if not cfg_data:
            train_args = dict(payload.get("train_args") or {})
            cfg_data = {
                "image_size": int(train_args.get("image_size", 224)),
                "patch_size": int(train_args.get("patch_size", 16)),
                "dim": int(train_args.get("dim", 192)),
                "depth": int(train_args.get("depth", 12)),
                "heads": int(train_args.get("heads", 3)),
                "mlp_ratio": float(train_args.get("mlp_ratio", 4.0)),
                "dropout": float(train_args.get("dropout", 0.0)),
                "attn_dropout": float(train_args.get("attn_dropout", 0.0)),
                "drop_path": float(train_args.get("drop_path", 0.05)),
            }
        self.cfg = ViTSSLConfig(**cfg_data)
        self.backbone = ViTSSLBackbone(self.cfg)
        state = payload.get("teacher") or payload.get("student")
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} did not contain teacher/student weights.")
        bb_state = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith("backbone.")}
        if not bb_state:
            raise ValueError(f"Checkpoint {checkpoint_path} did not contain backbone.* weights.")
        self.backbone.load_state_dict(bb_state, strict=True)
        self.image_size = int(self.cfg.image_size)
        if device is not None:
            self.to(device)

    def _prepare_inputs(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")
        x = images.detach()
        if tuple(x.shape[-2:]) != (self.image_size, self.image_size):
            x = F.interpolate(
                x.float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=images.dtype)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._prepare_inputs(images)
        use_amp = x.device.type == "cuda" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                tokens = self.backbone.forward_tokens(x)
        else:
            tokens = self.backbone.forward_tokens(x)
        return tokens.float()

    def _encoder(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)
