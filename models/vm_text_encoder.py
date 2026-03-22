from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lm import LMConfig, RotaryEmbedding, TransformerEncoderBlock


@dataclass
class VMTextEncoderConfig:
    vocab_size: int
    max_seq_len: int = 64
    dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    rope: bool = True


class VMTextEncoder(nn.Module):
    def __init__(self, cfg: VMTextEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(int(cfg.vocab_size), int(cfg.dim))
        self.pos_embed = None if bool(cfg.rope) else nn.Parameter(torch.zeros(1, int(cfg.max_seq_len), int(cfg.dim)))
        self.embed_dropout = nn.Dropout(float(cfg.dropout))
        block_cfg = LMConfig(
            vocab_size=int(cfg.vocab_size),
            embed_size=int(cfg.dim),
            num_heads=int(cfg.heads),
            mlp_ratio=int(cfg.mlp_ratio),
            layers=int(cfg.layers),
            max_seq_len=int(cfg.max_seq_len),
            causal_lm=False,
            dropout=float(cfg.dropout),
        )
        self.blocks = nn.ModuleList([TransformerEncoderBlock(block_cfg, idx=i) for i in range(int(cfg.layers))])
        head_dim = int(cfg.dim) // int(cfg.heads)
        self.rope = RotaryEmbedding(head_dim, max_len=int(cfg.max_seq_len)) if bool(cfg.rope) else None
        self.norm = nn.LayerNorm(int(cfg.dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.token_embed(input_ids)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, : int(x.shape[1]), :].to(device=x.device, dtype=x.dtype)
        x = self.embed_dropout(x)

        pad_mask = None
        if attention_mask is not None:
            pad_mask = ~attention_mask.to(device=x.device, dtype=torch.bool)
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask, rope=self.rope, is_causal=False)
        x = self.norm(x)

        if attention_mask is not None:
            weights = attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = x.mean(dim=1)
        pooled = F.normalize(pooled.float(), dim=-1)
        return {
            "hidden": x,
            "pooled": pooled,
        }

    def get_config(self) -> Dict[str, object]:
        return asdict(self.cfg)
