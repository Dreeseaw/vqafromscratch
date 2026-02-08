"""
language modeling components of vqa
"""
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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
class LMConfig:
    """
    Store all LM configurables - including arch & training params, all in one place
    """
    def __init__(
        self, 
        vocab_size: int = 16384, 
        embed_size: int = 768,
        mlp_ratio: int = 4,
        layers: int = 5,
        max_seq_len: int = 2048,
        tie_embeds: bool = False,
        config_file: str = None, 
    ):
        # load defaults/ params first
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.mlp_ratio = mlp_ratio
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.tie_embeds = tie_embeds
        self._config_file = config_file

        # if given, overwrite loaded params from yaml
        if self._config_file:
            with open(self._config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)


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
        # TODO: implement MHA over SDPA
        E = config.embed_size
        self._wk, self._wv, self._wq = nn.Linear(E, E), nn.Linear(E, E), nn.Linear(E, E)
        self._scalar = 1.0 / math.sqrt(E)
        self._ln1 = nn.LayerNorm(E)

        # MLP sublayer
        self._mlp = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size*config.mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(config.embed_size*config.mlp_ratio, config.embed_size),
            nn.ReLU(inplace=True),
        )
        self._ln2 = nn.LayerNorm(E)

    def forward(self, x: torch.Tensor, pad_mask=None, rope=None):
        """
            x: [B, S, E] tensor
        """
        wq = self._wq(x)
        wk = self._wk(x)
        if rope is not None:
            wq, wk = rope.apply(wq, wk)
        wk = torch.transpose(wk, 1, 2)
        wv = self._wv(x)
        scores = torch.bmm(wq, wk) * self._scalar
        if pad_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(pad_mask[:, None, :], mask_value)
        h = torch.bmm(F.softmax(scores, dim=2), wv)
        x = self._ln1(x + h)
        
        x = x + self._mlp(x)
        return self._ln2(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config: LMConfig, idx=-1):
        # allow different layer configs in same model via idx

        super().__init__()
        # CSA, MHA, & MLP thirds
        E = config.embed_size

        # CSA sublayer
        self._wk, self._wv, self._wq = nn.Linear(E, E), nn.Linear(E, E), nn.Linear(E, E)
        self._scalar = 1.0 / math.sqrt(E)
        self._ln1 = nn.LayerNorm(E)

        # Enc-MHA sublayer
        # TODO: implement MHA over SDPA
        self._wk2, self._wv2, self._wq2 = nn.Linear(E, E), nn.Linear(E, E), nn.Linear(E, E)
        self._ln2 = nn.LayerNorm(E)

        # MLP sublayer
        self._mlp = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size*config.mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(config.embed_size*config.mlp_ratio, config.embed_size),
            nn.ReLU(inplace=True),
        )
        self._ln3 = nn.LayerNorm(E)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, self_pad_mask=None, enc_pad_mask=None, rope=None):
        """
            x: [B, S, E] tensor
            kv: [B, S, E] (cache for that layer only)
        """
        (B, S, E) = x.shape
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        wq = self._wq(x)
        wk = self._wk(x)
        if rope is not None:
            wq, wk = rope.apply(wq, wk)
        wk = torch.transpose(wk, 1, 2)
        wv = self._wv(x)
        mask_value = torch.finfo(wq.dtype).min
        scores = torch.bmm(wq, wk) * self._scalar
        if self_pad_mask is not None:
            mask = mask | self_pad_mask[:, None, :]
        masked_scores = scores.masked_fill(mask, mask_value)
        h1 = torch.bmm(F.softmax(masked_scores, dim=2), wv)
        x = self._ln1(x + h1)
        
        wq2 = self._wq2(x)
        wk2 = self._wk2(kv)
        if rope is not None:
            wq2, wk2 = rope.apply(wq2, wk2)
        wk2 = torch.transpose(wk2, 1, 2)
        wv2 = self._wv2(kv)
        scores2 = torch.bmm(wq2, wk2) * self._scalar
        if enc_pad_mask is not None:
            scores2 = scores2.masked_fill(enc_pad_mask[:, None, :], mask_value)
        h2 = torch.bmm(F.softmax(scores2, dim=2), wv2)
        x = self._ln2(x + h2)

        return self._ln3(x + self._mlp(x))


class TransformerV1(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self._config = config
        # self._embed = nn.Linear(config.vocab_size, config.embed_size, bias=False)
        self._embed = nn.Embedding(config.vocab_size, config.embed_size)
        self._rope = RotaryEmbedding(config.embed_size, max_len=config.max_seq_len)
        self._enc_blocks = nn.ModuleList([TransformerEncoderBlock(config, i) for i in range(config.layers)])
        self._dec_blocks = nn.ModuleList([TransformerDecoderBlock(config, i) for i in range(config.layers)])
        self._unembed = nn.Linear(config.embed_size, config.vocab_size, bias=False)
        if config.tie_embeds:
            self._unembed.weight = self._embed.weight

    def _lm_head(self, h: torch.Tensor) -> torch.Tensor:
        """
            h: [B, E] tensor (embedded values)

            out: [B, E] tensor (logits)
        """
        return self._unembed(h)

    def _encode(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        """
            x: [B, S, E] tensor

            returns kv: [B, L, S, E] tensor  
        """
        kv = list()
        for block in self._enc_blocks:
            x = block(x, pad_mask=pad_mask, rope=self._rope)
            kv.append(x.clone())
        return torch.stack(kv, dim=1)
        
    def _decode(self, x: torch.Tensor, kv: torch.Tensor, self_pad_mask=None, enc_pad_mask=None) -> torch.Tensor:
        """
            x: [B, S, E] tensor
            kv: [B, L, S, S] tensor

            where L = layers (each layer needs kv cache from sister encoder layer)
            returns: [B, S, E] tensor
        """
        for i, block in enumerate(self._dec_blocks):
            x = block(x, kv[:, i], self_pad_mask=self_pad_mask, enc_pad_mask=enc_pad_mask, rope=self._rope)
        return x

    def forward(self, seq: torch.Tensor, pad_mask=None):
        """
            seq: [B, S] tensor
        """
        (B, S) = seq.shape
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=seq.device, dtype=torch.bool)
        embeds = self._embed(seq)
        kv_cache = self._encode(embeds, pad_mask=pad_mask)
        out = self._decode(embeds, kv_cache, self_pad_mask=pad_mask, enc_pad_mask=pad_mask)
        return self._lm_head(out)


if __name__=="__main__":
    # quick overfit test

    B, S, V, E, L = 10, 16, 64, 128, 2
    lmc = LMConfig(vocab_size=V, embed_size=E, layers=L)
    t = TransformerV1(lmc)

    t.train()
    opt = torch.optim.Adam(t.parameters(), lr=0.001, weight_decay=0.0001)

    batch = torch.randint(0, V, (B, S))
    for _ in range(10000):
        inputs  = batch[:, :-1]
        targets = batch[:, 1:]

        predictions = t(inputs)

        loss = F.cross_entropy(predictions.transpose(1, 2), targets)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.detach())
