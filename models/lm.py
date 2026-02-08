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
        num_heads: int = 8,
        mlp_ratio: int = 4,
        layers: int = 5,
        max_seq_len: int = 2048,
        tie_embeds: bool = False,
        attn_impl: str = "sdpa",
        sdp_backend: str = "auto",
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
        self.attn_impl = attn_impl
        self.sdp_backend = sdp_backend
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
        self._ln1 = nn.LayerNorm(E)

        # MLP sublayer
        self._mlp = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size*config.mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(config.embed_size*config.mlp_ratio, config.embed_size),
            nn.ReLU(inplace=True),
        )
        self._ln2 = nn.LayerNorm(E)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_mask=None, is_causal=False):
        """
            q, k, v: [B, H, S, D]
            pad_mask: [B, S] (True = mask)
        """
        attn_mask = None
        q_len = q.size(-2)
        k_len = k.size(-2)
        if pad_mask is not None:
            attn_mask = pad_mask[:, None, None, :].expand(-1, 1, q_len, -1)
        if is_causal:
            causal = torch.triu(
                torch.ones(q_len, k_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )[None, None, :, :]
            attn_mask = causal if attn_mask is None else (attn_mask | causal)
        if self._attn_impl == "sdpa":
            if q.device.type == "cuda" and self._sdp_backend != "auto":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=(self._sdp_backend == "flash"),
                    enable_mem_efficient=(self._sdp_backend == "mem_efficient"),
                    enable_math=(self._sdp_backend == "math"),
                ):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scalar
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
        return torch.matmul(F.softmax(scores, dim=-1), v)

    def forward(self, x: torch.Tensor, pad_mask=None, rope=None):
        """
            x: [B, S, E] tensor
        """
        B, S, E = x.shape
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
        h = self._attn(wq, wk, wv, pad_mask=pad_mask, is_causal=False)
        h = h.transpose(1, 2).contiguous().view(B, S, E)
        x = self._ln1(x + self._out_proj(h))
        
        x = x + self._mlp(x)
        return self._ln2(x)


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

        # CSA sublayer
        self._self_in_proj = nn.Linear(E, 3 * E)
        self._self_out_proj = nn.Linear(E, E)
        self._scalar = 1.0 / math.sqrt(self._head_dim)
        self._ln1 = nn.LayerNorm(E)

        # Enc-MHA sublayer
        self._cross_q = nn.Linear(E, E)
        self._cross_kv = nn.Linear(E, 2 * E)
        self._cross_out_proj = nn.Linear(E, E)
        self._ln2 = nn.LayerNorm(E)

        # MLP sublayer
        self._mlp = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size*config.mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(config.embed_size*config.mlp_ratio, config.embed_size),
            nn.ReLU(inplace=True),
        )
        self._ln3 = nn.LayerNorm(E)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_mask=None, is_causal=False):
        attn_mask = None
        q_len = q.size(-2)
        k_len = k.size(-2)
        if pad_mask is not None:
            attn_mask = pad_mask[:, None, None, :].expand(-1, 1, q_len, -1)
        if is_causal:
            causal = torch.triu(
                torch.ones(q_len, k_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )[None, None, :, :]
            attn_mask = causal if attn_mask is None else (attn_mask | causal)
        if self._attn_impl == "sdpa":
            if q.device.type == "cuda" and self._sdp_backend != "auto":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=(self._sdp_backend == "flash"),
                    enable_mem_efficient=(self._sdp_backend == "mem_efficient"),
                    enable_math=(self._sdp_backend == "math"),
                ):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scalar
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
        return torch.matmul(F.softmax(scores, dim=-1), v)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, self_pad_mask=None, enc_pad_mask=None, rope=None):
        """
            x: [B, S, E] tensor
            kv: [B, S, E] (cache for that layer only)
        """
        (B, S, E) = x.shape
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
        h1 = self._attn(wq, wk, wv, pad_mask=self_pad_mask, is_causal=True)
        h1 = h1.transpose(1, 2).contiguous().view(B, S, E)
        x = self._ln1(x + self._self_out_proj(h1))
        
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
        h2 = self._attn(wq2, wk2, wv2, pad_mask=enc_pad_mask, is_causal=False)
        h2 = h2.transpose(1, 2).contiguous().view(B, S, E)
        x = self._ln2(x + self._cross_out_proj(h2))

        return self._ln3(x + self._mlp(x))


class TransformerV1(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self._config = config
        # self._embed = nn.Linear(config.vocab_size, config.embed_size, bias=False)
        self._embed = nn.Embedding(config.vocab_size, config.embed_size)
        self._rope = RotaryEmbedding(config.embed_size // config.num_heads, max_len=config.max_seq_len)
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
            kv.append(x)
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
