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
        tie_embeds: bool = False,
        config_file: str = None, 
    ):
        # load defaults/ params first
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.mlp_ratio = mlp_ratio
        self.layers = layers
        self.tie_embeds = tie_embeds
        self._config_file = config_file

        # if given, overwrite loaded params from yaml
        if self._config_file:
            with open(self._config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [B, S, E] tensor
            returns [B, S, E] tensor, positionally encoded
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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

    def forward(self, x: torch.Tensor):
        """
            x: [B, S, E] tensor
        """
        wq = self._wq(x)
        wk = torch.transpose(self._wk(x), 1, 2) 
        wv = self._wv(x)
        h = torch.bmm(F.softmax(torch.bmm(wq, wk) * self._scalar, dim=2), wv)
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

    def forward(self, x: torch.Tensor, kv: torch.Tensor):
        """
            x: [B, S, E] tensor
            kv: [B, S, E] (cache for that layer only)
        """
        (B, S, E) = x.shape
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        wq = self._wq(x)
        wk = torch.transpose(self._wk(x), 1, 2) 
        wv = self._wv(x)
        mask_value = torch.finfo(wq.dtype).min
        masked_scores = (torch.bmm(wq, wk) * self._scalar).masked_fill(mask, mask_value)
        h1 = torch.bmm(F.softmax(masked_scores, dim=2), wv)
        x = self._ln1(x + h1)
        
        wq2 = self._wq2(x)
        wk2 = torch.transpose(self._wk2(kv), 1, 2) 
        wv2 = self._wv2(kv)
        h2 = torch.bmm(F.softmax(torch.bmm(wq2, wk2) * self._scalar, dim=2), wv2)
        x = self._ln2(x + h2)

        return self._ln3(x + self._mlp(x))


class TransformerV1(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self._config = config
        # self._embed = nn.Linear(config.vocab_size, config.embed_size, bias=False)
        self._embed = nn.Embedding(config.vocab_size, config.embed_size)
        self._pos_embed = PositionalEncoding(config.embed_size, dropout=0.0)
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

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [B, S, E] tensor

            returns kv: [B, L, S, E] tensor  
        """
        kv = list()
        for block in self._enc_blocks:
            x = block(x)
            kv.append(x.clone())
        return torch.stack(kv, dim=1)
        
    def _decode(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
            x: [B, S, E] tensor
            kv: [B, L, S, S] tensor

            where L = layers (each layer needs kv cache from sister encoder layer)
            returns: [B, S, E] tensor
        """
        for i, block in enumerate(self._dec_blocks):
            x = block(x, kv[:, i])
        return x

    def forward(self, seq: torch.Tensor):
        """
            seq: [B, S] tensor
        """
        (B, S) = seq.shape
        embeds = self._embed(seq)
        embeds = self._pos_embed(embeds)
        kv_cache = self._encode(embeds)
        out = self._decode(embeds, kv_cache)
        hN = out[:, -1, :]
        return self._lm_head(hN)


if __name__=="__main__":
    B, S, V = 10, 16, 64
    lmc = LMConfig(vocab_size=V, embed_size=128, layers=2)
    t = TransformerV1(lmc)

    batch = torch.randint(0, V, (B, S))
    targets = batch[:, 1:]

    t.train()
    opt = torch.optim.Adam(t.parameters(), lr=0.001, weight_decay=0.0001)
    # crit = nn.CrossEntropyLoss()

    for _ in range(10):
        nt = list()
        for idx in range(1, batch.shape[1]):
            nt.append(t(batch[:, :idx]))

        predictions = torch.stack(nt, dim=1)
        loss = F.cross_entropy(predictions.transpose(1, 2), targets)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.detach())
