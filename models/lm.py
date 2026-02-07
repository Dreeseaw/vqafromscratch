"""
language modeling components of vqa
"""
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # pseudo
            self = yaml.load(self._config_file)


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
        wk = self._wk(x)
        wq = torch.transpose(self._wq(x), 1, 2) 
        wv = self._wv(x)
        h = torch.bmm(F.softmax(torch.bmm(wk, wq) * self._scalar, dim=2), wv)
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
        self._scalar = torch.tensor(1.0 / math.sqrt(E))
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
        # TODO: masking!!! 
        wk = self._wk(x)
        wq = torch.transpose(self._wq(x), 1, 2) 
        wv = self._wv(x)
        h1 = torch.bmm(F.softmax(torch.bmm(wk, wq) * self._scalar, dim=2), wv)
        x = self._ln1(x + h1)
        
        wk2 = self._wk2(kv)
        wq2 = torch.transpose(self._wq2(x), 1, 2) 
        wv2 = self._wv2(kv)
        h2 = torch.bmm(F.softmax(torch.bmm(wk2, wq2) * self._scalar, dim=2), wv2)
        x = self._ln2(x + h2)

        return self._ln3(x + self._mlp(x))


class TransformerV1(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self._config = config
        self._embed = nn.Linear(config.vocab_size, config.embed_size)
        # TODO: positional embeddings
        #self._enc_blocks = nn.Sequential(
        #    *[TransformerEncoderBlock(config, i) for i in range(config.layers)]
        #)
        #self._dec_blocks = nn.Sequential(
        #    *[TransformerDecoderBlock(config, i) for i in range(config.layers)]
        #)
        self._enc_blocks = [TransformerEncoderBlock(config, i) for i in range(config.layers)]
        self._dec_blocks = [TransformerDecoderBlock(config, i) for i in range(config.layers)]
        self._unembed = nn.Linear(config.embed_size, config.vocab_size)
        if config.tie_embeds:
            self._unembed.weight = nn.Parameter(self._embed.weight.t())

    def _lm_head(self, h: torch.Tensor) -> torch.Tensor:
        """
            h: [B, E] tensor (embedded values)

            out: [B, E] tensor (logprobs)
        """
        return F.softmax(self._unembed(h), dim=1)

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
            seq: [B, S, V] tensor
        """
        (B, S, V) = seq.shape
        embeds = self._embed(seq)
        kv_cache = self._encode(embeds)
        out = self._decode(embeds, kv_cache)
        hN = out[:, -1, :]
        return self._lm_head(hN)


if __name__=="__main__":
    B, S, V = 10, 16, 64
    lmc = LMConfig(vocab_size=V, embed_size=128, layers=2)
    t = TransformerV1(lmc)

    batch = torch.randn(B, S, V)
    targets = batch[:, 1:, :]

    t.train()
    opt = torch.optim.Adam(t.parameters(), lr=0.001, weight_decay=0.0001)
    crit = nn.CrossEntropyLoss()

    nt = list()
    for idx in range(1, batch.shape[1]):
        nt.append(t(batch[:, :idx, :]))

    predictions = torch.stack(nt, dim=1)
    loss = crit(targets, predictions)

    opt.zero_grad()
    loss.backward()
    opt.step()
