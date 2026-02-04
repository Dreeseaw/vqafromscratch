from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Union
from time import perf_counter

import torch
import torch.nn as nn

Token = Tuple[int, ...]
Pair = Tuple[Token, Token]


@dataclass(frozen=True)
class BPESpecial:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    mask: str = "<mask>"


class ByteBPETokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.special = BPESpecial()
        self._special_tokens: Dict[str, Token] = {
            self.special.pad: (-1,),
            self.special.bos: (-2,),
            self.special.eos: (-3,),
            self.special.mask: (-4,),
        }
        self._special_names: Dict[Token, str] = {
            v: k for k, v in self._special_tokens.items()
        }
        self.merges: List[Pair] = []
        self._rebuild_vocab()

    # -------------------------
    # vocab + merges
    # -------------------------
    def _rebuild_vocab(self) -> None:
        self.id_to_token: List[Token] = [
            self._special_tokens[self.special.pad],
            self._special_tokens[self.special.bos],
            self._special_tokens[self.special.eos],
            self._special_tokens[self.special.mask],
        ]
        self.id_to_token.extend([(i,) for i in range(256)])
        for pair in self.merges:
            self.id_to_token.append(pair[0] + pair[1])
        self.token_to_id: Dict[Token, int] = {
            tok: i for i, tok in enumerate(self.id_to_token)
        }
        self.merge_ranks: Dict[Pair, int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

    def train_bpe(
        self,
        texts: Iterable[str],
        num_merges: int = 8000,
        min_pair_freq: int = 2,
    ) -> None:
        sequences: List[List[Token]] = []
        for text in texts:
            b = text.encode("utf-8")
            if b:
                sequences.append([(byte,) for byte in b])

        self.merges = []
        st = perf_counter()
        for i in range(num_merges):
            if i % 10 == 1: print(f"{i}: {perf_counter()-st}")
            pair_counts: Dict[Pair, int] = {}
            for seq in sequences:
                if len(seq) < 2:
                    continue
                prev = seq[0]
                for tok in seq[1:]:
                    pair = (prev, tok)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    prev = tok
            if not pair_counts:
                break
            best_pair, best_freq = max(pair_counts.items(), key=lambda x: x[1])
            if best_freq < min_pair_freq:
                break
            self.merges.append(best_pair)
            sequences = [self._merge_pair(seq, best_pair) for seq in sequences]

        self._rebuild_vocab()

    # -------------------------
    # encoding
    # -------------------------
    def _merge_pair(self, seq: List[Token], pair: Pair) -> List[Token]:
        a, b = pair
        out: List[Token] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def _bpe(self, tokens: List[Token]) -> List[Token]:
        if len(tokens) < 2 or not self.merges:
            return tokens
        while True:
            best_pair = None
            best_rank = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            tokens = self._merge_pair(tokens, best_pair)
        return tokens

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> torch.LongTensor:
        tokens = self._bpe([(b,) for b in text.encode("utf-8")])
        ids = [self.token_to_id[tok] for tok in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long)

    def decode(
        self, ids: Union[List[int], torch.Tensor], skip_special: bool = True
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if ids and isinstance(ids[0], list):
            ids = [i for row in ids for i in row]

        out_parts: List[str] = []
        buf = bytearray()
        for idx in ids:
            tok = self.id_to_token[int(idx)]
            if tok in self._special_names:
                if not skip_special:
                    if buf:
                        out_parts.append(bytes(buf).decode("utf-8", errors="replace"))
                        buf.clear()
                    out_parts.append(self._special_names[tok])
                continue
            buf.extend(tok)
        if buf:
            out_parts.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(out_parts)

    def forward(
        self,
        text_list: List[str],
        max_len: int = 64,
        return_attention_mask: bool = True,
    ):
        batch_ids: List[List[int]] = []
        for text in text_list:
            ids = self.encode(text, add_bos=True, add_eos=True).tolist()
            if max_len and len(ids) > max_len:
                ids = ids[:max_len]
            batch_ids.append(ids)

        if max_len:
            t = max_len
        else:
            t = max((len(ids) for ids in batch_ids), default=0)

        input_ids = torch.full((len(batch_ids), t), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            if not ids:
                continue
            trunc = ids[:t]
            input_ids[i, : len(trunc)] = torch.tensor(trunc, dtype=torch.long)

        attention_mask = input_ids != self.pad_id
        if return_attention_mask:
            return input_ids, attention_mask
        return input_ids

    # -------------------------
    # persistence
    # -------------------------
    def save(self, path: str) -> None:
        payload = {
            "merges": self.merges,
            "special_tokens": self._special_tokens,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device=None) -> "ByteBPETokenizer":
        data = torch.load(path, map_location=device if device else "cpu")
        tok = cls()
        if "special_tokens" in data:
            tok._special_tokens = data["special_tokens"]
            tok._special_names = {v: k for k, v in tok._special_tokens.items()}
        tok.merges = data.get("merges", [])
        tok._rebuild_vocab()
        return tok

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def mask_id(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)
