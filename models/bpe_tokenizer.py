from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Union
import unicodedata
from time import perf_counter

import torch
import torch.nn as nn

Token = Tuple[int, ...]
Pair = Tuple[Token, Token]

_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u201F": '"',
        "\u00AB": '"',
        "\u00BB": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\uFE63": "-",
        "\uFF0D": "-",
        "\u2044": "/",
        "\u2215": "/",
        "\uFF0F": "/",
        "\u2026": "...",
        "\u00A0": " ",
        "\u2007": " ",
        "\u202F": " ",
        "\u200B": "",
    }
)


@dataclass(frozen=True)
class BPESpecial:
    pad: str = "<pad>"
    unk: str = "<unk>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    mask: str = "<mask>"
    vis: str = "<vis>"
    txt: str = "<txt>"


class ByteBPETokenizer(nn.Module):
    def __init__(
        self,
        num_extra_tokens: int = 16,
        normalize_nfkc: bool = True,
        normalize_punct: bool = True,
        max_numeric_token_len: int = 4,
    ) -> None:
        super().__init__()
        self.special = BPESpecial()
        self.num_extra_tokens = num_extra_tokens
        self.normalize_nfkc = normalize_nfkc
        self.normalize_punct = normalize_punct
        self.max_numeric_token_len = max_numeric_token_len

        self.extra_tokens: List[str] = [
            f"<extra_{i}>" for i in range(self.num_extra_tokens)
        ]
        self._special_token_order: List[str] = [
            self.special.pad,
            self.special.unk,
            self.special.bos,
            self.special.eos,
            self.special.mask,
            self.special.vis,
            self.special.txt,
        ] + self.extra_tokens

        self._special_tokens: Dict[str, Token] = {}
        next_id = -1
        for name in self._special_token_order:
            self._special_tokens[name] = (next_id,)
            next_id -= 1
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
            self._special_tokens[name] for name in self._special_token_order
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

    def _normalize_text(self, text: str) -> str:
        if self.normalize_nfkc:
            text = unicodedata.normalize("NFKC", text)
        if self.normalize_punct:
            text = text.translate(_PUNCT_TRANSLATION)
        return text

    @staticmethod
    def _is_digit_token(tok: Token) -> bool:
        return bool(tok) and all(48 <= b <= 57 for b in tok)

    def _reject_pair(self, pair: Pair) -> bool:
        # Avoid merging long numeric strings into a single token.
        if self.max_numeric_token_len <= 0:
            return False
        combined = pair[0] + pair[1]
        if len(combined) > self.max_numeric_token_len and self._is_digit_token(combined):
            return True
        return False

    def train_bpe(
        self,
        texts: Iterable[str],
        num_merges: int = 8000,
        min_pair_freq: int = 2,
    ) -> None:
        sequences: List[List[Token]] = []
        for text in texts:
            norm = self._normalize_text(text)
            b = norm.encode("utf-8")
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
            best_pair = None
            best_freq = 0
            for pair, freq in pair_counts.items():
                if freq < min_pair_freq:
                    continue
                if self._reject_pair(pair):
                    continue
                if freq > best_freq:
                    best_pair = pair
                    best_freq = freq
            if best_pair is None:
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
        norm = self._normalize_text(text)
        tokens = self._bpe([(b,) for b in norm.encode("utf-8")])
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
            "special_token_order": self._special_token_order,
            "config": {
                "num_extra_tokens": self.num_extra_tokens,
                "normalize_nfkc": self.normalize_nfkc,
                "normalize_punct": self.normalize_punct,
                "max_numeric_token_len": self.max_numeric_token_len,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device=None) -> "ByteBPETokenizer":
        data = torch.load(path, map_location=device if device else "cpu")
        config = data.get("config", {})
        tok = cls(
            num_extra_tokens=int(config.get("num_extra_tokens", 16)),
            normalize_nfkc=bool(config.get("normalize_nfkc", True)),
            normalize_punct=bool(config.get("normalize_punct", True)),
            max_numeric_token_len=int(config.get("max_numeric_token_len", 4)),
        )
        if "special_tokens" in data:
            tok._special_tokens = data["special_tokens"]
            tok._special_names = {v: k for k, v in tok._special_tokens.items()}
            order = data.get("special_token_order")
            if isinstance(order, list) and order:
                tok._special_token_order = order
            else:
                # Backward compatibility: keep known tokens in default order.
                order = []
                for name in tok._special_token_order:
                    if name in tok._special_tokens:
                        order.append(name)
                tok._special_token_order = order
        tok.merges = data.get("merges", [])
        tok._rebuild_vocab()
        return tok

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self._special_tokens[self.special.pad]]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self._special_tokens[self.special.bos]]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self._special_tokens[self.special.eos]]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self._special_tokens[self.special.unk]]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[self._special_tokens[self.special.mask]]

    @property
    def vis_id(self) -> int:
        return self.token_to_id[self._special_tokens[self.special.vis]]

    @property
    def txt_id(self) -> int:
        return self.token_to_id[self._special_tokens[self.special.txt]]

    @property
    def special_tokens(self) -> List[str]:
        return list(self._special_token_order)

    @property
    def config(self) -> Dict[str, Union[int, bool]]:
        return {
            "num_extra_tokens": self.num_extra_tokens,
            "normalize_nfkc": self.normalize_nfkc,
            "normalize_punct": self.normalize_punct,
            "max_numeric_token_len": self.max_numeric_token_len,
        }

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)
