#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import inspect
import io
import json
import multiprocessing as mp
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import orjson as _orjson  # type: ignore
except Exception:  # pragma: no cover - optional speedup
    _orjson = None


TAG_RE = re.compile(r"<[^>\n]+>")
WIKI_LINK_WITH_PIPE_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
WIKI_HEADING_RE = re.compile(r"={2,}([^=]+)={2,}")
WIKI_BOLD_ITALIC_RE = re.compile(r"'{2,}")
WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}\n]*\}\}")
WORD_RE = re.compile(r"\S+")
PARA_SPLIT_RE = re.compile(r"(?:\n\s*){2,}")


def _json_loads(line: str) -> Dict:
    if _orjson is not None:
        return _orjson.loads(line)
    return json.loads(line)


def _is_gzip_path(path: str) -> bool:
    return path.endswith(".gz")


def _strip_gzip_suffix(path: str) -> str:
    if path.endswith(".gz"):
        return path[:-3]
    return path


def _infer_extension(path: str) -> str:
    base = _strip_gzip_suffix(path)
    return os.path.splitext(base)[1].lower()


def _open_text(path: str) -> io.TextIOBase:
    if _is_gzip_path(path):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _strip_wiki_markup(text: str) -> str:
    text = WIKI_LINK_WITH_PIPE_RE.sub(r"\2", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = WIKI_HEADING_RE.sub(r"\1", text)
    text = WIKI_BOLD_ITALIC_RE.sub("", text)
    text = WIKI_TEMPLATE_RE.sub(" ", text)
    return text


def clean_text(text: str, normalization: str) -> str:
    if normalization and normalization.upper() != "OFF":
        text = unicodedata.normalize(normalization.upper(), text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _strip_wiki_markup(text)
    text = text.replace("\t", " ")

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _hash_ids(ids: Sequence[int]) -> str:
    arr = np.asarray(ids, dtype="<i4")
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _stable_unit_interval(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return n / float(1 << 64)


def _assign_split(doc_id: str, split_train: float, split_val: float, split_test: float) -> str:
    total = split_train + split_val + split_test
    if total <= 0:
        raise ValueError("Split ratios must sum to > 0.")
    t = split_train / total
    v = split_val / total
    u = _stable_unit_interval(doc_id)
    if u < t:
        return "train"
    if u < (t + v):
        return "val"
    return "test"


def _split_paragraphs(text: str) -> List[str]:
    parts = PARA_SPLIT_RE.split(text)
    out: List[str] = []
    for part in parts:
        p = part.strip()
        if p:
            out.append(p)
    return out


def _pack_paragraph_ids(
    paragraph_ids: Sequence[Sequence[int]],
    sep_ids: Sequence[int],
    segment_token_cap: int,
) -> List[List[int]]:
    if segment_token_cap <= 0:
        raise ValueError("segment_token_cap must be > 0")
    packed: List[List[int]] = []
    cur: List[int] = []
    for para in paragraph_ids:
        if not para:
            continue
        need = len(para) if not cur else len(sep_ids) + len(para)
        if cur and (len(cur) + need > segment_token_cap):
            packed.append(cur)
            cur = []
        if cur and sep_ids:
            cur.extend(int(x) for x in sep_ids)
        cur.extend(int(x) for x in para)
        if len(cur) >= segment_token_cap:
            packed.append(cur)
            cur = []
    if cur:
        packed.append(cur)
    return packed


def _window_tokens(ids: Sequence[int], window_len: int, stride: int) -> List[List[int]]:
    if window_len <= 0:
        raise ValueError("window_len must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    n = len(ids)
    if n <= window_len:
        return [list(int(x) for x in ids)]
    last_start = n - window_len
    starts = list(range(0, last_start + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != last_start:
        starts.append(last_start)
    windows = [list(int(x) for x in ids[s : s + window_len]) for s in starts]
    for seq in windows:
        if len(seq) != window_len:
            raise AssertionError("Window length mismatch in sliding window generation.")
    return windows


def _percentile_from_hist(length_counts: np.ndarray, q: float) -> int:
    total = int(length_counts.sum())
    if total <= 0:
        return 0
    threshold = max(1, int(np.ceil(total * q)))
    csum = 0
    for length in range(1, int(length_counts.shape[0])):
        csum += int(length_counts[length])
        if csum >= threshold:
            return length
    return int(length_counts.shape[0] - 1)


@dataclass(frozen=True)
class InputSpec:
    path: str
    text_key: str
    word_count_key: str
    txt_docs: str
    json_list_key: str


def _parse_input_spec(arg: str, defaults: InputSpec) -> InputSpec:
    if "::" not in arg:
        return InputSpec(
            path=arg,
            text_key=defaults.text_key,
            word_count_key=defaults.word_count_key,
            txt_docs=defaults.txt_docs,
            json_list_key=defaults.json_list_key,
        )
    parts = arg.split("::")
    path = parts[0]
    text_key = defaults.text_key
    word_count_key = defaults.word_count_key
    txt_docs = defaults.txt_docs
    json_list_key = defaults.json_list_key
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"Invalid input override (missing '='): {part}")
        key, value = part.split("=", 1)
        key = key.strip()
        if key == "text_key":
            text_key = value
        elif key == "word_count_key":
            word_count_key = value
        elif key == "txt_docs":
            if value not in ("file", "line"):
                raise ValueError(f"Invalid txt_docs value: {value}")
            txt_docs = value
        elif key == "json_list_key":
            json_list_key = value
        else:
            raise ValueError(f"Unknown input override key: {key}")
    return InputSpec(
        path=path,
        text_key=text_key,
        word_count_key=word_count_key,
        txt_docs=txt_docs,
        json_list_key=json_list_key,
    )


def _expand_inputs(inputs: Sequence[InputSpec], exts: Sequence[str]) -> List[InputSpec]:
    out: List[InputSpec] = []
    ext_set = {e.lower() for e in exts}
    seen = set()
    for spec in inputs:
        path = spec.path
        if os.path.isfile(path):
            if path in seen:
                raise ValueError(f"Duplicate input path: {path}")
            seen.add(path)
            out.append(spec)
            continue
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Input not found: {path}")
        for root, _, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                ext = _infer_extension(full)
                if ext in ext_set:
                    if full in seen:
                        raise ValueError(f"Duplicate input path: {full}")
                    seen.add(full)
                    out.append(
                        InputSpec(
                            path=full,
                            text_key=spec.text_key,
                            word_count_key=spec.word_count_key,
                            txt_docs=spec.txt_docs,
                            json_list_key=spec.json_list_key,
                        )
                    )
    out.sort(key=lambda s: s.path)
    return out


def _iter_docs_from_txt(path: str, mode: str) -> Iterator[Tuple[str, str, int]]:
    with _open_text(path) as f:
        if mode == "file":
            text = f.read()
            if text:
                yield f"{path}::doc:0", text, _count_words(text)
            return
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield f"{path}::line:{line_idx}", line, _count_words(line)

def _iter_docs_from_jsonl(
    path: str, text_key: str, word_count_key: str
) -> Iterator[Tuple[str, str, Optional[int]]]:
    with _open_text(path) as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = _json_loads(line)
            text = obj.get(text_key)
            if isinstance(text, str) and text:
                wc = obj.get(word_count_key)
                hint = obj.get("doc_id")
                if hint is None:
                    hint = obj.get("id")
                if hint is None:
                    hint = obj.get("title")
                if isinstance(hint, (str, int)):
                    doc_id = f"{path}::line:{line_idx}::{hint}"
                else:
                    doc_id = f"{path}::line:{line_idx}"
                yield doc_id, text, wc if isinstance(wc, int) else None


def _iter_docs_from_json(
    path: str, text_key: str, word_count_key: str, list_key: str
) -> Iterator[Tuple[str, str, Optional[int]]]:
    with _open_text(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, dict):
                text = item.get(text_key)
                if isinstance(text, str) and text:
                    wc = item.get(word_count_key)
                    hint = item.get("doc_id")
                    if hint is None:
                        hint = item.get("id")
                    if hint is None:
                        hint = item.get("title")
                    if isinstance(hint, (str, int)):
                        doc_id = f"{path}::idx:{idx}::{hint}"
                    else:
                        doc_id = f"{path}::idx:{idx}"
                    yield doc_id, text, wc if isinstance(wc, int) else None
        return
    if isinstance(obj, dict):
        if list_key:
            items = obj.get(list_key)
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    if isinstance(item, dict):
                        text = item.get(text_key)
                        if isinstance(text, str) and text:
                            wc = item.get(word_count_key)
                            hint = item.get("doc_id")
                            if hint is None:
                                hint = item.get("id")
                            if hint is None:
                                hint = item.get("title")
                            if isinstance(hint, (str, int)):
                                doc_id = f"{path}::{list_key}:{idx}::{hint}"
                            else:
                                doc_id = f"{path}::{list_key}:{idx}"
                            yield doc_id, text, wc if isinstance(wc, int) else None
                return
        if isinstance(obj.get(text_key), str):
            wc = obj.get(word_count_key)
            hint = obj.get("doc_id")
            if hint is None:
                hint = obj.get("id")
            if hint is None:
                hint = obj.get("title")
            if isinstance(hint, (str, int)):
                doc_id = f"{path}::root::{hint}"
            else:
                doc_id = f"{path}::root"
            yield doc_id, obj[text_key], wc if isinstance(wc, int) else None
            return
        docs = obj.get("documents")
        if isinstance(docs, list):
            for idx, item in enumerate(docs):
                if isinstance(item, dict):
                    text = item.get(text_key)
                    if isinstance(text, str) and text:
                        wc = item.get(word_count_key)
                        hint = item.get("doc_id")
                        if hint is None:
                            hint = item.get("id")
                        if hint is None:
                            hint = item.get("title")
                        if isinstance(hint, (str, int)):
                            doc_id = f"{path}::documents:{idx}::{hint}"
                        else:
                            doc_id = f"{path}::documents:{idx}"
                        yield doc_id, text, wc if isinstance(wc, int) else None
            return
    raise ValueError(f"Unsupported JSON structure in {path}")


def iter_documents(
    path: str, text_key: str, txt_mode: str, word_count_key: str, list_key: str
) -> Iterator[Tuple[str, str, Optional[int]]]:
    ext = _infer_extension(path)
    if ext == ".txt":
        yield from _iter_docs_from_txt(path, txt_mode)
    elif ext == ".jsonl":
        yield from _iter_docs_from_jsonl(path, text_key, word_count_key)
    elif ext == ".json":
        yield from _iter_docs_from_json(path, text_key, word_count_key, list_key)
    else:
        raise ValueError(f"Unsupported file extension: {path}")


def _estimate_total_words(
    files: Sequence[str],
    text_key: str,
    txt_mode: str,
    word_count_key: str,
    list_key: str,
) -> Tuple[int, int]:
    total_words = 0
    total_docs = 0
    for path in files:
        for _doc_id, text, wc in iter_documents(path, text_key, txt_mode, word_count_key, list_key):
            total_docs += 1
            if isinstance(wc, int) and wc >= 0:
                total_words += wc
            else:
                total_words += _count_words(text)
    return total_words, total_docs


@dataclass
class ShardInfo:
    shard_id: int
    num_seqs: int
    num_tokens: int
    min_len: int
    max_len: int


class ShardWriter:
    def __init__(
        self,
        out_dir: str,
        start_id: int,
        max_seqs: int,
        max_tokens: int,
        manifest_path: str,
    ) -> None:
        self.out_dir = out_dir
        self.shards_dir = os.path.join(out_dir, "shards")
        self.max_seqs = max_seqs
        self.max_tokens = max_tokens
        self.manifest_path = manifest_path
        self.shard_id = start_id

        self.tokens: List[int] = []
        self.lengths: List[int] = []
        self.num_tokens = 0
        self.num_seqs = 0
        self.min_len = 0
        self.max_len = 0

        os.makedirs(self.shards_dir, exist_ok=True)

    def _reset(self) -> None:
        self.tokens = []
        self.lengths = []
        self.num_tokens = 0
        self.num_seqs = 0
        self.min_len = 0
        self.max_len = 0

    def _should_flush(self, next_len: int) -> bool:
        if self.num_seqs >= self.max_seqs:
            return True
        if self.num_tokens + next_len > self.max_tokens:
            return True
        return False

    def add(self, seq: Sequence[int]) -> Optional[ShardInfo]:
        if not seq:
            return None
        if self._should_flush(len(seq)):
            info = self.flush()
        else:
            info = None

        self.tokens.extend(int(x) for x in seq)
        self.lengths.append(len(seq))
        self.num_tokens += len(seq)
        self.num_seqs += 1
        if self.min_len == 0 or len(seq) < self.min_len:
            self.min_len = len(seq)
        if len(seq) > self.max_len:
            self.max_len = len(seq)
        return info

    def flush(self) -> Optional[ShardInfo]:
        if self.num_seqs == 0:
            return None
        tokens_arr = np.asarray(self.tokens, dtype=np.int32)
        lengths_arr = np.asarray(self.lengths, dtype=np.int32)

        shard_name = f"shard-{self.shard_id:05d}"
        tokens_path = os.path.join(self.shards_dir, f"{shard_name}.tokens.npy")
        lengths_path = os.path.join(self.shards_dir, f"{shard_name}.lengths.npy")
        np.save(tokens_path, tokens_arr)
        np.save(lengths_path, lengths_arr)

        info = ShardInfo(
            shard_id=self.shard_id,
            num_seqs=self.num_seqs,
            num_tokens=self.num_tokens,
            min_len=self.min_len,
            max_len=self.max_len,
        )
        meta_path = os.path.join(self.shards_dir, f"{shard_name}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info.__dict__, f, indent=2)

        entry = {
            "shard_id": self.shard_id,
            "tokens": os.path.relpath(tokens_path, self.out_dir),
            "lengths": os.path.relpath(lengths_path, self.out_dir),
            "num_seqs": info.num_seqs,
            "num_tokens": info.num_tokens,
            "min_len": info.min_len,
            "max_len": info.max_len,
        }
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self.shard_id += 1
        self._reset()
        return info


def _next_shard_id(shards_dir: str) -> int:
    if not os.path.isdir(shards_dir):
        return 0
    max_id = -1
    for name in os.listdir(shards_dir):
        if not name.startswith("shard-"):
            continue
        if not name.endswith(".tokens.npy"):
            continue
        base = name.split(".tokens.npy")[0]
        try:
            shard_id = int(base.split("shard-")[-1])
        except ValueError:
            continue
        max_id = max(max_id, shard_id)
    return max_id + 1


def _load_hashes(path: str) -> set:
    hashes = set()
    if not os.path.isfile(path):
        return hashes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                hashes.add(line)
    return hashes


def _load_tokenizer(tokenizer_spec: str):
    if os.path.isfile(tokenizer_spec) and tokenizer_spec.endswith(".pt"):
        from models.bpe_tokenizer import ByteBPETokenizer

        return ByteBPETokenizer.load(tokenizer_spec)

    if ":" in tokenizer_spec:
        module_name, attr = tokenizer_spec.split(":", 1)
        mod = __import__(module_name, fromlist=[attr])
        obj = getattr(mod, attr)
        if callable(obj):
            return obj()
        return obj

    raise ValueError(
        "Tokenizer spec must be a .pt file or module:attr callable"
    )


def _build_encode_fn(tokenizer):
    encode_fn = tokenizer.encode
    kwargs = {}
    try:
        sig = inspect.signature(encode_fn)
        if "add_bos" in sig.parameters:
            kwargs["add_bos"] = False
        if "add_eos" in sig.parameters:
            kwargs["add_eos"] = False
        if "add_special_tokens" in sig.parameters:
            kwargs["add_special_tokens"] = False
    except (TypeError, ValueError):
        pass

    def _encode(text: str) -> List[int]:
        out = encode_fn(text, **kwargs)
        if isinstance(out, np.ndarray):
            return out.astype(np.int64).tolist()
        if hasattr(out, "tolist"):
            return out.tolist()
        return list(out)

    return _encode

def _process_doc_impl(
    encode_fn,
    text: str,
    normalization: str,
    max_seq_len: int,
    stride: int,
    segment_token_cap: int,
    sep_ids: Sequence[int],
    dedup_docs: bool,
    add_bos: bool,
    add_eos: bool,
    bos_id: Optional[int],
    eos_id: Optional[int],
) -> Optional[Tuple[Optional[str], List[Sequence[int]]]]:
    cleaned = clean_text(text, normalization)
    if not cleaned:
        return None
    doc_hash = _hash_text(cleaned) if dedup_docs else None
    paragraphs = _split_paragraphs(cleaned)
    if not paragraphs:
        return None
    content_max_len = max_seq_len
    if add_bos:
        content_max_len -= 1
    if content_max_len <= 0:
        raise ValueError("max_seq_len too small to fit special tokens.")
    if stride > content_max_len:
        raise ValueError("stride must be <= max content length after BOS reservation.")

    para_ids: List[List[int]] = []
    for para in paragraphs:
        ids = encode_fn(para)
        if ids:
            para_ids.append([int(x) for x in ids])
    if not para_ids:
        return None

    segments = _pack_paragraph_ids(
        paragraph_ids=para_ids,
        sep_ids=sep_ids,
        segment_token_cap=segment_token_cap,
    )

    seqs: List[Sequence[int]] = []
    for segment_ids in segments:
        ids = list(int(x) for x in segment_ids)
        if add_eos:
            ids.append(int(eos_id))
        windows = _window_tokens(ids, content_max_len, stride)
        for window in windows:
            out = list(window)
            if add_bos:
                out = [int(bos_id)] + out
            if len(out) > max_seq_len:
                raise AssertionError("Window length exceeded max_seq_len.")
            seqs.append(out)
    if not seqs:
        return None
    return doc_hash, seqs


_WORKER_ENCODE = None
_WORKER_CFG: Dict[str, object] = {}


def _worker_init(
    tokenizer_spec: str,
    normalization: str,
    max_seq_len: int,
    stride: int,
    segment_token_cap: int,
    dedup_docs: bool,
    add_bos: bool,
    add_eos: bool,
    bos_id: Optional[int],
    eos_id: Optional[int],
) -> None:
    global _WORKER_ENCODE, _WORKER_CFG
    _WORKER_ENCODE = _build_encode_fn(_load_tokenizer(tokenizer_spec))
    sep_ids = _WORKER_ENCODE("\n")
    _WORKER_CFG = {
        "normalization": normalization,
        "max_seq_len": max_seq_len,
        "stride": stride,
        "segment_token_cap": segment_token_cap,
        "sep_ids": [int(x) for x in sep_ids],
        "dedup_docs": dedup_docs,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }
    try:  # reduce oversubscription if tokenizer uses torch
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def _worker_process_doc(
    payload: Tuple[str, str, Optional[int]]
) -> Tuple[str, Optional[Tuple[Optional[str], List[Sequence[int]]]]]:
    if _WORKER_ENCODE is None:
        raise RuntimeError("Worker tokenizer not initialized.")
    doc_id, text, _wc = payload
    result = _process_doc_impl(
        _WORKER_ENCODE,
        text=text,
        normalization=str(_WORKER_CFG["normalization"]),
        max_seq_len=int(_WORKER_CFG["max_seq_len"]),
        stride=int(_WORKER_CFG["stride"]),
        segment_token_cap=int(_WORKER_CFG["segment_token_cap"]),
        sep_ids=_WORKER_CFG["sep_ids"],  # type: ignore[arg-type]
        dedup_docs=bool(_WORKER_CFG["dedup_docs"]),
        add_bos=bool(_WORKER_CFG["add_bos"]),
        add_eos=bool(_WORKER_CFG["add_eos"]),
        bos_id=_WORKER_CFG["bos_id"],
        eos_id=_WORKER_CFG["eos_id"],
    )
    return doc_id, result


def _token_str(tokenizer, idx: int) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            s = tokenizer.decode([int(idx)])
            s = s.replace("\n", "\\n")
            s = s.replace("\t", "\\t")
            return s
        except Exception:
            pass
    return ""


def _write_meta(out_dir: str, meta: Dict) -> None:
    path = os.path.join(out_dir, "meta.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != meta:
            raise ValueError(
                "Output directory already contains meta.json with different config."
            )
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _print_histogram(length_counts, max_len: int, bin_size: int) -> None:
    if isinstance(length_counts, dict):
        if not length_counts:
            print("(no sequences)")
            return
        total = sum(length_counts.values())
        items = length_counts.items()
    else:
        if length_counts.sum() == 0:
            print("(no sequences)")
            return
        total = int(length_counts.sum())
        items = ((i, int(length_counts[i])) for i in range(1, max_len + 1))

    bins: Dict[Tuple[int, int], int] = {}
    for length, count in items:
        if count <= 0:
            continue
        start = ((length - 1) // bin_size) * bin_size + 1
        end = min(start + bin_size - 1, max_len)
        bins[(start, end)] = bins.get((start, end), 0) + count
    for (start, end) in sorted(bins.keys()):
        count = bins[(start, end)]
        pct = 100.0 * count / max(total, 1)
        print(f"{start:4d}-{end:4d}: {count:10d} ({pct:6.2f}%)")


def _print_token_stats(
    counts: np.ndarray,
    top_n: int,
    tokenizer=None,
    show_token_strings: bool = False,
) -> None:
    total = int(counts.sum())
    if total == 0:
        print("(no tokens)")
        return
    top_idx = np.argsort(counts)[::-1][:top_n]
    print("Top tokens:")
    for idx in top_idx:
        c = int(counts[idx])
        if c == 0:
            continue
        pct = 100.0 * c / total
        tok_str = ""
        if show_token_strings and tokenizer is not None:
            tok_str = _token_str(tokenizer, int(idx))
            if tok_str:
                tok_str = f" {tok_str!r}"
        print(f"  id {int(idx):5d}: {c:10d} ({pct:6.2f}%){tok_str}")

    zero = int(np.sum(counts == 0))
    le1 = int(np.sum(counts <= 1))
    le10 = int(np.sum(counts <= 10))
    print("Tail stats:")
    print(f"  tokens with count == 0: {zero}")
    print(f"  tokens with count <= 1: {le1}")
    print(f"  tokens with count <= 10: {le10}")
    top_sum = int(counts[top_idx].sum())
    print(f"  top-{top_n} coverage: {100.0 * top_sum / total:.2f}%")


def _length_buckets(length_counts: np.ndarray, max_seq_len: int) -> Dict[str, int]:
    b1_end = min(64, max_seq_len)
    b2_start = 65
    b2_end = min(128, max_seq_len)
    b3_start = 129
    b3_end = min(256, max_seq_len)

    out = {
        "1-64": int(length_counts[1 : b1_end + 1].sum()) if b1_end >= 1 else 0,
        "65-128": int(length_counts[b2_start : b2_end + 1].sum()) if b2_end >= b2_start else 0,
        "129-256": int(length_counts[b3_start : b3_end + 1].sum()) if b3_end >= b3_start else 0,
        "257+": int(length_counts[257:].sum()) if max_seq_len >= 257 else 0,
    }
    return out


def _write_split_manifest_json(
    out_dir: str,
    split: str,
    max_seq_len: int,
    stride: int,
    vocab_size: Optional[int],
    docs: int,
    windows: int,
    total_tokens: int,
    length_counts: np.ndarray,
    sample_doc_ids: Sequence[str],
) -> None:
    avg_len = float(total_tokens) / float(max(1, windows))
    hist = _length_buckets(length_counts, max_seq_len)
    p50 = _percentile_from_hist(length_counts, 0.50)
    p95 = _percentile_from_hist(length_counts, 0.95)
    p99 = _percentile_from_hist(length_counts, 0.99)
    payload = {
        "split": split,
        "max_seq_len": int(max_seq_len),
        "stride": int(stride),
        "tokenizer_vocab_size": None if vocab_size is None else int(vocab_size),
        "counts": {
            "docs": int(docs),
            "windows": int(windows),
            "examples": int(windows),
            "total_tokens_prepad": int(total_tokens),
            "avg_len": avg_len,
            "p50_len": int(p50),
            "p95_len": int(p95),
            "p99_len": int(p99),
        },
        "length_histogram": hist,
        "sample_doc_ids": list(sample_doc_ids),
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-tokenize and review a large text corpus for encoder-decoder training."
    )
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help=(
            "Input file(s) or directories. Optional per-input overrides: "
            "path::text_key=...::word_count_key=...::json_list_key=...::txt_docs=line"
        ),
    )
    ap.add_argument("--out-dir", required=True, help="Output directory for shards.")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer .pt path or module:attr.")
    ap.add_argument("--max-seq-len", "--max_seq_len", dest="max_seq_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument(
        "--segment-token-cap",
        type=int,
        default=1024,
        help="Paragraph packing cap before sliding windows.",
    )
    ap.add_argument("--split-train", "--split_train", dest="split_train", type=float, default=0.95)
    ap.add_argument("--split-val", "--split_val", dest="split_val", type=float, default=0.04)
    ap.add_argument("--split-test", "--split_test", dest="split_test", type=float, default=0.01)
    ap.add_argument("--add-bos", action="store_true", help="Prepend BOS to each sequence.")
    ap.add_argument("--add-eos", action="store_true", help="Append EOS to each sequence.")
    ap.add_argument("--no-add-bos", action="store_false", dest="add_bos")
    ap.add_argument("--no-add-eos", action="store_false", dest="add_eos")
    ap.add_argument("--bos-id", type=int, default=None)
    ap.add_argument("--eos-id", type=int, default=None)
    ap.add_argument("--pad-id", type=int, default=None)
    ap.add_argument("--shard-max-seqs", type=int, default=50000)
    ap.add_argument("--shard-max-tokens", type=int, default=20000000)
    ap.add_argument("--text-key", type=str, default="text")
    ap.add_argument("--txt-docs", choices=["file", "line"], default="file")
    ap.add_argument("--word-count-key", type=str, default="word_count")
    ap.add_argument(
        "--json-list-key",
        type=str,
        default="",
        help="If JSON root is a dict, optionally read documents from this list key.",
    )
    ap.add_argument(
        "--extensions",
        type=str,
        default=".txt,.jsonl,.json,.txt.gz,.jsonl.gz",
        help="Comma-separated list of file extensions to include.",
    )
    ap.add_argument(
        "--normalization",
        choices=["NFC", "NFKC", "OFF"],
        default="NFKC",
    )
    ap.add_argument(
        "--drop-last-window",
        action="store_true",
        help="Deprecated: ignored; final end-aligned window is always kept.",
    )
    ap.add_argument("--dedup-docs", action="store_true", dest="dedup_docs")
    ap.add_argument("--dedup-seqs", action="store_true", dest="dedup_seqs")
    ap.add_argument("--no-dedup-docs", action="store_false", dest="dedup_docs")
    ap.add_argument("--no-dedup-seqs", action="store_false", dest="dedup_seqs")
    ap.add_argument("--no-load-dedup", action="store_false", dest="load_dedup")
    ap.add_argument("--no-persist-dedup", action="store_false", dest="persist_dedup")
    ap.add_argument("--no-eta-scan", action="store_false", dest="eta_scan")
    ap.add_argument("--hist-bin", type=int, default=64)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--show-token-strings", action="store_true")
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--log-seconds", type=float, default=300.0)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mp-chunk", type=int, default=16)

    ap.set_defaults(
        dedup_docs=False,
        dedup_seqs=False,
        load_dedup=True,
        persist_dedup=True,
        eta_scan=True,
        add_bos=False,
        add_eos=True,
    )

    args = ap.parse_args()

    if args.max_seq_len <= 0:
        raise SystemExit("--max-seq-len must be > 0")
    if args.stride <= 0:
        raise SystemExit("--stride must be > 0")
    if args.segment_token_cap <= 0:
        raise SystemExit("--segment-token-cap must be > 0")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.mp_chunk <= 0:
        raise SystemExit("--mp-chunk must be > 0")
    if args.split_train < 0 or args.split_val < 0 or args.split_test < 0:
        raise SystemExit("Split ratios must be non-negative.")
    split_sum = args.split_train + args.split_val + args.split_test
    if split_sum <= 0:
        raise SystemExit("Split ratios must sum to > 0.")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = _load_tokenizer(args.tokenizer) if args.workers == 1 else None
    encode_fn = _build_encode_fn(tokenizer) if tokenizer is not None else None
    vocab_size = None
    tok_for_vocab = tokenizer if tokenizer is not None else _load_tokenizer(args.tokenizer)
    if hasattr(tok_for_vocab, "vocab_size"):
        vocab_size = int(tok_for_vocab.vocab_size)
    elif hasattr(tok_for_vocab, "id_to_token"):
        vocab_size = len(tok_for_vocab.id_to_token)

    bos_id = args.bos_id
    eos_id = args.eos_id
    pad_id = args.pad_id
    if bos_id is None and hasattr(tok_for_vocab, "bos_id"):
        bos_id = int(getattr(tok_for_vocab, "bos_id"))
    if eos_id is None and hasattr(tok_for_vocab, "eos_id"):
        eos_id = int(getattr(tok_for_vocab, "eos_id"))
    if pad_id is None and hasattr(tok_for_vocab, "pad_id"):
        pad_id = int(getattr(tok_for_vocab, "pad_id"))

    if args.add_bos and bos_id is None:
        raise SystemExit("BOS requested but --bos-id not set and tokenizer has no bos_id.")
    if args.add_eos and eos_id is None:
        raise SystemExit("EOS requested but --eos-id not set and tokenizer has no eos_id.")
    if pad_id is None:
        raise SystemExit("PAD token id is required (set --pad-id or use tokenizer with pad_id).")
    if eos_id is None:
        raise SystemExit("EOS token id is required (set --eos-id or use tokenizer with eos_id).")
    if int(pad_id) == int(eos_id):
        raise SystemExit("PAD and EOS must be distinct token ids.")

    content_max_len = args.max_seq_len - (1 if args.add_bos else 0)
    if content_max_len <= 0:
        raise SystemExit("--max-seq-len too small for requested special tokens.")
    if args.stride > content_max_len:
        raise SystemExit("--stride must be <= max content length after BOS reservation.")
    if args.drop_last_window:
        print("warning: --drop-last-window is deprecated and ignored (final end-aligned window is always kept).")

    meta_extensions = sorted(
        [e.strip() for e in args.extensions.split(",") if e.strip()]
    )
    meta = {
        "max_seq_len": args.max_seq_len,
        "content_max_len": content_max_len,
        "stride": args.stride,
        "segment_token_cap": args.segment_token_cap,
        "normalization": args.normalization,
        "dedup_docs": bool(args.dedup_docs),
        "dedup_seqs": bool(args.dedup_seqs),
        "text_key": args.text_key,
        "txt_docs": args.txt_docs,
        "word_count_key": args.word_count_key,
        "json_list_key": args.json_list_key,
        "extensions": meta_extensions,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "add_bos": bool(args.add_bos),
        "add_eos": bool(args.add_eos),
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "split_train": args.split_train,
        "split_val": args.split_val,
        "split_test": args.split_test,
    }
    if vocab_size is not None:
        if int(pad_id) >= vocab_size or int(eos_id) >= vocab_size:
            raise SystemExit("PAD/EOS ids must be within tokenizer vocab size.")
    _write_meta(out_dir, meta)

    doc_hashes: set = set()
    seq_hashes: set = set()
    dedup_dir = os.path.join(out_dir, "dedup")
    os.makedirs(dedup_dir, exist_ok=True)
    doc_hash_path = os.path.join(dedup_dir, "doc_hashes.txt")
    seq_hash_path = os.path.join(dedup_dir, "seq_hashes.txt")
    doc_hash_f = None
    seq_hash_f = None

    if args.load_dedup and args.dedup_docs:
        doc_hashes = _load_hashes(doc_hash_path)
    if args.load_dedup and args.dedup_seqs:
        seq_hashes = _load_hashes(seq_hash_path)
    if args.persist_dedup and args.dedup_docs:
        doc_hash_f = open(doc_hash_path, "a", encoding="utf-8")
    if args.persist_dedup and args.dedup_seqs:
        seq_hash_f = open(seq_hash_path, "a", encoding="utf-8")

    extensions = meta_extensions
    default_spec = InputSpec(
        path="",
        text_key=args.text_key,
        word_count_key=args.word_count_key,
        txt_docs=args.txt_docs,
        json_list_key=args.json_list_key,
    )
    input_specs = [_parse_input_spec(arg, default_spec) for arg in args.input]
    inputs = _expand_inputs(input_specs, extensions)
    if not inputs:
        raise SystemExit("No input files found.")

    split_names = ("train", "val", "test")
    split_writers: Dict[str, ShardWriter] = {}
    split_doc_ids: Dict[str, set] = {k: set() for k in split_names}
    split_docs: Dict[str, int] = {k: 0 for k in split_names}
    split_total_seqs: Dict[str, int] = {k: 0 for k in split_names}
    split_total_tokens: Dict[str, int] = {k: 0 for k in split_names}
    split_length_counts: Dict[str, np.ndarray] = {
        k: np.zeros(args.max_seq_len + 1, dtype=np.int64) for k in split_names
    }
    split_sample_doc_ids: Dict[str, List[str]] = {k: [] for k in split_names}
    split_seqs_at_max: Dict[str, int] = {k: 0 for k in split_names}

    for split in split_names:
        split_out = os.path.join(out_dir, split)
        os.makedirs(split_out, exist_ok=True)
        split_shards_dir = os.path.join(split_out, "shards")
        os.makedirs(split_shards_dir, exist_ok=True)
        _write_meta(split_out, {**meta, "split": split})
        next_id = _next_shard_id(split_shards_dir)
        split_writers[split] = ShardWriter(
            out_dir=split_out,
            start_id=next_id,
            max_seqs=args.shard_max_seqs,
            max_tokens=args.shard_max_tokens,
            manifest_path=os.path.join(split_out, "manifest.jsonl"),
        )

    length_counts = np.zeros(args.max_seq_len + 1, dtype=np.int64)
    seqs_at_max = 0
    total_seqs = 0
    total_tokens = 0
    total_docs = 0
    raw_docs = 0
    dup_docs = 0
    dup_seqs = 0
    empty_docs = 0
    doc_to_split: Dict[str, str] = {}

    if vocab_size is not None:
        token_counts = np.zeros(vocab_size, dtype=np.int64)
    else:
        token_counts = None
        token_counter: Dict[int, int] = {}

    start_t = time.perf_counter()
    last_log_t = start_t
    last_docs = 0
    last_seqs = 0
    last_tokens = 0

    total_words_est = 0
    total_docs_est = 0
    if args.eta_scan:
        eta_start = time.perf_counter()
        total_words_est = 0
        total_docs_est = 0
        for spec in inputs:
            words, docs = _estimate_total_words(
                files=[spec.path],
                text_key=spec.text_key,
                txt_mode=spec.txt_docs,
                word_count_key=spec.word_count_key,
                list_key=spec.json_list_key,
            )
            total_words_est += words
            total_docs_est += docs
        eta_elapsed = time.perf_counter() - eta_start
        print(
            f"ETA scan complete: docs={total_docs_est} words={total_words_est} "
            f"(elapsed {eta_elapsed:.1f}s)"
        )

    words_seen = 0

    def _doc_iter() -> Iterator[Tuple[str, str, Optional[int]]]:
        nonlocal raw_docs, words_seen
        for spec in inputs:
            for doc_id, doc, wc in iter_documents(
                spec.path,
                spec.text_key,
                spec.txt_docs,
                spec.word_count_key,
                spec.json_list_key,
            ):
                raw_docs += 1
                if args.max_docs and raw_docs > args.max_docs:
                    return
                if isinstance(wc, int) and wc >= 0:
                    words_seen += wc
                else:
                    words_seen += _count_words(doc)
                yield doc_id, doc, wc

    def _consume_result(
        doc_id: str,
        result: Optional[Tuple[Optional[str], List[Sequence[int]]]],
    ) -> None:
        nonlocal empty_docs, dup_docs, dup_seqs, total_docs, total_seqs, total_tokens, seqs_at_max, token_counts, token_counter, last_log_t, last_docs, last_seqs, last_tokens
        if result is None:
            empty_docs += 1
            return

        doc_hash, doc_seqs = result
        if args.dedup_docs and doc_hash is not None:
            if doc_hash in doc_hashes:
                dup_docs += 1
                return
            doc_hashes.add(doc_hash)
            if doc_hash_f is not None:
                doc_hash_f.write(doc_hash + "\n")

        split_name = _assign_split(
            doc_id,
            split_train=args.split_train,
            split_val=args.split_val,
            split_test=args.split_test,
        )
        prev_split = doc_to_split.get(doc_id)
        if prev_split is not None and prev_split != split_name:
            raise RuntimeError(f"Leak check failed: doc_id {doc_id!r} mapped to both {prev_split} and {split_name}.")
        doc_to_split[doc_id] = split_name
        if doc_id in split_doc_ids[split_name]:
            raise RuntimeError(f"Duplicate doc_id within split {split_name}: {doc_id!r}")
        split_doc_ids[split_name].add(doc_id)

        wrote_any = False
        for seq in doc_seqs:
            if args.dedup_seqs:
                sh = _hash_ids(seq)
                if sh in seq_hashes:
                    dup_seqs += 1
                    continue
                seq_hashes.add(sh)
                if seq_hash_f is not None:
                    seq_hash_f.write(sh + "\n")

            if len(seq) > args.max_seq_len:
                raise AssertionError("window lengths never exceed max_seq_len check failed")
            split_writers[split_name].add(seq)

            seqlen = len(seq)
            wrote_any = True
            length_counts[seqlen] += 1
            total_seqs += 1
            total_tokens += seqlen
            if seqlen == args.max_seq_len:
                seqs_at_max += 1
            split_length_counts[split_name][seqlen] += 1
            split_total_seqs[split_name] += 1
            split_total_tokens[split_name] += seqlen
            if seqlen == args.max_seq_len:
                split_seqs_at_max[split_name] += 1

            if token_counts is not None:
                arr = np.asarray(seq, dtype=np.int64)
                bc = np.bincount(arr, minlength=token_counts.shape[0])
                if bc.shape[0] > token_counts.shape[0]:
                    pad = bc.shape[0] - token_counts.shape[0]
                    token_counts = np.pad(token_counts, (0, pad), mode="constant")
                token_counts[: len(bc)] += bc
            else:
                for tid in seq:
                    token_counter[int(tid)] = token_counter.get(int(tid), 0) + 1

        if wrote_any:
            total_docs += 1
            split_docs[split_name] += 1
            if len(split_sample_doc_ids[split_name]) < 8:
                split_sample_doc_ids[split_name].append(doc_id)
        now = time.perf_counter()
        should_log = False
        if args.log_every and total_docs % args.log_every == 0:
            should_log = True
        if args.log_seconds > 0 and (now - last_log_t) >= args.log_seconds:
            should_log = True
        if should_log:
            elapsed = now - start_t
            delta_t = max(now - last_log_t, 1e-9)
            d_docs = total_docs - last_docs
            d_seqs = total_seqs - last_seqs
            d_tokens = total_tokens - last_tokens
            docs_per_s = d_docs / delta_t
            seqs_per_s = d_seqs / delta_t
            toks_per_s = d_tokens / delta_t
            avg_docs_per_s = total_docs / max(elapsed, 1e-9)
            eta_msg = ""
            if total_words_est > 0 and words_seen > 0:
                avg_words_per_s = words_seen / max(elapsed, 1e-9)
                remaining_words = max(total_words_est - words_seen, 0)
                eta_s = remaining_words / max(avg_words_per_s, 1e-9)
                eta_msg = f" words={words_seen} eta={eta_s/3600.0:.2f}h"
            print(
                "progress "
                f"elapsed={elapsed/3600.0:.2f}h "
                f"docs={total_docs} seqs={total_seqs} tokens={total_tokens}"
                f"{eta_msg} "
                f"rate(d/s,seq/s,tok/s)={docs_per_s:.2f},{seqs_per_s:.2f},{toks_per_s:.2f} "
                f"avg_docs/s={avg_docs_per_s:.2f} "
                f"dups(doc/seq)={dup_docs}/{dup_seqs} empty={empty_docs}"
            )
            last_log_t = now
            last_docs = total_docs
            last_seqs = total_seqs
            last_tokens = total_tokens

    if args.workers == 1:
        sep_ids = encode_fn("\n") if encode_fn is not None else []
        for doc_id, doc, _wc in _doc_iter():
            result = _process_doc_impl(
                encode_fn,
                text=doc,
                normalization=args.normalization,
                max_seq_len=args.max_seq_len,
                stride=args.stride,
                segment_token_cap=args.segment_token_cap,
                sep_ids=sep_ids,
                dedup_docs=args.dedup_docs,
                add_bos=args.add_bos,
                add_eos=args.add_eos,
                bos_id=bos_id,
                eos_id=eos_id,
            )
            _consume_result(doc_id, result)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(
                args.tokenizer,
                args.normalization,
                args.max_seq_len,
                args.stride,
                args.segment_token_cap,
                args.dedup_docs,
                args.add_bos,
                args.add_eos,
                bos_id,
                eos_id,
            ),
        ) as pool:
            for doc_id, result in pool.imap(
                _worker_process_doc, _doc_iter(), chunksize=args.mp_chunk
            ):
                _consume_result(doc_id, result)

    for split in split_names:
        split_writers[split].flush()
    train_val_leak = split_doc_ids["train"].intersection(split_doc_ids["val"])
    train_test_leak = split_doc_ids["train"].intersection(split_doc_ids["test"])
    val_test_leak = split_doc_ids["val"].intersection(split_doc_ids["test"])
    if train_val_leak or train_test_leak or val_test_leak:
        raise RuntimeError("Leak check failed: doc_id appears in multiple splits.")

    for split in split_names:
        bucket = _length_buckets(split_length_counts[split], args.max_seq_len)
        if int(bucket.get("257+", 0)) != 0:
            raise AssertionError(f"{split}: 257+ length bucket must be zero.")
        _write_split_manifest_json(
            out_dir=os.path.join(out_dir, split),
            split=split,
            max_seq_len=args.max_seq_len,
            stride=args.stride,
            vocab_size=vocab_size,
            docs=split_docs[split],
            windows=split_total_seqs[split],
            total_tokens=split_total_tokens[split],
            length_counts=split_length_counts[split],
            sample_doc_ids=split_sample_doc_ids[split],
        )
    if doc_hash_f is not None:
        doc_hash_f.close()
    if seq_hash_f is not None:
        seq_hash_f.close()

    print("\n=== Dataset Health Check ===")
    print(f"Input files: {len(inputs)}")
    print(f"Raw docs seen: {raw_docs}")
    print(f"Docs kept: {total_docs}")
    print(f"Docs dropped (dup): {dup_docs}")
    print(f"Docs dropped (empty): {empty_docs}")
    if total_docs > 0:
        print(
            "Split doc ratios (train/val/test): "
            f"{split_docs['train'] / total_docs:.4f} / "
            f"{split_docs['val'] / total_docs:.4f} / "
            f"{split_docs['test'] / total_docs:.4f}"
        )
    print(f"Sequences produced: {total_seqs}")
    print(f"Sequences dropped (dup): {dup_seqs}")
    print(f"Total tokens: {total_tokens}")
    if total_words_est > 0:
        print(f"Estimated total words (from '{args.word_count_key}'): {total_words_est}")
        print(f"Words seen: {words_seen}")

    if total_seqs > 0:
        frac_max = 100.0 * seqs_at_max / total_seqs
        avg_len = total_tokens / total_seqs
        pad_frac = 1.0 - (total_tokens / (total_seqs * args.max_seq_len))
        print(f"Sequences at max length: {seqs_at_max} ({frac_max:.2f}%)")
        print(f"Average sequence length: {avg_len:.2f}")
        print(f"Estimated padding per batch: {100.0 * pad_frac:.2f}%")

    print("\nSequence length histogram:")
    _print_histogram(length_counts, args.max_seq_len, args.hist_bin)

    print("\nToken frequency distribution:")
    if token_counts is None:
        if token_counter:
            max_id = max(token_counter.keys())
            token_counts = np.zeros(max_id + 1, dtype=np.int64)
            for idx, c in token_counter.items():
                token_counts[idx] = c
        else:
            token_counts = np.zeros(0, dtype=np.int64)

    tokenizer_stats = tok_for_vocab if args.show_token_strings else None
    _print_token_stats(
        token_counts,
        top_n=args.top_n,
        tokenizer=tokenizer_stats,
        show_token_strings=args.show_token_strings,
    )


if __name__ == "__main__":
    main()
