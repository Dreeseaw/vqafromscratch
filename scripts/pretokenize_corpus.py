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


def _iter_docs_from_txt(path: str, mode: str) -> Iterator[Tuple[str, int]]:
    with _open_text(path) as f:
        if mode == "file":
            text = f.read()
            if text:
                yield text, _count_words(text)
            return
        for line in f:
            line = line.strip()
            if line:
                yield line, _count_words(line)

def _iter_docs_from_jsonl(
    path: str, text_key: str, word_count_key: str
) -> Iterator[Tuple[str, Optional[int]]]:
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = _json_loads(line)
            text = obj.get(text_key)
            if isinstance(text, str) and text:
                wc = obj.get(word_count_key)
                yield text, wc if isinstance(wc, int) else None


def _iter_docs_from_json(
    path: str, text_key: str, word_count_key: str, list_key: str
) -> Iterator[Tuple[str, Optional[int]]]:
    with _open_text(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                text = item.get(text_key)
                if isinstance(text, str) and text:
                    wc = item.get(word_count_key)
                    yield text, wc if isinstance(wc, int) else None
        return
    if isinstance(obj, dict):
        if list_key:
            items = obj.get(list_key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        text = item.get(text_key)
                        if isinstance(text, str) and text:
                            wc = item.get(word_count_key)
                            yield text, wc if isinstance(wc, int) else None
                return
        if isinstance(obj.get(text_key), str):
            wc = obj.get(word_count_key)
            yield obj[text_key], wc if isinstance(wc, int) else None
            return
        docs = obj.get("documents")
        if isinstance(docs, list):
            for item in docs:
                if isinstance(item, dict):
                    text = item.get(text_key)
                    if isinstance(text, str) and text:
                        wc = item.get(word_count_key)
                        yield text, wc if isinstance(wc, int) else None
            return
    raise ValueError(f"Unsupported JSON structure in {path}")


def iter_documents(
    path: str, text_key: str, txt_mode: str, word_count_key: str, list_key: str
) -> Iterator[Tuple[str, Optional[int]]]:
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
        for text, wc in iter_documents(path, text_key, txt_mode, word_count_key, list_key):
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
    drop_last_window: bool,
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
    ids = encode_fn(cleaned)
    if not ids:
        return None
    content_max_len = max_seq_len
    if add_bos:
        content_max_len -= 1
    if add_eos:
        content_max_len -= 1
    if content_max_len <= 0:
        raise ValueError("max_seq_len too small to fit special tokens.")

    def _wrap(seq: Sequence[int]) -> List[int]:
        out: List[int] = []
        if add_bos:
            out.append(int(bos_id))
        out.extend(int(x) for x in seq)
        if add_eos:
            out.append(int(eos_id))
        return out

    if len(ids) <= content_max_len:
        return doc_hash, [_wrap(ids)]
    seqs: List[Sequence[int]] = []
    start = 0
    while start < len(ids):
        end = start + content_max_len
        if end > len(ids) and drop_last_window:
            break
        seqs.append(_wrap(ids[start:end]))
        start += stride
    return doc_hash, seqs


_WORKER_ENCODE = None
_WORKER_CFG: Dict[str, object] = {}


def _worker_init(
    tokenizer_spec: str,
    normalization: str,
    max_seq_len: int,
    stride: int,
    drop_last_window: bool,
    dedup_docs: bool,
    add_bos: bool,
    add_eos: bool,
    bos_id: Optional[int],
    eos_id: Optional[int],
) -> None:
    global _WORKER_ENCODE, _WORKER_CFG
    _WORKER_ENCODE = _build_encode_fn(_load_tokenizer(tokenizer_spec))
    _WORKER_CFG = {
        "normalization": normalization,
        "max_seq_len": max_seq_len,
        "stride": stride,
        "drop_last_window": drop_last_window,
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


def _worker_process_doc(payload: Tuple[str, Optional[int]]) -> Optional[Tuple[Optional[str], List[Sequence[int]]]]:
    if _WORKER_ENCODE is None:
        raise RuntimeError("Worker tokenizer not initialized.")
    text, _wc = payload
    return _process_doc_impl(
        _WORKER_ENCODE,
        text=text,
        normalization=str(_WORKER_CFG["normalization"]),
        max_seq_len=int(_WORKER_CFG["max_seq_len"]),
        stride=int(_WORKER_CFG["stride"]),
        drop_last_window=bool(_WORKER_CFG["drop_last_window"]),
        dedup_docs=bool(_WORKER_CFG["dedup_docs"]),
        add_bos=bool(_WORKER_CFG["add_bos"]),
        add_eos=bool(_WORKER_CFG["add_eos"]),
        bos_id=_WORKER_CFG["bos_id"],
        eos_id=_WORKER_CFG["eos_id"],
    )


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
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
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
    ap.add_argument("--drop-last-window", action="store_true")
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
        dedup_docs=True,
        dedup_seqs=True,
        load_dedup=True,
        persist_dedup=True,
        eta_scan=True,
        add_bos=False,
        add_eos=False,
    )

    args = ap.parse_args()

    if args.max_seq_len <= 0:
        raise SystemExit("--max-seq-len must be > 0")
    if args.stride <= 0:
        raise SystemExit("--stride must be > 0")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.mp_chunk <= 0:
        raise SystemExit("--mp-chunk must be > 0")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

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

    content_max_len = args.max_seq_len - (1 if args.add_bos else 0) - (1 if args.add_eos else 0)
    if content_max_len <= 0:
        raise SystemExit("--max-seq-len too small for requested special tokens.")
    if args.stride > content_max_len:
        raise SystemExit("--stride must be <= max content length after adding BOS/EOS.")

    meta_extensions = sorted(
        [e.strip() for e in args.extensions.split(",") if e.strip()]
    )
    meta = {
        "max_seq_len": args.max_seq_len,
        "content_max_len": content_max_len,
        "stride": args.stride,
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
    }
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

    next_id = _next_shard_id(shards_dir)
    writer = ShardWriter(
        out_dir=out_dir,
        start_id=next_id,
        max_seqs=args.shard_max_seqs,
        max_tokens=args.shard_max_tokens,
        manifest_path=manifest_path,
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

    def _doc_iter() -> Iterator[Tuple[str, Optional[int]]]:
        nonlocal raw_docs, words_seen
        for spec in inputs:
            for doc, wc in iter_documents(
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
                yield doc, wc

    def _consume_result(result: Optional[Tuple[Optional[str], List[Sequence[int]]]]) -> None:
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

        for seq in doc_seqs:
            if args.dedup_seqs:
                sh = _hash_ids(seq)
                if sh in seq_hashes:
                    dup_seqs += 1
                    continue
                seq_hashes.add(sh)
                if seq_hash_f is not None:
                    seq_hash_f.write(sh + "\n")

            writer.add(seq)

            seqlen = len(seq)
            length_counts[seqlen] += 1
            total_seqs += 1
            total_tokens += seqlen
            if seqlen == args.max_seq_len:
                seqs_at_max += 1

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

        total_docs += 1
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
        for doc, _wc in _doc_iter():
            result = _process_doc_impl(
                encode_fn,
                text=doc,
                normalization=args.normalization,
                max_seq_len=args.max_seq_len,
                stride=args.stride,
                drop_last_window=args.drop_last_window,
                dedup_docs=args.dedup_docs,
                add_bos=args.add_bos,
                add_eos=args.add_eos,
                bos_id=bos_id,
                eos_id=eos_id,
            )
            _consume_result(result)
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
                args.drop_last_window,
                args.dedup_docs,
                args.add_bos,
                args.add_eos,
                bos_id,
                eos_id,
            ),
        ) as pool:
            for result in pool.imap(
                _worker_process_doc, _doc_iter(), chunksize=args.mp_chunk
            ):
                _consume_result(result)

    writer.flush()
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
