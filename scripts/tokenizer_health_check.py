#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple

from models.bpe_tokenizer import ByteBPETokenizer

try:
    import orjson as _orjson  # type: ignore
except Exception:  # pragma: no cover - optional speedup
    _orjson = None


def _json_loads(line: str) -> Dict:
    if _orjson is not None:
        return _orjson.loads(line)
    return json.loads(line)


def _find_coco_captions_json(anno_dir: str) -> str:
    candidates = [
        os.path.join(anno_dir, "annotations/captions_train2017.json"),
        os.path.join(anno_dir, "annotations/captions_train2014.json"),
        os.path.join(anno_dir, "captions_train2017.json"),
        os.path.join(anno_dir, "captions_train2014.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find COCO captions json under {anno_dir}."
    )


def iter_coco_captions(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for ann in data.get("annotations", []):
        cap = ann.get("caption")
        if cap:
            yield cap


def iter_wiki_jsonl(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = _json_loads(line)
            text = obj.get("text", "")
            if text:
                yield text


def sample_iterable(
    iterable: Iterable[str],
    k: int,
    rng: random.Random,
    mode: str,
    max_items: int = 0,
    accept_prob: float = 0.0,
) -> Tuple[List[str], int]:
    sample: List[str] = []
    seen = 0
    if mode == "head":
        for item in iterable:
            seen += 1
            if max_items and seen > max_items:
                break
            sample.append(item)
            if k > 0 and len(sample) >= k:
                break
        return sample, seen

    if mode == "random":
        if accept_prob <= 0.0:
            if max_items and k > 0:
                accept_prob = min(1.0, float(k) / float(max_items))
            else:
                accept_prob = 0.001
        for item in iterable:
            seen += 1
            if max_items and seen > max_items:
                break
            if rng.random() <= accept_prob:
                sample.append(item)
                if k > 0 and len(sample) >= k:
                    break
        return sample, seen

    if mode != "reservoir":
        raise ValueError(f"Unknown sample mode: {mode}")
    for item in iterable:
        seen += 1
        if max_items and seen > max_items:
            break
        if k <= 0:
            sample.append(item)
            continue
        if len(sample) < k:
            sample.append(item)
        else:
            j = rng.randint(0, seen - 1)
            if j < k:
                sample[j] = item
    return sample, seen


def _percentile(sorted_vals: List[int], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if pct <= 0:
        return float(sorted_vals[0])
    if pct >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d = k - f
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * d)


def _format_piece(piece: str) -> str:
    piece = piece.replace("\\", "\\\\")
    piece = piece.replace("\n", "\\n").replace("\t", "\\t")
    return piece


def _token_to_str(tokenizer: ByteBPETokenizer, idx: int) -> str:
    tok = tokenizer.id_to_token[int(idx)]
    if tok in tokenizer._special_names:
        return tokenizer._special_names[tok]
    return tokenizer.decode([idx], skip_special=False)


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _load_tokenizer(path: str) -> ByteBPETokenizer:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return ByteBPETokenizer.load(path)


def _build_special_maps(tokenizer: ByteBPETokenizer) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for name in tokenizer.special_tokens:
        tok = tokenizer._special_tokens.get(name)
        if tok is None:
            continue
        idx = tokenizer.token_to_id[tok]
        name_to_id[name] = idx
        id_to_name[idx] = name
    return name_to_id, id_to_name


def _summarize_tokens(
    tokenizer: ByteBPETokenizer,
    texts: List[str],
    rng: random.Random,
    roundtrip_samples: int,
    example_samples: int,
    max_chars: int,
    skip_longer_than: int,
    log_slow_ms: float,
    max_slow_examples: int,
) -> None:
    name_to_id, id_to_name = _build_special_maps(tokenizer)
    special_ids = set(name_to_id.values())

    counts = [0 for _ in range(tokenizer.vocab_size)]
    lengths: List[int] = []
    total_chars = 0
    total_tokens = 0
    unk_count = 0
    special_counts: Dict[str, int] = {name: 0 for name in name_to_id}
    skipped_long = 0
    truncated = 0
    slow_examples: List[Tuple[float, int, str]] = []

    def _encode_ids_from_norm(norm: str) -> List[int]:
        tokens = tokenizer._bpe([(b,) for b in norm.encode("utf-8")])
        return [tokenizer.token_to_id[tok] for tok in tokens]

    for text in texts:
        norm = tokenizer._normalize_text(text)
        if skip_longer_than and len(norm) > skip_longer_than:
            skipped_long += 1
            continue
        if max_chars and len(norm) > max_chars:
            norm = norm[:max_chars]
            truncated += 1

        t0 = perf_counter()
        ids = _encode_ids_from_norm(norm)
        t1 = perf_counter()
        if log_slow_ms > 0:
            elapsed_ms = (t1 - t0) * 1000.0
            if elapsed_ms >= log_slow_ms and len(slow_examples) < max_slow_examples:
                preview = norm[:200].replace("\n", " ").replace("\t", " ")
                slow_examples.append((elapsed_ms, len(norm), preview))

        for idx in ids:
            counts[idx] += 1
            if idx in special_ids:
                special_counts[id_to_name[idx]] += 1
            if idx == tokenizer.unk_id:
                unk_count += 1

        total_tokens += len(ids)
        total_chars += len(norm)
        lengths.append(len(ids) + 2)

    non_special_ids = [i for i in range(tokenizer.vocab_size) if i not in special_ids]
    non_special_vocab = len(non_special_ids)
    seen = sum(1 for i in non_special_ids if counts[i] > 0)
    freq1 = sum(1 for i in non_special_ids if counts[i] == 1)
    freq0 = sum(1 for i in non_special_ids if counts[i] == 0)
    freq1_pct = (freq1 / non_special_vocab * 100.0) if non_special_vocab else 0.0

    _print_header("Frequency Summary (Sampled Corpus)")
    print(f"Sample texts: {len(texts)}")
    if skip_longer_than:
        print(f"Skipped (too long): {skipped_long}")
    if max_chars:
        print(f"Truncated (to max chars): {truncated}")
    print(f"Total tokens (no BOS/EOS): {total_tokens}")
    print(f"Non-special vocab size: {non_special_vocab}")
    print(f"Tokens seen at least once: {seen} ({seen / max(1, non_special_vocab) * 100.0:.2f}%)")
    print(f"Tokens with freq=1: {freq1} ({freq1_pct:.2f}% of non-special vocab)")
    print(f"Tokens with freq=0: {freq0} ({freq0 / max(1, non_special_vocab) * 100.0:.2f}% of non-special vocab)")

    top_ids = sorted(non_special_ids, key=lambda i: counts[i], reverse=True)[:50]
    bottom_candidates = [i for i in non_special_ids if counts[i] > 0]
    bottom_ids = sorted(bottom_candidates, key=lambda i: counts[i])[:50]

    _print_header("Top 50 Tokens (by frequency)")
    for rank, idx in enumerate(top_ids, start=1):
        piece = _format_piece(_token_to_str(tokenizer, idx))
        print(f"{rank:>2}. id={idx:>6} freq={counts[idx]:>8} token={repr(piece)}")

    _print_header("Bottom 50 Tokens (by frequency, freq>0 only)")
    if not bottom_ids:
        print("No tokens observed in the sample.")
    else:
        for rank, idx in enumerate(bottom_ids, start=1):
            piece = _format_piece(_token_to_str(tokenizer, idx))
            print(f"{rank:>2}. id={idx:>6} freq={counts[idx]:>8} token={repr(piece)}")

    _print_header("Sequence Length Stats (BOS/EOS included)")
    if lengths:
        lengths_sorted = sorted(lengths)
        avg_len = sum(lengths) / len(lengths)
        med_len = statistics.median(lengths)
        p90 = _percentile(lengths_sorted, 90)
        p95 = _percentile(lengths_sorted, 95)
        p99 = _percentile(lengths_sorted, 99)
        print(f"Avg length: {avg_len:.2f}")
        print(f"Median length: {med_len:.2f}")
        print(f"P90 length: {p90:.2f}")
        print(f"P95 length: {p95:.2f}")
        print(f"P99 length: {p99:.2f}")
        print(f"Min length: {lengths_sorted[0]}")
        print(f"Max length: {lengths_sorted[-1]}")
    else:
        print("No lengths computed (empty sample).")

    _print_header("Compression / UNK / Specials")
    if total_tokens > 0:
        cpt = total_chars / total_tokens
        tpc = total_tokens / max(1, total_chars)
        print(f"Chars per token (normalized text): {cpt:.4f}")
        print(f"Tokens per char (normalized text): {tpc:.4f}")
    else:
        print("Chars-per-token: N/A (no tokens)")
    if total_tokens > 0:
        unk_rate = unk_count / total_tokens * 100.0
        print(f"UNK token rate: {unk_rate:.4f}% ({unk_count} / {total_tokens})")
    else:
        print("UNK token rate: N/A (no tokens)")

    specials_found = {name: c for name, c in special_counts.items() if c > 0}
    if specials_found:
        print("Special tokens found in normal encoding (unexpected):")
        for name, count in specials_found.items():
            print(f"- {name}: {count}")
    else:
        print("Special tokens found in normal encoding: none (OK)")

    _print_header("Round-Trip Checks (decode(encode(text)))")
    if not texts:
        print("No samples to test.")
    else:
        n = min(roundtrip_samples, len(texts))
        sample_texts = rng.sample(texts, n)
        mismatches = 0
        for text in sample_texts:
            enc = tokenizer.encode(text, add_bos=False, add_eos=False)
            dec = tokenizer.decode(enc, skip_special=True)
            expected = tokenizer._normalize_text(text)
            if dec != expected:
                mismatches += 1
        print(f"Samples tested: {n}")
        print(f"Mismatches: {mismatches}")
        if mismatches:
            print("Note: round-trip compares against normalized text.")

    _print_header("Token Boundary Examples")
    if not texts:
        print("No examples to show.")
    else:
        n = min(example_samples, len(texts))
        for i, text in enumerate(rng.sample(texts, n), start=1):
            norm = tokenizer._normalize_text(text)
            if max_chars and len(norm) > max_chars:
                norm = norm[:max_chars]
            enc = [tokenizer.token_to_id[tok] for tok in tokenizer._bpe([(b,) for b in norm.encode("utf-8")])]
            pieces = [_format_piece(_token_to_str(tokenizer, idx)) for idx in enc]
            joined = " | ".join(repr(p) for p in pieces)
            print(f"[Example {i}]")
            print(f"Text: {repr(_format_piece(text))}")
            print(f"Normalized: {repr(_format_piece(norm))}")
            print(f"Token count: {len(enc)}")
            print(f"Tokens: {joined}")
            print("-" * 80)

    if slow_examples:
        _print_header("Slow Encode Examples")
        for elapsed_ms, length, preview in slow_examples:
            print(f"{elapsed_ms:8.2f} ms | chars={length:6d} | {preview!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Health check for a trained BPE tokenizer.")
    ap.add_argument("--tokenizer", type=str, default="", help="Path to tokenizer .pt")
    ap.add_argument("--run-id", type=str, default="", help="Run id under logs/")
    ap.add_argument(
        "--wiki-jsonl",
        type=str,
        default="./data/wiki_coco/articles.jsonl",
        help="Path to Wikipedia JSONL (with {text: ...} per line).",
    )
    ap.add_argument(
        "--coco-json",
        type=str,
        default="",
        help="Path to COCO captions json. If empty, auto-detect under --anno-dir.",
    )
    ap.add_argument(
        "--anno-dir",
        type=str,
        default="./annotations",
        help="Annotation dir for COCO auto-detection.",
    )
    ap.add_argument("--wiki-sample-texts", type=int, default=5000)
    ap.add_argument("--coco-sample-texts", type=int, default=5000)
    ap.add_argument(
        "--wiki-sample-mode",
        type=str,
        choices=["head", "random", "reservoir"],
        default="head",
        help="Sampling mode for wiki: head (fast), random (fast-ish), reservoir (uniform, slow).",
    )
    ap.add_argument(
        "--wiki-max-lines",
        type=int,
        default=200000,
        help="Max wiki lines to scan (0 = no limit).",
    )
    ap.add_argument(
        "--wiki-accept-prob",
        type=float,
        default=0.0,
        help="Random accept prob for --wiki-sample-mode random (0=auto).",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--roundtrip-samples", type=int, default=50)
    ap.add_argument("--example-samples", type=int, default=5)
    ap.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Truncate each normalized text to this many chars (0 = no truncation).",
    )
    ap.add_argument(
        "--skip-longer-than",
        type=int,
        default=0,
        help="Skip texts longer than this many normalized chars (0 = no skip).",
    )
    ap.add_argument(
        "--log-slow-ms",
        type=float,
        default=0.0,
        help="Log any sample whose encoding takes >= this many ms (0 = off).",
    )
    ap.add_argument(
        "--max-slow-examples",
        type=int,
        default=10,
        help="Max slow samples to report when --log-slow-ms is set.",
    )
    args = ap.parse_args()

    tok_path = args.tokenizer
    if not tok_path and args.run_id:
        tok_path = os.path.join("logs", args.run_id, "tokenizer.pt")
    if not tok_path:
        raise SystemExit("Provide --tokenizer or --run-id.")

    tokenizer = _load_tokenizer(tok_path)

    _print_header("Tokenizer Info")
    print(f"Tokenizer path: {tok_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    name_to_id, _ = _build_special_maps(tokenizer)
    print("Special tokens:")
    for name in tokenizer.special_tokens:
        idx = name_to_id.get(name)
        print(f"- {name} (id={idx})")

    rng = random.Random(args.seed)
    texts: List[str] = []

    if args.wiki_jsonl and os.path.isfile(args.wiki_jsonl):
        wiki_samples, wiki_total = sample_iterable(
            iter_wiki_jsonl(args.wiki_jsonl),
            args.wiki_sample_texts,
            rng,
            mode=args.wiki_sample_mode,
            max_items=args.wiki_max_lines,
            accept_prob=args.wiki_accept_prob,
        )
        print(
            f"\nLoaded wiki texts: {len(wiki_samples)} sampled / {wiki_total} seen "
            f"(mode={args.wiki_sample_mode}, max_lines={args.wiki_max_lines})"
        )
        texts.extend(wiki_samples)
    else:
        print(f"\nWiki jsonl not found: {args.wiki_jsonl} (skipping)")

    coco_path = args.coco_json
    if not coco_path:
        try:
            coco_path = _find_coco_captions_json(args.anno_dir)
        except FileNotFoundError:
            coco_path = ""
    if coco_path and os.path.isfile(coco_path):
        coco_samples, coco_total = sample_iterable(
            iter_coco_captions(coco_path),
            args.coco_sample_texts,
            rng,
            mode="reservoir",
        )
        print(f"Loaded COCO captions: {len(coco_samples)} sampled / {coco_total} seen")
        texts.extend(coco_samples)
    else:
        print(f"COCO captions json not found under: {coco_path or args.anno_dir} (skipping)")

    rng.shuffle(texts)
    _summarize_tokens(
        tokenizer,
        texts,
        rng,
        roundtrip_samples=args.roundtrip_samples,
        example_samples=args.example_samples,
        max_chars=args.max_chars,
        skip_longer_than=args.skip_longer_than,
        log_slow_ms=args.log_slow_ms,
        max_slow_examples=args.max_slow_examples,
    )


if __name__ == "__main__":
    main()
