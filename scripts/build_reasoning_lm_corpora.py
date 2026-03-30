#!/usr/bin/env python3
"""
Build reasoning-enriched LM pretraining corpora using the existing repo pipeline.

This script:
- downloads / formats SQuAD v2 and MultiNLI into JSONL docs
- generates synthetic counting/arithmetic docs
- token-budgets them with the existing mix_bpe_16k tokenizer
- runs scripts/pretokenize_corpus.py with the same config family as distill256_cleaned2
- validates the resulting shard format
- writes a mix config and a human-readable report

It intentionally uses the repo's existing `data/pretraining/...` layout rather than
creating a parallel `data/lm_pretrain/...` tree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bpe_tokenizer import ByteBPETokenizer


RAW_SQUAD_PATH = ROOT / "data" / "pretraining" / "reasoning_squad.jsonl"
RAW_MULTINLI_PATH = ROOT / "data" / "pretraining" / "reasoning_multinli.jsonl"
RAW_COUNTING_PATH = ROOT / "data" / "pretraining" / "reasoning_counting.jsonl"

SQUAD_OUT_DIR = ROOT / "data" / "pretraining" / "squad256_reasoning"
MULTINLI_OUT_DIR = ROOT / "data" / "pretraining" / "multinli256_reasoning"
COUNTING_OUT_DIR = ROOT / "data" / "pretraining" / "counting256_reasoning"

MIX_CONFIG_PATH = ROOT / "data" / "pretraining" / "reasoning_mix_config.json"
REPORT_MD_PATH = ROOT / "data" / "pretraining" / "reasoning_corpora_report.md"
REPORT_JSON_PATH = ROOT / "data" / "pretraining" / "reasoning_corpora_report.json"

TOKENIZER_PATH = ROOT / "logs" / "mix_bpe_16k" / "tokenizer.pt"

BUCKET_RANGES: Sequence[tuple[int, int]] = (
    (1, 64),
    (65, 128),
    (129, 256),
)

DISTILL_STYLE_PRETOKENIZE_ARGS = [
    "--max-seq-len",
    "256",
    "--stride",
    "64",
    "--window_stride",
    "64",
    "--max_windows_per_doc",
    "3",
    "--window_sampling",
    "random",
    "--segment-token-cap",
    "1024",
    "--extensions",
    ".jsonl",
    "--text-key",
    "text",
    "--word-count-key",
    "word_count",
    "--normalization",
    "NFKC",
    "--clean_wikipedia",
    "0",
    "--clean_markdown",
    "1",
    "--drop_list_pages",
    "1",
    "--drop_disambiguation_pages",
    "1",
    "--min_chars_after_clean",
    "1",
    "--log_clean_stats",
    "1",
    "--seed",
    "42",
    "--no-dedup-docs",
    "--no-dedup-seqs",
    "--no-add-bos",
    "--add-eos",
]


@dataclass
class SourceBuildResult:
    name: str
    raw_path: Path
    out_dir: Path
    raw_docs: int
    raw_token_budget: int
    raw_tokens_selected: int
    pretokenized_train_docs: int
    pretokenized_train_windows: int
    pretokenized_train_tokens: int
    bucket_stats: list[dict[str, Any]]
    raw_spot_checks: list[dict[str, Any]]
    decode_spot_checks: list[dict[str, Any]]


def collapse_ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def load_tokenizer(path: Path) -> ByteBPETokenizer:
    if not path.is_file():
        raise FileNotFoundError(f"Missing tokenizer: {path}")
    return ByteBPETokenizer.load(str(path))


def token_len(tok: ByteBPETokenizer, text: str) -> int:
    return int(tok.encode(text, add_bos=False, add_eos=True).numel())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            written += 1
    return written


def pretokenize(raw_path: Path, out_dir: Path, *, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "scripts.pretokenize_corpus",
        "--input",
        str(raw_path),
        "--out-dir",
        str(out_dir),
        "--tokenizer",
        str(TOKENIZER_PATH),
        "--workers",
        str(max(1, int(workers))),
    ] + DISTILL_STYLE_PRETOKENIZE_ARGS
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def read_manifest_counts(split_dir: Path) -> dict[str, Any]:
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest


def collect_length_stats(split_dir: Path) -> tuple[list[dict[str, Any]], int, int]:
    shard_dir = split_dir / "shards"
    bucket_seq_counts = [0 for _ in BUCKET_RANGES]
    bucket_token_counts = [0 for _ in BUCKET_RANGES]
    total_tokens = 0
    total_seqs = 0
    for lengths_path in sorted(shard_dir.glob("*.lengths.npy")):
        arr = np.load(lengths_path, mmap_mode="r")
        vals = np.asarray(arr, dtype=np.int64)
        total_tokens += int(vals.sum())
        total_seqs += int(vals.shape[0])
        for i, (lo, hi) in enumerate(BUCKET_RANGES):
            mask = (vals >= lo) & (vals <= hi)
            bucket_seq_counts[i] += int(mask.sum())
            if mask.any():
                bucket_token_counts[i] += int(vals[mask].sum())
    stats: list[dict[str, Any]] = []
    denom = float(total_tokens) if total_tokens > 0 else 1.0
    for (lo, hi), seqs, toks in zip(BUCKET_RANGES, bucket_seq_counts, bucket_token_counts):
        stats.append(
            {
                "range": [int(lo), int(hi)],
                "seqs": int(seqs),
                "tokens": int(toks),
                "prob": float(toks / denom),
            }
        )
    return stats, total_seqs, total_tokens


def validate_pretokenized_dir(path: Path) -> dict[str, Any]:
    from train.train_transformer import load_dataset

    required = [
        path / "meta.json",
        path / "train" / "manifest.json",
        path / "train" / "manifest.jsonl",
        path / "train" / "meta.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pretokenized artifacts under {path}: {missing}")
    ds = load_dataset(str(path / "train"), 256)
    if len(ds) <= 0:
        raise RuntimeError(f"Loaded empty dataset from {path}")
    first = ds[0]
    seq = first[0] if isinstance(first, tuple) else first
    seqlen = int(getattr(seq, "numel", lambda: len(seq))())
    return {"path": str(path), "train_examples": int(len(ds)), "first_seq_len": seqlen}


def sample_raw_rows(path: Path, count: int, *, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        return []
    rng = random.Random(seed)
    if len(rows) <= count:
        return rows
    return rng.sample(rows, count)


def build_decode_checks(tok: ByteBPETokenizer, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in raw_rows:
        text = str(row["text"])
        ids = tok.encode(text, add_bos=False, add_eos=True)
        out.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "seq_len": int(ids.numel()),
                "raw_text": text,
                "decoded_text": tok.decode(ids),
                "token_ids_head": ids[:32].tolist(),
            }
        )
    return out


def format_squad_row(row: dict[str, Any]) -> tuple[str, str] | None:
    context = collapse_ws(row.get("context", ""))
    question = collapse_ws(row.get("question", ""))
    answers = row.get("answers") or {}
    texts = answers.get("text") or []
    answer = collapse_ws(texts[0]) if texts else ""
    if not context or not question or not answer:
        return None
    text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    row_id = f"squad_v2:train:{row.get('id', '')}"
    return row_id, text


def count_blank_answer_rows(path: Path) -> int:
    blank = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = str(row.get("text", ""))
            for part in text.splitlines():
                if part.startswith("Answer:") and not part[len("Answer:") :].strip():
                    blank += 1
                    break
    return blank


def format_multinli_row(row: dict[str, Any], label_names: Sequence[str]) -> tuple[str, str] | None:
    label = int(row.get("label", -1))
    if label < 0 or label >= len(label_names):
        return None
    premise = collapse_ws(row.get("premise", ""))
    hypothesis = collapse_ws(row.get("hypothesis", ""))
    relation = collapse_ws(label_names[label])
    if not premise or not hypothesis or not relation:
        return None
    pair_id = row.get("pairID") or row.get("promptID") or ""
    row_id = f"multinli:train:{pair_id}"
    text = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation: {relation}"
    return row_id, text


ANIMALS = ["dog", "cat", "horse", "bird", "cow", "sheep", "elephant", "zebra", "giraffe", "bear"]
VEHICLES = ["car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"]
FOODS = ["apple", "banana", "orange", "pizza", "sandwich", "cake", "donut", "broccoli", "carrot"]
FURNITURE = ["chair", "table", "bench", "couch", "bed", "toilet", "sink"]
PEOPLE = ["man", "woman", "boy", "girl", "person", "child"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]

NOUN_GROUPS = [ANIMALS, VEHICLES, FOODS, FURNITURE, PEOPLE]


def generate_counting_text(rng: random.Random) -> str:
    template_type = rng.choice(["objects", "items", "word_total", "arith"])
    if template_type == "objects":
        nouns = rng.choice(NOUN_GROUPS)
        target = rng.choice(nouns)
        n = rng.randint(3, 15)
        items = [rng.choice(nouns) for _ in range(n)]
        if target not in items:
            items[rng.randrange(len(items))] = target
        answer = sum(1 for x in items if x == target)
        return f"Objects: {', '.join(items)}\nCount of {target}: {answer}"
    if template_type == "items":
        nouns = rng.choice(NOUN_GROUPS)
        noun = rng.choice(nouns)
        target_color = rng.choice(COLORS)
        n = rng.randint(4, 14)
        items = [f"{rng.choice(COLORS)} {noun}" for _ in range(n)]
        if not any(x == f"{target_color} {noun}" for x in items):
            items[rng.randrange(len(items))] = f"{target_color} {noun}"
        answer = sum(1 for x in items if x == f"{target_color} {noun}")
        return f"Items: {', '.join(items)}\nHow many {target_color} {noun}: {answer}"
    if template_type == "word_total":
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        noun_a = rng.choice(rng.choice(NOUN_GROUPS))
        noun_b = rng.choice(rng.choice(NOUN_GROUPS))
        return f"Question: There are {a} {noun_a}s and {b} {noun_b}s. How many things total?\nAnswer: {a + b}"
    a = rng.randint(1, 20)
    b = rng.randint(1, 20)
    op = rng.choice(["+", "-", "+"])
    if op == "-" and b > a:
        a, b = b, a
    answer = a + b if op == "+" else a - b
    return f"Question: {a} {op} {b}\nAnswer: {answer}"


def build_budgeted_jsonl(
    *,
    name: str,
    out_path: Path,
    target_tokens: int,
    iter_rows: Iterable[tuple[str, str]],
    tok: ByteBPETokenizer,
) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    total_tokens = 0
    total_docs = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row_id, text in iter_rows:
            text = text.strip()
            if not row_id or not text or row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            n_tok = token_len(tok, text)
            if total_docs > 0 and total_tokens >= target_tokens:
                break
            row = {
                "id": row_id,
                "text": text,
                "source": name,
                "word_count": len(text.split()),
                "token_count_est": n_tok,
            }
            out.write(json.dumps(row, ensure_ascii=True) + "\n")
            total_tokens += n_tok
            total_docs += 1
    return total_docs, total_tokens


def iter_squad_rows(seed: int) -> Iterable[tuple[str, str]]:
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    for idx in indices:
        row = ds[int(idx)]
        formatted = format_squad_row(row)
        if formatted is not None:
            yield formatted


def iter_multinli_rows(seed: int) -> Iterable[tuple[str, str]]:
    ds = load_dataset("nyu-mll/multi_nli", split="train")
    label_names = list(ds.features["label"].names)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    for idx in indices:
        row = ds[int(idx)]
        formatted = format_multinli_row(row, label_names)
        if formatted is not None:
            yield formatted


def iter_counting_rows(seed: int) -> Iterable[tuple[str, str]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    i = 0
    while True:
        text = generate_counting_text(rng).strip()
        if text in seen:
            continue
        seen.add(text)
        i += 1
        yield (f"counting:{i}", text)


def build_source(
    *,
    name: str,
    raw_path: Path,
    out_dir: Path,
    target_tokens: int,
    row_iter: Iterable[tuple[str, str]],
    tok: ByteBPETokenizer,
    workers: int,
    seed: int,
) -> SourceBuildResult:
    if raw_path.exists():
        raw_path.unlink()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    raw_docs, raw_tokens = build_budgeted_jsonl(
        name=name,
        out_path=raw_path,
        target_tokens=target_tokens,
        iter_rows=row_iter,
        tok=tok,
    )
    if name == "squad":
        blank_rows = count_blank_answer_rows(raw_path)
        if blank_rows > 0:
            raise RuntimeError(f"SQuAD rebuild produced {blank_rows} blank-answer rows in {raw_path}")
    pretokenize(raw_path, out_dir, workers=workers)
    validate_pretokenized_dir(out_dir)
    train_manifest = read_manifest_counts(out_dir / "train")
    bucket_stats, train_windows, train_tokens = collect_length_stats(out_dir / "train")
    raw_checks = sample_raw_rows(raw_path, 5, seed=seed + 11)
    decode_checks = build_decode_checks(tok, sample_raw_rows(raw_path, 10, seed=seed + 29))
    return SourceBuildResult(
        name=name,
        raw_path=raw_path,
        out_dir=out_dir,
        raw_docs=raw_docs,
        raw_token_budget=target_tokens,
        raw_tokens_selected=raw_tokens,
        pretokenized_train_docs=int(train_manifest["counts"]["docs"]),
        pretokenized_train_windows=int(train_windows),
        pretokenized_train_tokens=int(train_tokens),
        bucket_stats=bucket_stats,
        raw_spot_checks=raw_checks,
        decode_spot_checks=decode_checks,
    )


def existing_manifest_summary(path: Path) -> dict[str, Any]:
    manifest = read_manifest_counts(path / "train")
    bucket_stats, train_windows, train_tokens = collect_length_stats(path / "train")
    return {
        "path": str(path),
        "train_docs": int(manifest["counts"]["docs"]),
        "train_windows": int(train_windows),
        "train_tokens": int(train_tokens),
        "bucket_stats": bucket_stats,
    }


def combine_bucket_stats(named_stats: Sequence[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    seqs = [0 for _ in BUCKET_RANGES]
    toks = [0 for _ in BUCKET_RANGES]
    for _name, stats in named_stats:
        for i, row in enumerate(stats):
            seqs[i] += int(row["seqs"])
            toks[i] += int(row["tokens"])
    total_tokens = sum(toks) or 1
    out = []
    for (lo, hi), s, t in zip(BUCKET_RANGES, seqs, toks):
        out.append({"range": [lo, hi], "seqs": s, "tokens": t, "prob": float(t / total_tokens)})
    return out


def write_mix_config(existing: dict[str, Any], new_sources: Sequence[SourceBuildResult]) -> dict[str, Any]:
    new_total = sum(x.pretokenized_train_tokens for x in new_sources)
    reasoning_total_weight = 0.20
    source_payload: dict[str, Any] = {
        "wiki": {
            "tokens": int(existing["wiki"]["train_tokens"]),
            "weight": 0.55,
            "path": str(ROOT / "data" / "pretraining" / "wikicoco256_cleaned"),
        },
        "distill": {
            "tokens": int(existing["distill"]["train_tokens"]),
            "weight": 0.25,
            "path": str(ROOT / "data" / "pretraining" / "distill256_cleaned2"),
        },
    }
    for src in new_sources:
        rel_weight = (float(src.pretokenized_train_tokens) / float(new_total)) if new_total > 0 else 1.0 / max(1, len(new_sources))
        source_payload[src.name] = {
            "tokens": int(src.pretokenized_train_tokens),
            "weight": float(reasoning_total_weight * rel_weight),
            "path": str(src.out_dir),
        }
    total_tokens = sum(int(v["tokens"]) for v in source_payload.values())
    out = {
        "sources": source_payload,
        "total_tokens": int(total_tokens),
        "notes": (
            "Uses existing pretokenize_corpus formatting. "
            "Wiki+distill remain at 80% combined weight; the 20% reasoning slice is allocated "
            "proportionally across the realized reasoning-source token counts."
        ),
    }
    MIX_CONFIG_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def render_bucket_table(name: str, stats: list[dict[str, Any]]) -> str:
    lines = [f"Length-bucket stats [{name}] (range, seqs, tokens, prob):"]
    for row in stats:
        lo, hi = row["range"]
        lines.append(
            f"  {lo:4d}-{hi:4d}: seqs={int(row['seqs']):7d} "
            f"tokens={int(row['tokens']):10d} prob={float(row['prob']):.4f}"
        )
    return "\n".join(lines)


def write_report(
    *,
    existing: dict[str, Any],
    new_sources: Sequence[SourceBuildResult],
    combined_reasoning_stats: list[dict[str, Any]],
    combined_full_stats: list[dict[str, Any]],
    mix_config: dict[str, Any],
) -> None:
    new_source_payload = []
    for src in new_sources:
        new_source_payload.append(
            {
                **src.__dict__,
                "raw_path": str(src.raw_path),
                "out_dir": str(src.out_dir),
            }
        )
    report = {
        "pipeline": {
            "raw_converters": [
                "scripts/distill_to_pretokenize.py",
                "scripts/coco_to_pretokenize.py",
            ],
            "pretokenizer": "scripts/pretokenize_corpus.py",
            "tokenizer": str(TOKENIZER_PATH),
            "reference_dirs": [
                "data/pretraining/wikicoco256_cleaned",
                "data/pretraining/distill256_cleaned2",
            ],
            "pretokenize_style": {
                "max_seq_len": 256,
                "stride": 64,
                "window_stride": 64,
                "max_windows_per_doc": 3,
                "window_sampling": "random",
                "segment_token_cap": 1024,
                "normalization": "NFKC",
                "clean_wikipedia": False,
                "clean_markdown": True,
                "min_chars_after_clean": 1,
                "add_bos": False,
                "add_eos": True,
                "dedup_docs": False,
                "dedup_seqs": False,
            },
        },
        "existing": existing,
        "new_sources": new_source_payload,
        "combined_reasoning_stats": combined_reasoning_stats,
        "combined_full_stats": combined_full_stats,
        "mix_config": mix_config,
    }
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# Reasoning LM Corpora Report")
    lines.append("")
    lines.append("## Pipeline")
    lines.append(f"- tokenizer: `{TOKENIZER_PATH}`")
    lines.append("- reference pretokenized dirs:")
    lines.append("  - `data/pretraining/wikicoco256_cleaned`")
    lines.append("  - `data/pretraining/distill256_cleaned2`")
    lines.append("- raw text -> shard pipeline:")
    lines.append("  - `scripts/build_reasoning_lm_corpora.py`")
    lines.append("  - `scripts/pretokenize_corpus.py`")
    lines.append("")
    lines.append("## Existing Corpus Anchors")
    lines.append(f"- wiki train tokens: `{existing['wiki']['train_tokens']}`")
    lines.append(f"- distill train tokens: `{existing['distill']['train_tokens']}`")
    lines.append("")
    lines.append("## New Sources")
    for src in new_sources:
        lines.append(f"### {src.name}")
        lines.append(f"- raw path: `{src.raw_path}`")
        lines.append(f"- pretokenized dir: `{src.out_dir}`")
        lines.append(f"- raw docs: `{src.raw_docs}`")
        lines.append(f"- selected raw tokens: `{src.raw_tokens_selected}` (target `{src.raw_token_budget}`)")
        lines.append(f"- train docs: `{src.pretokenized_train_docs}`")
        lines.append(f"- train windows: `{src.pretokenized_train_windows}`")
        lines.append(f"- train tokens: `{src.pretokenized_train_tokens}`")
        lines.append("")
        lines.append("```text")
        lines.append(render_bucket_table(src.name, src.bucket_stats))
        lines.append("```")
        lines.append("")
    lines.append("## Combined Reasoning Bucket Stats")
    lines.append("")
    lines.append("```text")
    lines.append(render_bucket_table("reasoning_combined", combined_reasoning_stats))
    lines.append("```")
    lines.append("")
    lines.append("## Combined Full Mix Bucket Stats")
    lines.append("")
    lines.append("```text")
    lines.append(render_bucket_table("wiki+distill+reasoning", combined_full_stats))
    lines.append("```")
    lines.append("")
    lines.append("## Mix Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(mix_config, indent=2))
    lines.append("```")
    lines.append("")
    for src in new_sources:
        lines.append(f"## Spot Checks: {src.name}")
        lines.append("")
        for check in src.decode_spot_checks[:5]:
            lines.append(f"- id: `{check['id']}` seq_len=`{check['seq_len']}`")
            lines.append(f"  raw: `{check['raw_text'][:500]}`")
            lines.append(f"  decoded: `{check['decoded_text'][:500]}`")
        lines.append("")
    REPORT_MD_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build reasoning-enriched LM corpora using the existing pretokenize pipeline.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--squad-target-tokens", type=int, default=2_000_000)
    ap.add_argument("--multinli-target-tokens", type=int, default=1_500_000)
    ap.add_argument("--counting-target-tokens", type=int, default=1_500_000)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    tok = load_tokenizer(TOKENIZER_PATH)

    existing = {
        "wiki": existing_manifest_summary(ROOT / "data" / "pretraining" / "wikicoco256_cleaned"),
        "distill": existing_manifest_summary(ROOT / "data" / "pretraining" / "distill256_cleaned2"),
    }

    squad = build_source(
        name="squad",
        raw_path=RAW_SQUAD_PATH,
        out_dir=SQUAD_OUT_DIR,
        target_tokens=int(args.squad_target_tokens),
        row_iter=iter_squad_rows(seed=int(args.seed) + 1),
        tok=tok,
        workers=int(args.workers),
        seed=int(args.seed) + 100,
    )
    multinli = build_source(
        name="multinli",
        raw_path=RAW_MULTINLI_PATH,
        out_dir=MULTINLI_OUT_DIR,
        target_tokens=int(args.multinli_target_tokens),
        row_iter=iter_multinli_rows(seed=int(args.seed) + 2),
        tok=tok,
        workers=int(args.workers),
        seed=int(args.seed) + 200,
    )
    counting = build_source(
        name="counting",
        raw_path=RAW_COUNTING_PATH,
        out_dir=COUNTING_OUT_DIR,
        target_tokens=int(args.counting_target_tokens),
        row_iter=iter_counting_rows(seed=int(args.seed) + 3),
        tok=tok,
        workers=int(args.workers),
        seed=int(args.seed) + 300,
    )
    new_sources = [squad, multinli, counting]

    combined_reasoning_stats = combine_bucket_stats([(s.name, s.bucket_stats) for s in new_sources])
    combined_full_stats = combine_bucket_stats(
        [
            ("wiki", existing["wiki"]["bucket_stats"]),
            ("distill", existing["distill"]["bucket_stats"]),
            ("squad", squad.bucket_stats),
            ("multinli", multinli.bucket_stats),
            ("counting", counting.bucket_stats),
        ]
    )
    mix_config = write_mix_config(existing, new_sources)
    write_report(
        existing=existing,
        new_sources=new_sources,
        combined_reasoning_stats=combined_reasoning_stats,
        combined_full_stats=combined_full_stats,
        mix_config=mix_config,
    )

    print("Reasoning LM corpora complete.")
    for src in new_sources:
        print(
            f"- {src.name}: raw_docs={src.raw_docs} raw_tokens={src.raw_tokens_selected} "
            f"train_windows={src.pretokenized_train_windows} train_tokens={src.pretokenized_train_tokens}"
        )
        print(render_bucket_table(src.name, src.bucket_stats))
    print(render_bucket_table("reasoning_combined", combined_reasoning_stats))
    print(render_bucket_table("wiki+distill+reasoning", combined_full_stats))
    print(f"mix config: {MIX_CONFIG_PATH}")
    print(f"report: {REPORT_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
