from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a composite LM train bucket from distill + reasoning corpora.")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="data/pretraining/distill256_reasoningmix_v1/train",
        help="Output train bucket directory containing manifest.jsonl / manifest.json / meta.json",
    )
    ap.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Source train bucket dir to include, e.g. --source distill data/pretraining/distill256_cleaned2/train",
    )
    ap.add_argument(
        "--repeat",
        action="append",
        nargs=2,
        metavar=("NAME", "COUNT"),
        help="Optional integer repeat count per source name, e.g. --repeat squad 3",
    )
    ap.add_argument("--force", action="store_true", help="Rebuild output even if it already exists.")
    return ap.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object line in {path}")
            rows.append(row)
    return rows


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _default_sources() -> List[Tuple[str, str]]:
    return [
        ("distill", "data/pretraining/distill256_cleaned2/train"),
        ("squad", "data/pretraining/squad256_reasoning/train"),
        ("multinli", "data/pretraining/multinli256_reasoning/train"),
        ("counting", "data/pretraining/counting256_reasoning/train"),
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    root = output_dir.parent
    manifest_jsonl = output_dir / "manifest.jsonl"
    manifest_json = output_dir / "manifest.json"
    meta_json = output_dir / "meta.json"

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    sources = args.source if args.source else _default_sources()
    repeat_map = {str(name): int(count) for name, count in (args.repeat or [])}

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[Dict[str, Any]] = []
    source_summaries: List[Dict[str, Any]] = []
    base_meta: Dict[str, Any] | None = None
    total_examples = 0
    total_tokens = 0
    next_shard_id = 0

    for name, path_str in sources:
        src_dir = Path(path_str).resolve()
        src_manifest_jsonl = src_dir / "manifest.jsonl"
        src_manifest_json = src_dir / "manifest.json"
        src_meta_json = src_dir / "meta.json"
        if not src_manifest_jsonl.is_file():
            raise FileNotFoundError(f"Missing manifest.jsonl for source '{name}': {src_manifest_jsonl}")
        if not src_meta_json.is_file():
            raise FileNotFoundError(f"Missing meta.json for source '{name}': {src_meta_json}")

        rows = _load_jsonl(src_manifest_jsonl)
        meta = _load_json(src_meta_json)
        manifest = _load_json(src_manifest_json) if src_manifest_json.is_file() else {}
        if base_meta is None:
            base_meta = dict(meta)
        else:
            for key in ("max_seq_len", "vocab_size", "bos_id", "eos_id", "pad_id"):
                if key in base_meta and key in meta and base_meta[key] != meta[key]:
                    raise ValueError(
                        f"Source '{name}' meta mismatch for {key}: {meta[key]} vs base {base_meta[key]}"
                    )

        repeat = max(1, int(repeat_map.get(str(name), 1)))
        for rep_idx in range(repeat):
            for row in rows:
                new_row = dict(row)
                new_row["shard_id"] = int(next_shard_id)
                next_shard_id += 1
                tokens_rel = str(row["tokens"])
                lengths_rel = str(row["lengths"])
                new_row["tokens"] = os.path.relpath(src_dir / tokens_rel, output_dir)
                new_row["lengths"] = os.path.relpath(src_dir / lengths_rel, output_dir)
                combined_rows.append(new_row)
                total_examples += int(row.get("num_seqs", 0) or 0)
                total_tokens += int(row.get("num_tokens", 0) or 0)

        source_summaries.append(
            {
                "name": str(name),
                "path": str(src_dir),
                "repeat": int(repeat),
                "base_examples": int(manifest.get("counts", {}).get("examples", 0) or 0),
                "base_tokens": int(manifest.get("counts", {}).get("total_tokens_prepad", 0) or 0),
            }
        )

    if base_meta is None:
        raise RuntimeError("No sources were configured for the composite train bucket.")

    _save_jsonl(manifest_jsonl, combined_rows)
    manifest_payload = {
        "split": "train",
        "max_seq_len": int(base_meta.get("max_seq_len", 256) or 256),
        "stride": int(base_meta.get("stride", 64) or 64),
        "tokenizer_vocab_size": int(base_meta.get("vocab_size", 0) or 0),
        "counts": {
            "docs": int(total_examples),
            "windows": int(total_examples),
            "examples": int(total_examples),
            "total_tokens_prepad": int(total_tokens),
        },
        "sources": source_summaries,
        "notes": "Composite train bucket used to inject reasoning corpora through the existing distill bucket path.",
    }
    _save_json(manifest_json, manifest_payload)

    meta_payload = dict(base_meta)
    meta_payload["split"] = "train"
    meta_payload["composite_sources"] = source_summaries
    _save_json(meta_json, meta_payload)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_manifest_rows": len(combined_rows),
                "total_examples": int(total_examples),
                "total_tokens_prepad": int(total_tokens),
                "sources": source_summaries,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
