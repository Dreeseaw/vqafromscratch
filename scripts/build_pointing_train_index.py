#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = ROOT / "data" / "pointing" / "index"
TRAIN_INDEX_PATH = ROOT / "data" / "pointing" / "train_index.jsonl"
MIX_CONFIG_PATH = ROOT / "data" / "pointing" / "mix_config.json"
GRID_SIZE = 14
TOKEN_COUNT = GRID_SIZE * GRID_SIZE
SIGMA = 1.5


def resolve_image_path(path_value: str, db_conn: duckdb.DuckDBPyConnection | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = (ROOT / path).resolve()
    if candidate.exists():
        return candidate
    if db_conn is not None:
        row = db_conn.execute(
            "select local_path from images where local_path = ? limit 1",
            [path_value],
        ).fetchone()
        if row:
            resolved = (ROOT / str(row[0])).resolve()
            return resolved
    return candidate


def split_is_train(split: str | None) -> bool:
    text = str(split or "").strip().lower()
    return text == "train" or text.startswith("train_")


def project_point_to_grid(x: float, y: float) -> int:
    col = max(0, min(GRID_SIZE - 1, int(math.floor(float(x) * GRID_SIZE))))
    row = max(0, min(GRID_SIZE - 1, int(math.floor(float(y) * GRID_SIZE))))
    return row * GRID_SIZE + col


def gaussian_soft_target(indices: list[int]) -> list[float]:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij")
    for flat_idx in indices:
        row = flat_idx // GRID_SIZE
        col = flat_idx % GRID_SIZE
        dist2 = (xx - col) ** 2 + (yy - row) ** 2
        grid += np.exp(-dist2 / (2.0 * SIGMA * SIGMA))
    total = float(grid.sum())
    if total <= 0.0:
        raise ValueError("soft target sum was non-positive")
    return (grid / total).reshape(-1).astype(np.float32).tolist()


def load_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(INDEX_ROOT.glob("*.jsonl")):
        if path.name == "invalid_records.jsonl":
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training-ready pointing index with SigLIP grid targets.")
    parser.add_argument("--db-path", default=str(ROOT / "data" / "vm_ssl" / "db" / "vm_ssl.duckdb"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_conn = duckdb.connect(args.db_path, read_only=True) if Path(args.db_path).exists() else None
    raw_records = load_records()

    source_counts = Counter()
    has_vqa_target = 0
    grounding_only = 0
    point_count_sum = 0
    skip_reasons = Counter()
    output_records: list[dict[str, Any]] = []

    TRAIN_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    for record in raw_records:
        split = str((record.get("metadata") or {}).get("split") or "")
        if not split_is_train(split):
            skip_reasons["non_train_split"] += 1
            continue

        points = list(record.get("points") or [])
        if not points:
            skip_reasons["no_points"] += 1
            continue

        resolved_path = resolve_image_path(str(record["image_path"]), db_conn)
        if not resolved_path.exists():
            skip_reasons["missing_image"] += 1
            continue

        grid_indices = [project_point_to_grid(float(pt["x"]), float(pt["y"])) for pt in points]
        if len(points) > 1 and len(set(grid_indices)) == 1:
            skip_reasons["degenerate_same_grid_cell"] += 1
            continue

        soft_target = gaussian_soft_target(grid_indices)
        if not math.isclose(sum(soft_target), 1.0, rel_tol=0.0, abs_tol=1e-4):
            skip_reasons["soft_target_not_normalized"] += 1
            continue

        answer = record.get("answer")
        has_target = answer is not None and str(answer).strip() != ""

        output = {
            "sample_id": record["sample_id"],
            "source": record["source"],
            "image_path": str(resolved_path),
            "question": record["question"],
            "answer": answer,
            "points_normalized": points,
            "grid_targets": {
                "indices": grid_indices,
                "soft_target": soft_target,
            },
            "has_vqa_target": bool(has_target),
            "metadata": record.get("metadata") or {},
        }
        output_records.append(output)
        source_counts[record["source"]] += 1
        point_count_sum += len(points)
        if has_target:
            has_vqa_target += 1
        else:
            grounding_only += 1

    with TRAIN_INDEX_PATH.open("w", encoding="utf-8") as out:
        for record in output_records:
            out.write(json.dumps(record) + "\n")

    raw_weight_map = {source: math.sqrt(float(count)) for source, count in source_counts.items()}
    weight_total = sum(raw_weight_map.values()) or 1.0
    weights = {source: value / weight_total for source, value in raw_weight_map.items()}
    mix_config = {
        "total_samples": len(output_records),
        "source_counts": dict(source_counts),
        "recommended_sampling_weights": weights,
    }
    MIX_CONFIG_PATH.write_text(json.dumps(mix_config, indent=2) + "\n", encoding="utf-8")

    for record in output_records:
        path = Path(record["image_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        if any(idx < 0 or idx >= TOKEN_COUNT for idx in record["grid_targets"]["indices"]):
            raise ValueError(f"bad grid index in {record['sample_id']}")
        total = float(sum(record["grid_targets"]["soft_target"]))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f"soft target invalid for {record['sample_id']}: {total}")

    print("Train Index Summary")
    print(f"- total_records: {len(output_records)}")
    print(f"- source_counts: {dict(source_counts)}")
    print(f"- has_vqa_target: {has_vqa_target}")
    print(f"- grounding_only: {grounding_only}")
    avg_points = (point_count_sum / len(output_records)) if output_records else 0.0
    print(f"- avg_points_per_sample: {avg_points:.3f}")
    print(f"- skips: {dict(skip_reasons)}")

    rng = random.Random(args.seed)
    sample_records = output_records if len(output_records) <= 5 else rng.sample(output_records, 5)
    print("\nSpot Checks")
    for rec in sample_records:
        soft = np.array(rec["grid_targets"]["soft_target"], dtype=np.float64)
        top3 = np.argsort(-soft)[:3].tolist()
        print(
            json.dumps(
                {
                    "sample_id": rec["sample_id"],
                    "question": rec["question"],
                    "points": rec["points_normalized"],
                    "grid_indices": rec["grid_targets"]["indices"],
                    "top3_soft_indices": top3,
                }
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
