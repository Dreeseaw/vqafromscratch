#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "Photoroom/midjourney-v6-recap"
REPO_TYPE = "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and materialize a random Midjourney v6 recap subset.")
    parser.add_argument("--target-samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--max-shards", type=int, default=0, help="Optional debug cap on parquet shards.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/vm_ssl/archives/midjourney_v6_recap"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/vm_ssl/raw/midjourney_v6_recap_30k"))
    return parser.parse_args()


def existing_sample_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            seen.add(str(payload["sample_id"]))
    return seen


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    args.parquet_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    meta_dir = args.output_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "subset_manifest.jsonl"

    seen_ids = existing_sample_ids(manifest_path)
    if len(seen_ids) >= int(args.target_samples):
        print(f"midjourney subset already materialized: {len(seen_ids)} samples")
        return 0

    api = HfApi()
    shard_names = [name for name in api.list_repo_files(REPO_ID, repo_type=REPO_TYPE) if name.startswith("train_") and name.endswith(".parquet")]
    if int(args.max_shards) > 0:
        shard_names = shard_names[: int(args.max_shards)]
    rng.shuffle(shard_names)

    con = duckdb.connect()
    written = len(seen_ids)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for shard_index, shard_name in enumerate(shard_names, start=1):
            if written >= int(args.target_samples):
                break
            local_parquet = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    filename=shard_name,
                    local_dir=str(args.parquet_dir),
                )
            )
            remaining = int(args.target_samples) - written
            query = f"""
                select
                  id,
                  prompt,
                  llava,
                  llava_status,
                  gemini,
                  gemini_status,
                  qwen3,
                  qwen3_status,
                  image
                from read_parquet('{local_parquet}')
                using sample reservoir({remaining} rows) repeatable({int(args.seed) + shard_index})
            """
            rows = con.execute(query).fetchall()
            shard_written = 0
            for row in rows:
                row_id, prompt, llava, llava_status, gemini, gemini_status, qwen3, qwen3_status, image = row
                sample_id = f"midjourney_v6_recap_30k:{row_id}"
                if sample_id in seen_ids:
                    continue
                image_bytes = image["bytes"]
                image_name = f"{row_id}.jpg"
                image_path = images_dir / image_name
                meta_path = meta_dir / f"{row_id}.json"
                if not image_path.exists():
                    image_path.write_bytes(image_bytes)
                meta_path.write_text(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "id": row_id,
                            "prompt": prompt,
                            "llava": llava,
                            "llava_status": bool(llava_status),
                            "gemini": gemini,
                            "gemini_status": bool(gemini_status),
                            "qwen3": qwen3,
                            "qwen3_status": bool(qwen3_status),
                            "shard": shard_name,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                manifest.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "dataset_name": "midjourney_v6_recap_30k",
                            "pair_split": "train",
                            "image_path": str(image_path),
                            "meta_path": str(meta_path),
                            "shard": shard_name,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                manifest.flush()
                seen_ids.add(sample_id)
                written += 1
                shard_written += 1
                if written >= int(args.target_samples):
                    break
            print(f"[{shard_index}/{len(shard_names)}] shard={shard_name} kept={shard_written} total={written}", flush=True)

    print(f"midjourney subset materialized: {written} samples -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
