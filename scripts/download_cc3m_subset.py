#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "pixparse/cc3m-wds"
REPO_TYPE = "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and materialize a random CC3M WebDataset subset.")
    parser.add_argument("--target-pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--shard-prefix", type=str, default="cc3m-train-")
    parser.add_argument("--max-shards", type=int, default=0, help="Optional cap for debugging.")
    parser.add_argument("--archives-dir", type=Path, default=Path("data/vm_ssl/archives/cc3m_wds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/vm_ssl/raw/cc3m_subset_50k"))
    return parser.parse_args()


def iter_sample_keys(tf: tarfile.TarFile) -> dict[str, set[str]]:
    exts_by_key: dict[str, set[str]] = defaultdict(set)
    for member in tf.getmembers():
        if not member.isfile():
            continue
        path = Path(member.name)
        stem = path.stem
        suffix = path.suffix.lower()
        if not stem or suffix not in {".jpg", ".jpeg", ".png", ".txt", ".json"}:
            continue
        exts_by_key[stem].add(suffix)
    return exts_by_key


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))

    args.archives_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    meta_dir = args.output_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "subset_manifest.jsonl"

    existing = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for _ in f:
                existing += 1
    if existing >= int(args.target_pairs):
        print(f"cc3m subset already materialized: {existing} pairs")
        return 0

    api = HfApi()
    shard_names = [
        name
        for name in api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        if name.startswith(args.shard_prefix) and name.endswith(".tar")
    ]
    if int(args.max_shards) > 0:
        shard_names = shard_names[: int(args.max_shards)]
    rng.shuffle(shard_names)

    written = existing
    seen_keys: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                seen_keys.add(str(payload["sample_id"]))

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for shard_idx, shard_name in enumerate(shard_names, start=1):
            if written >= int(args.target_pairs):
                break
            local_tar = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    filename=shard_name,
                    local_dir=str(args.archives_dir),
                )
            )
            print(f"[{shard_idx}/{len(shard_names)}] shard={shard_name} local={local_tar}")
            with tarfile.open(local_tar, "r") as tf:
                exts_by_key = iter_sample_keys(tf)
                candidate_keys = [
                    key for key, exts in exts_by_key.items() if ".txt" in exts and (".jpg" in exts or ".jpeg" in exts or ".png" in exts)
                ]
                rng.shuffle(candidate_keys)
                members = {member.name: member for member in tf.getmembers() if member.isfile()}
                shard_written = 0
                for key in candidate_keys:
                    if written >= int(args.target_pairs):
                        break
                    sample_id = f"{Path(shard_name).stem}:{key}"
                    if sample_id in seen_keys:
                        continue
                    image_member_name = None
                    for ext in (".jpg", ".jpeg", ".png"):
                        name = f"{key}{ext}"
                        if name in members:
                            image_member_name = name
                            break
                    text_member_name = f"{key}.txt"
                    json_member_name = f"{key}.json"
                    if image_member_name is None or text_member_name not in members:
                        continue
                    image_bytes = tf.extractfile(members[image_member_name]).read()
                    text = tf.extractfile(members[text_member_name]).read().decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    meta = {}
                    if json_member_name in members:
                        try:
                            meta = json.loads(tf.extractfile(members[json_member_name]).read().decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            meta = {}

                    image_ext = Path(image_member_name).suffix.lower()
                    out_name = f"{Path(shard_name).stem}_{key}{image_ext}"
                    image_path = images_dir / out_name
                    meta_path = meta_dir / f"{Path(shard_name).stem}_{key}.json"
                    image_path.write_bytes(image_bytes)
                    meta_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
                    manifest.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "dataset_name": "cc3m_subset_50k",
                                "pair_split": "train",
                                "shard": shard_name,
                                "key": key,
                                "image_path": str(image_path),
                                "meta_path": str(meta_path),
                                "text": text,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    manifest.flush()
                    seen_keys.add(sample_id)
                    written += 1
                    shard_written += 1
                print(f"  kept {shard_written} samples from {shard_name}; total={written}")

    print(f"cc3m subset materialized: {written} pairs -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
