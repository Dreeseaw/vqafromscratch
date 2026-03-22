#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DATASETS = {
    "textcaps": {
        "repo_id": "lmms-lab/TextCaps",
        "repo_type": "dataset",
        "local_dir": "data/vm_ssl/raw/textcaps_hf",
    },
    "flickr30k": {
        "repo_id": "nlphuji/flickr30k",
        "repo_type": "dataset",
        "local_dir": "data/vm_ssl/raw/flickr30k_hf",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download image-text datasets for VM recipe experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["textcaps", "flickr30k"],
        help="Datasets to download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for dataset_name in args.datasets:
        cfg = DATASETS[dataset_name]
        local_dir = Path(cfg["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {dataset_name} -> {local_dir}")
        snapshot_download(
            repo_id=cfg["repo_id"],
            repo_type=cfg["repo_type"],
            local_dir=str(local_dir),
        )
        print(f"done {dataset_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
