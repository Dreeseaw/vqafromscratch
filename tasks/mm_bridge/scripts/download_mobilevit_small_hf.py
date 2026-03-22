from __future__ import annotations

import argparse
import os

from models.hf_vision import download_mobilevit_small


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", type=str, default="apple/mobilevit-small")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="logs/hf_vision/apple_mobilevit_small",
    )
    args = ap.parse_args()
    out_dir = download_mobilevit_small(args.repo_id, args.out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
