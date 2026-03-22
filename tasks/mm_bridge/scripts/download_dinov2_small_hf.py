from __future__ import annotations

import argparse

from models.hf_vision import download_dinov2_small


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", type=str, default="facebook/dinov2-small")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="logs/hf_vision/facebook_dinov2_small",
    )
    args = ap.parse_args()
    out_dir = download_dinov2_small(args.repo_id, args.out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
