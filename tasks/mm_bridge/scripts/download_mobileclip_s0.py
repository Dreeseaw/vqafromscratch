from __future__ import annotations

import argparse

from models.hf_vision import download_mobileclip_s0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        type=str,
        default="logs/hf_vision/apple_mobileclip_s0",
    )
    args = ap.parse_args()
    out_dir = download_mobileclip_s0(args.out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
