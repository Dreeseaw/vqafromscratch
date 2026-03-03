#!/usr/bin/env python3
"""
Convert COCO captions JSON into pretokenize_corpus-ready JSONL docs.
"""
import argparse
import json
import os
from pathlib import Path


def _find_captions_json(anno_dir: str) -> str:
    candidates = [
        os.path.join(anno_dir, "annotations", "captions_train2017.json"),
        os.path.join(anno_dir, "annotations", "captions_train2014.json"),
        os.path.join(anno_dir, "captions_train2017.json"),
        os.path.join(anno_dir, "captions_train2014.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find COCO captions json under {anno_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert COCO captions JSON into JSONL docs with a `text` field."
    )
    ap.add_argument("--captions_json", default="", help="Path to captions_train*.json. If unset, auto-detect.")
    ap.add_argument(
        "--anno_dir",
        default="./annotations",
        help="COCO annotations root used when --captions_json is not provided.",
    )
    ap.add_argument("--out_path", required=True, help="Output JSONL path.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    captions_path = args.captions_json.strip() or _find_captions_json(args.anno_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(captions_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    anns = payload.get("annotations", [])
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for i, ann in enumerate(anns, start=1):
            cap = ann.get("caption")
            if not isinstance(cap, str):
                continue
            text = cap.strip()
            if not text:
                continue
            ann_id = ann.get("id", i)
            image_id = ann.get("image_id")
            row = {
                "id": f"coco_caption:{ann_id}",
                "text": text,
                "source": "coco_caption",
                "word_count": len(text.split()),
            }
            if image_id is not None:
                row["image_id"] = image_id
            out.write(json.dumps(row, ensure_ascii=True) + "\n")
            written += 1

    print(f"Wrote {written} captions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
