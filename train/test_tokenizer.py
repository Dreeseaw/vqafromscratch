import argparse
import json
import os

from evals.probe import ANNO_DIR_DEFAULT
from models.bpe_tokenizer import ByteBPETokenizer


def _find_captions_json(anno_dir: str) -> str:
    candidates = [
        os.path.join(anno_dir, "annotations/captions_train2017.json"),
        os.path.join(anno_dir, "annotations/captions_train2014.json"),
        os.path.join(anno_dir, "captions_train2017.json"),
        os.path.join(anno_dir, "captions_train2014.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find COCO captions json under {anno_dir}.")


def load_caption(path: str, idx: int) -> str:
    with open(path, "r") as f:
        data = json.load(f)
    anns = data.get("annotations", [])
    if not anns:
        raise ValueError("No annotations found in captions json.")
    idx = max(0, min(idx, len(anns) - 1))
    return anns[idx].get("caption", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--idx", type=int, default=0)
    args = ap.parse_args()

    tok_path = os.path.join("logs", args.run_id, "tokenizer.pt")
    if not os.path.isfile(tok_path):
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    ann_path = _find_captions_json(ANNO_DIR_DEFAULT)
    caption = load_caption(ann_path, args.idx)

    tokenizer = ByteBPETokenizer.load(tok_path)
    enc = tokenizer.encode(caption)
    dec = tokenizer.decode(enc)

    print(f"caption: {caption}")
    print(f"encoding (idx): {enc}")
    print(f"encoding (mapped): {[tokenizer.decode([t,]) for t in enc]}")
    print(f"encoded length: {enc.numel()}")
    print(f"decoded: {dec}")


if __name__ == "__main__":
    main()
