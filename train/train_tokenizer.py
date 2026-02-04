import argparse
import json
import os
from typing import List

from evals.probe import ANNO_DIR_DEFAULT

from models.bpe_tokenizer import ByteBPETokenizer


LOGDIR = "logs/"


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
    raise FileNotFoundError(
        f"Could not find COCO captions json under {anno_dir}."
    )


def load_captions(path: str, limit: int = 0) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    captions: List[str] = []
    for ann in data.get("annotations", []):
        cap = ann.get("caption")
        if not cap:
            continue
        captions.append(cap)
        if limit > 0 and len(captions) >= limit:
            break
    return captions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--num_merges", type=int, default=8000)
    ap.add_argument("--min_pair_freq", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run_dir = os.path.join(LOGDIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    ann_path = _find_captions_json(ANNO_DIR_DEFAULT)
    captions = load_captions(ann_path, limit=args.limit)

    tokenizer = ByteBPETokenizer()
    tokenizer.train_bpe(
        captions, num_merges=args.num_merges, min_pair_freq=args.min_pair_freq
    )

    tok_path = os.path.join(run_dir, "tokenizer.pt")
    tokenizer.save(tok_path)

    info = {
        "vocab_size": tokenizer.vocab_size,
        "num_merges": args.num_merges,
        "min_pair_freq": args.min_pair_freq,
        "num_captions": len(captions),
    }
    with open(os.path.join(run_dir, "tokenizer_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"vocab_size: {tokenizer.vocab_size}")
    if captions:
        sample = captions[0]
        enc = tokenizer.encode(sample)
        dec = tokenizer.decode(enc)
        print(f"round_trip caption: {sample}")
        print(f"encoded length: {enc.numel()}")
        print(f"decoded: {dec}")
    else:
        print("round_trip caption: <none>")

    batch = captions[: min(4, len(captions))]
    if batch:
        input_ids, attention_mask = tokenizer(
            batch, max_len=args.max_len, return_attention_mask=True
        )
        print(
            f"batch shapes: input_ids={tuple(input_ids.shape)} "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    else:
        print("batch shapes: <none>")


if __name__ == "__main__":
    main()
