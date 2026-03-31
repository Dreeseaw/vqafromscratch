from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Sequence

from torch.utils.data import DataLoader, Subset

from train.mm import build_loader, evaluate_records, load_runtime_from_checkpoint, resolve_device, run_generation_predictions, set_seed


OCR_BUCKET_PATTERNS = {
    "sign_written": [
        "what does the sign",
        "what does this sign",
        "what is written",
        "written on the",
        "on the sign",
    ],
    "brand_logo": [
        "what brand",
        "what company",
        "what logo",
        "what store",
    ],
    "name_label": [
        "what is the name",
        "what name",
        "name of the",
        "what does the label",
    ],
    "word_text": [
        "what word",
        "what words",
        "what letter",
        "what number is",
        "what does the",
    ],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the heuristic OCR VQAv2 subset.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit_ocr", type=int, default=500)
    ap.add_argument("--semantic_eval_budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def bucket_for_question(question: str) -> str | None:
    q = str(question).strip().lower()
    for bucket, patterns in OCR_BUCKET_PATTERNS.items():
        if any(p in q for p in patterns):
            return bucket
    return None


def make_subset_loader(
    base_loader: DataLoader,
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    dataset = Subset(base_loader.dataset, list(indices))
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "collate_fn": base_loader.collate_fn,
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def collect_indices(dataset: Any, *, limit_ocr: int) -> tuple[list[int], dict[int, str]]:
    ocr_indices: list[int] = []
    bucket_by_index: dict[int, str] = {}
    items = getattr(dataset, "items", [])
    for idx, item in enumerate(items):
        bucket = bucket_for_question(item.get("question", ""))
        if bucket is None:
            continue
        ocr_indices.append(idx)
        bucket_by_index[idx] = bucket
        if len(ocr_indices) >= int(limit_ocr):
            break
    return ocr_indices, bucket_by_index


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    overrides = {
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "semantic_eval_budget": int(args.semantic_eval_budget),
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        args_override=overrides,
    )
    base_loader = build_loader(run_args, tokenizer=tokenizer, split=str(args.eval_split), train_mode=False, limit=0)
    ocr_indices, bucket_by_index = collect_indices(base_loader.dataset, limit_ocr=int(args.limit_ocr))
    loader = make_subset_loader(
        base_loader,
        ocr_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=int(args.prefetch_factor),
    )
    records = run_generation_predictions(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
        max_answer_length=int(run_args.max_answer_length),
        max_batches=0,
        logger=None,
        split_name=f"{args.eval_split}_ocr_subset",
        log_every=20,
        cuda_empty_cache_every=0,
    )
    summary = evaluate_records(records, qualitative_samples=0, confusion_top_k=0, scorer="official")
    bucket_counts = Counter(bucket_by_index.values())
    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "overall_accuracy": float(summary.get("overall_accuracy", 0.0) or 0.0),
        "answer_type_accuracy": dict(summary.get("answer_type_accuracy", {}) or {}),
        "record_count": int(len(records)),
        "bucket_count": {str(k): int(v) for k, v in sorted(bucket_counts.items())},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[eval:{args.eval_split}_ocr_subset] overall_accuracy={out['overall_accuracy']:.4f}", flush=True)
    print(f"[ocr-subset] wrote: {os.path.abspath(args.output_json)}", flush=True)


if __name__ == "__main__":
    main()
