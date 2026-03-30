from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from train.mm import (
    build_loader,
    evaluate_records,
    load_runtime_from_checkpoint,
    resolve_device,
    run_generation_predictions,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate an MM checkpoint on GQA with exact-match scoring.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gqa_root", type=str, default=None)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def _run_group(
    *,
    model: Any,
    tokenizer: Any,
    run_args: Any,
    device: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    limit_eval: int,
    eval_batches: int,
    group: str,
) -> Dict[str, Any]:
    run_args.eval_batch_size = int(batch_size)
    run_args.batch_size = int(batch_size)
    run_args.num_workers = int(num_workers)
    run_args.prefetch_factor = int(prefetch_factor)
    run_args.pin_memory = bool(pin_memory)
    run_args.gqa_eval_group = str(group or "")
    loader = build_loader(
        run_args,
        tokenizer=tokenizer,
        split="gqa_val",
        train_mode=False,
        limit=max(0, int(limit_eval)),
    )
    records = run_generation_predictions(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
        max_answer_length=int(run_args.max_answer_length),
        max_batches=int(eval_batches),
        logger=None,
        split_name="gqa_val",
        log_every=20,
    )
    summary = evaluate_records(records, qualitative_samples=0, confusion_top_k=0, scorer="exact")
    return {
        "overall_accuracy": float(summary.get("overall_accuracy", 0.0) or 0.0),
        "record_count": int(summary.get("num_samples", len(records))),
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    overrides = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "gqa_root": args.gqa_root,
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        args_override=overrides,
    )
    if args.gqa_root:
        run_args.gqa_root = args.gqa_root

    overall = _run_group(
        model=model,
        tokenizer=tokenizer,
        run_args=run_args,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        limit_eval=int(args.limit_eval),
        eval_batches=int(args.eval_batches),
        group="",
    )

    groups: Dict[str, Any] = {}
    for group in ("spatial", "attribute", "exist", "count"):
        groups[group] = _run_group(
            model=model,
            tokenizer=tokenizer,
            run_args=run_args,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            pin_memory=bool(args.pin_memory),
            limit_eval=int(args.limit_eval),
            eval_batches=int(args.eval_batches),
            group=group,
        )

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "scorer": "exact",
        "overall_accuracy": float(overall.get("overall_accuracy", 0.0)),
        "record_count": int(overall.get("record_count", 0)),
        "group_accuracy": {k: float(v.get("overall_accuracy", 0.0)) for k, v in groups.items()},
        "group_record_count": {k: int(v.get("record_count", 0)) for k, v in groups.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[gqa-eval] wrote: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
