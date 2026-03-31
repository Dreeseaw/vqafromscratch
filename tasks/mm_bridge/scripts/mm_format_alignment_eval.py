from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from train.mm import (
    build_loader,
    evaluate_records,
    load_prefix_remap_checkpoint,
    load_runtime_from_checkpoint,
    resolve_device,
    run_generation_predictions,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Eval helper for semantic bottleneck format-alignment checkpoints.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=0)
    ap.add_argument("--max_answer_length", type=int, default=None)
    ap.add_argument("--scorer", type=str, default="official", choices=["official", "proxy"])
    ap.add_argument("--apply_prefix_remap_in_forward", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--disable_lm_visual_adapters", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefix_remap_checkpoint", type=str, default="")
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--semantic_eval_budget", type=int, default=0)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    overrides: Dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "images_root": args.images_root,
        "annotations_root": args.annotations_root,
        "apply_prefix_remap_in_forward": bool(args.apply_prefix_remap_in_forward),
        "disable_lm_visual_adapters": bool(args.disable_lm_visual_adapters),
        "use_prefix_remap": bool(args.apply_prefix_remap_in_forward),
        "prefix_remap_present": bool(args.apply_prefix_remap_in_forward)
        or bool(str(args.prefix_remap_checkpoint or "").strip()),
        "prefix_remap_checkpoint": str(args.prefix_remap_checkpoint or ""),
        "semantic_eval_budget": int(args.semantic_eval_budget),
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        args_override=overrides,
    )
    if str(args.prefix_remap_checkpoint or "").strip():
        load_prefix_remap_checkpoint(
            model,
            checkpoint_path=str(args.prefix_remap_checkpoint),
            logger=None,
        )
    loader = build_loader(
        run_args,
        tokenizer=tokenizer,
        split=str(args.eval_split),
        train_mode=False,
        limit=max(0, int(args.limit_eval)),
    )
    max_answer_len = int(args.max_answer_length) if args.max_answer_length is not None else int(run_args.max_answer_length)
    records = run_generation_predictions(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
        max_answer_length=max_answer_len,
        max_batches=int(args.eval_batches),
        logger=None,
        split_name=str(args.eval_split),
        log_every=20,
        cuda_empty_cache_every=(400 if int(args.eval_batches) == 0 else 0),
    )
    summary = evaluate_records(
        records,
        qualitative_samples=0,
        confusion_top_k=0,
        scorer=str(args.scorer),
    )
    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "apply_prefix_remap_in_forward": bool(args.apply_prefix_remap_in_forward),
        "disable_lm_visual_adapters": bool(args.disable_lm_visual_adapters),
        "semantic_eval_budget": int(args.semantic_eval_budget),
        "scorer": str(summary.get("scorer", args.scorer)),
        "overall_accuracy": float(summary.get("overall_accuracy", 0.0)),
        "answer_type_accuracy": dict(summary.get("answer_type_accuracy", {})),
        "question_type_accuracy": dict(summary.get("question_type_accuracy", {})),
        "record_count": int(len(records)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[format-eval] wrote: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
