#!/usr/bin/env python3
"""
Compare serial vs batched eval KV-cache generation on a real MM bridge checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.mm import (
    build_loader,
    evaluate_records,
    load_runtime_from_checkpoint,
    resolve_device,
    run_generation_predictions,
)


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _answer_type_accuracy(summary: Dict[str, Any]) -> Dict[str, float]:
    raw = summary.get("answer_type_accuracy", {}) or {}
    return {str(k): float(v) for (k, v) in raw.items()}


def _run_mode(
    *,
    checkpoint_path: str,
    mode: str,
    split: str,
    device: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    limit: int,
    max_batches: int,
    progress_every: int,
    scorer: str,
    images_root: str | None,
    annotations_root: str | None,
) -> Dict[str, Any]:
    overrides = {
        "batch_size": int(batch_size),
        "eval_batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "prefetch_factor": int(prefetch_factor),
        "pin_memory": bool(pin_memory),
        "images_root": images_root,
        "annotations_root": annotations_root,
        "eval_use_kv_cache": True,
        "eval_kv_cache_mode": str(mode),
    }
    model, tokenizer, bridge_cfg, payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        args_override=overrides,
    )

    loader_args = argparse.Namespace(**vars(run_args))
    loader_args.batch_size = int(batch_size)
    loader_args.eval_batch_size = int(batch_size)
    loader_args.num_workers = int(num_workers)
    loader_args.prefetch_factor = int(prefetch_factor)
    loader_args.pin_memory = bool(pin_memory)
    if images_root:
        loader_args.images_root = str(images_root)
    if annotations_root:
        loader_args.annotations_root = str(annotations_root)

    loader = build_loader(
        loader_args,
        tokenizer=tokenizer,
        split=split,
        train_mode=False,
        limit=(int(limit) if int(limit) > 0 else 0),
    )

    t0 = time.perf_counter()
    records = run_generation_predictions(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
        max_answer_length=int(loader_args.max_answer_length),
        max_batches=int(max_batches),
        debug_shapes=False,
        logger=None,
        split_name=f"{split}:{mode}",
        log_every=int(progress_every),
    )
    elapsed_s = float(time.perf_counter() - t0)
    summary = evaluate_records(records, qualitative_samples=0, confusion_top_k=5, scorer=scorer)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return {
        "mode": str(mode),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "run_id": str(payload.get("train_args", {}).get("run_id", "")),
        "bridge_type": str(getattr(bridge_cfg, "bridge_type", "unknown")),
        "question_context_mode": str(getattr(run_args, "bridge_question_context_mode", "unknown")),
        "samples": int(len(records)),
        "elapsed_s": elapsed_s,
        "samples_per_s": (float(len(records)) / max(elapsed_s, 1e-6)),
        "overall_accuracy": float(summary.get("overall_accuracy", 0.0)),
        "answer_type_accuracy": _answer_type_accuracy(summary),
        "records": records,
    }


def _compare_records(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
    *,
    mismatch_limit: int,
) -> Dict[str, Any]:
    left_by_qid = {int(row["question_id"]): row for row in left}
    right_by_qid = {int(row["question_id"]): row for row in right}
    left_qids = set(left_by_qid.keys())
    right_qids = set(right_by_qid.keys())
    shared_qids = sorted(left_qids & right_qids)
    missing_left = sorted(right_qids - left_qids)
    missing_right = sorted(left_qids - right_qids)

    mismatches: List[Dict[str, Any]] = []
    for qid in shared_qids:
        lhs = left_by_qid[qid]
        rhs = right_by_qid[qid]
        if str(lhs.get("prediction", "")) == str(rhs.get("prediction", "")):
            continue
        mismatches.append(
            {
                "question_id": int(qid),
                "image_id": int(lhs.get("image_id", rhs.get("image_id", -1))),
                "question": str(lhs.get("question", "")),
                "serial_prediction": str(lhs.get("prediction", "")),
                "batched_prediction": str(rhs.get("prediction", "")),
                "canonical_answer": str(lhs.get("canonical_answer", "")),
            }
        )

    shared_count = int(len(shared_qids))
    mismatch_count = int(len(mismatches))
    exact_match_count = int(shared_count - mismatch_count)
    exact_match_ratio = float(exact_match_count / shared_count) if shared_count > 0 else 0.0

    return {
        "shared_question_count": shared_count,
        "missing_in_serial_count": int(len(missing_left)),
        "missing_in_batched_count": int(len(missing_right)),
        "missing_in_serial": missing_left[: int(mismatch_limit)],
        "missing_in_batched": missing_right[: int(mismatch_limit)],
        "prediction_mismatch_count": mismatch_count,
        "prediction_exact_match_count": exact_match_count,
        "prediction_exact_match_ratio": exact_match_ratio,
        "mismatch_examples": mismatches[: int(mismatch_limit)],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to logs/<run_id>/step_<N>.tar")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=10)
    ap.add_argument("--progress_every", type=int, default=2)
    ap.add_argument("--eval_scorer", type=str, default="official", choices=["official", "proxy"])
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--mismatch_limit", type=int, default=10)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--predictions_dir", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.device))

    serial = _run_mode(
        checkpoint_path=str(args.checkpoint),
        mode="serial",
        split=str(args.split),
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        limit=int(args.limit),
        max_batches=int(args.max_batches),
        progress_every=int(args.progress_every),
        scorer=str(args.eval_scorer),
        images_root=args.images_root,
        annotations_root=args.annotations_root,
    )
    batched = _run_mode(
        checkpoint_path=str(args.checkpoint),
        mode="batched",
        split=str(args.split),
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        limit=int(args.limit),
        max_batches=int(args.max_batches),
        progress_every=int(args.progress_every),
        scorer=str(args.eval_scorer),
        images_root=args.images_root,
        annotations_root=args.annotations_root,
    )

    comparison = _compare_records(
        serial["records"],
        batched["records"],
        mismatch_limit=int(args.mismatch_limit),
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "max_batches": int(args.max_batches),
        "eval_scorer": str(args.eval_scorer),
        "serial": {k: v for (k, v) in serial.items() if k != "records"},
        "batched": {k: v for (k, v) in batched.items() if k != "records"},
        "comparison": comparison,
        "accuracy_delta_batched_minus_serial": (
            float(batched.get("overall_accuracy", 0.0)) - float(serial.get("overall_accuracy", 0.0))
        ),
    }

    if args.predictions_dir:
        os.makedirs(args.predictions_dir, exist_ok=True)
        serial_path = os.path.join(args.predictions_dir, "serial_predictions.jsonl")
        batched_path = os.path.join(args.predictions_dir, "batched_predictions.jsonl")
        _write_jsonl(serial_path, serial["records"])
        _write_jsonl(batched_path, batched["records"])
        report["serial_predictions_path"] = serial_path
        report["batched_predictions_path"] = batched_path

    out_path = str(args.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
