from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from train.mm import (
    QACollator,
    _to_device,
    build_loader,
    evaluate_records,
    load_runtime_from_checkpoint,
    resolve_device,
    run_generation_predictions,
    set_seed,
)


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
    ap = argparse.ArgumentParser(description="OCR subset + attention routing analysis for dual-VM checkpoints.")
    ap.add_argument("--dual_checkpoint", type=str, required=True)
    ap.add_argument("--anchor_checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit_ocr", type=int, default=500)
    ap.add_argument("--limit_control", type=int, default=100)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def bucket_for_question(question: str) -> str | None:
    q = str(question).strip().lower()
    for bucket, patterns in OCR_BUCKET_PATTERNS.items():
        if any(p in q for p in patterns):
            return bucket
    return None


def make_subset_loader(base_loader: DataLoader, indices: Sequence[int], *, batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int) -> DataLoader:
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


def collect_indices(dataset: Any, *, limit_ocr: int, limit_control: int) -> tuple[list[int], list[int], dict[int, str]]:
    ocr_indices: list[int] = []
    control_indices: list[int] = []
    bucket_by_index: dict[int, str] = {}
    items = getattr(dataset, "items", [])
    for idx, item in enumerate(items):
        bucket = bucket_for_question(item.get("question", ""))
        if bucket is not None and len(ocr_indices) < int(limit_ocr):
            ocr_indices.append(idx)
            bucket_by_index[idx] = bucket
        elif bucket is None and len(control_indices) < int(limit_control):
            control_indices.append(idx)
        if len(ocr_indices) >= int(limit_ocr) and len(control_indices) >= int(limit_control):
            break
    return ocr_indices, control_indices, bucket_by_index


def evaluate_subset(
    checkpoint: str,
    *,
    device: str,
    eval_split: str,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> Dict[str, Any]:
    overrides = {
        "batch_size": int(batch_size),
        "eval_batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "prefetch_factor": int(prefetch_factor),
        "pin_memory": bool(pin_memory),
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=checkpoint,
        device=device,
        args_override=overrides,
    )
    base_loader = build_loader(run_args, tokenizer=tokenizer, split=str(eval_split), train_mode=False, limit=0)
    loader = make_subset_loader(
        base_loader,
        indices,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    records = run_generation_predictions(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
        max_answer_length=int(run_args.max_answer_length),
        max_batches=0,
        logger=None,
        split_name=f"{eval_split}_ocr_subset",
        log_every=20,
        cuda_empty_cache_every=0,
    )
    summary = evaluate_records(records, qualitative_samples=0, confusion_top_k=0, scorer="official")
    return {
        "overall_accuracy": float(summary.get("overall_accuracy", 0.0)),
        "answer_type_accuracy": dict(summary.get("answer_type_accuracy", {})),
        "record_count": int(len(records)),
    }


@torch.no_grad()
def collect_attention_stats(
    checkpoint: str,
    *,
    device: str,
    eval_split: str,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    force_bucket: str | None = None,
) -> Dict[str, Any]:
    overrides = {
        "batch_size": int(batch_size),
        "eval_batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "prefetch_factor": int(prefetch_factor),
        "pin_memory": bool(pin_memory),
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=checkpoint,
        device=device,
        args_override=overrides,
    )
    base_loader = build_loader(run_args, tokenizer=tokenizer, split=str(eval_split), train_mode=False, limit=0)
    loader = make_subset_loader(
        base_loader,
        indices,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    model.eval()
    bucket_vals: dict[str, list[float]] = defaultdict(list)
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        _, _k, aux = model.forward_logits(
            input_ids=batch["input_ids"],
            images=batch["images"],
            text_pad_mask=batch["text_pad_mask"],
            prompt_mask=batch.get("prompt_mask"),
            question_mask=batch.get("question_mask"),
            return_aux=True,
            return_bridge_attn=True,
        )
        per_example = aux.get("vitstr_attn_fraction_per_example")
        if isinstance(per_example, torch.Tensor):
            vals = per_example.detach().float().cpu().tolist()
        else:
            vals = [float(aux.get("vitstr_attn_fraction", 0.0))] * len(raw_batch["question_ids"])
        for question, val in zip(raw_batch["questions"], vals):
            bucket = force_bucket if force_bucket is not None else (bucket_for_question(question) or "other")
            bucket_vals[str(bucket)].append(float(val))
    summary = {
        bucket: {
            "count": len(vals),
            "mean_vitstr_attn_fraction": (sum(vals) / max(1, len(vals))),
        }
        for bucket, vals in sorted(bucket_vals.items())
    }
    return summary


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    base_model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=args.dual_checkpoint,
        device=device,
        args_override={
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "pin_memory": bool(args.pin_memory),
        },
    )
    base_loader = build_loader(run_args, tokenizer=tokenizer, split=str(args.eval_split), train_mode=False, limit=0)
    del base_model
    ocr_indices, control_indices, bucket_by_index = collect_indices(
        base_loader.dataset,
        limit_ocr=int(args.limit_ocr),
        limit_control=int(args.limit_control),
    )
    dual_subset = evaluate_subset(
        args.dual_checkpoint,
        device=device,
        eval_split=str(args.eval_split),
        indices=ocr_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=int(args.prefetch_factor),
    )
    anchor_subset = evaluate_subset(
        args.anchor_checkpoint,
        device=device,
        eval_split=str(args.eval_split),
        indices=ocr_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=int(args.prefetch_factor),
    )
    ocr_attn = collect_attention_stats(
        args.dual_checkpoint,
        device=device,
        eval_split=str(args.eval_split),
        indices=ocr_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=int(args.prefetch_factor),
    )
    control_attn = collect_attention_stats(
        args.dual_checkpoint,
        device=device,
        eval_split=str(args.eval_split),
        indices=control_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=int(args.prefetch_factor),
        force_bucket="control",
    )

    out = {
        "dual_checkpoint": os.path.abspath(args.dual_checkpoint),
        "anchor_checkpoint": os.path.abspath(args.anchor_checkpoint),
        "ocr_subset_size": len(ocr_indices),
        "control_subset_size": len(control_indices),
        "ocr_subset_dual": dual_subset,
        "ocr_subset_anchor": anchor_subset,
        "ocr_subset_delta_overall": float(dual_subset["overall_accuracy"] - anchor_subset["overall_accuracy"]),
        "ocr_attention_by_bucket": ocr_attn,
        "control_attention": control_attn,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[dualvm-ocr] wrote: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
