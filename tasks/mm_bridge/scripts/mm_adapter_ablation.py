from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch

from train.mm import (
    build_loader,
    evaluate_records,
    load_runtime_from_checkpoint,
    resolve_device,
    run_generation_predictions,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Progressive LM visual-adapter ablation for MM checkpoints.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--gqa_root", type=str, default=None)
    ap.add_argument("--gqa_eval_group", type=str, default="", choices=["", "spatial", "attribute", "count", "exist"])
    ap.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test", "gqa_train", "gqa_val"])
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=0)
    ap.add_argument("--max_answer_length", type=int, default=None)
    ap.add_argument("--keep_counts", type=str, default="3,2,1,0")
    ap.add_argument("--scorer", type=str, default="official", choices=["official", "proxy"])
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def _capture_adapter_gates(model: Any) -> Dict[int, tuple[torch.Tensor, torch.Tensor]]:
    saved: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in list(getattr(model, "visual_adapter_layer_ids", [])):
        mod = model.visual_adapters[str(layer_idx)]
        saved[int(layer_idx)] = (
            mod.cross_gate_logit.detach().clone(),
            mod.ff_gate_logit.detach().clone(),
        )
    return saved


def _set_adapter_keep_count(
    model: Any,
    keep_count: int,
    *,
    base_gates: Dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    keep_n = max(0, int(keep_count))
    layer_ids = list(getattr(model, "visual_adapter_layer_ids", []))
    keep_ids = set(layer_ids[-keep_n:]) if keep_n > 0 else set()
    for layer_idx in layer_ids:
        mod = model.visual_adapters[str(layer_idx)]
        enabled = layer_idx in keep_ids
        if enabled:
            cross_gate, ff_gate = base_gates[int(layer_idx)]
            mod.cross_gate_logit.data.copy_(cross_gate.to(device=mod.cross_gate_logit.device, dtype=mod.cross_gate_logit.dtype))
            mod.ff_gate_logit.data.copy_(ff_gate.to(device=mod.ff_gate_logit.device, dtype=mod.ff_gate_logit.dtype))
        else:
            logit = torch.logit(torch.tensor(1e-6)).item()
            mod.cross_gate_logit.data.fill_(float(logit))
            mod.ff_gate_logit.data.fill_(float(logit))


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    overrides = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "images_root": args.images_root,
        "annotations_root": args.annotations_root,
        "gqa_root": args.gqa_root,
        "gqa_eval_group": str(args.gqa_eval_group or ""),
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        args_override=overrides,
    )
    if args.images_root:
        run_args.images_root = args.images_root
    if args.annotations_root:
        run_args.annotations_root = args.annotations_root
    if args.gqa_root:
        run_args.gqa_root = args.gqa_root
    run_args.gqa_eval_group = str(args.gqa_eval_group or "")
    run_args.batch_size = int(args.batch_size)
    run_args.eval_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    run_args.prefetch_factor = int(args.prefetch_factor)
    run_args.pin_memory = bool(args.pin_memory)
    loader = build_loader(
        run_args,
        tokenizer=tokenizer,
        split=str(args.eval_split),
        train_mode=False,
        limit=max(0, int(args.limit_eval)),
    )
    max_answer_len = int(args.max_answer_length) if args.max_answer_length is not None else int(run_args.max_answer_length)
    base_gates = _capture_adapter_gates(model)

    keep_counts = [int(x.strip()) for x in str(args.keep_counts).split(",") if x.strip()]
    results: List[Dict[str, Any]] = []
    for keep_n in keep_counts:
        _set_adapter_keep_count(model, keep_n, base_gates=base_gates)
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
        )
        summary = evaluate_records(
            records,
            qualitative_samples=0,
            confusion_top_k=0,
            scorer=str(args.scorer),
        )
        row = {
            "keep_count": int(keep_n),
            "overall_accuracy": float(summary.get("overall_accuracy", 0.0)),
            "scorer": str(summary.get("scorer", args.scorer)),
            "answer_type_accuracy": dict(summary.get("answer_type_accuracy", {})),
            "question_type_accuracy": dict(summary.get("question_type_accuracy", {})),
            "heuristic_category_accuracy": dict(summary.get("heuristic_category_accuracy", {})),
            "record_count": int(len(records)),
        }
        results.append(row)
        print(f"[adapter-ablation] keep={keep_n} acc={row['overall_accuracy']:.4f}")

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[adapter-ablation] wrote: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
