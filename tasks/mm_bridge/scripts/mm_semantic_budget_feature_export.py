from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch

from train.mm import (
    _to_device,
    build_loader,
    load_prefix_remap_checkpoint,
    load_runtime_from_checkpoint,
    resolve_device,
    set_seed,
)


def _parse_budgets(raw: str) -> List[int]:
    values: List[int] = []
    seen: set[int] = set()
    for part in str(raw or "").split(","):
        item = part.strip()
        if not item:
            continue
        budget = max(1, int(item))
        if budget in seen:
            continue
        seen.add(budget)
        values.append(budget)
    return sorted(values)


def _trim_or_pad(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = int(x.shape[-1])
    if cur == int(target_dim):
        return x
    if cur > int(target_dim):
        return x[..., : int(target_dim)]
    pad = torch.zeros((*x.shape[:-1], int(target_dim) - cur), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _mean_pool_with_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    keep = (~mask).to(dtype=x.dtype)
    denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * keep.unsqueeze(-1)).sum(dim=1) / denom


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export pooled low-budget semantic and question features for learned budget prediction.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--budgets", type=str, default="2,4,8")
    ap.add_argument("--eval_split", type=str, default="val")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=1)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--gqa_root", type=str, default=None)
    ap.add_argument("--chartqa_db_path", type=str, default=None)
    ap.add_argument("--textocr_annotations_root", type=str, default=None)
    ap.add_argument("--textocr_images_root", type=str, default=None)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--disable_lm_visual_adapters", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--apply_prefix_remap_in_forward", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--prefix_remap_checkpoint", type=str, default="")
    ap.add_argument("--prefix_feature_dim", type=int, default=128)
    ap.add_argument("--question_feature_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_dir", type=str, required=True)
    return ap.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    budgets = _parse_budgets(args.budgets)
    if not budgets:
        raise ValueError("No budgets provided.")

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides: Dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "images_root": args.images_root,
        "annotations_root": args.annotations_root,
        "gqa_root": args.gqa_root,
        "chartqa_db_path": args.chartqa_db_path,
        "textocr_annotations_root": args.textocr_annotations_root,
        "textocr_images_root": args.textocr_images_root,
        "apply_prefix_remap_in_forward": bool(args.apply_prefix_remap_in_forward),
        "disable_lm_visual_adapters": bool(args.disable_lm_visual_adapters),
        "prefix_remap_present": bool(args.apply_prefix_remap_in_forward)
        or bool(str(args.prefix_remap_checkpoint or "").strip()),
        "prefix_remap_checkpoint": str(args.prefix_remap_checkpoint or ""),
        "semantic_eval_budget": 0,
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        args_override=overrides,
    )
    if str(args.prefix_remap_checkpoint or "").strip():
        load_prefix_remap_checkpoint(model, checkpoint_path=str(args.prefix_remap_checkpoint), logger=None)
    semantic_mod = getattr(getattr(model, "bridge", None), "semantic_bottleneck", None)
    if semantic_mod is None or not hasattr(semantic_mod, "set_eval_budget"):
        raise RuntimeError("Checkpoint does not expose a configurable semantic bottleneck.")

    loader = build_loader(
        run_args,
        tokenizer=tokenizer,
        split=str(args.eval_split),
        train_mode=False,
        limit=max(0, int(args.limit_eval)),
    )

    question_ids: List[int] = []
    questions: List[str] = []
    answer_types: List[str] = []
    dataset_names: List[str] = []
    question_feat_tensor: torch.Tensor | None = None

    for budget in budgets:
        semantic_mod.set_eval_budget(int(budget))
        prefix_features: List[torch.Tensor] = []
        question_features: List[torch.Tensor] = []
        prefix_norms: List[torch.Tensor] = []
        qids_budget: List[int] = []
        questions_budget: List[str] = []
        answer_types_budget: List[str] = []
        dataset_names_budget: List[str] = []

        print(f"[feature-export] START split={args.eval_split} budget={budget}", flush=True)
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            text_emb = model.lm._embed_dropout(model.lm._embed(batch["input_ids"]))
            prefix, _ = model._compute_visual_prefix(
                images=batch["images"],
                text_emb=text_emb,
                text_pad_mask=batch["text_pad_mask"],
                prompt_mask=batch.get("prompt_mask"),
                question_mask=batch.get("question_mask"),
            )
            prefix_mean = prefix.mean(dim=1)
            prefix_feat = _trim_or_pad(prefix_mean, int(args.prefix_feature_dim)).detach().cpu().to(dtype=torch.float16)
            question_mean = _mean_pool_with_mask(text_emb, batch.get("question_mask"))
            question_feat = _trim_or_pad(question_mean, int(args.question_feature_dim)).detach().cpu().to(dtype=torch.float16)
            prefix_norm = prefix.norm(dim=-1).mean(dim=1).detach().cpu().to(dtype=torch.float32)

            prefix_features.append(prefix_feat)
            question_features.append(question_feat)
            prefix_norms.append(prefix_norm)
            qids_budget.extend(int(v) for v in batch["question_ids"])
            questions_budget.extend(str(v) for v in batch["questions"])
            answer_types_budget.extend(str(m.get("answer_type", "other")) for m in batch["metadata"])
            dataset_names_budget.extend(str(m.get("source_dataset", "")) for m in batch["metadata"])

        if not prefix_features:
            raise RuntimeError(f"No features extracted for budget={budget}")

        prefix_tensor = torch.cat(prefix_features, dim=0)
        q_tensor = torch.cat(question_features, dim=0)
        norm_tensor = torch.cat(prefix_norms, dim=0)
        payload = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "eval_split": str(args.eval_split),
            "budget": int(budget),
            "question_ids": torch.tensor(qids_budget, dtype=torch.long),
            "prefix_features": prefix_tensor,
            "question_features": q_tensor,
            "prefix_mean_norm": norm_tensor,
            "questions": questions_budget,
            "answer_types": answer_types_budget,
            "dataset_names": dataset_names_budget,
        }
        out_path = output_dir / f"features_k{budget}.pt"
        torch.save(payload, out_path)
        if question_feat_tensor is None:
            question_ids = qids_budget
            questions = questions_budget
            answer_types = answer_types_budget
            dataset_names = dataset_names_budget
            question_feat_tensor = q_tensor
        else:
            if qids_budget != question_ids:
                raise RuntimeError(f"Mismatched question id order for budget={budget}")
        print(f"[feature-export] END   split={args.eval_split} budget={budget} wrote={out_path}", flush=True)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    meta_path = output_dir / "feature_manifest.json"
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": os.path.abspath(args.checkpoint),
                "eval_split": str(args.eval_split),
                "budgets": [int(b) for b in budgets],
                "record_count": len(question_ids),
                "question_ids": [int(v) for v in question_ids],
                "questions": questions,
                "answer_types": answer_types,
                "dataset_names": dataset_names,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"[feature-export] wrote manifest: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
