from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict

import torch

from torch.utils.data import DataLoader

from train.mm import QACollator, _to_device, load_runtime_from_checkpoint, resolve_device, set_seed
from train.vqa_data import PointingIndexDataset, build_image_transform


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate grounding target mass for MM checkpoints on the pointing index.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--pointing_index_path", type=str, required=True)
    ap.add_argument("--limit_eval", type=int, default=5000)
    ap.add_argument("--eval_batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


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
        "pointing_index_path": args.pointing_index_path,
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
    dataset = PointingIndexDataset(
        index_path=str(args.pointing_index_path),
        images_root=str(run_args.images_root),
        transform=build_image_transform(train_mode=False),
        limit=max(0, int(args.limit_eval)),
        skip_missing_images=True,
        target_len=196,
    )
    collator = QACollator(
        tokenizer=tokenizer,
        max_q=int(run_args.max_question_length),
        max_a=int(run_args.max_answer_length),
        max_text_tokens=int(run_args.max_text_tokens),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        collate_fn=collator,
    )
    model.eval()

    total = 0
    mass_sum = 0.0
    by_source_sum: Dict[str, float] = defaultdict(float)
    by_source_count: Dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for bidx, raw_batch in enumerate(loader):
            batch = _to_device(raw_batch, device)
            _ = model.forward_logits(
                input_ids=batch["input_ids"],
                images=batch["images"],
                text_pad_mask=batch["text_pad_mask"],
                prompt_mask=batch.get("prompt_mask"),
                question_mask=batch.get("question_mask"),
                return_aux=True,
                return_bridge_attn=True,
            )
            aux = getattr(model.bridge, "last_aux_info", {})
            perceiver_attn = aux.get("perceiver_final_attn")
            if not isinstance(perceiver_attn, torch.Tensor):
                raise RuntimeError("Grounding mass eval requires perceiver_final_attn in bridge aux.")
            attn_dist = perceiver_attn.float().mean(dim=1).mean(dim=1)
            if int(attn_dist.shape[-1]) > 196:
                attn_dist = attn_dist[:, :196]
            attn_dist = attn_dist / attn_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            target = batch["grounding_soft_target"].float()
            if int(target.shape[-1]) != int(attn_dist.shape[-1]):
                target = target[:, : int(attn_dist.shape[-1])]
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            mass = (attn_dist * target).sum(dim=-1)
            for i in range(int(mass.shape[0])):
                score = float(mass[i].item())
                meta = raw_batch["metadata"][i] if i < len(raw_batch["metadata"]) else {}
                source = str((meta or {}).get("source_dataset", "pointing"))
                mass_sum += score
                total += 1
                by_source_sum[source] += score
                by_source_count[source] += 1
            if int(args.eval_batches) > 0 and (bidx + 1) >= int(args.eval_batches):
                break

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "pointing_index_path": os.path.abspath(args.pointing_index_path),
        "record_count": int(total),
        "mean_target_mass": float(mass_sum / max(1, total)),
        "by_source": {
            src: {
                "count": int(by_source_count[src]),
                "mean_target_mass": float(by_source_sum[src] / max(1, by_source_count[src])),
            }
            for src in sorted(by_source_sum.keys())
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[grounding-mass] wrote: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
