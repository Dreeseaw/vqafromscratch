from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from evals.vqa import vqa_official_accuracy
from train.mm import _to_device, build_loader, load_runtime_from_checkpoint, resolve_device, set_seed
from train.vqa_data import COLOR_MEAN, COLOR_STD


def _denorm_image(image: torch.Tensor) -> np.ndarray:
    x = image.detach().float().cpu()
    mean = torch.tensor(COLOR_MEAN, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(COLOR_STD, dtype=x.dtype).view(3, 1, 1)
    x = (x * std) + mean
    x = x.clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def _reshape_token_map(values: np.ndarray) -> np.ndarray:
    n = int(values.shape[0])
    side = int(math.isqrt(n))
    if side * side == n:
        return values.reshape(side, side)
    h = max(1, side)
    w = int(math.ceil(float(n) / float(h)))
    out = np.zeros((h * w,), dtype=np.float32)
    out[:n] = values.astype(np.float32)
    return out.reshape(h, w)


def _heat_overlay(base: np.ndarray, heat: np.ndarray) -> Image.Image:
    heat = heat.astype(np.float32)
    if float(heat.max()) > float(heat.min()):
        heat = (heat - float(heat.min())) / float(heat.max() - float(heat.min()))
    else:
        heat = np.zeros_like(heat, dtype=np.float32)
    heat_img = Image.fromarray((heat * 255.0).round().astype(np.uint8)).resize(
        (int(base.shape[1]), int(base.shape[0])),
        resample=Image.Resampling.BILINEAR,
    )
    heat_arr = np.asarray(heat_img, dtype=np.float32) / 255.0
    overlay = base.astype(np.float32).copy()
    overlay[..., 0] = np.clip(overlay[..., 0] * 0.55 + 255.0 * heat_arr * 0.45, 0.0, 255.0)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - 0.25 * heat_arr), 0.0, 255.0)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1.0 - 0.25 * heat_arr), 0.0, 255.0)
    return Image.fromarray(overlay.astype(np.uint8))


def _prompt_views(model: Any, batch: Dict[str, Any], tokenizer: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_lengths = [len(x) for x in batch["prompt_ids"]]
    return model._pack_generation_text_batch(
        batch["prompt_ids"],
        prompt_lengths=prompt_lengths,
        pad_id=tokenizer.pad_id,
        device=batch["images"].device,
        legacy_prompt_mask=False,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--lm_checkpoint_override", type=str, default=None)
    ap.add_argument("--vision_checkpoint_override", type=str, default=None)
    ap.add_argument("--tokenizer_path_override", type=str, default=None)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--max_answer_length", type=int, default=None)
    ap.add_argument("--num_correct", type=int, default=100)
    ap.add_argument("--num_incorrect", type=int, default=100)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_dir", type=str, required=True)
    return ap.parse_args()


@torch.no_grad()
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
        "lm_checkpoint": args.lm_checkpoint_override,
        "vision_checkpoint": args.vision_checkpoint_override,
        "tokenizer_path": args.tokenizer_path_override,
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
    run_args.batch_size = int(args.batch_size)
    run_args.eval_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    run_args.prefetch_factor = int(args.prefetch_factor)
    run_args.pin_memory = bool(args.pin_memory)
    loader = build_loader(
        run_args,
        tokenizer=tokenizer,
        split=str(args.split),
        train_mode=False,
        limit=(int(args.limit_eval) if int(args.limit_eval) > 0 else 0),
    )
    max_answer_len = int(args.max_answer_length) if args.max_answer_length is not None else int(run_args.max_answer_length)

    out_dir = os.path.abspath(args.output_dir)
    correct_dir = os.path.join(out_dir, "correct")
    incorrect_dir = os.path.join(out_dir, "incorrect")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    target_correct = max(0, int(args.num_correct))
    target_incorrect = max(0, int(args.num_incorrect))
    keep_correct = 0
    keep_incorrect = 0

    model.eval()
    for bidx, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        prompt_input_ids, prompt_text_pad_mask, prompt_mask, question_mask = _prompt_views(model, batch, tokenizer)
        prompt_text_emb = model.lm._embed_dropout(model.lm._embed(prompt_input_ids))
        question_context, question_tokens, question_token_mask = model._compute_question_views(
            text_emb=prompt_text_emb,
            text_pad_mask=prompt_text_pad_mask,
            prompt_mask=prompt_mask,
            question_mask=question_mask,
        )

        visual_features = model.vision_adapter(batch["images"])
        if getattr(model, "visual_feature_adapter_type", "none") != "none":
            visual_features = model.visual_feature_adapter(visual_features)

        if not hasattr(model.bridge, "_prepare_visual_tokens"):
            raise RuntimeError("Grounding inspection currently supports query-style bridges only.")
        visual_tokens = model.bridge._prepare_visual_tokens(visual_features, question_context=question_context)
        bridge_out = model.bridge(
            visual_features,
            question_context=question_context,
            question_tokens=question_tokens,
            question_token_mask=question_token_mask,
            return_attn=True,
        )
        if not isinstance(bridge_out, tuple) or len(bridge_out) != 2:
            raise RuntimeError("Bridge did not return attention maps for grounding inspection.")
        _visual_prefix, attn_maps = bridge_out
        if not attn_maps:
            raise RuntimeError("Grounding inspection received an empty attention map list.")
        last_attn = attn_maps[-1].detach().float().cpu().numpy()
        per_query = last_attn.mean(axis=1)

        generated = model.generate_answers(
            images=batch["images"],
            prompt_ids=batch["prompt_ids"],
            pad_id=tokenizer.pad_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_answer_len,
        )

        for i in range(len(generated)):
            pred_text = tokenizer.decode(generated[i], skip_special=True).strip()
            accuracy = float(vqa_official_accuracy(pred_text, batch["all_answers_raw"][i]))
            if accuracy >= 0.999:
                bucket = "correct"
                if keep_correct >= target_correct:
                    continue
                bucket_dir = correct_dir
                keep_correct += 1
            elif accuracy <= 1e-8:
                bucket = "incorrect"
                if keep_incorrect >= target_incorrect:
                    continue
                bucket_dir = incorrect_dir
                keep_incorrect += 1
            else:
                continue

            sample_q = per_query[i]
            combined = sample_q.mean(axis=0)
            combined_grid = _reshape_token_map(combined)
            per_query_grids = np.stack([_reshape_token_map(q) for q in sample_q], axis=0)
            base = _denorm_image(batch["images"][i])
            overlay = _heat_overlay(base, combined_grid)
            stem = f"{bucket}_{int(batch['question_ids'][i])}"
            overlay.save(os.path.join(bucket_dir, f"{stem}.png"))
            torch.save(
                {
                    "per_query_maps": torch.from_numpy(per_query_grids),
                    "combined_map": torch.from_numpy(combined_grid),
                    "visual_token_count": int(visual_tokens.shape[1]),
                },
                os.path.join(bucket_dir, f"{stem}.pt"),
            )
            manifest.append(
                {
                    "bucket": bucket,
                    "question_id": int(batch["question_ids"][i]),
                    "image_id": int(batch["image_ids"][i]),
                    "question": batch["questions"][i],
                    "prediction": pred_text,
                    "accuracy": accuracy,
                    "visual_token_count": int(visual_tokens.shape[1]),
                    "overlay_path": os.path.join(bucket_dir, f"{stem}.png"),
                    "tensor_path": os.path.join(bucket_dir, f"{stem}.pt"),
                }
            )

        print(
            f"[ground] batch={bidx + 1}"
            + (f"/{int(args.max_batches)}" if int(args.max_batches) > 0 else "")
            + f" kept_correct={keep_correct}/{target_correct} kept_incorrect={keep_incorrect}/{target_incorrect}"
        )
        if keep_correct >= target_correct and keep_incorrect >= target_incorrect:
            break
        if int(args.max_batches) > 0 and (bidx + 1) >= int(args.max_batches):
            break

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": str(args.split),
        "num_correct_saved": int(keep_correct),
        "num_incorrect_saved": int(keep_incorrect),
        "manifest": manifest,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[ground] wrote: {out_dir}")


if __name__ == "__main__":
    main()
