from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from evals.vqa import vqa_official_accuracy
from train.mm import _to_device, build_loader, load_runtime_from_checkpoint, resolve_device, set_seed


def _trim_generated(ids: List[int], eos_id: int) -> List[int]:
    out: List[int] = []
    for tok in ids:
        if int(tok) == int(eos_id):
            break
        out.append(int(tok))
    if out:
        return out
    return [int(eos_id)]


def _ece_bins(records: List[Dict[str, Any]], n_bins: int) -> List[Dict[str, Any]]:
    bins: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, int(n_bins)))]
    for rec in records:
        conf = max(0.0, min(1.0, float(rec["confidence"])))
        idx = min(len(bins) - 1, int(math.floor(conf * len(bins))))
        bins[idx].append(rec)
    out: List[Dict[str, Any]] = []
    for idx, items in enumerate(bins):
        if not items:
            out.append(
                {
                    "bin_index": int(idx),
                    "count": 0,
                    "confidence_mean": 0.0,
                    "accuracy_mean": 0.0,
                    "confidence_lo": float(idx) / float(len(bins)),
                    "confidence_hi": float(idx + 1) / float(len(bins)),
                }
            )
            continue
        conf_mean = float(sum(float(x["confidence"]) for x in items) / float(len(items)))
        acc_mean = float(sum(float(x["accuracy"]) for x in items) / float(len(items)))
        out.append(
            {
                "bin_index": int(idx),
                "count": int(len(items)),
                "confidence_mean": conf_mean,
                "accuracy_mean": acc_mean,
                "confidence_lo": float(idx) / float(len(bins)),
                "confidence_hi": float(idx + 1) / float(len(bins)),
            }
        )
    return out


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Calibration Analysis",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- split: `{report['split']}`",
        f"- samples: `{report['num_samples']}`",
        f"- ECE: `{report['ece']:.6f}`",
        "",
        "| bin | n | conf | acc |",
        "|---|---:|---:|---:|",
    ]
    for row in report["bins"]:
        lines.append(
            f"| {row['bin_index']} | {row['count']} | {row['confidence_mean']:.4f} | {row['accuracy_mean']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=96)
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
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, default=None)
    ap.add_argument("--predictions_jsonl", type=str, default=None)
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

    model.eval()
    records: List[Dict[str, Any]] = []
    pred_out = None
    if args.predictions_jsonl:
        pred_path = os.path.abspath(args.predictions_jsonl)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        pred_out = open(pred_path, "w", encoding="utf-8")

    for bidx, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        generated = model.generate_answers(
            images=batch["images"],
            prompt_ids=batch["prompt_ids"],
            pad_id=tokenizer.pad_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_answer_len,
        )
        score_toks = [_trim_generated(list(seq), tokenizer.eos_id) for seq in generated]
        score_input_ids = [list(prompt) + toks for prompt, toks in zip(batch["prompt_ids"], score_toks)]
        prompt_lengths = [len(x) for x in batch["prompt_ids"]]
        input_ids, text_pad_mask, prompt_mask, question_mask = model._pack_generation_text_batch(
            score_input_ids,
            prompt_lengths=prompt_lengths,
            pad_id=tokenizer.pad_id,
            device=batch["images"].device,
            legacy_prompt_mask=False,
        )
        text_emb = model.lm._embed_dropout(model.lm._embed(input_ids))
        visual_prefix, visual_features = model._compute_visual_prefix(
            images=batch["images"],
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            prompt_mask=prompt_mask,
            question_mask=question_mask,
        )
        logits, prefix_k = model._decode_with_visual_prefix(
            input_ids=input_ids,
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            visual_prefix=visual_prefix,
            visual_features=visual_features,
            images=batch["images"],
        )
        text_logits = logits[:, prefix_k:, :]
        next_logits = text_logits[:, :-1, :]
        targets = input_ids[:, 1:]

        for i, toks in enumerate(score_toks):
            pred_ids = list(generated[i])
            pred_text = tokenizer.decode(pred_ids, skip_special=True).strip()
            start = max(0, int(prompt_lengths[i]) - 1)
            end = start + int(len(toks))
            token_logits = next_logits[i, start:end, :]
            token_targets = targets[i, start:end]
            token_log_probs = F.log_softmax(token_logits.float(), dim=-1)
            chosen_log_probs = token_log_probs.gather(-1, token_targets.unsqueeze(-1)).squeeze(-1)
            confidence = float(torch.exp(chosen_log_probs.mean()).item()) if int(len(toks)) > 0 else 0.0
            accuracy = float(vqa_official_accuracy(pred_text, batch["all_answers_raw"][i]))
            rec = {
                "question_id": int(batch["question_ids"][i]),
                "image_id": int(batch["image_ids"][i]),
                "question": batch["questions"][i],
                "prediction": pred_text,
                "confidence": confidence,
                "accuracy": accuracy,
                "token_count": int(len(toks)),
            }
            records.append(rec)
            if pred_out is not None:
                pred_out.write(json.dumps(rec, ensure_ascii=True) + "\n")

        print(
            f"[calib] batch={bidx + 1}"
            + (f"/{int(args.max_batches)}" if int(args.max_batches) > 0 else "")
            + f" records={len(records)}"
        )
        if int(args.max_batches) > 0 and (bidx + 1) >= int(args.max_batches):
            break

    if pred_out is not None:
        pred_out.close()

    bins = _ece_bins(records, int(args.bins))
    total = float(sum(row["count"] for row in bins))
    ece = 0.0
    if total > 0:
        for row in bins:
            if int(row["count"]) <= 0:
                continue
            ece += (float(row["count"]) / total) * abs(float(row["accuracy_mean"]) - float(row["confidence_mean"]))

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": str(args.split),
        "num_samples": int(len(records)),
        "ece": float(ece),
        "bins": bins,
        "mean_confidence": (float(sum(r["confidence"] for r in records)) / float(len(records))) if records else 0.0,
        "mean_accuracy": (float(sum(r["accuracy"] for r in records)) / float(len(records))) if records else 0.0,
    }

    out_json = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(f"[calib] wrote: {out_json}")

    if args.output_md:
        out_md = os.path.abspath(args.output_md)
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(_render_markdown(report))
        print(f"[calib] wrote: {out_md}")


if __name__ == "__main__":
    main()
