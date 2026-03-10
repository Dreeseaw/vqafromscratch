"""
Checkpoint-level diagnostics for frozen-bridge VQA models.

Purpose:
- Quantify how much predictions depend on image content.
- Measure visual-prefix geometry/stability seen by the frozen LM.
- Compare perturbation sensitivity across bridge variants.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from train.mm import (
    _to_device,
    build_loader,
    evaluate_records,
    load_runtime_from_checkpoint,
    resolve_device,
    set_seed,
)
from train.vqa_data import normalize_vqa_answer


VALID_MODES = ("clean", "shuffle", "zero", "noise", "fixed_image")


@dataclass
class PrefixStats:
    batches: int = 0
    samples: int = 0
    prefix_token_norm_mean_sum: float = 0.0
    prefix_token_norm_std_sum: float = 0.0
    prefix_global_std_sum: float = 0.0
    prefix_batch_variance_mean_sum: float = 0.0
    prefix_pairwise_cos_mean_sum: float = 0.0
    prefix_pairwise_cos_count: int = 0
    text_token_norm_mean_sum: float = 0.0
    prefix_text_norm_ratio_sum: float = 0.0
    delta_l2_vs_clean_sum: float = 0.0
    delta_cos_vs_clean_sum: float = 0.0
    delta_count: int = 0

    def update(
        self,
        prefix: torch.Tensor,
        text_emb: torch.Tensor,
        clean_prefix_flat: Optional[torch.Tensor] = None,
    ) -> None:
        if prefix.ndim != 3:
            raise ValueError(f"Expected prefix [B,K,D], got {tuple(prefix.shape)}")
        if text_emb.ndim != 3:
            raise ValueError(f"Expected text_emb [B,S,D], got {tuple(text_emb.shape)}")

        self.batches += 1
        b = int(prefix.shape[0])
        self.samples += b

        p = prefix.detach().float()
        t = text_emb.detach().float()

        token_norms = p.norm(dim=-1)
        self.prefix_token_norm_mean_sum += float(token_norms.mean().item())
        self.prefix_token_norm_std_sum += float(token_norms.std(unbiased=False).item())
        self.prefix_global_std_sum += float(p.std(unbiased=False).item())
        self.prefix_batch_variance_mean_sum += float(p.var(dim=0, unbiased=False).mean().item())

        t_norm = float(t.norm(dim=-1).mean().item())
        p_norm = float(token_norms.mean().item())
        self.text_token_norm_mean_sum += t_norm
        self.prefix_text_norm_ratio_sum += (p_norm / max(1e-8, t_norm))

        p_flat = p.reshape(b, -1)
        if b > 1:
            p_normed = F.normalize(p_flat, dim=-1, eps=1e-8)
            sim = torch.matmul(p_normed, p_normed.transpose(0, 1))
            mask = ~torch.eye(b, dtype=torch.bool, device=sim.device)
            pair_mean = float(sim[mask].mean().item()) if int(mask.sum().item()) > 0 else 0.0
            self.prefix_pairwise_cos_mean_sum += pair_mean
            self.prefix_pairwise_cos_count += 1

        if clean_prefix_flat is not None:
            if clean_prefix_flat.shape != p_flat.shape:
                raise ValueError(
                    f"clean/prefix shape mismatch: clean={tuple(clean_prefix_flat.shape)} mode={tuple(p_flat.shape)}"
                )
            c = clean_prefix_flat.detach().float()
            l2 = torch.norm(p_flat - c, dim=-1).mean().item()
            cos = F.cosine_similarity(p_flat, c, dim=-1, eps=1e-8).mean().item()
            self.delta_l2_vs_clean_sum += float(l2)
            self.delta_cos_vs_clean_sum += float(cos)
            self.delta_count += 1

    def to_dict(self) -> Dict[str, float]:
        n = max(1, self.batches)
        out = {
            "stats_batches": int(self.batches),
            "stats_samples": int(self.samples),
            "prefix_token_norm_mean": self.prefix_token_norm_mean_sum / n,
            "prefix_token_norm_std": self.prefix_token_norm_std_sum / n,
            "prefix_global_std": self.prefix_global_std_sum / n,
            "prefix_batch_variance_mean": self.prefix_batch_variance_mean_sum / n,
            "text_token_norm_mean": self.text_token_norm_mean_sum / n,
            "prefix_text_norm_ratio": self.prefix_text_norm_ratio_sum / n,
        }
        if self.prefix_pairwise_cos_count > 0:
            out["prefix_pairwise_cos_mean"] = self.prefix_pairwise_cos_mean_sum / self.prefix_pairwise_cos_count
        else:
            out["prefix_pairwise_cos_mean"] = float("nan")
        if self.delta_count > 0:
            out["prefix_delta_l2_vs_clean"] = self.delta_l2_vs_clean_sum / self.delta_count
            out["prefix_delta_cos_vs_clean"] = self.delta_cos_vs_clean_sum / self.delta_count
        else:
            out["prefix_delta_l2_vs_clean"] = float("nan")
            out["prefix_delta_cos_vs_clean"] = float("nan")
        return out


def _parse_modes(raw: str) -> List[str]:
    modes = [m.strip().lower() for m in str(raw).split(",") if m.strip()]
    if not modes:
        raise ValueError("No diagnostic modes specified.")
    bad = [m for m in modes if m not in VALID_MODES]
    if bad:
        raise ValueError(f"Unsupported mode(s): {bad}. Valid: {list(VALID_MODES)}")
    if "clean" in modes:
        modes = ["clean"] + [m for m in modes if m != "clean"]
    return modes


def _apply_mode(images: torch.Tensor, mode: str, noise_std: float, *, seed: int, batch_idx: int) -> torch.Tensor:
    if mode == "clean":
        return images
    if mode == "zero":
        return torch.zeros_like(images)
    if mode == "fixed_image":
        return images[:1].expand_as(images)
    if mode == "shuffle":
        if int(images.shape[0]) <= 1:
            return images
        g = torch.Generator(device=images.device)
        g.manual_seed(int(seed) + 7919 * int(batch_idx) + 17)
        perm = torch.randperm(int(images.shape[0]), generator=g, device=images.device)
        return images.index_select(0, perm)
    if mode == "noise":
        g = torch.Generator(device=images.device)
        g.manual_seed(int(seed) + 1543 * int(batch_idx) + 29)
        noise = torch.randn(images.shape, generator=g, device=images.device, dtype=images.dtype)
        return images + float(noise_std) * noise
    raise ValueError(f"Unsupported mode: {mode}")


def _make_record(batch: Dict[str, Any], i: int, prediction: str) -> Dict[str, Any]:
    return {
        "question_id": int(batch["question_ids"][i]),
        "image_id": int(batch["image_ids"][i]),
        "question": batch["questions"][i],
        "prediction": prediction,
        "canonical_answer": batch["answers"][i],
        "all_answers_raw": batch["all_answers_raw"][i],
        "all_answers": batch["all_answers"][i],
        "metadata": batch["metadata"][i],
    }


def _prediction_agreement(
    records: Sequence[Dict[str, Any]],
    clean_pred_by_qid: Dict[int, str],
) -> Dict[str, float]:
    if not clean_pred_by_qid:
        return {"agreement_with_clean": float("nan"), "agreement_count": 0}
    agree = 0
    total = 0
    for r in records:
        qid = int(r["question_id"])
        base = clean_pred_by_qid.get(qid)
        if base is None:
            continue
        pa = normalize_vqa_answer(str(r.get("prediction", "")))
        pb = normalize_vqa_answer(str(base))
        total += 1
        if pa == pb:
            agree += 1
    return {
        "agreement_with_clean": (float(agree) / float(total)) if total > 0 else float("nan"),
        "agreement_count": int(total),
    }


@torch.no_grad()
def _run_mode(
    *,
    mode: str,
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    max_answer_length: int,
    max_batches: int,
    stats_batches: int,
    noise_std: float,
    seed: int,
    clean_prefix_cache: Optional[List[torch.Tensor]] = None,
    progress_every: int = 10,
) -> Dict[str, Any]:
    model.eval()
    records: List[Dict[str, Any]] = []
    stats = PrefixStats()
    this_prefix_cache: List[torch.Tensor] = []

    for bidx, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        images_mode = _apply_mode(batch["images"], mode, noise_std, seed=seed, batch_idx=bidx)
        text_emb = model.lm._embed(batch["input_ids"])
        question_context: Optional[torch.Tensor] = None
        if bool(getattr(model.bridge, "supports_question_context", False)):
            valid = (~batch["text_pad_mask"]).unsqueeze(-1).float()
            denom = valid.sum(dim=1).clamp_min(1.0)
            question_context = (text_emb * valid).sum(dim=1) / denom

        needs_visual = bool(getattr(model.bridge, "requires_visual_features", True))
        if needs_visual:
            visual_features = model.vision_adapter(images_mode)
            if bool(getattr(model.bridge, "supports_question_context", False)):
                visual_prefix = model.bridge(visual_features, question_context=question_context)
            else:
                visual_prefix = model.bridge(visual_features)
        else:
            if bool(getattr(model.bridge, "supports_question_context", False)):
                visual_prefix = model.bridge(images_mode, question_context=question_context)
            else:
                visual_prefix = model.bridge(images_mode)
        if bidx < int(stats_batches):
            clean_ref = None
            if clean_prefix_cache is not None and bidx < len(clean_prefix_cache):
                clean_ref = clean_prefix_cache[bidx].to(device=visual_prefix.device, dtype=visual_prefix.dtype)
            stats.update(
                prefix=visual_prefix,
                text_emb=text_emb,
                clean_prefix_flat=(None if clean_ref is None else clean_ref.reshape(clean_ref.shape[0], -1)),
            )
            this_prefix_cache.append(visual_prefix.detach().float().reshape(visual_prefix.shape[0], -1).cpu())

        gens = model.generate_answers(
            images=images_mode,
            prompt_ids=batch["prompt_ids"],
            pad_id=tokenizer.pad_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=int(max_answer_length),
        )
        for i in range(len(gens)):
            pred = tokenizer.decode(gens[i], skip_special=True).strip()
            records.append(_make_record(batch, i, pred))

        if progress_every > 0 and (((bidx + 1) % int(progress_every) == 0) or bidx == 0):
            print(
                f"[diag] mode={mode} batch={bidx + 1}"
                + (f"/{max_batches}" if max_batches > 0 else "")
                + f" records={len(records)}"
            )
        if max_batches > 0 and (bidx + 1) >= int(max_batches):
            break

    return {
        "records": records,
        "prefix_stats": stats.to_dict(),
        "prefix_cache": this_prefix_cache,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# MM Bridge Diagnostics")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- checkpoint: `{report['checkpoint']}`")
    lines.append(f"- bridge_type: `{report['bridge_type']}`")
    lines.append(f"- requires_visual_features: `{int(bool(report['requires_visual_features']))}`")
    lines.append(f"- split: `{report['split']}`")
    lines.append(f"- scorer: `{report['scorer']}`")
    lines.append(f"- max_batches: `{report['max_batches']}`")
    lines.append(f"- modes: `{', '.join(report['modes'])}`")
    lines.append("")
    lines.append("## Accuracy and Agreement")
    lines.append("")
    lines.append("| mode | overall_acc | delta_vs_clean | agreement_with_clean | samples |")
    lines.append("|---|---:|---:|---:|---:|")
    clean_acc = report.get("clean_overall_accuracy")
    for mode in report["modes"]:
        row = report["results"][mode]
        acc = row.get("overall_accuracy")
        delta = row.get("delta_overall_accuracy_vs_clean")
        agree = row.get("agreement_with_clean")
        n = row.get("num_samples", 0)
        lines.append(
            "| {m} | {a} | {d} | {g} | {n} |".format(
                m=mode,
                a=("nan" if acc is None else f"{float(acc):.4f}"),
                d=("nan" if delta is None else f"{float(delta):+.4f}"),
                g=("nan" if agree != agree else f"{float(agree):.4f}"),
                n=int(n),
            )
        )
    lines.append("")
    lines.append("## Prefix Geometry")
    lines.append("")
    lines.append("| mode | prefix_batch_variance_mean | prefix_pairwise_cos_mean | prefix_text_norm_ratio | prefix_delta_l2_vs_clean | prefix_delta_cos_vs_clean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode in report["modes"]:
        ps = report["results"][mode]["prefix_stats"]
        lines.append(
            "| {m} | {v:.6f} | {c:.6f} | {r:.6f} | {l2:.6f} | {cc:.6f} |".format(
                m=mode,
                v=float(ps.get("prefix_batch_variance_mean", float("nan"))),
                c=float(ps.get("prefix_pairwise_cos_mean", float("nan"))),
                r=float(ps.get("prefix_text_norm_ratio", float("nan"))),
                l2=float(ps.get("prefix_delta_l2_vs_clean", float("nan"))),
                cc=float(ps.get("prefix_delta_cos_vs_clean", float("nan"))),
            )
        )
    lines.append("")
    if clean_acc is not None:
        lines.append("## Quick Read")
        lines.append("")
        for mode in report["modes"]:
            if mode == "clean":
                continue
            row = report["results"][mode]
            delta = row.get("delta_overall_accuracy_vs_clean")
            agree = row.get("agreement_with_clean")
            if delta is None:
                continue
            lines.append(
                f"- `{mode}` vs clean: accuracy delta `{float(delta):+.4f}`, prediction agreement `{float(agree):.4f}`"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="logs/<run_id>/step_<N>.tar")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--max_answer_length", type=int, default=None)
    ap.add_argument("--scorer", type=str, default="official", choices=["official", "proxy"])
    ap.add_argument("--qualitative_samples", type=int, default=0)
    ap.add_argument("--confusion_top_k", type=int, default=0)
    ap.add_argument("--modes", type=str, default="clean,shuffle,zero,noise,fixed_image")
    ap.add_argument("--noise_std", type=float, default=0.2)
    ap.add_argument("--stats_batches", type=int, default=50, help="Batches used for prefix-geometry stats.")
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    modes = _parse_modes(args.modes)

    overrides = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "images_root": args.images_root,
        "annotations_root": args.annotations_root,
    }
    model, tokenizer, bridge_cfg, payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        args_override=overrides,
    )
    if args.images_root:
        run_args.images_root = args.images_root
    if args.annotations_root:
        run_args.annotations_root = args.annotations_root
    run_args.batch_size = int(args.batch_size)
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

    report: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "run_id": str(getattr(run_args, "run_id", "")),
        "split": str(args.split),
        "device": str(device),
        "bridge_config": asdict(bridge_cfg),
        "bridge_type": str(bridge_cfg.bridge_type),
        "requires_visual_features": bool(getattr(model.bridge, "requires_visual_features", True)),
        "scorer": str(args.scorer),
        "modes": modes,
        "max_batches": int(args.max_batches),
        "stats_batches": int(args.stats_batches),
        "num_loader_samples": int(len(loader.dataset)),
        "results": {},
    }

    clean_pred_by_qid: Dict[int, str] = {}
    clean_overall_acc: Optional[float] = None
    clean_prefix_cache: Optional[List[torch.Tensor]] = None

    for mode in modes:
        mode_out = _run_mode(
            mode=mode,
            loader=loader,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_answer_length=max_answer_len,
            max_batches=int(args.max_batches),
            stats_batches=int(args.stats_batches),
            noise_std=float(args.noise_std),
            seed=int(args.seed),
            clean_prefix_cache=clean_prefix_cache,
            progress_every=int(args.progress_every),
        )
        records = mode_out["records"]
        summary = evaluate_records(
            records,
            qualitative_samples=int(args.qualitative_samples),
            confusion_top_k=int(args.confusion_top_k),
            scorer=str(args.scorer),
        )

        row: Dict[str, Any] = {
            "num_samples": int(len(records)),
            "overall_accuracy": summary.get("overall_accuracy"),
            "answer_type_accuracy": summary.get("answer_type_accuracy", {}),
            "question_type_accuracy": summary.get("question_type_accuracy", {}),
            "prefix_stats": mode_out["prefix_stats"],
        }

        if mode == "clean":
            clean_pred_by_qid = {int(r["question_id"]): str(r.get("prediction", "")) for r in records}
            clean_overall_acc = summary.get("overall_accuracy")
            clean_prefix_cache = mode_out["prefix_cache"]
            row["agreement_with_clean"] = 1.0
            row["agreement_count"] = int(len(records))
            row["delta_overall_accuracy_vs_clean"] = 0.0 if clean_overall_acc is not None else None
        else:
            agr = _prediction_agreement(records, clean_pred_by_qid)
            row.update(agr)
            if clean_overall_acc is not None and summary.get("overall_accuracy") is not None:
                row["delta_overall_accuracy_vs_clean"] = float(summary["overall_accuracy"]) - float(clean_overall_acc)
            else:
                row["delta_overall_accuracy_vs_clean"] = None

        report["results"][mode] = row

    report["clean_overall_accuracy"] = clean_overall_acc

    out_json = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(f"[diag] wrote: {out_json}")

    out_md = args.output_md
    if out_md:
        out_md = os.path.abspath(out_md)
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(_render_markdown(report))
        print(f"[diag] wrote: {out_md}")


if __name__ == "__main__":
    main()
