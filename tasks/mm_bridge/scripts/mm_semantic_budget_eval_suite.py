from __future__ import annotations

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

from evals.vqa import _answer_type_for_record, _question_type_for_record, _record_accuracy
from train.mm import (
    build_loader,
    evaluate_records,
    load_prefix_remap_checkpoint,
    load_runtime_from_checkpoint,
    resolve_device,
    run_generation_predictions,
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _print_answer_type_summary(tag: str, summary: Dict[str, Any]) -> None:
    by_answer = dict(summary.get("answer_type_accuracy", {}) or {})
    order = ("yes/no", "number", "other")
    rendered = [f"{key}={float(by_answer.get(key, 0.0)):.4f}" for key in order]
    for key in sorted(by_answer.keys()):
        if key not in order:
            rendered.append(f"{key}={float(by_answer.get(key, 0.0)):.4f}")
    print(f"[eval:{tag}] answer_type: " + " ".join(rendered), flush=True)


def _record_export_row(record: Dict[str, Any], *, budget: int, scorer: str) -> Dict[str, Any]:
    out = dict(record)
    out["budget"] = int(budget)
    out["accuracy"] = float(_record_accuracy(record, str(scorer)))
    out["answer_type"] = str(_answer_type_for_record(record))
    out["question_type"] = str(_question_type_for_record(record))
    stats = out.pop("generation_stats", None)
    if isinstance(stats, dict):
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                out[str(key)] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run fixed-K, oracle, and optional per-sample uncertainty artifact export for a semantic-budget checkpoint."
    )
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--budgets", type=str, default="2,4,8,16")
    ap.add_argument("--probe_budgets", type=str, default="1")
    ap.add_argument("--stats_budgets", type=str, default="2,4")
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
    ap.add_argument("--eval_split", type=str, default="val")
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=0)
    ap.add_argument("--max_answer_length", type=int, default=None)
    ap.add_argument("--scorer", type=str, default="official", choices=["official", "proxy", "exact", "text_exact"])
    ap.add_argument("--apply_prefix_remap_in_forward", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--disable_lm_visual_adapters", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefix_remap_checkpoint", type=str, default="")
    ap.add_argument("--eval_use_kv_cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval_kv_cache_mode", type=str, default="batched", choices=["serial", "batched"])
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_dir", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    budgets = _parse_budgets(args.budgets)
    probe_budgets = [budget for budget in _parse_budgets(args.probe_budgets) if budget not in budgets]
    stats_budgets = set(_parse_budgets(args.stats_budgets))
    if not budgets:
        raise ValueError("No oracle budgets provided.")

    all_budgets = sorted(set(budgets + probe_budgets))
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "gqa_root": args.gqa_root,
        "chartqa_db_path": args.chartqa_db_path,
        "textocr_annotations_root": args.textocr_annotations_root,
        "textocr_images_root": args.textocr_images_root,
        "apply_prefix_remap_in_forward": bool(args.apply_prefix_remap_in_forward),
        "disable_lm_visual_adapters": bool(args.disable_lm_visual_adapters),
        "use_prefix_remap": bool(args.apply_prefix_remap_in_forward),
        "prefix_remap_present": bool(args.apply_prefix_remap_in_forward)
        or bool(str(args.prefix_remap_checkpoint or "").strip()),
        "prefix_remap_checkpoint": str(args.prefix_remap_checkpoint or ""),
        "semantic_eval_budget": 0,
        "eval_use_kv_cache": bool(args.eval_use_kv_cache),
        "eval_kv_cache_mode": str(args.eval_kv_cache_mode),
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
    max_answer_len = int(args.max_answer_length) if args.max_answer_length is not None else int(run_args.max_answer_length)

    fixed_results: Dict[str, Dict[str, Any]] = {}
    qid_maps: Dict[int, Dict[int, Dict[str, Any]]] = {}
    expected_qids: set[int] | None = None

    for budget in all_budgets:
        collect_stats = int(budget) in stats_budgets
        print(f"[semantic-budget-suite] START budget={budget} collect_stats={int(collect_stats)}", flush=True)
        semantic_mod.set_eval_budget(int(budget))
        records = run_generation_predictions(
            model=model,
            loader=loader,
            tokenizer=tokenizer,
            device=device,
            max_answer_length=max_answer_len,
            max_batches=int(args.eval_batches),
            logger=None,
            split_name=f"{args.eval_split}_k{budget}",
            log_every=20,
            cuda_empty_cache_every=(400 if int(args.eval_batches) == 0 else 0),
            collect_generation_stats=collect_stats,
        )
        summary = evaluate_records(
            records,
            qualitative_samples=0,
            confusion_top_k=0,
            scorer=str(args.scorer),
        )
        qid_map: Dict[int, Dict[str, Any]] = {}
        export_rows: List[Dict[str, Any]] = []
        for record in records:
            qid = int(record["question_id"])
            qid_map[qid] = {
                "budget": int(budget),
                "accuracy": float(_record_accuracy(record, str(args.scorer))),
                "answer_type": str(_answer_type_for_record(record)),
                "record": record,
            }
            export_rows.append(_record_export_row(record, budget=int(budget), scorer=str(args.scorer)))
        qids = set(qid_map.keys())
        if expected_qids is None:
            expected_qids = qids
        elif qids != expected_qids:
            raise RuntimeError(f"Budget={budget} produced a mismatched question-id set.")
        qid_maps[int(budget)] = qid_map

        records_path = output_dir / f"fixed_k{budget}_records.jsonl"
        summary_path = output_dir / f"fixed_k{budget}_summary.json"
        _write_jsonl(records_path, export_rows)
        payload = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "budget": int(budget),
            "record_count": int(len(records)),
            "overall_accuracy": float(summary.get("overall_accuracy", 0.0) or 0.0),
            "answer_type_accuracy": dict(summary.get("answer_type_accuracy", {})),
            "question_type_accuracy": dict(summary.get("question_type_accuracy", {})),
            "scorer": str(summary.get("scorer", args.scorer)),
            "records_path": str(records_path),
            "stats_present": bool(collect_stats),
        }
        _write_json(summary_path, payload)
        fixed_results[str(budget)] = payload
        print(f"[eval:{args.eval_split}_k{budget}] overall_accuracy={payload['overall_accuracy']:.4f}", flush=True)
        print(
            f"[eval:{args.eval_split}_k{budget}] overall_accuracy={payload['overall_accuracy']:.4f} "
            f"scorer={payload['scorer']}",
            flush=True,
        )
        _print_answer_type_summary(f"{args.eval_split}_k{budget}", summary)
        print(f"[semantic-budget-suite] END   budget={budget} overall={payload['overall_accuracy']:.4f}", flush=True)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    if expected_qids is None:
        raise RuntimeError("No eval records produced.")

    oracle_records: List[Dict[str, Any]] = []
    selected_budgets: List[int] = []
    selected_budget_hist = Counter()
    for qid in sorted(expected_qids):
        candidates = [qid_maps[budget][qid] for budget in budgets]
        best_acc = max(float(c["accuracy"]) for c in candidates)
        best_budget = min(int(c["budget"]) for c in candidates if float(c["accuracy"]) == best_acc)
        chosen = next(c for c in candidates if int(c["budget"]) == best_budget and float(c["accuracy"]) == best_acc)
        selected_budgets.append(best_budget)
        selected_budget_hist[best_budget] += 1
        row = _record_export_row(chosen["record"], budget=int(best_budget), scorer=str(args.scorer))
        row["oracle_selected_budget"] = int(best_budget)
        oracle_records.append(row)

    oracle_summary = evaluate_records(
        oracle_records,
        qualitative_samples=0,
        confusion_top_k=0,
        scorer=str(args.scorer),
    )
    average_selected_budget = float(sum(selected_budgets)) / float(max(1, len(selected_budgets)))
    oracle_path = output_dir / "oracle_records.jsonl"
    _write_jsonl(oracle_path, oracle_records)

    best_fixed_budget = min(
        budgets,
        key=lambda budget: (-float(fixed_results[str(budget)]["overall_accuracy"]), int(budget)),
    )
    best_fixed_summary = fixed_results[str(best_fixed_budget)]
    best_fixed_overall = float(best_fixed_summary["overall_accuracy"])
    oracle_overall = float(oracle_summary.get("overall_accuracy", 0.0) or 0.0)
    tradeoff_rows: List[Dict[str, Any]] = []
    for budget in budgets:
        fixed = fixed_results[str(budget)]
        tradeoff_rows.append(
            {
                "label": f"fixed_k_{budget}",
                "mode": "fixed",
                "budget": int(budget),
                "average_budget": float(budget),
                "overall_accuracy": float(fixed["overall_accuracy"]),
            }
        )
    tradeoff_rows.append(
        {
            "label": "oracle",
            "mode": "oracle",
            "budget": None,
            "average_budget": float(average_selected_budget),
            "overall_accuracy": float(oracle_overall),
        }
    )

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "eval_split": str(args.eval_split),
        "scorer": str(oracle_summary.get("scorer", args.scorer)),
        "oracle_budgets": [int(x) for x in budgets],
        "probe_budgets": [int(x) for x in probe_budgets],
        "stats_budgets": [int(x) for x in sorted(stats_budgets)],
        "record_count": int(len(oracle_records)),
        "fixed_k": fixed_results,
        "best_fixed": {
            "budget": int(best_fixed_budget),
            "overall_accuracy": float(best_fixed_overall),
        },
        "oracle": {
            "overall_accuracy": float(oracle_overall),
            "answer_type_accuracy": dict(oracle_summary.get("answer_type_accuracy", {})),
            "question_type_accuracy": dict(oracle_summary.get("question_type_accuracy", {})),
            "average_selected_budget": float(average_selected_budget),
            "selected_budget_histogram": {str(k): int(v) for k, v in sorted(selected_budget_hist.items())},
            "selected_budget_fraction": {
                str(k): float(v) / float(max(1, len(selected_budgets))) for k, v in sorted(selected_budget_hist.items())
            },
            "records_path": str(oracle_path),
            "delta_vs_best_fixed": float(oracle_overall - best_fixed_overall),
        },
        "accuracy_vs_budget_tradeoff": tradeoff_rows,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, out)

    print(f"[eval:{args.eval_split}_oracle] overall_accuracy={oracle_overall:.4f}", flush=True)
    print(
        f"[oracle:{args.eval_split}] overall_accuracy={oracle_overall:.4f} "
        f"average_budget={average_selected_budget:.4f} best_fixed_k={best_fixed_budget}",
        flush=True,
    )
    print(f"[eval:{args.eval_split}_best_fixed] overall_accuracy={best_fixed_overall:.4f}", flush=True)
    print(
        f"[eval:best_fixed] overall_accuracy={best_fixed_overall:.4f} "
        f"scorer={best_fixed_summary['scorer']} budget={best_fixed_budget}",
        flush=True,
    )
    print(f"[semantic-budget-suite] wrote: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
