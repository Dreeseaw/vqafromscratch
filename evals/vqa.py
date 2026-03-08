"""
VQA evaluation utilities.

Role:
- Compute VQAv2-style metrics and breakdowns from prediction records.
- Provide standalone checkpoint eval (`--mm_checkpoint`) without running training.
- Evaluate outputs from an explicit vision-feature -> bridge -> LM setup (no implicit LM-ready vision tokens).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence

from train.vqa_data import (
    heuristic_answer_type,
    heuristic_question_category,
    majority_answer,
    normalize_vqa_answer,
    vqa_soft_accuracy,
)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _answer_type_for_record(record: Dict[str, Any]) -> str:
    meta = record.get("metadata") or {}
    at = meta.get("answer_type")
    if isinstance(at, str) and at.strip():
        return at.strip().lower()
    return heuristic_answer_type(str(record.get("canonical_answer", "")))


def _question_type_for_record(record: Dict[str, Any]) -> str:
    meta = record.get("metadata") or {}
    qt = meta.get("question_type")
    if isinstance(qt, str) and qt.strip():
        return qt.strip().lower()
    return heuristic_question_category(str(record.get("question", "")))


def _heuristic_question_category_for_record(record: Dict[str, Any]) -> str:
    return heuristic_question_category(str(record.get("question", "")))


def summarize_vqa_predictions(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "num_samples": int(len(records)),
        "overall_accuracy": None,
        "answer_type_accuracy": {},
        "question_type_accuracy": {},
        "heuristic_category_accuracy": {},
    }
    if not records:
        return out

    has_labels = any(bool(r.get("all_answers")) for r in records)
    if not has_labels:
        return out

    total_acc = 0.0
    at_sum = defaultdict(float)
    at_n = defaultdict(int)
    qt_sum = defaultdict(float)
    qt_n = defaultdict(int)
    hc_sum = defaultdict(float)
    hc_n = defaultdict(int)

    for r in records:
        answers = r.get("all_answers") or []
        if not answers:
            continue
        pred = str(r.get("prediction", ""))
        acc = vqa_soft_accuracy(pred, answers)
        total_acc += acc

        at = _answer_type_for_record(r)
        qt = _question_type_for_record(r)
        hc = _heuristic_question_category_for_record(r)
        at_sum[at] += acc
        at_n[at] += 1
        qt_sum[qt] += acc
        qt_n[qt] += 1
        hc_sum[hc] += acc
        hc_n[hc] += 1

    n = sum(1 for r in records if r.get("all_answers"))
    out["overall_accuracy"] = _safe_div(total_acc, float(n))
    out["answer_type_accuracy"] = {k: _safe_div(at_sum[k], float(at_n[k])) for k in sorted(at_sum.keys())}
    out["question_type_accuracy"] = {k: _safe_div(qt_sum[k], float(qt_n[k])) for k in sorted(qt_sum.keys())}
    out["heuristic_category_accuracy"] = {k: _safe_div(hc_sum[k], float(hc_n[k])) for k in sorted(hc_sum.keys())}
    return out


def format_qualitative_samples(records: Sequence[Dict[str, Any]], n: int = 8, seed: int = 35) -> List[Dict[str, Any]]:
    if n <= 0 or not records:
        return []
    idxs = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    picked = idxs[: int(n)]
    out: List[Dict[str, Any]] = []
    for i in picked:
        r = records[i]
        gt = majority_answer(r.get("all_answers") or [])
        out.append(
            {
                "question_id": r.get("question_id"),
                "question": r.get("question", ""),
                "prediction": r.get("prediction", ""),
                "canonical_answer": r.get("canonical_answer", ""),
                "gt_majority": gt,
                "answer_type": _answer_type_for_record(r),
                "question_type": _question_type_for_record(r),
            }
        )
    return out


def build_confusion_summary(records: Sequence[Dict[str, Any]], top_k: int = 20, short_max_words: int = 2) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    pair_counts = Counter()
    gt_counts = Counter()
    for r in records:
        answers = r.get("all_answers") or []
        if not answers:
            continue
        gt = majority_answer(answers)
        pr = normalize_vqa_answer(str(r.get("prediction", "")))
        if not gt or not pr:
            continue
        if gt == pr:
            continue
        if len(gt.split()) > short_max_words or len(pr.split()) > short_max_words:
            continue
        gt_counts[gt] += 1
        pair_counts[(gt, pr)] += 1

    rows: List[Dict[str, Any]] = []
    for (gt, pr), c in pair_counts.most_common(int(top_k)):
        rows.append(
            {
                "gt": gt,
                "pred": pr,
                "count": int(c),
                "gt_total": int(gt_counts[gt]),
                "rate_within_gt": _safe_div(float(c), float(gt_counts[gt])),
            }
        )
    return rows


def _load_predictions_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _save_predictions_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def _print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_jsonl", type=str, default=None)
    ap.add_argument("--mm_checkpoint", type=str, default=None)

    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--max_answer_length", type=int, default=None)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--debug_shapes", action="store_true")

    ap.add_argument("--qualitative_samples", type=int, default=8)
    ap.add_argument("--confusion_top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=35)

    ap.add_argument("--save_predictions_jsonl", type=str, default=None)
    ap.add_argument("--save_summary_json", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions_jsonl and not args.mm_checkpoint:
        raise SystemExit("Specify either --predictions_jsonl or --mm_checkpoint")

    if args.predictions_jsonl:
        records = _load_predictions_jsonl(args.predictions_jsonl)
    else:
        from train.mm import run_predictions_from_checkpoint

        records = run_predictions_from_checkpoint(
            checkpoint_path=args.mm_checkpoint,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            images_root=args.images_root,
            annotations_root=args.annotations_root,
            limit=args.limit,
            max_batches=args.max_batches,
            max_answer_length=args.max_answer_length,
            debug_shapes=args.debug_shapes,
            progress_every=args.log_every,
        )
        if args.save_predictions_jsonl:
            _save_predictions_jsonl(args.save_predictions_jsonl, records)

    summary = summarize_vqa_predictions(records)
    summary["qualitative"] = format_qualitative_samples(records, n=args.qualitative_samples, seed=args.seed)
    summary["confusions"] = build_confusion_summary(records, top_k=args.confusion_top_k)
    _print_summary(summary)

    if args.save_summary_json:
        with open(args.save_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
