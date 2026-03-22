from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

from evals.vqa import vqa_official_accuracy
from train.vqa_data import resolve_annotation_file, resolve_question_file


_CATEGORY_PATTERNS: Sequence[tuple[str, re.Pattern[str]]] = (
    ("reading", re.compile(r"\b(what does .* say|what is written|what .* written|what .* sign say)\b")),
    ("counting", re.compile(r"\b(how many|number of|count)\b")),
    ("color", re.compile(r"\bwhat colou?r\b")),
    (
        "spatial",
        re.compile(r"\b(where|which side|left|right|above|below|under|behind|in front of|next to|between)\b"),
    ),
    ("action", re.compile(r"\bwhat (?:is|are) .* doing\b")),
    ("object_id", re.compile(r"\b(what is this|what is that|what kind|what type|which kind|which type)\b")),
    ("yesno_existence", re.compile(r"^(is|are) there\b")),
    ("yesno_attribute", re.compile(r"^(is|are|does|do|did|can|could|was|were|will)\b")),
)


def _load_answers_row(path: str, tag: str | None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    if tag is None:
        return rows[-1]
    tagged = [row for row in rows if str(row.get("tag", "")) == str(tag)]
    if not tagged:
        raise ValueError(f"No row with tag={tag!r} found in {path}")
    return tagged[-1]


def _normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _categorize(question: str) -> str:
    q = _normalize_question(question)
    for name, pattern in _CATEGORY_PATTERNS:
        if pattern.search(q):
            return name
    return "other"


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Fine-Grained VQA Breakdown",
        "",
        f"- answers: `{report['answers_jsonl']}`",
        f"- tag: `{report['selected_tag']}`",
        f"- split: `{report['split']}`",
        f"- samples: `{report['num_samples']}`",
        "",
        "| category | n | accuracy |",
        "|---|---:|---:|",
    ]
    for row in report["categories"]:
        lines.append(f"| {row['category']} | {row['count']} | {row['accuracy']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers_jsonl", type=str, required=True)
    ap.add_argument("--annotations_root", type=str, default="data/vqav2")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    row = _load_answers_row(str(args.answers_jsonl), args.tag)
    q_path = resolve_question_file(str(args.annotations_root), str(args.split))
    a_path = resolve_annotation_file(str(args.annotations_root), str(args.split))
    if a_path is None:
        raise FileNotFoundError(f"Missing annotations for split={args.split} under {args.annotations_root}")

    with open(q_path, "r", encoding="utf-8") as f:
        question_rows = json.load(f).get("questions", [])
    with open(a_path, "r", encoding="utf-8") as f:
        annotation_rows = json.load(f).get("annotations", [])

    question_by_qid = {int(q["question_id"]): str(q.get("question", "")) for q in question_rows}
    ann_by_qid = {int(a["question_id"]): a for a in annotation_rows}

    grouped_scores: Dict[str, List[float]] = defaultdict(list)
    missing = 0
    for answer_row in row.get("answers", []):
        qid = int(answer_row["question_id"])
        ann = ann_by_qid.get(qid)
        if ann is None:
            missing += 1
            continue
        question = question_by_qid.get(qid, str(answer_row.get("question", "")))
        category = _categorize(question)
        gt_answers = [str(x.get("answer", "")) for x in ann.get("answers", [])]
        prediction = str(answer_row.get("prediction") or "")
        grouped_scores[category].append(vqa_official_accuracy(prediction, gt_answers))

    cats: List[Dict[str, Any]] = []
    for category, scores in sorted(grouped_scores.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n = len(scores)
        cats.append(
            {
                "category": category,
                "count": int(n),
                "accuracy": (float(sum(scores)) / float(n)) if n > 0 else 0.0,
            }
        )

    report = {
        "answers_jsonl": os.path.abspath(args.answers_jsonl),
        "annotations_root": os.path.abspath(args.annotations_root),
        "split": str(args.split),
        "selected_tag": str(row.get("tag", "latest")),
        "global_step": int(row.get("global_step", -1)),
        "num_samples": int(sum(x["count"] for x in cats)),
        "missing_annotations": int(missing),
        "categories": cats,
    }

    out_json = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(f"[finebreak] wrote: {out_json}")

    if args.output_md:
        out_md = os.path.abspath(args.output_md)
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(_render_markdown(report))
        print(f"[finebreak] wrote: {out_md}")


if __name__ == "__main__":
    main()
