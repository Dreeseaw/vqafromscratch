"""
VQA evaluation utilities.

Role:
- Compute VQAv2 metrics and breakdowns from prediction records.
- Provide standalone checkpoint eval (`--mm_checkpoint`) without running training.
- Support both a fast proxy scorer and an official-style VQAv2 scorer.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence

from train.vqa_data import (
    heuristic_answer_type,
    heuristic_question_category,
    normalize_vqa_answer,
    vqa_soft_accuracy,
)


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id've": "i'd've",
    "i'dve": "i'd've",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

_MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_ARTICLES = {"a", "an", "the"}
_PUNCT = [
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_WS_RE = re.compile(r"\s+")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in out) or (" " + p in out) or (_COMMA_STRIP.search(out) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out)
    return out


def _process_digit_article(text: str) -> str:
    out: List[str] = []
    for w in text.lower().split():
        w = _MANUAL_MAP.get(w, w)
        if w not in _ARTICLES:
            out.append(_CONTRACTIONS.get(w, w))
    return " ".join(out)


def normalize_vqa_official(text: str) -> str:
    out = str(text or "").replace("\n", " ").replace("\t", " ").strip().lower()
    out = _process_punctuation(out)
    out = _process_digit_article(out)
    out = _WS_RE.sub(" ", out).strip()
    return out


def _majority_answer(answers: Sequence[str], scorer: str) -> str:
    if not answers:
        return ""
    if scorer == "official":
        norm = [normalize_vqa_official(a) for a in answers if str(a).strip()]
    else:
        norm = [normalize_vqa_answer(a) for a in answers if str(a).strip()]
    norm = [x for x in norm if x]
    if not norm:
        return ""
    return Counter(norm).most_common(1)[0][0]


def vqa_official_accuracy(prediction: str, gt_answers: Sequence[str]) -> float:
    if not gt_answers:
        return 0.0
    pred = normalize_vqa_official(prediction)
    gts = [normalize_vqa_official(a) for a in gt_answers if str(a).strip()]
    if not gts:
        return 0.0

    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1 :]
        match_count = sum(1 for a in others if a == pred)
        accs.append(min(1.0, float(match_count) / 3.0))
    return float(sum(accs) / len(accs))


def _record_answers_for_scorer(record: Dict[str, Any], scorer: str) -> List[str]:
    if scorer == "official":
        raw = record.get("all_answers_raw")
        if isinstance(raw, list) and raw:
            return [str(x) for x in raw if str(x).strip()]
    ans = record.get("all_answers")
    if isinstance(ans, list):
        return [str(x) for x in ans if str(x).strip()]
    return []


def _record_accuracy(record: Dict[str, Any], scorer: str) -> float:
    answers = _record_answers_for_scorer(record, scorer)
    pred = str(record.get("prediction", ""))
    if scorer == "official":
        return vqa_official_accuracy(pred, answers)
    return vqa_soft_accuracy(pred, answers)


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


def summarize_vqa_predictions(records: Sequence[Dict[str, Any]], scorer: str = "official") -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "num_samples": int(len(records)),
        "scorer": str(scorer),
        "overall_accuracy": None,
        "answer_type_accuracy": {},
        "question_type_accuracy": {},
        "heuristic_category_accuracy": {},
        "official_fallback_to_normalized_count": 0,
    }
    if not records:
        return out

    has_labels = any(bool(_record_answers_for_scorer(r, scorer)) for r in records)
    if not has_labels:
        return out

    total_acc = 0.0
    at_sum = defaultdict(float)
    at_n = defaultdict(int)
    qt_sum = defaultdict(float)
    qt_n = defaultdict(int)
    hc_sum = defaultdict(float)
    hc_n = defaultdict(int)
    n = 0

    for r in records:
        answers = _record_answers_for_scorer(r, scorer)
        if not answers:
            continue
        if scorer == "official" and not (r.get("all_answers_raw") or []):
            out["official_fallback_to_normalized_count"] += 1
        acc = _record_accuracy(r, scorer)
        total_acc += acc
        n += 1

        at = _answer_type_for_record(r)
        qt = _question_type_for_record(r)
        hc = _heuristic_question_category_for_record(r)
        at_sum[at] += acc
        at_n[at] += 1
        qt_sum[qt] += acc
        qt_n[qt] += 1
        hc_sum[hc] += acc
        hc_n[hc] += 1

    out["overall_accuracy"] = _safe_div(total_acc, float(n))
    out["answer_type_accuracy"] = {k: _safe_div(at_sum[k], float(at_n[k])) for k in sorted(at_sum.keys())}
    out["question_type_accuracy"] = {k: _safe_div(qt_sum[k], float(qt_n[k])) for k in sorted(qt_sum.keys())}
    out["heuristic_category_accuracy"] = {k: _safe_div(hc_sum[k], float(hc_n[k])) for k in sorted(hc_sum.keys())}
    return out


def format_qualitative_samples(
    records: Sequence[Dict[str, Any]], n: int = 8, seed: int = 35, scorer: str = "official"
) -> List[Dict[str, Any]]:
    if n <= 0 or not records:
        return []
    idxs = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    picked = idxs[: int(n)]
    out: List[Dict[str, Any]] = []
    for i in picked:
        r = records[i]
        answers = _record_answers_for_scorer(r, scorer)
        gt = _majority_answer(answers, scorer)
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


def build_confusion_summary(
    records: Sequence[Dict[str, Any]], top_k: int = 20, short_max_words: int = 2, scorer: str = "official"
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    pair_counts = Counter()
    gt_counts = Counter()
    for r in records:
        answers = _record_answers_for_scorer(r, scorer)
        if not answers:
            continue
        gt = _majority_answer(answers, scorer)
        if scorer == "official":
            pr = normalize_vqa_official(str(r.get("prediction", "")))
        else:
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
    ap.add_argument("--scorer", type=str, default="official", choices=["official", "proxy"])
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

    summary = summarize_vqa_predictions(records, scorer=args.scorer)
    summary["qualitative"] = format_qualitative_samples(records, n=args.qualitative_samples, seed=args.seed, scorer=args.scorer)
    summary["confusions"] = build_confusion_summary(records, top_k=args.confusion_top_k, scorer=args.scorer)
    _print_summary(summary)

    if args.save_summary_json:
        with open(args.save_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
