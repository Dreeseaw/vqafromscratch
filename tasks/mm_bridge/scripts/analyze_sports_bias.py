#!/usr/bin/env python3
"""
Audit sports-topic skew in the lm_boom2 pretraining data and relate it to
`what sport is` VQA performance across representative mm_bridge runs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WIKI_JSONL = REPO_ROOT / "data/wiki_coco/articles.jsonl"
WIKI_CLEAN_STATS = REPO_ROOT / "data/pretraining/wikicoco256_cleaned/clean_stats.jsonl"
DISTILL_RAW = REPO_ROOT / "data/distill/q1/raw.jsonl"
VQA_VAL_QUESTIONS = REPO_ROOT / "data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_VAL_ANNOTATIONS = REPO_ROOT / "data/vqav2/v2_mscoco_val2014_annotations.json"


RUN_SPECS = [
    {
        "label": "original_vm",
        "desc": "Original frozen VM frontier (Nail)",
        "path": REPO_ROOT / "logs/mmnail_v1_lmmeanqquery_dynbudget_adapter_d3_cap64/logfile.txt",
    },
    {
        "label": "mobilevit",
        "desc": "MobileViT stage-1 comparable run",
        "path": REPO_ROOT / "logs/mmplank_v1_mobilevit_attnqquery_dynbudget_adapter_d3_cap64/logfile.txt",
    },
    {
        "label": "mobileclip",
        "desc": "MobileCLIP comparable run",
        "path": REPO_ROOT / "logs/mmcrane_v1_20260314_mobileclip_attnqquery_dynbudget_adapter_d3_cap64/logfile.txt",
    },
    {
        "label": "dinov2s_dyn",
        "desc": "DINOv2-S comparable run",
        "path": REPO_ROOT / "logs/mmcrane_v1_20260314_dinov2s_attnqquery_dynbudget_adapter_d3_cap64/logfile.txt",
    },
    {
        "label": "dinov2s_nodyn",
        "desc": "DINOv2-S frontier run",
        "path": REPO_ROOT / "logs/mmcrane_v1_20260314_dinov2s_attnqquery_nodynbudget_adapter_d3/logfile.txt",
    },
]


TITLE_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = [
    ("super_bowl", re.compile(r"\bsuper bowl\b", re.I)),
    ("world_series", re.compile(r"\bworld series\b", re.I)),
    ("formula_one", re.compile(r"\b(formula one|formula 1)\b", re.I)),
    ("grand_prix", re.compile(r"\bgrand prix\b", re.I)),
    ("nascar", re.compile(r"\bnascar\b", re.I)),
    ("indycar", re.compile(r"\bindycar\b", re.I)),
    ("motogp", re.compile(r"\bmotogp\b", re.I)),
    ("nfl", re.compile(r"\b(nfl|national football league|american football)\b", re.I)),
    ("mlb", re.compile(r"\b(mlb|major league baseball)\b", re.I)),
    ("nba", re.compile(r"\b(nba|national basketball association)\b", re.I)),
    ("nhl", re.compile(r"\b(nhl|national hockey league)\b", re.I)),
    ("premier_league", re.compile(r"\bpremier league\b", re.I)),
    ("uefa_fifa", re.compile(r"\b(uefa|fifa|world cup|champions league|euros?)\b", re.I)),
    ("olympics", re.compile(r"\b(olympic|olympics)\b", re.I)),
    ("baseball", re.compile(r"\bbaseball\b", re.I)),
    ("basketball", re.compile(r"\bbasketball\b", re.I)),
    ("american_football", re.compile(r"\bfootball\b", re.I)),
    ("soccer", re.compile(r"\b(soccer|association football)\b", re.I)),
    ("hockey", re.compile(r"\b(hockey|stanley cup)\b", re.I)),
    ("tennis", re.compile(r"\b(tennis|wimbledon|atp|wta)\b", re.I)),
    ("golf", re.compile(r"\bgolf\b", re.I)),
    ("cricket", re.compile(r"\bcricket\b", re.I)),
    ("rugby", re.compile(r"\brugby\b", re.I)),
    ("boxing_mma", re.compile(r"\b(mma|ufc|boxing)\b", re.I)),
    ("volleyball", re.compile(r"\bvolleyball\b", re.I)),
    ("badminton", re.compile(r"\bbadminton\b", re.I)),
    ("ski_snow", re.compile(r"\b(skiing|snowboarding)\b", re.I)),
    ("surf_skate", re.compile(r"\b(surfing|skateboarding)\b", re.I)),
    ("cycling", re.compile(r"\b(cycling|tour de france)\b", re.I)),
    ("horse_racing", re.compile(r"\b(horse racing|kentucky derby)\b", re.I)),
]
TEXT_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = [
    ("super_bowl", re.compile(r"\bsuper bowl\b", re.I)),
    ("world_series", re.compile(r"\bworld series\b", re.I)),
    ("formula_one", re.compile(r"\b(formula one|formula 1)\b", re.I)),
    ("grand_prix", re.compile(r"\bgrand prix\b", re.I)),
    ("nascar", re.compile(r"\bnascar\b", re.I)),
    ("indycar", re.compile(r"\bindycar\b", re.I)),
    ("motogp", re.compile(r"\bmotogp\b", re.I)),
    ("nfl", re.compile(r"\b(nfl|national football league|american football)\b", re.I)),
    ("mlb", re.compile(r"\b(mlb|major league baseball)\b", re.I)),
    ("nba", re.compile(r"\b(nba|national basketball association)\b", re.I)),
    ("nhl", re.compile(r"\b(nhl|national hockey league)\b", re.I)),
    ("premier_league", re.compile(r"\bpremier league\b", re.I)),
    ("uefa_fifa", re.compile(r"\b(uefa|fifa|world cup|champions league|euros?)\b", re.I)),
    ("olympics", re.compile(r"\b(olympic games|summer olympics|winter olympics)\b", re.I)),
    ("baseball", re.compile(r"\bbaseball\b", re.I)),
    ("basketball", re.compile(r"\bbasketball\b", re.I)),
    ("soccer", re.compile(r"\b(soccer|association football)\b", re.I)),
    ("hockey", re.compile(r"\b(hockey|stanley cup)\b", re.I)),
    ("tennis", re.compile(r"\b(tennis|wimbledon|atp|wta)\b", re.I)),
    ("golf", re.compile(r"\bgolf\b", re.I)),
    ("cricket", re.compile(r"\bcricket\b", re.I)),
    ("rugby", re.compile(r"\brugby\b", re.I)),
    ("boxing_mma", re.compile(r"\b(mma|ufc)\b", re.I)),
    ("volleyball", re.compile(r"\bvolleyball\b", re.I)),
    ("badminton", re.compile(r"\bbadminton\b", re.I)),
    ("ski_snow", re.compile(r"\b(skiing|snowboarding)\b", re.I)),
    ("surf_skate", re.compile(r"\b(surfing|skateboarding)\b", re.I)),
    ("cycling", re.compile(r"\b(cycling|tour de france)\b", re.I)),
    ("horse_racing", re.compile(r"\b(horse racing|kentucky derby)\b", re.I)),
]
TITLE_SEASON_RE = re.compile(
    r"\b(19|20)\d{2}\b.*\b(season|grand prix|super bowl|world series|cup|open|championship|playoffs?)\b",
    re.I,
)
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "let's": "let's",
    "mightve": "might've",
    "mustve": "must've",
    "shant": "shan't",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "wheres": "where's",
    "wholl": "who'll",
    "whos": "who's",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
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
_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit sports skew in LM pretraining and VQA bridge evals.")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional path for machine-readable JSON output.")
    return ap.parse_args()


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _norm_doc_id(doc_id: str) -> str:
    out = str(doc_id).strip()
    if out.startswith("./"):
        out = out[2:]
    return out


def _wiki_doc_id(line_no: int, row_id: str) -> str:
    return _norm_doc_id(f"data/wiki_coco/articles.jsonl::line:{line_no}::{row_id}")


def normalize_vqa_official(text: str) -> str:
    out = str(text or "").replace("\n", " ").replace("\t", " ").strip().lower()
    for p in _PUNCT:
        if (p + " " in out) or (" " + p in out) or (_COMMA_STRIP.search(out) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out)
    words = []
    for w in out.split():
        w = _MANUAL_MAP.get(w, w)
        if w not in _ARTICLES:
            words.append(_CONTRACTIONS.get(w, w))
    return _WS_RE.sub(" ", " ".join(words)).strip()


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


def classify_sports(title: str, text: str) -> Tuple[bool, bool, List[str], bool]:
    intro = text[:2400]
    title_labels = sorted({label for (label, pattern) in TITLE_PATTERNS if pattern.search(title)})
    intro_counts = Counter()
    for label, pattern in TEXT_PATTERNS:
        intro_counts[label] += len(pattern.findall(intro))
    intro_labels = sorted(label for label, count in intro_counts.items() if count > 0)
    intro_hit_total = int(sum(intro_counts.values()))
    conservative = bool(title_labels)
    broad = conservative or intro_hit_total >= 3 or len(intro_labels) >= 2 or any(count >= 2 for count in intro_counts.values())
    labels = title_labels if conservative else intro_labels if broad else []
    season_like = bool((conservative or broad) and TITLE_SEASON_RE.search(title))
    return conservative, broad, labels, season_like


def load_clean_stats(path: Path) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for row in _iter_jsonl(path):
        stats[_norm_doc_id(str(row.get("doc_id", "")))] = row
    return stats


def load_wiki_audit() -> Tuple[dict, Dict[str, dict]]:
    clean_stats = load_clean_stats(WIKI_CLEAN_STATS)
    totals = Counter()
    sports_conservative = Counter()
    sports_broad = Counter()
    keyword_docs = Counter()
    keyword_windows = Counter()
    examples: List[dict] = []
    doc_map: Dict[str, dict] = {}

    with WIKI_JSONL.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            doc_id = _wiki_doc_id(line_no, str(row["id"]))
            title = str(row.get("title", ""))
            text = str(row.get("text", ""))
            word_count = int(row.get("word_count", len(text.split())))
            clean = clean_stats.get(doc_id, {})
            sampled_windows = int(clean.get("num_sampled_windows", 0))
            cleaned_chars = int(clean.get("cleaned_chars", len(text)))
            is_sports_conservative, is_sports_broad, labels, season_like = classify_sports(title, text)

            totals["docs"] += 1
            totals["word_count"] += word_count
            totals["cleaned_chars"] += cleaned_chars
            totals["sampled_windows"] += sampled_windows

            meta = {
                "title": title,
                "word_count": word_count,
                "cleaned_chars": cleaned_chars,
                "sampled_windows": sampled_windows,
                "is_sports_conservative": is_sports_conservative,
                "is_sports_broad": is_sports_broad,
                "labels": labels,
                "season_like": season_like,
            }
            doc_map[doc_id] = meta

            if is_sports_broad:
                sports_broad["docs"] += 1
                sports_broad["word_count"] += word_count
                sports_broad["cleaned_chars"] += cleaned_chars
                sports_broad["sampled_windows"] += sampled_windows
                if season_like:
                    sports_broad["season_docs"] += 1
                    sports_broad["season_windows"] += sampled_windows
                    sports_broad["season_chars"] += cleaned_chars
            if is_sports_conservative:
                sports_conservative["docs"] += 1
                sports_conservative["word_count"] += word_count
                sports_conservative["cleaned_chars"] += cleaned_chars
                sports_conservative["sampled_windows"] += sampled_windows
                if season_like:
                    sports_conservative["season_docs"] += 1
                    sports_conservative["season_windows"] += sampled_windows
                    sports_conservative["season_chars"] += cleaned_chars
                for label in labels:
                    keyword_docs[label] += 1
                    keyword_windows[label] += sampled_windows
                examples.append(
                    {
                        "title": title,
                        "labels": labels,
                        "sampled_windows": sampled_windows,
                        "cleaned_chars": cleaned_chars,
                        "season_like": season_like,
                    }
                )

    examples.sort(key=lambda x: (x["sampled_windows"], x["cleaned_chars"]), reverse=True)
    return {
        "totals": dict(totals),
        "sports_conservative": dict(sports_conservative),
        "sports_broad": dict(sports_broad),
        "keyword_docs": keyword_docs.most_common(15),
        "keyword_windows": keyword_windows.most_common(15),
        "top_examples": examples[:20],
    }, doc_map


def load_distill_audit(doc_map: Dict[str, dict]) -> dict:
    if not DISTILL_RAW.exists():
        return {"available": False}

    totals = Counter()
    sports = Counter()
    question_prefix = Counter()
    for row in _iter_jsonl(DISTILL_RAW):
        totals["examples"] += 1
        source_doc_id = _norm_doc_id(str(row.get("source_doc_id", "")))
        src = doc_map.get(source_doc_id)
        if src and src.get("is_sports_conservative"):
            sports["examples_conservative"] += 1
            if src.get("season_like"):
                sports["season_examples_conservative"] += 1
        if src and src.get("is_sports_broad"):
            sports["examples_broad"] += 1
            if src.get("season_like"):
                sports["season_examples_broad"] += 1
        q = str(row.get("question", "")).strip().lower()
        if q:
            prefix = " ".join(q.split()[:3])
            question_prefix[prefix] += 1
            if src and src.get("is_sports_conservative"):
                question_prefix[f"sports::{prefix}"] += 1

    out = {
        "available": True,
        "total_examples": int(totals["examples"]),
        "sports_examples_conservative": int(sports["examples_conservative"]),
        "season_examples_conservative": int(sports["season_examples_conservative"]),
        "sports_share_conservative": _safe_div(sports["examples_conservative"], totals["examples"]),
        "season_share_conservative": _safe_div(sports["season_examples_conservative"], totals["examples"]),
        "sports_examples_broad": int(sports["examples_broad"]),
        "season_examples_broad": int(sports["season_examples_broad"]),
        "sports_share_broad": _safe_div(sports["examples_broad"], totals["examples"]),
        "season_share_broad": _safe_div(sports["season_examples_broad"], totals["examples"]),
    }
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_vqa_val_records() -> List[dict]:
    with VQA_VAL_QUESTIONS.open("r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with VQA_VAL_ANNOTATIONS.open("r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    q_by_id = {int(q["question_id"]): q for q in questions}
    records: List[dict] = []
    for ann in annotations:
        qid = int(ann["question_id"])
        q = q_by_id[qid]
        answers_raw = [str(a.get("answer", "")) for a in ann.get("answers", [])]
        records.append(
            {
                "question_id": qid,
                "question": str(q["question"]),
                "question_type": str(ann.get("question_type", "")).strip().lower(),
                "answer_type": str(ann.get("answer_type", "")),
                "multiple_choice_answer": str(ann.get("multiple_choice_answer", "")),
                "all_answers_raw": answers_raw,
            }
        )
    return records


def compute_vqa_qtype_profile(records: Sequence[dict], qtypes: Sequence[str]) -> dict:
    by_qtype: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_qtype[r["question_type"]].append(r)

    support_ranking = sorted(((qt, len(rows)) for qt, rows in by_qtype.items()), key=lambda x: x[1], reverse=True)
    support_rank = {qt: idx + 1 for idx, (qt, _) in enumerate(support_ranking)}

    profiles = {}
    for qtype in qtypes:
        rows = by_qtype.get(qtype, [])
        ans_counter = Counter(r["multiple_choice_answer"] for r in rows)
        top_answer, top_count = ("", 0)
        if ans_counter:
            top_answer, top_count = ans_counter.most_common(1)[0]
        official_const = 0.0
        for r in rows:
            official_const += vqa_official_accuracy(top_answer, r["all_answers_raw"])
        profiles[qtype] = {
            "support": len(rows),
            "support_rank": support_rank.get(qtype),
            "num_unique_answers": len(ans_counter),
            "top_answers": ans_counter.most_common(8),
            "majority_answer": top_answer,
            "majority_answer_share": _safe_div(top_count, len(rows)),
            "official_constant_baseline": _safe_div(official_const, len(rows)),
        }
    return {
        "num_val_questions": len(records),
        "support_top10": support_ranking[:10],
        "profiles": profiles,
    }


def _parse_log_qtypes(rendered: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in rendered.split("|"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        name, value = item.rsplit(":", 1)
        try:
            out[name.strip().lower()] = float(value.strip())
        except ValueError:
            continue
    return out


def audit_runs(vqa_records: Sequence[dict], qtypes: Sequence[str]) -> List[dict]:
    qtype_support = compute_vqa_qtype_profile(vqa_records, qtypes)["profiles"]
    out = []
    for spec in RUN_SPECS:
        path = spec["path"]
        if not path.exists():
            out.append({"label": spec["label"], "desc": spec["desc"], "available": False})
            continue
        overall: Optional[float] = None
        top_qtypes: Dict[str, float] = {}
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if "[eval:val] overall_accuracy=" in line:
                    m = re.search(r"overall_accuracy=([0-9.]+)", line)
                    if m:
                        overall = float(m.group(1))
                if "[eval:val] top question-type accuracy:" in line:
                    rendered = line.split("top question-type accuracy:", 1)[1].strip()
                    top_qtypes = _parse_log_qtypes(rendered)

        qtype_metrics = {}
        for qtype in qtypes:
            qtype_metrics[qtype] = {
                "accuracy": top_qtypes.get(qtype),
                "support": int(qtype_support[qtype]["support"]),
                "in_top10": qtype in top_qtypes,
            }
        out.append(
            {
                "label": spec["label"],
                "desc": spec["desc"],
                "available": True,
                "overall_accuracy": overall,
                "qtypes": qtype_metrics,
                "top_qtypes": top_qtypes,
            }
        )
    return out


def format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> int:
    args = parse_args()
    target_qtypes = ["what sport is", "what room is", "what animal is"]

    wiki_audit, doc_map = load_wiki_audit()
    distill_audit = load_distill_audit(doc_map)
    vqa_records = load_vqa_val_records()
    vqa_qtype = compute_vqa_qtype_profile(vqa_records, target_qtypes)
    run_audit = audit_runs(vqa_records, target_qtypes)

    totals = wiki_audit["totals"]
    sports_cons = wiki_audit["sports_conservative"]
    sports_broad = wiki_audit["sports_broad"]
    corpus_summary = {
        "sports_doc_share_conservative": _safe_div(sports_cons.get("docs", 0), totals.get("docs", 0)),
        "sports_word_share_conservative": _safe_div(sports_cons.get("word_count", 0), totals.get("word_count", 0)),
        "sports_cleaned_char_share_conservative": _safe_div(sports_cons.get("cleaned_chars", 0), totals.get("cleaned_chars", 0)),
        "sports_window_share_conservative": _safe_div(sports_cons.get("sampled_windows", 0), totals.get("sampled_windows", 0)),
        "sports_season_doc_share_conservative": _safe_div(sports_cons.get("season_docs", 0), totals.get("docs", 0)),
        "sports_doc_share_broad": _safe_div(sports_broad.get("docs", 0), totals.get("docs", 0)),
        "sports_word_share_broad": _safe_div(sports_broad.get("word_count", 0), totals.get("word_count", 0)),
        "sports_cleaned_char_share_broad": _safe_div(sports_broad.get("cleaned_chars", 0), totals.get("cleaned_chars", 0)),
        "sports_window_share_broad": _safe_div(sports_broad.get("sampled_windows", 0), totals.get("sampled_windows", 0)),
        "sports_season_doc_share_broad": _safe_div(sports_broad.get("season_docs", 0), totals.get("docs", 0)),
    }

    result = {
        "wiki_audit": wiki_audit,
        "distill_audit": distill_audit,
        "corpus_summary": corpus_summary,
        "vqa_qtype": vqa_qtype,
        "run_audit": run_audit,
    }

    print("Sports Bias Audit")
    print("=================")
    print()
    print("LM corpus")
    print(
        f"- Conservative sports docs (title-led): {sports_cons.get('docs', 0)}/{totals.get('docs', 0)} "
        f"({format_pct(corpus_summary['sports_doc_share_conservative'])})"
    )
    print(
        f"- Conservative sports words: {sports_cons.get('word_count', 0)}/{totals.get('word_count', 0)} "
        f"({format_pct(corpus_summary['sports_word_share_conservative'])})"
    )
    print(
        f"- Conservative sports cleaned chars: {sports_cons.get('cleaned_chars', 0)}/{totals.get('cleaned_chars', 0)} "
        f"({format_pct(corpus_summary['sports_cleaned_char_share_conservative'])})"
    )
    print(
        f"- Conservative sports sampled windows: {sports_cons.get('sampled_windows', 0)}/{totals.get('sampled_windows', 0)} "
        f"({format_pct(corpus_summary['sports_window_share_conservative'])})"
    )
    print(
        f"- Conservative sports docs that look season/event-history pages: {sports_cons.get('season_docs', 0)} "
        f"({format_pct(corpus_summary['sports_season_doc_share_conservative'])} of all wiki docs)"
    )
    print(
        f"- Broad sports docs (title or strong intro): {sports_broad.get('docs', 0)}/{totals.get('docs', 0)} "
        f"({format_pct(corpus_summary['sports_doc_share_broad'])})"
    )
    print(
        f"- Broad sports sampled windows: {sports_broad.get('sampled_windows', 0)}/{totals.get('sampled_windows', 0)} "
        f"({format_pct(corpus_summary['sports_window_share_broad'])})"
    )
    print(f"- Top sports keyword hits by docs: {wiki_audit['keyword_docs'][:8]}")
    print(f"- Top sports keyword hits by sampled windows: {wiki_audit['keyword_windows'][:8]}")
    print()
    print("Distill mix")
    if distill_audit.get("available"):
        print(
            f"- Distill examples from conservative sports source docs: "
            f"{distill_audit['sports_examples_conservative']}/{distill_audit['total_examples']} "
            f"({format_pct(distill_audit['sports_share_conservative'])})"
        )
        print(
            f"- Distill examples from conservative season/event-history sports docs: "
            f"{distill_audit['season_examples_conservative']}/{distill_audit['total_examples']} "
            f"({format_pct(distill_audit['season_share_conservative'])})"
        )
        print(
            f"- Distill examples from broad sports source docs: "
            f"{distill_audit['sports_examples_broad']}/{distill_audit['total_examples']} "
            f"({format_pct(distill_audit['sports_share_broad'])})"
        )
    else:
        print("- Distill raw file not available.")
    print()
    print("VQAv2 question-type profile")
    print(f"- Val questions: {vqa_qtype['num_val_questions']}")
    print(f"- Top question types by support: {vqa_qtype['support_top10']}")
    for qtype in target_qtypes:
        prof = vqa_qtype["profiles"][qtype]
        print(
            f"- {qtype!r}: support={prof['support']} rank={prof['support_rank']} "
            f"unique_answers={prof['num_unique_answers']} "
            f"majority='{prof['majority_answer']}' share={format_pct(prof['majority_answer_share'])} "
            f"const-baseline={format_pct(prof['official_constant_baseline'])}"
        )
        print(f"  top answers: {prof['top_answers'][:6]}")
    print()
    print("Representative run audit")
    for row in run_audit:
        if not row.get("available"):
            print(f"- {row['label']}: missing")
            continue
        sport = row["qtypes"]["what sport is"]
        room = row["qtypes"]["what room is"]
        sport_render = "not in final top-10"
        if sport["accuracy"] is not None:
            sport_render = f"{format_pct(sport['accuracy'])} on {sport['support']} qs"
        room_render = "not in final top-10"
        if room["accuracy"] is not None:
            room_render = f"{format_pct(room['accuracy'])} on {room['support']} qs"
        print(
            f"- {row['label']} ({row['desc']}): overall={format_pct(row['overall_accuracy'])}, "
            f"what sport is={sport_render}, "
            f"what room is={room_render}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON report to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
