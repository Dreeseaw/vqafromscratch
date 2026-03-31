from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from train.vqa_data import heuristic_question_category, normalize_vqa_answer


OCR_BUCKET_PATTERNS = {
    "sign_written": [
        "what does the sign",
        "what does this sign",
        "what is written",
        "written on the",
        "on the sign",
    ],
    "brand_logo": [
        "what brand",
        "what company",
        "what logo",
        "what store",
    ],
    "name_label": [
        "what is the name",
        "what name",
        "name of the",
        "what does the label",
    ],
    "word_text": [
        "what word",
        "what words",
        "what letter",
        "what number is",
        "what does the",
    ],
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def _budget_list(raw: str) -> List[int]:
    values: List[int] = []
    seen: set[int] = set()
    for part in str(raw or "").split(","):
        item = part.strip()
        if not item:
            continue
        value = max(1, int(item))
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return sorted(values)


def _ocr_bucket(question: str) -> str | None:
    q = str(question).strip().lower()
    for bucket, patterns in OCR_BUCKET_PATTERNS.items():
        if any(pattern in q for pattern in patterns):
            return bucket
    return None


def _question_length_bucket(question: str) -> str:
    words = [tok for tok in str(question).strip().split() if tok]
    n = len(words)
    if n <= 4:
        return "short_0_4"
    if n <= 8:
        return "medium_5_8"
    return "long_9_plus"


def _question_prefix(question: str, n: int = 2) -> str:
    toks = [tok for tok in normalize_vqa_answer(question).split() if tok]
    return " ".join(toks[:n]) if toks else ""


def _row_answer_type(row: Dict[str, Any]) -> str:
    raw = str(row.get("answer_type") or (row.get("metadata") or {}).get("answer_type") or "other").strip().lower()
    if raw in ("yes/no", "yesno", "yes_no", "yes-no"):
        return "yes/no"
    if raw in ("number", "num", "numeric", "numerical"):
        return "number"
    return "other"


def _row_question_type(row: Dict[str, Any]) -> str:
    raw = str(row.get("question_type") or (row.get("metadata") or {}).get("question_type") or "").strip().lower()
    if raw:
        return raw
    return heuristic_question_category(str(row.get("question", "")))


def _severity_scores(question: str, row: Dict[str, Any]) -> Dict[str, float]:
    qcat = heuristic_question_category(question)
    is_ocr_like = 1.0 if _ocr_bucket(question) is not None else 0.0
    is_count_like = 1.0 if qcat == "count" or normalize_vqa_answer(question).startswith("how many") else 0.0
    mean_selected_prob = float(row.get("mean_selected_prob", 0.0) or 0.0)
    mean_margin = float(row.get("mean_margin", 0.0) or 0.0)
    mean_entropy = float(row.get("mean_entropy", 0.0) or 0.0)
    return {
        "conf_mean": 1.0 - mean_selected_prob,
        "margin_mean": 1.0 - mean_margin,
        "entropy_mean": mean_entropy,
        "hybrid_conf_ocr_count": (1.0 - mean_selected_prob) + 0.20 * is_ocr_like + 0.10 * is_count_like,
    }


def _load_budget_maps(eval_dir: Path, budgets: Sequence[int]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[int, Dict[str, Any]]]]:
    meta_by_qid: Dict[int, Dict[str, Any]] = {}
    out: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for budget in budgets:
        budget_rows: Dict[int, Dict[str, Any]] = {}
        for row in _iter_jsonl(eval_dir / f"fixed_k{budget}_records.jsonl"):
            qid = int(row["question_id"])
            if qid not in meta_by_qid:
                meta_by_qid[qid] = {
                    "question": str(row.get("question", "")),
                    "answer_type": str(row.get("answer_type", "")),
                    "question_type": str(row.get("question_type", "")),
                }
            min_row: Dict[str, Any] = {
                "accuracy": float(row.get("accuracy", 0.0) or 0.0),
            }
            for key in ("mean_selected_prob", "mean_margin", "mean_entropy"):
                if key in row:
                    min_row[key] = float(row.get(key, 0.0) or 0.0)
            budget_rows[qid] = min_row
        out[int(budget)] = budget_rows
    return meta_by_qid, out


def _qid_split(qids: Sequence[int], *, tune_mod: int, tune_remainder: int) -> Dict[str, List[int]]:
    tune: List[int] = []
    report: List[int] = []
    for qid in qids:
        if int(qid) % int(tune_mod) == int(tune_remainder):
            tune.append(int(qid))
        else:
            report.append(int(qid))
    return {
        "tune": sorted(tune),
        "report": sorted(report),
        "full": sorted(int(qid) for qid in qids),
    }


def _evaluate_selected_budgets(
    qids: Sequence[int],
    *,
    meta_by_qid: Dict[int, Dict[str, Any]],
    budget_maps: Dict[int, Dict[int, Dict[str, Any]]],
    selected_budget_by_qid: Dict[int, int],
    base_budget: int,
    best_fixed_budget: int,
    oracle_budget_by_qid: Dict[int, int],
) -> Dict[str, Any]:
    total_acc = 0.0
    total_budget = 0.0
    escalated = 0
    by_answer_sum = defaultdict(float)
    by_answer_n = defaultdict(int)
    budget_hist = Counter()

    base_acc_sum = 0.0
    best_fixed_acc_sum = 0.0
    oracle_acc_sum = 0.0
    n = 0

    for qid in qids:
        budget = int(selected_budget_by_qid[int(qid)])
        row = budget_maps[budget][int(qid)]
        meta = meta_by_qid[int(qid)]
        acc = float(row["accuracy"])
        answer_type = _row_answer_type(meta)
        total_acc += acc
        total_budget += float(budget)
        budget_hist[budget] += 1
        by_answer_sum[answer_type] += acc
        by_answer_n[answer_type] += 1
        if int(budget) > int(base_budget):
            escalated += 1

        base_acc_sum += float(budget_maps[int(base_budget)][int(qid)]["accuracy"])
        best_fixed_acc_sum += float(budget_maps[int(best_fixed_budget)][int(qid)]["accuracy"])
        oracle_acc_sum += float(budget_maps[int(oracle_budget_by_qid[int(qid)])][int(qid)]["accuracy"])
        n += 1

    denom = float(max(1, n))
    overall = total_acc / denom
    base_acc = base_acc_sum / denom
    best_fixed_acc = best_fixed_acc_sum / denom
    oracle_acc = oracle_acc_sum / denom
    oracle_gap = oracle_acc - base_acc
    recovered = (overall - base_acc) / oracle_gap if oracle_gap > 1e-12 else 0.0
    return {
        "count": int(n),
        "overall_accuracy": float(overall),
        "answer_type_accuracy": {
            key: float(by_answer_sum[key] / float(max(1, by_answer_n[key])))
            for key in ("yes/no", "number", "other")
        },
        "average_budget": float(total_budget / denom),
        "percent_escalated": float(escalated / denom),
        "selected_budget_histogram": {str(key): int(val) for key, val in sorted(budget_hist.items())},
        "delta_vs_fixed_base": float(overall - base_acc),
        "delta_vs_best_fixed": float(overall - best_fixed_acc),
        "oracle_gap_recovered": float(recovered),
        "base_accuracy": float(base_acc),
        "best_fixed_accuracy": float(best_fixed_acc),
        "oracle_accuracy": float(oracle_acc),
    }


def _select_single_threshold(
    qids: Sequence[int],
    *,
    ranking: Sequence[int],
    base_budget: int,
    high_budget: int,
    pct: float,
) -> Dict[int, int]:
    n = len(qids)
    count = max(0, min(n, int(math.floor(float(pct) * float(n)))))
    escalate = set(int(qid) for qid in ranking[:count])
    return {int(qid): (int(high_budget) if int(qid) in escalate else int(base_budget)) for qid in qids}


def _select_two_threshold(
    qids: Sequence[int],
    *,
    ranking: Sequence[int],
    base_budget: int,
    mid_budget: int,
    high_budget: int,
    pct_mid_total: float,
    pct_high: float,
) -> Dict[int, int]:
    n = len(qids)
    high_count = max(0, min(n, int(math.floor(float(pct_high) * float(n)))))
    mid_total_count = max(high_count, min(n, int(math.floor(float(pct_mid_total) * float(n)))))
    high_set = set(int(qid) for qid in ranking[:high_count])
    mid_set = set(int(qid) for qid in ranking[:mid_total_count]) - high_set
    out: Dict[int, int] = {}
    for qid in qids:
        iqid = int(qid)
        if iqid in high_set:
            out[iqid] = int(high_budget)
        elif iqid in mid_set:
            out[iqid] = int(mid_budget)
        else:
            out[iqid] = int(base_budget)
    return out


def _tune_frontier(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[split]["average_budget"]),
            -float(row[split]["overall_accuracy"]),
            row["policy_id"],
        ),
    )
    best_acc = -1.0
    frontier: List[Dict[str, Any]] = []
    for row in ordered:
        acc = float(row[split]["overall_accuracy"])
        if acc > best_acc + 1e-12:
            frontier.append(row)
            best_acc = acc
    return frontier


def _pick_recommended(rows: List[Dict[str, Any]], *, split: str, max_average_budget: float) -> Dict[str, Any] | None:
    eligible = [row for row in rows if float(row[split]["average_budget"]) <= float(max_average_budget) + 1e-12]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            -float(row[split]["overall_accuracy"]),
            float(row[split]["average_budget"]),
            row["policy_id"],
        )
    )
    return eligible[0]


def _oracle_subset_summary(
    qids: Sequence[int],
    *,
    meta_by_qid: Dict[int, Dict[str, Any]],
    budget_maps: Dict[int, Dict[int, Dict[str, Any]]],
    subset_budgets: Sequence[int],
) -> Dict[str, Any]:
    selected_budget_by_qid: Dict[int, int] = {}
    for qid in qids:
        best_acc = max(float(budget_maps[int(budget)][int(qid)]["accuracy"]) for budget in subset_budgets)
        best_budget = min(
            int(budget)
            for budget in subset_budgets
            if float(budget_maps[int(budget)][int(qid)]["accuracy"]) == best_acc
        )
        selected_budget_by_qid[int(qid)] = int(best_budget)
    return _evaluate_selected_budgets(
        qids,
        meta_by_qid=meta_by_qid,
        budget_maps=budget_maps,
        selected_budget_by_qid=selected_budget_by_qid,
        base_budget=min(int(b) for b in subset_budgets),
        best_fixed_budget=min(int(b) for b in subset_budgets),
        oracle_budget_by_qid=selected_budget_by_qid,
    )


def _tail_breakdown(
    qids: Sequence[int],
    *,
    meta_by_qid: Dict[int, Dict[str, Any]],
    budget_maps: Dict[int, Dict[int, Dict[str, Any]]],
    selected_budget_by_qid: Dict[int, int],
    base_budget: int,
) -> Dict[str, Any]:
    tail_qids = [int(qid) for qid in qids if int(selected_budget_by_qid[int(qid)]) > int(base_budget)]
    budget_hist = Counter()
    answer_hist = Counter()
    qtype_hist = Counter()
    qlen_hist = Counter()
    prefix_hist = Counter()
    ocr_hist = Counter()
    improve = 0
    worsen = 0

    for qid in tail_qids:
        base_row = budget_maps[int(base_budget)][int(qid)]
        chosen_budget = int(selected_budget_by_qid[int(qid)])
        chosen_row = budget_maps[chosen_budget][int(qid)]
        meta = meta_by_qid[int(qid)]
        question = str(meta.get("question", ""))
        budget_hist[chosen_budget] += 1
        answer_hist[_row_answer_type(meta)] += 1
        qtype_hist[_row_question_type(meta)] += 1
        qlen_hist[_question_length_bucket(question)] += 1
        prefix = _question_prefix(question, n=2)
        if prefix:
            prefix_hist[prefix] += 1
        bucket = _ocr_bucket(question)
        ocr_hist["ocr_like" if bucket is not None else "non_ocr"] += 1
        base_acc = float(base_row["accuracy"])
        chosen_acc = float(chosen_row["accuracy"])
        if chosen_acc > base_acc:
            improve += 1
        elif chosen_acc < base_acc:
            worsen += 1

    total = float(max(1, len(tail_qids)))
    return {
        "count": int(len(tail_qids)),
        "fraction": float(len(tail_qids) / float(max(1, len(qids)))),
        "selected_budget_histogram": {str(k): int(v) for k, v in sorted(budget_hist.items())},
        "answer_type_fraction": {str(k): float(v) / total for k, v in sorted(answer_hist.items())},
        "ocr_fraction": {str(k): float(v) / total for k, v in sorted(ocr_hist.items())},
        "question_length_fraction": {str(k): float(v) / total for k, v in sorted(qlen_hist.items())},
        "top_question_types": [{"question_type": str(k), "count": int(v)} for k, v in qtype_hist.most_common(10)],
        "top_question_prefixes": [{"prefix": str(k), "count": int(v)} for k, v in prefix_hist.most_common(12)],
        "improve_fraction_vs_base": float(improve / total),
        "worsen_fraction_vs_base": float(worsen / total),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep simple post-hoc semantic-budget schedulers from saved eval artifacts.")
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--main_base_budget", type=int, default=4)
    ap.add_argument("--side_base_budget", type=int, default=2)
    ap.add_argument("--tune_mod", type=int, default=5)
    ap.add_argument("--tune_remainder", type=int, default=0)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(os.path.abspath(args.eval_dir))
    summary = _read_json(eval_dir / "summary.json")
    oracle_budgets = [int(x) for x in summary.get("oracle_budgets", [])]
    all_budgets = sorted(set(oracle_budgets))
    meta_by_qid, budget_maps = _load_budget_maps(eval_dir, all_budgets)
    oracle_budget_by_qid = {
        int(row["question_id"]): int(row["oracle_selected_budget"])
        for row in _iter_jsonl(eval_dir / "oracle_records.jsonl")
    }
    qids = sorted(int(qid) for qid in oracle_budget_by_qid.keys())
    split_qids = _qid_split(qids, tune_mod=int(args.tune_mod), tune_remainder=int(args.tune_remainder))

    best_fixed_budget = int((summary.get("best_fixed") or {}).get("budget", int(args.main_base_budget)))
    main_base_budget = int(args.main_base_budget)
    side_base_budget = int(args.side_base_budget)
    if main_base_budget not in budget_maps:
        raise RuntimeError(f"Missing base budget K={main_base_budget} in {eval_dir}")
    if side_base_budget not in budget_maps:
        raise RuntimeError(f"Missing side base budget K={side_base_budget} in {eval_dir}")

    main_signals = ("conf_mean", "margin_mean", "entropy_mean", "hybrid_conf_ocr_count")
    side_signals = ("conf_mean", "margin_mean", "entropy_mean", "hybrid_conf_ocr_count")
    pct_grid = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
    pct_grid_2stage_high = (0.0, 0.005, 0.01, 0.02, 0.05)
    pct_grid_2stage_total = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)

    severity_by_budget_split: Dict[tuple[int, str, str], Dict[int, float]] = {}
    ranking_by_budget_split: Dict[tuple[int, str, str], List[int]] = {}
    for base_budget, signal_names in ((main_base_budget, main_signals), (side_base_budget, side_signals)):
        for split_name, split_ids in split_qids.items():
            for signal_name in signal_names:
                scores = {
                    int(qid): float(
                        _severity_scores(
                            str(meta_by_qid[int(qid)].get("question", "")),
                            budget_maps[int(base_budget)][int(qid)],
                        )[signal_name]
                    )
                    for qid in split_ids
                }
                severity_by_budget_split[(int(base_budget), split_name, signal_name)] = scores
                ranking_by_budget_split[(int(base_budget), split_name, signal_name)] = sorted(
                    split_ids,
                    key=lambda qid: (-scores[int(qid)], int(qid)),
                )

    def build_selected_by_split(
        *,
        family: str,
        signal_name: str,
        base_budget: int,
        params: Dict[str, float],
    ) -> Dict[str, Dict[int, int]]:
        selected_by_split: Dict[str, Dict[int, int]] = {}
        for split_name, split_ids in split_qids.items():
            ranking = ranking_by_budget_split[(int(base_budget), split_name, signal_name)]
            if family in ("k4_to8", "k4_to16", "k2_to4", "k2_to8"):
                selected_by_split[split_name] = _select_single_threshold(
                    split_ids,
                    ranking=ranking,
                    base_budget=int(base_budget),
                    high_budget=int(params["high_budget"]),
                    pct=float(params["pct"]),
                )
            else:
                selected_by_split[split_name] = _select_two_threshold(
                    split_ids,
                    ranking=ranking,
                    base_budget=int(base_budget),
                    mid_budget=int(params["mid_budget"]),
                    high_budget=int(params["high_budget"]),
                    pct_mid_total=float(params["pct_mid_total"]),
                    pct_high=float(params["pct_high"]),
                )
        return selected_by_split

    def eval_policy(
        policy_id: str,
        family: str,
        signal_name: str,
        base_budget: int,
        params: Dict[str, float],
        selected_by_split: Dict[str, Dict[int, int]],
    ) -> Dict[str, Any]:
        sections: Dict[str, Any] = {}
        for split_name, split_ids in split_qids.items():
            sections[split_name] = _evaluate_selected_budgets(
                split_ids,
                meta_by_qid=meta_by_qid,
                budget_maps=budget_maps,
                selected_budget_by_qid=selected_by_split[split_name],
                base_budget=int(base_budget),
                best_fixed_budget=int(best_fixed_budget),
                oracle_budget_by_qid=oracle_budget_by_qid,
            )
        return {
            "policy_id": str(policy_id),
            "family": str(family),
            "signal": str(signal_name),
            "base_budget": int(base_budget),
            "params": dict(params),
            "tune": sections["tune"],
            "report": sections["report"],
            "full": sections["full"],
        }

    main_rows: List[Dict[str, Any]] = []
    has_main_mid = 8 in budget_maps
    has_main_high = 16 in budget_maps
    for signal_name in main_signals:
        if has_main_mid:
            for pct in pct_grid:
                params = {"high_budget": 8.0, "pct": float(pct)}
                selected_by_split = build_selected_by_split(
                    family="k4_to8",
                    signal_name=signal_name,
                    base_budget=main_base_budget,
                    params=params,
                )
                main_rows.append(
                    eval_policy(f"main:k4_to8:{signal_name}:p{pct:.3f}", "k4_to8", signal_name, main_base_budget, params, selected_by_split)
                )
        if has_main_high:
            for pct in pct_grid:
                params = {"high_budget": 16.0, "pct": float(pct)}
                selected_by_split = build_selected_by_split(
                    family="k4_to16",
                    signal_name=signal_name,
                    base_budget=main_base_budget,
                    params=params,
                )
                main_rows.append(
                    eval_policy(f"main:k4_to16:{signal_name}:p{pct:.3f}", "k4_to16", signal_name, main_base_budget, params, selected_by_split)
                )
        if has_main_mid and has_main_high:
            for pct_high in pct_grid_2stage_high:
                for pct_total in pct_grid_2stage_total:
                    if float(pct_total) < float(pct_high):
                        continue
                    params = {"mid_budget": 8.0, "high_budget": 16.0, "pct_mid_total": float(pct_total), "pct_high": float(pct_high)}
                    selected_by_split = build_selected_by_split(
                        family="k4_to8_to16",
                        signal_name=signal_name,
                        base_budget=main_base_budget,
                        params=params,
                    )
                    main_rows.append(
                        eval_policy(
                            f"main:k4_to8_to16:{signal_name}:p8{pct_total:.3f}:p16{pct_high:.3f}",
                            "k4_to8_to16",
                            signal_name,
                            main_base_budget,
                            params,
                            selected_by_split,
                        )
                    )

    side_rows: List[Dict[str, Any]] = []
    has_side_mid = 4 in budget_maps
    has_side_high = 8 in budget_maps
    for signal_name in side_signals:
        if has_side_mid:
            for pct in pct_grid:
                params = {"high_budget": 4.0, "pct": float(pct)}
                selected_by_split = build_selected_by_split(
                    family="k2_to4",
                    signal_name=signal_name,
                    base_budget=side_base_budget,
                    params=params,
                )
                side_rows.append(
                    eval_policy(f"side:k2_to4:{signal_name}:p{pct:.3f}", "k2_to4", signal_name, side_base_budget, params, selected_by_split)
                )
        if has_side_high:
            for pct in pct_grid:
                params = {"high_budget": 8.0, "pct": float(pct)}
                selected_by_split = build_selected_by_split(
                    family="k2_to8",
                    signal_name=signal_name,
                    base_budget=side_base_budget,
                    params=params,
                )
                side_rows.append(
                    eval_policy(f"side:k2_to8:{signal_name}:p{pct:.3f}", "k2_to8", signal_name, side_base_budget, params, selected_by_split)
                )
        if has_side_mid and has_side_high:
            for pct_high in pct_grid_2stage_high:
                for pct_total in pct_grid_2stage_total:
                    if float(pct_total) < float(pct_high):
                        continue
                    params = {"mid_budget": 4.0, "high_budget": 8.0, "pct_mid_total": float(pct_total), "pct_high": float(pct_high)}
                    selected_by_split = build_selected_by_split(
                        family="k2_to4_to8",
                        signal_name=signal_name,
                        base_budget=side_base_budget,
                        params=params,
                    )
                    side_rows.append(
                        eval_policy(
                            f"side:k2_to4_to8:{signal_name}:p4{pct_total:.3f}:p8{pct_high:.3f}",
                            "k2_to4_to8",
                            signal_name,
                            side_base_budget,
                            params,
                            selected_by_split,
                        )
                    )

    main_frontier_tune = _tune_frontier(main_rows, "tune")
    side_frontier_tune = _tune_frontier(side_rows, "tune")
    recommended_main = _pick_recommended(main_frontier_tune, split="tune", max_average_budget=4.5)
    recommended_side = _pick_recommended(side_frontier_tune, split="tune", max_average_budget=2.4)

    oracle_subset_rows: Dict[str, Any] = {}
    for label, subset in (
        ("oracle_4_8", [4, 8]),
        ("oracle_4_8_16", [4, 8, 16]),
        ("oracle_2_4", [2, 4]),
        ("oracle_2_4_8", [2, 4, 8]),
    ):
        if all(int(budget) in budget_maps for budget in subset):
            oracle_subset_rows[label] = _oracle_subset_summary(
                split_qids["full"],
                meta_by_qid=meta_by_qid,
                budget_maps=budget_maps,
                subset_budgets=subset,
            )

    recommended_main_tail = None
    if recommended_main is not None:
        full_selected = build_selected_by_split(
            family=str(recommended_main["family"]),
            signal_name=str(recommended_main["signal"]),
            base_budget=int(recommended_main["base_budget"]),
            params=dict(recommended_main["params"]),
        )["full"]
        recommended_main_tail = {
            "policy_id": str(recommended_main["policy_id"]),
            "tail_breakdown": _tail_breakdown(
                split_qids["full"],
                meta_by_qid=meta_by_qid,
                budget_maps=budget_maps,
                selected_budget_by_qid=full_selected,
                base_budget=main_base_budget,
            ),
        }

    recommended_side_tail = None
    if recommended_side is not None:
        full_selected = build_selected_by_split(
            family=str(recommended_side["family"]),
            signal_name=str(recommended_side["signal"]),
            base_budget=int(recommended_side["base_budget"]),
            params=dict(recommended_side["params"]),
        )["full"]
        recommended_side_tail = {
            "policy_id": str(recommended_side["policy_id"]),
            "tail_breakdown": _tail_breakdown(
                split_qids["full"],
                meta_by_qid=meta_by_qid,
                budget_maps=budget_maps,
                selected_budget_by_qid=full_selected,
                base_budget=side_base_budget,
            ),
        }

    oracle_tail_main = {
        "tail_breakdown": _tail_breakdown(
            split_qids["full"],
            meta_by_qid=meta_by_qid,
            budget_maps=budget_maps,
            selected_budget_by_qid=oracle_budget_by_qid,
            base_budget=main_base_budget,
        )
    }
    oracle_tail_side = {
        "tail_breakdown": _tail_breakdown(
            split_qids["full"],
            meta_by_qid=meta_by_qid,
            budget_maps=budget_maps,
            selected_budget_by_qid=oracle_budget_by_qid,
            base_budget=side_base_budget,
        )
    }

    out = {
        "eval_dir": str(eval_dir),
        "checkpoint": str(summary.get("checkpoint", "")),
        "tuning_protocol": {
            "rule": f"qid % {int(args.tune_mod)} == {int(args.tune_remainder)} => tune, else report",
            "tune_count": int(len(split_qids["tune"])),
            "report_count": int(len(split_qids["report"])),
            "full_count": int(len(split_qids["full"])),
            "note": "Candidate thresholds are fixed percentile grids. Report-split tables are disjoint from the tuning slice.",
        },
        "best_fixed_budget": int(best_fixed_budget),
        "main_policies": {
            "base_budget": int(main_base_budget),
            "candidates": main_rows,
            "tune_frontier": main_frontier_tune,
            "recommended": recommended_main,
        },
        "side_probe_policies": {
            "base_budget": int(side_base_budget),
            "candidates": side_rows,
            "tune_frontier": side_frontier_tune,
            "recommended": recommended_side,
        },
        "oracle_subset_upper_bounds": oracle_subset_rows,
        "tail_analysis": {
            "oracle_base4": oracle_tail_main,
            "oracle_base2": oracle_tail_side,
            "recommended_main": recommended_main_tail,
            "recommended_side": recommended_side_tail,
        },
    }

    output_path = Path(os.path.abspath(args.output_json))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")

    if recommended_main is not None:
        report = recommended_main["report"]
        print(
            "[scheduler:main] recommended "
            f"policy={recommended_main['policy_id']} "
            f"report_acc={float(report['overall_accuracy']):.4f} "
            f"report_avg_k={float(report['average_budget']):.4f} "
            f"report_gap_recovered={float(report['oracle_gap_recovered']):.4f}",
            flush=True,
        )
    if recommended_side is not None:
        report = recommended_side["report"]
        print(
            "[scheduler:side] recommended "
            f"policy={recommended_side['policy_id']} "
            f"report_acc={float(report['overall_accuracy']):.4f} "
            f"report_avg_k={float(report['average_budget']):.4f} "
            f"report_gap_recovered={float(report['oracle_gap_recovered']):.4f}",
            flush=True,
        )
    print(f"[scheduler] wrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
