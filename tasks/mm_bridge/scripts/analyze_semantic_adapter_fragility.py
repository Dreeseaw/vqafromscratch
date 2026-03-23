from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize semantic-compression adapter fragility results.")
    ap.add_argument("--anchor_ablation", type=str, required=True)
    ap.add_argument("--k32_ablation", type=str, required=True)
    ap.add_argument("--k8_ablation", type=str, required=True)
    ap.add_argument("--k8_probe", type=str, required=True)
    ap.add_argument("--gqa_results", type=str, action="append", default=[])
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, required=True)
    return ap.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ablation_map(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in payload.get("results", []):
        out[int(row.get("keep_count", 0))] = dict(row)
    return out


def _safe_get_category(row: Dict[str, Any], key: str) -> float:
    raw = dict(row.get("answer_type_accuracy", {}) or {})
    return float(raw.get(key, 0.0))


def _format(v: float) -> str:
    return f"{float(v):.4f}"


def _gqa_table(gqa_entries: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, payload in gqa_entries:
        row_map = _ablation_map(payload)
        keep3 = float(row_map.get(3, {}).get("overall_accuracy", 0.0))
        keep0 = float(row_map.get(0, {}).get("overall_accuracy", 0.0))
        rows.append(
            {
                "label": label,
                "keep3": keep3,
                "keep0": keep0,
                "delta_3_to_0": keep3 - keep0,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    anchor = _load_json(args.anchor_ablation)
    k32 = _load_json(args.k32_ablation)
    k8 = _load_json(args.k8_ablation)
    k8_probe = _load_json(args.k8_probe)

    rows_by_model = {
        "anchor": _ablation_map(anchor),
        "k32": _ablation_map(k32),
        "k8": _ablation_map(k8),
    }
    categories = ["yes/no", "number", "other"]

    primary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, row_map in rows_by_model.items():
        primary[model_name] = {}
        for category in categories:
            keep3 = _safe_get_category(row_map.get(3, {}), category)
            keep2 = _safe_get_category(row_map.get(2, {}), category)
            keep1 = _safe_get_category(row_map.get(1, {}), category)
            keep0 = _safe_get_category(row_map.get(0, {}), category)
            primary[model_name][category] = {
                "keep3": keep3,
                "keep2": keep2,
                "keep1": keep1,
                "keep0": keep0,
                "delta_3_to_0": keep3 - keep0,
            }

    fragility_ratios: Dict[str, Dict[str, float]] = {"k32_vs_anchor": {}, "k8_vs_anchor": {}}
    for category in categories:
        anchor_delta = primary["anchor"][category]["delta_3_to_0"]
        denom = anchor_delta if abs(anchor_delta) > 1e-12 else 1e-12
        fragility_ratios["k32_vs_anchor"][category] = primary["k32"][category]["delta_3_to_0"] / denom
        fragility_ratios["k8_vs_anchor"][category] = primary["k8"][category]["delta_3_to_0"] / denom

    probe_by_type = dict((k8_probe.get("best") or {}).get("by_answer_type", {}) or {})
    probe_ablation_gap = {}
    for category in categories:
        probe_acc = float(probe_by_type.get(category, 0.0))
        keep0 = primary["k8"][category]["keep0"]
        probe_ablation_gap[category] = {
            "probe_accuracy": probe_acc,
            "k8_keep0_accuracy": keep0,
            "probe_minus_keep0": probe_acc - keep0,
        }

    gqa_entries: List[Tuple[str, Dict[str, Any]]] = []
    for spec in list(args.gqa_results or []):
        if ":" not in str(spec):
            raise SystemExit(f"--gqa_results entries must be label:path, got {spec!r}")
        label, path = str(spec).split(":", 1)
        gqa_entries.append((label, _load_json(path)))
    gqa_table = _gqa_table(gqa_entries)

    out = {
        "primary": primary,
        "fragility_ratios": fragility_ratios,
        "probe_ablation_gap": probe_ablation_gap,
        "gqa": gqa_table,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)

    lines: List[str] = []
    lines.append("# Semantic Adapter Fragility Analysis")
    lines.append("")
    lines.append("## Primary Table")
    lines.append("")
    lines.append("| Model | Category | Keep 3 | Keep 2 | Keep 1 | Keep 0 | Delta 3->0 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for model_name in ("anchor", "k32", "k8"):
        for category in categories:
            row = primary[model_name][category]
            lines.append(
                f"| {model_name} | {category} | {_format(row['keep3'])} | {_format(row['keep2'])} | "
                f"{_format(row['keep1'])} | {_format(row['keep0'])} | {_format(row['delta_3_to_0'])} |"
            )
    lines.append("")
    lines.append("## Fragility Ratios")
    lines.append("")
    lines.append("| Category | K32 / Anchor | K8 / Anchor |")
    lines.append("|---|---:|---:|")
    for category in categories:
        lines.append(
            f"| {category} | {_format(fragility_ratios['k32_vs_anchor'][category])} | "
            f"{_format(fragility_ratios['k8_vs_anchor'][category])} |"
        )
    lines.append("")
    lines.append("## K8 Probe-Ablation Gap")
    lines.append("")
    lines.append("| Category | Probe | K8 Keep 0 | Probe - Keep 0 |")
    lines.append("|---|---:|---:|---:|")
    for category in categories:
        row = probe_ablation_gap[category]
        lines.append(
            f"| {category} | {_format(row['probe_accuracy'])} | {_format(row['k8_keep0_accuracy'])} | "
            f"{_format(row['probe_minus_keep0'])} |"
        )
    if gqa_table:
        lines.append("")
        lines.append("## GQA")
        lines.append("")
        lines.append("| Label | Keep 3 | Keep 0 | Delta |")
        lines.append("|---|---:|---:|---:|")
        for row in gqa_table:
            lines.append(
                f"| {row['label']} | {_format(row['keep3'])} | {_format(row['keep0'])} | {_format(row['delta_3_to_0'])} |"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[fragility] wrote: {os.path.abspath(args.output_json)}")
    print(f"[fragility] wrote: {os.path.abspath(args.output_md)}")


if __name__ == "__main__":
    main()
