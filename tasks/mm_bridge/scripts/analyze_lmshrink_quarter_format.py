from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


QUARTER_BRIDGE = {"overall": 0.6044, "yes/no": 0.7430, "number": 0.4497, "other": 0.5400}
QUARTER_COMP = {"overall": 0.5705, "yes/no": 0.7315, "number": 0.4432, "other": 0.4816}
QUARTER_PROBE = 0.4915
REF_40M = {"overall": 0.5900, "yes/no": 0.7393, "number": 0.4499, "other": 0.5134, "probe": 0.5103}
REF_HALF = {"overall": 0.5927, "yes/no": 0.7487, "number": 0.4511, "other": 0.5116, "probe": 0.4983}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze quarter-LM remap + format refinement experiment.")
    ap.add_argument("--bundle_dir", type=str, required=True)
    return ap.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _phase1_trace(bundle_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(bundle_dir.glob("phase1_eval_step_*_with_remap.json")):
        step = int(path.stem.split("_")[3])
        data = _read_json(path)
        rows.append(
            {
                "step": step,
                "overall": float(data.get("overall_accuracy", 0.0)),
                "answer_type_accuracy": dict(data.get("answer_type_accuracy", {})),
            }
        )
    return rows


def _phase2_trace(bundle_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(bundle_dir.glob("phase2_eval_step_*_no_remap.json")):
        step = int(path.stem.split("_")[3])
        data = _read_json(path)
        rows.append(
            {
                "step": step,
                "overall": float(data.get("overall_accuracy", 0.0)),
                "answer_type_accuracy": dict(data.get("answer_type_accuracy", {})),
            }
        )
    return rows


def _success_tier(overall: float) -> str:
    if overall >= 0.58:
        return "strong"
    if overall >= 0.575:
        return "moderate"
    return "weak"


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()

    phase1_trace = _phase1_trace(bundle_dir)
    phase1_best = max(phase1_trace, key=lambda x: x["overall"]) if phase1_trace else None
    phase1_best_gain = None
    if phase1_best is not None:
        phase1_best_gain = float(phase1_best["overall"]) - QUARTER_COMP["overall"]

    decision = _read_json(bundle_dir / "phase1_decision.json")
    proceed = bool(decision.get("proceed_phase2", False))

    lines: List[str] = []
    lines.append("# Quarter-LM Format Refinement Report")
    lines.append("")
    lines.append(f"Bundle: `{bundle_dir}`")
    lines.append("")
    lines.append("## Phase 1")
    lines.append("")
    lines.append("| Condition | Overall | Yes/No | Number | Other |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Quarter bridge | {QUARTER_BRIDGE['overall']:.4f} | {QUARTER_BRIDGE['yes/no']:.4f} | {QUARTER_BRIDGE['number']:.4f} | {QUARTER_BRIDGE['other']:.4f} |"
    )
    lines.append(
        f"| Quarter compressed | {QUARTER_COMP['overall']:.4f} | {QUARTER_COMP['yes/no']:.4f} | {QUARTER_COMP['number']:.4f} | {QUARTER_COMP['other']:.4f} |"
    )
    if phase1_best is not None:
        at = phase1_best["answer_type_accuracy"]
        lines.append(
            f"| Quarter + remap keep-0 | {phase1_best['overall']:.4f} | {float(at.get('yes/no', 0.0)):.4f} | {float(at.get('number', 0.0)):.4f} | {float(at.get('other', 0.0)):.4f} |"
        )
        lines.append("")
        lines.append(f"- Phase 1 best step: `{phase1_best['step']}`")
        lines.append(f"- Gain over quarter compressed baseline: `{phase1_best_gain:+.4f}`")
    lines.append(f"- Phase 2 decision: `{'proceed' if proceed else 'stop'}`")

    if phase1_trace:
        lines.append("")
        lines.append("### Phase 1 Trace")
        lines.append("")
        lines.append("| Step | Overall | Yes/No | Number | Other |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in phase1_trace:
            at = row["answer_type_accuracy"]
            lines.append(
                f"| {row['step']} | {row['overall']:.4f} | {float(at.get('yes/no', 0.0)):.4f} | {float(at.get('number', 0.0)):.4f} | {float(at.get('other', 0.0)):.4f} |"
            )

    report_json: Dict[str, Any] = {
        "bundle_dir": str(bundle_dir),
        "phase1_trace": phase1_trace,
        "phase1_decision": decision,
    }

    if not proceed:
        lines.append("")
        lines.append("## Read")
        lines.append("")
        lines.append("Quarter-specific remap did not clear the proceed threshold, so the prior quarter compression should be treated as already close to format-optimal.")
        out_md = bundle_dir / "quarter_format_report.md"
        out_json = bundle_dir / "quarter_format_report.json"
        out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        out_json.write_text(json.dumps(report_json, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[quarter-format] wrote: {out_md}")
        print(f"[quarter-format] wrote: {out_json}")
        return

    phase2_trace = _phase2_trace(bundle_dir)
    best_step = int(decision.get("phase2_best_step", 5000))
    full_eval = _read_json(bundle_dir / f"phase2_best_step_{best_step}_full.json")
    probe = _read_json(bundle_dir / "phase2_tiny_head_probe_best.json")
    best_probe = float(probe.get("best", {}).get("accuracy", 0.0))
    full_at = dict(full_eval.get("answer_type_accuracy", {}))
    tier = _success_tier(float(full_eval.get("overall_accuracy", 0.0)))

    lines.append("")
    lines.append("## Phase 2")
    lines.append("")
    lines.append("| Condition | Overall | Yes/No | Number | Other | Probe |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| 40M format-aligned K=8 | {REF_40M['overall']:.4f} | {REF_40M['yes/no']:.4f} | {REF_40M['number']:.4f} | {REF_40M['other']:.4f} | {REF_40M['probe']:.4f} |"
    )
    lines.append(
        f"| Half-LM compressed | {REF_HALF['overall']:.4f} | {REF_HALF['yes/no']:.4f} | {REF_HALF['number']:.4f} | {REF_HALF['other']:.4f} | {REF_HALF['probe']:.4f} |"
    )
    lines.append(
        f"| Quarter compressed | {QUARTER_COMP['overall']:.4f} | {QUARTER_COMP['yes/no']:.4f} | {QUARTER_COMP['number']:.4f} | {QUARTER_COMP['other']:.4f} | {QUARTER_PROBE:.4f} |"
    )
    lines.append(
        f"| Quarter refined | {float(full_eval.get('overall_accuracy', 0.0)):.4f} | {float(full_at.get('yes/no', 0.0)):.4f} | {float(full_at.get('number', 0.0)):.4f} | {float(full_at.get('other', 0.0)):.4f} | {best_probe:.4f} |"
    )
    lines.append("")
    lines.append(f"- Best checkpoint: `step_{best_step}.tar`")
    lines.append(f"- Success tier: `{tier}`")
    lines.append(
        f"- Gain over original quarter compressed: `{float(full_eval.get('overall_accuracy', 0.0)) - QUARTER_COMP['overall']:+.4f}`"
    )
    lines.append(
        f"- Improvement in `other`: `{float(full_at.get('other', 0.0)) - QUARTER_COMP['other']:+.4f}`"
    )

    if phase2_trace:
        lines.append("")
        lines.append("### Phase 2 Trace")
        lines.append("")
        lines.append("| Step | Overall | Yes/No | Number | Other |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in phase2_trace:
            at = row["answer_type_accuracy"]
            lines.append(
                f"| {row['step']} | {row['overall']:.4f} | {float(at.get('yes/no', 0.0)):.4f} | {float(at.get('number', 0.0)):.4f} | {float(at.get('other', 0.0)):.4f} |"
            )

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if tier == "strong":
        lines.append("Quarter LM looks like a real deployment frontier. The refined quarter-specific format alignment closes enough of the gap that 12.2M should remain the active small-LM line.")
    elif tier == "moderate":
        lines.append("Quarter LM improves meaningfully but not completely. It remains a promising small-LM option, but Half is still the safer deployment frontier.")
    else:
        lines.append("Quarter-specific refinement did not close enough of the gap. Half should remain the deployment frontier, and further quarter work should only continue if there is a specific product need for the smaller LM.")

    report_json.update(
        {
            "phase2_trace": phase2_trace,
            "best_step": best_step,
            "full_eval": full_eval,
            "probe": probe,
            "success_tier": tier,
        }
    )
    out_md = bundle_dir / "quarter_format_report.md"
    out_json = bundle_dir / "quarter_format_report.json"
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(report_json, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[quarter-format] wrote: {out_md}")
    print(f"[quarter-format] wrote: {out_json}")


if __name__ == "__main__":
    main()
