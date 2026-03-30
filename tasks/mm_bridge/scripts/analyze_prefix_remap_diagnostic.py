from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List


KEEP3 = {"overall": 0.6154, "yes/no": 0.7611, "number": 0.4583, "other": 0.5462}
KEEP0 = {"overall": 0.3744, "yes/no": 0.4510, "number": 0.2757, "other": 0.3423}
PROBE = {"overall": 0.5031, "yes/no": 0.6199, "number": 0.3511, "other": 0.4316}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze prefix-remap diagnostic run outputs.")
    ap.add_argument("--run_dir", type=str, required=True)
    return ap.parse_args()


def _parse_eval_blocks(logfile: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    last_step = None
    step_re = re.compile(r"\[mm\] step=(\d+)")
    overall_re = re.compile(r"\[eval:val\] overall_accuracy=([0-9.]+) scorer=([a-z]+)")
    answer_re = re.compile(r"\[eval:val\] answer_type: (.*)")
    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            m = step_re.search(line)
            if m:
                last_step = int(m.group(1))
                continue
            m = overall_re.search(line)
            if m:
                blocks.append(
                    {
                        "step_hint": last_step,
                        "overall": float(m.group(1)),
                        "scorer": str(m.group(2)),
                        "answer_type": {},
                    }
                )
                continue
            m = answer_re.search(line)
            if m and blocks:
                metrics: Dict[str, float] = {}
                for chunk in str(m.group(1)).split():
                    if "=" not in chunk:
                        continue
                    key, value = chunk.split("=", 1)
                    metrics[str(key)] = float(value)
                blocks[-1]["answer_type"] = metrics
    return blocks


def _load_fixed_eval_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pair_eval_rows(blocks: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    paired: List[Dict[str, Any]] = []
    n = min(len(blocks), len(rows))
    for i in range(n):
        out = dict(rows[i])
        out["overall"] = float(blocks[i]["overall"])
        out["scorer"] = str(blocks[i]["scorer"])
        out["answer_type"] = dict(blocks[i].get("answer_type", {}))
        out["step_hint"] = blocks[i].get("step_hint")
        paired.append(out)
    return paired


def _find_final_full_eval(paired: List[Dict[str, Any]]) -> Dict[str, Any]:
    for row in reversed(paired):
        if str(row.get("tag", "")) == "final_eval":
            return row
    if paired:
        return paired[-1]
    raise RuntimeError("No eval rows found in remap diagnostic run.")


def _outcome(yes_no: float) -> str:
    if yes_no >= 0.60:
        return "A"
    if yes_no >= 0.50:
        return "B"
    return "C"


def _recommendation(code: str) -> str:
    if code == "A":
        return "Format mismatch dominates. Next step should be direct format-alignment pressure on bottleneck outputs, not more LM-side capacity."
    if code == "B":
        return "A linear remap recovers part of the gap. Next step should be a tiny nonlinear remap or a trained bottleneck format-alignment loss."
    return "Linear remap is not enough. Keep some adapter-like capacity or move to a bottleneck training regime that explicitly pressures the LM-facing interface under adapter removal."


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    logfile = os.path.join(run_dir, "logfile.txt")
    fixed_eval = os.path.join(run_dir, "fixed_eval_val_answers.jsonl")
    if not os.path.isfile(logfile):
        raise SystemExit(f"Missing logfile: {logfile}")
    if not os.path.isfile(fixed_eval):
        raise SystemExit(f"Missing fixed eval rows: {fixed_eval}")

    blocks = _parse_eval_blocks(logfile)
    rows = _load_fixed_eval_rows(fixed_eval)
    paired = _pair_eval_rows(blocks, rows)
    final_full = _find_final_full_eval(paired)
    final_metrics = {
        "overall": float(final_full.get("overall", 0.0)),
        "yes/no": float(final_full.get("answer_type", {}).get("yes/no", 0.0)),
        "number": float(final_full.get("answer_type", {}).get("number", 0.0)),
        "other": float(final_full.get("answer_type", {}).get("other", 0.0)),
    }

    recovery = {}
    for key in ("overall", "yes/no", "number", "other"):
        denom = KEEP3[key] - KEEP0[key]
        recovery[key] = (final_metrics[key] - KEEP0[key]) / denom if abs(denom) > 1e-12 else 0.0

    outcome = _outcome(final_metrics["yes/no"])
    report = {
        "run_dir": run_dir,
        "trainable_params_line": None,
        "eval_rows": paired,
        "final_full_eval": final_full,
        "comparison": {
            "keep3": KEEP3,
            "keep0": KEEP0,
            "probe": PROBE,
            "remap_keep0": final_metrics,
        },
        "recovery_fraction": recovery,
        "outcome": outcome,
        "recommendation": _recommendation(outcome),
    }

    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            if "trainable_params=" in line:
                report["trainable_params_line"] = line.strip()
                break

    md = []
    md.append("# Prefix Remap Diagnostic")
    md.append("")
    md.append(f"Run dir: `{run_dir}`")
    if report["trainable_params_line"]:
        md.append(f"- {report['trainable_params_line']}")
    md.append("")
    md.append("## Comparison")
    md.append("")
    md.append("| Condition | Overall | Yes/No | Number | Other |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| K=8 keep-3 (full system) | {KEEP3['overall']:.4f} | {KEEP3['yes/no']:.4f} | {KEEP3['number']:.4f} | {KEEP3['other']:.4f} |")
    md.append(f"| K=8 keep-0 (no adapters, no remap) | {KEEP0['overall']:.4f} | {KEEP0['yes/no']:.4f} | {KEEP0['number']:.4f} | {KEEP0['other']:.4f} |")
    md.append(f"| K=8 probe (linear head, no LM) | {PROBE['overall']:.4f} | {PROBE['yes/no']:.4f} | {PROBE['number']:.4f} | {PROBE['other']:.4f} |")
    md.append(f"| K=8 + PrefixRemap keep-0 | {final_metrics['overall']:.4f} | {final_metrics['yes/no']:.4f} | {final_metrics['number']:.4f} | {final_metrics['other']:.4f} |")
    md.append("")
    md.append("## Periodic Eval Trace")
    md.append("")
    md.append("| Step | Tag | Overall | Yes/No | Number | Other |")
    md.append("|---|---|---:|---:|---:|---:|")
    for row in paired:
        at = dict(row.get("answer_type", {}))
        md.append(
            f"| {int(row.get('global_step', row.get('step_hint') or 0))} | {row.get('tag', '')} | "
            f"{float(row.get('overall', 0.0)):.4f} | {float(at.get('yes/no', 0.0)):.4f} | "
            f"{float(at.get('number', 0.0)):.4f} | {float(at.get('other', 0.0)):.4f} |"
        )
    md.append("")
    md.append("## Outcome")
    md.append("")
    md.append(f"Outcome `{outcome}`")
    md.append("")
    if outcome == "A":
        md.append("- Interpretation: the failure is mostly a linear format mismatch.")
    elif outcome == "B":
        md.append("- Interpretation: part of the gap is linear format, but part is deeper routing.")
    else:
        md.append("- Interpretation: the problem is deeper than a linear format mismatch.")
    md.append("")
    md.append("## Recovery Fraction")
    md.append("")
    md.append("| Category | Fraction of adapter work recovered |")
    md.append("|---|---:|")
    for key in ("overall", "yes/no", "number", "other"):
        md.append(f"| {key} | {recovery[key]:.4f} |")
    md.append("")
    md.append("## Recommendation")
    md.append("")
    md.append(report["recommendation"])

    out_md = os.path.join(run_dir, "remap_diagnostic.md")
    out_json = os.path.join(run_dir, "remap_diagnostic.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(f"[remap-diagnostic] wrote: {out_md}")
    print(f"[remap-diagnostic] wrote: {out_json}")


if __name__ == "__main__":
    main()
