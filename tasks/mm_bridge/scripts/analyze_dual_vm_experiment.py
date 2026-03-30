from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


ANCHOR = {
    "overall": 0.6163,
    "yes/no": 0.7589,
    "number": 0.4573,
    "other": 0.5499,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze dual-VM OCR experiment.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--full_eval_json", type=str, required=True)
    ap.add_argument("--ocr_analysis_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    return ap.parse_args()


def _parse_answer_type_line(line: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ("yes/no", "number", "other"):
        m = re.search(rf"{re.escape(key)}=([0-9.]+)", line)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_logfile(log_path: Path) -> Dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    periodic: List[Dict[str, Any]] = []
    attn_trace: List[Dict[str, float]] = []
    pending_eval: Dict[str, Any] | None = None
    for line in lines:
        m_step = re.search(r"step=(\d+).*?vitstr_attn=([0-9.]+)", line)
        if m_step:
            attn_trace.append({"step": float(m_step.group(1)), "vitstr_attn_fraction": float(m_step.group(2))})
        m_eval = re.search(r"\[eval:val\] overall_accuracy=([0-9.]+)", line)
        if m_eval:
            pending_eval = {"overall": float(m_eval.group(1))}
            continue
        if pending_eval is not None and "[eval:val] answer_type:" in line:
            pending_eval["answer_type_accuracy"] = _parse_answer_type_line(line)
            continue
        m_tag = re.search(r"fixed-eval answers appended: .* step=(\d+) tag=(periodic_eval|final_eval)", line)
        if m_tag and pending_eval is not None:
            pending_eval["step"] = int(m_tag.group(1))
            pending_eval["tag"] = str(m_tag.group(2))
            if str(m_tag.group(2)) == "periodic_eval":
                periodic.append(dict(pending_eval))
            pending_eval = None
    return {
        "periodic_evals": periodic,
        "vitstr_attn_trace": attn_trace,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    full_eval = json.loads(Path(args.full_eval_json).read_text(encoding="utf-8"))
    ocr = json.loads(Path(args.ocr_analysis_json).read_text(encoding="utf-8"))
    parsed = parse_logfile(run_dir / "logfile.txt")
    periodic = parsed["periodic_evals"]
    best_periodic = max(periodic, key=lambda x: float(x.get("overall", 0.0))) if periodic else None
    attn_trace = parsed["vitstr_attn_trace"]

    best = {
        "checkpoint": str(full_eval["checkpoint"]),
        "overall": float(full_eval["overall_accuracy"]),
        "answer_type_accuracy": dict(full_eval.get("answer_type_accuracy", {})),
    }
    summary = {
        "run_dir": str(run_dir.resolve()),
        "best_periodic": best_periodic,
        "full_eval": best,
        "ocr_analysis": ocr,
        "vitstr_attn_trace": attn_trace,
        "anchor": ANCHOR,
    }

    md = []
    md.append("# Dual-VM OCR Report")
    md.append("")
    md.append(f"Run dir: `{run_dir.resolve()}`")
    md.append(f"Best checkpoint: `{Path(full_eval['checkpoint']).name}`")
    md.append("")
    md.append("## Reference Comparison")
    md.append("")
    md.append("| Condition | Overall | Yes/No | Number | Other |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| Cement champion (SigLIP only) | {ANCHOR['overall']:.4f} | {ANCHOR['yes/no']:.4f} | {ANCHOR['number']:.4f} | {ANCHOR['other']:.4f} |")
    md.append(
        f"| Dual-VM best checkpoint | {best['overall']:.4f} | {best['answer_type_accuracy'].get('yes/no', 0.0):.4f} | "
        f"{best['answer_type_accuracy'].get('number', 0.0):.4f} | {best['answer_type_accuracy'].get('other', 0.0):.4f} |"
    )
    md.append("")
    md.append("## Periodic Eval Trace")
    md.append("")
    md.append("| Step | Overall | Yes/No | Number | Other |")
    md.append("|---|---:|---:|---:|---:|")
    for row in periodic:
        acc = row.get("answer_type_accuracy", {})
        md.append(
            f"| {int(row['step'])} | {float(row.get('overall', 0.0)):.4f} | {float(acc.get('yes/no', 0.0)):.4f} | "
            f"{float(acc.get('number', 0.0)):.4f} | {float(acc.get('other', 0.0)):.4f} |"
        )
    md.append("")
    md.append("## OCR Subset")
    md.append("")
    md.append(f"- subset size: `{int(ocr.get('ocr_subset_size', 0))}`")
    md.append(f"- dual overall: `{float(ocr.get('ocr_subset_dual', {}).get('overall_accuracy', 0.0)):.4f}`")
    md.append(f"- anchor overall: `{float(ocr.get('ocr_subset_anchor', {}).get('overall_accuracy', 0.0)):.4f}`")
    md.append(f"- delta overall: `{float(ocr.get('ocr_subset_delta_overall', 0.0)):.4f}`")
    md.append("")
    md.append("## ViTSTR Routing")
    md.append("")
    if attn_trace:
        md.append(
            f"- training vitstr_attn_fraction: start `{attn_trace[0]['vitstr_attn_fraction']:.4f}`, "
            f"end `{attn_trace[-1]['vitstr_attn_fraction']:.4f}`"
        )
    ocr_buckets = ocr.get("ocr_attention_by_bucket", {})
    control = ocr.get("control_attention", {})
    for bucket, row in sorted(ocr_buckets.items()):
        md.append(
            f"- `{bucket}`: mean_vitstr_attn_fraction=`{float(row.get('mean_vitstr_attn_fraction', 0.0)):.4f}` "
            f"count=`{int(row.get('count', 0))}`"
        )
    if "control" in control:
        row = control["control"]
        md.append(
            f"- `control`: mean_vitstr_attn_fraction=`{float(row.get('mean_vitstr_attn_fraction', 0.0)):.4f}` "
            f"count=`{int(row.get('count', 0))}`"
        )
    md.append("")
    md.append("## Recommendation")
    md.append("")
    overall_delta = best["overall"] - ANCHOR["overall"]
    other_delta = float(best["answer_type_accuracy"].get("other", 0.0)) - ANCHOR["other"]
    if overall_delta > 0.002:
        rec = "Dual-VM helped overall. Keep the line and follow up on OCR-targeted routing."
    elif overall_delta < -0.005:
        rec = "Dual-VM hurt the Cement baseline. The added OCR stream is not being filtered cleanly enough yet."
    else:
        rec = "Dual-VM was roughly neutral overall. The decision should come from OCR-subset lift versus non-OCR regression."
    md.append(f"- {rec}")
    md.append(f"- overall delta vs anchor: `{overall_delta:.4f}`")
    md.append(f"- other delta vs anchor: `{other_delta:.4f}`")

    output_md = Path(args.output_md) if args.output_md else (run_dir / "dual_vm_report.md")
    output_json = Path(args.output_json) if args.output_json else (run_dir / "dual_vm_report.json")
    output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[dualvm-analysis] wrote: {output_md.resolve()}")
    print(f"[dualvm-analysis] wrote: {output_json.resolve()}")


if __name__ == "__main__":
    main()
