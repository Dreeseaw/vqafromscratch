from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


REFERENCE = {
    "keep3": {"overall": 0.6154, "yes/no": 0.7611, "number": 0.4583, "other": 0.5462},
    "keep0": {"overall": 0.3744, "yes/no": 0.4510, "number": 0.2757, "other": 0.3423},
    "remap": {"overall": 0.5762, "yes/no": 0.7456, "number": 0.4377, "other": 0.4839},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze semantic bottleneck format-alignment experiment outputs.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--output_md", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    return ap.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scan_step_evals(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("format_eval_step_*_*.json")):
        m = re.match(r"format_eval_step_(\d+)_(no_remap|with_remap)\.json$", path.name)
        if not m:
            continue
        data = _read_json(path)
        rows.append(
            {
                "step": int(m.group(1)),
                "mode": str(m.group(2)),
                "overall": float(data.get("overall_accuracy", 0.0)),
                "answer_type_accuracy": dict(data.get("answer_type_accuracy", {})),
                "path": str(path),
            }
        )
    rows.sort(key=lambda x: (x["step"], x["mode"]))
    return rows


def _group_step_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        step = int(row["step"])
        slot = grouped.setdefault(step, {"step": step})
        slot[str(row["mode"])] = row
    return [grouped[k] for k in sorted(grouped)]


def _best_checkpoint(grouped: List[Dict[str, Any]]) -> int:
    best_step = -1
    best_acc = -1.0
    for row in grouped:
        base = row.get("no_remap")
        if not base:
            continue
        acc = float(base.get("overall", 0.0))
        if acc > best_acc:
            best_acc = acc
            best_step = int(row["step"])
    return best_step


def _parse_training_trace(logfile: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"^\[mm\] step=(?P<step>\d+) .*?loss=(?P<loss>[0-9.]+) .*?loss_vqa=(?P<loss_vqa>[0-9.]+).*?"
        r"(?:distill=(?P<distill>[0-9.]+))?.*?"
        r"(?:sem_format=(?P<sem_format>[0-9.]+))?.*?"
        r"(?:sem_format_w=(?P<sem_format_w>[0-9.]+))?.*?"
        r"(?:format_cos=(?P<format_cos>[0-9.]+))?.*?"
        r"(?:comp_attn_ent=(?P<attn>[0-9.]+))?.*$"
    )
    for line in logfile.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        gd = m.groupdict()
        rows.append(
            {
                "step": int(gd["step"]),
                "loss_total": float(gd["loss"]),
                "loss_vqa": float(gd["loss_vqa"]),
                "loss_distill": float(gd["distill"]) if gd.get("distill") else None,
                "loss_format": float(gd["sem_format"]) if gd.get("sem_format") else None,
                "format_weight": float(gd["sem_format_w"]) if gd.get("sem_format_w") else None,
                "format_cosine_sim": float(gd["format_cos"]) if gd.get("format_cos") else None,
                "compression_attn_entropy": float(gd["attn"]) if gd.get("attn") else None,
            }
        )
    return rows


def _success_tier(full_no_remap: Dict[str, Any]) -> str:
    overall = float(full_no_remap.get("overall_accuracy", 0.0))
    yesno = float(full_no_remap.get("answer_type_accuracy", {}).get("yes/no", 0.0))
    if overall >= 0.57 and yesno >= 0.74:
        return "strong"
    if overall >= 0.50:
        return "moderate"
    return "weak"


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    rows = _scan_step_evals(run_dir)
    grouped = _group_step_rows(rows)
    best_step = _best_checkpoint(grouped)
    if best_step < 0:
        raise RuntimeError(f"No checkpoint eval JSONs found in {run_dir}")

    full_no = _read_json(run_dir / f"format_eval_best_step_{best_step}_no_remap_full.json")
    full_yes = _read_json(run_dir / f"format_eval_best_step_{best_step}_with_remap_full.json")
    probe = _read_json(run_dir / "tiny_head_probe_best.json")
    ablation = _read_json(run_dir / "adapter_ablation_best.json")
    trace = _parse_training_trace(run_dir / "logfile.txt")
    tier = _success_tier(full_no)

    lines: List[str] = []
    lines.append("# Format Alignment Report")
    lines.append("")
    lines.append(f"Run dir: `{run_dir}`")
    lines.append(f"Best checkpoint: `step_{best_step}.tar`")
    lines.append(f"Success tier: `{tier}`")
    lines.append("")
    lines.append("## Reference Comparison")
    lines.append("")
    lines.append("| Condition | Overall | Yes/No | Number | Other |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| K=8 keep-3 full system | {REFERENCE['keep3']['overall']:.4f} | {REFERENCE['keep3']['yes/no']:.4f} | {REFERENCE['keep3']['number']:.4f} | {REFERENCE['keep3']['other']:.4f} |"
    )
    lines.append(
        f"| K=8 keep-0 no remap | {REFERENCE['keep0']['overall']:.4f} | {REFERENCE['keep0']['yes/no']:.4f} | {REFERENCE['keep0']['number']:.4f} | {REFERENCE['keep0']['other']:.4f} |"
    )
    lines.append(
        f"| K=8 + remap keep-0 | {REFERENCE['remap']['overall']:.4f} | {REFERENCE['remap']['yes/no']:.4f} | {REFERENCE['remap']['number']:.4f} | {REFERENCE['remap']['other']:.4f} |"
    )
    lines.append(
        f"| Format-aligned K=8, no remap | {float(full_no['overall_accuracy']):.4f} | {float(full_no['answer_type_accuracy'].get('yes/no', 0.0)):.4f} | {float(full_no['answer_type_accuracy'].get('number', 0.0)):.4f} | {float(full_no['answer_type_accuracy'].get('other', 0.0)):.4f} |"
    )
    lines.append(
        f"| Format-aligned K=8, with remap | {float(full_yes['overall_accuracy']):.4f} | {float(full_yes['answer_type_accuracy'].get('yes/no', 0.0)):.4f} | {float(full_yes['answer_type_accuracy'].get('number', 0.0)):.4f} | {float(full_yes['answer_type_accuracy'].get('other', 0.0)):.4f} |"
    )
    lines.append("")
    lines.append("## Checkpoint Eval Trace")
    lines.append("")
    lines.append("| Step | No Remap Overall | No Remap Y/N | No Remap Num | No Remap Other | With Remap Overall | With Remap Y/N | With Remap Num | With Remap Other |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in grouped:
        no = row.get("no_remap", {})
        yes = row.get("with_remap", {})
        no_at = no.get("answer_type_accuracy", {})
        yes_at = yes.get("answer_type_accuracy", {})
        lines.append(
            f"| {row['step']} | {_fmt(no.get('overall'))} | {_fmt(no_at.get('yes/no'))} | {_fmt(no_at.get('number'))} | {_fmt(no_at.get('other'))} | "
            f"{_fmt(yes.get('overall'))} | {_fmt(yes_at.get('yes/no'))} | {_fmt(yes_at.get('number'))} | {_fmt(yes_at.get('other'))} |"
        )
    lines.append("")
    lines.append("## Training Trace")
    lines.append("")
    lines.append("| Step | Loss | VQA | Distill | Format | Format W | Format Cos | Attn Ent |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in trace:
        if row["step"] % 100 != 0 and row["step"] != 1:
            continue
        lines.append(
            f"| {row['step']} | {_fmt(row['loss_total'])} | {_fmt(row['loss_vqa'])} | {_fmt(row['loss_distill'])} | {_fmt(row['loss_format'])} | {_fmt(row['format_weight'])} | {_fmt(row['format_cosine_sim'])} | {_fmt(row['compression_attn_entropy'])} |"
        )
    lines.append("")
    lines.append("## Secondary Diagnostics")
    lines.append("")
    lines.append("### Adapter Ablation With Adapters Re-enabled")
    lines.append("")
    lines.append("| Keep | Overall | Yes/No | Number | Other |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in ablation.get("results", []):
        at = row.get("answer_type_accuracy", {})
        lines.append(
            f"| {int(row['keep_count'])} | {float(row.get('overall_accuracy', 0.0)):.4f} | {float(at.get('yes/no', 0.0)):.4f} | {float(at.get('number', 0.0)):.4f} | {float(at.get('other', 0.0)):.4f} |"
        )
    lines.append("")
    lines.append("### Tiny-Head Probe")
    lines.append("")
    best_probe = probe.get("best", {})
    lines.append(
        f"- overall: {float(best_probe.get('accuracy', 0.0)):.4f} vs prior K=8 probe 0.5031"
    )
    by_type = best_probe.get("by_answer_type", {})
    if by_type:
        lines.append(
            f"- yes/no: {float(by_type.get('yes/no', 0.0)):.4f}, number: {float(by_type.get('number', 0.0)):.4f}, other: {float(by_type.get('other', 0.0)):.4f}"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if tier == "strong":
        lines.append("Format alignment succeeded. The next move should be to test whether adapters are now redundant, complementary, or mildly additive on top of the aligned bottleneck, rather than spending more effort on raw LM-side rescue.")
    elif tier == "moderate":
        lines.append("Format alignment helped but did not fully internalize the remap. The next move should be either longer bottleneck-only training or a thin permanent remap layer.")
    else:
        lines.append("Format alignment did not solve the interface problem. The next move should shift toward minimal adapters or another lightweight in-network routing mechanism.")

    out_md = Path(args.output_md) if args.output_md else run_dir / "format_alignment_report.md"
    out_json = Path(args.output_json) if args.output_json else run_dir / "format_alignment_report.json"
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    out = {
        "run_dir": str(run_dir),
        "best_step": int(best_step),
        "success_tier": tier,
        "checkpoint_trace": grouped,
        "full_no_remap": full_no,
        "full_with_remap": full_yes,
        "probe": probe,
        "adapter_ablation": ablation,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[format-analysis] wrote: {out_md}")
    print(f"[format-analysis] wrote: {out_json}")


if __name__ == "__main__":
    main()
