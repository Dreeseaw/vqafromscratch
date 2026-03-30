from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DUAL_FULL = {"overall": 0.6307, "yes/no": 0.7918, "number": 0.4699, "other": 0.5508, "ocr": 0.2930}
CEMENT = {"overall": 0.6163, "yes/no": 0.7589, "number": 0.4573, "other": 0.5499, "ocr": 0.2510}
SIGLIP_K8 = {"overall": 0.5900, "yes/no": 0.7393, "number": 0.4499, "other": 0.5134}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze dual-VM compressed experiment bundle.")
    ap.add_argument("--bundle_dir", type=str, required=True)
    return ap.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_json_path(bundle_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = bundle_dir / path
    if candidate.exists():
        return candidate
    candidate = bundle_dir / path.name
    if candidate.exists():
        return candidate
    return candidate


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


def _success_tier(overall: float, ocr: float) -> str:
    if overall > SIGLIP_K8["overall"] and ocr > 0.27:
        return "strong"
    if overall >= 0.58 and ocr > CEMENT["ocr"]:
        return "moderate"
    return "weak"


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()

    phase1_decision = _read_json(bundle_dir / "phase1_decision.json")
    phase1_trace = _phase1_trace(bundle_dir)
    keep0 = _read_json(bundle_dir / "phase1_dual_keep0_full.json")
    remap_full = _read_json(_resolve_json_path(bundle_dir, str(phase1_decision["phase1_best_full_eval_path"])))
    proceed = bool(phase1_decision.get("proceed_phase2", False))

    lines: List[str] = []
    lines.append("# Dual-VM Compression Report")
    lines.append("")
    lines.append(f"Bundle: `{bundle_dir}`")
    lines.append("")
    lines.append("## Phase 1")
    lines.append("")
    lines.append("| Condition | Overall | Yes/No | Number | Other |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Dual-VM warm-start keep-3 | {DUAL_FULL['overall']:.4f} | {DUAL_FULL['yes/no']:.4f} | {DUAL_FULL['number']:.4f} | {DUAL_FULL['other']:.4f} |"
    )
    keep0_at = keep0.get("answer_type_accuracy", {})
    remap_at = remap_full.get("answer_type_accuracy", {})
    lines.append(
        f"| Dual-VM keep-0 | {float(keep0.get('overall_accuracy', 0.0)):.4f} | {float(keep0_at.get('yes/no', 0.0)):.4f} | {float(keep0_at.get('number', 0.0)):.4f} | {float(keep0_at.get('other', 0.0)):.4f} |"
    )
    lines.append(
        f"| Dual-VM + remap keep-0 | {float(remap_full.get('overall_accuracy', 0.0)):.4f} | {float(remap_at.get('yes/no', 0.0)):.4f} | {float(remap_at.get('number', 0.0)):.4f} | {float(remap_at.get('other', 0.0)):.4f} |"
    )
    lines.append("")
    lines.append(
        f"- keep-0 drop from full dual-VM: `{float(keep0.get('overall_accuracy', 0.0)) - DUAL_FULL['overall']:+.4f}`"
    )
    lines.append(
        f"- remap recovery over keep-0: `{float(remap_full.get('overall_accuracy', 0.0)) - float(keep0.get('overall_accuracy', 0.0)):+.4f}`"
    )
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

    out: Dict[str, Any] = {
        "bundle_dir": str(bundle_dir),
        "phase1_decision": phase1_decision,
        "phase1_keep0": keep0,
        "phase1_remap_full": remap_full,
        "phase1_trace": phase1_trace,
    }

    if not proceed:
        lines.append("")
        lines.append("## Read")
        lines.append("")
        lines.append("The dual-VM remap gate did not justify running compression. The warm-start dual-VM line should be treated as too adapter-dependent or too format-fragile for this K=8 compression recipe.")
        out_md = bundle_dir / "dualvm_compression_report.md"
        out_json = bundle_dir / "dualvm_compression_report.json"
        out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        out_json.write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[dualvm-compression] wrote: {out_md}")
        print(f"[dualvm-compression] wrote: {out_json}")
        return

    phase2_trace = _phase2_trace(bundle_dir)
    best_step = int(phase1_decision.get("phase2_best_step", 3000))
    full_eval = _read_json(bundle_dir / f"phase2_best_step_{best_step}_full.json")
    probe = _read_json(bundle_dir / "phase2_tiny_head_probe_best.json")
    ocr = _read_json(bundle_dir / "phase2_ocr_analysis.json")
    full_at = full_eval.get("answer_type_accuracy", {})
    best_probe = float(probe.get("best", {}).get("accuracy", 0.0))
    ocr_overall = float(ocr.get("ocr_subset_dual", {}).get("overall_accuracy", 0.0))
    tier = _success_tier(float(full_eval.get("overall_accuracy", 0.0)), ocr_overall)

    lines.append("")
    lines.append("## Phase 2")
    lines.append("")
    lines.append("| Condition | Overall | Yes/No | Number | Other | OCR subset |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Dual-VM warm-start | {DUAL_FULL['overall']:.4f} | {DUAL_FULL['yes/no']:.4f} | {DUAL_FULL['number']:.4f} | {DUAL_FULL['other']:.4f} | {DUAL_FULL['ocr']:.4f} |"
    )
    lines.append(
        f"| Cement anchor | {CEMENT['overall']:.4f} | {CEMENT['yes/no']:.4f} | {CEMENT['number']:.4f} | {CEMENT['other']:.4f} | {CEMENT['ocr']:.4f} |"
    )
    lines.append(
        f"| SigLIP-only K=8 | {SIGLIP_K8['overall']:.4f} | {SIGLIP_K8['yes/no']:.4f} | {SIGLIP_K8['number']:.4f} | {SIGLIP_K8['other']:.4f} | - |"
    )
    lines.append(
        f"| Dual-VM K=8 | {float(full_eval.get('overall_accuracy', 0.0)):.4f} | {float(full_at.get('yes/no', 0.0)):.4f} | {float(full_at.get('number', 0.0)):.4f} | {float(full_at.get('other', 0.0)):.4f} | {ocr_overall:.4f} |"
    )
    lines.append("")
    lines.append(f"- Best checkpoint: `step_{best_step}.tar`")
    lines.append(f"- Probe: `{best_probe:.4f}`")
    lines.append(f"- Success tier: `{tier}`")
    lines.append(
        f"- OCR retention vs warm-start: `{ocr_overall - DUAL_FULL['ocr']:+.4f}`"
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
        lines.append("Combine dual-VM and compression going forward. The OCR signal survived K=8 and the compressed dual-VM line is strictly better than the SigLIP-only compressed baseline.")
    elif tier == "moderate":
        lines.append("The combination is useful for OCR-heavy targets, but not a clear general replacement for SigLIP-only compression. Keep both lines alive.")
    else:
        lines.append("Do not combine dual-VM and compression as the default path. Use the dual-VM line uncompressed and keep compression on the simpler SigLIP-only branch.")

    out.update(
        {
            "phase2_trace": phase2_trace,
            "best_step": best_step,
            "phase2_full_eval": full_eval,
            "phase2_probe": probe,
            "phase2_ocr": ocr,
            "success_tier": tier,
        }
    )
    out_md = bundle_dir / "dualvm_compression_report.md"
    out_json = bundle_dir / "dualvm_compression_report.json"
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[dualvm-compression] wrote: {out_md}")
    print(f"[dualvm-compression] wrote: {out_json}")


if __name__ == "__main__":
    main()
