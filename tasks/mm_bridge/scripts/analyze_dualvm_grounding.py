from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DUAL_WARM = {"overall": 0.6307, "yes/no": 0.7918, "number": 0.4699, "other": 0.5508, "ocr": 0.2930}
DUAL_K8 = {"overall": 0.6109, "yes/no": 0.7866, "number": 0.4611, "other": 0.5168, "ocr": 0.2106, "probe": 0.5337}
SIGLIP_K8 = {"overall": 0.5900, "yes/no": 0.7393, "number": 0.4499, "other": 0.5134, "probe": 0.5103}
CEMENT = {"overall": 0.6163, "yes/no": 0.7589, "number": 0.4573, "other": 0.5499, "ocr": 0.2510}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze the dual-VM grounding experiment bundle.")
    ap.add_argument("--bundle_dir", type=str, required=True)
    return ap.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trace(bundle_dir: Path, prefix: str, suffix: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(bundle_dir.glob(f"{prefix}_step_*_{suffix}.json")):
        step = int(path.stem.split("_")[2])
        data = _read_json(path)
        rows.append(
            {
                "step": step,
                "overall": float(data.get("overall_accuracy", 0.0)),
                "answer_type_accuracy": dict(data.get("answer_type_accuracy", {})),
            }
        )
    return rows


def _success_tier(overall: float, other: float) -> str:
    if overall >= 0.6163 and other >= 0.53:
        return "strong"
    if overall >= 0.6150 and other >= (DUAL_K8["other"] + 0.01):
        return "moderate"
    return "weak"


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()

    control_best_step = int(_read_json(bundle_dir / "control_best_step.json")["best_step"])
    mixed_best_step = int(_read_json(bundle_dir / "mixed_best_step.json")["best_step"])

    control_full = _read_json(bundle_dir / f"control_best_step_{control_best_step}_full.json")
    mixed_full = _read_json(bundle_dir / f"mixed_best_step_{mixed_best_step}_full.json")
    control_gqa = _read_json(bundle_dir / f"control_best_step_{control_best_step}_gqa.json")
    mixed_gqa = _read_json(bundle_dir / f"mixed_best_step_{mixed_best_step}_gqa.json")
    control_ocr = _read_json(bundle_dir / "control_ocr_analysis.json")
    mixed_ocr = _read_json(bundle_dir / "mixed_ocr_analysis.json")
    control_probe = _read_json(bundle_dir / "control_probe.json")
    mixed_probe = _read_json(bundle_dir / "mixed_probe.json")
    control_ground = _read_json(bundle_dir / "control_grounding_mass.json")
    mixed_ground = _read_json(bundle_dir / "mixed_grounding_mass.json")
    gqa_sanity = _read_json(bundle_dir / "gqa_exact_sanity.json")

    control_trace = _trace(bundle_dir, "control", "vqa")
    mixed_trace = _trace(bundle_dir, "mixed", "vqa")

    control_at = control_full.get("answer_type_accuracy", {})
    mixed_at = mixed_full.get("answer_type_accuracy", {})
    control_probe_best = float(control_probe.get("best", {}).get("accuracy", 0.0))
    mixed_probe_best = float(mixed_probe.get("best", {}).get("accuracy", 0.0))
    control_ocr_acc = float(control_ocr.get("ocr_subset_dual", {}).get("overall_accuracy", 0.0))
    mixed_ocr_acc = float(mixed_ocr.get("ocr_subset_dual", {}).get("overall_accuracy", 0.0))
    tier = _success_tier(float(mixed_full.get("overall_accuracy", 0.0)), float(mixed_at.get("other", 0.0)))

    lines: List[str] = []
    lines.append("# Dual-VM Grounding Report")
    lines.append("")
    lines.append(f"Bundle: `{bundle_dir}`")
    lines.append("")
    lines.append("## Sanity")
    lines.append("")
    lines.append(f"- GQA exact-match sanity overall: `{float(gqa_sanity.get('overall_accuracy', 0.0)):.4f}`")
    lines.append("")
    lines.append("## Main Comparison")
    lines.append("")
    lines.append("| Condition | Overall | Yes/No | Number | Other | OCR subset | Probe |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Dual-VM warm-start | {DUAL_WARM['overall']:.4f} | {DUAL_WARM['yes/no']:.4f} | {DUAL_WARM['number']:.4f} | {DUAL_WARM['other']:.4f} | {DUAL_WARM['ocr']:.4f} | - |"
    )
    lines.append(
        f"| Dual-VM K=8 prior | {DUAL_K8['overall']:.4f} | {DUAL_K8['yes/no']:.4f} | {DUAL_K8['number']:.4f} | {DUAL_K8['other']:.4f} | {DUAL_K8['ocr']:.4f} | {DUAL_K8['probe']:.4f} |"
    )
    lines.append(
        f"| Dual-VM K=8 control | {float(control_full.get('overall_accuracy', 0.0)):.4f} | {float(control_at.get('yes/no', 0.0)):.4f} | {float(control_at.get('number', 0.0)):.4f} | {float(control_at.get('other', 0.0)):.4f} | {control_ocr_acc:.4f} | {control_probe_best:.4f} |"
    )
    lines.append(
        f"| Dual-VM K=8 + grounding+GQA | {float(mixed_full.get('overall_accuracy', 0.0)):.4f} | {float(mixed_at.get('yes/no', 0.0)):.4f} | {float(mixed_at.get('number', 0.0)):.4f} | {float(mixed_at.get('other', 0.0)):.4f} | {mixed_ocr_acc:.4f} | {mixed_probe_best:.4f} |"
    )
    lines.append(
        f"| SigLIP-only K=8 | {SIGLIP_K8['overall']:.4f} | {SIGLIP_K8['yes/no']:.4f} | {SIGLIP_K8['number']:.4f} | {SIGLIP_K8['other']:.4f} | - | {SIGLIP_K8['probe']:.4f} |"
    )
    lines.append(
        f"| Cement anchor | {CEMENT['overall']:.4f} | {CEMENT['yes/no']:.4f} | {CEMENT['number']:.4f} | {CEMENT['other']:.4f} | {CEMENT['ocr']:.4f} | - |"
    )
    lines.append("")
    lines.append("## Deltas")
    lines.append("")
    lines.append(f"- mixed minus control overall: `{float(mixed_full.get('overall_accuracy', 0.0)) - float(control_full.get('overall_accuracy', 0.0)):+.4f}`")
    lines.append(f"- mixed minus control other: `{float(mixed_at.get('other', 0.0)) - float(control_at.get('other', 0.0)):+.4f}`")
    lines.append(f"- mixed minus prior dual-VM K=8 overall: `{float(mixed_full.get('overall_accuracy', 0.0)) - DUAL_K8['overall']:+.4f}`")
    lines.append(f"- mixed minus prior dual-VM K=8 other: `{float(mixed_at.get('other', 0.0)) - DUAL_K8['other']:+.4f}`")
    lines.append(f"- mixed OCR delta vs control: `{mixed_ocr_acc - control_ocr_acc:+.4f}`")
    lines.append(f"- mixed grounding mass delta vs control: `{float(mixed_ground.get('mean_target_mass', 0.0)) - float(control_ground.get('mean_target_mass', 0.0)):+.4f}`")
    lines.append(f"- success tier: `{tier}`")
    lines.append("")
    lines.append("## GQA")
    lines.append("")
    lines.append("| Condition | Overall | Spatial | Attribute | Exist | Count |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Control | {float(control_gqa.get('overall_accuracy', 0.0)):.4f} | {float(control_gqa.get('group_accuracy', {}).get('spatial', 0.0)):.4f} | {float(control_gqa.get('group_accuracy', {}).get('attribute', 0.0)):.4f} | {float(control_gqa.get('group_accuracy', {}).get('exist', 0.0)):.4f} | {float(control_gqa.get('group_accuracy', {}).get('count', 0.0)):.4f} |"
    )
    lines.append(
        f"| Grounding+GQA | {float(mixed_gqa.get('overall_accuracy', 0.0)):.4f} | {float(mixed_gqa.get('group_accuracy', {}).get('spatial', 0.0)):.4f} | {float(mixed_gqa.get('group_accuracy', {}).get('attribute', 0.0)):.4f} | {float(mixed_gqa.get('group_accuracy', {}).get('exist', 0.0)):.4f} | {float(mixed_gqa.get('group_accuracy', {}).get('count', 0.0)):.4f} |"
    )
    lines.append("")
    lines.append("## Trace")
    lines.append("")
    lines.append("### Control VQAv2")
    lines.append("")
    lines.append("| Step | Overall | Yes/No | Number | Other |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in control_trace:
        at = row["answer_type_accuracy"]
        lines.append(
            f"| {row['step']} | {row['overall']:.4f} | {float(at.get('yes/no', 0.0)):.4f} | {float(at.get('number', 0.0)):.4f} | {float(at.get('other', 0.0)):.4f} |"
        )
    lines.append("")
    lines.append("### Grounding+GQA VQAv2")
    lines.append("")
    lines.append("| Step | Overall | Yes/No | Number | Other |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in mixed_trace:
        at = row["answer_type_accuracy"]
        lines.append(
            f"| {row['step']} | {row['overall']:.4f} | {float(at.get('yes/no', 0.0)):.4f} | {float(at.get('number', 0.0)):.4f} | {float(at.get('other', 0.0)):.4f} |"
        )

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if tier == "strong":
        lines.append("Use grounding+GQA as the new default compressed dual-VM recipe. It materially improves the K=8 bottleneck and reaches Cement-level quality.")
    elif tier == "moderate":
        lines.append("Keep the grounding+GQA recipe alive. It improves compressed dual-VM behavior, especially on the broad `other` regime, but does not fully replace the uncompressed dual-VM frontier.")
    else:
        lines.append("Do not adopt grounding+GQA as the default compressed dual-VM recipe. The extra supervision is not paying for its added complexity under the current bottleneck design.")

    out = {
        "bundle_dir": str(bundle_dir),
        "control_best_step": control_best_step,
        "mixed_best_step": mixed_best_step,
        "control_full": control_full,
        "mixed_full": mixed_full,
        "control_gqa": control_gqa,
        "mixed_gqa": mixed_gqa,
        "control_ocr": control_ocr,
        "mixed_ocr": mixed_ocr,
        "control_probe": control_probe,
        "mixed_probe": mixed_probe,
        "control_grounding_mass": control_ground,
        "mixed_grounding_mass": mixed_ground,
        "success_tier": tier,
    }
    (bundle_dir / "grounding_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (bundle_dir / "grounding_report.json").write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[dualvm-grounding] wrote: {bundle_dir / 'grounding_report.md'}")
    print(f"[dualvm-grounding] wrote: {bundle_dir / 'grounding_report.json'}")


if __name__ == "__main__":
    main()
