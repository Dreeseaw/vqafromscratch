#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-./.venv_local/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmgrid_direct_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"
PROGRESS_MD="${BUNDLE_DIR}/progress.md"
TIMELINE_LOG="${BUNDLE_DIR}/timeline.log"

CEMENT_CKPT="${CEMENT_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
DUALVM_CKPT="${DUALVM_CKPT:-logs/mmdualvm_v1_20260324_rerun/step_9000.tar}"
SIGLIP_REMAP="${SIGLIP_REMAP:-logs/mmsemantic_remap_v1_debug/step_500.tar}"
DUALVM_REMAP="${DUALVM_REMAP:-logs/dualvm_compressed_v1_20260325_234134_phase1_remap/step_500.tar}"
SIGLIP_DIR="${SIGLIP_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"
VITSTR_CKPT="${VITSTR_CKPT:-logs/hf_vision/vitstr_tiny_patch16_224/vitstr_tiny_patch16_224_aug.pth}"
SIGLIP_BASELINE="${SIGLIP_BASELINE:-0.5900}"
DUALVM_BASELINE="${DUALVM_BASELINE:-0.6109}"
SEED="${SEED:-42}"

mkdir -p "${BUNDLE_DIR}"
echo "# Direct Grid Bottleneck Progress" > "${PROGRESS_MD}"
touch "${TIMELINE_LOG}"

timestamp() {
  date --iso-8601=seconds
}

log_timeline() {
  echo "[$(timestamp)] $*" | tee -a "${TIMELINE_LOG}"
}

latest_step() {
  local run_dir="$1"
  local latest
  latest="$(find "${run_dir}" -maxdepth 1 -name 'step_*.tar' -printf '%f\n' 2>/dev/null | sed -E 's/^step_([0-9]+)\.tar$/\1/' | sort -n | tail -1)"
  echo "${latest:-0}"
}

run_mm_resume() {
  local run_id="$1"
  local target_step="$2"
  shift 2
  local run_dir="logs/${run_id}"
  local step
  step="$(latest_step "${run_dir}")"
  if [[ "${step}" -ge "${target_step}" ]]; then
    return 0
  fi
  if [[ "${step}" -gt 0 ]]; then
    ./runmm_v1.sh "${run_id}" "${step}" "$@"
  else
    ./runmm_v1.sh "${run_id}" "$@"
  fi
}

run_eval_sweep() {
  local run_id="$1"
  local eval_bs="$2"
  local out_prefix="$3"
  shift 3
  local ckpt
  for step in 500 1000 1500 2000 2500 3000; do
    ckpt="logs/${run_id}/step_${step}.tar"
    [[ -f "${ckpt}" ]] || continue
    if [[ ! -f "${BUNDLE_DIR}/${out_prefix}_step_${step}.json" ]]; then
      "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
        --checkpoint "${ckpt}" \
        --batch_size "${eval_bs}" \
        --eval_batches 100 \
        --disable_lm_visual_adapters \
        --output_json "${BUNDLE_DIR}/${out_prefix}_step_${step}.json" \
        "$@"
    fi
  done
}

best_step_from_prefix() {
  local prefix="$1"
  "${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}" "${prefix}"
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1])
prefix = sys.argv[2]
best_step = 3000
best_acc = -1.0
for path in sorted(bundle.glob(f"{prefix}_step_*.json")):
    try:
        step = int(path.stem.split("_")[-1])
    except Exception:
        continue
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    acc = float(data.get("overall_accuracy", 0.0))
    if acc > best_acc:
        best_acc = acc
        best_step = step
print(best_step)
PY
}

materialize_full_eval_from_log() {
  local run_id="$1"
  local out_json="$2"
  local run_dir="logs/${run_id}"
  local logfile="${run_dir}/logfile.txt"
  [[ -f "${logfile}" ]] || return 1
  "${PYTHON_BIN}" - <<'PY' "${logfile}" "${out_json}"
import json
import re
import sys
from pathlib import Path

logfile = Path(sys.argv[1])
out_json = Path(sys.argv[2])
overall = None
answer_types = None
record_count = None
tag_seen = False
for line in logfile.read_text(encoding="utf-8").splitlines():
    m = re.search(r"\[eval:val\] overall_accuracy=([0-9.]+)", line)
    if m:
        overall = float(m.group(1))
        continue
    m = re.search(r"\[eval:val\] answer_type: yes/no=([0-9.]+) number=([0-9.]+) other=([0-9.]+)", line)
    if m:
        answer_types = {
            "yes/no": float(m.group(1)),
            "number": float(m.group(2)),
            "other": float(m.group(3)),
        }
        continue
    m = re.search(r"\[mm\] eval split=val samples=([0-9]+)", line)
    if m:
        record_count = int(m.group(1))
        continue
    if "tag=final_eval" in line:
        tag_seen = True
if not tag_seen or overall is None or answer_types is None:
    raise SystemExit(1)
payload = {
    "overall_accuracy": overall,
    "answer_type_accuracy": answer_types,
    "record_count": record_count,
    "source": str(logfile),
    "derived_from_logfile_final_eval": True,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
with out_json.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True)
print(out_json)
PY
}

append_progress() {
  local name="$1"
  local started="$2"
  local status="$3"
  local best_step="$4"
  local json_path="$5"
  local notes="$6"
  "${PYTHON_BIN}" - <<'PY' "${PROGRESS_MD}" "${name}" "${started}" "${status}" "${best_step}" "${json_path}" "${notes}"
import json, sys
from pathlib import Path
progress, name, started, status, best_step, json_path, notes = sys.argv[1:]
md = []
md.append(f"\n## {name}")
md.append(f"- Started: {started}")
md.append(f"- Status: {status}")
if best_step and best_step != "0":
    md.append(f"- Best checkpoint: step_{best_step}")
if json_path and json_path != "none":
    data = json.load(open(json_path, "r", encoding="utf-8"))
    aty = data.get("answer_type_accuracy", {})
    md.append(
        "- Key result: overall "
        f"{float(data.get('overall_accuracy', 0.0)):.4f}, "
        f"y/n {float(aty.get('yes/no', 0.0)):.4f}, "
        f"num {float(aty.get('number', 0.0)):.4f}, "
        f"other {float(aty.get('other', 0.0)):.4f}"
    )
if notes:
    md.append(f"- Notes: {notes}")
with open(progress, "a", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
PY
}

run_siglip_variant() {
  local variant_key="$1"
  local run_id="$2"
  local extra_bottleneck_args=("${@:3}")
  local started
  started="$(timestamp)"
  log_timeline "START ${run_id}"
  run_mm_resume "${run_id}" 3000 \
    --vision_model siglip_base \
    --vision_checkpoint "${SIGLIP_DIR}" \
    --seed "${SEED}" \
    --max_steps 3000 \
    --manual_max_steps \
    --batch_size 64 \
    --grad_accum_steps 3 \
    --eval_batch_size 96 \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    --log_every 10 \
    --eval_every 500 \
    --eval_batches 100 \
    --ckpt_every 500 \
    --final_eval_batches 0 \
    --lr 0.0002 \
    --lr_schedule cosine \
    --lr_warmup_steps 200 \
    --freeze_mode semantic_bottleneck_only \
    --semantic_bottleneck \
    --semantic_tokens 8 \
    --semantic_latent_dim 256 \
    --semantic_recon_loss_weight 0.1 \
    --semantic_consistency_loss_weight 0.0 \
    --disable_lm_visual_adapters \
    --prefix_remap_present \
    --no-apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${SIGLIP_REMAP}" \
    --semantic_format_loss_weight 0.3 \
    --semantic_format_loss_final_weight 0.0 \
    --semantic_format_anneal_start_step 2250 \
    --semantic_format_anneal_end_step 3000 \
    --init_from_mm_checkpoint "${CEMENT_CKPT}" \
    --min_train_steps_per_s 0 \
    "${extra_bottleneck_args[@]}"
  run_eval_sweep "${run_id}" 96 "${variant_key}_periodic"
  local best_step
  best_step="$(best_step_from_prefix "${variant_key}_periodic")"
  local best_ckpt="logs/${run_id}/step_${best_step}.tar"
  local full_json="${BUNDLE_DIR}/${variant_key}_best_full.json"
  if [[ ! -f "${full_json}" ]]; then
    if [[ "${best_step}" == "3000" ]] && materialize_full_eval_from_log "${run_id}" "${full_json}" >/dev/null 2>&1; then
      :
    else
      "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
        --checkpoint "${best_ckpt}" \
        --batch_size 96 \
        --eval_batches 0 \
        --disable_lm_visual_adapters \
        --output_json "${full_json}"
    fi
  fi
  local probe_json="${BUNDLE_DIR}/${variant_key}_probe.json"
  if [[ ! -f "${probe_json}" ]]; then
    "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_semantic_probe \
      --checkpoint "${best_ckpt}" \
      --batch_size 96 \
      --probe_batch_size 256 \
      --num_workers 1 \
      --prefetch_factor 1 \
      --no-pin_memory \
      --limit_train 10000 \
      --limit_val 5000 \
      --answer_top_k 3000 \
      --epochs 10 \
      --lr 0.001 \
      --feature_pool flatten \
      --output_json "${probe_json}"
  fi
  append_progress "${variant_key}" "${started}" "COMPLETE" "${best_step}" "${full_json}" "SigLIP-only direct-grid bottleneck"
  log_timeline "END ${run_id}"
}

run_dualvm_variant() {
  local variant_key="$1"
  local run_id="$2"
  local extra_bottleneck_args=("${@:3}")
  local started
  started="$(timestamp)"
  log_timeline "START ${run_id}"
  run_mm_resume "${run_id}" 3000 \
    --vision_model siglip_vitstr_tiny_dual \
    --vision_checkpoint "${SIGLIP_DIR}" \
    --vision_aux_checkpoint "${VITSTR_CKPT}" \
    --seed "${SEED}" \
    --max_steps 3000 \
    --manual_max_steps \
    --batch_size 48 \
    --grad_accum_steps 4 \
    --eval_batch_size 64 \
    --num_workers 0 \
    --prefetch_factor 1 \
    --no-pin_memory \
    --log_every 10 \
    --eval_every 500 \
    --eval_batches 100 \
    --ckpt_every 500 \
    --final_eval_batches 0 \
    --lr 0.0002 \
    --lr_schedule cosine \
    --lr_warmup_steps 200 \
    --freeze_mode semantic_bottleneck_only \
    --semantic_bottleneck \
    --semantic_tokens 8 \
    --semantic_latent_dim 256 \
    --semantic_recon_loss_weight 0.1 \
    --semantic_consistency_loss_weight 0.0 \
    --disable_lm_visual_adapters \
    --prefix_remap_present \
    --no-apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${DUALVM_REMAP}" \
    --semantic_format_loss_weight 0.3 \
    --semantic_format_loss_final_weight 0.0 \
    --semantic_format_anneal_start_step 2250 \
    --semantic_format_anneal_end_step 3000 \
    --init_from_mm_checkpoint "${DUALVM_CKPT}" \
    --min_train_steps_per_s 0 \
    "${extra_bottleneck_args[@]}"
  run_eval_sweep "${run_id}" 64 "${variant_key}_periodic"
  local best_step
  best_step="$(best_step_from_prefix "${variant_key}_periodic")"
  local best_ckpt="logs/${run_id}/step_${best_step}.tar"
  local full_json="${BUNDLE_DIR}/${variant_key}_best_full.json"
  if [[ ! -f "${full_json}" ]]; then
    if [[ "${best_step}" == "3000" ]] && materialize_full_eval_from_log "${run_id}" "${full_json}" >/dev/null 2>&1; then
      :
    else
      "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
        --checkpoint "${best_ckpt}" \
        --batch_size 64 \
        --eval_batches 0 \
        --disable_lm_visual_adapters \
        --output_json "${full_json}"
    fi
  fi
  local probe_json="${BUNDLE_DIR}/${variant_key}_probe.json"
  if [[ ! -f "${probe_json}" ]]; then
    "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_semantic_probe \
      --checkpoint "${best_ckpt}" \
      --batch_size 64 \
      --probe_batch_size 192 \
      --num_workers 0 \
      --prefetch_factor 1 \
      --no-pin_memory \
      --limit_train 10000 \
      --limit_val 5000 \
      --answer_top_k 3000 \
      --epochs 10 \
      --lr 0.001 \
      --feature_pool flatten \
      --output_json "${probe_json}"
  fi
  local ocr_json="${BUNDLE_DIR}/${variant_key}_ocr_analysis.json"
  if [[ ! -f "${ocr_json}" ]]; then
    "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_dualvm_ocr_analysis \
      --dual_checkpoint "${best_ckpt}" \
      --anchor_checkpoint "${CEMENT_CKPT}" \
      --batch_size 64 \
      --limit_ocr 500 \
      --limit_control 100 \
      --output_json "${ocr_json}"
  fi
  append_progress "${variant_key}" "${started}" "COMPLETE" "${best_step}" "${full_json}" "dual-VM direct-grid bottleneck"
  log_timeline "END ${run_id}"
}

extract_overall() {
  "${PYTHON_BIN}" - <<'PY' "$1"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(float(data.get("overall_accuracy", 0.0)))
PY
}

RUN_A="${BUNDLE_ID}_siglip_concat_k8"
RUN_B="${BUNDLE_ID}_siglip_derived_k8"

run_siglip_variant "variant_a_siglip" "${RUN_A}" --semantic_grid_access
run_siglip_variant "variant_b_siglip" "${RUN_B}" --semantic_grid_access --semantic_query_derivation

A_ACC="$(extract_overall "${BUNDLE_DIR}/variant_a_siglip_best_full.json")"
B_ACC="$(extract_overall "${BUNDLE_DIR}/variant_b_siglip_best_full.json")"

WINNER_KEY="variant_a_siglip"
WINNER_ARGS=(--semantic_grid_access)
WINNER_ACC="${A_ACC}"
if "${PYTHON_BIN}" - <<'PY' "${A_ACC}" "${B_ACC}"
import sys
a = float(sys.argv[1]); b = float(sys.argv[2])
raise SystemExit(0 if b > a else 1)
PY
then
  WINNER_KEY="variant_b_siglip"
  WINNER_ARGS=(--semantic_grid_access --semantic_query_derivation)
  WINNER_ACC="${B_ACC}"
fi

echo "{\"variant_a_siglip\": ${A_ACC}, \"variant_b_siglip\": ${B_ACC}, \"winner\": \"${WINNER_KEY}\"}" > "${BUNDLE_DIR}/siglip_summary.json"

if "${PYTHON_BIN}" - <<'PY' "${WINNER_ACC}" "${SIGLIP_BASELINE}"
import sys
winner = float(sys.argv[1]); baseline = float(sys.argv[2])
raise SystemExit(0 if winner > baseline else 1)
PY
then
  RUN_DUAL="${BUNDLE_ID}_${WINNER_KEY/variant_/dual_}"
  run_dualvm_variant "${WINNER_KEY/variant_/variant_}_dualvm" "${RUN_DUAL}" "${WINNER_ARGS[@]}"
fi

echo "${BUNDLE_ID}"
