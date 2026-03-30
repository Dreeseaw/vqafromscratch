#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmovernight_batch_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"
TIMELINE="${BUNDLE_DIR}/timeline.log"
PROGRESS_MD="${BUNDLE_DIR}/progress.md"
REPORT_MD="${BUNDLE_DIR}/overnight_batch_report.md"

mkdir -p "${BUNDLE_DIR}"
if [[ ! -f "${TIMELINE}" ]]; then
  : > "${TIMELINE}"
fi
if [[ ! -f "${PROGRESS_MD}" ]]; then
cat > "${PROGRESS_MD}" <<'EOF'
# Overnight Progress

EOF
fi

GQA_BRIDGE_CKPT="${GQA_BRIDGE_CKPT:-logs/mmcement_gqa7030_v1_96x2/step_9000.tar}"
DUALVM_CKPT="${DUALVM_CKPT:-logs/mmdualvm_v1_20260324_rerun/step_9000.tar}"
CEMENT_ANCHOR_CKPT="${CEMENT_ANCHOR_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
SIGLIP_DIR="${SIGLIP_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"
VITSTR_CKPT="${VITSTR_CKPT:-logs/hf_vision/vitstr_tiny_patch16_224/vitstr_tiny_patch16_224_aug.pth}"
SIGLIP_REMAP_CKPT="${SIGLIP_REMAP_CKPT:-logs/mmsemantic_remap_v1_debug/step_500.tar}"
DUALVM_REMAP_CKPT="${DUALVM_REMAP_CKPT:-logs/dualvm_compressed_v1_20260325_234134_phase1_remap/step_500.tar}"
GQA_ROOT="${GQA_ROOT:-data/gqa}"

SEED="${SEED:-42}"
SAFE_TRAIN_BS="${SAFE_TRAIN_BS:-64}"
SAFE_TRAIN_GA="${SAFE_TRAIN_GA:-3}"
SAFE_EVAL_BS="${SAFE_EVAL_BS:-128}"
SAFE_WORKERS="${SAFE_WORKERS:-1}"
SAFE_PREFETCH="${SAFE_PREFETCH:-1}"
MIX_TRAIN_BS="${MIX_TRAIN_BS:-64}"
MIX_TRAIN_GA="${MIX_TRAIN_GA:-3}"
MIX_EVAL_BS="${MIX_EVAL_BS:-96}"
MIX_WORKERS="${MIX_WORKERS:-0}"
MIX_PREFETCH="${MIX_PREFETCH:-1}"
GQA_EVAL_BS="${GQA_EVAL_BS:-64}"
TRACE_EVAL_BATCHES="${TRACE_EVAL_BATCHES:-100}"
OCR_LIMIT="${OCR_LIMIT:-500}"
CTRL_LIMIT="${CTRL_LIMIT:-100}"
GQA_LIMIT_5K="${GQA_LIMIT_5K:-5000}"
PROBE_LIMIT_TRAIN="${PROBE_LIMIT_TRAIN:-9999}"
PROBE_LIMIT_VAL="${PROBE_LIMIT_VAL:-4319}"

RUN2_ID="${BUNDLE_ID}_run2_gqabridge_k8"
RUN3_ID="${BUNDLE_ID}_run3_dualvm_k8_gqa15"
RUN4_ID="${BUNDLE_ID}_run4_dualvm_k12"
RUN5_ID="${BUNDLE_ID}_run5_gqabridge_k12"

BATCH_START_TS="$(date +%s)"

log_line() {
  local line="[$(date)] $*"
  echo "${line}" | tee -a "${TIMELINE}"
}

run_stdout_log() {
  local name="$1"
  echo "${BUNDLE_DIR}/${name}.stdout.log"
}

latest_step() {
  local run_dir="$1"
  local best=0
  local path base step
  shopt -s nullglob
  for path in "${run_dir}"/step_*.tar; do
    base="$(basename "${path}")"
    step="${base#step_}"
    step="${step%.tar}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > best )); then
      best="${step}"
    fi
  done
  shopt -u nullglob
  echo "${best}"
}

vram_snapshot() {
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
}

clear_vram() {
  log_line "VRAM before clear: $(vram_snapshot | tr '\n' ' ')"
  runtime_exec_python - <<'PY'
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass
gc.collect()
PY
  sleep 2
  log_line "VRAM after clear: $(vram_snapshot | tr '\n' ' ')"
}

pick_best_step() {
  local glob_pat="$1"
  runtime_exec_python - <<'PY' "${BUNDLE_DIR}" "${glob_pat}"
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1])
glob_pat = sys.argv[2]
best_step = None
best_acc = -1.0
for path in sorted(bundle.glob(glob_pat)):
    parts = path.stem.split("_")
    step = None
    for i, part in enumerate(parts):
        if part == "step" and i + 1 < len(parts):
            try:
                step = int(parts[i + 1])
            except Exception:
                step = None
            break
    if step is None:
        continue
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    acc = float(data.get("overall_accuracy", 0.0) or 0.0)
    if acc > best_acc:
        best_acc = acc
        best_step = step
print(best_step or 3000)
PY
}

append_progress_entry() {
  local name="$1"
  local status="$2"
  local started="$3"
  local duration="$4"
  local best_step="$5"
  local json_path="$6"
  local notes="$7"
  runtime_exec_python - <<'PY' "${PROGRESS_MD}" "${name}" "${status}" "${started}" "${duration}" "${best_step}" "${json_path}" "${notes}"
import json, sys
from pathlib import Path
md = Path(sys.argv[1])
name, status, started, duration, best_step, json_path, notes = sys.argv[2:]
line = f"## {name}\n- Started: {started}\n- Status: {status}\n- Duration: {duration}\n"
if best_step:
    line += f"- Best checkpoint: step_{best_step}\n"
if json_path and Path(json_path).exists():
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    overall = float(data.get("overall_accuracy", 0.0) or 0.0)
    at = data.get("answer_type_accuracy", {}) or {}
    yn = float(at.get("yes/no", 0.0) or 0.0)
    num = float(at.get("number", 0.0) or 0.0)
    other = float(at.get("other", 0.0) or 0.0)
    line += f"- Key result: overall {overall:.4f}, y/n {yn:.4f}, num {num:.4f}, other {other:.4f}\n"
line += f"- Notes: {notes or '—'}\n\n"
with md.open("a", encoding="utf-8") as f:
    f.write(line)
PY
}

hours_since_start() {
  runtime_exec_python - <<'PY' "${BATCH_START_TS}"
import sys, time
start = float(sys.argv[1])
print((time.time() - start) / 3600.0)
PY
}

resume_runmm() {
  local run_id="$1"
  local target_step="$2"
  shift 2
  local run_dir="logs/${run_id}"
  local step
  step="$(latest_step "${run_dir}")"
  if (( step >= target_step )); then
    return 0
  fi
  if (( step > 0 )); then
    ./runmm_v1.sh "${run_id}" "${step}" "$@"
  else
    ./runmm_v1.sh "${run_id}" "$@"
  fi
}

eval_ckpt_vqa() {
  local ckpt="$1"
  local out_json="$2"
  local batch_size="$3"
  shift 3
  if [[ -f "${out_json}" ]]; then
    return 0
  fi
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${ckpt}" \
    --batch_size "${batch_size}" \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    "$@" \
    --output_json "${out_json}"
}

eval_trace_vqa() {
  local prefix="$1"
  local run_dir="$2"
  local batch_size="$3"
  shift 3
  local step ckpt out
  for step in 500 1000 1500 2000 2500 3000; do
    ckpt="${run_dir}/step_${step}.tar"
    out="${BUNDLE_DIR}/${prefix}_step_${step}_vqa.json"
    if [[ ! -f "${ckpt}" || -f "${out}" ]]; then
      continue
    fi
    eval_ckpt_vqa "${ckpt}" "${out}" "${batch_size}" --eval_batches "${TRACE_EVAL_BATCHES}" "$@"
  done
}

run_probe() {
  local ckpt="$1"
  local out_json="$2"
  local batch_size="$3"
  local epochs="${4:-1}"
  if [[ -f "${out_json}" ]]; then
    return 0
  fi
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_probe \
    --checkpoint "${ckpt}" \
    --batch_size "${batch_size}" \
    --probe_batch_size 256 \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    --limit_train "${PROBE_LIMIT_TRAIN}" \
    --limit_val "${PROBE_LIMIT_VAL}" \
    --answer_top_k 3000 \
    --epochs "${epochs}" \
    --lr 0.001 \
    --feature_pool flatten \
    --output_json "${out_json}"
}

run1_eval_gqa_bridge() {
  local stdout_log
  stdout_log="$(run_stdout_log run1_eval_gqa_bridge)"
  if [[ -f "${BUNDLE_DIR}/run1_vqa_full.json" && -f "${BUNDLE_DIR}/run1_gqa_5k.json" ]]; then
    echo "already complete"
    return 0
  fi
  log_line "START run1_eval_gqa_bridge"
  {
    eval_ckpt_vqa "${GQA_BRIDGE_CKPT}" "${BUNDLE_DIR}/run1_vqa_full.json" "${SAFE_EVAL_BS}" --eval_batches 0 --no-disable_lm_visual_adapters
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_gqa_eval \
      --checkpoint "${GQA_BRIDGE_CKPT}" \
      --batch_size "${GQA_EVAL_BS}" \
      --num_workers 0 \
      --prefetch_factor 1 \
      --no-pin_memory \
      --gqa_root "${GQA_ROOT}" \
      --limit_eval "${GQA_LIMIT_5K}" \
      --output_json "${BUNDLE_DIR}/run1_gqa_5k.json"
  } > >(tee -a "${stdout_log}") 2>&1
  local status=$?
  if (( status == 0 )); then
    log_line "END   run1_eval_gqa_bridge"
  else
    log_line "FAIL  run1_eval_gqa_bridge"
  fi
  return "${status}"
}

run2_gqabridge_k8() {
  local stdout_log
  stdout_log="$(run_stdout_log run2_gqabridge_k8)"
  local run_dir="logs/${RUN2_ID}"
  log_line "START run2_gqabridge_k8"
  {
    resume_runmm "${RUN2_ID}" 3000 \
      --vision_model siglip_base \
      --vision_checkpoint "${SIGLIP_DIR}" \
      --seed "${SEED}" \
      --max_steps 3000 \
      --manual_max_steps \
      --batch_size "${SAFE_TRAIN_BS}" \
      --grad_accum_steps "${SAFE_TRAIN_GA}" \
      --eval_batch_size "${SAFE_EVAL_BS}" \
      --num_workers "${SAFE_WORKERS}" \
      --prefetch_factor "${SAFE_PREFETCH}" \
      --no-pin_memory \
      --eval_every 500 \
      --eval_batches "${TRACE_EVAL_BATCHES}" \
      --ckpt_every 500 \
      --final_eval_batches 50 \
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
      --prefix_remap_checkpoint "${SIGLIP_REMAP_CKPT}" \
      --semantic_format_loss_weight 0.3 \
      --semantic_format_loss_final_weight 0.0 \
      --semantic_format_anneal_start_step 2250 \
      --semantic_format_anneal_end_step 3000 \
      --init_from_mm_checkpoint "${GQA_BRIDGE_CKPT}" \
      --min_train_steps_per_s 0

    eval_trace_vqa "run2" "${run_dir}" "${SAFE_EVAL_BS}" --eval_batches "${TRACE_EVAL_BATCHES}" --disable_lm_visual_adapters
    local best_step
    best_step="$(pick_best_step 'run2_step_*_vqa.json')"
    printf '{\n  "best_step": %s\n}\n' "${best_step}" > "${BUNDLE_DIR}/run2_best_step.json"
    eval_ckpt_vqa "${run_dir}/step_${best_step}.tar" "${BUNDLE_DIR}/run2_best_full.json" "${SAFE_EVAL_BS}" --eval_batches 0 --disable_lm_visual_adapters
  } > >(tee -a "${stdout_log}") 2>&1
  local status=$?
  if (( status == 0 )); then
    log_line "END   run2_gqabridge_k8"
  else
    log_line "FAIL  run2_gqabridge_k8"
  fi
  return "${status}"
}

run3_dualvm_k8_gqa15() {
  local stdout_log
  stdout_log="$(run_stdout_log run3_dualvm_k8_gqa15)"
  local run_dir="logs/${RUN3_ID}"
  log_line "START run3_dualvm_k8_gqa15"
  {
    resume_runmm "${RUN3_ID}" 3000 \
      --vision_model siglip_vitstr_tiny_dual \
      --vision_checkpoint "${SIGLIP_DIR}" \
      --vision_aux_checkpoint "${VITSTR_CKPT}" \
      --seed "${SEED}" \
      --max_steps 3000 \
      --manual_max_steps \
      --batch_size "${MIX_TRAIN_BS}" \
      --grad_accum_steps "${MIX_TRAIN_GA}" \
      --eval_batch_size "${MIX_EVAL_BS}" \
      --num_workers "${MIX_WORKERS}" \
      --prefetch_factor "${MIX_PREFETCH}" \
      --no-pin_memory \
      --eval_every 500 \
      --eval_batches "${TRACE_EVAL_BATCHES}" \
      --ckpt_every 500 \
      --final_eval_batches 50 \
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
      --prefix_remap_checkpoint "${DUALVM_REMAP_CKPT}" \
      --semantic_format_loss_weight 0.3 \
      --semantic_format_loss_final_weight 0.0 \
      --semantic_format_anneal_start_step 2250 \
      --semantic_format_anneal_end_step 3000 \
      --gqa_root "${GQA_ROOT}" \
      --gqa_train_mix_ratio 0.15625 \
      --gqa_train_fraction 0.1 \
      --init_from_mm_checkpoint "${DUALVM_CKPT}" \
      --min_train_steps_per_s 0

    eval_trace_vqa "run3" "${run_dir}" "${MIX_EVAL_BS}" --eval_batches "${TRACE_EVAL_BATCHES}" --disable_lm_visual_adapters
    local best_step
    best_step="$(pick_best_step 'run3_step_*_vqa.json')"
    printf '{\n  "best_step": %s\n}\n' "${best_step}" > "${BUNDLE_DIR}/run3_best_step.json"
    eval_ckpt_vqa "${run_dir}/step_${best_step}.tar" "${BUNDLE_DIR}/run3_best_full.json" "${MIX_EVAL_BS}" --eval_batches 0 --disable_lm_visual_adapters
  } > >(tee -a "${stdout_log}") 2>&1
  local status=$?
  if (( status == 0 )); then
    log_line "END   run3_dualvm_k8_gqa15"
  else
    log_line "FAIL  run3_dualvm_k8_gqa15"
  fi
  return "${status}"
}

run4_dualvm_k12() {
  local stdout_log
  stdout_log="$(run_stdout_log run4_dualvm_k12)"
  local run_dir="logs/${RUN4_ID}"
  log_line "START run4_dualvm_k12"
  {
    resume_runmm "${RUN4_ID}" 3000 \
      --vision_model siglip_vitstr_tiny_dual \
      --vision_checkpoint "${SIGLIP_DIR}" \
      --vision_aux_checkpoint "${VITSTR_CKPT}" \
      --seed "${SEED}" \
      --max_steps 3000 \
      --manual_max_steps \
      --batch_size "${SAFE_TRAIN_BS}" \
      --grad_accum_steps "${SAFE_TRAIN_GA}" \
      --eval_batch_size "${SAFE_EVAL_BS}" \
      --num_workers "${SAFE_WORKERS}" \
      --prefetch_factor "${SAFE_PREFETCH}" \
      --no-pin_memory \
      --eval_every 500 \
      --eval_batches "${TRACE_EVAL_BATCHES}" \
      --ckpt_every 500 \
      --final_eval_batches 50 \
      --lr 0.0002 \
      --lr_schedule cosine \
      --lr_warmup_steps 200 \
      --freeze_mode semantic_bottleneck_only \
      --semantic_bottleneck \
      --semantic_tokens 12 \
      --semantic_latent_dim 256 \
      --semantic_recon_loss_weight 0.1 \
      --semantic_consistency_loss_weight 0.0 \
      --disable_lm_visual_adapters \
      --init_from_mm_checkpoint "${DUALVM_CKPT}" \
      --min_train_steps_per_s 0

    eval_trace_vqa "run4" "${run_dir}" "${SAFE_EVAL_BS}" --eval_batches "${TRACE_EVAL_BATCHES}" --disable_lm_visual_adapters
    local best_step
    best_step="$(pick_best_step 'run4_step_*_vqa.json')"
    printf '{\n  "best_step": %s\n}\n' "${best_step}" > "${BUNDLE_DIR}/run4_best_step.json"
    eval_ckpt_vqa "${run_dir}/step_${best_step}.tar" "${BUNDLE_DIR}/run4_best_full.json" "${SAFE_EVAL_BS}" --eval_batches 0 --disable_lm_visual_adapters
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_dualvm_ocr_analysis \
      --dual_checkpoint "${run_dir}/step_${best_step}.tar" \
      --anchor_checkpoint "${CEMENT_ANCHOR_CKPT}" \
      --batch_size 96 \
      --limit_ocr "${OCR_LIMIT}" \
      --limit_control "${CTRL_LIMIT}" \
      --output_json "${BUNDLE_DIR}/run4_ocr_analysis.json"
  } > >(tee -a "${stdout_log}") 2>&1
  local status=$?
  if (( status == 0 )); then
    log_line "END   run4_dualvm_k12"
  else
    log_line "FAIL  run4_dualvm_k12"
  fi
  return "${status}"
}

run5_gqabridge_k12() {
  local stdout_log
  stdout_log="$(run_stdout_log run5_gqabridge_k12)"
  local run_dir="logs/${RUN5_ID}"
  log_line "START run5_gqabridge_k12"
  {
    resume_runmm "${RUN5_ID}" 3000 \
      --vision_model siglip_base \
      --vision_checkpoint "${SIGLIP_DIR}" \
      --seed "${SEED}" \
      --max_steps 3000 \
      --manual_max_steps \
      --batch_size "${SAFE_TRAIN_BS}" \
      --grad_accum_steps "${SAFE_TRAIN_GA}" \
      --eval_batch_size "${SAFE_EVAL_BS}" \
      --num_workers "${SAFE_WORKERS}" \
      --prefetch_factor "${SAFE_PREFETCH}" \
      --no-pin_memory \
      --eval_every 500 \
      --eval_batches "${TRACE_EVAL_BATCHES}" \
      --ckpt_every 500 \
      --final_eval_batches 50 \
      --lr 0.0002 \
      --lr_schedule cosine \
      --lr_warmup_steps 200 \
      --freeze_mode semantic_bottleneck_only \
      --semantic_bottleneck \
      --semantic_tokens 12 \
      --semantic_latent_dim 256 \
      --semantic_recon_loss_weight 0.1 \
      --semantic_consistency_loss_weight 0.0 \
      --disable_lm_visual_adapters \
      --init_from_mm_checkpoint "${GQA_BRIDGE_CKPT}" \
      --min_train_steps_per_s 0

    eval_trace_vqa "run5" "${run_dir}" "${SAFE_EVAL_BS}" --eval_batches "${TRACE_EVAL_BATCHES}" --disable_lm_visual_adapters
    local best_step
    best_step="$(pick_best_step 'run5_step_*_vqa.json')"
    printf '{\n  "best_step": %s\n}\n' "${best_step}" > "${BUNDLE_DIR}/run5_best_step.json"
    eval_ckpt_vqa "${run_dir}/step_${best_step}.tar" "${BUNDLE_DIR}/run5_best_full.json" "${SAFE_EVAL_BS}" --eval_batches 0 --disable_lm_visual_adapters
  } > >(tee -a "${stdout_log}") 2>&1
  local status=$?
  if (( status == 0 )); then
    log_line "END   run5_gqabridge_k12"
  else
    log_line "FAIL  run5_gqabridge_k12"
  fi
  return "${status}"
}

run6_probes() {
  local stdout_log
  stdout_log="$(run_stdout_log run6_probes)"
  log_line "START run6_probes"
  {
    if [[ -f "${BUNDLE_DIR}/run2_best_step.json" ]]; then
      local s2
      s2="$(runtime_exec_python - <<'PY' "${BUNDLE_DIR}/run2_best_step.json"
import json, sys
print(json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('best_step', 3000))
PY
)"
      run_probe "logs/${RUN2_ID}/step_${s2}.tar" "${BUNDLE_DIR}/run2_probe.json" 64 1
    fi
    if [[ -f "${BUNDLE_DIR}/run3_best_step.json" ]]; then
      local s3
      s3="$(runtime_exec_python - <<'PY' "${BUNDLE_DIR}/run3_best_step.json"
import json, sys
print(json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('best_step', 3000))
PY
)"
      run_probe "logs/${RUN3_ID}/step_${s3}.tar" "${BUNDLE_DIR}/run3_probe.json" 64 1
    fi
    if [[ -f "${BUNDLE_DIR}/run4_best_step.json" ]]; then
      local s4
      s4="$(runtime_exec_python - <<'PY' "${BUNDLE_DIR}/run4_best_step.json"
import json, sys
print(json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('best_step', 3000))
PY
)"
      run_probe "logs/${RUN4_ID}/step_${s4}.tar" "${BUNDLE_DIR}/run4_probe.json" 64 1
    fi
    if [[ -f "${BUNDLE_DIR}/run5_best_step.json" ]]; then
      local s5
      s5="$(runtime_exec_python - <<'PY' "${BUNDLE_DIR}/run5_best_step.json"
import json, sys
print(json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('best_step', 3000))
PY
)"
      run_probe "logs/${RUN5_ID}/step_${s5}.tar" "${BUNDLE_DIR}/run5_probe.json" 64 1
    fi
  } > >(tee -a "${stdout_log}") 2>&1
  local status=$?
  if (( status == 0 )); then
    log_line "END   run6_probes"
  else
    log_line "FAIL  run6_probes"
  fi
  return "${status}"
}

write_bundle_report() {
  runtime_exec_python - <<'PY' "${BUNDLE_DIR}" "${REPORT_MD}"
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1])
report = Path(sys.argv[2])
rows = []
for name in ["run1_vqa_full","run2_best_full","run3_best_full","run4_best_full","run5_best_full"]:
    path = bundle / f"{name}.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    at = data.get("answer_type_accuracy", {}) or {}
    rows.append((name, float(data.get("overall_accuracy", 0.0)), float(at.get("yes/no", 0.0) or 0.0), float(at.get("number", 0.0) or 0.0), float(at.get("other", 0.0) or 0.0)))
lines = ["# Overnight Batch Report", ""]
for name, overall, yn, num, other in rows:
    lines.append(f"- {name}: overall {overall:.4f}, y/n {yn:.4f}, num {num:.4f}, other {other:.4f}")
lines.append("")
lines.append(f"Progress file: `{bundle / 'progress.md'}`")
report.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(report)
PY
}

run_with_progress() {
  local name="$1"
  local func="$2"
  local best_step_json="$3"
  local result_json="$4"
  local notes="$5"
  local started_human started_ts ended_ts duration_human best_step status
  started_human="$(date -Iseconds)"
  started_ts="$(date +%s)"
  if "${func}"; then
    status="COMPLETE"
  else
    status="FAILED"
  fi
  ended_ts="$(date +%s)"
  duration_human="$(( (ended_ts-started_ts)/3600 ))h $(( ((ended_ts-started_ts)%3600)/60 ))m"
  best_step=""
  if [[ -n "${best_step_json}" && -f "${best_step_json}" ]]; then
    best_step="$(runtime_exec_python - <<'PY' "${best_step_json}"
import json, sys
print(json.load(open(sys.argv[1], 'r', encoding='utf-8')).get('best_step', ''))
PY
)"
  fi
  append_progress_entry "${name}" "${status}" "${started_human}" "${duration_human}" "${best_step}" "${result_json}" "${notes}"
  clear_vram
  return 0
}

log_line "BATCH START ${BUNDLE_ID}"
log_line "VRAM start: $(vram_snapshot | tr '\n' ' ')"

run_with_progress "Run 1: GQA-Bridge Full Eval" run1_eval_gqa_bridge "" "${BUNDLE_DIR}/run1_vqa_full.json" "Full VQAv2 + 5K GQA exact-match eval on the finished Cement+GQA bridge checkpoint."
run_with_progress "Run 2: GQA-Bridge K8 Compression" run2_gqabridge_k8 "${BUNDLE_DIR}/run2_best_step.json" "${BUNDLE_DIR}/run2_best_full.json" "SigLIP-only format-alignment compression on the GQA-trained bridge perceiver."
run_with_progress "Run 3: Dual-VM K8 GQA-Only Compression" run3_dualvm_k8_gqa15 "${BUNDLE_DIR}/run3_best_step.json" "${BUNDLE_DIR}/run3_best_full.json" "Dual-VM compression with VQAv2+GQA mix only, no grounding loss."
run_with_progress "Run 4: Dual-VM K12 Compression" run4_dualvm_k12 "${BUNDLE_DIR}/run4_best_step.json" "${BUNDLE_DIR}/run4_best_full.json" "K=12 token-budget test on the dual-VM warm-start perceiver."

ELAPSED_AFTER_RUN4="$(hours_since_start)"
RUN5_OK="$(runtime_exec_python - <<'PY' "${ELAPSED_AFTER_RUN4}"
import sys
print("1" if float(sys.argv[1]) < 6.5 else "0")
PY
)"
if [[ "${RUN5_OK}" == "1" ]]; then
  run_with_progress "Run 5: GQA-Bridge K12 Compression" run5_gqabridge_k12 "${BUNDLE_DIR}/run5_best_step.json" "${BUNDLE_DIR}/run5_best_full.json" "K=12 token-budget test on the GQA-trained bridge perceiver."
fi

ELAPSED_AFTER_RUN5="$(hours_since_start)"
RUN6_OK="$(runtime_exec_python - <<'PY' "${ELAPSED_AFTER_RUN5}"
import sys
print("1" if float(sys.argv[1]) < 7.5 else "0")
PY
)"
if [[ "${RUN6_OK}" == "1" ]]; then
  run_with_progress "Run 6: Tiny-Head Probes" run6_probes "" "" "One-epoch probes for the new compressed checkpoints."
fi

write_bundle_report >/dev/null
log_line "BATCH COMPLETE ${BUNDLE_ID}"
echo "${BUNDLE_ID}"
