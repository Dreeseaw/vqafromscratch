#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmsemantic_varbudget_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"
TIMELINE="${BUNDLE_DIR}/timeline.log"
PROGRESS_MD="${BUNDLE_DIR}/progress.md"
LATEST_LINK="logs/mmsemantic_varbudget_v1_latest"

TRAIN_RUN_ID="${TRAIN_RUN_ID:-${BUNDLE_ID}_varprefix_k16}"
EVAL_RUN_ID="${EVAL_RUN_ID:-${BUNDLE_ID}_oracle_eval}"
SOURCE_CKPT="${SOURCE_CKPT:-logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar}"
LM_CKPT="${LM_CKPT:-logs/lm_final/step_45000.tar}"
SIGLIP2_DIR="${SIGLIP2_DIR:-logs/hf_vision/openclip_siglip2_b16_webli}"
REMAP_CKPT="${REMAP_CKPT:-logs/mmsemantic_remap_v1_debug/step_500.tar}"

SEED="${SEED:-35}"
BUDGETS="${BUDGETS:-4,8,12,16}"
TRAIN_BS="${TRAIN_BS:-96}"
TRAIN_GA="${TRAIN_GA:-2}"
EVAL_BS="${EVAL_BS:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH="${PREFETCH:-1}"
PIN_MEMORY="${PIN_MEMORY:-0}"

mkdir -p "${BUNDLE_DIR}"
if [[ ! -f "${TIMELINE}" ]]; then
  : > "${TIMELINE}"
fi
if [[ ! -f "${PROGRESS_MD}" ]]; then
cat > "${PROGRESS_MD}" <<'EOF'
# Variable-K Semantic Budgeting Progress

EOF
fi
ln -sfn "${BUNDLE_ID}" "${LATEST_LINK}"

log_line() {
  local line="[$(date)] $*"
  echo "${line}" | tee -a "${TIMELINE}"
}

run_stdout_log() {
  local name="$1"
  echo "${BUNDLE_DIR}/${name}.stdout.log"
}

format_duration() {
  runtime_exec_python - <<'PY' "$1" "$2"
import sys
start = float(sys.argv[1]); end = float(sys.argv[2])
delta = max(0, int(round(end - start)))
h = delta // 3600
m = (delta % 3600) // 60
print(f"{h}h {m}m")
PY
}

append_progress_entry() {
  local title="$1"
  local status="$2"
  local started="$3"
  local duration="$4"
  local notes="$5"
  runtime_exec_python - <<'PY' "${PROGRESS_MD}" "${title}" "${status}" "${started}" "${duration}" "${notes}"
import sys
from pathlib import Path
md = Path(sys.argv[1])
title, status, started, duration, notes = sys.argv[2:]
with md.open("a", encoding="utf-8") as f:
    f.write(f"## {title}\n")
    f.write(f"- Started: {started}\n")
    f.write(f"- Status: {status}\n")
    f.write(f"- Duration: {duration}\n")
    f.write(f"- Notes: {notes}\n\n")
PY
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

pin_memory_flag() {
  local enabled="${1:-0}"
  if [[ "${enabled}" == "1" ]]; then
    printf '%s\n' "--pin_memory"
  else
    printf '%s\n' "--no-pin_memory"
  fi
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

clear_vram() {
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
}

run_train_phase() {
  local started_h started_ts end_ts duration stdout_log status=0
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log train)"
  log_line "START ${TRAIN_RUN_ID} phase=train"
  resume_runmm "${TRAIN_RUN_ID}" 3000 \
    --vision_model siglip2_b16 \
    --vision_checkpoint "${SIGLIP2_DIR}" \
    --lm_checkpoint "${LM_CKPT}" \
    --seed "${SEED}" \
    --max_steps 3000 \
    --manual_max_steps \
    --batch_size "${TRAIN_BS}" \
    --grad_accum_steps "${TRAIN_GA}" \
    --eval_batch_size "${EVAL_BS}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH}" \
    "$(pin_memory_flag "${PIN_MEMORY}")" \
    --log_every 20 \
    --eval_every 0 \
    --ckpt_every 500 \
    --final_eval_batches 1 \
    --lr 0.0002 \
    --lr_schedule cosine \
    --lr_warmup_steps 200 \
    --freeze_mode semantic_bottleneck_only \
    --bridge_question_context_mode question_only \
    --bridge_query_bank_mode question_hidden_attn \
    --semantic_bottleneck \
    --semantic_tokens 16 \
    --semantic_latent_dim 256 \
    --semantic_recon_loss_weight 0.1 \
    --semantic_consistency_loss_weight 0.0 \
    --disable_lm_visual_adapters \
    --prefix_remap_present \
    --no-apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${REMAP_CKPT}" \
    --semantic_format_loss_weight 0.3 \
    --semantic_format_loss_final_weight 0.0 \
    --semantic_format_anneal_start_step 2250 \
    --semantic_format_anneal_end_step 3000 \
    --semantic_budget_options "${BUDGETS}" \
    --semantic_budget_sample_mode batch \
    --init_from_mm_checkpoint "${SOURCE_CKPT}" \
    --min_train_steps_per_s 0 2>&1 | tee -a "${stdout_log}" || status=$?
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END   ${TRAIN_RUN_ID} status=0 phase=train"
    append_progress_entry "Train variable-K compression" "COMPLETE" "${started_h}" "${duration}" \
      "Fresh-bridge source compressed with max K=16 and sampled budgets ${BUDGETS}."
  else
    log_line "END   ${TRAIN_RUN_ID} status=${status} phase=train"
    append_progress_entry "Train variable-K compression" "FAILED" "${started_h}" "${duration}" \
      "Training failed or was partial."
  fi
  return "${status}"
}

run_eval_phase() {
  local started_h started_ts end_ts duration stdout_log status=0 final_ckpt output_json eval_run_dir eval_log
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log oracle_eval)"
  final_ckpt="logs/${TRAIN_RUN_ID}/step_3000.tar"
  output_json="${BUNDLE_DIR}/oracle_eval_summary.json"
  eval_run_dir="logs/${EVAL_RUN_ID}"
  eval_log="${eval_run_dir}/logfile.txt"
  mkdir -p "${eval_run_dir}"
  if [[ ! -f "${final_ckpt}" ]]; then
    log_line "FAIL  ${EVAL_RUN_ID} status=missing_ckpt phase=oracle_eval"
    log_line "ORACLE_EVAL_END status=missing_ckpt"
    append_progress_entry "Oracle eval" "FAILED" "${started_h}" "0h 0m" "Missing checkpoint ${final_ckpt}."
    return 1
  fi
  log_line "START ${EVAL_RUN_ID} phase=oracle_eval"
  log_line "ORACLE_EVAL_BEGIN checkpoint=${final_ckpt}"
  {
    printf '[oracle_eval] eval_only=1 bundle=%s checkpoint=%s budgets=%s\n' "${BUNDLE_ID}" "${final_ckpt}" "${BUDGETS}"
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_budget_oracle_eval \
      --checkpoint "${final_ckpt}" \
      --budgets "${BUDGETS}" \
      --batch_size "${EVAL_BS}" \
      --num_workers "${NUM_WORKERS}" \
      --prefetch_factor "${PREFETCH}" \
      "$(pin_memory_flag "${PIN_MEMORY}")" \
      --disable_lm_visual_adapters \
      --output_json "${output_json}"
  } 2>&1 | tee -a "${stdout_log}" "${eval_log}" || status=$?
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END   ${EVAL_RUN_ID} status=0 phase=oracle_eval"
    log_line "ORACLE_EVAL_END status=0"
    append_progress_entry "Oracle eval" "COMPLETE" "${started_h}" "${duration}" \
      "Fixed-K and oracle eval completed for budgets ${BUDGETS}; summary at ${output_json}."
  else
    log_line "FAIL  ${EVAL_RUN_ID} status=${status} phase=oracle_eval"
    log_line "ORACLE_EVAL_END status=${status}"
    append_progress_entry "Oracle eval" "FAILED" "${started_h}" "${duration}" \
      "Oracle eval failed."
  fi
  return "${status}"
}

main() {
  log_line "BUNDLE ${BUNDLE_ID} START"
  log_line "source_ckpt=${SOURCE_CKPT}"
  log_line "exec_env: $(runtime_log_env_summary "$(runtime_resolve_mode)")"
  clear_vram
  run_train_phase || true
  clear_vram
  run_eval_phase || true
  clear_vram
  log_line "BUNDLE ${BUNDLE_ID} COMPLETE"
}

main "$@"
