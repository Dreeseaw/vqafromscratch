#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmdynbudget_clean9k_overnight_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"
TIMELINE="${BUNDLE_DIR}/timeline.log"
PROGRESS_MD="${BUNDLE_DIR}/progress.md"
LATEST_LINK="logs/mmdynbudget_clean9k_overnight_latest"
DB_LOG="${BUNDLE_DIR}/db_build.log"

TRAIN_RUN_ID="${TRAIN_RUN_ID:-${BUNDLE_ID}_frontier24_train}"
NEW_FIXED_RUN_ID="${NEW_FIXED_RUN_ID:-${BUNDLE_ID}_frontier24_fixed_eval}"
NEW_SCHED_RUN_ID="${NEW_SCHED_RUN_ID:-${BUNDLE_ID}_frontier24_scheduler}"
BASELINE_FIXED_RUN_ID="${BASELINE_FIXED_RUN_ID:-${BUNDLE_ID}_baseline416_fixed_eval}"
BASELINE_SCHED_RUN_ID="${BASELINE_SCHED_RUN_ID:-${BUNDLE_ID}_baseline416_scheduler}"

SOURCE_CKPT="${SOURCE_CKPT:-logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar}"
BASELINE_VARPREFIX_CKPT="${BASELINE_VARPREFIX_CKPT:-logs/mmsemantic_varbudget_v1_20260330_180744_varprefix_k16/step_3000.tar}"
LM_CKPT="${LM_CKPT:-logs/lm_final/step_45000.tar}"
SIGLIP2_DIR="${SIGLIP2_DIR:-logs/hf_vision/openclip_siglip2_b16_webli}"
REMAP_CKPT="${REMAP_CKPT:-logs/mmsemantic_remap_v1_debug/step_500.tar}"

SEED="${SEED:-35}"
TRAIN_BUDGETS="${TRAIN_BUDGETS:-2,4,8,16}"
EVAL_BUDGETS="${EVAL_BUDGETS:-2,4,8,16}"
PROBE_BUDGETS="${PROBE_BUDGETS:-1}"
STATS_BUDGETS="${STATS_BUDGETS:-2,4}"

TRAIN_BS="${TRAIN_BS:-96}"
TRAIN_GA="${TRAIN_GA:-2}"
TRAIN_EVAL_BS="${TRAIN_EVAL_BS:-96}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
TRAIN_PREFETCH="${TRAIN_PREFETCH:-1}"
TRAIN_PIN_MEMORY="${TRAIN_PIN_MEMORY:-0}"

POSTHOC_EVAL_BS="${POSTHOC_EVAL_BS:-128}"
POSTHOC_NUM_WORKERS="${POSTHOC_NUM_WORKERS:-2}"
POSTHOC_PREFETCH="${POSTHOC_PREFETCH:-1}"
POSTHOC_PIN_MEMORY="${POSTHOC_PIN_MEMORY:-0}"

mkdir -p "${BUNDLE_DIR}"
: > "${TIMELINE}"
if [[ ! -f "${PROGRESS_MD}" ]]; then
cat > "${PROGRESS_MD}" <<'EOF'
# Dynamic Budget Overnight Bundle Progress

EOF
fi
ln -sfn "${BUNDLE_ID}" "${LATEST_LINK}"

log_line() {
  local line="[$(date)] $*"
  echo "${line}" | tee -a "${TIMELINE}"
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

rebuild_db() {
  local reason="$1"
  {
    echo "[$(date)] rebuild_db reason=${reason}"
    python3 scripts/build_experiment_db.py --task mm_bridge
  } >> "${DB_LOG}" 2>&1 || true
}

schedule_delayed_rebuild() {
  local reason="$1"
  local delay_s="${2:-30}"
  (
    sleep "${delay_s}"
    cd "${REPO_ROOT}"
    {
      echo "[$(date)] rebuild_db reason=${reason}"
      python3 scripts/build_experiment_db.py --task mm_bridge
    } >> "${DB_LOG}" 2>&1 || true
  ) >/dev/null 2>&1 &
}

prepare_run_dir() {
  local run_id="$1"
  local eval_only="${2:-0}"
  local run_dir="logs/${run_id}"
  local log_path="${run_dir}/logfile.txt"
  mkdir -p "${run_dir}"
  touch "${log_path}"
  if [[ "${eval_only}" == "1" ]]; then
    printf '[launcher] eval_only=1 bundle=%s run=%s\n' "${BUNDLE_ID}" "${run_id}" >> "${log_path}"
  else
    printf '[launcher] bundle=%s run=%s\n' "${BUNDLE_ID}" "${run_id}" >> "${log_path}"
  fi
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
  local started_h started_ts end_ts duration status=0
  local run_dir="logs/${TRAIN_RUN_ID}"
  prepare_run_dir "${TRAIN_RUN_ID}" 0
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  log_line "START ${TRAIN_RUN_ID} phase=frontier24_train"
  rebuild_db "train_start"
  schedule_delayed_rebuild "train_start_delayed" 45
  resume_runmm "${TRAIN_RUN_ID}" 3000 \
    --vision_model siglip2_b16 \
    --vision_checkpoint "${SIGLIP2_DIR}" \
    --lm_checkpoint "${LM_CKPT}" \
    --seed "${SEED}" \
    --max_steps 3000 \
    --manual_max_steps \
    --batch_size "${TRAIN_BS}" \
    --grad_accum_steps "${TRAIN_GA}" \
    --eval_batch_size "${TRAIN_EVAL_BS}" \
    --num_workers "${TRAIN_NUM_WORKERS}" \
    --prefetch_factor "${TRAIN_PREFETCH}" \
    "$(pin_memory_flag "${TRAIN_PIN_MEMORY}")" \
    --log_every 20 \
    --eval_every 0 \
    --ckpt_every 500 \
    --final_eval_batches 1 \
    --lr 0.0002 \
    --lr_schedule cosine \
    --lr_warmup_steps 200 \
    --lr_min_ratio 0.15 \
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
    --semantic_budget_options "${TRAIN_BUDGETS}" \
    --semantic_budget_sample_mode batch \
    --init_from_mm_checkpoint "${SOURCE_CKPT}" \
    --eval_use_kv_cache \
    --eval_kv_cache_mode batched \
    --min_train_steps_per_s 0 || status=$?
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END   ${TRAIN_RUN_ID} status=0 phase=frontier24_train"
    append_progress_entry "Train clean-9k frontier variable-prefix compressor" "COMPLETE" "${started_h}" "${duration}" \
      "Source=${SOURCE_CKPT}; budgets=${TRAIN_BUDGETS}; checkpoint=${run_dir}/step_3000.tar."
  else
    log_line "FAIL  ${TRAIN_RUN_ID} status=${status} phase=frontier24_train"
    append_progress_entry "Train clean-9k frontier variable-prefix compressor" "FAILED" "${started_h}" "${duration}" \
      "Training failed or stopped early."
  fi
  rebuild_db "train_end"
  return "${status}"
}

run_eval_suite_phase() {
  local title="$1"
  local run_id="$2"
  local checkpoint="$3"
  local output_subdir="$4"
  local started_h started_ts end_ts duration status=0
  local eval_dir="${BUNDLE_DIR}/${output_subdir}"
  local run_dir="logs/${run_id}"
  local run_log="${run_dir}/logfile.txt"
  prepare_run_dir "${run_id}" 1
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  log_line "START ${run_id} phase=${output_subdir}"
  rebuild_db "${output_subdir}_start"
  schedule_delayed_rebuild "${output_subdir}_start_delayed" 15
  {
    printf '[eval_suite] eval_only=1 bundle=%s run=%s checkpoint=%s budgets=%s probe=%s stats=%s\n' \
      "${BUNDLE_ID}" "${run_id}" "${checkpoint}" "${EVAL_BUDGETS}" "${PROBE_BUDGETS}" "${STATS_BUDGETS}"
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_budget_eval_suite \
      --checkpoint "${checkpoint}" \
      --budgets "${EVAL_BUDGETS}" \
      --probe_budgets "${PROBE_BUDGETS}" \
      --stats_budgets "${STATS_BUDGETS}" \
      --batch_size "${POSTHOC_EVAL_BS}" \
      --num_workers "${POSTHOC_NUM_WORKERS}" \
      --prefetch_factor "${POSTHOC_PREFETCH}" \
      "$(pin_memory_flag "${POSTHOC_PIN_MEMORY}")" \
      --disable_lm_visual_adapters \
      --eval_use_kv_cache \
      --eval_kv_cache_mode batched \
      --output_dir "${eval_dir}"
  } 2>&1 | tee -a "${run_log}" || status=$?
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END   ${run_id} status=0 phase=${output_subdir}"
    append_progress_entry "${title}" "COMPLETE" "${started_h}" "${duration}" \
      "Artifacts=${eval_dir}/summary.json."
  else
    log_line "FAIL  ${run_id} status=${status} phase=${output_subdir}"
    append_progress_entry "${title}" "FAILED" "${started_h}" "${duration}" \
      "Eval suite failed."
  fi
  rebuild_db "${output_subdir}_end"
  return "${status}"
}

run_scheduler_phase() {
  local title="$1"
  local run_id="$2"
  local eval_dir="$3"
  local started_h started_ts end_ts duration status=0
  local run_dir="logs/${run_id}"
  local run_log="${run_dir}/logfile.txt"
  local output_json="${BUNDLE_DIR}/$(basename "${eval_dir}")_scheduler_summary.json"
  prepare_run_dir "${run_id}" 1
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  log_line "START ${run_id} phase=$(basename "${eval_dir}")_scheduler"
  rebuild_db "$(basename "${eval_dir}")_scheduler_start"
  schedule_delayed_rebuild "$(basename "${eval_dir}")_scheduler_start_delayed" 15
  {
    printf '[scheduler_run] eval_only=1 bundle=%s run=%s eval_dir=%s\n' "${BUNDLE_ID}" "${run_id}" "${eval_dir}"
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_budget_scheduler_sweep \
      --eval_dir "${eval_dir}" \
      --main_base_budget 4 \
      --side_base_budget 2 \
      --output_json "${output_json}"
  } 2>&1 | tee -a "${run_log}" || status=$?
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END   ${run_id} status=0 phase=$(basename "${eval_dir}")_scheduler"
    append_progress_entry "${title}" "COMPLETE" "${started_h}" "${duration}" \
      "Artifacts=${output_json}."
  else
    log_line "FAIL  ${run_id} status=${status} phase=$(basename "${eval_dir}")_scheduler"
    append_progress_entry "${title}" "FAILED" "${started_h}" "${duration}" \
      "Scheduler sweep failed."
  fi
  rebuild_db "$(basename "${eval_dir}")_scheduler_end"
  return "${status}"
}

main() {
  log_line "BUNDLE ${BUNDLE_ID} START"
  log_line "source_ckpt=${SOURCE_CKPT}"
  log_line "baseline_varprefix_ckpt=${BASELINE_VARPREFIX_CKPT}"
  log_line "exec_env: $(runtime_log_env_summary "$(runtime_resolve_mode)")"
  rebuild_db "bundle_start"

  local new_ckpt="logs/${TRAIN_RUN_ID}/step_3000.tar"
  local new_eval_summary="${BUNDLE_DIR}/frontier24_eval/summary.json"
  local new_sched_summary="${BUNDLE_DIR}/frontier24_eval_scheduler_summary.json"
  local baseline_eval_summary="${BUNDLE_DIR}/baseline416_eval/summary.json"
  local baseline_sched_summary="${BUNDLE_DIR}/baseline416_eval_scheduler_summary.json"

  if [[ -f "${new_ckpt}" ]]; then
    log_line "SKIP  ${TRAIN_RUN_ID} status=existing_ckpt phase=frontier24_train checkpoint=${new_ckpt}"
  else
    clear_vram
    run_train_phase || true
  fi

  if [[ -f "${new_ckpt}" ]]; then
    if [[ -f "${new_eval_summary}" ]]; then
      log_line "SKIP  ${NEW_FIXED_RUN_ID} status=existing_summary phase=frontier24_eval summary=${new_eval_summary}"
    else
      clear_vram
      run_eval_suite_phase "Fixed-K plus oracle eval on clean-9k {2,4,8,16} variable-prefix checkpoint" "${NEW_FIXED_RUN_ID}" "${new_ckpt}" "frontier24_eval" || true
    fi
    if [[ -f "${new_sched_summary}" ]]; then
      log_line "SKIP  ${NEW_SCHED_RUN_ID} status=existing_summary phase=frontier24_scheduler summary=${new_sched_summary}"
    else
      clear_vram
      run_scheduler_phase "Scheduler sweep on clean-9k {2,4,8,16} variable-prefix checkpoint" "${NEW_SCHED_RUN_ID}" "${BUNDLE_DIR}/frontier24_eval" || true
    fi
  else
    log_line "FAIL  ${NEW_FIXED_RUN_ID} status=missing_ckpt phase=frontier24_eval"
    log_line "FAIL  ${NEW_SCHED_RUN_ID} status=missing_ckpt phase=frontier24_scheduler"
  fi

  if [[ -f "${baseline_eval_summary}" ]]; then
    log_line "SKIP  ${BASELINE_FIXED_RUN_ID} status=existing_summary phase=baseline416_eval summary=${baseline_eval_summary}"
  else
    clear_vram
    run_eval_suite_phase "Fixed-K plus oracle eval on baseline clean-9k {4,8,12,16} variable-prefix checkpoint" "${BASELINE_FIXED_RUN_ID}" "${BASELINE_VARPREFIX_CKPT}" "baseline416_eval" || true
  fi

  if [[ -f "${baseline_sched_summary}" ]]; then
    log_line "SKIP  ${BASELINE_SCHED_RUN_ID} status=existing_summary phase=baseline416_scheduler summary=${baseline_sched_summary}"
  else
    clear_vram
    run_scheduler_phase "Scheduler sweep on baseline clean-9k {4,8,12,16} variable-prefix checkpoint" "${BASELINE_SCHED_RUN_ID}" "${BUNDLE_DIR}/baseline416_eval" || true
  fi

  clear_vram
  rebuild_db "bundle_end"
  log_line "BUNDLE ${BUNDLE_ID} COMPLETE"
}

main "$@"
