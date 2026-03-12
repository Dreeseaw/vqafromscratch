#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-mmarch_safeqcond_frontier_v1_20260311}"
TARGET_STEP="${TARGET_STEP:-9000}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
DRY_RUN="${DRY_RUN:-0}"

latest_ckpt_step() {
  local run_id="$1"
  local run_dir="logs/${run_id}"
  local max_step=0
  local f base step
  shopt -s nullglob
  for f in "${run_dir}"/step_*.tar; do
    base="${f##*/}"
    step="${base#step_}"
    step="${step%.tar}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > max_step )); then
      max_step="${step}"
    fi
  done
  shopt -u nullglob
  echo "${max_step}"
}

has_completed_eval() {
  local run_id="$1"
  local answers_path="logs/${run_id}/fixed_eval_val_answers.jsonl"
  local pattern="\"global_step\": ${TARGET_STEP}.*\"tag\": \"(final_eval|eval_only)\""
  if [[ ! -f "${answers_path}" ]]; then
    return 1
  fi
  if command -v rg >/dev/null 2>&1; then
    rg -q "${pattern}" "${answers_path}"
  else
    grep -Eq "${pattern}" "${answers_path}"
  fi
}

DONE_CKPT="logs/${RUN_ID}/step_${TARGET_STEP}.tar"
if [[ -f "${DONE_CKPT}" ]] && has_completed_eval "${RUN_ID}"; then
  echo "[safeqcond-frontier] SKIP ${RUN_ID} (already complete: ${DONE_CKPT})"
  exit 0
fi

RESUME_STEP="$(latest_ckpt_step "${RUN_ID}")"
CMD=(./runmm.sh "${RUN_ID}")
if [[ -f "${DONE_CKPT}" ]] || (( RESUME_STEP >= TARGET_STEP )); then
  CMD+=("${TARGET_STEP}")
  CMD+=(
    --precision bf16
    --batch_size "${BATCH_SIZE}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --epochs 400
    --max_steps "${TARGET_STEP}"
    --manual_max_steps
    --log_every "${LOG_EVERY}"
    --eval_every "${EVAL_EVERY}"
    --eval_log_every 20
    --ckpt_every "${CKPT_EVERY}"
    --vision_feature_source posterior_mu
    --num_visual_tokens 49
    --bridge_token_reduce all
    --bridge_add_2d_pos_emb
    --bridge_num_heads 8
    --bridge_type perceiver_resampler
    --bridge_query_depth 3
    --bridge_pre_mixer_type none
    --bridge_question_conditioning
    --bridge_question_context_mode prompt_only
    --bridge_token_selector_type none
    --bridge_token_select_k 0
    --prefix_calibration
    --prefix_calib_layernorm
    --prefix_calib_bias
    --prefix_calib_gate_init 1.0
    --prefix_norm_target_ratio 4.0
    --prefix_norm_reg_weight 0.005
    --prefix_batchvar_reg_weight 0.0002
    --prefix_dropout 0.03
    --freeze_mode bridge_plus_top_lm
    --train_top_lm_layers 2
    --lr 0.0002
    --lr_schedule cosine
    --lr_warmup_steps 600
    --lr_min_ratio 0.15
    --eval_only
    --eval_batches 0
    --eval_fraction 1.0
    --eval_log_every 20
    --eval_scorer official
    --final_sanity_count 0
    --cuda_empty_cache_after_eval
  )
else
  if (( RESUME_STEP > 0 )); then
    CMD+=("${RESUME_STEP}")
  fi
  CMD+=(
    --precision bf16
    --batch_size "${BATCH_SIZE}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --epochs 400
    --max_steps "${TARGET_STEP}"
    --manual_max_steps
    --log_every "${LOG_EVERY}"
    --eval_every "${EVAL_EVERY}"
    --eval_batches "${EVAL_BATCHES}"
    --eval_log_every 20
    --eval_fraction 1.0
    --ckpt_every "${CKPT_EVERY}"
    --eval_scorer official
    --final_sanity_count 0
    --cuda_empty_cache_after_eval
    --vision_feature_source posterior_mu
    --num_visual_tokens 49
    --bridge_token_reduce all
    --bridge_add_2d_pos_emb
    --bridge_num_heads 8
    --bridge_type perceiver_resampler
    --bridge_query_depth 3
    --bridge_pre_mixer_type none
    --bridge_question_conditioning
    --bridge_question_context_mode prompt_only
    --bridge_token_selector_type none
    --bridge_token_select_k 0
    --prefix_calibration
    --prefix_calib_layernorm
    --prefix_calib_bias
    --prefix_calib_gate_init 1.0
    --prefix_norm_target_ratio 4.0
    --prefix_norm_reg_weight 0.005
    --prefix_batchvar_reg_weight 0.0002
    --prefix_dropout 0.03
    --freeze_mode bridge_plus_top_lm
    --train_top_lm_layers 2
    --lr 0.0002
    --lr_schedule cosine
    --lr_warmup_steps 600
    --lr_min_ratio 0.15
  )
fi

echo "[safeqcond-frontier] RUN_ID=${RUN_ID} TARGET_STEP=${TARGET_STEP} RESUME_STEP=${RESUME_STEP}"
echo "[safeqcond-frontier] BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "[safeqcond-frontier] CMD: ${CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if "${CMD[@]}"; then
  exit 0
fi

status=$?
exit "${status}"
