#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/mm_run_budget.sh"

RUN_ID="${RUN_ID:-mmarch_cov_v1_20260310_perceiver_oracle196_topk64_h1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-3}"
TARGET_STEP="${TARGET_STEP:-$(mm_budget_steps_for_bs_ga "${BATCH_SIZE}" "${GRAD_ACCUM_STEPS}")}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

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

DONE_CKPT="logs/${RUN_ID}/step_${TARGET_STEP}.tar"
if [[ -f "${DONE_CKPT}" ]]; then
  echo "[oracle-topk64] SKIP ${RUN_ID} (already complete: ${DONE_CKPT})"
  exit 0
fi

RESUME_STEP="$(latest_ckpt_step "${RUN_ID}")"
CMD=(./runmm.sh "${RUN_ID}")
if (( RESUME_STEP > 0 )); then
  CMD+=("${RESUME_STEP}")
fi
CMD+=(
  --precision bf16
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --epochs 300
  --max_steps "${TARGET_STEP}"
  --log_every 20
  --eval_every 0
  --eval_batches 0
  --eval_log_every 20
  --ckpt_every 1000
  --eval_scorer official
  --final_sanity_count 0
  --cuda_empty_cache_after_eval
  --vision_feature_source posterior_mu
  --num_visual_tokens 196
  --bridge_token_reduce adaptive_pool
  --bridge_add_2d_pos_emb
  --bridge_num_heads 8
  --bridge_type perceiver_resampler
  --bridge_query_depth 3
  --bridge_pre_mixer_type none
  --no-bridge_question_conditioning
  --bridge_token_selector_type topk
  --bridge_token_select_k 64
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

echo "[oracle-topk64] RUN_ID=${RUN_ID} TARGET_STEP=${TARGET_STEP} RESUME_STEP=${RESUME_STEP}"
echo "[oracle-topk64] CMD: ${CMD[*]}"
"${CMD[@]}"
