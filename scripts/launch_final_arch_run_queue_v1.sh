#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/mm_run_budget.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmarch_final_queue_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmarch_final_queue_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmarch_final_v1_20260310}"
HORIZON_HOURS="${HORIZON_HOURS:-10}"
LOG_EVERY="${LOG_EVERY:-20}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DRY_RUN="${DRY_RUN:-0}"

START_TS="$(date +%s)"
HORIZON_SEC="$(( HORIZON_HOURS * 3600 ))"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Final Architecture Run Queue V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Policy:
- final eval only
- eval on half the eval split
- fixed train sample budget across runs
- restart-safe skip/resume

Runtime knobs:
- RUN_PREFIX=${RUN_PREFIX}
- HORIZON_HOURS=${HORIZON_HOURS}
- LOG_EVERY=${LOG_EVERY}
- CKPT_EVERY=${CKPT_EVERY}
- NUM_WORKERS=${NUM_WORKERS}
- PREFETCH_FACTOR=${PREFETCH_FACTOR}
EOF

COMMON_ARGS=(
  --precision bf16
  --epochs 300
  --log_every "${LOG_EVERY}"
  --eval_every 0
  --eval_batches 0
  --eval_log_every 20
  --eval_fraction 0.5
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

within_horizon() {
  local now elapsed
  now="$(date +%s)"
  elapsed="$((now - START_TS))"
  [[ "${elapsed}" -lt "${HORIZON_SEC}" ]]
}

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

run_one() {
  local suffix="$1"
  local batch_size="$2"
  local grad_accum_steps="$3"
  shift 3
  local target_step
  target_step="$(mm_budget_steps_for_bs_ga "${batch_size}" "${grad_accum_steps}")"
  local run_id="${RUN_PREFIX}_${suffix}"
  local done_ckpt="logs/${run_id}/step_${target_step}.tar"

  if ! within_horizon; then
    echo "[$(date)] STOP horizon reached before ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi

  if [[ -f "${done_ckpt}" ]]; then
    echo "[$(date)] SKIP  ${run_id} (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_ckpt_step "${run_id}")"
  if (( resume_step >= target_step )); then
    echo "[$(date)] SKIP  ${run_id} (latest checkpoint step ${resume_step} >= target ${target_step})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local cmd=(./runmm.sh "${run_id}")
  if (( resume_step > 0 )); then
    cmd+=("${resume_step}")
  fi
  cmd+=(
    "${COMMON_ARGS[@]}"
    --batch_size "${batch_size}"
    --grad_accum_steps "${grad_accum_steps}"
    --max_steps "${target_step}"
    "$@"
  )

  echo "[$(date)] START ${run_id} target_step=${target_step} resume_step=${resume_step} bs=${batch_size} ga=${grad_accum_steps}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
    echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  echo "[$(date)] FAIL  ${run_id} (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
  return 1
}

run_one "safeqcond_d3_main" 192 1 \
  --bridge_question_conditioning \
  --bridge_question_context_mode prompt_only \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

run_one "multiscale_d3_main" 128 2 \
  --bridge_type multiscale_perceiver \
  --vision_feature_source encoder_plus_posterior_mu \
  --bridge_token_reduce adaptive_pool || true

run_one "earlylayer_encoder_d3_main" 192 1 \
  --vision_feature_source encoder \
  --bridge_token_reduce adaptive_pool \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

run_one "oracle196_topk64_main" 64 3 \
  --num_visual_tokens 196 \
  --bridge_token_reduce adaptive_pool \
  --bridge_token_selector_type topk \
  --bridge_token_select_k 64 || true

run_one "geomcal_d3_main" 192 1 \
  --prefix_geom_mlp_ratio 0.5 \
  --prefix_geom_token_mixer_layers 1 || true

run_one "topk32_d3_main" 192 1 \
  --bridge_token_selector_type topk \
  --bridge_token_select_k 32 || true

run_one "structuredroles_d3_exp" 192 1 \
  --bridge_type structured_roles \
  --bridge_num_roles 4 || true

run_one "evidencesparse_d3_exp" 192 1 \
  --bridge_type evidence_sparse \
  --bridge_evidence_topk 24 || true

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
