#!/bin/bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmarch_coverage_big_v1_6h_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmarch_coverage_big_v1_6h_latest"

HORIZON_HOURS="${HORIZON_HOURS:-6}"
RUN_PREFIX="${RUN_PREFIX:-mmarch_cov_v1_20260310}"
MAX_STEPS_MAIN="${MAX_STEPS_MAIN:-7000}"
MAX_STEPS_EXP="${MAX_STEPS_EXP:-5000}"
MAX_STEPS_HEAVY="${MAX_STEPS_HEAVY:-4000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-160}"
LOG_EVERY="${LOG_EVERY:-20}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DRY_RUN="${DRY_RUN:-0}"

START_TS="$(date +%s)"
HORIZON_SEC="$(( HORIZON_HOURS * 3600 ))"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Architecture Coverage Big Sweep V1 (6h Horizon)

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Key goals:
- Question-conditioned perceiver bridge coverage
- Early-layer VM feature-source coverage
- Large-token oracle coverage
- Adaptive token-selection coverage

Runtime knobs:
- RUN_PREFIX=${RUN_PREFIX}
- HORIZON_HOURS=${HORIZON_HOURS}
- MAX_STEPS_MAIN=${MAX_STEPS_MAIN}
- MAX_STEPS_EXP=${MAX_STEPS_EXP}
- MAX_STEPS_HEAVY=${MAX_STEPS_HEAVY}
- EVAL_EVERY=${EVAL_EVERY}
- EVAL_BATCHES=${EVAL_BATCHES}
- BATCH_SIZE=${BATCH_SIZE}
- GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
- NUM_WORKERS=${NUM_WORKERS}
- PREFETCH_FACTOR=${PREFETCH_FACTOR}

Restart/skip behavior:
- A run is treated as complete when \`logs/<run_id>/step_<target>.tar\` exists.
- Complete runs are skipped automatically.
- Partial runs auto-resume from latest available checkpoint step.
- Keep RUN_PREFIX fixed across restarts to preserve resume/skip behavior.
EOF

COMMON_ARGS=(
  --precision bf16
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --epochs 300
  --log_every "${LOG_EVERY}"
  --eval_every "${EVAL_EVERY}"
  --eval_batches "${EVAL_BATCHES}"
  --eval_log_every 20
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
  local target_step="$2"
  shift 2
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
  cmd+=("${COMMON_ARGS[@]}" --max_steps "${target_step}" "$@")

  echo "[$(date)] START ${run_id} target_step=${target_step} resume_step=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
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

# 0) Anchor (non-qcond perceiver) for same-day comparability.
run_one "perceiver_d3_anchor" "${MAX_STEPS_MAIN}" \
  --no-bridge_question_conditioning \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

# 1) Question-conditioned perceiver (core architecture thread).
run_one "perceiver_d3_qcond" "${MAX_STEPS_MAIN}" \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.50 \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

# 2) Question-conditioned perceiver + early-layer feature source.
run_one "perceiver_d3_qcond_encoder" "${MAX_STEPS_MAIN}" \
  --vision_feature_source encoder \
  --bridge_token_reduce adaptive_pool \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.50 \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

# 3) Adaptive token selection (top-k) without q-conditioning.
run_one "perceiver_d3_topk24" "${MAX_STEPS_EXP}" \
  --no-bridge_question_conditioning \
  --bridge_token_selector_type topk \
  --bridge_token_select_k 24 || true

# 4) Adaptive token selection + question conditioning.
run_one "perceiver_d3_qcond_topk24" "${MAX_STEPS_EXP}" \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.50 \
  --bridge_token_selector_type topk \
  --bridge_token_select_k 24 || true

# 5) Adaptive token selection + qcond + early-layer features.
run_one "perceiver_d3_qcond_topk24_encoder" "${MAX_STEPS_EXP}" \
  --vision_feature_source encoder \
  --bridge_token_reduce adaptive_pool \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.50 \
  --bridge_token_selector_type topk \
  --bridge_token_select_k 24 || true

# 6) Large-token oracle (196 tokens) with conservative batching.
run_one "perceiver_oracle196" "${MAX_STEPS_HEAVY}" \
  --num_visual_tokens 196 \
  --bridge_token_reduce adaptive_pool \
  --batch_size 96 \
  --grad_accum_steps 2 \
  --no-bridge_question_conditioning \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

# 7) Large-token oracle + q-conditioning.
run_one "perceiver_oracle196_qcond" "${MAX_STEPS_HEAVY}" \
  --num_visual_tokens 196 \
  --bridge_token_reduce adaptive_pool \
  --batch_size 96 \
  --grad_accum_steps 2 \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.50 \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

# 8) Large-token oracle + q-conditioning + early-layer features.
run_one "perceiver_oracle196_qcond_encoder" "${MAX_STEPS_HEAVY}" \
  --vision_feature_source encoder \
  --num_visual_tokens 196 \
  --bridge_token_reduce adaptive_pool \
  --batch_size 96 \
  --grad_accum_steps 2 \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.50 \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

# 9) Perceiver capacity check around qcond thread.
run_one "perceiver_d4_qcond" "${MAX_STEPS_EXP}" \
  --bridge_query_depth 4 \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.50 \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 || true

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
