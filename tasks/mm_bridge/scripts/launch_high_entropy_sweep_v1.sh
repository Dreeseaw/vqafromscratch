#!/bin/bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmarch_high_entropy_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmarch_high_entropy_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmarch_high_entropy_v1_20260311}"
TARGET_STEP="${TARGET_STEP:-9000}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-0}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DRY_RUN="${DRY_RUN:-0}"

if (( BATCH_SIZE * GRAD_ACCUM_STEPS != 192 )); then
  echo "[high-entropy] ERROR effective batch must be 192, got $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
  exit 1
fi

cat > "${SWEEP_DIR}/README.md" <<EOF
# High-Entropy Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Final draft order source:
- tasks/mm_bridge/docs/20_high_entropy_sweep_plan_2026-03-11.md

Comparison policy:
- effective batch size = 192
- eval_every = 1000
- eval_batches = 100
- final_eval_batches = 0 (full final eval)
- eval_fraction = 1.0
- target_step = ${TARGET_STEP}

Runtime knobs:
- RUN_PREFIX=${RUN_PREFIX}
- BATCH_SIZE=${BATCH_SIZE}
- GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
- LOG_EVERY=${LOG_EVERY}
- EVAL_EVERY=${EVAL_EVERY}
- EVAL_BATCHES=${EVAL_BATCHES}
- FINAL_EVAL_BATCHES=${FINAL_EVAL_BATCHES}
- CKPT_EVERY=${CKPT_EVERY}
- NUM_WORKERS=${NUM_WORKERS}
- PREFETCH_FACTOR=${PREFETCH_FACTOR}

Restart behavior:
- complete runs are skipped only if both step_${TARGET_STEP}.tar and a completed final-eval marker exist
- partial runs resume from latest available checkpoint
- restart with the same RUN_PREFIX to continue the sweep
EOF

COMMON_ARGS=(
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
  --final_eval_batches "${FINAL_EVAL_BATCHES}"
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

run_one() {
  local suffix="$1"
  shift
  local run_id="${RUN_PREFIX}_${suffix}"
  local done_ckpt="logs/${run_id}/step_${TARGET_STEP}.tar"

  if [[ -f "${done_ckpt}" ]] && has_completed_eval "${run_id}"; then
    echo "[$(date)] SKIP  ${run_id} (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_ckpt_step "${run_id}")"

  local cmd=(./runmm.sh "${run_id}")
  if [[ -f "${done_ckpt}" ]] || (( resume_step >= TARGET_STEP )); then
    cmd+=("${TARGET_STEP}")
    cmd+=(
      "${COMMON_ARGS[@]}"
      "$@"
      --eval_only
      --eval_batches 0
      --eval_fraction 1.0
      --eval_log_every 20
      --eval_scorer official
      --final_sanity_count 0
      --cuda_empty_cache_after_eval
    )
    echo "[$(date)] RESUME-EVAL ${run_id} from step_${TARGET_STEP}.tar" | tee -a "${SWEEP_DIR}/timeline.log"
  else
    if (( resume_step > 0 )); then
      cmd+=("${resume_step}")
    fi
    cmd+=("${COMMON_ARGS[@]}" "$@")
  fi

  echo "[$(date)] START ${run_id} target_step=${TARGET_STEP} resume_step=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
    echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  else
    local status=$?
    echo "[$(date)] FAIL  ${run_id} (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
    return "${status}"
  fi
}

# 1) safe qcond frontier harden
run_one "safeqcond_frontier" \
  --bridge_question_conditioning \
  --bridge_question_context_mode prompt_only \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0

# 2) structured roles frontier harden
run_one "structuredroles_frontier" \
  --bridge_type structured_roles \
  --bridge_num_roles 4

# 3) early-layer encoder frontier harden
run_one "earlylayer_encoder_frontier" \
  --vision_feature_source encoder \
  --bridge_token_reduce adaptive_pool \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0

# 4) safe qcond + early-layer encoder
run_one "safeqcond_earlylayer_frontier" \
  --vision_feature_source encoder \
  --bridge_token_reduce adaptive_pool \
  --bridge_question_conditioning \
  --bridge_question_context_mode prompt_only \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0

# 5) safe qcond + geometry-aware calibration
run_one "safeqcond_geomcal_frontier" \
  --bridge_question_conditioning \
  --bridge_question_context_mode prompt_only \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 \
  --prefix_geom_mlp_ratio 0.5 \
  --prefix_geom_token_mixer_layers 1

# 6) structured roles + geometry-aware calibration
run_one "structuredroles_geomcal_frontier" \
  --bridge_type structured_roles \
  --bridge_num_roles 4 \
  --prefix_geom_mlp_ratio 0.5 \
  --prefix_geom_token_mixer_layers 1

# 7) safe qcond + early-layer encoder + geometry-aware calibration
run_one "safeqcond_earlylayer_geomcal_frontier" \
  --vision_feature_source encoder \
  --bridge_token_reduce adaptive_pool \
  --bridge_question_conditioning \
  --bridge_question_context_mode prompt_only \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0 \
  --prefix_geom_mlp_ratio 0.5 \
  --prefix_geom_token_mixer_layers 1

# 8) safe qcond + multiscale perceiver
run_one "safeqcond_multiscale_frontier" \
  --vision_feature_source encoder_plus_posterior_mu \
  --bridge_type multiscale_perceiver \
  --bridge_token_reduce adaptive_pool \
  --bridge_question_conditioning \
  --bridge_question_context_mode prompt_only \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0

# 9) safe qcond + hybrid constant/image bridge
run_one "safeqcond_hybrid_tok075_frontier" \
  --bridge_type hybrid_const_image \
  --bridge_hybrid_image_bridge_type perceiver_resampler \
  --bridge_hybrid_alpha_mode token \
  --bridge_hybrid_alpha_init 0.75 \
  --bridge_question_conditioning \
  --bridge_question_context_mode prompt_only \
  --bridge_token_selector_type none \
  --bridge_token_select_k 0

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
