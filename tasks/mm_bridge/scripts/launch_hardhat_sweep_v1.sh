#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmhardhat_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmhardhat_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmhardhat_v1_$(date +%Y%m%d)}"
TARGET_STEP="${TARGET_STEP:-9000}"
TARGET_STEP_18K="${TARGET_STEP_18K:-18000}"

# DINOv2-S: b96a2 (confirmed Crane probes)
DINOV2S_BS="${DINOV2S_BS:-96}"
DINOV2S_GA="${DINOV2S_GA:-2}"
DINOV2S_EVAL_BS="${DINOV2S_EVAL_BS:-96}"

# SigLIP-B: b192a1 train, b96 eval (confirmed Hardhat probes)
SIGLIP_BS="${SIGLIP_BS:-192}"
SIGLIP_GA="${SIGLIP_GA:-1}"
SIGLIP_EVAL_BS="${SIGLIP_EVAL_BS:-96}"

# Caption-align
CAPALIGN_BS="${CAPALIGN_BS:-96}"
CAPALIGN_STEPS="${CAPALIGN_STEPS:-3000}"

# Model directories
DINOV2S_MODEL_DIR="${DINOV2S_MODEL_DIR:-logs/hf_vision/facebook_dinov2_small}"
SIGLIP_MODEL_DIR="${SIGLIP_MODEL_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"

LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-0}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DRY_RUN="${DRY_RUN:-0}"
MIN_TRAIN_SPS="${MIN_TRAIN_SPS:-1.0}"
MIN_TRAIN_WINDOW="${MIN_TRAIN_WINDOW:-100}"
LOW_TRAIN_SPS_EXIT_CODE="${LOW_TRAIN_SPS_EXIT_CODE:-86}"
MAX_LOW_SPS_RESTARTS="${MAX_LOW_SPS_RESTARTS:-8}"

# Skip controls
SKIP_TIER1="${SKIP_TIER1:-0}"
SKIP_TIER2="${SKIP_TIER2:-0}"
SKIP_TIER3="${SKIP_TIER3:-0}"
SKIP_TIER4="${SKIP_TIER4:-0}"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Hardhat Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Plan: tasks/mm_bridge/docs/37claude_hardhat_sweep_plan_2026-03-15.md

Phase 1 layout:
  DINOv2-S: ${DINOV2S_BS}x${DINOV2S_GA} train / ${DINOV2S_EVAL_BS} eval
  SigLIP-B: ${SIGLIP_BS}x${SIGLIP_GA} train / ${SIGLIP_EVAL_BS} eval

Target step: ${TARGET_STEP} (${TARGET_STEP_18K} for 18k runs)

Skip controls: SKIP_TIER1=${SKIP_TIER1} SKIP_TIER2=${SKIP_TIER2} SKIP_TIER3=${SKIP_TIER3} SKIP_TIER4=${SKIP_TIER4}
EOF

# ============================================================================
# Common args (identical to Crane)
# ============================================================================

COMMON_ARGS=(
  --precision bf16
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
  --eval_use_kv_cache
  --eval_kv_cache_mode batched
  --vision_feature_source encoder
  --num_visual_tokens 49
  --bridge_token_reduce adaptive_pool
  --bridge_add_2d_pos_emb
  --bridge_num_heads 8
  --bridge_type perceiver_resampler
  --bridge_query_depth 3
  --bridge_pre_mixer_type none
  --bridge_question_conditioning
  --bridge_question_context_mode prompt_only
  --prefix_calibration
  --prefix_calib_layernorm
  --prefix_calib_bias
  --prefix_calib_gate_init 1.0
  --prefix_geom_mlp_ratio 0.5
  --prefix_geom_token_mixer_layers 1
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
  --min_train_steps_per_s "${MIN_TRAIN_SPS}"
  --min_train_steps_window "${MIN_TRAIN_WINDOW}"
)

# Hardhat default: nodynbudget + attnqquery + adapter d3
NODYN_ATTNQ_ADAPTER_ARGS=(
  --bridge_query_bank_mode question_hidden_attn
  --bridge_qquery_scale 1.0
  --bridge_token_selector_type none
  --bridge_token_select_k 0
  --lm_visual_adapter_type cross_attn
  --lm_visual_adapter_layers 3
  --lm_visual_adapter_num_heads 8
  --lm_visual_adapter_dropout 0.0
  --lm_visual_adapter_gate_init 0.5
)

DINOV2S_ARGS=(
  --vision_model dinov2_small
  --vision_checkpoint "${DINOV2S_MODEL_DIR}"
  --vision_feature_mode auto
  --batch_size "${DINOV2S_BS}"
  --grad_accum_steps "${DINOV2S_GA}"
  --eval_batch_size "${DINOV2S_EVAL_BS}"
)

SIGLIP_ARGS=(
  --vision_model siglip_base
  --vision_checkpoint "${SIGLIP_MODEL_DIR}"
  --vision_feature_mode auto
  --batch_size "${SIGLIP_BS}"
  --grad_accum_steps "${SIGLIP_GA}"
  --eval_batch_size "${SIGLIP_EVAL_BS}"
)

# ============================================================================
# Utility functions (same as Crane launcher)
# ============================================================================

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
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( 10#${step} > max_step )); then
      max_step=$((10#${step}))
    fi
  done
  shopt -u nullglob
  echo "${max_step}"
}

has_completed_eval() {
  local run_id="$1"
  local target="${2:-${TARGET_STEP}}"
  local answers_path="logs/${run_id}/fixed_eval_val_answers.jsonl"
  local pattern="\"global_step\": ${target}.*\"tag\": \"(final_eval|eval_only)\""
  if [[ ! -f "${answers_path}" ]]; then
    return 1
  fi
  grep -Eq "${pattern}" "${answers_path}"
}

run_one() {
  local suffix="$1"
  local target_step="$2"
  shift 2

  local run_id="${RUN_PREFIX}_${suffix}"
  local done_ckpt="logs/${run_id}/step_${target_step}.tar"

  if [[ -f "${done_ckpt}" ]] && has_completed_eval "${run_id}" "${target_step}"; then
    echo "[$(date)] SKIP  ${run_id} (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_ckpt_step "${run_id}")"

  local restart_count=0
  while true; do
    local cmd=(./runmm.sh "${run_id}")
    if [[ -f "${done_ckpt}" ]] || (( resume_step >= target_step )); then
      cmd+=("${target_step}")
      cmd+=(
        "${COMMON_ARGS[@]}"
        --max_steps "${target_step}"
        "$@"
        --eval_only
        --eval_batches 0
        --eval_fraction 1.0
        --eval_log_every 20
        --eval_scorer official
        --final_sanity_count 0
        --cuda_empty_cache_after_eval
      )
      echo "[$(date)] RESUME-EVAL ${run_id} from step_${target_step}.tar" | tee -a "${SWEEP_DIR}/timeline.log"
    else
      if (( resume_step > 0 )); then
        cmd+=("${resume_step}")
      fi
      cmd+=(
        "${COMMON_ARGS[@]}"
        --max_steps "${target_step}"
        "$@"
      )
    fi

    echo "[$(date)] START ${run_id} target=${target_step} resume=${resume_step} restarts=${restart_count}" | tee -a "${SWEEP_DIR}/timeline.log"
    echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
      return 0
    fi

    if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
      echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
      return 0
    fi

    local status=$?
    if (( status == LOW_TRAIN_SPS_EXIT_CODE )); then
      restart_count=$((restart_count + 1))
      if (( restart_count > MAX_LOW_SPS_RESTARTS )); then
        echo "[$(date)] FAIL  ${run_id} exceeded low-sps restart budget" | tee -a "${SWEEP_DIR}/timeline.log"
        return "${status}"
      fi
      resume_step="$(latest_ckpt_step "${run_id}")"
      if (( resume_step <= 0 )); then
        echo "[$(date)] FAIL  ${run_id} low-sps exit without checkpoint" | tee -a "${SWEEP_DIR}/timeline.log"
        return "${status}"
      fi
      echo "[$(date)] RESTART ${run_id} low-sps exit resume=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
      continue
    fi

    echo "[$(date)] FAIL  ${run_id} (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
    return "${status}"
  done
}

run_capalign() {
  # NOTE: stdout is captured — log messages go to stderr.
  local suffix="$1"
  shift

  local run_id="${RUN_PREFIX}_capalign_${suffix}"
  local done_ckpt="logs/${run_id}/step_$(printf '%06d' "${CAPALIGN_STEPS}").tar"

  if [[ -f "${done_ckpt}" ]]; then
    echo "[$(date)] SKIP  ${run_id} (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log" >&2
    echo "${done_ckpt}"
    return 0
  fi

  local cmd=(
    ./runcapalign.sh "${run_id}"
    --batch_size "${CAPALIGN_BS}"
    --max_steps "${CAPALIGN_STEPS}"
    --precision bf16
    --bridge_type perceiver_resampler
    --num_visual_tokens 49
    --bridge_token_reduce adaptive_pool
    --bridge_add_2d_pos_emb
    --bridge_num_heads 8
    --bridge_query_depth 3
    --bridge_pre_mixer_type none
    --prefix_calibration
    --prefix_calib_layernorm
    --prefix_calib_bias
    --prefix_calib_gate_init 1.0
    --prefix_geom_mlp_ratio 0.5
    --prefix_geom_token_mixer_layers 1
    "$@"
  )

  echo "[$(date)] START caption-align ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log" >&2
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log" >&2
    echo "DRYRUN"
    return 0
  fi

  if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
    echo "[$(date)] END   caption-align ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log" >&2
    echo "${done_ckpt}"
    return 0
  fi

  echo "[$(date)] FAIL  caption-align ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log" >&2
  return 1
}

run_twostage() {
  local suffix="$1"
  local target_step="$2"
  local capalign_ckpt="$3"
  shift 3

  local run_id="${RUN_PREFIX}_${suffix}"

  if [[ "${capalign_ckpt}" == "DRYRUN" ]]; then
    echo "[$(date)] DRY_RUN skip two-stage ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  if [[ ! -f "${capalign_ckpt}" ]]; then
    echo "[$(date)] FAIL  two-stage ${run_id}: caption-align ckpt not found: ${capalign_ckpt}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi

  local capalign_step
  capalign_step="$(echo "${capalign_ckpt}" | grep -oP 'step_\K[0-9]+')"
  capalign_step=$((10#${capalign_step}))  # strip leading zeros

  local done_ckpt="logs/${run_id}/step_${target_step}.tar"
  if [[ -f "${done_ckpt}" ]] && has_completed_eval "${run_id}" "${target_step}"; then
    echo "[$(date)] SKIP  ${run_id} (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_ckpt_step "${run_id}")"

  local restart_count=0
  while true; do
    local cmd=(./runmm.sh "${run_id}")
    if [[ -f "${done_ckpt}" ]] || (( resume_step >= target_step )); then
      cmd+=("${target_step}")
      cmd+=(
        "${COMMON_ARGS[@]}"
        --max_steps "${target_step}"
        "$@"
        --eval_only
        --eval_batches 0
        --eval_fraction 1.0
        --eval_log_every 20
        --eval_scorer official
        --final_sanity_count 0
        --cuda_empty_cache_after_eval
      )
    elif (( resume_step > 0 )); then
      # Resuming mid-VQA training — schedule already reset on first launch
      cmd+=("${resume_step}")
      cmd+=(
        "${COMMON_ARGS[@]}"
        --max_steps "${target_step}"
        "$@"
      )
    else
      # First launch: load caption-align weights, reset schedule for clean 9k VQA
      mkdir -p "logs/${run_id}"
      if [[ ! -f "logs/${run_id}/step_${capalign_step}.tar" ]]; then
        cp "${capalign_ckpt}" "logs/${run_id}/step_${capalign_step}.tar"
      fi
      cmd+=("${capalign_step}")
      cmd+=(
        "${COMMON_ARGS[@]}"
        --max_steps "${target_step}"
        --reset_schedule
        "$@"
      )
    fi

    echo "[$(date)] START two-stage ${run_id} target=${target_step} resume=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
    echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
      return 0
    fi

    if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
      echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
      return 0
    fi

    local status=$?
    if (( status == LOW_TRAIN_SPS_EXIT_CODE )); then
      restart_count=$((restart_count + 1))
      if (( restart_count > MAX_LOW_SPS_RESTARTS )); then
        echo "[$(date)] FAIL  ${run_id} exceeded low-sps restart budget" | tee -a "${SWEEP_DIR}/timeline.log"
        return "${status}"
      fi
      resume_step="$(latest_ckpt_step "${run_id}")"
      echo "[$(date)] RESTART ${run_id} low-sps exit resume=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
      continue
    fi

    echo "[$(date)] FAIL  ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return "${status}"
  done
}


# ============================================================================
# PHASE 1 — All non-conditional runs
# ============================================================================

# ============================================================================
# TIER 1: DINOv2-S Nodynbudget Solidification (4 runs, ~2.8h)
# ============================================================================

if [[ "${SKIP_TIER1}" != "1" ]]; then
  echo "[$(date)] === TIER 1: DINOv2-S Nodynbudget Solidification ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 1: seed2 of 0.5762 frontier
  run_one "dinov2s_attnqquery_nodynbudget_adapter_d3_seed2" "${TARGET_STEP}" \
    "${DINOV2S_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    --seed 53

  # Run 2: questiononly
  run_one "dinov2s_questiononly_attnqquery_nodynbudget_adapter_d3" "${TARGET_STEP}" \
    "${DINOV2S_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    --bridge_question_context_mode question_only

  # Run 3: d4 adapters
  run_one "dinov2s_attnqquery_nodynbudget_adapter_d4" "${TARGET_STEP}" \
    "${DINOV2S_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    --lm_visual_adapter_layers 4

  # Run 4: perceiver query_depth=4
  run_one "dinov2s_attnqquery_nodynbudget_adapter_d3_qdepth4" "${TARGET_STEP}" \
    "${DINOV2S_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    --bridge_query_depth 4
fi


# ============================================================================
# TIER 2: Longer Training (1-2 runs, ~1.3-2.6h)
# ============================================================================

if [[ "${SKIP_TIER2}" != "1" ]]; then
  echo "[$(date)] === TIER 2: Longer Training ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 5: 18k baseline
  run_one "dinov2s_attnqquery_nodynbudget_adapter_d3_18k" "${TARGET_STEP_18K}" \
    "${DINOV2S_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    --lr_warmup_steps 1200

  # Run 6 is CONDITIONAL on Tier 1 — deferred to Phase 2 launcher
fi


# ============================================================================
# TIER 3: Corrected Caption-Align (1 run, ~0.9h)
# ============================================================================

if [[ "${SKIP_TIER3}" != "1" ]]; then
  echo "[$(date)] === TIER 3: Corrected Caption-Align ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Caption-align stage (DINOv2-S, ~7 min)
  capalign_ckpt_dinov2s="$(run_capalign "dinov2s_nodyn" \
    --vision_model dinov2_small \
    --vision_checkpoint "${DINOV2S_MODEL_DIR}" \
    --vision_feature_source encoder \
    --vision_feature_mode auto)"

  # Run 7: VQA with --reset_schedule (clean 9k from pre-trained bridge)
  run_twostage "dinov2s_captionalign_attnqquery_nodynbudget_adapter_d3" "${TARGET_STEP}" \
    "${capalign_ckpt_dinov2s}" \
    "${DINOV2S_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}"
fi


# ============================================================================
# TIER 4: SigLIP-B/16 Baseline (1 run, ~0.7h)
# ============================================================================

if [[ "${SKIP_TIER4}" != "1" ]]; then
  echo "[$(date)] === TIER 4: SigLIP-B/16 ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 8: SigLIP-B nodynbudget baseline
  run_one "siglip_attnqquery_nodynbudget_adapter_d3" "${TARGET_STEP}" \
    "${SIGLIP_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}"
fi


echo "[$(date)] PHASE 1 COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
echo ""
echo "============================================================================"
echo "Phase 1 done. Review Tier 1-4 results, then update and run Phase 2:"
echo "  tasks/mm_bridge/scripts/launch_hardhat_sweep_v1_phase2.sh"
echo "============================================================================"
