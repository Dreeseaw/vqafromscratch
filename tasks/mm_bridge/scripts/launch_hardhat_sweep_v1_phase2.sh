#!/bin/bash
set -euo pipefail
# ============================================================================
# Hardhat Sweep V1 — Phase 2 (Conditional Runs)
#
# This launcher is a STUB. Update the placeholder args below after reviewing
# Phase 1 results (Tiers 1-4). Then run it.
#
# Decision gates to resolve before running:
#
#   TIER 1 RESULTS → did any of {questiononly, d4, qdepth4} beat 0.5762?
#     YES → set BEST_T1_DELTA below to the winning flag(s)
#     NO  → leave BEST_T1_DELTA empty, Run 6 and Run 9 will be skipped
#
#   TIER 2 RESULTS → did 18k show continued slope (delta > 0.003)?
#     YES → Run 11 should use 18k; set USE_18K=1
#     NO  → Run 11 uses 9k; leave USE_18K=0
#
#   TIER 3 RESULTS → did corrected caption-align beat the frontier?
#     YES → Run 11 should include caption-align; set USE_CAPALIGN=1
#     NO  → leave USE_CAPALIGN=0
#
#   TIER 4 RESULTS → SigLIP-B vs DINOv2-S frontier?
#     SigLIP wins    → BEST_VM=siglip, build Eng-3 (DINOv2-B) for Run 10
#     Within ±0.01   → BEST_VM=siglip, build Eng-3 for Run 10
#     DINOv2-S wins  → BEST_VM=dinov2s, skip Run 10
#     → also set BEST_VM_ARGS and BEST_VM_RUN_NAME below
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# --- FILL THESE IN AFTER PHASE 1 ---

# Best Tier 1 delta(s). Leave empty array if Tier 1 was flat.
# Examples:
#   BEST_T1_DELTA=(--bridge_question_context_mode question_only)
#   BEST_T1_DELTA=(--lm_visual_adapter_layers 4)
#   BEST_T1_DELTA=(--bridge_query_depth 4)
#   BEST_T1_DELTA=(--bridge_question_context_mode question_only --bridge_query_depth 4)
BEST_T1_DELTA=()
BEST_T1_SUFFIX=""  # e.g. "questiononly" or "d4_qdepth4"

# 18k decision
USE_18K="${USE_18K:-0}"

# Caption-align decision
USE_CAPALIGN="${USE_CAPALIGN:-0}"

# Best VM decision
# Options: "dinov2s" or "siglip"
BEST_VM="${BEST_VM:-dinov2s}"

# Whether to run DINOv2-B capacity-matched comparison (Tier 5)
RUN_DINOV2B="${RUN_DINOV2B:-0}"
DINOV2B_MODEL_DIR="${DINOV2B_MODEL_DIR:-logs/hf_vision/facebook_dinov2_base}"
DINOV2B_BS="${DINOV2B_BS:-96}"
DINOV2B_GA="${DINOV2B_GA:-2}"
DINOV2B_EVAL_BS="${DINOV2B_EVAL_BS:-96}"

# --- END USER CONFIG ---

RUN_PREFIX="${RUN_PREFIX:-mmhardhat_v1_$(date +%Y%m%d)}"
TARGET_STEP="${TARGET_STEP:-9000}"
TARGET_STEP_18K="${TARGET_STEP_18K:-18000}"
DRY_RUN="${DRY_RUN:-0}"

DINOV2S_BS="${DINOV2S_BS:-96}"
DINOV2S_GA="${DINOV2S_GA:-2}"
DINOV2S_EVAL_BS="${DINOV2S_EVAL_BS:-96}"
SIGLIP_BS="${SIGLIP_BS:-192}"
SIGLIP_GA="${SIGLIP_GA:-1}"
SIGLIP_EVAL_BS="${SIGLIP_EVAL_BS:-96}"
CAPALIGN_BS="${CAPALIGN_BS:-96}"
CAPALIGN_STEPS="${CAPALIGN_STEPS:-3000}"
DINOV2S_MODEL_DIR="${DINOV2S_MODEL_DIR:-logs/hf_vision/facebook_dinov2_small}"
SIGLIP_MODEL_DIR="${SIGLIP_MODEL_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"

NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-0}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
MIN_TRAIN_SPS="${MIN_TRAIN_SPS:-1.0}"
MIN_TRAIN_WINDOW="${MIN_TRAIN_WINDOW:-100}"
LOW_TRAIN_SPS_EXIT_CODE="${LOW_TRAIN_SPS_EXIT_CODE:-86}"
MAX_LOW_SPS_RESTARTS="${MAX_LOW_SPS_RESTARTS:-8}"

# Reuse Phase 1 sweep dir
SWEEP_DIR="$(readlink -f logs/mmhardhat_v1_latest 2>/dev/null || echo "logs/mmhardhat_v1_phase2_$(date +%Y%m%d_%H%M%S)")"
mkdir -pv "${SWEEP_DIR}"

echo "[$(date)] === PHASE 2 START ===" | tee -a "${SWEEP_DIR}/timeline.log"

# ---- Redefine same infra as Phase 1 ----

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

DINOV2B_ARGS=(
  --vision_model dinov2_small  # TODO: update to dinov2_base after Eng-3
  --vision_checkpoint "${DINOV2B_MODEL_DIR}"
  --vision_feature_mode auto
  --batch_size "${DINOV2B_BS}"
  --grad_accum_steps "${DINOV2B_GA}"
  --eval_batch_size "${DINOV2B_EVAL_BS}"
)

# ---- Utility functions (copied from Phase 1) ----

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
    echo "[$(date)] FAIL  two-stage ${run_id}: ckpt not found: ${capalign_ckpt}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi

  local capalign_step
  capalign_step="$(echo "${capalign_ckpt}" | grep -oP 'step_\K[0-9]+')"
  capalign_step=$((10#${capalign_step}))

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
      cmd+=("${resume_step}")
      cmd+=(
        "${COMMON_ARGS[@]}"
        --max_steps "${target_step}"
        "$@"
      )
    else
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
# TIER 2 CONDITIONAL: Run 6 — best_config_18k
# ============================================================================

if [[ ${#BEST_T1_DELTA[@]} -gt 0 && -n "${BEST_T1_SUFFIX}" ]]; then
  echo "[$(date)] === TIER 2 (conditional): best_config_18k ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 6: Stack Tier 1 winner(s) with 18k
  run_one "dinov2s_${BEST_T1_SUFFIX}_attnqquery_nodynbudget_adapter_18k" "${TARGET_STEP_18K}" \
    "${DINOV2S_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    "${BEST_T1_DELTA[@]}" \
    --lr_warmup_steps 1200
else
  echo "[$(date)] SKIP Run 6: Tier 1 was flat, no best_config_18k needed" | tee -a "${SWEEP_DIR}/timeline.log"
fi


# ============================================================================
# TIER 4 CONDITIONAL: Run 9 — SigLIP with best bridge config
# ============================================================================

if [[ ${#BEST_T1_DELTA[@]} -gt 0 && -n "${BEST_T1_SUFFIX}" ]]; then
  echo "[$(date)] === TIER 4 (conditional): SigLIP + best bridge ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 9: SigLIP-B with Tier 1 winning delta(s)
  run_one "siglip_${BEST_T1_SUFFIX}_attnqquery_nodynbudget_adapter" "${TARGET_STEP}" \
    "${SIGLIP_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    "${BEST_T1_DELTA[@]}"
else
  echo "[$(date)] SKIP Run 9: Tier 1 was flat, no SigLIP bridge transfer test" | tee -a "${SWEEP_DIR}/timeline.log"
fi


# ============================================================================
# TIER 5: DINOv2-B Capacity-Matched Comparison (conditional)
# ============================================================================

if [[ "${RUN_DINOV2B}" == "1" ]]; then
  echo "[$(date)] === TIER 5: DINOv2-B Capacity Match ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 10: DINOv2-B nodynbudget
  run_one "dinov2b_attnqquery_nodynbudget_adapter_d3" "${TARGET_STEP}" \
    "${DINOV2B_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}"
else
  echo "[$(date)] SKIP Tier 5: DINOv2-B not requested" | tee -a "${SWEEP_DIR}/timeline.log"
fi


# ============================================================================
# TIER 6: Final Frontier
# ============================================================================

echo "[$(date)] === TIER 6: Final Frontier ===" | tee -a "${SWEEP_DIR}/timeline.log"

# Determine best VM args for the max-out run
if [[ "${BEST_VM}" == "siglip" ]]; then
  FINAL_VM_ARGS=("${SIGLIP_ARGS[@]}")
  FINAL_VM_TAG="siglip"
else
  FINAL_VM_ARGS=("${DINOV2S_ARGS[@]}")
  FINAL_VM_TAG="dinov2s"
fi

# Build suffix
FINAL_SUFFIX="${FINAL_VM_TAG}"
FINAL_EXTRA_ARGS=()

if [[ ${#BEST_T1_DELTA[@]} -gt 0 && -n "${BEST_T1_SUFFIX}" ]]; then
  FINAL_SUFFIX="${FINAL_SUFFIX}_${BEST_T1_SUFFIX}"
  FINAL_EXTRA_ARGS+=("${BEST_T1_DELTA[@]}")
fi

FINAL_TARGET="${TARGET_STEP}"
if [[ "${USE_18K}" == "1" ]]; then
  FINAL_SUFFIX="${FINAL_SUFFIX}_18k"
  FINAL_TARGET="${TARGET_STEP_18K}"
  FINAL_EXTRA_ARGS+=(--lr_warmup_steps 1200)
fi

if [[ "${USE_CAPALIGN}" == "1" ]]; then
  FINAL_SUFFIX="${FINAL_SUFFIX}_captionalign"

  # Caption-align for best VM
  if [[ "${BEST_VM}" == "siglip" ]]; then
    capalign_ckpt_final="$(run_capalign "siglip_final" \
      --vision_model siglip_base \
      --vision_checkpoint "${SIGLIP_MODEL_DIR}" \
      --vision_feature_source encoder \
      --vision_feature_mode auto)"
  else
    # Reuse the DINOv2-S caption-align from Tier 3 if it exists
    capalign_ckpt_final="$(run_capalign "dinov2s_nodyn" \
      --vision_model dinov2_small \
      --vision_checkpoint "${DINOV2S_MODEL_DIR}" \
      --vision_feature_source encoder \
      --vision_feature_mode auto)"
  fi

  # Run 11: max-out with caption-align
  run_twostage "${FINAL_SUFFIX}_attnqquery_nodynbudget_adapter_maxout" "${FINAL_TARGET}" \
    "${capalign_ckpt_final}" \
    "${FINAL_VM_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    "${FINAL_EXTRA_ARGS[@]}"
else
  # Run 11: max-out without caption-align
  run_one "${FINAL_SUFFIX}_attnqquery_nodynbudget_adapter_maxout" "${FINAL_TARGET}" \
    "${FINAL_VM_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    "${FINAL_EXTRA_ARGS[@]}"
fi

# Run 12: seed2 of the max-out run
if [[ "${USE_CAPALIGN}" == "1" ]]; then
  run_twostage "${FINAL_SUFFIX}_attnqquery_nodynbudget_adapter_maxout_seed2" "${FINAL_TARGET}" \
    "${capalign_ckpt_final}" \
    "${FINAL_VM_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    "${FINAL_EXTRA_ARGS[@]}" \
    --seed 53
else
  run_one "${FINAL_SUFFIX}_attnqquery_nodynbudget_adapter_maxout_seed2" "${FINAL_TARGET}" \
    "${FINAL_VM_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
    "${FINAL_EXTRA_ARGS[@]}" \
    --seed 53
fi

echo "[$(date)] PHASE 2 COMPLETE" | tee -a "${SWEEP_DIR}/timeline.log"
