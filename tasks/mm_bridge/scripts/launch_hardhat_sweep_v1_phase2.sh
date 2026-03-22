#!/bin/bash
set -euo pipefail
# ============================================================================
# Hardhat Sweep V1 — Phase 2 (Revised)
#
# Phase 1 results (2026-03-15 / 2026-03-16):
#
#   T1 seed2:           0.5658 (seed variance ~0.01)
#   T1 questiononly:     0.5803 ← WINNER (+0.004)
#   T1 d4:              0.5508 (hurts)
#   T1 qdepth4:          0.5762 (flat)
#   T2 18k:              0.5911 (+0.015 over 9k, still climbing)
#   T3 caption-align:    0.5421 (dead)
#   T4 SigLIP-B:         0.6095 (+0.033 over DINOv2-S frontier)
#
# Phase 2 plan (3 training runs + 2 eval-only passes):
#
#   Run 11: siglip_questiononly_18k           (max-out)
#           → full eval at 9k for legacy comparison
#   Run 12: siglip_questiononly_18k_seed2     (frontier stability)
#           → full eval at 9k for seed variance
#   Run 10: dinov2b_nodynbudget_d3 @ 9k      (capacity-matched comparison)
#
# SigLIP-B at +3.3% over DINOv2-S could be capacity (86M vs 22M), not
# alignment. DINOv2-B (86M, 256 tok, 768d) vs SigLIP-B (86M, 196 tok, 768d)
# isolates pre-training objective at matched capacity.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_PREFIX="${RUN_PREFIX:-mmhardhat_v1_20260315}"
TARGET_STEP="${TARGET_STEP:-9000}"
TARGET_STEP_18K="${TARGET_STEP_18K:-18000}"
DRY_RUN="${DRY_RUN:-0}"

# SigLIP-B: b192a1 train, b96 eval (confirmed Hardhat probes)
SIGLIP_BS="${SIGLIP_BS:-96}"
SIGLIP_GA="${SIGLIP_GA:-2}"
SIGLIP_EVAL_BS="${SIGLIP_EVAL_BS:-96}"
SIGLIP_MODEL_DIR="${SIGLIP_MODEL_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"

# DINOv2-B: b192a1 train, b96 eval (confirmed 2.89 sps @ b192a1)
DINOV2B_BS="${DINOV2B_BS:-96}"
DINOV2B_GA="${DINOV2B_GA:-2}"
DINOV2B_EVAL_BS="${DINOV2B_EVAL_BS:-96}"
DINOV2B_MODEL_DIR="${DINOV2B_MODEL_DIR:-logs/hf_vision/facebook_dinov2_base}"

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

# ============================================================================
# Common args
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

SIGLIP_ARGS=(
  --vision_model siglip_base
  --vision_checkpoint "${SIGLIP_MODEL_DIR}"
  --vision_feature_mode auto
  --batch_size "${SIGLIP_BS}"
  --grad_accum_steps "${SIGLIP_GA}"
  --eval_batch_size "${SIGLIP_EVAL_BS}"
)

DINOV2B_ARGS=(
  --vision_model dinov2_base
  --vision_checkpoint "${DINOV2B_MODEL_DIR}"
  --vision_feature_mode auto
  --batch_size "${DINOV2B_BS}"
  --grad_accum_steps "${DINOV2B_GA}"
  --eval_batch_size "${DINOV2B_EVAL_BS}"
)

# ============================================================================
# Utility functions
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

: <<'COMMENT'
restart with just dinov2-b run cus claude is fucking dumb

# ============================================================================
# Run 11: SigLIP + questiononly @ 18k (max-out)
# ============================================================================

echo "[$(date)] === Run 11: SigLIP questiononly 18k ===" | tee -a "${SWEEP_DIR}/timeline.log"

run_one "siglip_questiononly_attnqquery_nodynbudget_adapter_18k" "${TARGET_STEP_18K}" \
  "${SIGLIP_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
  --bridge_question_context_mode question_only \
  --lr_warmup_steps 1200

# Full eval at step 9k for legacy comparison against Phase 1 runs
echo "[$(date)] === Run 11: full eval at step 9k ===" | tee -a "${SWEEP_DIR}/timeline.log"

run_one "siglip_questiononly_attnqquery_nodynbudget_adapter_18k" "${TARGET_STEP}" \
  "${SIGLIP_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
  --bridge_question_context_mode question_only \
  --lr_warmup_steps 1200


# ============================================================================
# Run 12: SigLIP + questiononly @ 18k seed2 (frontier stability)
# ============================================================================

echo "[$(date)] === Run 12: SigLIP questiononly 18k seed2 ===" | tee -a "${SWEEP_DIR}/timeline.log"

run_one "siglip_questiononly_attnqquery_nodynbudget_adapter_18k_seed2" "${TARGET_STEP_18K}" \
  "${SIGLIP_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
  --bridge_question_context_mode question_only \
  --lr_warmup_steps 1200 \
  --seed 53

# Full eval at step 9k for seed variance measurement
echo "[$(date)] === Run 12: full eval at step 9k ===" | tee -a "${SWEEP_DIR}/timeline.log"

run_one "siglip_questiononly_attnqquery_nodynbudget_adapter_18k_seed2" "${TARGET_STEP}" \
  "${SIGLIP_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}" \
  --bridge_question_context_mode question_only \
  --lr_warmup_steps 1200 \
  --seed 53

COMMENT

# ============================================================================
# Run 10: DINOv2-B capacity-matched comparison @ 9k
# ============================================================================

echo "[$(date)] === Run 10: DINOv2-B nodynbudget (capacity match) ===" | tee -a "${SWEEP_DIR}/timeline.log"

run_one "dinov2b_attnqquery_nodynbudget_adapter_d3" "${TARGET_STEP}" \
  "${DINOV2B_ARGS[@]}" "${NODYN_ATTNQ_ADAPTER_ARGS[@]}"


echo "[$(date)] PHASE 2 COMPLETE" | tee -a "${SWEEP_DIR}/timeline.log"
