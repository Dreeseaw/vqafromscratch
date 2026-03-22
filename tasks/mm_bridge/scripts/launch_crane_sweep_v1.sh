#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmcrane_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmcrane_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmcrane_v1_20260314}"
TARGET_STEP="${TARGET_STEP:-9000}"
TARGET_STEP_18K="${TARGET_STEP_18K:-18000}"

# All VMs use b96a2 (confirmed by perf probes 2026-03-14)
BS="${BS:-96}"
GA="${GA:-2}"
EVAL_BS="${EVAL_BS:-96}"
CAPALIGN_BS="${CAPALIGN_BS:-96}"
CAPALIGN_STEPS="${CAPALIGN_STEPS:-3000}"

MOBILEVIT_MODEL_DIR="${MOBILEVIT_MODEL_DIR:-logs/hf_vision/apple_mobilevit_small}"
MOBILECLIP_MODEL_DIR="${MOBILECLIP_MODEL_DIR:-logs/hf_vision/apple_mobileclip_s0}"
DINOV2_MODEL_DIR="${DINOV2_MODEL_DIR:-logs/hf_vision/facebook_dinov2_small}"

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

SKIP_TIER1="${SKIP_TIER1:-0}"
SKIP_TIER2="${SKIP_TIER2:-0}"
SKIP_TIER3="${SKIP_TIER3:-0}"
SKIP_TIER4="${SKIP_TIER4:-0}"
SKIP_TIER5="${SKIP_TIER5:-0}"
SKIP_TIER6="${SKIP_TIER6:-0}"
SKIP_RUN3_18K="${SKIP_RUN3_18K:-0}"

# MobileViT-specific loader overrides (lower workers to avoid OOM in loader)
MOBILEVIT_NUM_WORKERS="${MOBILEVIT_NUM_WORKERS:-2}"
MOBILEVIT_PREFETCH_FACTOR="${MOBILEVIT_PREFETCH_FACTOR:-1}"
MOBILEVIT_PIN_MEMORY="${MOBILEVIT_PIN_MEMORY:-0}"

if (( BS * GA != 192 )); then
  echo "[crane] ERROR effective batch must be 192, got $((BS * GA))"
  exit 1
fi

cat > "${SWEEP_DIR}/README.md" <<EOF
# Crane Extended Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Plan: tasks/mm_bridge/docs/34_crane_extended_sweep_plan_2026-03-14.md

Layout: ${BS}x${GA} for all VMs (effective batch 192)
Target step: ${TARGET_STEP} (${TARGET_STEP_18K} for 18k runs)

Skip controls: SKIP_TIER1=${SKIP_TIER1} SKIP_TIER2=${SKIP_TIER2} SKIP_TIER3=${SKIP_TIER3} SKIP_TIER4=${SKIP_TIER4} SKIP_TIER5=${SKIP_TIER5} SKIP_TIER6=${SKIP_TIER6}
EOF

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

DYN_ADAPTER_ARGS=(
  --eval_batch_size "${EVAL_BS}"
  --bridge_token_selector_type qadaptive
  --bridge_token_select_k 64
  --bridge_token_select_k_min 24
  --lm_visual_adapter_type cross_attn
  --lm_visual_adapter_layers 3
  --lm_visual_adapter_num_heads 8
  --lm_visual_adapter_dropout 0.0
  --lm_visual_adapter_gate_init 0.5
)

ATTNQQUERY_ARGS=(
  --bridge_query_bank_mode question_hidden_attn
  --bridge_qquery_scale 1.0
)

MOBILEVIT_ARGS=(
  --vision_model mobilevit_hf
  --vision_checkpoint "${MOBILEVIT_MODEL_DIR}"
  --vision_feature_mode auto
)

MOBILEVIT_LOADER_ARGS=(
  --num_workers "${MOBILEVIT_NUM_WORKERS}"
  --prefetch_factor "${MOBILEVIT_PREFETCH_FACTOR}"
)
if [[ "${MOBILEVIT_PIN_MEMORY}" == "0" ]]; then
  MOBILEVIT_LOADER_ARGS+=(--no-pin_memory)
fi

MOBILECLIP_ARGS=(
  --vision_model mobileclip_s0
  --vision_checkpoint "${MOBILECLIP_MODEL_DIR}"
  --vision_feature_mode auto
)

DINOV2_ARGS=(
  --vision_model dinov2_small
  --vision_checkpoint "${DINOV2_MODEL_DIR}"
  --vision_feature_mode auto
)

# --- Utility functions (same pattern as Plank launcher) ---

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
        --batch_size "${BS}"
        --grad_accum_steps "${GA}"
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
        --batch_size "${BS}"
        --grad_accum_steps "${GA}"
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
  # NOTE: This function is called inside $(...) so all log messages go to stderr.
  # Only the checkpoint path goes to stdout for capture.
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
  capalign_step=$((10#${capalign_step}))  # strip leading zeros to match mm.py naming

  # VQA stage uses the checkpoint from caption-align
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
        --batch_size "${BS}"
        --grad_accum_steps "${GA}"
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
        --batch_size "${BS}"
        --grad_accum_steps "${GA}"
        "$@"
      )
    else
      # First launch: use caption-align checkpoint
      cmd+=("${capalign_step}")
      cmd+=(
        "${COMMON_ARGS[@]}"
        --max_steps "${target_step}"
        --batch_size "${BS}"
        --grad_accum_steps "${GA}"
        "$@"
      )
      # Copy the caption-align checkpoint into the run dir for runmm.sh to find
      mkdir -p "logs/${run_id}"
      if [[ ! -f "logs/${run_id}/step_${capalign_step}.tar" ]]; then
        cp "${capalign_ckpt}" "logs/${run_id}/step_${capalign_step}.tar"
      fi
    fi

    echo "[$(date)] START two-stage ${run_id} target=${target_step} resume=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
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
# TIER 1: MobileViT Completion
# ============================================================================

if [[ "${SKIP_TIER1}" != "1" ]]; then
  echo "[$(date)] === TIER 1: MobileViT Completion ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 1: questiononly
  run_one "mobilevit_questiononly_attnqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
    "${MOBILEVIT_ARGS[@]}" "${MOBILEVIT_LOADER_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --bridge_question_context_mode question_only

  # Run 2: adapter d4
  run_one "mobilevit_attnqquery_dynbudget_adapter_d4_cap64" "${TARGET_STEP}" \
    "${MOBILEVIT_ARGS[@]}" "${MOBILEVIT_LOADER_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --lm_visual_adapter_layers 4

  # Run 3: 18k training (skip-controllable)
  if [[ "${SKIP_RUN3_18K}" != "1" ]]; then
    run_one "mobilevit_attnqquery_dynbudget_adapter_d3_cap64_18k" "${TARGET_STEP_18K}" \
      "${MOBILEVIT_ARGS[@]}" "${MOBILEVIT_LOADER_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
      --lr_warmup_steps 1200
  fi

  # Run 19: seed2
  run_one "mobilevit_attnqquery_dynbudget_adapter_d3_cap64_seed2" "${TARGET_STEP}" \
    "${MOBILEVIT_ARGS[@]}" "${MOBILEVIT_LOADER_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --seed 53
fi


# ============================================================================
# TIER 2: New VM Baselines
# ============================================================================

if [[ "${SKIP_TIER2}" != "1" ]]; then
  echo "[$(date)] === TIER 2: New VM Baselines ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 4: MobileCLIP with attnqquery
  run_one "mobileclip_attnqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
    "${MOBILECLIP_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}"

  # Run 5: DINOv2 with attnqquery + dynbudget
  run_one "dinov2s_attnqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
    "${DINOV2_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}"

  # Run 6: DINOv2 with lmmeanqquery + dynbudget
  run_one "dinov2s_lmmeanqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
    "${DINOV2_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" \
    --bridge_query_bank_mode question_hidden_mean \
    --bridge_qquery_scale 1.0
fi


# ============================================================================
# TIER 3: DINOv2 Dynbudget Sweep
# ============================================================================

if [[ "${SKIP_TIER3}" != "1" ]]; then
  echo "[$(date)] === TIER 3: DINOv2 Dynbudget Sweep ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 7: DINOv2 no dynbudget (all 256 tokens to perceiver)
  run_one "dinov2s_attnqquery_nodynbudget_adapter_d3" "${TARGET_STEP}" \
    "${DINOV2_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --eval_batch_size "${EVAL_BS}" \
    --bridge_token_selector_type none \
    --bridge_token_select_k 0 \
    --lm_visual_adapter_type cross_attn \
    --lm_visual_adapter_layers 3 \
    --lm_visual_adapter_num_heads 8 \
    --lm_visual_adapter_dropout 0.0 \
    --lm_visual_adapter_gate_init 0.5

  # Run 8: DINOv2 cap128
  run_one "dinov2s_attnqquery_dynbudget_adapter_d3_cap128" "${TARGET_STEP}" \
    "${DINOV2_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --eval_batch_size "${EVAL_BS}" \
    --bridge_token_selector_type qadaptive \
    --bridge_token_select_k 128 \
    --bridge_token_select_k_min 48 \
    --lm_visual_adapter_type cross_attn \
    --lm_visual_adapter_layers 3 \
    --lm_visual_adapter_num_heads 8 \
    --lm_visual_adapter_dropout 0.0 \
    --lm_visual_adapter_gate_init 0.5

  # Run 9: DINOv2 cap32
  run_one "dinov2s_attnqquery_dynbudget_adapter_d3_cap32" "${TARGET_STEP}" \
    "${DINOV2_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --eval_batch_size "${EVAL_BS}" \
    --bridge_token_selector_type qadaptive \
    --bridge_token_select_k 32 \
    --bridge_token_select_k_min 12 \
    --lm_visual_adapter_type cross_attn \
    --lm_visual_adapter_layers 3 \
    --lm_visual_adapter_num_heads 8 \
    --lm_visual_adapter_dropout 0.0 \
    --lm_visual_adapter_gate_init 0.5
fi


# ============================================================================
# TIER 4: Caption-Align Pre-Training
# ============================================================================

if [[ "${SKIP_TIER4}" != "1" ]]; then
  echo "[$(date)] === TIER 4: Caption-Align Pre-Training ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 10: MobileViT caption-align → VQA
  capalign_ckpt_mobilevit="$(run_capalign "mobilevit" \
    --vision_model mobilevit_hf \
    --vision_checkpoint "${MOBILEVIT_MODEL_DIR}" \
    --vision_feature_source encoder \
    --vision_feature_mode auto)"
  run_twostage "mobilevit_captionalign_attnqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
    "${capalign_ckpt_mobilevit}" \
    "${MOBILEVIT_ARGS[@]}" "${MOBILEVIT_LOADER_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}"

  # Run 11: DINOv2 caption-align → VQA
  capalign_ckpt_dinov2="$(run_capalign "dinov2s" \
    --vision_model dinov2_small \
    --vision_checkpoint "${DINOV2_MODEL_DIR}" \
    --vision_feature_source encoder \
    --vision_feature_mode auto)"
  run_twostage "dinov2s_captionalign_attnqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
    "${capalign_ckpt_dinov2}" \
    "${DINOV2_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}"

  # Run 12: MobileCLIP caption-align → VQA (low priority)
  capalign_ckpt_mobileclip="$(run_capalign "mobileclip" \
    --vision_model mobileclip_s0 \
    --vision_checkpoint "${MOBILECLIP_MODEL_DIR}" \
    --vision_feature_source encoder \
    --vision_feature_mode auto)"
  run_twostage "mobileclip_captionalign_attnqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
    "${capalign_ckpt_mobileclip}" \
    "${MOBILECLIP_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}"
fi


# ============================================================================
# TIER 5: Stacking Winners (configs selected post-hoc based on tier 1-4 results)
# ============================================================================

if [[ "${SKIP_TIER5}" != "1" ]]; then
  echo "[$(date)] === TIER 5: Stacking Winners ===" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] NOTE: Tier 5 runs must be configured manually based on Tier 1-4 results." | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] Edit this section after analyzing results from Tiers 1-4." | tee -a "${SWEEP_DIR}/timeline.log"

  # Example stacking run (uncomment and configure after Tier 1-4 results):
  # run_one "best_vm_stacked_attnqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
  #   "${DINOV2_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
  #   --bridge_question_context_mode question_only
fi


# ============================================================================
# TIER 6: Diagnostics (low priority)
# ============================================================================

if [[ "${SKIP_TIER6}" != "1" ]]; then
  echo "[$(date)] === TIER 6: Diagnostics ===" | tee -a "${SWEEP_DIR}/timeline.log"

  # Run 16: questiononly on DINOv2
  run_one "dinov2s_questiononly_attnqquery_dynbudget_adapter_d3_cap64" "${TARGET_STEP}" \
    "${DINOV2_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --bridge_question_context_mode question_only

  # Run 17: d4 adapters on MobileCLIP
  run_one "mobileclip_attnqquery_dynbudget_adapter_d4_cap64" "${TARGET_STEP}" \
    "${MOBILECLIP_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --lm_visual_adapter_layers 4

  # Run 18: d5 adapters on DINOv2
  run_one "dinov2s_attnqquery_dynbudget_adapter_d5_cap64" "${TARGET_STEP}" \
    "${DINOV2_ARGS[@]}" "${DYN_ADAPTER_ARGS[@]}" "${ATTNQQUERY_ARGS[@]}" \
    --lm_visual_adapter_layers 5
fi


echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
