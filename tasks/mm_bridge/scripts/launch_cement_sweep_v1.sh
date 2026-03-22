#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmcement_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmcement_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmcement_v1_20260316}"
TARGET_STEP="${TARGET_STEP:-9000}"
DRY_RUN="${DRY_RUN:-0}"

SIGLIP_BS="${SIGLIP_BS:-96}"
SIGLIP_GA="${SIGLIP_GA:-2}"
SIGLIP_EVAL_BS="${SIGLIP_EVAL_BS:-96}"
SIGLIP_MODEL_DIR="${SIGLIP_MODEL_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"

LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-0}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MIN_TRAIN_SPS="${MIN_TRAIN_SPS:-1.0}"
MIN_TRAIN_WINDOW="${MIN_TRAIN_WINDOW:-100}"
LOW_TRAIN_SPS_EXIT_CODE="${LOW_TRAIN_SPS_EXIT_CODE:-86}"
MAX_LOW_SPS_RESTARTS="${MAX_LOW_SPS_RESTARTS:-8}"

SEED_A="${SEED_A:-42}"
SEED_B="${SEED_B:-53}"
SEED_C="${SEED_C:-97}"

SKIP_PROMPT_ARM="${SKIP_PROMPT_ARM:-0}"
SKIP_QUESTION_ARM="${SKIP_QUESTION_ARM:-0}"
SKIP_SEED42="${SKIP_SEED42:-0}"
SKIP_SEED53="${SKIP_SEED53:-0}"
SKIP_SEED97="${SKIP_SEED97:-0}"

if (( SIGLIP_BS * SIGLIP_GA != 192 )); then
  echo "[cement] ERROR effective batch must be 192, got $((SIGLIP_BS * SIGLIP_GA))"
  exit 1
fi

cat > "${SWEEP_DIR}/README.md" <<EOF
# Cement Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Plan source:
- tasks/mm_bridge/docs/41_cement_sweep_plan_2026-03-16.md

Purpose:
- run the Tier 1 six-run SigLIP head-to-head
- compare prompt_only vs question_only at matched seeds, schedule, and batch layout
- seed-check the winning arm

Locked layout:
- batch_size=${SIGLIP_BS}
- grad_accum_steps=${SIGLIP_GA}
- eval_batch_size=${SIGLIP_EVAL_BS}
- effective_batch_size=$((SIGLIP_BS * SIGLIP_GA))

Locked model family:
- VM: SigLIP-B/16 (${SIGLIP_MODEL_DIR})
- bridge: perceiver resampler
- query path: attnqquery
- token selector: none (nodynbudget)
- LM adapters: cross_attn depth 3
- training horizon: ${TARGET_STEP} steps
- lr warmup: 600

Run matrix:
- prompt_only seeds: ${SEED_A}, ${SEED_B}, ${SEED_C}
- question_only seeds: ${SEED_A}, ${SEED_B}, ${SEED_C}

Skip controls:
- SKIP_PROMPT_ARM=1
- SKIP_QUESTION_ARM=1
- SKIP_SEED42=1
- SKIP_SEED53=1
- SKIP_SEED97=1

Watchdog:
- min_train_steps_per_s=${MIN_TRAIN_SPS}
- min_train_steps_window=${MIN_TRAIN_WINDOW}
- restart budget=${MAX_LOW_SPS_RESTARTS}
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
  --vision_model siglip_base
  --vision_checkpoint "${SIGLIP_MODEL_DIR}"
  --vision_feature_source encoder
  --vision_feature_mode auto
  --batch_size "${SIGLIP_BS}"
  --grad_accum_steps "${SIGLIP_GA}"
  --eval_batch_size "${SIGLIP_EVAL_BS}"
  --num_visual_tokens 49
  --bridge_type perceiver_resampler
  --bridge_query_depth 3
  --bridge_num_heads 8
  --bridge_token_reduce adaptive_pool
  --bridge_add_2d_pos_emb
  --bridge_pre_mixer_type none
  --bridge_question_conditioning
  --bridge_query_bank_mode question_hidden_attn
  --bridge_qquery_scale 1.0
  --bridge_token_selector_type none
  --bridge_token_select_k 0
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
  --lm_visual_adapter_type cross_attn
  --lm_visual_adapter_layers 3
  --lm_visual_adapter_num_heads 8
  --lm_visual_adapter_dropout 0.0
  --lm_visual_adapter_gate_init 0.5
  --lr 0.0002
  --lr_schedule cosine
  --lr_warmup_steps 600
  --lr_min_ratio 0.15
  --min_train_steps_per_s "${MIN_TRAIN_SPS}"
  --min_train_steps_window "${MIN_TRAIN_WINDOW}"
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
  shift 1

  local run_id="${RUN_PREFIX}_${suffix}"
  local done_ckpt="logs/${run_id}/step_${TARGET_STEP}.tar"

  if [[ -f "${done_ckpt}" ]] && has_completed_eval "${run_id}" "${TARGET_STEP}"; then
    echo "[$(date)] SKIP  ${run_id} (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_ckpt_step "${run_id}")"
  local restart_count=0

  while true; do
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
      cmd+=(
        "${COMMON_ARGS[@]}"
        "$@"
      )
    fi

    echo "[$(date)] START ${run_id} target=${TARGET_STEP} resume=${resume_step} restarts=${restart_count}" | tee -a "${SWEEP_DIR}/timeline.log"
    echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
      return 0
    fi

    local status=0
    if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
      echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
      return 0
    fi
    status=$?

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

run_seed_pair() {
  local seed="$1"
  local seed_tag="s${seed}"

  if [[ "${SKIP_PROMPT_ARM}" != "1" ]]; then
    run_one "siglip_cement_promptonly_${seed_tag}" \
      --seed "${seed}" \
      --bridge_question_context_mode prompt_only
  fi

  if [[ "${SKIP_QUESTION_ARM}" != "1" ]]; then
    run_one "siglip_cement_questiononly_${seed_tag}" \
      --seed "${seed}" \
      --bridge_question_context_mode question_only
  fi
}

echo "[$(date)] === CEMENT TIER 1 START ===" | tee -a "${SWEEP_DIR}/timeline.log"

if [[ "${SKIP_SEED42}" != "1" ]]; then
  echo "[$(date)] === Seed ${SEED_A} pair ===" | tee -a "${SWEEP_DIR}/timeline.log"
  run_seed_pair "${SEED_A}"
fi

if [[ "${SKIP_SEED53}" != "1" ]]; then
  echo "[$(date)] === Seed ${SEED_B} pair ===" | tee -a "${SWEEP_DIR}/timeline.log"
  run_seed_pair "${SEED_B}"
fi

if [[ "${SKIP_SEED97}" != "1" ]]; then
  echo "[$(date)] === Seed ${SEED_C} pair ===" | tee -a "${SWEEP_DIR}/timeline.log"
  run_seed_pair "${SEED_C}"
fi

echo "[$(date)] CEMENT TIER 1 COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
echo
echo "===================================================================="
echo "Cement Tier 1 queue complete."
echo "Review the six SigLIP runs, pick the winning arm, then run diagnostics:"
echo "  ./tasks/mm_bridge/scripts/run_diagnostics.sh <diag_run_id> --checkpoint logs/<winner_run>/step_${TARGET_STEP}.tar"
echo "===================================================================="
