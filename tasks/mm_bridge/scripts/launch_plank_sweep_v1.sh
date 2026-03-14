#!/bin/bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmplank_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmplank_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmplank_v1_20260313}"
TARGET_STEP="${TARGET_STEP:-9000}"

RUN_BS="${RUN_BS:-192}"
RUN_GA="${RUN_GA:-1}"
RUN_EVAL_BS="${RUN_EVAL_BS:-192}"
MOBILEVIT_BS="${MOBILEVIT_BS:-96}"
MOBILEVIT_GA="${MOBILEVIT_GA:-2}"
MOBILEVIT_EVAL_BS="${MOBILEVIT_EVAL_BS:-96}"

MOBILEVIT_MODEL_DIR="${MOBILEVIT_MODEL_DIR:-logs/hf_vision/apple_mobilevit_small}"
MOBILEVIT_SEED2="${MOBILEVIT_SEED2:-53}"
MOBILEVIT_NUM_WORKERS="${MOBILEVIT_NUM_WORKERS:-2}"
MOBILEVIT_PREFETCH_FACTOR="${MOBILEVIT_PREFETCH_FACTOR:-1}"
MOBILEVIT_PIN_MEMORY="${MOBILEVIT_PIN_MEMORY:-0}"

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

SKIP_MOBILEVIT_STAGE1="${SKIP_MOBILEVIT_STAGE1:-0}"
SKIP_MOBILEVIT_SEED2="${SKIP_MOBILEVIT_SEED2:-0}"
SKIP_CORE_STAGE1="${SKIP_CORE_STAGE1:-0}"
SKIP_VISUAL_ADAPTER="${SKIP_VISUAL_ADAPTER:-0}"

if (( RUN_BS * RUN_GA != 192 )); then
  echo "[plank] ERROR effective batch must be 192, got $((RUN_BS * RUN_GA))"
  exit 1
fi
if (( MOBILEVIT_BS * MOBILEVIT_GA != 192 )); then
  echo "[plank] ERROR MobileViT effective batch must be 192, got $((MOBILEVIT_BS * MOBILEVIT_GA))"
  exit 1
fi

cat > "${SWEEP_DIR}/README.md" <<EOF
# Plank Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Plan sources:
- tasks/mm_bridge/docs/29_plank_sweep_plan_2026-03-13.md
- tasks/mm_bridge/docs/28_nail_sweep_report_2026-03-13.md
- tasks/mm_bridge/docs/30_mobilevit_perf_tuning_2026-03-13.md

Queue policy:
- follow the practical draft queue from the Plank plan
- start with the narrow MobileViT "same bridge, better vision" branch
- then return to the original-VM qquery-sharpening branch
- keep effective batch fixed at 192 with ${RUN_BS}x${RUN_GA}
- keep eval batch at ${RUN_EVAL_BS} for the original-VM branch
- keep MobileViT at ${MOBILEVIT_BS}x${MOBILEVIT_GA} with eval batch ${MOBILEVIT_EVAL_BS}

MobileViT policy:
- use the optimized frozen MobileViT HF path at ${MOBILEVIT_MODEL_DIR}
- keep cap fixed at 64
- keep role specialization out
- use a lower-memory MobileViT train layout by default: ${MOBILEVIT_BS}x${MOBILEVIT_GA}
- use a safer MobileViT loader by default: workers=${MOBILEVIT_NUM_WORKERS}, prefetch=${MOBILEVIT_PREFETCH_FACTOR}, pin_memory=${MOBILEVIT_PIN_MEMORY}

Deferred from this launcher:
- vm_lastblock_tune_lmmeanqquery_dynbudget_adapter_d3_cap64
- bridge-pretraining variants

Skip controls:
- SKIP_MOBILEVIT_STAGE1=1 to skip the MobileViT branch
- SKIP_MOBILEVIT_SEED2=1 to skip the MobileViT seed-2 rerun
- SKIP_CORE_STAGE1=1 to skip the original-VM qquery-sharpening branch
- SKIP_VISUAL_ADAPTER=1 to skip the visual-adapter branch

Watchdog policy:
- train watchdog threshold defaults to ${MIN_TRAIN_SPS} steps/s over ${MIN_TRAIN_WINDOW} steps
- low-throughput exits use code ${LOW_TRAIN_SPS_EXIT_CODE}
- launcher auto-restarts from the latest checkpoint up to ${MAX_LOW_SPS_RESTARTS} times
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
  local bs="$2"
  local ga="$3"
  shift 3

  if (( bs * ga != 192 )); then
    echo "[plank] ERROR ${suffix} requested invalid effective batch: ${bs}x${ga}"
    exit 1
  fi

  local run_id="${RUN_PREFIX}_${suffix}"
  local done_ckpt="logs/${run_id}/step_${TARGET_STEP}.tar"

  if [[ -f "${done_ckpt}" ]] && has_completed_eval "${run_id}"; then
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
        --batch_size "${bs}"
        --grad_accum_steps "${ga}"
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
        --batch_size "${bs}"
        --grad_accum_steps "${ga}"
        "$@"
      )
    fi

    echo "[$(date)] START ${run_id} bs=${bs} ga=${ga} target_step=${TARGET_STEP} resume_step=${resume_step} restart_count=${restart_count}" | tee -a "${SWEEP_DIR}/timeline.log"
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
        echo "[$(date)] FAIL  ${run_id} exceeded low-sps restart budget ${MAX_LOW_SPS_RESTARTS}" | tee -a "${SWEEP_DIR}/timeline.log"
        return "${status}"
      fi
      resume_step="$(latest_ckpt_step "${run_id}")"
      if (( resume_step <= 0 )); then
        echo "[$(date)] FAIL  ${run_id} low-sps watchdog exit without checkpoint" | tee -a "${SWEEP_DIR}/timeline.log"
        return "${status}"
      fi
      echo "[$(date)] RESTART ${run_id} after low-sps watchdog exit_code=${status} resume_step=${resume_step} restart_count=${restart_count}" | tee -a "${SWEEP_DIR}/timeline.log"
      continue
    fi

    echo "[$(date)] FAIL  ${run_id} (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
    return "${status}"
  done
}

COMMON_DYN_ADAPTER_ARGS=(
  --eval_batch_size "${RUN_EVAL_BS}"
  --bridge_token_selector_type qadaptive
  --bridge_token_select_k 64
  --bridge_token_select_k_min 24
  --lm_visual_adapter_type cross_attn
  --lm_visual_adapter_layers 3
  --lm_visual_adapter_num_heads 8
  --lm_visual_adapter_dropout 0.0
  --lm_visual_adapter_gate_init 0.5
)

MOBILEVIT_DYN_ADAPTER_ARGS=(
  --eval_batch_size "${MOBILEVIT_EVAL_BS}"
  --bridge_token_selector_type qadaptive
  --bridge_token_select_k 64
  --bridge_token_select_k_min 24
  --lm_visual_adapter_type cross_attn
  --lm_visual_adapter_layers 3
  --lm_visual_adapter_num_heads 8
  --lm_visual_adapter_dropout 0.0
  --lm_visual_adapter_gate_init 0.5
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

# 1) MobileViT Stage 1: same bridge, better vision
if [[ "${SKIP_MOBILEVIT_STAGE1}" != "1" ]]; then
  run_one "mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64" "${MOBILEVIT_BS}" "${MOBILEVIT_GA}" \
    "${MOBILEVIT_ARGS[@]}" \
    "${MOBILEVIT_LOADER_ARGS[@]}" \
    "${MOBILEVIT_DYN_ADAPTER_ARGS[@]}" \
    --bridge_query_bank_mode question_hidden_mean \
    --bridge_qquery_scale 1.0

  run_one "mobilevit_qquery_dynbudget_adapter_d3_cap64" "${MOBILEVIT_BS}" "${MOBILEVIT_GA}" \
    "${MOBILEVIT_ARGS[@]}" \
    "${MOBILEVIT_LOADER_ARGS[@]}" \
    "${MOBILEVIT_DYN_ADAPTER_ARGS[@]}" \
    --bridge_query_bank_mode question_mix \
    --bridge_qquery_basis_count 4 \
    --bridge_qquery_scale 1.0

  run_one "mobilevit_attnqquery_dynbudget_adapter_d3_cap64" "${MOBILEVIT_BS}" "${MOBILEVIT_GA}" \
    "${MOBILEVIT_ARGS[@]}" \
    "${MOBILEVIT_LOADER_ARGS[@]}" \
    "${MOBILEVIT_DYN_ADAPTER_ARGS[@]}" \
    --bridge_query_bank_mode question_hidden_attn \
    --bridge_qquery_scale 1.0

  if [[ "${SKIP_MOBILEVIT_SEED2}" != "1" ]]; then
    run_one "mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64_seed2" "${MOBILEVIT_BS}" "${MOBILEVIT_GA}" \
      "${MOBILEVIT_ARGS[@]}" \
      "${MOBILEVIT_LOADER_ARGS[@]}" \
      "${MOBILEVIT_DYN_ADAPTER_ARGS[@]}" \
      --seed "${MOBILEVIT_SEED2}" \
      --bridge_query_bank_mode question_hidden_mean \
      --bridge_qquery_scale 1.0
  fi
fi

# 2) Original-VM qquery sharpening branch
if [[ "${SKIP_CORE_STAGE1}" != "1" ]]; then
  run_one "questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
    "${COMMON_DYN_ADAPTER_ARGS[@]}" \
    --bridge_question_context_mode question_only \
    --bridge_query_bank_mode question_hidden_mean \
    --bridge_qquery_scale 1.0

  run_one "multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
    "${COMMON_DYN_ADAPTER_ARGS[@]}" \
    --bridge_query_bank_mode question_hidden_mean_multi \
    --bridge_qquery_multi_count 4 \
    --bridge_qquery_scale 1.0

  run_one "hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
    "${COMMON_DYN_ADAPTER_ARGS[@]}" \
    --bridge_query_bank_mode question_hidden_hybrid \
    --bridge_qquery_scale 1.0 \
    --bridge_qquery_hybrid_gate_init 0.5

  run_one "iter2_lmmeanqquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
    "${COMMON_DYN_ADAPTER_ARGS[@]}" \
    --bridge_query_bank_mode question_hidden_mean \
    --bridge_qquery_scale 1.0 \
    --bridge_iterative_qquery_steps 2 \
    --bridge_iterative_qquery_residual_scale 1.0
fi

# 3) Small visual adaptation branch
if [[ "${SKIP_VISUAL_ADAPTER}" != "1" ]]; then
  run_one "visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
    "${COMMON_DYN_ADAPTER_ARGS[@]}" \
    --bridge_query_bank_mode question_hidden_mean \
    --bridge_qquery_scale 1.0 \
    --visual_feature_adapter_type res_mlp \
    --visual_feature_adapter_hidden_dim 0 \
    --visual_feature_adapter_dropout 0.0
fi

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
