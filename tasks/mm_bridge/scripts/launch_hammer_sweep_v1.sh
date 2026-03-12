#!/bin/bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmhammer_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmhammer_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmhammer_v1_20260312}"
TARGET_STEP="${TARGET_STEP:-9000}"

BRIDGE_ONLY_BS="${BRIDGE_ONLY_BS:-64}"
BRIDGE_ONLY_GA="${BRIDGE_ONLY_GA:-3}"
ADAPTER_BS="${ADAPTER_BS:-192}"
ADAPTER_GA="${ADAPTER_GA:-1}"

LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-0}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_NONCOMPLIANT="${INCLUDE_NONCOMPLIANT:-0}"

if (( BRIDGE_ONLY_BS * BRIDGE_ONLY_GA != 192 )); then
  echo "[hammer] ERROR bridge-only effective batch must be 192, got $((BRIDGE_ONLY_BS * BRIDGE_ONLY_GA))"
  exit 1
fi
if (( ADAPTER_BS * ADAPTER_GA != 192 )); then
  echo "[hammer] ERROR adapter effective batch must be 192, got $((ADAPTER_BS * ADAPTER_GA))"
  exit 1
fi

cat > "${SWEEP_DIR}/README.md" <<EOF
# Hammer Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Plan source:
- tasks/mm_bridge/docs/22_hammer_sweep_plan_2026-03-12.md

Performance-tuning source:
- tasks/mm_bridge/docs/23_hammer_perf_tuning_report_2026-03-12.md
- logs/mmhammer_perfprobe_v1_latest/recommended_layouts.tsv

Default queue policy:
- run the Hammer anchor plus every architecture family that cleared
  train steps/s > 3.0 and eval steps/s > 0.8 at effective batch 192
- keep bridge-only qquery-family runs at ${BRIDGE_ONLY_BS}x${BRIDGE_ONLY_GA}
- keep adapter-family runs at ${ADAPTER_BS}x${ADAPTER_GA}
- omit noncompliant dynbudget-only runs unless INCLUDE_NONCOMPLIANT=1

Omitted by default:
- dynbudget_qscore_earlylayer_geomcal
- qquery_dynbudget_earlylayer_geomcal

Those two variants never found a bs x ga layout that satisfied both
throughput floors at effective batch 192 in the probe sweep.
Set INCLUDE_NONCOMPLIANT=1 to enqueue them anyway with 64x3, which was the
fastest projected overnight choice under the 9000-train-step / 2020-eval-step
runtime model.
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
    echo "[hammer] ERROR ${suffix} requested invalid effective batch: ${bs}x${ga}"
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

  echo "[$(date)] START ${run_id} bs=${bs} ga=${ga} target_step=${TARGET_STEP} resume_step=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
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
  echo "[$(date)] FAIL  ${run_id} (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
  return "${status}"
}

# 1) hammer control anchor
run_one "anchor_safeqcond_earlylayer_geomcal" "${BRIDGE_ONLY_BS}" "${BRIDGE_ONLY_GA}"

# 2) question-conditioned query bank
run_one "qquery_earlylayer_geomcal" "${BRIDGE_ONLY_BS}" "${BRIDGE_ONLY_GA}" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0

# 3) residual LM visual adapters
run_one "adapter_safeqcond_earlylayer_geomcal" "${ADAPTER_BS}" "${ADAPTER_GA}" \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 2 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

if [[ "${INCLUDE_NONCOMPLIANT}" == "1" ]]; then
  run_one "dynbudget_qscore_earlylayer_geomcal" 64 3 \
    --bridge_token_selector_type qadaptive \
    --bridge_token_select_k 64 \
    --bridge_token_select_k_min 24
fi

# 4) qquery + adapters
run_one "qquery_adapter_earlylayer_geomcal" "${ADAPTER_BS}" "${ADAPTER_GA}" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 2 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

if [[ "${INCLUDE_NONCOMPLIANT}" == "1" ]]; then
  run_one "qquery_dynbudget_earlylayer_geomcal" 64 3 \
    --bridge_query_bank_mode question_mix \
    --bridge_qquery_basis_count 4 \
    --bridge_qquery_scale 1.0 \
    --bridge_token_selector_type qadaptive \
    --bridge_token_select_k 64 \
    --bridge_token_select_k_min 24
fi

# 5) dynbudget + adapters
run_one "dynbudget_adapter_earlylayer_geomcal" "${ADAPTER_BS}" "${ADAPTER_GA}" \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 2 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

# 6) full compliant hammer stack
run_one "qquery_dynbudget_adapter_earlylayer_geomcal" "${ADAPTER_BS}" "${ADAPTER_GA}" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 2 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
