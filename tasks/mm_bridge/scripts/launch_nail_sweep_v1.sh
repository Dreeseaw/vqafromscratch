#!/bin/bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmnail_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmnail_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmnail_v1_20260313}"
TARGET_STEP="${TARGET_STEP:-9000}"

RUN_BS="${RUN_BS:-192}"
RUN_GA="${RUN_GA:-1}"
RUN_EVAL_BS="${RUN_EVAL_BS:-192}"

BEST_SOURCE_RUN="${BEST_SOURCE_RUN:-mmhammer_v2_qquery_dynbudget_adapter_earlylayer_geomcal}"
BEST_SOURCE_STEP="${BEST_SOURCE_STEP:-9000}"

LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-0}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DRY_RUN="${DRY_RUN:-0}"

if (( RUN_BS * RUN_GA != 192 )); then
  echo "[nail] ERROR effective batch must be 192, got $((RUN_BS * RUN_GA))"
  exit 1
fi

cat > "${SWEEP_DIR}/README.md" <<EOF
# Nail Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Plan sources:
- tasks/mm_bridge/docs/27_nail_sweep_plan_2026-03-13.md
- tasks/mm_bridge/docs/26_hammer_v2_sweep_report_2026-03-13.md

Queue policy:
- use the revised Nail execution priority from the plan appendix
- keep the stable Hammer runtime policy: effective batch 192, eval batch ${RUN_EVAL_BS}
- keep the adapter-centered Hammer baseline as the default family
- run corruption diagnostics first via eval-only checkpoint aliases of ${BEST_SOURCE_RUN}

Runtime assumptions:
- batched KV-cache eval path stays enabled
- base family is qquery + dynbudget + LM visual adapters
- deeper adapters, richer qquery generation, and role-specialized queries are the main new architecture axes
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

prepare_checkpoint_alias() {
  local alias_run_id="$1"
  local source_run_id="$2"
  local source_step="$3"
  local alias_dir="logs/${alias_run_id}"
  local source_ckpt="logs/${source_run_id}/step_${source_step}.tar"
  local alias_ckpt="${alias_dir}/step_${TARGET_STEP}.tar"

  if [[ ! -f "${source_ckpt}" ]]; then
    echo "[nail] ERROR missing source checkpoint: ${source_ckpt}"
    exit 1
  fi
  mkdir -pv "${alias_dir}" >/dev/null
  ln -sfn "../${source_run_id}/step_${source_step}.tar" "${alias_ckpt}"
}

run_one() {
  local suffix="$1"
  local bs="$2"
  local ga="$3"
  shift 3

  if (( bs * ga != 192 )); then
    echo "[nail] ERROR ${suffix} requested invalid effective batch: ${bs}x${ga}"
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

  local status=0
  if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
    echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  else
    status=$?
    echo "[$(date)] FAIL  ${run_id} (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
    return "${status}"
  fi
}

run_eval_alias() {
  local suffix="$1"
  local source_run_id="$2"
  local source_step="$3"
  shift 3

  local run_id="${RUN_PREFIX}_${suffix}"
  local done_ckpt="logs/${run_id}/step_${TARGET_STEP}.tar"

  if [[ -f "${done_ckpt}" ]] && has_completed_eval "${run_id}"; then
    echo "[$(date)] SKIP  ${run_id} (complete eval alias)" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    prepare_checkpoint_alias "${run_id}" "${source_run_id}" "${source_step}"
  fi

  local cmd=(
    ./runmm.sh "${run_id}" "${TARGET_STEP}"
    "${COMMON_ARGS[@]}"
    --batch_size "${RUN_BS}"
    --grad_accum_steps "${RUN_GA}"
    --eval_only
    --eval_batches 0
    --eval_fraction 1.0
    --eval_log_every 20
    --eval_scorer official
    --final_sanity_count 0
    --cuda_empty_cache_after_eval
    "$@"
  )

  echo "[$(date)] START ${run_id} eval-alias source=${source_run_id}/step_${source_step}.tar" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local status=0
  if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
    echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  else
    status=$?
    echo "[$(date)] FAIL  ${run_id} (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
    return "${status}"
  fi
}

# 1) best checkpoint corruption suite
run_eval_alias "best_ckpt_image_shuffle" "${BEST_SOURCE_RUN}" "${BEST_SOURCE_STEP}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
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
  --lm_visual_adapter_gate_init 0.5 \
  --eval_image_corruption shuffle

run_eval_alias "best_ckpt_image_zero" "${BEST_SOURCE_RUN}" "${BEST_SOURCE_STEP}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
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
  --lm_visual_adapter_gate_init 0.5 \
  --eval_image_corruption zero

run_eval_alias "best_ckpt_random_image_swap" "${BEST_SOURCE_RUN}" "${BEST_SOURCE_STEP}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
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
  --lm_visual_adapter_gate_init 0.5 \
  --eval_image_corruption random_swap

# 2) deeper adapters on the current best family
run_one "qquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 3 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

# 3) larger dynbudget cap on the current best family
run_one "qquery_dynbudget_adapter_d2_cap96" "${RUN_BS}" "${RUN_GA}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 96 \
  --bridge_token_select_k_min 48 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 2 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

# 4) LM-hidden-state mean-projection qquery
run_one "lmmeanqquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
  --bridge_query_bank_mode question_hidden_mean \
  --bridge_qquery_scale 1.0 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 3 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

# 5) attention-derived qquery generation
run_one "attnqquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
  --bridge_query_bank_mode question_hidden_attn \
  --bridge_qquery_scale 1.0 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 3 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

# 6) role-specialized query slots on the best family
run_one "rolespecial_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 \
  --bridge_query_role_specialization \
  --bridge_num_roles 4 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 3 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

# 7) richer qquery plus larger dynbudget cap
run_one "lmmeanqquery_dynbudget_adapter_d3_cap96" "${RUN_BS}" "${RUN_GA}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
  --bridge_query_bank_mode question_hidden_mean \
  --bridge_qquery_scale 1.0 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 96 \
  --bridge_token_select_k_min 48 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 3 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

# 8) strongest combined frontier probe
run_one "rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64" "${RUN_BS}" "${RUN_GA}" \
  --eval_batch_size "${RUN_EVAL_BS}" \
  --bridge_query_bank_mode question_hidden_mean \
  --bridge_qquery_scale 1.0 \
  --bridge_query_role_specialization \
  --bridge_num_roles 4 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 3 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
