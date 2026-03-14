#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmhammer_perfprobe_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmhammer_perfprobe_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmhammer_perf_v1_20260312}"
MAX_STEPS="${MAX_STEPS:-40}"
FINAL_EVAL_BATCHES="${FINAL_EVAL_BATCHES:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TRAIN_MIN_STEPS_PER_S="${TRAIN_MIN_STEPS_PER_S:-3.0}"
EVAL_MIN_STEPS_PER_S="${EVAL_MIN_STEPS_PER_S:-0.8}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_PATH="${SWEEP_DIR}/probe_results.tsv"
CHOICES_PATH="${SWEEP_DIR}/recommended_layouts.tsv"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Hammer Performance Probes V1

Purpose:
- verify the new Hammer architecture families can keep effective batch size 192
- pick the fastest batch_size x grad_accum layout that clears throughput floors

Thresholds:
- train steps/s > ${TRAIN_MIN_STEPS_PER_S}
- eval steps/s > ${EVAL_MIN_STEPS_PER_S}

Runtime:
- MAX_STEPS=${MAX_STEPS}
- FINAL_EVAL_BATCHES=${FINAL_EVAL_BATCHES}
- NUM_WORKERS=${NUM_WORKERS}
- PREFETCH_FACTOR=${PREFETCH_FACTOR}

Probe order:
- try 192x1 first
- fall back to 96x2
- then 64x3
- then 48x4
- then 32x6
EOF

COMMON_ARGS=(
  --precision bf16
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --epochs 20
  --max_steps "${MAX_STEPS}"
  --manual_max_steps
  --log_every 20
  --eval_every 0
  --eval_batches 0
  --final_eval_batches "${FINAL_EVAL_BATCHES}"
  --eval_log_every 1
  --ckpt_every 0
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
  --lr_warmup_steps 200
  --lr_min_ratio 0.15
)

extract_metric() {
  local pattern="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    return 1
  fi
  local line
  line="$(rg "${pattern}" "${path}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    return 1
  fi
  echo "${line}" | sed -E 's/.*steps_per_s=([0-9.]+).*/\1/'
}

passes_thresholds() {
  local train_sps="$1"
  local eval_sps="$2"
  awk -v train_sps="${train_sps}" -v eval_sps="${eval_sps}" \
      -v train_min="${TRAIN_MIN_STEPS_PER_S}" -v eval_min="${EVAL_MIN_STEPS_PER_S}" \
      'BEGIN { exit !((train_sps + 0.0) > (train_min + 0.0) && (eval_sps + 0.0) > (eval_min + 0.0)) }'
}

run_attempt() {
  local variant="$1"
  local bs="$2"
  local ga="$3"
  shift 3

  local run_id="${RUN_PREFIX}_${variant}_b${bs}a${ga}"
  local run_log="logs/${run_id}/logfile.txt"
  local stdout_log="${SWEEP_DIR}/${run_id}.stdout.log"
  local cmd=(
    ./runmm.sh "${run_id}"
    "${COMMON_ARGS[@]}"
    --batch_size "${bs}"
    --grad_accum_steps "${ga}"
    "$@"
  )

  echo "[$(date)] START ${run_id} bs=${bs} ga=${ga}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"

  if [[ -f "${run_log}" ]]; then
    local existing_train_sps=""
    local existing_eval_sps=""
    existing_train_sps="$(extract_metric '^\[mm\] step=.*steps_per_s=' "${run_log}" || true)"
    existing_eval_sps="$(extract_metric '^\[eval:.*steps_per_s=' "${run_log}" || true)"
    if [[ -n "${existing_train_sps}" && -n "${existing_eval_sps}" ]]; then
      local existing_status="slow"
      if passes_thresholds "${existing_train_sps}" "${existing_eval_sps}"; then
        existing_status="pass"
      fi
      echo -e "${variant}\t${run_id}\t${bs}\t${ga}\t${existing_train_sps}\t${existing_eval_sps}\treuse_${existing_status}" >> "${RESULTS_PATH}"
      echo "[$(date)] REUSE ${run_id} train_sps=${existing_train_sps} eval_sps=${existing_eval_sps} status=${existing_status}" | tee -a "${SWEEP_DIR}/timeline.log"
      [[ "${existing_status}" == "pass" ]]
      return
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo -e "${variant}\t${run_id}\t${bs}\t${ga}\tNA\tNA\tdry_run" >> "${RESULTS_PATH}"
    return 1
  fi

  if ! "${cmd[@]}" >> "${stdout_log}" 2>&1; then
    echo -e "${variant}\t${run_id}\t${bs}\t${ga}\tNA\tNA\tfail" >> "${RESULTS_PATH}"
    echo "[$(date)] FAIL  ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi

  local train_sps=""
  local eval_sps=""
  train_sps="$(extract_metric '^\[mm\] step=.*steps_per_s=' "${run_log}" || true)"
  eval_sps="$(extract_metric '^\[eval:.*steps_per_s=' "${run_log}" || true)"

  if [[ -z "${train_sps}" || -z "${eval_sps}" ]]; then
    echo -e "${variant}\t${run_id}\t${bs}\t${ga}\t${train_sps:-NA}\t${eval_sps:-NA}\tmissing_metrics" >> "${RESULTS_PATH}"
    echo "[$(date)] MISS  ${run_id} train_sps=${train_sps:-NA} eval_sps=${eval_sps:-NA}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi

  local status="slow"
  if passes_thresholds "${train_sps}" "${eval_sps}"; then
    status="pass"
  fi
  echo -e "${variant}\t${run_id}\t${bs}\t${ga}\t${train_sps}\t${eval_sps}\t${status}" >> "${RESULTS_PATH}"
  echo "[$(date)] END   ${run_id} train_sps=${train_sps} eval_sps=${eval_sps} status=${status}" | tee -a "${SWEEP_DIR}/timeline.log"

  [[ "${status}" == "pass" ]]
}

probe_variant() {
  local variant="$1"
  shift

  local choices=(
    "192 1"
    "96 2"
    "64 3"
    "48 4"
    "32 6"
  )
  local choice
  for choice in "${choices[@]}"; do
    local bs ga
    read -r bs ga <<< "${choice}"
    if run_attempt "${variant}" "${bs}" "${ga}" "$@"; then
      echo -e "${variant}\t${bs}\t${ga}\tpass" >> "${CHOICES_PATH}"
      return 0
    fi
  done

  echo -e "${variant}\tNA\tNA\tno_pass" >> "${CHOICES_PATH}"
  return 1
}

echo -e "variant\trun_id\tbatch_size\tgrad_accum\ttrain_steps_per_s\teval_steps_per_s\tstatus" > "${RESULTS_PATH}"
echo -e "variant\tbatch_size\tgrad_accum\tstatus" > "${CHOICES_PATH}"

overall_status=0

probe_variant \
  "qquery_earlylayer_geomcal" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 || overall_status=1

probe_variant \
  "adapter_safeqcond_earlylayer_geomcal" \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 2 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5 || overall_status=1

probe_variant \
  "dynbudget_qscore_earlylayer_geomcal" \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 || overall_status=1

probe_variant \
  "qquery_adapter_earlylayer_geomcal" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 2 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5 || overall_status=1

probe_variant \
  "qquery_dynbudget_earlylayer_geomcal" \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 || overall_status=1

probe_variant \
  "dynbudget_adapter_earlylayer_geomcal" \
  --bridge_token_selector_type qadaptive \
  --bridge_token_select_k 64 \
  --bridge_token_select_k_min 24 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 2 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5 || overall_status=1

probe_variant \
  "qquery_dynbudget_adapter_earlylayer_geomcal" \
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
  --lm_visual_adapter_gate_init 0.5 || overall_status=1

echo "[$(date)] PROBES COMPLETE ${SWEEP_ID} overall_status=${overall_status}" | tee -a "${SWEEP_DIR}/timeline.log"
exit "${overall_status}"
