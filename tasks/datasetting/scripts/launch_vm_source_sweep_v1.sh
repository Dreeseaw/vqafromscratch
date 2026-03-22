#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="ds_vmsource_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/ds_vmsource_v1_latest"

DRY_RUN="${DRY_RUN:-0}"

VM_BATCH_SIZE="${VM_BATCH_SIZE:-128}"
VM_NUM_WORKERS="${VM_NUM_WORKERS:-10}"
VM_PREFETCH_FACTOR="${VM_PREFETCH_FACTOR:-1}"
VM_PIN_MEMORY="${VM_PIN_MEMORY:-1}"

MM_TARGET_STEP="${MM_TARGET_STEP:-9000}"
MM_BATCH_SIZE="${MM_BATCH_SIZE:-96}"
MM_GRAD_ACCUM="${MM_GRAD_ACCUM:-2}"
MM_EVAL_BATCH_SIZE="${MM_EVAL_BATCH_SIZE:-96}"
MM_NUM_WORKERS="${MM_NUM_WORKERS:-4}"
MM_PREFETCH_FACTOR="${MM_PREFETCH_FACTOR:-2}"
MM_SEED="${MM_SEED:-35}"

cat > "${SWEEP_DIR}/README.md" <<EOF
# VM Source Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Plan doc: tasks/datasetting/docs/05_vm_source_sweep_v1_2026-03-18.md

This sweep compares image-source families at matched total image-exposure budget.
EOF

printf "family\tvm_run_id\tmm_run_id\tvm_epochs\tvm_checkpoint\tdataset_mix_json\n" > "${SWEEP_DIR}/families.tsv"

COMMON_MM_ARGS=(
  --precision bf16
  --num_workers "${MM_NUM_WORKERS}"
  --prefetch_factor "${MM_PREFETCH_FACTOR}"
  --epochs 400
  --max_steps "${MM_TARGET_STEP}"
  --manual_max_steps
  --log_every 20
  --eval_every 1000
  --eval_batches 100
  --final_eval_batches 0
  --eval_log_every 20
  --eval_fraction 1.0
  --ckpt_every 1000
  --eval_scorer official
  --final_sanity_count 0
  --cuda_empty_cache_after_eval
  --eval_use_kv_cache
  --eval_kv_cache_mode batched
  --vision_feature_source encoder
  --vision_feature_mode auto
  --batch_size "${MM_BATCH_SIZE}"
  --grad_accum_steps "${MM_GRAD_ACCUM}"
  --eval_batch_size "${MM_EVAL_BATCH_SIZE}"
  --num_visual_tokens 49
  --bridge_type perceiver_resampler
  --bridge_query_depth 3
  --bridge_num_heads 8
  --bridge_token_reduce adaptive_pool
  --bridge_add_2d_pos_emb
  --bridge_pre_mixer_type none
  --bridge_question_conditioning
  --bridge_query_bank_mode question_hidden_attn
  --bridge_question_context_mode question_only
  --bridge_qquery_scale 1.0
  --bridge_qcond_scale 0.5
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
  --seed "${MM_SEED}"
  --vision_model dinovit_ssl
)

latest_step_checkpoint() {
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

has_mm_completed_eval() {
  local run_id="$1"
  local target="${2:-${MM_TARGET_STEP}}"
  local answers_path="logs/${run_id}/fixed_eval_val_answers.jsonl"
  local pattern="\"global_step\": ${target}.*\"tag\": \"(final_eval|eval_only)\""
  if [[ ! -f "${answers_path}" ]]; then
    return 1
  fi
  grep -Eq "${pattern}" "${answers_path}"
}

run_vm_stage() {
  local run_id="$1"
  local epochs="$2"
  local dataset_mix_json="$3"
  local final_epoch_ckpt="logs/${run_id}/epoch_${epochs}.tar"

  if [[ -f "${final_epoch_ckpt}" ]]; then
    echo "[$(date)] SKIP  ${run_id} stage=vm (complete: ${final_epoch_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_step_checkpoint "${run_id}")"
  local cmd=(./rundino.sh "${run_id}")
  if (( resume_step > 0 )); then
    cmd+=("${resume_step}")
  fi
  cmd+=(
    --batch_size "${VM_BATCH_SIZE}"
    --num_workers "${VM_NUM_WORKERS}"
    --prefetch_factor "${VM_PREFETCH_FACTOR}"
    --precision bf16
    --epochs "${epochs}"
  )
  if [[ "${VM_PIN_MEMORY}" == "1" ]]; then
    cmd+=(--pin_memory)
  fi
  if [[ -n "${dataset_mix_json}" ]]; then
    cmd+=(--dataset_mix "${dataset_mix_json}")
  fi

  echo "[$(date)] START ${run_id} stage=vm target_epoch=${epochs} resume_step=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
    if [[ -f "${final_epoch_ckpt}" ]]; then
      echo "[$(date)] END   ${run_id} stage=vm checkpoint=${final_epoch_ckpt}" | tee -a "${SWEEP_DIR}/timeline.log"
      return 0
    fi
    echo "[$(date)] FAIL  ${run_id} stage=vm missing checkpoint=${final_epoch_ckpt}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi

  echo "[$(date)] FAIL  ${run_id} stage=vm (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
  return 1
}

run_mm_stage() {
  local run_id="$1"
  local vm_checkpoint="$2"
  local done_ckpt="logs/${run_id}/step_${MM_TARGET_STEP}.tar"

  if [[ -f "${done_ckpt}" ]] && has_mm_completed_eval "${run_id}" "${MM_TARGET_STEP}"; then
    echo "[$(date)] SKIP  ${run_id} stage=mm (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_step_checkpoint "${run_id}")"
  local cmd=(./runmm.sh "${run_id}")
  if [[ -f "${done_ckpt}" ]] || (( resume_step >= MM_TARGET_STEP )); then
    cmd+=("${MM_TARGET_STEP}")
    cmd+=(
      "${COMMON_MM_ARGS[@]}"
      --vision_checkpoint "${vm_checkpoint}"
      --eval_only
      --eval_batches 0
      --eval_fraction 1.0
      --final_sanity_count 0
    )
    echo "[$(date)] RESUME-EVAL ${run_id} stage=mm checkpoint=${vm_checkpoint}" | tee -a "${SWEEP_DIR}/timeline.log"
  else
    if (( resume_step > 0 )); then
      cmd+=("${resume_step}")
    fi
    cmd+=(
      "${COMMON_MM_ARGS[@]}"
      --vision_checkpoint "${vm_checkpoint}"
    )
  fi

  echo "[$(date)] START ${run_id} stage=mm target_step=${MM_TARGET_STEP} resume_step=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
    echo "[$(date)] END   ${run_id} stage=mm" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  echo "[$(date)] FAIL  ${run_id} stage=mm (see ${SWEEP_DIR}/${run_id}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
  return 1
}

run_family() {
  local family="$1"
  local epochs="$2"
  local dataset_mix_json="$3"

  local vm_run_id="vm_dinovit_srcsweep1_${family}"
  local mm_run_id="mm_dinovit_srcsweep1_${family}"
  local vm_checkpoint="logs/${vm_run_id}/epoch_${epochs}.tar"

  echo "[$(date)] FAMILY ${family} vm=${vm_run_id} mm=${mm_run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
  if ! run_vm_stage "${vm_run_id}" "${epochs}" "${dataset_mix_json}"; then
    echo "[$(date)] FAIL  family=${family} stage=vm" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi
  if [[ "${DRY_RUN}" != "1" ]] && [[ ! -f "${vm_checkpoint}" ]]; then
    echo "[$(date)] FAIL  family=${family} missing vm checkpoint ${vm_checkpoint}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${family}" "${vm_run_id}" "${mm_run_id}" "${epochs}" "${vm_checkpoint}" "${dataset_mix_json}" >> "${SWEEP_DIR}/families.tsv"
  run_mm_stage "${mm_run_id}" "${vm_checkpoint}"
}

run_family "cocoonly" "80" "" || true
run_family "textocr" "65" '{"coco_local:train2014":100,"textocr:train":100}' || true
run_family "cocotext" "67" '{"coco_local:train2014":100,"coco_text:train":100}' || true
run_family "ocrboth" "56" '{"coco_local:train2014":100,"textocr:train":100,"coco_text:train":100}' || true
run_family "inat10" "50" '{"coco_local:train2014":100,"inat2021:train_mini":10}' || true
run_family "ocrboth_inat10" "40" '{"coco_local:train2014":100,"textocr:train":100,"coco_text:train":100,"inat2021:train_mini":10}' || true

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
