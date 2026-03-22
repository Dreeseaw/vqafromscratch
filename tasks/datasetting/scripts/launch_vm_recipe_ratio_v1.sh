#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"

DRY_RUN="${DRY_RUN:-0}"
FAMILY="${FAMILY:-ocrmix1}"

STAGE1_RATIO_PERCENT="${STAGE1_RATIO_PERCENT:-80}"
STAGE2_RATIO_PERCENT="${STAGE2_RATIO_PERCENT:-20}"
STAGE3_RATIO_PERCENT="${STAGE3_RATIO_PERCENT:-0}"

for value in "${STAGE1_RATIO_PERCENT}" "${STAGE2_RATIO_PERCENT}" "${STAGE3_RATIO_PERCENT}"; do
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "[vmrecipe] ERROR stage ratios must be integers"
    exit 1
  fi
done
if (( STAGE1_RATIO_PERCENT <= 0 )); then
  echo "[vmrecipe] ERROR stage 1 ratio must be > 0"
  exit 1
fi
if (( STAGE2_RATIO_PERCENT < 0 || STAGE3_RATIO_PERCENT < 0 )); then
  echo "[vmrecipe] ERROR stage 2/3 ratios must be >= 0"
  exit 1
fi
if (( STAGE2_RATIO_PERCENT == 0 && STAGE3_RATIO_PERCENT == 0 )); then
  echo "[vmrecipe] ERROR stage 2 and stage 3 cannot both be 0"
  exit 1
fi

RATIO_TAG="s1_${STAGE1_RATIO_PERCENT}_s2_${STAGE2_RATIO_PERCENT}_s3_${STAGE3_RATIO_PERCENT}"
SWEEP_ID="ds_vmrecipe_${RATIO_TAG}_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/ds_vmrecipe_${RATIO_TAG}_v1_latest"

VM_RUN_ID="${VM_RUN_ID:-vm_recipev1_${RATIO_TAG}_${FAMILY}}"
MM_RUN_ID="${MM_RUN_ID:-mm_recipev1_${RATIO_TAG}_${FAMILY}}"

DINO_DATASET_MIX="${DINO_DATASET_MIX-}"
PAIR_MIX="${PAIR_MIX-}"
case "${FAMILY}" in
  limbo_coco)
    if [[ -z "${DINO_DATASET_MIX}" ]]; then
      DINO_DATASET_MIX='{"coco_local:train2014":100}'
    fi
    if [[ -z "${PAIR_MIX}" ]]; then
      PAIR_MIX='{"coco_captions_2014:train2014":100}'
    fi
    ;;
  limbo_ocrmix|limbo_ocrtail)
    if [[ -z "${DINO_DATASET_MIX}" ]]; then
      DINO_DATASET_MIX='{"coco_local:train2014":100,"textocr:train":100,"coco_text:train":100}'
    fi
    if [[ -z "${PAIR_MIX}" ]]; then
      PAIR_MIX='{"coco_captions_2014:train2014":100,"coco_text_captions:train":100}'
    fi
    ;;
  *)
    if [[ -z "${DINO_DATASET_MIX}" ]]; then
      DINO_DATASET_MIX='{"coco_local:train2014":100,"textocr:train":100,"coco_text:train":100}'
    fi
    if [[ -z "${PAIR_MIX}" ]]; then
      PAIR_MIX='{"coco_captions_2014:train2014":100,"coco_text_captions:train":100}'
    fi
    ;;
esac

DINO_EPOCHS="${DINO_EPOCHS:-70}"
VM_BATCH_SIZE="${VM_BATCH_SIZE:-128}"
VM_NUM_WORKERS="${VM_NUM_WORKERS:-10}"
VM_PREFETCH_FACTOR="${VM_PREFETCH_FACTOR:-1}"
VM_PIN_MEMORY="${VM_PIN_MEMORY:-1}"

CROSS_BATCH_SIZE="${CROSS_BATCH_SIZE:-176}"
CROSS_NUM_WORKERS="${CROSS_NUM_WORKERS:-6}"
CROSS_PREFETCH_FACTOR="${CROSS_PREFETCH_FACTOR:-2}"
CROSS_WARMUP_STEPS="${CROSS_WARMUP_STEPS:-500}"
CROSS_LR="${CROSS_LR:-0.0002}"
CROSS_DINO_WEIGHT_SCHEDULE="${CROSS_DINO_WEIGHT_SCHEDULE:-0.5@0.0}"
PAIR_MAX_PAIRS="${PAIR_MAX_PAIRS:-0}"

ALIGN_BATCH_SIZE="${ALIGN_BATCH_SIZE:-128}"
ALIGN_NUM_WORKERS="${ALIGN_NUM_WORKERS:-4}"
ALIGN_PREFETCH_FACTOR="${ALIGN_PREFETCH_FACTOR:-2}"
ALIGN_WARMUP_STEPS="${ALIGN_WARMUP_STEPS:-500}"
ALIGN_LR="${ALIGN_LR:-0.0002}"
VM_MAX_IMAGES="${VM_MAX_IMAGES:-0}"

MM_TARGET_STEP="${MM_TARGET_STEP:-9000}"
MM_BATCH_SIZE="${MM_BATCH_SIZE:-96}"
MM_GRAD_ACCUM="${MM_GRAD_ACCUM:-2}"
MM_EVAL_BATCH_SIZE="${MM_EVAL_BATCH_SIZE:-96}"
MM_NUM_WORKERS="${MM_NUM_WORKERS:-4}"
MM_PREFETCH_FACTOR="${MM_PREFETCH_FACTOR:-2}"
MM_SEED="${MM_SEED:-35}"
MM_LOG_EVERY="${MM_LOG_EVERY:-20}"
MM_EVAL_EVERY="${MM_EVAL_EVERY:-1000}"
MM_EVAL_BATCHES="${MM_EVAL_BATCHES:-100}"
MM_FINAL_EVAL_BATCHES="${MM_FINAL_EVAL_BATCHES:-0}"
MM_CKPT_EVERY="${MM_CKPT_EVERY:-1000}"

normalize_json_object() {
  local label="$1"
  local raw="$2"
  python3 - "${label}" "${raw}" <<'PY'
import json
import sys

label = sys.argv[1]
raw = sys.argv[2].strip()
try:
    value = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f"[vmrecipe] ERROR {label} must be valid JSON object text: {raw!r} ({exc})")
if not isinstance(value, dict) or not value:
    raise SystemExit(f"[vmrecipe] ERROR {label} must be a non-empty JSON object: {raw!r}")
print(json.dumps(value, separators=(',', ':'), sort_keys=True))
PY
}

DINO_DATASET_MIX="$(normalize_json_object "DINO_DATASET_MIX" "${DINO_DATASET_MIX}")"
PAIR_MIX="$(normalize_json_object "PAIR_MIX" "${PAIR_MIX}")"

cat > "${SWEEP_DIR}/README.md" <<EOF
# VM Recipe Three-Stage Ratio Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Stages:
1. SSL-only DINO
2. SSL + SigLIP-style image-text cross-training
3. SigLIP-only image-text alignment
4. Fixed downstream MM eval

Family: ${FAMILY}
VM run: ${VM_RUN_ID}
MM run: ${MM_RUN_ID}

STAGE1_RATIO_PERCENT=${STAGE1_RATIO_PERCENT}
STAGE2_RATIO_PERCENT=${STAGE2_RATIO_PERCENT}
STAGE3_RATIO_PERCENT=${STAGE3_RATIO_PERCENT}
DINO_DATASET_MIX=${DINO_DATASET_MIX}
PAIR_MIX=${PAIR_MIX}
DINO_EPOCHS=${DINO_EPOCHS}
CROSS_DINO_WEIGHT_SCHEDULE=${CROSS_DINO_WEIGHT_SCHEDULE}
EOF

printf "family\tvm_run_id\tmm_run_id\tstage1_ratio\tstage2_ratio\tstage3_ratio\tdino_epochs\tstage2_steps\tstage3_steps\tdino_dataset_mix\tpair_mix\tcross_dino_weight_schedule\n" > "${SWEEP_DIR}/families.tsv"

COMMON_MM_ARGS=(
  --precision bf16
  --num_workers "${MM_NUM_WORKERS}"
  --prefetch_factor "${MM_PREFETCH_FACTOR}"
  --epochs 400
  --max_steps "${MM_TARGET_STEP}"
  --manual_max_steps
  --log_every "${MM_LOG_EVERY}"
  --eval_every "${MM_EVAL_EVERY}"
  --eval_batches "${MM_EVAL_BATCHES}"
  --final_eval_batches "${MM_FINAL_EVAL_BATCHES}"
  --eval_log_every 20
  --eval_fraction 1.0
  --ckpt_every "${MM_CKPT_EVERY}"
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

compute_stage_steps() {
  local stage1_steps="$1"
  local stage_ratio="$2"
  if (( stage1_steps <= 0 || stage_ratio == 0 )); then
    echo "0"
    return
  fi
  local computed=$(( (stage1_steps * stage_ratio + STAGE1_RATIO_PERCENT / 2) / STAGE1_RATIO_PERCENT ))
  if (( computed <= 0 )); then
    computed=1
  fi
  echo "${computed}"
}

run_vm_stage() {
  local final_epoch_ckpt="logs/${VM_RUN_ID}/epoch_${DINO_EPOCHS}.tar"
  if [[ -f "${final_epoch_ckpt}" ]]; then
    echo "[$(date)] SKIP  ${VM_RUN_ID} stage=ssl (complete: ${final_epoch_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_step_checkpoint "${VM_RUN_ID}")"
  local cmd=(./rundino.sh "${VM_RUN_ID}")
  if (( resume_step > 0 )); then
    cmd+=("${resume_step}")
  fi
  cmd+=(
    --batch_size "${VM_BATCH_SIZE}"
    --num_workers "${VM_NUM_WORKERS}"
    --prefetch_factor "${VM_PREFETCH_FACTOR}"
    --precision bf16
    --epochs "${DINO_EPOCHS}"
    --max_images "${VM_MAX_IMAGES}"
    --dataset_mix "${DINO_DATASET_MIX}"
  )
  if [[ "${VM_PIN_MEMORY}" == "1" ]]; then
    cmd+=(--pin_memory)
  fi

  echo "[$(date)] START ${VM_RUN_ID} stage=ssl target_epoch=${DINO_EPOCHS} resume_step=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${VM_RUN_ID} stage=ssl" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  if "${cmd[@]}" >> "${SWEEP_DIR}/${VM_RUN_ID}.stdout.log" 2>&1; then
    if [[ -f "${final_epoch_ckpt}" ]]; then
      echo "[$(date)] END   ${VM_RUN_ID} stage=ssl checkpoint=${final_epoch_ckpt}" | tee -a "${SWEEP_DIR}/timeline.log"
      return 0
    fi
    echo "[$(date)] FAIL  ${VM_RUN_ID} stage=ssl missing checkpoint=${final_epoch_ckpt}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi
  echo "[$(date)] FAIL  ${VM_RUN_ID} stage=ssl (see ${SWEEP_DIR}/${VM_RUN_ID}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
  return 1
}

run_cross_stage() {
  local stage1_steps="$1"
  local stage2_steps="$2"
  if (( STAGE2_RATIO_PERCENT == 0 )); then
    echo "[$(date)] SKIP  ${VM_RUN_ID} stage=cross ratio=0" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${VM_RUN_ID} stage=cross target_phase_steps=${stage2_steps}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  local dino_final_step
  dino_final_step="$(latest_step_checkpoint "${VM_RUN_ID}")"
  if (( dino_final_step < stage1_steps )); then
    echo "[$(date)] FAIL  ${VM_RUN_ID} stage=cross missing SSL checkpoint" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi
  local cmd=(./runvmcrosssiglip.sh "${VM_RUN_ID}")
  cmd+=(
    --pair_mix "${PAIR_MIX}"
    --max_pairs "${PAIR_MAX_PAIRS}"
    --batch_size "${CROSS_BATCH_SIZE}"
    --num_workers "${CROSS_NUM_WORKERS}"
    --prefetch_factor "${CROSS_PREFETCH_FACTOR}"
    --phase_steps "${stage2_steps}"
    --warmup_steps "${CROSS_WARMUP_STEPS}"
    --lr "${CROSS_LR}"
    --dino_weight_schedule "${CROSS_DINO_WEIGHT_SCHEDULE}"
  )
  echo "[$(date)] START ${VM_RUN_ID} stage=cross target_phase_steps=${stage2_steps}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if "${cmd[@]}" >> "${SWEEP_DIR}/${VM_RUN_ID}.stdout.log" 2>&1; then
    echo "[$(date)] END   ${VM_RUN_ID} stage=cross" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  echo "[$(date)] FAIL  ${VM_RUN_ID} stage=cross (see ${SWEEP_DIR}/${VM_RUN_ID}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
  return 1
}

run_align_stage() {
  local stage3_steps="$1"
  if (( STAGE3_RATIO_PERCENT == 0 )); then
    echo "[$(date)] SKIP  ${VM_RUN_ID} stage=align ratio=0" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${VM_RUN_ID} stage=align target_phase_steps=${stage3_steps}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  local cmd=(./runvmsiglip.sh "${VM_RUN_ID}")
  cmd+=(
    --pair_mix "${PAIR_MIX}"
    --max_pairs "${PAIR_MAX_PAIRS}"
    --batch_size "${ALIGN_BATCH_SIZE}"
    --num_workers "${ALIGN_NUM_WORKERS}"
    --prefetch_factor "${ALIGN_PREFETCH_FACTOR}"
    --phase_steps "${stage3_steps}"
    --warmup_steps "${ALIGN_WARMUP_STEPS}"
    --lr "${ALIGN_LR}"
  )
  echo "[$(date)] START ${VM_RUN_ID} stage=align target_phase_steps=${stage3_steps}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if "${cmd[@]}" >> "${SWEEP_DIR}/${VM_RUN_ID}.stdout.log" 2>&1; then
    echo "[$(date)] END   ${VM_RUN_ID} stage=align" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  echo "[$(date)] FAIL  ${VM_RUN_ID} stage=align (see ${SWEEP_DIR}/${VM_RUN_ID}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
  return 1
}

run_mm_stage() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${MM_RUN_ID} stage=mm" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local final_vm_step
  final_vm_step="$(latest_step_checkpoint "${VM_RUN_ID}")"
  if (( final_vm_step <= 0 )); then
    echo "[$(date)] FAIL  ${MM_RUN_ID} stage=mm missing VM checkpoint" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi
  local vm_checkpoint="logs/${VM_RUN_ID}/step_${final_vm_step}.tar"
  local done_ckpt="logs/${MM_RUN_ID}/step_${MM_TARGET_STEP}.tar"

  if [[ -f "${done_ckpt}" ]] && has_mm_completed_eval "${MM_RUN_ID}" "${MM_TARGET_STEP}"; then
    echo "[$(date)] SKIP  ${MM_RUN_ID} stage=mm (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  local resume_step
  resume_step="$(latest_step_checkpoint "${MM_RUN_ID}")"
  local cmd=(./runmm.sh "${MM_RUN_ID}")
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
    echo "[$(date)] RESUME-EVAL ${MM_RUN_ID} stage=mm checkpoint=${vm_checkpoint}" | tee -a "${SWEEP_DIR}/timeline.log"
  else
    if (( resume_step > 0 )); then
      cmd+=("${resume_step}")
    fi
    cmd+=(
      "${COMMON_MM_ARGS[@]}"
      --vision_checkpoint "${vm_checkpoint}"
    )
  fi

  echo "[$(date)] START ${MM_RUN_ID} stage=mm target_step=${MM_TARGET_STEP} resume_step=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if "${cmd[@]}" >> "${SWEEP_DIR}/${MM_RUN_ID}.stdout.log" 2>&1; then
    echo "[$(date)] END   ${MM_RUN_ID} stage=mm" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  echo "[$(date)] FAIL  ${MM_RUN_ID} stage=mm (see ${SWEEP_DIR}/${MM_RUN_ID}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
  return 1
}

STAGE1_STEPS_FOR_DRY_RUN=52320
STAGE2_STEPS="$(compute_stage_steps "${STAGE1_STEPS_FOR_DRY_RUN}" "${STAGE2_RATIO_PERCENT}")"
STAGE3_STEPS="$(compute_stage_steps "${STAGE1_STEPS_FOR_DRY_RUN}" "${STAGE3_RATIO_PERCENT}")"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "${FAMILY}" "${VM_RUN_ID}" "${MM_RUN_ID}" "${STAGE1_RATIO_PERCENT}" "${STAGE2_RATIO_PERCENT}" "${STAGE3_RATIO_PERCENT}" \
  "${DINO_EPOCHS}" "${STAGE2_STEPS}" "${STAGE3_STEPS}" "${DINO_DATASET_MIX}" "${PAIR_MIX}" "${CROSS_DINO_WEIGHT_SCHEDULE}" >> "${SWEEP_DIR}/families.tsv"

echo "[$(date)] FAMILY ${FAMILY} vm=${VM_RUN_ID} mm=${MM_RUN_ID} ratios=${STAGE1_RATIO_PERCENT}/${STAGE2_RATIO_PERCENT}/${STAGE3_RATIO_PERCENT}" | tee -a "${SWEEP_DIR}/timeline.log"
run_vm_stage

REAL_STAGE1_STEPS="$(latest_step_checkpoint "${VM_RUN_ID}")"
if [[ "${DRY_RUN}" == "1" ]]; then
  REAL_STAGE1_STEPS="${STAGE1_STEPS_FOR_DRY_RUN}"
fi
STAGE2_STEPS="$(compute_stage_steps "${REAL_STAGE1_STEPS}" "${STAGE2_RATIO_PERCENT}")"
STAGE3_STEPS="$(compute_stage_steps "${REAL_STAGE1_STEPS}" "${STAGE3_RATIO_PERCENT}")"

run_cross_stage "${REAL_STAGE1_STEPS}" "${STAGE2_STEPS}"
run_align_stage "${STAGE3_STEPS}"
run_mm_stage

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
