#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmsemantic_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmsemantic_v1_latest"

RUN_PREFIX="${RUN_PREFIX:-mmsemantic_v1_20260322}"
ANCHOR_CKPT="${ANCHOR_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
SEED="${SEED:-53}"
MAX_STEPS="${MAX_STEPS:-4000}"
TARGET_STEP="${TARGET_STEP:-${MAX_STEPS}}"
EVAL_EVERY="${EVAL_EVERY:-500}"
CKPT_EVERY="${CKPT_EVERY:-500}"
RUN_FULL_SWEEP="${RUN_FULL_SWEEP:-0}"
SKIP_K16="${SKIP_K16:-0}"
DRY_RUN="${DRY_RUN:-0}"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Semantic Compression Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Plan source:
- tasks/mm_bridge/docs/48_semantic_compression_week_plan_2026-03-22.md

Correct Cement anchor:
- ${ANCHOR_CKPT}
- use completed full-eval Cement question_only anchor, not the step_8000 mini-eval peak

Default pilot policy:
- run K=16 first
- max_steps=${TARGET_STEP}
- eval_every=${EVAL_EVERY}
- ckpt_every=${CKPT_EVERY}
- full K sweep only if RUN_FULL_SWEEP=1
- skip pilot repeat only if SKIP_K16=1

Locked training mode:
- freeze_mode=semantic_adapter_only
- init_from_mm_checkpoint=${ANCHOR_CKPT}
- semantic_teacher_checkpoint=${ANCHOR_CKPT}
- semantic_latent_dim=256
- semantic_recon_loss_weight=0.1
- semantic_consistency_loss_weight=0.1
EOF

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

has_completion_marker() {
  local run_id="$1"
  local target="$2"
  local answers_path="logs/${run_id}/fixed_eval_val_answers.jsonl"
  local pattern="\"global_step\": ${target}.*\"tag\": \"(final_eval|eval_only)\""
  [[ -f "${answers_path}" ]] && rg -q "${pattern}" "${answers_path}"
}

run_one() {
  local run_name="$1"
  local semantic_tokens="$2"
  local done_ckpt="logs/${run_name}/step_${TARGET_STEP}.tar"
  local resume_step
  resume_step="$(latest_ckpt_step "${run_name}")"
  local stdout_log="${SWEEP_DIR}/${run_name}.stdout.log"
  local cmd=()

  if [[ -f "${done_ckpt}" ]] && has_completion_marker "${run_name}" "${TARGET_STEP}"; then
    echo "[$(date)] SKIP  ${run_name} (complete: ${done_ckpt})" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi

  if [[ -f "${done_ckpt}" ]] || (( resume_step >= TARGET_STEP )); then
    cmd=(
      ./runmm_v1.sh "${run_name}" "${TARGET_STEP}"
      --eval_only
      --eval_batches 0
      --eval_batch_size 96
    )
    echo "[$(date)] RESUME-EVAL ${run_name} from step_${TARGET_STEP}.tar" | tee -a "${SWEEP_DIR}/timeline.log"
  else
    cmd=(
      ./runmm_v1.sh "${run_name}"
      --seed "${SEED}"
      --max_steps "${TARGET_STEP}"
      --manual_max_steps
      --eval_every "${EVAL_EVERY}"
      --ckpt_every "${CKPT_EVERY}"
      --freeze_mode semantic_adapter_only
      --bridge_question_context_mode question_only
      --semantic_bottleneck
      --semantic_tokens "${semantic_tokens}"
      --semantic_latent_dim 256
      --semantic_recon_loss_weight 0.1
      --semantic_consistency_loss_weight 0.1
      --init_from_mm_checkpoint "${ANCHOR_CKPT}"
      --semantic_teacher_checkpoint "${ANCHOR_CKPT}"
      --grad_accum_steps=1
      --batch_size=192
      --eval_batch_size=96
    )
    if (( resume_step > 0 )); then
      cmd=(
        ./runmm_v1.sh "${run_name}" "${resume_step}"
        "${cmd[@]:2}"
      )
    fi
  fi

  echo "[$(date)] START ${run_name} semantic_tokens=${semantic_tokens} target=${TARGET_STEP} resume=${resume_step}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${run_name}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  "${cmd[@]}" >> "${stdout_log}" 2>&1
  echo "[$(date)] END   ${run_name}" | tee -a "${SWEEP_DIR}/timeline.log"
}

if [[ "${SKIP_K16}" != "1" ]]; then
  run_one "${RUN_PREFIX}_k16" 16
fi

if [[ "${RUN_FULL_SWEEP}" == "1" ]]; then
  run_one "${RUN_PREFIX}_k32" 32
  run_one "${RUN_PREFIX}_k8" 8
  run_one "${RUN_PREFIX}_k4" 4
  run_one "${RUN_PREFIX}_k2" 2
fi

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
