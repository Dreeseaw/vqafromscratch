#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-mmcement_gqa7030_v1_${STAMP}}"
RUN_DIR="logs/${RUN_ID}"
SIGLIP_DIR="${SIGLIP_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"
LM_CKPT="${LM_CKPT:-logs/lm_final/step_45000.tar}"
GQA_ROOT="${GQA_ROOT:-data/gqa}"
GQA_TRAIN_FRACTION="${GQA_TRAIN_FRACTION:-0.1}"
SEED="${SEED:-42}"

mkdir -p "${RUN_DIR}"

latest_step() {
  local run_dir="$1"
  local best=0
  local path base step
  shopt -s nullglob
  for path in "${run_dir}"/step_*.tar; do
    base="$(basename "${path}")"
    step="${base#step_}"
    step="${step%.tar}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > best )); then
      best="${step}"
    fi
  done
  shopt -u nullglob
  echo "${best}"
}

RUN_ARGS=(
  --vision_model siglip_base
  --vision_checkpoint "${SIGLIP_DIR}"
  --lm_checkpoint "${LM_CKPT}"
  --seed "${SEED}"
  --max_steps 9000
  --manual_max_steps
  --log_every 10
  --num_workers 2
  --prefetch_factor 1
  --no-pin_memory
  --batch_size 96
  --grad_accum_steps 2
  --eval_batch_size 160
  --gqa_root "${GQA_ROOT}"
  --gqa_train_mix_ratio 0.3
  --gqa_train_fraction "${GQA_TRAIN_FRACTION}"
  --min_train_steps_per_s 0
)

STEP="$(latest_step "${RUN_DIR}")"
if (( STEP < 9000 )); then
  if (( STEP > 0 )); then
    ./runmm_v1.sh "${RUN_ID}" "${STEP}" "${RUN_ARGS[@]}"
  else
    ./runmm_v1.sh "${RUN_ID}" "${RUN_ARGS[@]}"
  fi
fi

BEST_STEP="$(RUN_ID_ENV="${RUN_ID}" python3 - <<'PY'
import os
import re
from pathlib import Path

run_dir = Path("logs") / Path(os.environ["RUN_ID_ENV"])
log_path = run_dir / "logfile.txt"
best_step = 9000
best_acc = -1.0
pending = None
for line in log_path.read_text(encoding="utf-8").splitlines():
    m_eval = re.search(r"\[eval:val\] overall_accuracy=([0-9.]+)", line)
    if m_eval:
        pending = float(m_eval.group(1))
        continue
    m_tag = re.search(r"fixed-eval answers appended: .* step=(\d+) tag=periodic_eval", line)
    if m_tag and pending is not None:
        step = int(m_tag.group(1))
        if pending > best_acc:
            best_acc = pending
            best_step = step
        pending = None
print(best_step)
PY
)"

BEST_CKPT="${RUN_DIR}/step_${BEST_STEP}.tar"

if [[ ! -f "${RUN_DIR}/best_step.json" ]]; then
  printf '{\n  "best_step": %s\n}\n' "${BEST_STEP}" > "${RUN_DIR}/best_step.json"
fi

if [[ ! -f "${RUN_DIR}/best_full_eval.json" ]]; then
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_dualvm_eval \
    --checkpoint "${BEST_CKPT}" \
    --batch_size 160 \
    --num_workers 1 \
    --no-pin_memory \
    --eval_batches 0 \
    --output_json "${RUN_DIR}/best_full_eval.json"
fi

if [[ ! -f "${RUN_DIR}/best_gqa_eval_5k.json" ]]; then
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_gqa_eval \
    --checkpoint "${BEST_CKPT}" \
    --batch_size 64 \
    --num_workers 1 \
    --no-pin_memory \
    --gqa_root "${GQA_ROOT}" \
    --limit_eval 5000 \
    --output_json "${RUN_DIR}/best_gqa_eval_5k.json"
fi

echo "${RUN_ID}"
