#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-./.venv_local/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-mmdualvm_v1_${STAMP}}"
RUN_DIR="logs/${RUN_ID}"
ANCHOR_CKPT="${ANCHOR_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
INIT_MODE="${INIT_MODE:-warm_cement}"
SIGLIP_DIR="${SIGLIP_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"
VITSTR_CKPT="${VITSTR_CKPT:-logs/hf_vision/vitstr_tiny_patch16_224/vitstr_tiny_patch16_224_aug.pth}"
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

TRAIN_ARGS=(
  --vision_model siglip_vitstr_tiny_dual
  --vision_checkpoint "${SIGLIP_DIR}"
  --vision_aux_checkpoint "${VITSTR_CKPT}"
  --seed "${SEED}"
)

case "${INIT_MODE}" in
  warm_cement)
    TRAIN_ARGS+=(--init_from_mm_checkpoint "${ANCHOR_CKPT}")
    ;;
  fresh_bridge)
    ;;
  *)
    echo "Unknown INIT_MODE=${INIT_MODE} (expected warm_cement or fresh_bridge)" >&2
    exit 2
    ;;
esac

STEP="$(latest_step "${RUN_DIR}")"
if (( STEP < 9000 )); then
  if (( STEP > 0 )); then
    ./runmm_v1.sh "${RUN_ID}" "${STEP}" "${TRAIN_ARGS[@]}"
  else
    ./runmm_v1.sh "${RUN_ID}" "${TRAIN_ARGS[@]}"
  fi
fi

BEST_STEP="$(RUN_ID_ENV="${RUN_ID}" "${PYTHON_BIN}" - <<'PY'
import re
import os
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

if [[ ! -f "${RUN_DIR}/dual_best_full_eval.json" ]]; then
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_dualvm_eval \
    --checkpoint "${BEST_CKPT}" \
    --batch_size 96 \
    --eval_batches 0 \
    --output_json "${RUN_DIR}/dual_best_full_eval.json"
fi

if [[ ! -f "${RUN_DIR}/ocr_analysis.json" ]]; then
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_dualvm_ocr_analysis \
    --dual_checkpoint "${BEST_CKPT}" \
    --anchor_checkpoint "${ANCHOR_CKPT}" \
    --batch_size 96 \
    --limit_ocr 500 \
    --limit_control 100 \
    --output_json "${RUN_DIR}/ocr_analysis.json"
fi

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.analyze_dual_vm_experiment \
  --run_dir "${RUN_DIR}" \
  --full_eval_json "${RUN_DIR}/dual_best_full_eval.json" \
  --ocr_analysis_json "${RUN_DIR}/ocr_analysis.json"

echo "${RUN_ID}"
