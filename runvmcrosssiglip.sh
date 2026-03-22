#!/bin/bash
set -e

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id> [checkpoint_step] [extra cross-stage args...]"
  echo
  echo "Example:"
  echo "  $0 vm_recipev1_dino80_cross20_ocrmix --pair_mix '{\"coco_captions_2014:train2014\":100}' --dino_weight_schedule '0.5@0.0,0.1@0.5'"
  exit 1
fi
shift || true

CKPT_STEP=""
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  CKPT_STEP="$1"
  shift || true
fi
EXTRA_ARGS=("$@")
LAUNCH_LOG="logs/$RUN_ID/launcher_siglip_cross.log"

mkdir -pv "logs/$RUN_ID"
if [[ -n "${CKPT_STEP}" ]]; then
  cat models/vm_text_encoder.py models/vit_ssl.py train/vm_recipe_data.py train/vm_siglip_align.py > "logs/$RUN_ID/code_vm_siglip_cross_${CKPT_STEP}.py"
else
  cat models/vm_text_encoder.py models/vit_ssl.py train/vm_recipe_data.py train/vm_siglip_align.py > "logs/$RUN_ID/code_vm_siglip_cross.py"
fi

echo "[$(date)] START run_id=${RUN_ID} checkpoint=${CKPT_STEP:-none}" >> "${LAUNCH_LOG}"
echo "[$(date)] EXTRA_ARGS ${EXTRA_ARGS[*]}" >> "${LAUNCH_LOG}"

PY_CMD=(
  python -m train.vm_siglip_align "$RUN_ID"
)
if [[ -n "${CKPT_STEP}" ]]; then
  PY_CMD+=("${CKPT_STEP}")
fi
PY_CMD+=(
  --device cuda
  --precision bf16
  --phase_name cross
  --allow_tf32
  --matmul_precision high
  --channels_last
  --pretokenize_text
  --optimizer_fused
  --optimizer_foreach
  --log_cuda_memory
)
PY_CMD+=("${EXTRA_ARGS[@]}")

status=0
docker run --rm --gpus all --ipc=host \
  -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "$(pwd)":/app -w /app \
  myrepo:gpu \
  "${PY_CMD[@]}" || status=$?

echo "[$(date)] EXIT status=${status}" >> "${LAUNCH_LOG}"
exit "${status}"
