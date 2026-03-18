#!/bin/bash
set -e

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id> [checkpoint_step] [extra dino args...]"
  echo
  echo "Example:"
  echo "  $0 dino_vittiny_scratch --data_dir images/train2014 --epochs 100"
  exit 1
fi
shift || true

CKPT_STEP=""
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  CKPT_STEP="$1"
  shift || true
fi
EXTRA_ARGS=("$@")
LAUNCH_LOG="logs/$RUN_ID/launcher.log"

mkdir -pv "logs/$RUN_ID"
if [[ -n "${CKPT_STEP}" ]]; then
  cat models/vit_ssl.py train/dino_ssl.py evals/ssl_knn.py > "logs/$RUN_ID/code_dino_ssl_${CKPT_STEP}.py"
else
  cat models/vit_ssl.py train/dino_ssl.py evals/ssl_knn.py > "logs/$RUN_ID/code_dino_ssl.py"
fi

CMD=(
  python -m train.dino_ssl "$RUN_ID"
)

if [[ -n "${CKPT_STEP}" ]]; then
  CMD+=("${CKPT_STEP}")
fi

echo "[$(date)] START run_id=${RUN_ID} checkpoint=${CKPT_STEP:-none}" >> "${LAUNCH_LOG}"
echo "[$(date)] CMD docker run --rm --gpus all --ipc=host -e PYTORCH_ENABLE_MPS_FALLBACK=1 -v $(pwd):/app -w /app myrepo:gpu ${CMD[*]} --device cuda --batch_size 128 --num_workers 10 --prefetch_factor 1 --pin_memory --precision bf16 ${EXTRA_ARGS[*]}" >> "${LAUNCH_LOG}"

status=0
docker run --rm --gpus all --ipc=host \
  -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
  -v "$(pwd)":/app -w /app myrepo:gpu \
  "${CMD[@]}" \
  --device cuda \
  --batch_size 128 \
  --num_workers 10 \
  --prefetch_factor 1 \
  --pin_memory \
  --precision bf16 \
  "${EXTRA_ARGS[@]}" || status=$?

echo "[$(date)] EXIT status=${status}" >> "${LAUNCH_LOG}"
exit "${status}"
