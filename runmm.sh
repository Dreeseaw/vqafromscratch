#!/bin/bash
set -e

source "$(cd "$(dirname "$0")" && pwd)/scripts/runtime_exec.sh"

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id> [checkpoint_step] [extra mm args...]"
  echo
  echo "Example:"
  echo "  $0 mm_exp1 --vision_model vaer --vision_checkpoint logs/vae_fast3/step_4001.tar --lm_checkpoint logs/lm_final/step_5000.tar"
  exit 1
fi
shift || true

CKPT_STEP=""
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  CKPT_STEP="$1"
  shift || true
fi
EXTRA_ARGS=("$@")

mkdir -pv "logs/$RUN_ID"
if [[ -n "${CKPT_STEP}" ]]; then
  cat models/bridge.py train/vqa_data.py train/mm.py evals/vqa.py > "logs/$RUN_ID/code_mm_$CKPT_STEP.py"
else
  cat models/bridge.py train/vqa_data.py train/mm.py evals/vqa.py > "logs/$RUN_ID/code_mm.py"
fi

CMD=(
  -m train.mm "$RUN_ID"
  --auto_download
  --no-download_images
  --images_root images
  --annotations_root data/vqav2
  --tokenizer_path logs/mix_bpe_16k/tokenizer.pt
  --mm_sdp_backend math
)

if [[ -n "${CKPT_STEP}" ]]; then
  CMD+=(--checkpoint "$CKPT_STEP")
fi

runtime_exec_python \
  "${CMD[@]}" \
  --vision_model=vae --vision_checkpoint=logs/vm_base2/step_15001.tar --lm_checkpoint=logs/lm_final/step_45000.tar \
  --batch_size=256 --epochs=10 --eval_every=0 --eval_batches=0 --limit_eval=0 --eval_scorer=official \
  --num_visual_tokens 49 --bridge_token_reduce all \
  "${EXTRA_ARGS[@]}"
