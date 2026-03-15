#!/bin/bash
set -e

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id> [extra caption_pretrain args...]"
  exit 1
fi
shift || true
EXTRA_ARGS=("$@")

mkdir -pv "logs/$RUN_ID"
cat models/bridge.py train/caption_data.py train/caption_pretrain.py > "logs/$RUN_ID/code_capalign.py"

docker run --rm --gpus all --ipc=host \
  -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
  -v "$(pwd)":/app -w /app myrepo:gpu \
  python -m train.caption_pretrain "$RUN_ID" \
  --tokenizer_path logs/mix_bpe_16k/tokenizer.pt \
  --lm_checkpoint logs/lm_boom2/step_45000.tar \
  --images_root images \
  --annotations_root data/vqav2 \
  "${EXTRA_ARGS[@]}"
