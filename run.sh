#!/bin/bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_id> [checkpoint_step]"
  exit 1
fi

RUN_ID="$1"

mkdir -pv "logs/$RUN_ID" 
if [ $# -ge 2 ]; then
  CKPT_STEP="$2"
  cat models/vae.py train/train.py > "logs/$RUN_ID/code_$CKPT_STEP.py"
  python -m train.train "$RUN_ID" "$CKPT_STEP"
else
  cat models/vae.py train/train.py > "logs/$RUN_ID/code.py"
  python -m train.train "$RUN_ID"
fi
