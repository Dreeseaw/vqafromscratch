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
  cat model.py train.py > "logs/$RUN_ID/code_$CKPT_STEP.py"
  python train.py "$RUN_ID" "$CKPT_STEP"
else
  cat model.py train.py > "logs/$RUN_ID/code.py"
  python train.py "$RUN_ID"
fi
