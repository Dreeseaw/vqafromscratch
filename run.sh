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
  python train.py "$RUN_ID" "$CKPT_STEP"
else
  python train.py "$RUN_ID"
fi
