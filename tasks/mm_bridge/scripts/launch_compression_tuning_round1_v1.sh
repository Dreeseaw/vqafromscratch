#!/bin/bash
set -euo pipefail

RUN_ID="${RUN_ID:-mmcompress_round1_k16_v1}"
ANCHOR_CKPT="${ANCHOR_CKPT:-/home/wdree/percy/vqafromscratch/logs/mmcement_v1_20260316_siglip_cement_questiononly_s53/step_8000.tar}"

exec ./runmm_v1.sh "${RUN_ID}" \
  --freeze_mode semantic_adapter_only \
  --init_from_mm_checkpoint "${ANCHOR_CKPT}" \
  --semantic_teacher_checkpoint "${ANCHOR_CKPT}" \
  --use_compression \
  --compression_k 16 \
  --compression_distill_weight 0.1 \
  --semantic_consistency_loss_weight 0.0 \
  --no-use_grounding_loss \
  --batch_size 96 \
  --grad_accum_steps 2 \
  --eval_batch_size 96 \
  --max_steps 3000 \
  --manual_max_steps \
  --lr 2e-4 \
  --lr_schedule cosine \
  --lr_warmup_steps 200 \
  --lr_min_ratio 0.15 \
  --eval_every 500 \
  --ckpt_every 500 \
  --eval_batches 100 \
  --final_eval_batches 0 \
  --seed 35 \
  "$@"
