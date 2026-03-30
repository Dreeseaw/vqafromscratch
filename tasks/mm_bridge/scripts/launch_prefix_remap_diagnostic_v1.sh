#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-mmsemantic_remap_v1_${STAMP}}"
K8_CKPT="${K8_CKPT:-logs/mmsemantic_v1_20260322_k8/step_4000.tar}"
SEED="${SEED:-53}"

./runmm_v1.sh "${RUN_ID}" \
  --vision_model siglip_base \
  --vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224 \
  --lm_checkpoint logs/lm_final/step_45000.tar \
  --seed "${SEED}" \
  --max_steps 500 \
  --manual_max_steps \
  --batch_size 192 \
  --grad_accum_steps 1 \
  --eval_batch_size 96 \
  --eval_every 100 \
  --eval_batches 100 \
  --ckpt_every 100 \
  --final_eval_batches 0 \
  --lr 0.001 \
  --lr_schedule constant \
  --lr_warmup_steps 0 \
  --freeze_mode prefix_remap_only \
  --bridge_question_context_mode question_only \
  --bridge_query_bank_mode question_hidden_attn \
  --semantic_bottleneck \
  --semantic_tokens 8 \
  --semantic_latent_dim 256 \
  --use_prefix_remap \
  --disable_lm_visual_adapters \
  --init_from_mm_checkpoint "${K8_CKPT}" \
  --min_train_steps_per_s 0

./.venv_local/bin/python -m tasks.mm_bridge.scripts.analyze_prefix_remap_diagnostic \
  --run_dir "logs/${RUN_ID}"

echo "${RUN_ID}"
