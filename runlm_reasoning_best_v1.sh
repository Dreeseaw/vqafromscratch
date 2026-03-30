#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/scripts/runtime_exec.sh"

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id> [checkpoint_step] [extra train args...]"
  exit 1
fi
shift

CHECKPOINT_STEP=""
if [[ $# -gt 0 && "${1}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT_STEP="$1"
  shift
fi
EXTRA_ARGS=("$@")

DISTILL_REASONING_BUCKET="${DISTILL_REASONING_BUCKET:-data/pretraining/distill256_reasoningmix_v1/train}"
REPEAT_DISTILL="${REPEAT_DISTILL:-1}"
REPEAT_SQUAD="${REPEAT_SQUAD:-1}"
REPEAT_MULTINLI="${REPEAT_MULTINLI:-1}"
REPEAT_COUNTING="${REPEAT_COUNTING:-1}"

runtime_exec_python scripts/build_lm_reasoning_mix_bucket.py \
  --output_dir "${DISTILL_REASONING_BUCKET}" \
  --source distill data/pretraining/distill256_cleaned2/train \
  --source squad data/pretraining/squad256_reasoning/train \
  --source multinli data/pretraining/multinli256_reasoning/train \
  --source counting data/pretraining/counting256_reasoning/train \
  --repeat distill "${REPEAT_DISTILL}" \
  --repeat squad "${REPEAT_SQUAD}" \
  --repeat multinli "${REPEAT_MULTINLI}" \
  --repeat counting "${REPEAT_COUNTING}"

TRAIN_CMD=(
  -m train.train_transformer "${RUN_ID}"
  --train_data data/pretraining/wikicoco256_cleaned
  --train_bucket_wiki data/pretraining/wikicoco256_cleaned/train
  --train_bucket_distill "${DISTILL_REASONING_BUCKET}"
  --mix_schedule configs/mix_schedule1.json
  --val_data data/pretraining/distill256_cleaned2/val
  --test_data data/pretraining/wikicoco256_cleaned/test
  --tokenizer logs/mix_bpe_16k/tokenizer.pt
  --tie_embeddings
  --debug_cuda_empty_cache=1
  --epochs=100
  --warmup_ratio=0.004
  --num_workers=8
  --persistent_workers
  --prefetch_factor 8
  --run_probes=1000
  --probe_after_log_only
  --eval_every_steps=5000
  --decoder_only
  --dec_layers=12
  --ff_mult=2
  --n_heads=8
  --d_model=512
  --attn_impl sdpa
  --no_activation_checkpointing
  --sdp_backend=flash
  --precision=bf16
  --swiglu
  --row_max_norm_c=2.0
  --probe_layers=0,3,7,11
  --muon
  --muon_ns_steps=5
  --muon_min_matrix_dim=100
  --lr=0.001
)

if [[ -n "${CHECKPOINT_STEP}" ]]; then
  TRAIN_CMD+=(--checkpoint "${CHECKPOINT_STEP}")
fi
TRAIN_CMD+=("${EXTRA_ARGS[@]}")

runtime_exec_python "${TRAIN_CMD[@]}"
