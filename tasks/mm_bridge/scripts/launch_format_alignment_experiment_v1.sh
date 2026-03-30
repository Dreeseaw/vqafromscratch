#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-./.venv_local/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-mmsemantic_format_v1_${STAMP}}"
RUN_DIR="logs/${RUN_ID}"
CEMENT_CKPT="${CEMENT_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s53/step_8000.tar}"
REMAP_CKPT="${REMAP_CKPT:-logs/mmsemantic_remap_v1_debug/step_500.tar}"
SEED="${SEED:-53}"

if [[ ! -f "${RUN_DIR}/step_3000.tar" ]]; then
  ./runmm_v1.sh "${RUN_ID}" \
    --vision_model siglip_base \
    --vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224 \
    --lm_checkpoint logs/lm_final/step_45000.tar \
    --seed "${SEED}" \
    --max_steps 3000 \
    --manual_max_steps \
    --batch_size 96 \
    --grad_accum_steps 2 \
    --eval_batch_size 96 \
    --log_every 1 \
    --eval_every 0 \
    --ckpt_every 500 \
    --final_eval_batches 1 \
    --lr 0.0002 \
    --lr_schedule cosine \
    --lr_warmup_steps 200 \
    --freeze_mode semantic_bottleneck_only \
    --bridge_question_context_mode question_only \
    --bridge_query_bank_mode question_hidden_attn \
    --semantic_bottleneck \
    --semantic_tokens 8 \
    --semantic_latent_dim 256 \
    --semantic_recon_loss_weight 0.1 \
    --semantic_consistency_loss_weight 0.0 \
    --disable_lm_visual_adapters \
    --prefix_remap_present \
    --no-apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${REMAP_CKPT}" \
    --semantic_format_loss_weight 0.3 \
    --semantic_format_loss_final_weight 0.0 \
    --semantic_format_anneal_start_step 2250 \
    --semantic_format_anneal_end_step 3000 \
    --init_from_mm_checkpoint "${CEMENT_CKPT}" \
    --min_train_steps_per_s 0
fi

for STEP in 500 1000 1500 2000 2500 3000; do
  CKPT="${RUN_DIR}/step_${STEP}.tar"
  if [[ ! -f "${CKPT}" ]]; then
    continue
  fi
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${CKPT}" \
    --batch_size 96 \
    --eval_batches 100 \
    --disable_lm_visual_adapters \
    --output_json "${RUN_DIR}/format_eval_step_${STEP}_no_remap.json"
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${CKPT}" \
    --batch_size 96 \
    --eval_batches 100 \
    --disable_lm_visual_adapters \
    --apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${REMAP_CKPT}" \
    --output_json "${RUN_DIR}/format_eval_step_${STEP}_with_remap.json"
done

BEST_STEP="$(RUN_DIR_ENV="${RUN_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
run_dir = Path(os.environ["RUN_DIR_ENV"])
best_step = None
best_acc = -1.0
for path in sorted(run_dir.glob("format_eval_step_*_no_remap.json")):
    step = int(path.stem.split("_")[3])
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    acc = float(data.get("overall_accuracy", 0.0))
    if acc > best_acc:
        best_acc = acc
        best_step = step
print(best_step or 3000)
PY
)"
BEST_CKPT="${RUN_DIR}/step_${BEST_STEP}.tar"

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
  --checkpoint "${BEST_CKPT}" \
  --batch_size 96 \
  --eval_batches 0 \
  --disable_lm_visual_adapters \
  --output_json "${RUN_DIR}/format_eval_best_step_${BEST_STEP}_no_remap_full.json"

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
  --checkpoint "${BEST_CKPT}" \
  --batch_size 96 \
  --eval_batches 0 \
  --disable_lm_visual_adapters \
  --apply_prefix_remap_in_forward \
  --prefix_remap_checkpoint "${REMAP_CKPT}" \
  --output_json "${RUN_DIR}/format_eval_best_step_${BEST_STEP}_with_remap_full.json"

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_adapter_ablation \
  --checkpoint "${BEST_CKPT}" \
  --batch_size 96 \
  --keep_counts 3,2,1 \
  --eval_batches 0 \
  --no-disable_lm_visual_adapters \
  --output_json "${RUN_DIR}/adapter_ablation_best.json"

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_semantic_probe \
  --checkpoint "${BEST_CKPT}" \
  --batch_size 96 \
  --probe_batch_size 256 \
  --limit_train 10000 \
  --limit_val 5000 \
  --answer_top_k 3000 \
  --epochs 10 \
  --lr 0.001 \
  --feature_pool flatten \
  --output_json "${RUN_DIR}/tiny_head_probe_best.json"

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.analyze_format_alignment_experiment \
  --run_dir "${RUN_DIR}"

echo "${RUN_ID}"
