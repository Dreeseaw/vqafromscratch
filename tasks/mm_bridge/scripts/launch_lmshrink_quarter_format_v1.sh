#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-./.venv_local/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-lmshrink_quarter_format_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"

QUARTER_PRETRAIN_CKPT="${QUARTER_PRETRAIN_CKPT:-logs/mmsemantic_lmshrink_v1_20260324_221950_pretrain_quarter/step_45000.tar}"
QUARTER_BRIDGE_CKPT="${QUARTER_BRIDGE_CKPT:-logs/mmsemantic_lmshrink_v1_20260324_221950_bridge_quarter/step_9000.tar}"
QUARTER_COMP_CKPT="${QUARTER_COMP_CKPT:-logs/mmsemantic_lmshrink_v1_20260324_221950_compression_quarter/step_3000.tar}"
QUARTER_COMP_FULL="${QUARTER_COMP_FULL:-logs/mmsemantic_lmshrink_v1_20260324_221950_compression_quarter/compression_full_eval.json}"
PROCEED_GAIN="${PROCEED_GAIN:-0.005}"
SEED="${SEED:-35}"

PHASE1_RUN_ID="${BUNDLE_ID}_phase1_remap"
PHASE1_RUN_DIR="logs/${PHASE1_RUN_ID}"
PHASE2_RUN_ID="${BUNDLE_ID}_phase2_format"
PHASE2_RUN_DIR="logs/${PHASE2_RUN_ID}"

mkdir -p "${BUNDLE_DIR}"

if [[ ! -f "${PHASE1_RUN_DIR}/step_500.tar" ]]; then
  ./runmm_v1.sh "${PHASE1_RUN_ID}" \
    --vision_model siglip_base \
    --vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224 \
    --lm_checkpoint "${QUARTER_PRETRAIN_CKPT}" \
    --lm_d_model 384 \
    --lm_num_heads 6 \
    --lm_layers 4 \
    --lm_mlp_ratio 2 \
    --lm_dropout 0.1 \
    --lm_max_seq_len 256 \
    --seed "${SEED}" \
    --max_steps 500 \
    --manual_max_steps \
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
    --apply_prefix_remap_in_forward \
    --disable_lm_visual_adapters \
    --init_from_mm_checkpoint "${QUARTER_COMP_CKPT}" \
    --min_train_steps_per_s 0
fi

for STEP in 100 200 300 400 500; do
  CKPT="${PHASE1_RUN_DIR}/step_${STEP}.tar"
  if [[ ! -f "${CKPT}" ]]; then
    continue
  fi
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${CKPT}" \
    --batch_size 96 \
    --eval_batches 100 \
    --disable_lm_visual_adapters \
    --apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${CKPT}" \
    --output_json "${BUNDLE_DIR}/phase1_eval_step_${STEP}_with_remap.json"
done

"${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}" "${QUARTER_COMP_FULL}" "${PROCEED_GAIN}"
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
baseline = json.load(open(sys.argv[2], "r", encoding="utf-8"))
threshold = float(sys.argv[3])

rows = []
for path in sorted(bundle.glob("phase1_eval_step_*_with_remap.json")):
    step = int(path.stem.split("_")[3])
    data = json.load(open(path, "r", encoding="utf-8"))
    rows.append((step, float(data.get("overall_accuracy", 0.0)), data))

if not rows:
    raise SystemExit("No Phase 1 eval rows found.")

best_step, best_overall, best_data = max(rows, key=lambda x: x[1])
baseline_overall = float(baseline.get("overall_accuracy", 0.0))
gain = best_overall - baseline_overall
out = {
    "quarter_compressed_baseline": baseline,
    "phase1_best_step": best_step,
    "phase1_best_overall": best_overall,
    "phase1_best_eval": best_data,
    "phase1_gain_over_baseline": gain,
    "proceed_phase2": bool(gain >= threshold),
    "threshold": threshold,
}
with open(bundle / "phase1_decision.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=True)
print(json.dumps(out, indent=2))
PY

PROCEED_PHASE2="$("${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}/phase1_decision.json"
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print("1" if data.get("proceed_phase2") else "0")
PY
)"

if [[ "${PROCEED_PHASE2}" != "1" ]]; then
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.analyze_lmshrink_quarter_format \
    --bundle_dir "${BUNDLE_DIR}"
  echo "${BUNDLE_ID}"
  exit 0
fi

if [[ ! -f "${PHASE2_RUN_DIR}/step_5000.tar" ]]; then
  ./runmm_v1.sh "${PHASE2_RUN_ID}" \
    --vision_model siglip_base \
    --vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224 \
    --lm_checkpoint "${QUARTER_PRETRAIN_CKPT}" \
    --lm_d_model 384 \
    --lm_num_heads 6 \
    --lm_layers 4 \
    --lm_mlp_ratio 2 \
    --lm_dropout 0.1 \
    --lm_max_seq_len 256 \
    --seed "${SEED}" \
    --max_steps 5000 \
    --manual_max_steps \
    --eval_every 500 \
    --eval_batches 100 \
    --ckpt_every 500 \
    --final_eval_batches 0 \
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
    --prefix_remap_checkpoint "${PHASE1_RUN_DIR}/step_500.tar" \
    --semantic_format_loss_weight 0.3 \
    --semantic_format_loss_final_weight 0.0 \
    --semantic_format_anneal_start_step 3750 \
    --semantic_format_anneal_end_step 5000 \
    --init_from_mm_checkpoint "${QUARTER_BRIDGE_CKPT}" \
    --min_train_steps_per_s 0
fi

for STEP in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000; do
  CKPT="${PHASE2_RUN_DIR}/step_${STEP}.tar"
  if [[ ! -f "${CKPT}" ]]; then
    continue
  fi
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${CKPT}" \
    --batch_size 96 \
    --eval_batches 100 \
    --disable_lm_visual_adapters \
    --output_json "${BUNDLE_DIR}/phase2_eval_step_${STEP}_no_remap.json"
done

BEST_STEP="$("${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}"
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
best_step = 5000
best_acc = -1.0
for path in sorted(bundle.glob("phase2_eval_step_*_no_remap.json")):
    step = int(path.stem.split("_")[3])
    data = json.load(open(path, "r", encoding="utf-8"))
    acc = float(data.get("overall_accuracy", 0.0))
    if acc > best_acc:
        best_acc = acc
        best_step = step
print(best_step)
PY
)"

BEST_CKPT="${PHASE2_RUN_DIR}/step_${BEST_STEP}.tar"

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
  --checkpoint "${BEST_CKPT}" \
  --batch_size 96 \
  --eval_batches 0 \
  --disable_lm_visual_adapters \
  --output_json "${BUNDLE_DIR}/phase2_best_step_${BEST_STEP}_full.json"

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
  --output_json "${BUNDLE_DIR}/phase2_tiny_head_probe_best.json"

"${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}/phase1_decision.json" "${BEST_STEP}"
import json
import sys
path = sys.argv[1]
best_step = int(sys.argv[2])
data = json.load(open(path, "r", encoding="utf-8"))
data["phase2_best_step"] = best_step
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=True)
PY

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.analyze_lmshrink_quarter_format \
  --bundle_dir "${BUNDLE_DIR}"

echo "${BUNDLE_ID}"
