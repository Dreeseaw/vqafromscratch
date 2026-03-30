#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-./.venv_local/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-dualvm_compressed_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"

DUAL_CKPT="${DUAL_CKPT:-logs/mmdualvm_v1_20260324_rerun/step_9000.tar}"
ANCHOR_CKPT="${ANCHOR_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
SIGLIP_DIR="${SIGLIP_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"
VITSTR_CKPT="${VITSTR_CKPT:-logs/hf_vision/vitstr_tiny_patch16_224/vitstr_tiny_patch16_224_aug.pth}"
PROCEED_GAIN="${PROCEED_GAIN:-0.01}"
SEED="${SEED:-35}"

PHASE1_RUN_ID="${BUNDLE_ID}_phase1_remap"
PHASE1_RUN_DIR="logs/${PHASE1_RUN_ID}"
PHASE2_RUN_ID="${BUNDLE_ID}_phase2_k8"
PHASE2_RUN_DIR="logs/${PHASE2_RUN_ID}"

mkdir -p "${BUNDLE_DIR}"

if [[ ! -f "${BUNDLE_DIR}/phase1_dual_keep0_full.json" ]]; then
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${DUAL_CKPT}" \
    --batch_size 96 \
    --eval_batches 0 \
    --disable_lm_visual_adapters \
    --output_json "${BUNDLE_DIR}/phase1_dual_keep0_full.json"
fi

if [[ ! -f "${PHASE1_RUN_DIR}/step_500.tar" ]]; then
  ./runmm_v1.sh "${PHASE1_RUN_ID}" \
    --vision_model siglip_vitstr_tiny_dual \
    --vision_checkpoint "${SIGLIP_DIR}" \
    --vision_aux_checkpoint "${VITSTR_CKPT}" \
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
    --use_prefix_remap \
    --apply_prefix_remap_in_forward \
    --disable_lm_visual_adapters \
    --init_from_mm_checkpoint "${DUAL_CKPT}" \
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

PHASE1_BEST_STEP="$("${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}"
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1])
best_step = 500
best_acc = -1.0
for path in sorted(bundle.glob("phase1_eval_step_*_with_remap.json")):
    step = int(path.stem.split("_")[3])
    data = json.load(open(path, "r", encoding="utf-8"))
    acc = float(data.get("overall_accuracy", 0.0))
    if acc > best_acc:
        best_acc = acc
        best_step = step
print(best_step)
PY
)"

PHASE1_BEST_CKPT="${PHASE1_RUN_DIR}/step_${PHASE1_BEST_STEP}.tar"
PHASE1_BEST_FULL="${BUNDLE_DIR}/phase1_best_step_${PHASE1_BEST_STEP}_with_remap_full.json"
if [[ ! -f "${PHASE1_BEST_FULL}" ]]; then
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${PHASE1_BEST_CKPT}" \
    --batch_size 96 \
    --eval_batches 0 \
    --disable_lm_visual_adapters \
    --apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${PHASE1_BEST_CKPT}" \
    --output_json "${PHASE1_BEST_FULL}"
fi

"${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}/phase1_dual_keep0_full.json" "${PHASE1_BEST_FULL}" "${BUNDLE_DIR}/phase1_decision.json" "${PROCEED_GAIN}"
import json, sys
keep0 = json.load(open(sys.argv[1], "r", encoding="utf-8"))
remap = json.load(open(sys.argv[2], "r", encoding="utf-8"))
threshold = float(sys.argv[4])
gain = float(remap.get("overall_accuracy", 0.0)) - float(keep0.get("overall_accuracy", 0.0))
out = {
    "phase1_keep0_full_path": sys.argv[1],
    "phase1_best_full_eval_path": sys.argv[2],
    "phase1_gain_over_keep0": gain,
    "threshold": threshold,
    "proceed_phase2": bool(gain >= threshold),
}
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=True)
print(json.dumps(out, indent=2))
PY

PROCEED_PHASE2="$("${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}/phase1_decision.json"
import json, sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print("1" if data.get("proceed_phase2") else "0")
PY
)"

if [[ "${PROCEED_PHASE2}" == "1" ]]; then
  if [[ ! -f "${PHASE2_RUN_DIR}/step_3000.tar" ]]; then
    ./runmm_v1.sh "${PHASE2_RUN_ID}" \
      --vision_model siglip_vitstr_tiny_dual \
      --vision_checkpoint "${SIGLIP_DIR}" \
      --vision_aux_checkpoint "${VITSTR_CKPT}" \
      --seed "${SEED}" \
      --max_steps 3000 \
      --manual_max_steps \
      --eval_every 500 \
      --eval_batches 100 \
      --ckpt_every 500 \
      --final_eval_batches 0 \
      --lr 0.0002 \
      --lr_schedule cosine \
      --lr_warmup_steps 200 \
      --freeze_mode semantic_bottleneck_only \
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
      --semantic_format_anneal_start_step 2250 \
      --semantic_format_anneal_end_step 3000 \
      --init_from_mm_checkpoint "${DUAL_CKPT}" \
      --min_train_steps_per_s 0
  fi

  for STEP in 500 1000 1500 2000 2500 3000; do
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

  PHASE2_BEST_STEP="$("${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}"
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1])
best_step = 3000
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
  PHASE2_BEST_CKPT="${PHASE2_RUN_DIR}/step_${PHASE2_BEST_STEP}.tar"

  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${PHASE2_BEST_CKPT}" \
    --batch_size 96 \
    --eval_batches 0 \
    --disable_lm_visual_adapters \
    --output_json "${BUNDLE_DIR}/phase2_best_step_${PHASE2_BEST_STEP}_full.json"

  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_dualvm_ocr_analysis \
    --dual_checkpoint "${PHASE2_BEST_CKPT}" \
    --anchor_checkpoint "${ANCHOR_CKPT}" \
    --batch_size 96 \
    --limit_ocr 500 \
    --limit_control 100 \
    --output_json "${BUNDLE_DIR}/phase2_ocr_analysis.json"

  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_semantic_probe \
    --checkpoint "${PHASE2_BEST_CKPT}" \
    --batch_size 96 \
    --probe_batch_size 256 \
    --limit_train 10000 \
    --limit_val 5000 \
    --answer_top_k 3000 \
    --epochs 10 \
    --lr 0.001 \
    --feature_pool flatten \
    --output_json "${BUNDLE_DIR}/phase2_tiny_head_probe_best.json"

  "${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}/phase1_decision.json" "${PHASE2_BEST_STEP}"
import json, sys
path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
data["phase2_best_step"] = int(sys.argv[2])
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=True)
PY
fi

"${PYTHON_BIN}" -m tasks.mm_bridge.scripts.analyze_dualvm_compressed \
  --bundle_dir "${BUNDLE_DIR}"

echo "${BUNDLE_ID}"
