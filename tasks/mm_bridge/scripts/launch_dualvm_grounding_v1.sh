#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-dualvm_grounding_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"

DUAL_CKPT="${DUAL_CKPT:-logs/mmdualvm_v1_20260324_rerun/step_9000.tar}"
DUAL_REMAP_CKPT="${DUAL_REMAP_CKPT:-logs/dualvm_compressed_v1_20260325_234134_phase1_remap/step_500.tar}"
ANCHOR_CKPT="${ANCHOR_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
SIGLIP_DIR="${SIGLIP_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"
VITSTR_CKPT="${VITSTR_CKPT:-logs/hf_vision/vitstr_tiny_patch16_224/vitstr_tiny_patch16_224_aug.pth}"
POINTING_INDEX_PATH="${POINTING_INDEX_PATH:-data/pointing/train_index.jsonl}"
GQA_ROOT="${GQA_ROOT:-data/gqa}"
SEED="${SEED:-35}"

TRACE_EVAL_BATCHES="${TRACE_EVAL_BATCHES:-100}"
GQA_TRACE_LIMIT="${GQA_TRACE_LIMIT:-5000}"
GROUND_LIMIT="${GROUND_LIMIT:-5000}"
MIXED_BATCH_SIZE="${MIXED_BATCH_SIZE:-64}"
MIXED_GRAD_ACCUM="${MIXED_GRAD_ACCUM:-3}"
MIXED_EVAL_BATCH_SIZE="${MIXED_EVAL_BATCH_SIZE:-32}"
MIXED_NUM_WORKERS="${MIXED_NUM_WORKERS:-0}"
MIXED_PREFETCH_FACTOR="${MIXED_PREFETCH_FACTOR:-1}"
MIXED_EVAL_EVERY="${MIXED_EVAL_EVERY:-250}"
MIXED_CKPT_EVERY="${MIXED_CKPT_EVERY:-250}"
MIXED_EVAL_BATCHES="${MIXED_EVAL_BATCHES:-50}"
MIXED_FINAL_EVAL_BATCHES="${MIXED_FINAL_EVAL_BATCHES:-50}"

CONTROL_RUN_ID="${BUNDLE_ID}_control"
CONTROL_RUN_DIR="logs/${CONTROL_RUN_ID}"
MIXED_RUN_ID="${BUNDLE_ID}_groundgqa"
MIXED_RUN_DIR="logs/${MIXED_RUN_ID}"

mkdir -p "${BUNDLE_DIR}"

if [[ ! -f "${BUNDLE_DIR}/gqa_exact_sanity.json" ]]; then
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_gqa_eval \
    --checkpoint "${DUAL_CKPT}" \
    --batch_size 96 \
    --limit_eval 256 \
    --gqa_root "${GQA_ROOT}" \
    --output_json "${BUNDLE_DIR}/gqa_exact_sanity.json"
fi

run_control() {
  if [[ -f "${CONTROL_RUN_DIR}/step_3000.tar" ]]; then
    return
  fi
  ./runmm_v1.sh "${CONTROL_RUN_ID}" \
    --vision_model siglip_vitstr_tiny_dual \
    --vision_checkpoint "${SIGLIP_DIR}" \
    --vision_aux_checkpoint "${VITSTR_CKPT}" \
    --seed "${SEED}" \
    --max_steps 3000 \
    --manual_max_steps \
    --eval_every 500 \
    --eval_batches "${TRACE_EVAL_BATCHES}" \
    --ckpt_every 500 \
    --final_eval_batches "${TRACE_EVAL_BATCHES}" \
    --freeze_mode semantic_bottleneck_only \
    --semantic_bottleneck \
    --semantic_tokens 8 \
    --semantic_latent_dim 256 \
    --semantic_recon_loss_weight 0.1 \
    --semantic_consistency_loss_weight 0.0 \
    --disable_lm_visual_adapters \
    --prefix_remap_present \
    --no-apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${DUAL_REMAP_CKPT}" \
    --semantic_format_loss_weight 0.3 \
    --semantic_format_loss_final_weight 0.0 \
    --semantic_format_anneal_start_step 2250 \
    --semantic_format_anneal_end_step 3000 \
    --init_from_mm_checkpoint "${DUAL_CKPT}" \
    --min_train_steps_per_s 0
}

run_mixed() {
  if [[ -f "${MIXED_RUN_DIR}/step_3000.tar" ]]; then
    return
  fi
  ./runmm_v1.sh "${MIXED_RUN_ID}" \
    --vision_model siglip_vitstr_tiny_dual \
    --vision_checkpoint "${SIGLIP_DIR}" \
    --vision_aux_checkpoint "${VITSTR_CKPT}" \
    --seed "${SEED}" \
    --batch_size "${MIXED_BATCH_SIZE}" \
    --grad_accum_steps "${MIXED_GRAD_ACCUM}" \
    --eval_batch_size "${MIXED_EVAL_BATCH_SIZE}" \
    --num_workers "${MIXED_NUM_WORKERS}" \
    --prefetch_factor "${MIXED_PREFETCH_FACTOR}" \
    --no-pin_memory \
    --max_steps 3000 \
    --manual_max_steps \
    --eval_every "${MIXED_EVAL_EVERY}" \
    --eval_batches "${MIXED_EVAL_BATCHES}" \
    --ckpt_every "${MIXED_CKPT_EVERY}" \
    --final_eval_batches "${MIXED_FINAL_EVAL_BATCHES}" \
    --freeze_mode semantic_bottleneck_only \
    --semantic_bottleneck \
    --semantic_tokens 8 \
    --semantic_latent_dim 256 \
    --semantic_recon_loss_weight 0.1 \
    --semantic_consistency_loss_weight 0.0 \
    --disable_lm_visual_adapters \
    --prefix_remap_present \
    --no-apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${DUAL_REMAP_CKPT}" \
    --semantic_format_loss_weight 0.3 \
    --semantic_format_loss_final_weight 0.0 \
    --semantic_format_anneal_start_step 2250 \
    --semantic_format_anneal_end_step 3000 \
    --init_from_mm_checkpoint "${DUAL_CKPT}" \
    --use_grounding_loss \
    --grounding_loss_weight 0.05 \
    --pointing_index_path "${POINTING_INDEX_PATH}" \
    --pointing_mix_ratio 0.1666667 \
    --gqa_train_mix_ratio 0.15625 \
    --min_train_steps_per_s 0
}

eval_vqa_trace() {
  local prefix="$1"
  local run_dir="$2"
  for STEP in 500 1000 1500 2000 2500 3000; do
    local ckpt="${run_dir}/step_${STEP}.tar"
    local out="${BUNDLE_DIR}/${prefix}_step_${STEP}_vqa.json"
    if [[ ! -f "${ckpt}" || -f "${out}" ]]; then
      continue
    fi
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
      --checkpoint "${ckpt}" \
      --batch_size 96 \
      --eval_batches "${TRACE_EVAL_BATCHES}" \
      --disable_lm_visual_adapters \
      --output_json "${out}"
  done
}

eval_gqa_trace() {
  local prefix="$1"
  local run_dir="$2"
  for STEP in 500 1000 1500 2000 2500 3000; do
    local ckpt="${run_dir}/step_${STEP}.tar"
    local out="${BUNDLE_DIR}/${prefix}_step_${STEP}_gqa.json"
    if [[ ! -f "${ckpt}" || -f "${out}" ]]; then
      continue
    fi
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_gqa_eval \
      --checkpoint "${ckpt}" \
      --batch_size 96 \
      --gqa_root "${GQA_ROOT}" \
      --limit_eval "${GQA_TRACE_LIMIT}" \
      --output_json "${out}"
  done
}

pick_best_step() {
  local prefix="$1"
  runtime_exec_python - <<'PY' "${BUNDLE_DIR}" "${prefix}"
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1])
prefix = sys.argv[2]
best_step = 3000
best_acc = -1.0
for path in sorted(bundle.glob(f"{prefix}_step_*_vqa.json")):
    step = int(path.stem.split("_")[2])
    data = json.load(open(path, "r", encoding="utf-8"))
    acc = float(data.get("overall_accuracy", 0.0))
    if acc > best_acc:
        best_acc = acc
        best_step = step
print(best_step)
PY
}

eval_best_suite() {
  local prefix="$1"
  local run_dir="$2"
  local best_step="$3"
  local ckpt="${run_dir}/step_${best_step}.tar"

  if [[ ! -f "${BUNDLE_DIR}/${prefix}_best_step_${best_step}_full.json" ]]; then
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
      --checkpoint "${ckpt}" \
      --batch_size 96 \
      --eval_batches 0 \
      --disable_lm_visual_adapters \
      --output_json "${BUNDLE_DIR}/${prefix}_best_step_${best_step}_full.json"
  fi

  if [[ ! -f "${BUNDLE_DIR}/${prefix}_best_step_${best_step}_gqa.json" ]]; then
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_gqa_eval \
      --checkpoint "${ckpt}" \
      --batch_size 96 \
      --gqa_root "${GQA_ROOT}" \
      --limit_eval 0 \
      --output_json "${BUNDLE_DIR}/${prefix}_best_step_${best_step}_gqa.json"
  fi

  if [[ ! -f "${BUNDLE_DIR}/${prefix}_ocr_analysis.json" ]]; then
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_dualvm_ocr_analysis \
      --dual_checkpoint "${ckpt}" \
      --anchor_checkpoint "${ANCHOR_CKPT}" \
      --batch_size 96 \
      --limit_ocr 500 \
      --limit_control 100 \
      --output_json "${BUNDLE_DIR}/${prefix}_ocr_analysis.json"
  fi

  if [[ ! -f "${BUNDLE_DIR}/${prefix}_probe.json" ]]; then
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_probe \
      --checkpoint "${ckpt}" \
      --batch_size 96 \
      --probe_batch_size 256 \
      --limit_train 10000 \
      --limit_val 5000 \
      --answer_top_k 3000 \
      --epochs 10 \
      --lr 0.001 \
      --feature_pool flatten \
      --output_json "${BUNDLE_DIR}/${prefix}_probe.json"
  fi

  if [[ ! -f "${BUNDLE_DIR}/${prefix}_grounding_mass.json" ]]; then
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_grounding_mass_eval \
      --checkpoint "${ckpt}" \
      --batch_size 96 \
      --images_root images \
      --annotations_root data/vqav2 \
      --pointing_index_path "${POINTING_INDEX_PATH}" \
      --limit_eval "${GROUND_LIMIT}" \
      --output_json "${BUNDLE_DIR}/${prefix}_grounding_mass.json"
  fi
}

run_control
eval_vqa_trace "control" "${CONTROL_RUN_DIR}"
eval_gqa_trace "control" "${CONTROL_RUN_DIR}"
CONTROL_BEST_STEP="$(pick_best_step control)"
printf '{\n  "best_step": %s\n}\n' "${CONTROL_BEST_STEP}" > "${BUNDLE_DIR}/control_best_step.json"
eval_best_suite "control" "${CONTROL_RUN_DIR}" "${CONTROL_BEST_STEP}"

run_mixed
eval_vqa_trace "mixed" "${MIXED_RUN_DIR}"
eval_gqa_trace "mixed" "${MIXED_RUN_DIR}"
MIXED_BEST_STEP="$(pick_best_step mixed)"
printf '{\n  "best_step": %s\n}\n' "${MIXED_BEST_STEP}" > "${BUNDLE_DIR}/mixed_best_step.json"
eval_best_suite "mixed" "${MIXED_RUN_DIR}" "${MIXED_BEST_STEP}"

runtime_exec_python -m tasks.mm_bridge.scripts.analyze_dualvm_grounding \
  --bundle_dir "${BUNDLE_DIR}"

echo "${BUNDLE_ID}"
