#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-mmsemantic_diag_v1_${STAMP}}"
OUT_DIR="logs/${RUN_ID}"
mkdir -pv "${OUT_DIR}"

ANCHOR_CKPT="${ANCHOR_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
BEST_COMPRESSED_CKPT="${BEST_COMPRESSED_CKPT:-logs/mmsemantic_v1_20260322_k32/step_4000.tar}"
FRONTIER_COMPRESSED_CKPT="${FRONTIER_COMPRESSED_CKPT:-logs/mmsemantic_v1_20260322_k8/step_4000.tar}"
INCLUDE_FRONTIER="${INCLUDE_FRONTIER:-1}"
INCLUDE_GROUNDING="${INCLUDE_GROUNDING:-1}"
DRY_RUN="${DRY_RUN:-0}"

PROBE_LIMIT_TRAIN="${PROBE_LIMIT_TRAIN:-10000}"
PROBE_LIMIT_VAL="${PROBE_LIMIT_VAL:-5000}"
PROBE_ANSWER_TOP_K="${PROBE_ANSWER_TOP_K:-3000}"
PROBE_EPOCHS="${PROBE_EPOCHS:-10}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-32}"
PROBE_TRAIN_BATCH_SIZE="${PROBE_TRAIN_BATCH_SIZE:-256}"

ABLATION_BATCH_SIZE="${ABLATION_BATCH_SIZE:-96}"
ABLATION_EVAL_BATCHES="${ABLATION_EVAL_BATCHES:-0}"
ABLATION_KEEP_COUNTS="${ABLATION_KEEP_COUNTS:-3,2,1,0}"

GROUND_BATCH_SIZE="${GROUND_BATCH_SIZE:-32}"
GROUND_LIMIT_EVAL="${GROUND_LIMIT_EVAL:-5000}"
GROUND_MAX_BATCHES="${GROUND_MAX_BATCHES:-0}"
GROUND_NUM_CORRECT="${GROUND_NUM_CORRECT:-50}"
GROUND_NUM_INCORRECT="${GROUND_NUM_INCORRECT:-50}"

cat > "${OUT_DIR}/README.md" <<EOF
# Semantic Compression Diagnostics V1

Created: $(date)

Purpose:
- run the downstream evaluation stack defined in doc 48
- compare the Cement anchor against the best completed semantic-compression run
- include the K=8 frontier run by default because it stayed nearly tied with K=16 on full eval

Checkpoints:
- anchor: ${ANCHOR_CKPT}
- best compressed: ${BEST_COMPRESSED_CKPT}
- frontier compressed: ${FRONTIER_COMPRESSED_CKPT}

Prepared tasks:
- semantic linear probe
- LM visual-adapter ablation
- grounding inspection
EOF

run_step() {
  local label="$1"
  shift
  local cmd=("$@")
  echo "[$(date)] START ${label}" | tee -a "${OUT_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${OUT_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${label}" | tee -a "${OUT_DIR}/timeline.log"
    return 0
  fi
  runtime_exec_python "${cmd[@]}" >> "${OUT_DIR}/${label}.stdout.log" 2>&1
  echo "[$(date)] END   ${label}" | tee -a "${OUT_DIR}/timeline.log"
}

run_probe() {
  local name="$1"
  local ckpt="$2"
  run_step "probe_${name}" \
    -m tasks.mm_bridge.scripts.mm_semantic_probe \
    --checkpoint "${ckpt}" \
    --device cuda \
    --batch_size "${PROBE_BATCH_SIZE}" \
    --probe_batch_size "${PROBE_TRAIN_BATCH_SIZE}" \
    --limit_train "${PROBE_LIMIT_TRAIN}" \
    --limit_val "${PROBE_LIMIT_VAL}" \
    --answer_top_k "${PROBE_ANSWER_TOP_K}" \
    --epochs "${PROBE_EPOCHS}" \
    --output_json "${OUT_DIR}/${name}_semantic_probe.json"
}

run_ablation() {
  local name="$1"
  local ckpt="$2"
  run_step "ablation_${name}" \
    -m tasks.mm_bridge.scripts.mm_adapter_ablation \
    --checkpoint "${ckpt}" \
    --device cuda \
    --batch_size "${ABLATION_BATCH_SIZE}" \
    --eval_batches "${ABLATION_EVAL_BATCHES}" \
    --keep_counts "${ABLATION_KEEP_COUNTS}" \
    --output_json "${OUT_DIR}/${name}_adapter_ablation.json"
}

run_grounding() {
  local name="$1"
  local ckpt="$2"
  run_step "ground_${name}" \
    -m tasks.mm_bridge.scripts.mm_grounding_inspection \
    --checkpoint "${ckpt}" \
    --device cuda \
    --batch_size "${GROUND_BATCH_SIZE}" \
    --limit_eval "${GROUND_LIMIT_EVAL}" \
    --max_batches "${GROUND_MAX_BATCHES}" \
    --num_correct "${GROUND_NUM_CORRECT}" \
    --num_incorrect "${GROUND_NUM_INCORRECT}" \
    --output_dir "${OUT_DIR}/${name}_grounding"
}

run_probe "anchor" "${ANCHOR_CKPT}"
run_probe "best" "${BEST_COMPRESSED_CKPT}"
run_ablation "anchor" "${ANCHOR_CKPT}"
run_ablation "best" "${BEST_COMPRESSED_CKPT}"

if [[ "${INCLUDE_FRONTIER}" == "1" ]]; then
  run_probe "k8" "${FRONTIER_COMPRESSED_CKPT}"
  run_ablation "k8" "${FRONTIER_COMPRESSED_CKPT}"
fi

if [[ "${INCLUDE_GROUNDING}" == "1" ]]; then
  run_grounding "anchor" "${ANCHOR_CKPT}"
  run_grounding "best" "${BEST_COMPRESSED_CKPT}"
  if [[ "${INCLUDE_FRONTIER}" == "1" ]]; then
    run_grounding "k8" "${FRONTIER_COMPRESSED_CKPT}"
  fi
fi

echo "[$(date)] DIAGNOSTICS COMPLETE ${RUN_ID}" | tee -a "${OUT_DIR}/timeline.log"
