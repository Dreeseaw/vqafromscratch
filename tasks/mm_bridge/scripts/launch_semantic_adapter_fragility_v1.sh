#!/usr/bin/env bash
set -euo pipefail

RUN_PREFIX="${RUN_PREFIX:-mmsemantic_fragility_v1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="${RUN_PREFIX}_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
LATEST_LINK="logs/${RUN_PREFIX}_latest"

ANCHOR_CKPT="${ANCHOR_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
K32_CKPT="${K32_CKPT:-logs/mmsemantic_v1_20260322_k32/step_4000.tar}"
K8_CKPT="${K8_CKPT:-logs/mmsemantic_v1_20260322_k8/step_4000.tar}"
K8_PROBE_JSON="${K8_PROBE_JSON:-logs/mmsemantic_diag_v1_20260322_234722/k8_semantic_probe.json}"

BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"
LIMIT_EVAL="${LIMIT_EVAL:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"

RUN_GQA="${RUN_GQA:-1}"
GQA_BATCH_SIZE="${GQA_BATCH_SIZE:-32}"
GQA_LIMIT_EVAL="${GQA_LIMIT_EVAL:-2000}"
GQA_GROUPS="${GQA_GROUPS:-spatial attribute count exist}"
VQA_SCORER="${VQA_SCORER:-official}"
GQA_SCORER="${GQA_SCORER:-proxy}"

mkdir -p "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "${LATEST_LINK}"

README="${SWEEP_DIR}/README.md"
TIMELINE="${SWEEP_DIR}/timeline.log"

cat > "${README}" <<EOF
# Semantic Adapter Fragility Experiment

Bundle: \`${SWEEP_ID}\`

Purpose:
- per-category adapter ablation for anchor, K=32, K=8
- optional GQA keep-3 vs keep-0 subset checks
- derived fragility-ratio and probe-gap analysis
EOF

touch "${TIMELINE}"

log_line() {
  echo "[$(date)] $*" | tee -a "${TIMELINE}"
}

run_ablation() {
  local label="$1"
  local ckpt="$2"
  shift 2
  log_line "START ${label}"
  log_line "CMD -m tasks.mm_bridge.scripts.mm_adapter_ablation --checkpoint ${ckpt} $*"
  ./.venv_local/bin/python -m tasks.mm_bridge.scripts.mm_adapter_ablation \
    --checkpoint "${ckpt}" \
    "$@" \
    > "${SWEEP_DIR}/${label}.stdout.log" 2>&1
  log_line "END   ${label}"
}

run_ablation "anchor_vqa" "${ANCHOR_CKPT}" \
  --device cuda \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  $( [[ "${PIN_MEMORY}" == "1" ]] && echo "--pin_memory" || echo "--no-pin_memory" ) \
  --eval_split val \
  --limit_eval "${LIMIT_EVAL}" \
  --eval_batches "${EVAL_BATCHES}" \
  --scorer "${VQA_SCORER}" \
  --keep_counts 3,2,1,0 \
  --output_json "${SWEEP_DIR}/anchor_vqa_ablation.json"

run_ablation "k32_vqa" "${K32_CKPT}" \
  --device cuda \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  $( [[ "${PIN_MEMORY}" == "1" ]] && echo "--pin_memory" || echo "--no-pin_memory" ) \
  --eval_split val \
  --limit_eval "${LIMIT_EVAL}" \
  --eval_batches "${EVAL_BATCHES}" \
  --scorer "${VQA_SCORER}" \
  --keep_counts 3,2,1,0 \
  --output_json "${SWEEP_DIR}/k32_vqa_ablation.json"

run_ablation "k8_vqa" "${K8_CKPT}" \
  --device cuda \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  $( [[ "${PIN_MEMORY}" == "1" ]] && echo "--pin_memory" || echo "--no-pin_memory" ) \
  --eval_split val \
  --limit_eval "${LIMIT_EVAL}" \
  --eval_batches "${EVAL_BATCHES}" \
  --scorer "${VQA_SCORER}" \
  --keep_counts 3,2,1,0 \
  --output_json "${SWEEP_DIR}/k8_vqa_ablation.json"

GQA_RESULT_SPECS=()
if [[ "${RUN_GQA}" == "1" ]]; then
  for group in ${GQA_GROUPS}; do
    run_ablation "anchor_gqa_${group}" "${ANCHOR_CKPT}" \
      --device cuda \
      --batch_size "${GQA_BATCH_SIZE}" \
      --num_workers 0 \
      --prefetch_factor 2 \
      --no-pin_memory \
      --eval_split gqa_val \
      --gqa_root data/gqa \
      --gqa_eval_group "${group}" \
      --limit_eval "${GQA_LIMIT_EVAL}" \
      --eval_batches 0 \
      --scorer "${GQA_SCORER}" \
      --keep_counts 3,0 \
      --output_json "${SWEEP_DIR}/anchor_gqa_${group}.json"
    GQA_RESULT_SPECS+=("anchor_${group}:${SWEEP_DIR}/anchor_gqa_${group}.json")

    run_ablation "k8_gqa_${group}" "${K8_CKPT}" \
      --device cuda \
      --batch_size "${GQA_BATCH_SIZE}" \
      --num_workers 0 \
      --prefetch_factor 2 \
      --no-pin_memory \
      --eval_split gqa_val \
      --gqa_root data/gqa \
      --gqa_eval_group "${group}" \
      --limit_eval "${GQA_LIMIT_EVAL}" \
      --eval_batches 0 \
      --scorer "${GQA_SCORER}" \
      --keep_counts 3,0 \
      --output_json "${SWEEP_DIR}/k8_gqa_${group}.json"
    GQA_RESULT_SPECS+=("k8_${group}:${SWEEP_DIR}/k8_gqa_${group}.json")
  done
fi

log_line "START analysis"
ANALYZE_CMD=(
  ./.venv_local/bin/python -m tasks.mm_bridge.scripts.analyze_semantic_adapter_fragility
  --anchor_ablation "${SWEEP_DIR}/anchor_vqa_ablation.json"
  --k32_ablation "${SWEEP_DIR}/k32_vqa_ablation.json"
  --k8_ablation "${SWEEP_DIR}/k8_vqa_ablation.json"
  --k8_probe "${K8_PROBE_JSON}"
  --output_json "${SWEEP_DIR}/fragility_analysis.json"
  --output_md "${SWEEP_DIR}/fragility_analysis.md"
)
for spec in "${GQA_RESULT_SPECS[@]}"; do
  ANALYZE_CMD+=(--gqa_results "${spec}")
done
printf '[%s] CMD ' "$(date)" | tee -a "${TIMELINE}"
printf '%q ' "${ANALYZE_CMD[@]}" | tee -a "${TIMELINE}"
printf '\n' | tee -a "${TIMELINE}"
"${ANALYZE_CMD[@]}" > "${SWEEP_DIR}/analysis.stdout.log" 2>&1
log_line "END   analysis"
log_line "EXPERIMENT COMPLETE ${SWEEP_ID}"
