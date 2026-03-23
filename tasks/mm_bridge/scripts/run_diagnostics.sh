#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <diag_run_id> --checkpoint <path> [--skip-training-probes]"
  exit 1
fi
shift || true

CHECKPOINT=""
LM_CHECKPOINT_OVERRIDE=""
VISION_CHECKPOINT_OVERRIDE=""
TOKENIZER_PATH_OVERRIDE=""
IMAGES_ROOT="images"
ANNOTATIONS_ROOT="data/vqav2"
BATCH_SIZE="96"
NUM_WORKERS="2"
PREFETCH_FACTOR="1"
PIN_MEMORY="0"
LIMIT_EVAL="0"
MAX_BATCHES="0"
GROUND_NUM_CORRECT="100"
GROUND_NUM_INCORRECT="100"
SKIP_TRAINING_PROBES="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --images_root)
      IMAGES_ROOT="$2"
      shift 2
      ;;
    --lm_checkpoint_override)
      LM_CHECKPOINT_OVERRIDE="$2"
      shift 2
      ;;
    --vision_checkpoint_override)
      VISION_CHECKPOINT_OVERRIDE="$2"
      shift 2
      ;;
    --tokenizer_path_override)
      TOKENIZER_PATH_OVERRIDE="$2"
      shift 2
      ;;
    --annotations_root)
      ANNOTATIONS_ROOT="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --prefetch_factor)
      PREFETCH_FACTOR="$2"
      shift 2
      ;;
    --limit_eval)
      LIMIT_EVAL="$2"
      shift 2
      ;;
    --max_batches)
      MAX_BATCHES="$2"
      shift 2
      ;;
    --ground_num_correct)
      GROUND_NUM_CORRECT="$2"
      shift 2
      ;;
    --ground_num_incorrect)
      GROUND_NUM_INCORRECT="$2"
      shift 2
      ;;
    --pin_memory)
      PIN_MEMORY="1"
      shift
      ;;
    --no-pin_memory)
      PIN_MEMORY="0"
      shift
      ;;
    --skip-training-probes)
      SKIP_TRAINING_PROBES="1"
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "--checkpoint is required"
  exit 1
fi

DIAG_DIR="logs/${RUN_ID}/diagnostics"
mkdir -pv "${DIAG_DIR}"
cat \
  tasks/mm_bridge/scripts/mm_bridge_diagnostics.py \
  tasks/mm_bridge/scripts/mm_finegrained_breakdown.py \
  tasks/mm_bridge/scripts/mm_calibration_analysis.py \
  tasks/mm_bridge/scripts/mm_grounding_inspection.py \
  train/mm.py \
  train/vqa_data.py \
  evals/vqa.py \
  models/bridge.py \
  > "${DIAG_DIR}/code_diagnostics.py"

PIN_FLAG="--no-pin_memory"
if [[ "${PIN_MEMORY}" == "1" ]]; then
  PIN_FLAG="--pin_memory"
fi

COMMON_RUNTIME=(
  runtime_exec_python
)

OVERRIDE_ARGS=()
if [[ -n "${LM_CHECKPOINT_OVERRIDE}" ]]; then
  OVERRIDE_ARGS+=(--lm_checkpoint_override "${LM_CHECKPOINT_OVERRIDE}")
fi
if [[ -n "${VISION_CHECKPOINT_OVERRIDE}" ]]; then
  OVERRIDE_ARGS+=(--vision_checkpoint_override "${VISION_CHECKPOINT_OVERRIDE}")
fi
if [[ -n "${TOKENIZER_PATH_OVERRIDE}" ]]; then
  OVERRIDE_ARGS+=(--tokenizer_path_override "${TOKENIZER_PATH_OVERRIDE}")
fi

ANSWERS_PATH="$(dirname "${CHECKPOINT}")/fixed_eval_val_answers.jsonl"

"${COMMON_RUNTIME[@]}" \
  -m tasks.mm_bridge.scripts.mm_bridge_diagnostics \
  --checkpoint "${CHECKPOINT}" \
  --images_root "${IMAGES_ROOT}" \
  --annotations_root "${ANNOTATIONS_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  "${PIN_FLAG}" \
  "${OVERRIDE_ARGS[@]}" \
  --limit_eval "${LIMIT_EVAL}" \
  --max_batches "${MAX_BATCHES}" \
  --modes "clean,zero,noise" \
  --output_json "${DIAG_DIR}/visual_sufficiency.json" \
  --output_md "${DIAG_DIR}/visual_sufficiency.md"

if [[ -f "${ANSWERS_PATH}" ]]; then
  "${COMMON_RUNTIME[@]}" \
    -m tasks.mm_bridge.scripts.mm_finegrained_breakdown \
    --answers_jsonl "${ANSWERS_PATH}" \
    --annotations_root "${ANNOTATIONS_ROOT}" \
    --split val \
    --output_json "${DIAG_DIR}/finegrained_breakdown.json" \
    --output_md "${DIAG_DIR}/finegrained_breakdown.md"
else
  echo "[run_diagnostics] skipping fine-grained breakdown; missing ${ANSWERS_PATH}"
fi

"${COMMON_RUNTIME[@]}" \
  -m tasks.mm_bridge.scripts.mm_calibration_analysis \
  --checkpoint "${CHECKPOINT}" \
  --images_root "${IMAGES_ROOT}" \
  --annotations_root "${ANNOTATIONS_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  "${PIN_FLAG}" \
  "${OVERRIDE_ARGS[@]}" \
  --limit_eval "${LIMIT_EVAL}" \
  --max_batches "${MAX_BATCHES}" \
  --output_json "${DIAG_DIR}/calibration.json" \
  --output_md "${DIAG_DIR}/calibration.md" \
  --predictions_jsonl "${DIAG_DIR}/calibration_predictions.jsonl"

"${COMMON_RUNTIME[@]}" \
  -m tasks.mm_bridge.scripts.mm_grounding_inspection \
  --checkpoint "${CHECKPOINT}" \
  --images_root "${IMAGES_ROOT}" \
  --annotations_root "${ANNOTATIONS_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  "${PIN_FLAG}" \
  "${OVERRIDE_ARGS[@]}" \
  --limit_eval "${LIMIT_EVAL}" \
  --max_batches "${MAX_BATCHES}" \
  --num_correct "${GROUND_NUM_CORRECT}" \
  --num_incorrect "${GROUND_NUM_INCORRECT}" \
  --output_dir "${DIAG_DIR}/grounding"

"${COMMON_RUNTIME[@]}" \
  -m tasks.mm_bridge.scripts.launch_cement_query_count_probes \
    --checkpoint "${CHECKPOINT}" \
    --run_prefix "${RUN_ID}_queryprobe"
