#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmhammer_kvcorrect_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmhammer_kvcorrect_v1_latest"

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_BATCHES="${MAX_BATCHES:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-2}"
EVAL_SCORER="${EVAL_SCORER:-official}"
DOCKER_IMAGE="${DOCKER_IMAGE:-myrepo:gpu}"

RESULTS_PATH="${SWEEP_DIR}/results.tsv"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Hammer KV-Cache Correctness Sweep V1

Purpose:
- compare eval KV-cache serial vs batched continuation on real MM bridge checkpoints
- verify exact prediction agreement before using batched mode for future runs

Runtime:
- DEVICE=${DEVICE}
- SPLIT=${SPLIT}
- BATCH_SIZE=${BATCH_SIZE}
- MAX_BATCHES=${MAX_BATCHES}
- NUM_WORKERS=${NUM_WORKERS}
- PREFETCH_FACTOR=${PREFETCH_FACTOR}
- PIN_MEMORY=${PIN_MEMORY}
- PROGRESS_EVERY=${PROGRESS_EVERY}
- EVAL_SCORER=${EVAL_SCORER}
- DOCKER_IMAGE=${DOCKER_IMAGE}

Pass rule:
- zero prediction mismatches between serial and batched
- zero missing question ids on either side
EOF

echo -e "label\tcheckpoint\tsamples\tserial_acc\tbatched_acc\tacc_delta\tserial_samples_per_s\tbatched_samples_per_s\texact_match_ratio\tmismatch_count\tmissing_serial\tmissing_batched\tstatus" > "${RESULTS_PATH}"

run_probe() {
  local label="$1"
  local checkpoint="$2"
  local json_path="${SWEEP_DIR}/${label}.json"
  local stdout_log="${SWEEP_DIR}/${label}.stdout.log"
  local preds_dir="${SWEEP_DIR}/${label}_preds"
  local cmd=(
    docker run --rm --gpus all --ipc=host
    -e PYTORCH_ENABLE_MPS_FALLBACK=1
    -v "$(pwd)":/app -w /app "${DOCKER_IMAGE}"
    python tasks/mm_bridge/scripts/mm_kvcache_correctness_probe.py
    --checkpoint "${checkpoint}"
    --device "${DEVICE}"
    --split "${SPLIT}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --progress_every "${PROGRESS_EVERY}"
    --max_batches "${MAX_BATCHES}"
    --eval_scorer "${EVAL_SCORER}"
    --output_json "${json_path}"
    --predictions_dir "${preds_dir}"
  )
  if [[ "${PIN_MEMORY}" == "0" ]]; then
    cmd+=(--no-pin_memory)
  fi

  echo "[$(date)] START ${label} checkpoint=${checkpoint}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if ! "${cmd[@]}" > "${stdout_log}" 2>&1; then
    echo -e "${label}\t${checkpoint}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tfail" >> "${RESULTS_PATH}"
    echo "[$(date)] FAIL  ${label}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi

  python3 - "${label}" "${checkpoint}" "${json_path}" "${RESULTS_PATH}" <<'PY'
import json
import sys

label, checkpoint, json_path, results_path = sys.argv[1:]
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
serial = data["serial"]
batched = data["batched"]
cmp = data["comparison"]
status = "pass"
if (
    int(cmp.get("prediction_mismatch_count", 0)) > 0
    or int(cmp.get("missing_in_serial_count", 0)) > 0
    or int(cmp.get("missing_in_batched_count", 0)) > 0
):
    status = "mismatch"
row = [
    label,
    checkpoint,
    str(int(serial.get("samples", 0))),
    f"{float(serial.get('overall_accuracy', 0.0)):.6f}",
    f"{float(batched.get('overall_accuracy', 0.0)):.6f}",
    f"{float(data.get('accuracy_delta_batched_minus_serial', 0.0)):.6f}",
    f"{float(serial.get('samples_per_s', 0.0)):.3f}",
    f"{float(batched.get('samples_per_s', 0.0)):.3f}",
    f"{float(cmp.get('prediction_exact_match_ratio', 0.0)):.6f}",
    str(int(cmp.get("prediction_mismatch_count", 0))),
    str(int(cmp.get("missing_in_serial_count", 0))),
    str(int(cmp.get("missing_in_batched_count", 0))),
    status,
]
with open(results_path, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
print(status)
PY
  local status
  status="$(tail -n 1 "${RESULTS_PATH}" | awk -F'\t' '{print $NF}')"
  echo "[$(date)] END   ${label} status=${status}" | tee -a "${SWEEP_DIR}/timeline.log"
  [[ "${status}" == "pass" ]]
}

overall_status=0

run_probe \
  "structuredroles_frontier_step9000" \
  "logs/mmarch_high_entropy_v1_20260311_structuredroles_frontier/step_9000.tar" || overall_status=1

run_probe \
  "safeqcond_earlylayer_geomcal_frontier_step9000" \
  "logs/mmarch_high_entropy_v1_20260311_safeqcond_earlylayer_geomcal_frontier/step_9000.tar" || overall_status=1

run_probe \
  "hammer_qquery_step2000" \
  "logs/mmhammer_v1_qquery_earlylayer_geomcal/step_2000.tar" || overall_status=1

run_probe \
  "perf_dynbudget_qscore_step40" \
  "logs/mmhammer_perf_v1_20260312_dynbudget_qscore_earlylayer_geomcal_b64a3/step_40.tar" || overall_status=1

run_probe \
  "perf_qquery_dynbudget_step40" \
  "logs/mmhammer_perf_v1_20260312_qquery_dynbudget_earlylayer_geomcal_b64a3/step_40.tar" || overall_status=1

exit "${overall_status}"
