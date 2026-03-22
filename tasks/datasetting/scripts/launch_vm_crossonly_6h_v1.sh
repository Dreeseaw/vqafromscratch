#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
DRY_RUN="${DRY_RUN:-0}"
FAMILY="${FAMILY:-fullpair1}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-logs/vm_dinovit_mixed2/step_9000.tar}"
VM_RUN_ID="${VM_RUN_ID:-vm_recipev1_crossonly6h_${FAMILY}}"

CROSS_BATCH_SIZE="${CROSS_BATCH_SIZE:-176}"
CROSS_NUM_WORKERS="${CROSS_NUM_WORKERS:-6}"
CROSS_PREFETCH_FACTOR="${CROSS_PREFETCH_FACTOR:-2}"
CROSS_WARMUP_STEPS="${CROSS_WARMUP_STEPS:-500}"
CROSS_LR="${CROSS_LR:-0.0002}"
CROSS_DINO_WEIGHT_SCHEDULE="${CROSS_DINO_WEIGHT_SCHEDULE:-0.5@0.0}"
PAIR_MAX_PAIRS="${PAIR_MAX_PAIRS:-0}"
LOG_EVERY="${LOG_EVERY:-50}"
CKPT_EVERY="${CKPT_EVERY:-1000}"

CROSS_TARGET_HOURS="${CROSS_TARGET_HOURS:-5.75}"
MEASURED_CROSS_STEPS_PER_S="${MEASURED_CROSS_STEPS_PER_S:-1.36}"
CROSS_PHASE_STEPS="${CROSS_PHASE_STEPS:-}"

PAIR_MIX="${PAIR_MIX:-}"
if [[ -z "${PAIR_MIX}" ]]; then
  PAIR_MIX='{"coco_captions_2014:train2014":100,"coco_text_captions:train":100,"flickr30k:train":100,"cc3m_subset_50k:train":100,"midjourney_v6_recap_llava:train":100,"midjourney_v6_recap_gemini:train":100,"midjourney_v6_recap_qwen3:train":100}'
fi

normalize_json_object() {
  local label="$1"
  local raw="$2"
  python3 - "${label}" "${raw}" <<'PY'
import json
import sys

label = sys.argv[1]
raw = sys.argv[2].strip()
try:
    value = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f"[crossonly] ERROR {label} must be valid JSON object text: {raw!r} ({exc})")
if not isinstance(value, dict) or not value:
    raise SystemExit(f"[crossonly] ERROR {label} must be a non-empty JSON object: {raw!r}")
print(json.dumps(value, separators=(',', ':'), sort_keys=True))
PY
}

PAIR_MIX="$(normalize_json_object "PAIR_MIX" "${PAIR_MIX}")"

if [[ -z "${CROSS_PHASE_STEPS}" ]]; then
  CROSS_PHASE_STEPS="$(python3 - "${CROSS_TARGET_HOURS}" "${MEASURED_CROSS_STEPS_PER_S}" <<'PY'
import sys

hours = float(sys.argv[1])
steps_per_s = float(sys.argv[2])
target = int(round(hours * 3600.0 * steps_per_s))
target = max(1000, target)
target = int(round(target / 8.0) * 8)
print(target)
PY
)"
fi

SWEEP_ID="ds_vmcrossonly6h_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/ds_vmcrossonly6h_v1_latest"

latest_step_checkpoint() {
  local run_id="$1"
  local run_dir="logs/${run_id}"
  local max_step=0
  local f base step
  shopt -s nullglob
  for f in "${run_dir}"/step_*.tar; do
    base="${f##*/}"
    step="${base#step_}"
    step="${step%.tar}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( 10#${step} > max_step )); then
      max_step=$((10#${step}))
    fi
  done
  shopt -u nullglob
  echo "${max_step}"
}

cat > "${SWEEP_DIR}/README.md" <<EOF
# VM Cross-Only 6h V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Run type: DINO + SigLIP cross-training only
VM run: ${VM_RUN_ID}
Base checkpoint: ${BASE_CHECKPOINT}

Target wall time (approx): ${CROSS_TARGET_HOURS} hours
Measured cross steps/s: ${MEASURED_CROSS_STEPS_PER_S}
Target phase steps: ${CROSS_PHASE_STEPS}

Batch size: ${CROSS_BATCH_SIZE}
Workers: ${CROSS_NUM_WORKERS}
Prefetch factor: ${CROSS_PREFETCH_FACTOR}
Warmup steps: ${CROSS_WARMUP_STEPS}
LR: ${CROSS_LR}
DINO weight schedule: ${CROSS_DINO_WEIGHT_SCHEDULE}
Pair max pairs: ${PAIR_MAX_PAIRS}
Pair mix: ${PAIR_MIX}
EOF

printf "family\tvm_run_id\tbase_checkpoint\ttarget_hours\tmeasured_steps_per_s\tphase_steps\tbatch_size\tnum_workers\tprefetch_factor\tpair_mix\tdino_weight_schedule\n" > "${SWEEP_DIR}/families.tsv"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "${FAMILY}" "${VM_RUN_ID}" "${BASE_CHECKPOINT}" "${CROSS_TARGET_HOURS}" "${MEASURED_CROSS_STEPS_PER_S}" "${CROSS_PHASE_STEPS}" \
  "${CROSS_BATCH_SIZE}" "${CROSS_NUM_WORKERS}" "${CROSS_PREFETCH_FACTOR}" "${PAIR_MIX}" "${CROSS_DINO_WEIGHT_SCHEDULE}" >> "${SWEEP_DIR}/families.tsv"

LATEST_STEP="$(latest_step_checkpoint "${VM_RUN_ID}")"
CMD=(./runvmcrosssiglip.sh "${VM_RUN_ID}")
CMD+=(
  --base_checkpoint "${BASE_CHECKPOINT}"
  --pair_mix "${PAIR_MIX}"
  --max_pairs "${PAIR_MAX_PAIRS}"
  --batch_size "${CROSS_BATCH_SIZE}"
  --num_workers "${CROSS_NUM_WORKERS}"
  --prefetch_factor "${CROSS_PREFETCH_FACTOR}"
  --phase_steps "${CROSS_PHASE_STEPS}"
  --warmup_steps "${CROSS_WARMUP_STEPS}"
  --lr "${CROSS_LR}"
  --log_every "${LOG_EVERY}"
  --ckpt_every "${CKPT_EVERY}"
  --dino_weight_schedule "${CROSS_DINO_WEIGHT_SCHEDULE}"
)

echo "[$(date)] FAMILY ${FAMILY} vm=${VM_RUN_ID} base=${BASE_CHECKPOINT} latest_step=${LATEST_STEP} target_phase_steps=${CROSS_PHASE_STEPS}" | tee -a "${SWEEP_DIR}/timeline.log"
echo "[$(date)] CMD ${CMD[*]}" >> "${SWEEP_DIR}/timeline.log"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[$(date)] DRY_RUN skip ${VM_RUN_ID} stage=cross" | tee -a "${SWEEP_DIR}/timeline.log"
  exit 0
fi

if "${CMD[@]}" >> "${SWEEP_DIR}/${VM_RUN_ID}.stdout.log" 2>&1; then
  FINAL_STEP="$(latest_step_checkpoint "${VM_RUN_ID}")"
  echo "[$(date)] END   ${VM_RUN_ID} stage=cross final_step=${FINAL_STEP}" | tee -a "${SWEEP_DIR}/timeline.log"
  exit 0
fi

echo "[$(date)] FAIL  ${VM_RUN_ID} stage=cross (see ${SWEEP_DIR}/${VM_RUN_ID}.stdout.log)" | tee -a "${SWEEP_DIR}/timeline.log"
exit 1
