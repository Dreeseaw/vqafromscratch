#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run_id> [expected_step]" >&2
  exit 2
fi

RUN_ID="$1"
EXPECTED_STEP="${2:-}"
INTERVAL_SECS="${INTERVAL_SECS:-240}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOGFILE="${RUN_DIR}/logfile.txt"
BABYSIT_LOG="${RUN_DIR}/babysit.log"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "run dir not found: ${RUN_DIR}" >&2
  exit 2
fi

touch "${BABYSIT_LOG}"

log_line() {
  local line="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${line}" | tee -a "${BABYSIT_LOG}"
}

latest_step() {
  if [[ -f "${LOGFILE}" ]]; then
    grep -Eo 'step=[0-9]+' "${LOGFILE}" | tail -n 1 | sed 's/step=//' || true
  fi
}

latest_checkpoint() {
  local ckpt
  ckpt="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'step_*.tar' | sort -V | tail -n 1 || true)"
  if [[ -n "${ckpt}" ]]; then
    basename "${ckpt}"
  fi
}

latest_steps_per_s() {
  if [[ -f "${LOGFILE}" ]]; then
    grep -Eo 'steps_per_s=[0-9.]+$' "${LOGFILE}" | tail -n 1 | sed 's/steps_per_s=//' || true
  fi
}

format_eta() {
  local current_step="$1"
  local sps="$2"
  if [[ -z "${EXPECTED_STEP}" || -z "${current_step}" || -z "${sps}" ]]; then
    return 0
  fi
  python3 - <<'PY' "${EXPECTED_STEP}" "${current_step}" "${sps}"
import sys
try:
    target = int(sys.argv[1]); current = int(sys.argv[2]); sps = float(sys.argv[3])
    if sps <= 0 or current >= target:
        print("0m")
    else:
        secs = int(round((target - current) / sps))
        h = secs // 3600
        m = (secs % 3600) // 60
        if h:
            print(f"{h}h {m}m")
        else:
            print(f"{m}m")
except Exception:
    pass
PY
}

run_pids() {
  pgrep -f "train\\.mm ${RUN_ID}|runmm\\.sh ${RUN_ID}|runmm_v1\\.sh ${RUN_ID}" | paste -sd, - || true
}

gpu_snapshot() {
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '\r' || true
}

log_line "babysit start run_id=${RUN_ID} expected_step=${EXPECTED_STEP:-none} interval=${INTERVAL_SECS}s"

while true; do
  PIDS="$(run_pids)"
  STEP="$(latest_step)"
  CKPT="$(latest_checkpoint)"
  SPS="$(latest_steps_per_s)"
  ETA="$(format_eta "${STEP}" "${SPS}")"
  GPU="$(gpu_snapshot)"
  LAST_LINE=""
  if [[ -f "${LOGFILE}" ]]; then
    LAST_LINE="$(tail -n 1 "${LOGFILE}" | tr '\t' ' ' | tr -s ' ')"
  fi

  log_line "check stuff! pids=${PIDS:-none} latest_step=${STEP:-none} latest_ckpt=${CKPT:-none} steps_per_s=${SPS:-none} eta=${ETA:-none} gpu=${GPU:-none} last='${LAST_LINE}'"

  if [[ -n "${EXPECTED_STEP}" && -f "${RUN_DIR}/step_${EXPECTED_STEP}.tar" ]]; then
    log_line "check stuff! expected checkpoint reached: step_${EXPECTED_STEP}.tar"
    exit 0
  fi

  if [[ -z "${PIDS}" ]]; then
    log_line "check stuff! process missing before expected completion"
    exit 1
  fi

  sleep "${INTERVAL_SECS}"
done
