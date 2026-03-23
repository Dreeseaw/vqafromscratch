#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <diag_run_id> --checkpoint <path> [extra diagnostics args...]"
  echo
  echo "Example:"
  echo "  $0 mmdiag_lt1 --checkpoint logs/mmbr_basesweep_lt1/step_17330.tar --max_batches 200"
  exit 1
fi
shift || true

EXTRA_ARGS=("$@")

mkdir -pv "logs/${RUN_ID}"
cat "${SCRIPT_DIR}/mm_bridge_diagnostics.py" train/mm.py train/vqa_data.py evals/vqa.py > "logs/${RUN_ID}/code_mm_diag.py"

CMD=(
  -m tasks.mm_bridge.scripts.mm_bridge_diagnostics
  --output_json "logs/${RUN_ID}/diag_report.json"
  --output_md "logs/${RUN_ID}/diag_report.md"
)

runtime_exec_python \
  "${CMD[@]}" \
  "${EXTRA_ARGS[@]}"
