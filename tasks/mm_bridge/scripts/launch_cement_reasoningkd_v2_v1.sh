#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"
source "${REPO_ROOT}/tasks/mm_bridge/scripts/experiment_bundle_lib.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmcement_reasoningkd_v2_v1_${STAMP}}"
LATEST_LINK="logs/mmcement_reasoningkd_v2_v1_latest"
RUN_ID="${RUN_ID:-${BUNDLE_ID}_main}"
PROGRESS_MD="logs/${BUNDLE_ID}/progress.md"
REPORT_MD="logs/${BUNDLE_ID}/report.md"

LM_RUN_ID="${LM_RUN_ID:-lm_reasoningmix_v2_20260329_133946}"
LM_STEP="${LM_STEP:-45000}"
LM_CKPT="${LM_CKPT:-logs/${LM_RUN_ID}/step_${LM_STEP}.tar}"
VISION_CKPT="${VISION_CKPT:-logs/hf_vision/google_siglip_base_patch16_224}"
TEACHER_DATA_DIR="${TEACHER_DATA_DIR:-data/distillation/qwen25vl3b_vqav2_train_v1}"

SEED="${SEED:-35}"
TRAIN_BS="${TRAIN_BS:-96}"
TRAIN_GA="${TRAIN_GA:-2}"
EVAL_BS="${EVAL_BS:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH="${PREFETCH:-1}"
KD_WEIGHT="${KD_WEIGHT:-0.3}"
KD_TEMP="${KD_TEMP:-4.0}"

format_duration() {
  runtime_exec_python - <<'PY' "$1" "$2"
import sys
start = float(sys.argv[1]); end = float(sys.argv[2])
delta = max(0, int(round(end - start)))
h = delta // 3600
m = (delta % 3600) // 60
print(f"{h}h {m}m")
PY
}

latest_step() {
  local run_dir="$1"
  local best=0
  local path base step
  shopt -s nullglob
  for path in "${run_dir}"/step_*.tar; do
    base="$(basename "${path}")"
    step="${base#step_}"
    step="${step%.tar}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > best )); then
      best="${step}"
    fi
  done
  shopt -u nullglob
  echo "${best}"
}

resume_runmm() {
  local run_id="$1"
  local target_step="$2"
  shift 2
  local run_dir="logs/${run_id}"
  local step
  step="$(latest_step "${run_dir}")"
  if (( step >= target_step )); then
    return 0
  fi
  if (( step > 0 )); then
    ./runmm_v1.sh "${run_id}" "${step}" "$@"
  else
    ./runmm_v1.sh "${run_id}" "$@"
  fi
}

eval_ckpt_vqa() {
  local ckpt="$1"
  local out_json="$2"
  local batch_size="$3"
  shift 3
  if [[ -f "${out_json}" ]]; then
    return 0
  fi
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${ckpt}" \
    --batch_size "${batch_size}" \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    "$@" \
    --output_json "${out_json}"
}

append_progress_entry() {
  local status="$1"
  local started="$2"
  local duration="$3"
  local best_step="$4"
  local json_path="$5"
  local notes="$6"
  runtime_exec_python - <<'PY' "${PROGRESS_MD}" "${status}" "${started}" "${duration}" "${best_step}" "${json_path}" "${notes}"
import json, sys
from pathlib import Path
md = Path(sys.argv[1])
status, started, duration, best_step, json_path, notes = sys.argv[2:]
line = f"## Cement + ReasoningLM v2 + Qwen KD\n- Started: {started}\n- Status: {status}\n- Duration: {duration}\n"
if best_step:
    line += f"- Best checkpoint: step_{best_step}\n"
path = Path(json_path)
if json_path and path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = float(data.get("overall_accuracy", 0.0) or 0.0)
    at = dict(data.get("answer_type_accuracy", {}) or {})
    yn = float(at.get("yes/no", 0.0) or 0.0)
    num = float(at.get("number", 0.0) or 0.0)
    other = float(at.get("other", 0.0) or 0.0)
    line += f"- Key result: overall {overall:.4f}, y/n {yn:.4f}, num {num:.4f}, other {other:.4f}\n"
line += f"- Notes: {notes}\n\n"
with md.open("a", encoding="utf-8") as f:
    f.write(line)
PY
}

write_report() {
  runtime_exec_python - <<'PY' "${REPORT_MD}" "logs/${RUN_ID}" "logs/${RUN_ID}/step_9000.tar" "logs/${BUNDLE_ID}/bridge_full.json" "${LM_CKPT}"
import json, sys
from pathlib import Path
report_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
ckpt = Path(sys.argv[3])
eval_json = Path(sys.argv[4])
lm_ckpt = sys.argv[5]

def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

eval_data = load_json(eval_json)
lines = ["# Cement + ReasoningLM v2 + Qwen KD", ""]
lines.append(f"- run dir: `{run_dir}`")
lines.append(f"- LM checkpoint: `{lm_ckpt}`")
lines.append(f"- final checkpoint: `{ckpt}`")
if eval_data:
    at = eval_data.get("answer_type_accuracy", {}) or {}
    lines.append(
        f"- full eval: {float(eval_data.get('overall_accuracy', 0.0) or 0.0):.4f}"
        f" | y/n {float(at.get('yes/no', 0.0) or 0.0):.4f}"
        f" | num {float(at.get('number', 0.0) or 0.0):.4f}"
        f" | other {float(at.get('other', 0.0) or 0.0):.4f}"
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

clear_vram() {
  runtime_exec_python - <<'PY'
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass
gc.collect()
PY
}

run_main() {
  local started_h started_ts end_ts duration status=0
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  bundle_mark_start "${RUN_ID}" "lm=$(basename "${LM_CKPT}") seed=${SEED}"
  resume_runmm "${RUN_ID}" 9000 \
    --vision_model siglip_base \
    --vision_checkpoint "${VISION_CKPT}" \
    --lm_checkpoint "${LM_CKPT}" \
    --seed "${SEED}" \
    --batch_size "${TRAIN_BS}" \
    --grad_accum_steps "${TRAIN_GA}" \
    --eval_batch_size "${EVAL_BS}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH}" \
    --no-pin_memory \
    --answer_kd_labels_path "${TEACHER_DATA_DIR}" \
    --answer_kd_weight "${KD_WEIGHT}" \
    --answer_kd_temp "${KD_TEMP}" \
    --min_train_steps_per_s 0 || status=$?
  if [[ -f "logs/${RUN_ID}/step_9000.tar" ]]; then
    eval_ckpt_vqa "logs/${RUN_ID}/step_9000.tar" "logs/${BUNDLE_ID}/bridge_full.json" "${EVAL_BS}" --eval_batches 0 || status=$?
  fi
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    bundle_mark_end "${RUN_ID}" "status=0"
    append_progress_entry "COMPLETE" "${started_h}" "${duration}" "9000" "logs/${BUNDLE_ID}/bridge_full.json" "Fresh Cement bridge with reasoning LM v2 and Qwen answer-KD"
  else
    bundle_mark_fail "${RUN_ID}" "status=${status}"
    append_progress_entry "FAILED" "${started_h}" "${duration}" "$(latest_step "logs/${RUN_ID}")" "logs/${BUNDLE_ID}/bridge_full.json" "Run failed or partial"
  fi
  write_report
  bundle_log_line "EXPERIMENT COMPLETE ${BUNDLE_ID}"
  bundle_refresh_experiment_db
  return "${status}"
}

bundle_init "${BUNDLE_ID}" "${LATEST_LINK}"
if [[ ! -f "${PROGRESS_MD}" ]]; then
  printf '# Cement + ReasoningLM v2 + Qwen KD\n\n' > "${PROGRESS_MD}"
fi
bundle_log_line "BUNDLE START ${BUNDLE_ID}"
bundle_log_line "LM checkpoint target: ${LM_CKPT}"
bundle_log_line "Teacher labels: ${TEACHER_DATA_DIR}"

if [[ ! -f "${LM_CKPT}" ]]; then
  bundle_log_line "FAIL ${RUN_ID} missing_lm_checkpoint ${LM_CKPT}"
  bundle_refresh_experiment_db
  exit 1
fi

clear_vram
run_main
