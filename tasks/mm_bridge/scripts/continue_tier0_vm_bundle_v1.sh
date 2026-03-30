#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export BUNDLE_ID="${BUNDLE_ID:-mmtier0_vm_v1_20260330_001641}"
source "${REPO_ROOT}/tasks/mm_bridge/scripts/launch_tier0_vm_bundle_v1.sh"

RUN_WINNER_FTKD="${BUNDLE_ID}_winner_ftkd_bridge"
RUN_FTKD_K8="${BUNDLE_ID}_winner_ftkd_k8"
TIMELINE="${BUNDLE_DIR}/timeline.log"

main() {
  if [[ ! -d "${BUNDLE_DIR}" ]]; then
    echo "Missing bundle dir: ${BUNDLE_DIR}" >&2
    return 1
  fi

  bundle_log_line "CONTINUE ${BUNDLE_ID} for KD-initialized FT + evals"
  bundle_refresh_experiment_db

  local frozen_winner_full frozen_winner_vm frozen_winner_dir
  frozen_winner_full="$(select_winner_by_full_eval "${BUNDLE_DIR}/siglip2_bridge_peak_full.json" "${BUNDLE_DIR}/pecore_bridge_peak_full.json")"
  if [[ -z "${frozen_winner_full}" ]]; then
    echo "No frozen winner trace found" >&2
    return 1
  fi
  if [[ "${frozen_winner_full}" == *siglip2* ]]; then
    frozen_winner_vm="siglip2_b16"
    frozen_winner_dir="${SIGLIP2_DIR}"
  else
    frozen_winner_vm="pe_core_b16"
    frozen_winner_dir="${PECORE_DIR}"
  fi

  local kd_final_step kd_peak_ckpt
  kd_final_step="$(full_eval_step_for_run "${RUN_WINNER_KD}")"
  kd_final_step="${kd_final_step:-9000}"
  kd_peak_ckpt="logs/${RUN_WINNER_KD}/step_${kd_final_step}.tar"
  if [[ ! -f "${kd_peak_ckpt}" ]]; then
    echo "Missing KD final checkpoint: ${kd_peak_ckpt}" >&2
    return 1
  fi

  clear_vram
  run_bridge_baseline "winner_ftkd" "${RUN_WINNER_FTKD}" "${frozen_winner_vm}" "${frozen_winner_dir}" \
    "${FINETUNE_BS}" "${FINETUNE_GA}" "${FINETUNE_EVAL_BS}" "${FINETUNE_NUM_WORKERS}" "${FINETUNE_PREFETCH}" \
    --answer_kd_labels_path "${TEACHER_DATA_DIR}" \
    --answer_kd_weight 0.3 \
    --answer_kd_temp 4.0 \
    --init_from_mm_checkpoint "${kd_peak_ckpt}" \
    --train_vision_last_n_blocks "${FINETUNE_LAST_N_BLOCKS}" \
    --vision_lr_scale "${FINETUNE_VISION_LR_SCALE}" || true
  clear_vram

  local kd_full ft_full kd_acc ft_acc
  kd_full="${BUNDLE_DIR}/winner_kd_bridge_peak_full.json"
  ft_full="${BUNDLE_DIR}/winner_ftkd_bridge_peak_full.json"
  kd_acc="$(runtime_exec_python - <<'PY' "${kd_full}"
import json, sys
print(float(json.loads(open(sys.argv[1], 'r', encoding='utf-8').read()).get('overall_accuracy', 0.0) or 0.0))
PY
)"
  ft_acc="$(runtime_exec_python - <<'PY' "${ft_full}"
import json, sys
print(float(json.loads(open(sys.argv[1], 'r', encoding='utf-8').read()).get('overall_accuracy', 0.0) or 0.0))
PY
)"
  kd_acc="${kd_acc:-0}"
  ft_acc="${ft_acc:-0}"

  local ft_final_step ft_peak_ckpt
  ft_final_step="$(full_eval_step_for_run "${RUN_WINNER_FTKD}")"
  ft_final_step="${ft_final_step:-9000}"
  ft_peak_ckpt="logs/${RUN_WINNER_FTKD}/step_${ft_final_step}.tar"

  if runtime_exec_python - <<'PY' "${ft_acc}" "${kd_acc}"
import sys
ft = float(sys.argv[1] or 0.0)
kd = float(sys.argv[2] or 0.0)
raise SystemExit(0 if ft > kd else 1)
PY
  then
    bundle_log_line "winner_stacked ftkd step=${ft_final_step}"
    clear_vram
    run_k8_compression "winner_ftkd_k8" "${RUN_FTKD_K8}" "${ft_peak_ckpt}" "${frozen_winner_vm}" "${frozen_winner_dir}" || true
    clear_vram
  else
    bundle_log_line "winner_stacked remains kd step=${kd_final_step}"
  fi

  bundle_mark_start "${BUNDLE_ID}_hard_suite_resume" "ftkd_and_missing"
  if [[ -f "${ft_peak_ckpt}" ]]; then
    run_hard_suite "winner_ftkd_bridge" "${ft_peak_ckpt}" 96 "winner_ftkd_bridge"
  fi
  if [[ -f "${BUNDLE_DIR}/winner_ftkd_k8_peak_full.json" ]]; then
    run_hard_suite "winner_ftkd_k8" "$(full_eval_ckpt_for_run "${RUN_FTKD_K8}")" 96 "winner_ftkd_k8"
  fi
  bundle_mark_end "${BUNDLE_ID}_hard_suite_resume" "done"

  write_report
  bundle_log_line "CONTINUE COMPLETE ${BUNDLE_ID}"
  bundle_refresh_experiment_db
}

main "$@"
