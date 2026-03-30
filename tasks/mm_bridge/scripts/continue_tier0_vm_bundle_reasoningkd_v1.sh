#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

BUNDLE_ID="${BUNDLE_ID:-mmtier0_vm_v1_20260330_001641}"
LM_CKPT="${LM_CKPT:-logs/lm_reasoningmix_v2_20260329_133946/step_45000.tar}"
RUN_ID="${RUN_ID:-${BUNDLE_ID}_winner_reasoningkd_bridge}"
KEY="${KEY:-winner_reasoningkd}"

source "${REPO_ROOT}/tasks/mm_bridge/scripts/launch_tier0_vm_bundle_v1.sh"

main_continue() {
  bundle_init "${BUNDLE_ID}" "${LATEST_LINK}"
  bundle_log_line "CONTINUE ${BUNDLE_ID} for fresh-init reasoning-LM v2 + KD SigLIP2 bridge"
  bundle_log_line "reasoningkd_lm_ckpt ${LM_CKPT}"

  clear_vram
  run_bridge_baseline "${KEY}" "${RUN_ID}" "siglip2_b16" "${SIGLIP2_DIR}" \
    "${BRIDGE_BS}" "${BRIDGE_GA}" "${BRIDGE_EVAL_BS}" "${BRIDGE_NUM_WORKERS}" "${BRIDGE_PREFETCH}" \
    --answer_kd_labels_path "${TEACHER_DATA_DIR}" \
    --answer_kd_weight 0.3 \
    --answer_kd_temp 4.0 || true
  clear_vram

  local ckpt
  ckpt="$(full_eval_ckpt_for_run "${RUN_ID}")"
  if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
    run_hard_suite "winner_reasoningkd_bridge" "${ckpt}" 96 "winner_reasoningkd_bridge" || true
  fi

  write_report
  bundle_log_line "CONTINUE COMPLETE ${BUNDLE_ID} reasoningkd"
  bundle_refresh_experiment_db
}

main_continue "$@"
