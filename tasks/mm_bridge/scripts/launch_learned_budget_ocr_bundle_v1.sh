#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmlearnbudget_ocrmix_clean9k_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"
TIMELINE="${BUNDLE_DIR}/timeline.log"
PROGRESS_MD="${BUNDLE_DIR}/progress.md"
LATEST_LINK="logs/mmlearnbudget_ocrmix_clean9k_v1_latest"
DB_LOG="${BUNDLE_DIR}/db_build.log"

TRAIN_RUN_ID="${TRAIN_RUN_ID:-${BUNDLE_ID}_ocrmix_train}"

NEW_VQA_EVAL_RUN_ID="${NEW_VQA_EVAL_RUN_ID:-${BUNDLE_ID}_ocrmix_vqa_eval}"
NEW_CHARTQA_EVAL_RUN_ID="${NEW_CHARTQA_EVAL_RUN_ID:-${BUNDLE_ID}_ocrmix_chartqa_eval}"
NEW_TEXTOCR_EVAL_RUN_ID="${NEW_TEXTOCR_EVAL_RUN_ID:-${BUNDLE_ID}_ocrmix_textocr_eval}"
NEW_GQA_EVAL_RUN_ID="${NEW_GQA_EVAL_RUN_ID:-${BUNDLE_ID}_ocrmix_gqa_eval}"

NEW_VQA_SCHED_RUN_ID="${NEW_VQA_SCHED_RUN_ID:-${BUNDLE_ID}_ocrmix_vqa_scheduler}"
NEW_CHARTQA_SCHED_RUN_ID="${NEW_CHARTQA_SCHED_RUN_ID:-${BUNDLE_ID}_ocrmix_chartqa_scheduler}"
NEW_TEXTOCR_SCHED_RUN_ID="${NEW_TEXTOCR_SCHED_RUN_ID:-${BUNDLE_ID}_ocrmix_textocr_scheduler}"
NEW_GQA_SCHED_RUN_ID="${NEW_GQA_SCHED_RUN_ID:-${BUNDLE_ID}_ocrmix_gqa_scheduler}"

NEW_VQA_FEATURE_RUN_ID="${NEW_VQA_FEATURE_RUN_ID:-${BUNDLE_ID}_ocrmix_vqa_features}"
NEW_CHARTQA_FEATURE_RUN_ID="${NEW_CHARTQA_FEATURE_RUN_ID:-${BUNDLE_ID}_ocrmix_chartqa_features}"
NEW_TEXTOCR_FEATURE_RUN_ID="${NEW_TEXTOCR_FEATURE_RUN_ID:-${BUNDLE_ID}_ocrmix_textocr_features}"
NEW_GQA_FEATURE_RUN_ID="${NEW_GQA_FEATURE_RUN_ID:-${BUNDLE_ID}_ocrmix_gqa_features}"

LEARNED_PREDICTOR_RUN_ID="${LEARNED_PREDICTOR_RUN_ID:-${BUNDLE_ID}_learned_budget}"
NEW_OCR_SUBSET_K2_RUN_ID="${NEW_OCR_SUBSET_K2_RUN_ID:-${BUNDLE_ID}_ocrmix_ocrsubset_k2}"
BASELINE_OCR_SUBSET_K2_RUN_ID="${BASELINE_OCR_SUBSET_K2_RUN_ID:-${BUNDLE_ID}_baseline_ocrsubset_k2}"
NEW_GQA_SLICE_K2_RUN_ID="${NEW_GQA_SLICE_K2_RUN_ID:-${BUNDLE_ID}_ocrmix_gqa_slices_k2}"
NEW_GQA_SLICE_K8_RUN_ID="${NEW_GQA_SLICE_K8_RUN_ID:-${BUNDLE_ID}_ocrmix_gqa_slices_k8}"
NEW_PROBE_K2_RUN_ID="${NEW_PROBE_K2_RUN_ID:-${BUNDLE_ID}_ocrmix_probe_k2}"
NEW_PROBE_K8_RUN_ID="${NEW_PROBE_K8_RUN_ID:-${BUNDLE_ID}_ocrmix_probe_k8}"

BASELINE_VQA_EVAL_RUN_ID="${BASELINE_VQA_EVAL_RUN_ID:-${BUNDLE_ID}_baseline_vqa_eval}"
BASELINE_CHARTQA_EVAL_RUN_ID="${BASELINE_CHARTQA_EVAL_RUN_ID:-${BUNDLE_ID}_baseline_chartqa_eval}"
BASELINE_TEXTOCR_EVAL_RUN_ID="${BASELINE_TEXTOCR_EVAL_RUN_ID:-${BUNDLE_ID}_baseline_textocr_eval}"
BASELINE_GQA_EVAL_RUN_ID="${BASELINE_GQA_EVAL_RUN_ID:-${BUNDLE_ID}_baseline_gqa_eval}"

SOURCE_CKPT="${SOURCE_CKPT:-logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar}"
BASELINE_CKPT="${BASELINE_CKPT:-logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_train/step_3000.tar}"
LM_CKPT="${LM_CKPT:-logs/lm_final/step_45000.tar}"
SIGLIP2_DIR="${SIGLIP2_DIR:-logs/hf_vision/openclip_siglip2_b16_webli}"
REMAP_CKPT="${REMAP_CKPT:-logs/mmsemantic_remap_v1_debug/step_500.tar}"

CHARTQA_DB_PATH="${CHARTQA_DB_PATH:-data/vm_ssl/db/vm_ssl.duckdb}"
TEXTOCR_ANN_ROOT="${TEXTOCR_ANN_ROOT:-data/vm_ssl/raw/textocr_full}"
TEXTOCR_IMG_ROOT="${TEXTOCR_IMG_ROOT:-data/vm_ssl/raw/textocr_trainval}"
GQA_ROOT="${GQA_ROOT:-data/gqa}"

SEED="${SEED:-35}"
TRAIN_BUDGETS="${TRAIN_BUDGETS:-2,4,8,16}"
EVAL_BUDGETS="${EVAL_BUDGETS:-2,4,8,16}"
PREDICTOR_BUDGETS="${PREDICTOR_BUDGETS:-2,4,8}"
NEW_STATS_BUDGETS="${NEW_STATS_BUDGETS:-2,4,8}"
BASELINE_STATS_BUDGETS="${BASELINE_STATS_BUDGETS:-}"

TRAIN_BS="${TRAIN_BS:-96}"
TRAIN_GA="${TRAIN_GA:-2}"
TRAIN_EVAL_BS="${TRAIN_EVAL_BS:-160}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
TRAIN_PREFETCH="${TRAIN_PREFETCH:-1}"
TRAIN_PIN_MEMORY="${TRAIN_PIN_MEMORY:-0}"

EVAL_BS="${EVAL_BS:-160}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
EVAL_PREFETCH="${EVAL_PREFETCH:-2}"
EVAL_PIN_MEMORY="${EVAL_PIN_MEMORY:-1}"

FEATURE_BS="${FEATURE_BS:-160}"
FEATURE_NUM_WORKERS="${FEATURE_NUM_WORKERS:-4}"
FEATURE_PREFETCH="${FEATURE_PREFETCH:-2}"
FEATURE_PIN_MEMORY="${FEATURE_PIN_MEMORY:-1}"

PROBE_BS="${PROBE_BS:-32}"
PROBE_BATCH_BS="${PROBE_BATCH_BS:-256}"
PROBE_NUM_WORKERS="${PROBE_NUM_WORKERS:-4}"
PROBE_PREFETCH="${PROBE_PREFETCH:-2}"
PROBE_PIN_MEMORY="${PROBE_PIN_MEMORY:-1}"

OCR_SUBSET_LIMIT="${OCR_SUBSET_LIMIT:-5000}"
GQA_SLICE_LIMIT="${GQA_SLICE_LIMIT:-100000}"
PROBE_LIMIT_TRAIN="${PROBE_LIMIT_TRAIN:-10000}"
PROBE_LIMIT_VAL="${PROBE_LIMIT_VAL:-5000}"

PREDICTOR_EPOCHS="${PREDICTOR_EPOCHS:-12}"
PREDICTOR_BATCH_SIZE="${PREDICTOR_BATCH_SIZE:-2048}"
PREDICTOR_HIDDEN_DIM="${PREDICTOR_HIDDEN_DIM:-128}"
PREDICTOR_DROPOUT="${PREDICTOR_DROPOUT:-0.1}"
PREDICTOR_LR="${PREDICTOR_LR:-0.001}"
PREDICTOR_WD="${PREDICTOR_WD:-0.0001}"

NEW_VQA_EVAL_DIR="${BUNDLE_DIR}/ocrmix_vqa_eval"
NEW_CHARTQA_EVAL_DIR="${BUNDLE_DIR}/ocrmix_chartqa_eval"
NEW_TEXTOCR_EVAL_DIR="${BUNDLE_DIR}/ocrmix_textocr_eval"
NEW_GQA_EVAL_DIR="${BUNDLE_DIR}/ocrmix_gqa_eval"

BASELINE_VQA_EVAL_DIR="${BUNDLE_DIR}/baseline_vqa_eval"
BASELINE_CHARTQA_EVAL_DIR="${BUNDLE_DIR}/baseline_chartqa_eval"
BASELINE_TEXTOCR_EVAL_DIR="${BUNDLE_DIR}/baseline_textocr_eval"
BASELINE_GQA_EVAL_DIR="${BUNDLE_DIR}/baseline_gqa_eval"

NEW_VQA_FEATURE_DIR="${BUNDLE_DIR}/ocrmix_vqa_features"
NEW_CHARTQA_FEATURE_DIR="${BUNDLE_DIR}/ocrmix_chartqa_features"
NEW_TEXTOCR_FEATURE_DIR="${BUNDLE_DIR}/ocrmix_textocr_features"
NEW_GQA_FEATURE_DIR="${BUNDLE_DIR}/ocrmix_gqa_features"

NEW_VQA_SCHED_JSON="${BUNDLE_DIR}/ocrmix_vqa_scheduler_summary.json"
NEW_CHARTQA_SCHED_JSON="${BUNDLE_DIR}/ocrmix_chartqa_scheduler_summary.json"
NEW_TEXTOCR_SCHED_JSON="${BUNDLE_DIR}/ocrmix_textocr_scheduler_summary.json"
NEW_GQA_SCHED_JSON="${BUNDLE_DIR}/ocrmix_gqa_scheduler_summary.json"
LEARNED_PREDICTOR_JSON="${BUNDLE_DIR}/learned_budget_predictor_summary.json"
LEARNED_PREDICTOR_MANIFEST="${BUNDLE_DIR}/learned_budget_predictor_manifest.json"

mkdir -p "${BUNDLE_DIR}"
if [[ ! -f "${TIMELINE}" ]]; then
  : > "${TIMELINE}"
fi
if [[ ! -f "${PROGRESS_MD}" ]]; then
cat > "${PROGRESS_MD}" <<'EOF'
# Learned Budget OCR Bundle Progress

EOF
fi
ln -sfn "${BUNDLE_ID}" "${LATEST_LINK}"

log_line() {
  local line="[$(date)] $*"
  echo "${line}" | tee -a "${TIMELINE}"
}

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

append_progress_entry() {
  local title="$1"
  local status="$2"
  local started="$3"
  local duration="$4"
  local notes="$5"
  runtime_exec_python - <<'PY' "${PROGRESS_MD}" "${title}" "${status}" "${started}" "${duration}" "${notes}"
import sys
from pathlib import Path
md = Path(sys.argv[1])
title, status, started, duration, notes = sys.argv[2:]
with md.open("a", encoding="utf-8") as f:
    f.write(f"## {title}\n")
    f.write(f"- Started: {started}\n")
    f.write(f"- Status: {status}\n")
    f.write(f"- Duration: {duration}\n")
    f.write(f"- Notes: {notes}\n\n")
PY
}

rebuild_db() {
  local reason="$1"
  {
    echo "[$(date)] rebuild_db reason=${reason}"
    python3 scripts/build_experiment_db.py --task mm_bridge
  } >> "${DB_LOG}" 2>&1 || true
}

schedule_delayed_rebuild() {
  local reason="$1"
  local delay_s="${2:-30}"
  (
    sleep "${delay_s}"
    cd "${REPO_ROOT}"
    {
      echo "[$(date)] rebuild_db reason=${reason}"
      python3 scripts/build_experiment_db.py --task mm_bridge
    } >> "${DB_LOG}" 2>&1 || true
  ) >/dev/null 2>&1 &
}

prepare_run_dir() {
  local run_id="$1"
  local eval_only="${2:-0}"
  local run_dir="logs/${run_id}"
  local log_path="${run_dir}/logfile.txt"
  mkdir -p "${run_dir}"
  touch "${log_path}"
  if [[ "${eval_only}" == "1" ]]; then
    printf '[launcher] eval_only=1 bundle=%s run=%s\n' "${BUNDLE_ID}" "${run_id}" >> "${log_path}"
  else
    printf '[launcher] bundle=%s run=%s\n' "${BUNDLE_ID}" "${run_id}" >> "${log_path}"
  fi
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

pin_memory_flag() {
  local enabled="${1:-0}"
  if [[ "${enabled}" == "1" ]]; then
    printf '%s\n' "--pin_memory"
  else
    printf '%s\n' "--no-pin_memory"
  fi
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

run_logged_phase() {
  local title="$1"
  local run_id="$2"
  local phase="$3"
  local done_path="$4"
  local success_note="$5"
  shift 5

  if [[ -n "${done_path}" && -f "${done_path}" ]]; then
    log_line "SKIP  ${run_id} phase=${phase} artifact=${done_path}"
    return 0
  fi

  prepare_run_dir "${run_id}" 1
  local run_dir="logs/${run_id}"
  local run_log="${run_dir}/logfile.txt"
  local started_h started_ts end_ts duration status=0
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"

  log_line "START ${run_id} phase=${phase}"
  rebuild_db "${phase}_start"
  schedule_delayed_rebuild "${phase}_start_delayed" 45
  {
    printf '[phase] eval_only=1 bundle=%s run=%s phase=%s\n' "${BUNDLE_ID}" "${run_id}" "${phase}"
    PYTHONUNBUFFERED=1 "$@"
  } >> "${run_log}" 2>&1 || status=$?
  if (( status == 0 )) && [[ -n "${done_path}" ]] && [[ ! -f "${done_path}" ]]; then
    status=11
    printf '[phase] missing_expected_artifact=%s\n' "${done_path}" >> "${run_log}"
  fi
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END   ${run_id} status=0 phase=${phase}"
    append_progress_entry "${title}" "COMPLETE" "${started_h}" "${duration}" "${success_note}"
  else
    log_line "FAIL  ${run_id} status=${status} phase=${phase}"
    append_progress_entry "${title}" "FAILED" "${started_h}" "${duration}" "See ${run_log}."
  fi
  rebuild_db "${phase}_end"
  clear_vram
  return "${status}"
}

run_train_phase() {
  local run_dir="logs/${TRAIN_RUN_ID}"
  local final_ckpt="${run_dir}/step_3000.tar"
  local started_h started_ts end_ts duration status=0

  prepare_run_dir "${TRAIN_RUN_ID}" 0
  if [[ -f "${final_ckpt}" ]]; then
    log_line "SKIP  ${TRAIN_RUN_ID} phase=train artifact=${final_ckpt}"
    return 0
  fi

  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  log_line "START ${TRAIN_RUN_ID} phase=train"
  rebuild_db "train_start"
  schedule_delayed_rebuild "train_start_delayed" 45

  resume_runmm "${TRAIN_RUN_ID}" 3000 \
    --vision_model siglip2_b16 \
    --vision_checkpoint "${SIGLIP2_DIR}" \
    --lm_checkpoint "${LM_CKPT}" \
    --seed "${SEED}" \
    --mm_sdp_backend math \
    --max_steps 3000 \
    --manual_max_steps \
    --batch_size "${TRAIN_BS}" \
    --grad_accum_steps "${TRAIN_GA}" \
    --eval_batch_size "${TRAIN_EVAL_BS}" \
    --num_workers "${TRAIN_NUM_WORKERS}" \
    --prefetch_factor "${TRAIN_PREFETCH}" \
    "$(pin_memory_flag "${TRAIN_PIN_MEMORY}")" \
    --log_every 20 \
    --eval_every 0 \
    --ckpt_every 500 \
    --final_eval_batches 1 \
    --lr 0.0002 \
    --lr_schedule cosine \
    --lr_warmup_steps 200 \
    --lr_min_ratio 0.15 \
    --freeze_mode semantic_bottleneck_only \
    --bridge_question_context_mode question_only \
    --bridge_query_bank_mode question_hidden_attn \
    --semantic_bottleneck \
    --semantic_tokens 16 \
    --semantic_latent_dim 256 \
    --semantic_recon_loss_weight 0.1 \
    --semantic_consistency_loss_weight 0.0 \
    --disable_lm_visual_adapters \
    --prefix_remap_present \
    --no-apply_prefix_remap_in_forward \
    --prefix_remap_checkpoint "${REMAP_CKPT}" \
    --semantic_format_loss_weight 0.3 \
    --semantic_format_loss_final_weight 0.0 \
    --semantic_format_anneal_start_step 2250 \
    --semantic_format_anneal_end_step 3000 \
    --semantic_budget_options "${TRAIN_BUDGETS}" \
    --semantic_budget_sample_mode batch \
    --init_from_mm_checkpoint "${SOURCE_CKPT}" \
    --eval_use_kv_cache \
    --eval_kv_cache_mode batched \
    --chartqa_train_mix_ratio 0.25 \
    --textocr_train_mix_ratio 0.10 \
    --chartqa_db_path "${CHARTQA_DB_PATH}" \
    --textocr_annotations_root "${TEXTOCR_ANN_ROOT}" \
    --textocr_images_root "${TEXTOCR_IMG_ROOT}" \
    --min_train_steps_per_s 0 || status=$?

  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )) && [[ -f "${final_ckpt}" ]]; then
    log_line "END   ${TRAIN_RUN_ID} status=0 phase=train"
    append_progress_entry "Train OCR/chart-aware clean-9k variable-prefix compressor" "COMPLETE" "${started_h}" "${duration}" \
      "Source=${SOURCE_CKPT}; budgets=${TRAIN_BUDGETS}; mix=0.65/0.25/0.10 approx; checkpoint=${final_ckpt}."
  else
    log_line "FAIL  ${TRAIN_RUN_ID} status=${status} phase=train"
    append_progress_entry "Train OCR/chart-aware clean-9k variable-prefix compressor" "FAILED" "${started_h}" "${duration}" \
      "Training failed or stopped early."
  fi
  rebuild_db "train_end"
  clear_vram
  return "${status}"
}

run_eval_suite_phase() {
  local title="$1"
  local run_id="$2"
  local checkpoint="$3"
  local output_dir="$4"
  local eval_split="$5"
  local scorer="$6"
  local stats_budgets="$7"
  run_logged_phase "${title}" "${run_id}" "${eval_split}_eval" "${output_dir}/summary.json" "${title} wrote ${output_dir}/summary.json." \
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_budget_eval_suite \
      --checkpoint "${checkpoint}" \
      --budgets "${EVAL_BUDGETS}" \
      --probe_budgets "" \
      --stats_budgets "${stats_budgets}" \
      --eval_split "${eval_split}" \
      --scorer "${scorer}" \
      --batch_size "${EVAL_BS}" \
      --num_workers "${EVAL_NUM_WORKERS}" \
      --prefetch_factor "${EVAL_PREFETCH}" \
      "$(pin_memory_flag "${EVAL_PIN_MEMORY}")" \
      --gqa_root "${GQA_ROOT}" \
      --chartqa_db_path "${CHARTQA_DB_PATH}" \
      --textocr_annotations_root "${TEXTOCR_ANN_ROOT}" \
      --textocr_images_root "${TEXTOCR_IMG_ROOT}" \
      --eval_use_kv_cache \
      --eval_kv_cache_mode batched \
      --output_dir "${output_dir}"
}

run_scheduler_phase() {
  local title="$1"
  local run_id="$2"
  local eval_dir="$3"
  local output_json="$4"
  run_logged_phase "${title}" "${run_id}" "scheduler" "${output_json}" "${title} wrote ${output_json}." \
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_budget_scheduler_sweep \
      --eval_dir "${eval_dir}" \
      --main_base_budget 4 \
      --side_base_budget 2 \
      --output_json "${output_json}"
}

run_feature_export_phase() {
  local title="$1"
  local run_id="$2"
  local checkpoint="$3"
  local eval_split="$4"
  local output_dir="$5"
  run_logged_phase "${title}" "${run_id}" "${eval_split}_features" "${output_dir}/feature_manifest.json" "${title} wrote ${output_dir}/feature_manifest.json." \
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_budget_feature_export \
      --checkpoint "${checkpoint}" \
      --budgets "${PREDICTOR_BUDGETS}" \
      --eval_split "${eval_split}" \
      --batch_size "${FEATURE_BS}" \
      --num_workers "${FEATURE_NUM_WORKERS}" \
      --prefetch_factor "${FEATURE_PREFETCH}" \
      "$(pin_memory_flag "${FEATURE_PIN_MEMORY}")" \
      --gqa_root "${GQA_ROOT}" \
      --chartqa_db_path "${CHARTQA_DB_PATH}" \
      --textocr_annotations_root "${TEXTOCR_ANN_ROOT}" \
      --textocr_images_root "${TEXTOCR_IMG_ROOT}" \
      --output_dir "${output_dir}"
}

write_predictor_manifest() {
  runtime_exec_python - <<'PY' "${LEARNED_PREDICTOR_MANIFEST}" "${NEW_VQA_EVAL_DIR}" "${NEW_VQA_FEATURE_DIR}" "${NEW_CHARTQA_EVAL_DIR}" "${NEW_CHARTQA_FEATURE_DIR}" "${NEW_TEXTOCR_EVAL_DIR}" "${NEW_TEXTOCR_FEATURE_DIR}"
import json
import os
import sys

manifest_path = os.path.abspath(sys.argv[1])
payload = {
    "sources": [
        {"dataset_name": "vqav2", "eval_dir": os.path.abspath(sys.argv[2]), "feature_dir": os.path.abspath(sys.argv[3])},
        {"dataset_name": "chartqa", "eval_dir": os.path.abspath(sys.argv[4]), "feature_dir": os.path.abspath(sys.argv[5])},
        {"dataset_name": "textocr_readout", "eval_dir": os.path.abspath(sys.argv[6]), "feature_dir": os.path.abspath(sys.argv[7])},
    ]
}
os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True)
print(manifest_path)
PY
}

run_predictor_phase() {
  write_predictor_manifest >/dev/null
  run_logged_phase "Train learned cascade budget predictor" "${LEARNED_PREDICTOR_RUN_ID}" "learned_budget" "${LEARNED_PREDICTOR_JSON}" "Learned budget predictor wrote ${LEARNED_PREDICTOR_JSON}." \
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_budget_learned_predictor \
      --manifest_json "${LEARNED_PREDICTOR_MANIFEST}" \
      --hidden_dim "${PREDICTOR_HIDDEN_DIM}" \
      --dropout "${PREDICTOR_DROPOUT}" \
      --epochs "${PREDICTOR_EPOCHS}" \
      --batch_size "${PREDICTOR_BATCH_SIZE}" \
      --lr "${PREDICTOR_LR}" \
      --weight_decay "${PREDICTOR_WD}" \
      --output_json "${LEARNED_PREDICTOR_JSON}"
}

run_ocr_subset_phase() {
  local title="$1"
  local run_id="$2"
  local checkpoint="$3"
  local budget="$4"
  local output_json="$5"
  run_logged_phase "${title}" "${run_id}" "ocr_subset_k${budget}" "${output_json}" "${title} wrote ${output_json}." \
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_ocr_subset_eval \
      --checkpoint "${checkpoint}" \
      --batch_size "${EVAL_BS}" \
      --num_workers "${EVAL_NUM_WORKERS}" \
      --prefetch_factor "${EVAL_PREFETCH}" \
      "$(pin_memory_flag "${EVAL_PIN_MEMORY}")" \
      --limit_ocr "${OCR_SUBSET_LIMIT}" \
      --semantic_eval_budget "${budget}" \
      --output_json "${output_json}"
}

run_gqa_slice_phase() {
  local title="$1"
  local run_id="$2"
  local checkpoint="$3"
  local budget="$4"
  local output_json="$5"
  run_logged_phase "${title}" "${run_id}" "gqa_slices_k${budget}" "${output_json}" "${title} wrote ${output_json}." \
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_gqa_eval \
      --checkpoint "${checkpoint}" \
      --batch_size "${EVAL_BS}" \
      --num_workers "${EVAL_NUM_WORKERS}" \
      --prefetch_factor "${EVAL_PREFETCH}" \
      "$(pin_memory_flag "${EVAL_PIN_MEMORY}")" \
      --gqa_root "${GQA_ROOT}" \
      --semantic_eval_budget "${budget}" \
      --limit_eval "${GQA_SLICE_LIMIT}" \
      --output_json "${output_json}"
}

run_probe_phase() {
  local title="$1"
  local run_id="$2"
  local checkpoint="$3"
  local budget="$4"
  local output_json="$5"
  run_logged_phase "${title}" "${run_id}" "probe_k${budget}" "${output_json}" "${title} wrote ${output_json}." \
    runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_probe \
      --checkpoint "${checkpoint}" \
      --batch_size "${PROBE_BS}" \
      --probe_batch_size "${PROBE_BATCH_BS}" \
      --num_workers "${PROBE_NUM_WORKERS}" \
      --prefetch_factor "${PROBE_PREFETCH}" \
      "$(pin_memory_flag "${PROBE_PIN_MEMORY}")" \
      --limit_train "${PROBE_LIMIT_TRAIN}" \
      --limit_val "${PROBE_LIMIT_VAL}" \
      --semantic_eval_budget "${budget}" \
      --output_json "${output_json}"
}

main() {
  local new_ckpt="logs/${TRAIN_RUN_ID}/step_3000.tar"
  local status=0

  prepare_run_dir "${TRAIN_RUN_ID}" 0
  rebuild_db "bundle_start"
  schedule_delayed_rebuild "bundle_start_delayed" 10
  log_line "BUNDLE ${BUNDLE_ID} source_ckpt=${SOURCE_CKPT} baseline_ckpt=${BASELINE_CKPT}"

  run_train_phase || return $?

  run_eval_suite_phase "New OCR-aware VQAv2 fixed/oracle eval" "${NEW_VQA_EVAL_RUN_ID}" "${new_ckpt}" "${NEW_VQA_EVAL_DIR}" "val" "official" "${NEW_STATS_BUDGETS}" || return $?
  run_eval_suite_phase "New OCR-aware ChartQA fixed/oracle eval" "${NEW_CHARTQA_EVAL_RUN_ID}" "${new_ckpt}" "${NEW_CHARTQA_EVAL_DIR}" "chartqa_val" "text_exact" "${NEW_STATS_BUDGETS}" || return $?
  run_eval_suite_phase "New OCR-aware TextOCR fixed/oracle eval" "${NEW_TEXTOCR_EVAL_RUN_ID}" "${new_ckpt}" "${NEW_TEXTOCR_EVAL_DIR}" "textocr_val" "text_exact" "${NEW_STATS_BUDGETS}" || return $?
  run_scheduler_phase "New VQAv2 cheap scheduler sweep" "${NEW_VQA_SCHED_RUN_ID}" "${NEW_VQA_EVAL_DIR}" "${NEW_VQA_SCHED_JSON}" || return $?
  run_scheduler_phase "New ChartQA cheap scheduler sweep" "${NEW_CHARTQA_SCHED_RUN_ID}" "${NEW_CHARTQA_EVAL_DIR}" "${NEW_CHARTQA_SCHED_JSON}" || return $?
  run_scheduler_phase "New TextOCR cheap scheduler sweep" "${NEW_TEXTOCR_SCHED_RUN_ID}" "${NEW_TEXTOCR_EVAL_DIR}" "${NEW_TEXTOCR_SCHED_JSON}" || return $?

  run_feature_export_phase "Export VQAv2 low-budget features" "${NEW_VQA_FEATURE_RUN_ID}" "${new_ckpt}" "val" "${NEW_VQA_FEATURE_DIR}" || return $?
  run_feature_export_phase "Export ChartQA low-budget features" "${NEW_CHARTQA_FEATURE_RUN_ID}" "${new_ckpt}" "chartqa_val" "${NEW_CHARTQA_FEATURE_DIR}" || return $?
  run_feature_export_phase "Export TextOCR low-budget features" "${NEW_TEXTOCR_FEATURE_RUN_ID}" "${new_ckpt}" "textocr_val" "${NEW_TEXTOCR_FEATURE_DIR}" || return $?

  run_predictor_phase || return $?

  run_ocr_subset_phase "New OCR subset eval at K=2" "${NEW_OCR_SUBSET_K2_RUN_ID}" "${new_ckpt}" 2 "${BUNDLE_DIR}/ocrmix_ocr_subset_k2.json" || return $?
  run_gqa_slice_phase "New GQA slices at K=2" "${NEW_GQA_SLICE_K2_RUN_ID}" "${new_ckpt}" 2 "${BUNDLE_DIR}/ocrmix_gqa_slices_k2.json" || return $?
  run_gqa_slice_phase "New GQA slices at K=8" "${NEW_GQA_SLICE_K8_RUN_ID}" "${new_ckpt}" 8 "${BUNDLE_DIR}/ocrmix_gqa_slices_k8.json" || return $?
  run_probe_phase "New semantic probe at K=2" "${NEW_PROBE_K2_RUN_ID}" "${new_ckpt}" 2 "${BUNDLE_DIR}/ocrmix_probe_k2.json" || return $?
  run_probe_phase "New semantic probe at K=8" "${NEW_PROBE_K8_RUN_ID}" "${new_ckpt}" 8 "${BUNDLE_DIR}/ocrmix_probe_k8.json" || return $?

  run_eval_suite_phase "Baseline clean varprefix VQAv2 fixed/oracle eval" "${BASELINE_VQA_EVAL_RUN_ID}" "${BASELINE_CKPT}" "${BASELINE_VQA_EVAL_DIR}" "val" "official" "${BASELINE_STATS_BUDGETS}" || return $?
  run_eval_suite_phase "Baseline clean varprefix ChartQA fixed/oracle eval" "${BASELINE_CHARTQA_EVAL_RUN_ID}" "${BASELINE_CKPT}" "${BASELINE_CHARTQA_EVAL_DIR}" "chartqa_val" "text_exact" "${BASELINE_STATS_BUDGETS}" || return $?
  run_eval_suite_phase "Baseline clean varprefix TextOCR fixed/oracle eval" "${BASELINE_TEXTOCR_EVAL_RUN_ID}" "${BASELINE_CKPT}" "${BASELINE_TEXTOCR_EVAL_DIR}" "textocr_val" "text_exact" "${BASELINE_STATS_BUDGETS}" || return $?
  run_ocr_subset_phase "Baseline OCR subset eval at K=2" "${BASELINE_OCR_SUBSET_K2_RUN_ID}" "${BASELINE_CKPT}" 2 "${BUNDLE_DIR}/baseline_ocr_subset_k2.json" || return $?

  rebuild_db "bundle_end"
  log_line "BUNDLE COMPLETE ${BUNDLE_ID} status=0"
}

main "$@"
