#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-./.venv_local/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmgrid_cotrain_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"
PROGRESS_MD="${BUNDLE_DIR}/progress.md"
TIMELINE_LOG="${BUNDLE_DIR}/timeline.log"

SIGLIP_DIR="${SIGLIP_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"
SIGLIP_REMAP="${SIGLIP_REMAP:-logs/mmsemantic_remap_v1_debug/step_500.tar}"
SEED="${SEED:-35}"
BRIDGE_BATCH_SIZE="${BRIDGE_BATCH_SIZE:-96}"
BRIDGE_GRAD_ACCUM="${BRIDGE_GRAD_ACCUM:-2}"
BRIDGE_EVAL_BATCH_SIZE="${BRIDGE_EVAL_BATCH_SIZE:-128}"
BRIDGE_NUM_WORKERS="${BRIDGE_NUM_WORKERS:-4}"
BRIDGE_PREFETCH_FACTOR="${BRIDGE_PREFETCH_FACTOR:-1}"
COMP_BATCH_SIZE="${COMP_BATCH_SIZE:-96}"
COMP_GRAD_ACCUM="${COMP_GRAD_ACCUM:-2}"
COMP_EVAL_BATCH_SIZE="${COMP_EVAL_BATCH_SIZE:-96}"
COMP_NUM_WORKERS="${COMP_NUM_WORKERS:-2}"
COMP_PREFETCH_FACTOR="${COMP_PREFETCH_FACTOR:-1}"

mkdir -p "${BUNDLE_DIR}"
echo "# Variant A Co-Training Progress" > "${PROGRESS_MD}"
touch "${TIMELINE_LOG}"

timestamp() {
  date --iso-8601=seconds
}

log_timeline() {
  echo "[$(timestamp)] $*" | tee -a "${TIMELINE_LOG}"
}

latest_step() {
  local run_dir="$1"
  local latest
  latest="$(find "${run_dir}" -maxdepth 1 -name 'step_*.tar' -printf '%f\n' 2>/dev/null | sed -E 's/^step_([0-9]+)\.tar$/\1/' | sort -n | tail -1)"
  echo "${latest:-0}"
}

run_mm_resume() {
  local run_id="$1"
  local target_step="$2"
  shift 2
  local run_dir="logs/${run_id}"
  local step
  step="$(latest_step "${run_dir}")"
  if [[ "${step}" -ge "${target_step}" ]]; then
    return 0
  fi
  if [[ "${step}" -gt 0 ]]; then
    ./runmm_v1.sh "${run_id}" "${step}" "$@"
  else
    ./runmm_v1.sh "${run_id}" "$@"
  fi
}

materialize_full_eval_from_log() {
  local run_id="$1"
  local out_json="$2"
  local run_dir="logs/${run_id}"
  local logfile="${run_dir}/logfile.txt"
  [[ -f "${logfile}" ]] || return 1
  "${PYTHON_BIN}" - <<'PY' "${logfile}" "${out_json}"
import json
import re
import sys
from pathlib import Path

logfile = Path(sys.argv[1])
out_json = Path(sys.argv[2])
overall = None
answer_types = None
record_count = None
tag_seen = False
for line in logfile.read_text(encoding="utf-8").splitlines():
    m = re.search(r"\[eval:val\] overall_accuracy=([0-9.]+)", line)
    if m:
        overall = float(m.group(1))
        continue
    m = re.search(r"\[eval:val\] answer_type: yes/no=([0-9.]+) number=([0-9.]+) other=([0-9.]+)", line)
    if m:
        answer_types = {
            "yes/no": float(m.group(1)),
            "number": float(m.group(2)),
            "other": float(m.group(3)),
        }
        continue
    m = re.search(r"\[mm\] eval split=val samples=([0-9]+)", line)
    if m:
        record_count = int(m.group(1))
        continue
    if "tag=final_eval" in line:
        tag_seen = True
if not tag_seen or overall is None or answer_types is None:
    raise SystemExit(1)
payload = {
    "overall_accuracy": overall,
    "answer_type_accuracy": answer_types,
    "record_count": record_count,
    "source": str(logfile),
    "derived_from_logfile_final_eval": True,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
with out_json.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True)
print(out_json)
PY
}

append_progress() {
  local name="$1"
  local started="$2"
  local status="$3"
  local best_step="$4"
  local json_path="$5"
  local notes="$6"
  "${PYTHON_BIN}" - <<'PY' "${PROGRESS_MD}" "${name}" "${started}" "${status}" "${best_step}" "${json_path}" "${notes}"
import json, sys
progress, name, started, status, best_step, json_path, notes = sys.argv[1:]
md = []
md.append(f"\n## {name}")
md.append(f"- Started: {started}")
md.append(f"- Status: {status}")
if best_step and best_step != "0":
    md.append(f"- Best checkpoint: step_{best_step}")
if json_path and json_path != "none":
    data = json.load(open(json_path, "r", encoding="utf-8"))
    aty = data.get("answer_type_accuracy", {})
    md.append(
        "- Key result: overall "
        f"{float(data.get('overall_accuracy', 0.0)):.4f}, "
        f"y/n {float(aty.get('yes/no', 0.0)):.4f}, "
        f"num {float(aty.get('number', 0.0)):.4f}, "
        f"other {float(aty.get('other', 0.0)):.4f}"
    )
if notes:
    md.append(f"- Notes: {notes}")
with open(progress, "a", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
PY
}

best_step_from_prefix() {
  local prefix="$1"
  "${PYTHON_BIN}" - <<'PY' "${BUNDLE_DIR}" "${prefix}"
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1])
prefix = sys.argv[2]
best_step = 3000
best_acc = -1.0
for path in sorted(bundle.glob(f"{prefix}_step_*.json")):
    step = int(path.stem.split("_")[-1])
    data = json.load(open(path, "r", encoding="utf-8"))
    acc = float(data.get("overall_accuracy", 0.0))
    if acc > best_acc:
        best_acc = acc
        best_step = step
print(best_step)
PY
}

BRIDGE_RUN="${BUNDLE_ID}_bridge"
COMP_RUN="${BUNDLE_ID}_compression"

bridge_started="$(timestamp)"
log_timeline "START ${BRIDGE_RUN}"
run_mm_resume "${BRIDGE_RUN}" 9000 \
  --vision_model siglip_base \
  --vision_checkpoint "${SIGLIP_DIR}" \
  --seed "${SEED}" \
  --max_steps 9000 \
  --manual_max_steps \
  --batch_size "${BRIDGE_BATCH_SIZE}" \
  --grad_accum_steps "${BRIDGE_GRAD_ACCUM}" \
  --eval_batch_size "${BRIDGE_EVAL_BATCH_SIZE}" \
  --num_workers "${BRIDGE_NUM_WORKERS}" \
  --prefetch_factor "${BRIDGE_PREFETCH_FACTOR}" \
  --no-pin_memory \
  --log_every 20 \
  --eval_every 1000 \
  --eval_batches 100 \
  --ckpt_every 1000 \
  --final_eval_batches 0 \
  --lr 0.0002 \
  --lr_schedule cosine \
  --lr_warmup_steps 600 \
  --lr_min_ratio 0.15 \
  --freeze_mode bridge_plus_top_lm \
  --train_top_lm_layers 2 \
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 3 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_dropout 0.0 \
  --lm_visual_adapter_gate_init 0.5 \
  --semantic_bottleneck \
  --semantic_tokens 8 \
  --semantic_latent_dim 256 \
  --semantic_grid_access \
  --semantic_recon_loss_weight 0.0 \
  --semantic_consistency_loss_weight 0.0 \
  --semantic_format_loss_weight 0.0 \
  --semantic_format_loss_final_weight 0.0 \
  --min_train_steps_per_s 0

BRIDGE_FULL_JSON="${BUNDLE_DIR}/bridge_step_9000_full.json"
if [[ ! -f "${BRIDGE_FULL_JSON}" ]]; then
  if ! materialize_full_eval_from_log "${BRIDGE_RUN}" "${BRIDGE_FULL_JSON}" >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
      --checkpoint "logs/${BRIDGE_RUN}/step_9000.tar" \
      --batch_size "${BRIDGE_EVAL_BATCH_SIZE}" \
      --num_workers "${BRIDGE_NUM_WORKERS}" \
      --prefetch_factor "${BRIDGE_PREFETCH_FACTOR}" \
      --no-pin_memory \
      --eval_batches 0 \
      --no-disable_lm_visual_adapters \
      --output_json "${BRIDGE_FULL_JSON}"
  fi
fi
append_progress "bridge_cotrain" "${bridge_started}" "COMPLETE" "9000" "${BRIDGE_FULL_JSON}" "fresh bridge training with co-trained Variant A bottleneck"
log_timeline "END ${BRIDGE_RUN}"

comp_started="$(timestamp)"
log_timeline "START ${COMP_RUN}"
run_mm_resume "${COMP_RUN}" 3000 \
  --vision_model siglip_base \
  --vision_checkpoint "${SIGLIP_DIR}" \
  --seed "${SEED}" \
  --max_steps 3000 \
  --manual_max_steps \
  --batch_size "${COMP_BATCH_SIZE}" \
  --grad_accum_steps "${COMP_GRAD_ACCUM}" \
  --eval_batch_size "${COMP_EVAL_BATCH_SIZE}" \
  --num_workers "${COMP_NUM_WORKERS}" \
  --prefetch_factor "${COMP_PREFETCH_FACTOR}" \
  --no-pin_memory \
  --log_every 10 \
  --eval_every 500 \
  --eval_batches 100 \
  --ckpt_every 500 \
  --final_eval_batches 0 \
  --lr 0.0002 \
  --lr_schedule cosine \
  --lr_warmup_steps 200 \
  --freeze_mode semantic_bottleneck_only \
  --semantic_bottleneck \
  --semantic_tokens 8 \
  --semantic_latent_dim 256 \
  --semantic_grid_access \
  --semantic_recon_loss_weight 0.1 \
  --semantic_consistency_loss_weight 0.0 \
  --disable_lm_visual_adapters \
  --prefix_remap_present \
  --no-apply_prefix_remap_in_forward \
  --prefix_remap_checkpoint "${SIGLIP_REMAP}" \
  --semantic_format_loss_weight 0.3 \
  --semantic_format_loss_final_weight 0.0 \
  --semantic_format_anneal_start_step 2250 \
  --semantic_format_anneal_end_step 3000 \
  --init_from_mm_checkpoint "logs/${BRIDGE_RUN}/step_9000.tar" \
  --min_train_steps_per_s 0

for step in 500 1000 1500 2000 2500 3000; do
  ckpt="logs/${COMP_RUN}/step_${step}.tar"
  [[ -f "${ckpt}" ]] || continue
  out_json="${BUNDLE_DIR}/compression_periodic_step_${step}.json"
  if [[ ! -f "${out_json}" ]]; then
    "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
        --checkpoint "${ckpt}" \
      --batch_size "${COMP_EVAL_BATCH_SIZE}" \
      --num_workers "${COMP_NUM_WORKERS}" \
      --prefetch_factor "${COMP_PREFETCH_FACTOR}" \
      --no-pin_memory \
      --eval_batches 100 \
      --disable_lm_visual_adapters \
      --output_json "${out_json}"
  fi
done

BEST_STEP="$(best_step_from_prefix compression_periodic)"
BEST_CKPT="logs/${COMP_RUN}/step_${BEST_STEP}.tar"
COMP_FULL_JSON="${BUNDLE_DIR}/compression_best_full.json"
if [[ ! -f "${COMP_FULL_JSON}" ]]; then
  if [[ "${BEST_STEP}" == "3000" ]] && materialize_full_eval_from_log "${COMP_RUN}" "${COMP_FULL_JSON}" >/dev/null 2>&1; then
    :
  else
    "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
      --checkpoint "${BEST_CKPT}" \
      --batch_size "${COMP_EVAL_BATCH_SIZE}" \
      --num_workers "${COMP_NUM_WORKERS}" \
      --prefetch_factor "${COMP_PREFETCH_FACTOR}" \
      --no-pin_memory \
      --eval_batches 0 \
      --disable_lm_visual_adapters \
      --output_json "${COMP_FULL_JSON}"
  fi
fi

PROBE_JSON="${BUNDLE_DIR}/compression_probe.json"
if [[ ! -f "${PROBE_JSON}" ]]; then
  "${PYTHON_BIN}" -m tasks.mm_bridge.scripts.mm_semantic_probe \
    --checkpoint "${BEST_CKPT}" \
    --batch_size "${COMP_EVAL_BATCH_SIZE}" \
    --probe_batch_size 256 \
    --num_workers "${COMP_NUM_WORKERS}" \
    --prefetch_factor "${COMP_PREFETCH_FACTOR}" \
    --no-pin_memory \
    --limit_train 10000 \
    --limit_val 5000 \
    --answer_top_k 3000 \
    --epochs 10 \
    --lr 0.001 \
    --feature_pool flatten \
    --output_json "${PROBE_JSON}"
fi

append_progress "compression_cotrain" "${comp_started}" "COMPLETE" "${BEST_STEP}" "${COMP_FULL_JSON}" "compression tuning on top of co-trained Variant A bridge"
log_timeline "END ${COMP_RUN}"

echo "${BUNDLE_ID}"
