#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/runtime_exec.sh"
source "${REPO_ROOT}/tasks/mm_bridge/scripts/experiment_bundle_lib.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmtier0_vm_v1_${STAMP}}"
LATEST_LINK="logs/mmtier0_vm_v1_latest"
BUNDLE_DIR="logs/${BUNDLE_ID}"
PROGRESS_MD="${BUNDLE_DIR}/progress.md"
REPORT_MD="${BUNDLE_DIR}/report.md"

LM_CKPT="${LM_CKPT:-logs/lm_final/step_45000.tar}"
TEACHER_DATA_DIR="${TEACHER_DATA_DIR:-data/distillation/qwen25vl3b_vqav2_train_v1}"
SIGLIP2_DIR="${SIGLIP2_DIR:-logs/hf_vision/openclip_siglip2_b16_webli}"
PECORE_DIR="${PECORE_DIR:-logs/hf_vision/openclip_pe_core_b16_meta}"
SIGLIP_BASE_DIR="${SIGLIP_BASE_DIR:-logs/hf_vision/google_siglip_base_patch16_224}"
REFERENCE_BRIDGE_CKPT="${REFERENCE_BRIDGE_CKPT:-logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar}"
REFERENCE_COMPRESS_CKPT="${REFERENCE_COMPRESS_CKPT:-logs/mmqwenkd_v1_20260327_224015_distill_k8/step_3000.tar}"
REMAP_CKPT="${REMAP_CKPT:-logs/mmsemantic_remap_v1_debug/step_500.tar}"

SEED="${SEED:-35}"

BRIDGE_BS="${BRIDGE_BS:-96}"
BRIDGE_GA="${BRIDGE_GA:-2}"
BRIDGE_EVAL_BS="${BRIDGE_EVAL_BS:-128}"
BRIDGE_NUM_WORKERS="${BRIDGE_NUM_WORKERS:-2}"
BRIDGE_PREFETCH="${BRIDGE_PREFETCH:-1}"

FINETUNE_BS="${FINETUNE_BS:-64}"
FINETUNE_GA="${FINETUNE_GA:-3}"
FINETUNE_EVAL_BS="${FINETUNE_EVAL_BS:-96}"
FINETUNE_NUM_WORKERS="${FINETUNE_NUM_WORKERS:-2}"
FINETUNE_PREFETCH="${FINETUNE_PREFETCH:-1}"
FINETUNE_LAST_N_BLOCKS="${FINETUNE_LAST_N_BLOCKS:-2}"
FINETUNE_VISION_LR_SCALE="${FINETUNE_VISION_LR_SCALE:-0.1}"

COMPRESS_BS="${COMPRESS_BS:-96}"
COMPRESS_GA="${COMPRESS_GA:-2}"
COMPRESS_EVAL_BS="${COMPRESS_EVAL_BS:-96}"
COMPRESS_NUM_WORKERS="${COMPRESS_NUM_WORKERS:-2}"
COMPRESS_PREFETCH="${COMPRESS_PREFETCH:-1}"
COMPRESS_K="${COMPRESS_K:-8}"

GQA_EVAL_LIMIT="${GQA_EVAL_LIMIT:-5000}"
GQA_EVAL_BS="${GQA_EVAL_BS:-96}"
OCR_LIMIT="${OCR_LIMIT:-500}"
PROBE_BATCH="${PROBE_BATCH:-256}"
PROBE_TRAIN_LIMIT="${PROBE_TRAIN_LIMIT:-10000}"
PROBE_VAL_LIMIT="${PROBE_VAL_LIMIT:-5000}"

DO_COMPRESS_WINNING_FROZEN="${DO_COMPRESS_WINNING_FROZEN:-1}"
DO_COMPRESS_WINNING_STACKED="${DO_COMPRESS_WINNING_STACKED:-1}"

RUN_SIGLIP2="${BUNDLE_ID}_siglip2_frozen_bridge"
RUN_PECORE="${BUNDLE_ID}_pecore_frozen_bridge"
RUN_WINNER_KD="${BUNDLE_ID}_winner_kd_bridge"
RUN_WINNER_FT="${BUNDLE_ID}_winner_ft_bridge"
RUN_WINNER_K8="${BUNDLE_ID}_winner_frozen_k8"
RUN_STACKED_K8="${BUNDLE_ID}_winner_stacked_k8"

run_stdout_log() {
  local name="$1"
  printf '%s\n' "${BUNDLE_DIR}/${name}.stdout.log"
}

timestamp_h() {
  date '+%Y-%m-%d %H:%M:%S'
}

timestamp_iso() {
  date --iso-8601=seconds
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

gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true
  fi
}

host_ram_snapshot() {
  free -m | awk '/^Mem:/ {printf "%s/%s MB", $3, $2}'
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

append_progress() {
  local name="$1"
  local started="$2"
  local status="$3"
  local duration="$4"
  local best_step="$5"
  local json_path="$6"
  local notes="$7"
  runtime_exec_python - <<'PY' "${PROGRESS_MD}" "${name}" "${started}" "${status}" "${duration}" "${best_step}" "${json_path}" "${notes}"
import json, sys
from pathlib import Path
progress, name, started, status, duration, best_step, json_path, notes = sys.argv[1:]
lines = [f"## {name}", f"- Started: {started}", f"- Status: {status}", f"- Duration: {duration}"]
if best_step and best_step != "0":
    lines.append(f"- Best checkpoint: step_{best_step}")
path = Path(json_path)
if json_path and json_path != "none" and path.is_file():
    data = json.loads(path.read_text(encoding="utf-8"))
    aty = dict(data.get("answer_type_accuracy", {}) or {})
    lines.append(
        "- Key result: overall "
        f"{float(data.get('overall_accuracy', 0.0) or 0.0):.4f}, "
        f"y/n {float(aty.get('yes/no', 0.0) or 0.0):.4f}, "
        f"num {float(aty.get('number', 0.0) or 0.0):.4f}, "
        f"other {float(aty.get('other', 0.0) or 0.0):.4f}"
    )
if notes:
    lines.append(f"- Notes: {notes}")
with open(progress, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n\n")
PY
}

parse_eval_log() {
  local logfile="$1"
  local out_json="$2"
  runtime_exec_python - <<'PY' "${logfile}" "${out_json}"
import json, re, sys
from pathlib import Path

logfile = Path(sys.argv[1])
out_json = Path(sys.argv[2])
if not logfile.is_file():
    raise SystemExit(1)

periodic = []
final_eval = None
current_step = None
current_samples = None
overall = None

step_re = re.compile(r"step=([0-9]+)")
overall_re = re.compile(r"\[eval:val\] overall_accuracy=([0-9.]+)")
atype_re = re.compile(r"\[eval:val\] answer_type: yes/no=([0-9.]+) number=([0-9.]+) other=([0-9.]+)")
split_re = re.compile(r"\[mm\] eval split=val samples=([0-9]+)")

for raw in logfile.read_text(encoding="utf-8").splitlines():
    m = step_re.search(raw)
    if m and "step=" in raw and ("loss=" in raw or "answers/s" in raw or "steps/s" in raw):
        current_step = int(m.group(1))
    m = split_re.search(raw)
    if m:
        current_samples = int(m.group(1))
    m = overall_re.search(raw)
    if m:
        overall = float(m.group(1))
        continue
    m = atype_re.search(raw)
    if m and overall is not None:
        rec = {
            "step": int(current_step or 0),
            "overall_accuracy": float(overall),
            "answer_type_accuracy": {
                "yes/no": float(m.group(1)),
                "number": float(m.group(2)),
                "other": float(m.group(3)),
            },
            "record_count": int(current_samples or 0),
        }
        if "tag=final_eval" in raw or "final eval" in raw.lower():
            final_eval = rec
        else:
            periodic.append(rec)
        overall = None

peak = max(periodic, key=lambda r: float(r.get("overall_accuracy", 0.0))) if periodic else None
payload = {
    "source_log": str(logfile),
    "periodic": periodic,
    "peak_periodic": peak,
    "final_eval": final_eval,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
with out_json.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True)
print(out_json)
PY
}

json_get() {
  runtime_exec_python - <<'PY' "$1" "$2" "$3"
import json, sys
path, obj_key, leaf_key = sys.argv[1:]
data = json.loads(open(path, "r", encoding="utf-8").read())
obj = data.get(obj_key) or {}
val = obj.get(leaf_key) if isinstance(obj, dict) else None
if val is None:
    if obj_key == "peak_periodic" and isinstance(data.get(obj_key), dict):
        val = data[obj_key].get(leaf_key)
print("" if val is None else val)
PY
}

materialize_eval_json_from_log() {
  local logfile="$1"
  local out_json="$2"
  runtime_exec_python - <<'PY' "${logfile}" "${out_json}"
import json, re, sys
from pathlib import Path

logfile = Path(sys.argv[1])
out_json = Path(sys.argv[2])
if not logfile.is_file():
    raise SystemExit(1)

overall = None
current_samples = None
last = None
overall_re = re.compile(r"\[eval:val\] overall_accuracy=([0-9.]+)")
atype_re = re.compile(r"\[eval:val\] answer_type: yes/no=([0-9.]+) number=([0-9.]+) other=([0-9.]+)")
split_re = re.compile(r"\[mm\] eval split=val samples=([0-9]+)")
for raw in logfile.read_text(encoding="utf-8").splitlines():
    m = split_re.search(raw)
    if m:
        current_samples = int(m.group(1))
    m = overall_re.search(raw)
    if m:
        overall = float(m.group(1))
        continue
    m = atype_re.search(raw)
    if m and overall is not None:
        last = {
            "overall_accuracy": float(overall),
            "answer_type_accuracy": {
                "yes/no": float(m.group(1)),
                "number": float(m.group(2)),
                "other": float(m.group(3)),
            },
            "record_count": int(current_samples or 0),
            "source_log": str(logfile),
        }
        overall = None
if last is None:
    raise SystemExit(1)
out_json.parent.mkdir(parents=True, exist_ok=True)
with out_json.open("w", encoding="utf-8") as f:
    json.dump(last, f, indent=2, ensure_ascii=True)
print(out_json)
PY
}

full_eval_bridge_checkpoint() {
  local run_id="$1"
  local step="$2"
  local out_json="$3"
  local eval_bs="$4"
  if [[ -f "${out_json}" ]]; then
    return 0
  fi
  local stdout_log
  stdout_log="$(run_stdout_log "${run_id}_peak_full_eval")"
  bundle_mark_start "${run_id}_peak_eval" "step=${step}"
  ./runmm_v1.sh "${run_id}" "${step}" \
    --eval_only \
    --eval_batches 0 \
    --eval_batch_size "${eval_bs}" \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    --min_train_steps_per_s 0 2>&1 | tee -a "${stdout_log}"
  materialize_eval_json_from_log "logs/${run_id}/logfile_from_${step}.txt" "${out_json}"
  bundle_mark_end "${run_id}_peak_eval" "step=${step}"
}

full_eval_compression_checkpoint() {
  local ckpt="$1"
  local out_json="$2"
  local eval_bs="$3"
  if [[ -f "${out_json}" ]]; then
    return 0
  fi
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_format_alignment_eval \
    --checkpoint "${ckpt}" \
    --batch_size "${eval_bs}" \
    --eval_batches 0 \
    --disable_lm_visual_adapters \
    --output_json "${out_json}"
}

run_gqa_eval() {
  local ckpt="$1"
  local out_json="$2"
  if [[ -f "${out_json}" ]]; then
    return 0
  fi
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_gqa_eval \
    --checkpoint "${ckpt}" \
    --batch_size "${GQA_EVAL_BS}" \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    --limit_eval "${GQA_EVAL_LIMIT}" \
    --output_json "${out_json}"
}

run_probe() {
  local ckpt="$1"
  local out_json="$2"
  local batch_size="$3"
  if [[ -f "${out_json}" ]]; then
    return 0
  fi
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_semantic_probe \
    --checkpoint "${ckpt}" \
    --batch_size "${batch_size}" \
    --probe_batch_size "${PROBE_BATCH}" \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    --limit_train "${PROBE_TRAIN_LIMIT}" \
    --limit_val "${PROBE_VAL_LIMIT}" \
    --answer_top_k 3000 \
    --epochs 10 \
    --lr 0.001 \
    --feature_pool flatten \
    --output_json "${out_json}"
}

run_ocr_eval() {
  local ckpt="$1"
  local out_json="$2"
  local batch_size="$3"
  if [[ -f "${out_json}" ]]; then
    return 0
  fi
  runtime_exec_python -m tasks.mm_bridge.scripts.mm_ocr_subset_eval \
    --checkpoint "${ckpt}" \
    --batch_size "${batch_size}" \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    --limit_ocr "${OCR_LIMIT}" \
    --output_json "${out_json}"
}

write_report() {
  runtime_exec_python - <<'PY' "${REPORT_MD}" "${BUNDLE_DIR}"
import json, os, sys
from pathlib import Path

report_path = Path(sys.argv[1])
bundle_dir = Path(sys.argv[2])

def load(name: str):
    path = bundle_dir / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def fmt_metric(data, key="overall_accuracy"):
    if not isinstance(data, dict):
        return "n/a"
    return f"{float(data.get(key, 0.0) or 0.0):.4f}"

def fmt_types(data):
    if not isinstance(data, dict):
        return "n/a"
    aty = dict(data.get("answer_type_accuracy", {}) or {})
    return (
        f"y/n {float(aty.get('yes/no', 0.0) or 0.0):.4f} | "
        f"num {float(aty.get('number', 0.0) or 0.0):.4f} | "
        f"other {float(aty.get('other', 0.0) or 0.0):.4f}"
    )

siglip2_peak = load("siglip2_bridge_peak_full.json")
pecore_peak = load("pecore_bridge_peak_full.json")
winner_frozen = load("winner_frozen_bridge_peak_full.json")
winner_kd = load("winner_kd_bridge_peak_full.json")
winner_reasoningkd = load("winner_reasoningkd_bridge_peak_full.json")
winner_ftclean = load("winner_ftclean_bridge_peak_full.json")
winner_ft = load("winner_ft_bridge_peak_full.json")
winner_ftkd = load("winner_ftkd_bridge_peak_full.json")
winner_k8 = load("winner_frozen_k8_peak_full.json")
stacked_k8 = load("winner_stacked_k8_peak_full.json")
ft_stacked_k8 = load("winner_ft_stacked_k8_peak_full.json")
ftkd_k8 = load("winner_ftkd_k8_peak_full.json")

lines = ["# Tier 0 VM Bundle", ""]
lines.append("- plan: `tasks/mm_bridge/docs/78_tier0_vm_bundle_plan_2026-03-29.md`")
lines.append("")
lines.append("## Peak Full-Eval Summary")
for label, data in [
    ("SigLIP2 frozen bridge", siglip2_peak),
    ("PE-Core frozen bridge", pecore_peak),
    ("Winning frozen bridge", winner_frozen),
    ("Winning VM + KD bridge", winner_kd),
    ("Winning VM + reasoning-LM v2 + KD bridge", winner_reasoningkd),
    ("Winning VM + fresh-init KD + top-layer finetune bridge", winner_ftclean),
    ("Winning VM + top-layer finetune bridge", winner_ft),
    ("Winning VM + KD + top-layer finetune bridge", winner_ftkd),
    ("Winning frozen VM K=8", winner_k8),
    ("Winning stacked VM K=8", stacked_k8),
    ("Winning FT-stacked VM K=8", ft_stacked_k8),
    ("Winning FTKD-stacked VM K=8", ftkd_k8),
]:
    if data is None:
        continue
    lines.append(f"- {label}: {fmt_metric(data)} | {fmt_types(data)}")
lines.append("")
lines.append("## Artifact Index")
for path in sorted(bundle_dir.glob("*.json")):
    lines.append(f"- `{path.name}`")
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

run_download_phase() {
  local started_h started_ts end_ts duration stdout_log status=0
  started_h="$(timestamp_h)"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log run0_download_verify)"
  bundle_mark_start "${BUNDLE_ID}_download_verify" "siglip2 pe_core"
  runtime_exec_python -m tasks.mm_bridge.scripts.download_tier0_vm_backbones \
    --siglip2_dir "${SIGLIP2_DIR}" \
    --pe_core_dir "${PECORE_DIR}" \
    --device cpu \
    --output_json "${BUNDLE_DIR}/backbone_verify.json" 2>&1 | tee -a "${stdout_log}" || status=$?
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    bundle_mark_end "${BUNDLE_ID}_download_verify" "status=0"
    append_progress "Run 0: Backbone Download + Verify" "${started_h}" "COMPLETE" "${duration}" "0" "${BUNDLE_DIR}/backbone_verify.json" "Downloaded local SigLIP2 and PE-Core checkpoints and verified feature shapes."
  else
    bundle_mark_fail "${BUNDLE_ID}_download_verify" "status=${status}"
    append_progress "Run 0: Backbone Download + Verify" "${started_h}" "FAILED" "${duration}" "0" "none" "Backbone download or shape verification failed."
  fi
  return "${status}"
}

run_bridge_baseline() {
  local key="$1"
  local run_id="$2"
  local vm_name="$3"
  local vm_ckpt="$4"
  local train_bs="$5"
  local train_ga="$6"
  local eval_bs="$7"
  local workers="$8"
  local prefetch="$9"
  shift 9
  local extra_args=("$@")

  local started_h started_ts end_ts duration stdout_log status=0 trace_json final_step peak_json notes
  started_h="$(timestamp_h)"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log "${key}_bridge")"
  notes="Frozen-VM bridge baseline."
  local has_kd=0
  local has_ft=0
  for arg in "${extra_args[@]}"; do
    if [[ "${arg}" == "--answer_kd_labels_path" ]]; then
      has_kd=1
    elif [[ "${arg}" == "--train_vision_last_n_blocks" ]]; then
      has_ft=1
    fi
  done
  if (( has_kd == 1 && has_ft == 1 )); then
    notes="Top-layer VM finetuning stacked on the Qwen-KD frontier."
  elif (( has_kd == 1 )); then
    notes="Qwen answer-KD bridge on the winning frozen VM."
  elif (( has_ft == 1 )); then
    notes="Top-layer VM finetuning on the winning frozen VM."
  fi
  bundle_mark_start "${run_id}" "vm=${vm_name}"
  resume_runmm "${run_id}" 9000 \
    --vision_model "${vm_name}" \
    --vision_checkpoint "${vm_ckpt}" \
    --lm_checkpoint "${LM_CKPT}" \
    --seed "${SEED}" \
    --batch_size "${train_bs}" \
    --grad_accum_steps "${train_ga}" \
    --eval_batch_size "${eval_bs}" \
    --num_workers "${workers}" \
    --prefetch_factor "${prefetch}" \
    --no-pin_memory \
    --min_train_steps_per_s 0 \
    "${extra_args[@]}" 2>&1 | tee -a "${stdout_log}" || status=$?

  trace_json="${BUNDLE_DIR}/${key}_bridge_trace.json"
  parse_eval_log "logs/${run_id}/logfile.txt" "${trace_json}" || status=$?
  final_step="$(full_eval_step_for_run "${run_id}")"
  final_step="${final_step:-9000}"
  peak_json="${BUNDLE_DIR}/${key}_bridge_peak_full.json"
  if [[ "${final_step}" == "9000" ]]; then
    materialize_eval_json_from_log "logs/${run_id}/logfile.txt" "${peak_json}" || status=$?
  else
    full_eval_bridge_checkpoint "${run_id}" "${final_step}" "${peak_json}" "${eval_bs}" || status=$?
  fi

  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    bundle_mark_end "${run_id}" "status=0 final=${final_step}"
    append_progress "Run: ${key} bridge" "${started_h}" "COMPLETE" "${duration}" "${final_step}" "${peak_json}" "${notes}"
  else
    bundle_mark_fail "${run_id}" "status=${status} final=${final_step}"
    append_progress "Run: ${key} bridge" "${started_h}" "FAILED" "${duration}" "${final_step}" "${peak_json}" "${notes}"
  fi
  return "${status}"
}

run_k8_compression() {
  local key="$1"
  local run_id="$2"
  local init_ckpt="$3"
  local vm_name="$4"
  local vm_ckpt="$5"
  local started_h started_ts end_ts duration stdout_log status=0 trace_json final_step peak_json best_ckpt
  started_h="$(timestamp_h)"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log "${key}_compress")"
  if [[ ! -f "${init_ckpt}" ]]; then
    append_progress "Run: ${key} K=8" "${started_h}" "FAILED" "0h 0m" "0" "none" "Missing init checkpoint ${init_ckpt}"
    return 1
  fi
  bundle_mark_start "${run_id}" "compress_from=$(basename "${init_ckpt}") vm=${vm_name}"
  resume_runmm "${run_id}" 3000 \
    --vision_model "${vm_name}" \
    --vision_checkpoint "${vm_ckpt}" \
    --lm_checkpoint "${LM_CKPT}" \
    --seed "${SEED}" \
    --max_steps 3000 \
    --manual_max_steps \
    --batch_size "${COMPRESS_BS}" \
    --grad_accum_steps "${COMPRESS_GA}" \
    --eval_batch_size "${COMPRESS_EVAL_BS}" \
    --num_workers "${COMPRESS_NUM_WORKERS}" \
    --prefetch_factor "${COMPRESS_PREFETCH}" \
    --no-pin_memory \
    --log_every 20 \
    --eval_every 500 \
    --eval_batches 100 \
    --final_eval_batches 0 \
    --ckpt_every 500 \
    --lr 0.0002 \
    --lr_schedule cosine \
    --lr_warmup_steps 200 \
    --freeze_mode semantic_bottleneck_only \
    --semantic_bottleneck \
    --semantic_tokens "${COMPRESS_K}" \
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
    --init_from_mm_checkpoint "${init_ckpt}" \
    --min_train_steps_per_s 0 2>&1 | tee -a "${stdout_log}" || status=$?

  trace_json="${BUNDLE_DIR}/${key}_compress_trace.json"
  parse_eval_log "logs/${run_id}/logfile.txt" "${trace_json}" || status=$?
  final_step="$(full_eval_step_for_run "${run_id}")"
  final_step="${final_step:-3000}"
  peak_json="${BUNDLE_DIR}/${key}_peak_full.json"
  best_ckpt="logs/${run_id}/step_${final_step}.tar"
  if [[ "${final_step}" == "3000" ]]; then
    materialize_eval_json_from_log "logs/${run_id}/logfile.txt" "${peak_json}" || status=$?
  else
    full_eval_compression_checkpoint "${best_ckpt}" "${peak_json}" "${COMPRESS_EVAL_BS}" || status=$?
  fi
  run_probe "${best_ckpt}" "${BUNDLE_DIR}/${key}_probe.json" "${COMPRESS_EVAL_BS}" || status=$?

  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    bundle_mark_end "${run_id}" "status=0 final=${final_step}"
    append_progress "Run: ${key} K=8" "${started_h}" "COMPLETE" "${duration}" "${final_step}" "${peak_json}" "Standard format-alignment K=8 compression."
  else
    bundle_mark_fail "${run_id}" "status=${status} final=${final_step}"
    append_progress "Run: ${key} K=8" "${started_h}" "FAILED" "${duration}" "${final_step}" "${peak_json}" "Compression failed or partial."
  fi
  return "${status}"
}

run_hard_suite() {
  local label="$1"
  local ckpt="$2"
  local batch_size="$3"
  local prefix="$4"
  bundle_mark_start "${prefix}_hardevals" "label=${label}"
  run_gqa_eval "${ckpt}" "${BUNDLE_DIR}/${prefix}_gqa.json"
  run_ocr_eval "${ckpt}" "${BUNDLE_DIR}/${prefix}_ocr.json" "${batch_size}"
  run_probe "${ckpt}" "${BUNDLE_DIR}/${prefix}_probe.json" "${batch_size}"
  bundle_mark_end "${prefix}_hardevals" "label=${label}"
}

select_winner_by_peak() {
  runtime_exec_python - <<'PY' "$1" "$2"
import json, os, sys
paths = sys.argv[1:]
best_path = ""
best_acc = -1.0
for path in paths:
    if not os.path.isfile(path):
        continue
    data = json.loads(open(path, "r", encoding="utf-8").read())
    peak = data.get("peak_periodic") or {}
    acc = float(peak.get("overall_accuracy", -1.0) or -1.0)
    if acc > best_acc:
        best_acc = acc
        best_path = path
print(best_path)
PY
}

select_winner_by_full_eval() {
  runtime_exec_python - <<'PY' "$1" "$2"
import json, os, sys
paths = sys.argv[1:]
best_path = ""
best_acc = -1.0
for path in paths:
    if not os.path.isfile(path):
        continue
    data = json.loads(open(path, "r", encoding="utf-8").read())
    acc = float(data.get("overall_accuracy", -1.0) or -1.0)
    if acc > best_acc:
        best_acc = acc
        best_path = path
print(best_path)
PY
}

full_eval_step_for_run() {
  local run_id="$1"
  latest_step "logs/${run_id}"
}

full_eval_ckpt_for_run() {
  local run_id="$1"
  local step
  step="$(full_eval_step_for_run "${run_id}")"
  if [[ -n "${step}" && "${step}" != "0" ]]; then
    printf 'logs/%s/step_%s.tar\n' "${run_id}" "${step}"
  fi
}

main() {
  mkdir -p "${BUNDLE_DIR}"
  printf '# Tier 0 VM Bundle\n\n' > "${PROGRESS_MD}"
  cat > "${BUNDLE_DIR}/README.md" <<EOF
# Tier 0 VM Bundle

Bundle: ${BUNDLE_ID}
Start: $(timestamp_h)

Plan:
- tasks/mm_bridge/docs/78_tier0_vm_bundle_plan_2026-03-29.md

Fixed recipe:
- Cement-style bridge stack
- LM checkpoint: ${LM_CKPT}
- teacher soft labels: ${TEACHER_DATA_DIR}
- reference bridge checkpoint: ${REFERENCE_BRIDGE_CKPT}
EOF

  bundle_init "${BUNDLE_ID}" "${LATEST_LINK}"
  bundle_log_line "BUNDLE START ${BUNDLE_ID}"
  bundle_log_line "env $(runtime_log_env_summary "$(runtime_resolve_mode)")"
  bundle_log_line "host_ram $(host_ram_snapshot)"
  bundle_log_line "gpu $(gpu_snapshot)"
  bundle_log_line "lm_ckpt ${LM_CKPT}"
  bundle_log_line "teacher_data ${TEACHER_DATA_DIR}"
  bundle_log_line "siglip2_dir ${SIGLIP2_DIR}"
  bundle_log_line "pecore_dir ${PECORE_DIR}"

  clear_vram
  run_download_phase || true
  clear_vram

  run_bridge_baseline "siglip2" "${RUN_SIGLIP2}" "siglip2_b16" "${SIGLIP2_DIR}" \
    "${BRIDGE_BS}" "${BRIDGE_GA}" "${BRIDGE_EVAL_BS}" "${BRIDGE_NUM_WORKERS}" "${BRIDGE_PREFETCH}" || true
  clear_vram

  run_bridge_baseline "pecore" "${RUN_PECORE}" "pe_core_b16" "${PECORE_DIR}" \
    "${BRIDGE_BS}" "${BRIDGE_GA}" "${BRIDGE_EVAL_BS}" "${BRIDGE_NUM_WORKERS}" "${BRIDGE_PREFETCH}" || true
  clear_vram

  local frozen_winner_full frozen_winner_key frozen_winner_run frozen_winner_vm frozen_winner_dir frozen_winner_final_step frozen_winner_final_ckpt
  frozen_winner_full="$(select_winner_by_full_eval "${BUNDLE_DIR}/siglip2_bridge_peak_full.json" "${BUNDLE_DIR}/pecore_bridge_peak_full.json")"
  if [[ -z "${frozen_winner_full}" ]]; then
    bundle_log_line "No valid frozen-VM winner trace found; stopping bundle."
    write_report
    bundle_refresh_experiment_db
    return 1
  fi
  if [[ "${frozen_winner_full}" == *siglip2* ]]; then
    frozen_winner_key="siglip2"
    frozen_winner_run="${RUN_SIGLIP2}"
    frozen_winner_vm="siglip2_b16"
    frozen_winner_dir="${SIGLIP2_DIR}"
  else
    frozen_winner_key="pecore"
    frozen_winner_run="${RUN_PECORE}"
    frozen_winner_vm="pe_core_b16"
    frozen_winner_dir="${PECORE_DIR}"
  fi
  frozen_winner_final_step="$(full_eval_step_for_run "${frozen_winner_run}")"
  frozen_winner_final_step="${frozen_winner_final_step:-9000}"
  frozen_winner_final_ckpt="logs/${frozen_winner_run}/step_${frozen_winner_final_step}.tar"
  cp -f "${BUNDLE_DIR}/${frozen_winner_key}_bridge_peak_full.json" "${BUNDLE_DIR}/winner_frozen_bridge_peak_full.json" || true
  bundle_log_line "winner_frozen ${frozen_winner_key} step=${frozen_winner_final_step}"

  if [[ "${DO_COMPRESS_WINNING_FROZEN}" == "1" ]]; then
    clear_vram
    run_k8_compression "winner_frozen_k8" "${RUN_WINNER_K8}" "${frozen_winner_final_ckpt}" "${frozen_winner_vm}" "${frozen_winner_dir}" || true
    clear_vram
  fi

  run_bridge_baseline "winner_kd" "${RUN_WINNER_KD}" "${frozen_winner_vm}" "${frozen_winner_dir}" \
    "${BRIDGE_BS}" "${BRIDGE_GA}" "${BRIDGE_EVAL_BS}" "${BRIDGE_NUM_WORKERS}" "${BRIDGE_PREFETCH}" \
    --answer_kd_labels_path "${TEACHER_DATA_DIR}" \
    --answer_kd_weight 0.3 \
    --answer_kd_temp 4.0 || true
  clear_vram

  run_bridge_baseline "winner_ft" "${RUN_WINNER_FT}" "${frozen_winner_vm}" "${frozen_winner_dir}" \
    "${FINETUNE_BS}" "${FINETUNE_GA}" "${FINETUNE_EVAL_BS}" "${FINETUNE_NUM_WORKERS}" "${FINETUNE_PREFETCH}" \
    --train_vision_last_n_blocks "${FINETUNE_LAST_N_BLOCKS}" \
    --vision_lr_scale "${FINETUNE_VISION_LR_SCALE}" || true
  clear_vram

  local stacked_winner_full stacked_winner_run stacked_winner_final_step stacked_winner_final_ckpt
  stacked_winner_full="$(select_winner_by_full_eval "${BUNDLE_DIR}/winner_kd_bridge_peak_full.json" "${BUNDLE_DIR}/winner_ft_bridge_peak_full.json")"
  if [[ -z "${stacked_winner_full}" ]]; then
    bundle_log_line "No valid stacked winner trace found; skipping stacked compression."
    DO_COMPRESS_WINNING_STACKED=0
  fi
  if [[ -n "${stacked_winner_full}" ]]; then
    if [[ "${stacked_winner_full}" == *winner_kd* ]]; then
      stacked_winner_run="${RUN_WINNER_KD}"
    else
      stacked_winner_run="${RUN_WINNER_FT}"
    fi
    stacked_winner_final_step="$(full_eval_step_for_run "${stacked_winner_run}")"
    stacked_winner_final_step="${stacked_winner_final_step:-9000}"
    stacked_winner_final_ckpt="logs/${stacked_winner_run}/step_${stacked_winner_final_step}.tar"
    bundle_log_line "winner_stacked $(basename "${stacked_winner_run}") step=${stacked_winner_final_step}"
  fi

  if [[ "${DO_COMPRESS_WINNING_STACKED}" == "1" ]]; then
    run_k8_compression "winner_stacked_k8" "${RUN_STACKED_K8}" "${stacked_winner_final_ckpt}" "${frozen_winner_vm}" "${frozen_winner_dir}" || true
    clear_vram
  fi

  bundle_mark_start "${BUNDLE_ID}_hard_suite" "reference_and_winners"
  run_hard_suite "reference_bridge" "${REFERENCE_BRIDGE_CKPT}" 96 "reference_bridge"
  if [[ -f "${REFERENCE_COMPRESS_CKPT}" ]]; then
    run_hard_suite "reference_compressed" "${REFERENCE_COMPRESS_CKPT}" 96 "reference_compressed"
  fi
  run_hard_suite "winner_frozen_bridge" "${frozen_winner_final_ckpt}" 96 "winner_frozen_bridge"
  run_hard_suite "winner_kd_bridge" "$(full_eval_ckpt_for_run "${RUN_WINNER_KD}")" 96 "winner_kd_bridge"
  run_hard_suite "winner_ft_bridge" "$(full_eval_ckpt_for_run "${RUN_WINNER_FT}")" 96 "winner_ft_bridge"
  if [[ -f "${BUNDLE_DIR}/winner_frozen_k8_peak_full.json" ]]; then
    run_hard_suite "winner_frozen_k8" "$(full_eval_ckpt_for_run "${RUN_WINNER_K8}")" 96 "winner_frozen_k8"
  fi
  if [[ -f "${BUNDLE_DIR}/winner_stacked_k8_peak_full.json" ]]; then
    run_hard_suite "winner_stacked_k8" "$(full_eval_ckpt_for_run "${RUN_STACKED_K8}")" 96 "winner_stacked_k8"
  fi
  bundle_mark_end "${BUNDLE_ID}_hard_suite" "done"

  write_report
  bundle_log_line "BUNDLE COMPLETE ${BUNDLE_ID}"
  bundle_refresh_experiment_db
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
