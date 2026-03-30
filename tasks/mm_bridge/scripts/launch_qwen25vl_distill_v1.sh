#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/runtime_exec.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ID="${BUNDLE_ID:-mmqwenkd_v1_${STAMP}}"
BUNDLE_DIR="logs/${BUNDLE_ID}"
TIMELINE="${BUNDLE_DIR}/timeline.log"
PROGRESS_MD="${BUNDLE_DIR}/progress.md"
REPORT_MD="${BUNDLE_DIR}/qwen_distill_report.md"
LATEST_LINK="logs/mmqwenkd_v1_latest"

TEACHER_DATA_DIR="${TEACHER_DATA_DIR:-data/distillation/qwen25vl3b_vqav2_train_v1}"
TEACHER_RUN_ID="${BUNDLE_ID}_teacher"
CONTROL_RUN_ID="${BUNDLE_ID}_control_bridge"
DISTILL_RUN_ID="${BUNDLE_ID}_distill_bridge"
COMPRESS_RUN_ID="${BUNDLE_ID}_distill_k8"

LM_CKPT="${LM_CKPT:-logs/lm_final/step_45000.tar}"
TOKENIZER_PATH="${TOKENIZER_PATH:-logs/mix_bpe_16k/tokenizer.pt}"
VISION_CKPT="${VISION_CKPT:-logs/hf_vision/google_siglip_base_patch16_224}"
REMAP_CKPT="${REMAP_CKPT:-logs/mmsemantic_remap_v1_debug/step_500.tar}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"

SEED="${SEED:-35}"
TRAIN_BS="${TRAIN_BS:-96}"
TRAIN_GA="${TRAIN_GA:-2}"
EVAL_BS="${EVAL_BS:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH="${PREFETCH:-1}"
TEACHER_BS="${TEACHER_BS:-4}"
TEACHER_SHARD="${TEACHER_SHARD:-4096}"
ANSWER_TOP_K="${ANSWER_TOP_K:-3000}"
KD_WEIGHT="${KD_WEIGHT:-0.3}"
KD_TEMP="${KD_TEMP:-4.0}"
COMPRESS_K="${COMPRESS_K:-8}"
TRACE_EVAL_BATCHES="${TRACE_EVAL_BATCHES:-100}"

mkdir -p "${BUNDLE_DIR}"
if [[ ! -f "${TIMELINE}" ]]; then
  : > "${TIMELINE}"
fi
if [[ ! -f "${PROGRESS_MD}" ]]; then
cat > "${PROGRESS_MD}" <<'EOF'
# Qwen Distillation Progress

EOF
fi
ln -sfn "${BUNDLE_ID}" "${LATEST_LINK}"

log_line() {
  local line="[$(date)] $*"
  echo "${line}" | tee -a "${TIMELINE}"
}

run_stdout_log() {
  local name="$1"
  echo "${BUNDLE_DIR}/${name}.stdout.log"
}

vram_snapshot() {
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
}

clear_vram() {
  log_line "VRAM before clear: $(vram_snapshot | tr '\n' ' ')"
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
  sleep 2
  log_line "VRAM after clear: $(vram_snapshot | tr '\n' ' ')"
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

teacher_complete() {
  runtime_exec_python - <<'PY' "${TEACHER_DATA_DIR}"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
meta = root / "meta.json"
progress = root / "progress.json"
if not meta.is_file() or not progress.is_file():
    print("0")
    raise SystemExit(0)
with meta.open("r", encoding="utf-8") as f:
    meta_data = json.load(f)
with progress.open("r", encoding="utf-8") as f:
    progress_data = json.load(f)
total = int(meta_data.get("total_items", 0) or 0)
next_index = int(progress_data.get("next_index", 0) or 0)
print("1" if total > 0 and next_index >= total else "0")
PY
}

append_progress_entry() {
  local name="$1"
  local status="$2"
  local started="$3"
  local duration="$4"
  local best_step="$5"
  local json_path="$6"
  local notes="$7"
  runtime_exec_python - <<'PY' "${PROGRESS_MD}" "${name}" "${status}" "${started}" "${duration}" "${best_step}" "${json_path}" "${notes}"
import json, sys
from pathlib import Path
md = Path(sys.argv[1])
name, status, started, duration, best_step, json_path, notes = sys.argv[2:]
line = f"## {name}\n- Started: {started}\n- Status: {status}\n- Duration: {duration}\n"
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
line += f"- Notes: {notes or '—'}\n\n"
with md.open("a", encoding="utf-8") as f:
    f.write(line)
PY
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

pick_best_step() {
  local glob_pat="$1"
  runtime_exec_python - <<'PY' "${BUNDLE_DIR}" "${glob_pat}"
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1])
glob_pat = sys.argv[2]
best_step = None
best_acc = -1.0
for path in sorted(bundle.glob(glob_pat)):
    parts = path.stem.split("_")
    step = None
    for i, part in enumerate(parts):
        if part == "step" and i + 1 < len(parts):
            try:
                step = int(parts[i + 1])
            except Exception:
                step = None
            break
    if step is None:
        continue
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    acc = float(data.get("overall_accuracy", 0.0) or 0.0)
    if acc > best_acc:
        best_acc = acc
        best_step = step
print(best_step or 3000)
PY
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

eval_trace_vqa() {
  local prefix="$1"
  local run_dir="$2"
  local batch_size="$3"
  shift 3
  local step ckpt out
  for step in 500 1000 1500 2000 2500 3000; do
    ckpt="${run_dir}/step_${step}.tar"
    out="${BUNDLE_DIR}/${prefix}_step_${step}_vqa.json"
    if [[ ! -f "${ckpt}" || -f "${out}" ]]; then
      continue
    fi
    eval_ckpt_vqa "${ckpt}" "${out}" "${batch_size}" --eval_batches "${TRACE_EVAL_BATCHES}" "$@"
  done
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
    --probe_batch_size 256 \
    --num_workers 1 \
    --prefetch_factor 1 \
    --no-pin_memory \
    --limit_train 10000 \
    --limit_val 5000 \
    --answer_top_k "${ANSWER_TOP_K}" \
    --epochs 10 \
    --lr 0.001 \
    --feature_pool flatten \
    --output_json "${out_json}"
}

write_report() {
  runtime_exec_python - <<'PY' "${REPORT_MD}" "${BUNDLE_DIR}" "${TEACHER_DATA_DIR}"
import json, sys
from pathlib import Path
report_path = Path(sys.argv[1])
bundle = Path(sys.argv[2])
teacher_dir = Path(sys.argv[3])

def load_json(name):
    path = bundle / name
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

teacher_meta = None
teacher_progress = None
if (teacher_dir / "meta.json").is_file():
    teacher_meta = json.loads((teacher_dir / "meta.json").read_text(encoding="utf-8"))
if (teacher_dir / "progress.json").is_file():
    teacher_progress = json.loads((teacher_dir / "progress.json").read_text(encoding="utf-8"))
control = load_json("control_bridge_full.json")
distill = load_json("distill_bridge_full.json")
compress = load_json("distill_compress_best_full.json")
probe = load_json("distill_compress_probe.json")

def fmt_eval(data):
    if not data:
        return "pending"
    at = data.get("answer_type_accuracy", {}) or {}
    return (
        f"{float(data.get('overall_accuracy', 0.0) or 0.0):.4f} overall"
        f" | y/n {float(at.get('yes/no', 0.0) or 0.0):.4f}"
        f" | num {float(at.get('number', 0.0) or 0.0):.4f}"
        f" | other {float(at.get('other', 0.0) or 0.0):.4f}"
    )

lines = ["# Qwen KD Distillation", ""]
if teacher_meta and teacher_progress:
    lines.append("## Teacher")
    lines.append(
        f"- samples: {int(teacher_progress.get('samples_completed', 0) or 0)} / {int(teacher_meta.get('total_items', 0) or 0)}"
    )
    lines.append(f"- answer vocab: {int(teacher_meta.get('answer_vocab_size', 0) or 0)}")
    lines.append(f"- shards: {len(list(teacher_meta.get('shards') or []))}")
    lines.append("")
lines.append("## Bridge")
lines.append(f"- control: {fmt_eval(control)}")
lines.append(f"- distilled: {fmt_eval(distill)}")
lines.append("")
lines.append("## Compression")
lines.append(f"- compressed: {fmt_eval(compress)}")
if probe:
    best = probe.get("best") or {}
    lines.append(f"- probe: {float(best.get('accuracy', 0.0) or 0.0):.4f}")
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

run_teacher_phase() {
  local started_h started_ts end_ts duration status=0 stdout_log
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log run1_teacher)"
  log_line "START ${TEACHER_RUN_ID}"
  mkdir -p "logs/${TEACHER_RUN_ID}"
  cp tasks/mm_bridge/scripts/build_qwen25vl_vqa_teacher.py "logs/${TEACHER_RUN_ID}/code_teacher.py"
  runtime_exec_python -m tasks.mm_bridge.scripts.build_qwen25vl_vqa_teacher \
    --run_id "${TEACHER_RUN_ID}" \
    --output_dir "${TEACHER_DATA_DIR}" \
    --model_name "${MODEL_NAME}" \
    --images_root images \
    --annotations_root data/vqav2 \
    --student_tokenizer_path "${TOKENIZER_PATH}" \
    --answer_top_k "${ANSWER_TOP_K}" \
    --batch_size "${TEACHER_BS}" \
    --shard_size "${TEACHER_SHARD}" \
    --seed "${SEED}" 2>&1 | tee -a "${stdout_log}" || status=$?
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if [[ "$(teacher_complete)" == "1" ]]; then
    status=0
  fi
  if (( status == 0 )); then
    log_line "END ${TEACHER_RUN_ID} status=0"
    append_progress_entry "Run 1: Teacher inference" "COMPLETE" "${started_h}" "${duration}" "" "" "Qwen2.5-VL-3B teacher extraction"
  else
    log_line "END ${TEACHER_RUN_ID} status=${status}"
    append_progress_entry "Run 1: Teacher inference" "FAILED" "${started_h}" "${duration}" "" "" "Teacher extraction failed or incomplete"
  fi
  return "${status}"
}

run_control_bridge_phase() {
  local started_h started_ts end_ts duration status=0 stdout_log
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log run2_control_bridge)"
  log_line "START ${CONTROL_RUN_ID}"
  resume_runmm "${CONTROL_RUN_ID}" 9000 \
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
    --min_train_steps_per_s 0 2>&1 | tee -a "${stdout_log}" || status=$?
  if [[ -f "logs/${CONTROL_RUN_ID}/step_9000.tar" ]]; then
    eval_ckpt_vqa "logs/${CONTROL_RUN_ID}/step_9000.tar" "${BUNDLE_DIR}/control_bridge_full.json" "${EVAL_BS}" --eval_batches 0 || status=$?
  fi
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END ${CONTROL_RUN_ID} status=0"
    append_progress_entry "Run 2: Fresh control bridge" "COMPLETE" "${started_h}" "${duration}" "9000" "${BUNDLE_DIR}/control_bridge_full.json" "Fresh Cement-style control, no KD"
  else
    log_line "END ${CONTROL_RUN_ID} status=${status}"
    append_progress_entry "Run 2: Fresh control bridge" "FAILED" "${started_h}" "${duration}" "$(latest_step "logs/${CONTROL_RUN_ID}")" "${BUNDLE_DIR}/control_bridge_full.json" "Control bridge failed or partial"
  fi
  return "${status}"
}

run_distill_bridge_phase() {
  local started_h started_ts end_ts duration status=0 stdout_log
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log run3_distill_bridge)"
  log_line "START ${DISTILL_RUN_ID}"
  resume_runmm "${DISTILL_RUN_ID}" 9000 \
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
    --min_train_steps_per_s 0 2>&1 | tee -a "${stdout_log}" || status=$?
  if [[ -f "logs/${DISTILL_RUN_ID}/step_9000.tar" ]]; then
    eval_ckpt_vqa "logs/${DISTILL_RUN_ID}/step_9000.tar" "${BUNDLE_DIR}/distill_bridge_full.json" "${EVAL_BS}" --eval_batches 0 || status=$?
  fi
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END ${DISTILL_RUN_ID} status=0"
    append_progress_entry "Run 3: Distilled bridge" "COMPLETE" "${started_h}" "${duration}" "9000" "${BUNDLE_DIR}/distill_bridge_full.json" "Fresh bridge with answer-vocab KL distillation"
  else
    log_line "END ${DISTILL_RUN_ID} status=${status}"
    append_progress_entry "Run 3: Distilled bridge" "FAILED" "${started_h}" "${duration}" "$(latest_step "logs/${DISTILL_RUN_ID}")" "${BUNDLE_DIR}/distill_bridge_full.json" "Distilled bridge failed or partial"
  fi
  return "${status}"
}

run_compression_phase() {
  local started_h started_ts end_ts duration status=0 stdout_log best_step best_ckpt
  started_h="$(date '+%Y-%m-%d %H:%M:%S')"
  started_ts="$(date +%s)"
  stdout_log="$(run_stdout_log run4_distill_compress)"
  log_line "START ${COMPRESS_RUN_ID}"
  if [[ ! -f "logs/${DISTILL_RUN_ID}/step_9000.tar" ]]; then
    log_line "END ${COMPRESS_RUN_ID} status=missing_distilled_bridge"
    append_progress_entry "Run 4: Distilled compression" "FAILED" "${started_h}" "0h 0m" "" "" "Missing distilled bridge step_9000 checkpoint"
    return 1
  fi
  resume_runmm "${COMPRESS_RUN_ID}" 3000 \
    --vision_model siglip_base \
    --vision_checkpoint "${VISION_CKPT}" \
    --lm_checkpoint "${LM_CKPT}" \
    --seed "${SEED}" \
    --max_steps 3000 \
    --manual_max_steps \
    --batch_size "${TRAIN_BS}" \
    --grad_accum_steps "${TRAIN_GA}" \
    --eval_batch_size "${EVAL_BS}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH}" \
    --no-pin_memory \
    --log_every 20 \
    --eval_every 0 \
    --ckpt_every 500 \
    --final_eval_batches 0 \
    --lr 0.0002 \
    --lr_schedule cosine \
    --lr_warmup_steps 200 \
    --freeze_mode semantic_bottleneck_only \
    --bridge_question_context_mode question_only \
    --bridge_query_bank_mode question_hidden_attn \
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
    --init_from_mm_checkpoint "logs/${DISTILL_RUN_ID}/step_9000.tar" \
    --min_train_steps_per_s 0 2>&1 | tee -a "${stdout_log}" || status=$?
  eval_trace_vqa "distill_compress" "logs/${COMPRESS_RUN_ID}" "${EVAL_BS}" --disable_lm_visual_adapters || status=$?
  best_step="$(pick_best_step "distill_compress_step_*_vqa.json")"
  best_ckpt="logs/${COMPRESS_RUN_ID}/step_${best_step}.tar"
  if [[ -f "${best_ckpt}" ]]; then
    eval_ckpt_vqa "${best_ckpt}" "${BUNDLE_DIR}/distill_compress_best_full.json" "${EVAL_BS}" --eval_batches 0 --disable_lm_visual_adapters || status=$?
    run_probe "${best_ckpt}" "${BUNDLE_DIR}/distill_compress_probe.json" "${TRAIN_BS}" || status=$?
  fi
  end_ts="$(date +%s)"
  duration="$(format_duration "${started_ts}" "${end_ts}")"
  if (( status == 0 )); then
    log_line "END ${COMPRESS_RUN_ID} status=0"
    append_progress_entry "Run 4: Distilled compression" "COMPLETE" "${started_h}" "${duration}" "${best_step}" "${BUNDLE_DIR}/distill_compress_best_full.json" "Format-aligned K=8 on top of the distilled bridge"
  else
    log_line "END ${COMPRESS_RUN_ID} status=${status}"
    append_progress_entry "Run 4: Distilled compression" "FAILED" "${started_h}" "${duration}" "${best_step}" "${BUNDLE_DIR}/distill_compress_best_full.json" "Compression failed or partial"
  fi
  return "${status}"
}

main() {
  log_line "BUNDLE ${BUNDLE_ID} START"
  log_line "Teacher data dir: ${TEACHER_DATA_DIR}"
  log_line "exec_env: $(runtime_log_env_summary "$(runtime_resolve_mode)")"
  clear_vram
  run_teacher_phase || true
  clear_vram
  run_control_bridge_phase || true
  clear_vram
  if [[ "$(teacher_complete)" == "1" ]]; then
    run_distill_bridge_phase || true
    clear_vram
    run_compression_phase || true
    clear_vram
  else
    log_line "Teacher extraction incomplete; distilled phases deferred"
  fi
  write_report
  log_line "BUNDLE ${BUNDLE_ID} COMPLETE"
}

main "$@"
