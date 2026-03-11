#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/mm_run_budget.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmarch_memprobe_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmarch_memprobe_v1_latest"

MAX_STEPS="${MAX_STEPS:-60}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
RESULTS_PATH="${SWEEP_DIR}/probe_results.tsv"
DRY_RUN="${DRY_RUN:-0}"

cat > "${SWEEP_DIR}/README.md" <<EOF
# New Architecture Memory Probes V1

Purpose:
- determine safe batch_size / grad_accum combinations
- avoid periodic eval overhead during probing

Runtime:
- MAX_STEPS=${MAX_STEPS}
- NUM_WORKERS=${NUM_WORKERS}
- PREFETCH_FACTOR=${PREFETCH_FACTOR}

Probe policy:
- try a more aggressive BS/GA first
- if it fails, retry a safer fallback
- no periodic evals (\`eval_every=0\`)
- tiny final eval only (\`eval_batches=1\`, \`limit_eval=64\`)
EOF

COMMON_ARGS=(
  --precision bf16
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --epochs 20
  --max_steps "${MAX_STEPS}"
  --manual_max_steps
  --log_every 20
  --eval_every 0
  --eval_batches 1
  --eval_log_every 1
  --ckpt_every 0
  --limit_eval 64
  --eval_scorer official
  --final_sanity_count 0
  --cuda_empty_cache_after_eval
  --num_visual_tokens 49
  --bridge_token_reduce all
  --bridge_add_2d_pos_emb
  --bridge_num_heads 8
  --bridge_type perceiver_resampler
  --bridge_query_depth 3
  --bridge_pre_mixer_type none
  --prefix_calibration
  --prefix_calib_layernorm
  --prefix_calib_bias
  --prefix_calib_gate_init 1.0
  --prefix_norm_target_ratio 4.0
  --prefix_norm_reg_weight 0.005
  --prefix_batchvar_reg_weight 0.0002
  --prefix_dropout 0.03
  --freeze_mode bridge_plus_top_lm
  --train_top_lm_layers 2
  --lr 0.0002
  --lr_schedule cosine
  --lr_warmup_steps 200
  --lr_min_ratio 0.15
)

run_attempt() {
  local run_id="$1"
  local bs="$2"
  local ga="$3"
  shift 3
  local done_ckpt="logs/${run_id}/step_${MAX_STEPS}.tar"
  if [[ -f "${done_ckpt}" ]]; then
    echo -e "${run_id}\t${bs}\t${ga}\tskip_complete" | tee -a "${RESULTS_PATH}"
    return 0
  fi
  local cmd=(
    ./runmm.sh "${run_id}"
    "${COMMON_ARGS[@]}"
    --batch_size "${bs}"
    --grad_accum_steps "${ga}"
    "$@"
  )
  echo "[$(date)] START ${run_id} bs=${bs} ga=${ga}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ${cmd[*]}" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo -e "${run_id}\t${bs}\t${ga}\tdry_run" | tee -a "${RESULTS_PATH}"
    return 0
  fi
  if "${cmd[@]}" >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1; then
    echo -e "${run_id}\t${bs}\t${ga}\tsuccess" | tee -a "${RESULTS_PATH}"
    echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  echo -e "${run_id}\t${bs}\t${ga}\tfail" | tee -a "${RESULTS_PATH}"
  echo "[$(date)] FAIL  ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
  return 1
}

run_probe_with_fallback() {
  local stem="$1"
  local bs_a="$2"
  local ga_a="$3"
  local bs_b="$4"
  local ga_b="$5"
  shift 5
  if run_attempt "${stem}_b${bs_a}a${ga_a}" "${bs_a}" "${ga_a}" "$@"; then
    return 0
  fi
  run_attempt "${stem}_b${bs_b}a${ga_b}" "${bs_b}" "${ga_b}" "$@" || true
}

echo -e "run_id\tbatch_size\tgrad_accum\tstatus" > "${RESULTS_PATH}"

run_probe_with_fallback \
  "mmprobe_safeqcond" 192 1 128 2 \
  --bridge_question_conditioning \
  --bridge_question_context_mode prompt_only

run_probe_with_fallback \
  "mmprobe_multiscale" 128 2 64 3 \
  --bridge_type multiscale_perceiver \
  --vision_feature_source encoder_plus_posterior_mu \
  --bridge_token_reduce adaptive_pool

run_probe_with_fallback \
  "mmprobe_geomcal" 192 1 128 2 \
  --prefix_geom_mlp_ratio 0.5 \
  --prefix_geom_token_mixer_layers 1

run_probe_with_fallback \
  "mmprobe_structroles" 192 1 128 2 \
  --bridge_type structured_roles \
  --bridge_num_roles 4

run_probe_with_fallback \
  "mmprobe_evidencesparse" 192 1 128 2 \
  --bridge_type evidence_sparse \
  --bridge_evidence_topk 24

echo "[$(date)] PROBES COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
