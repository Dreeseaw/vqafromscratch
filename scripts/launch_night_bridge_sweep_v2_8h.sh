#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/mm_run_budget.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmnight_bridge_v2_8h_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmnight_bridge_v2_8h_latest"

BATCH_SIZE="${BATCH_SIZE:-192}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
DEFAULT_MAX_STEPS="$(mm_budget_steps_for_bs_ga "${BATCH_SIZE}" "${GRAD_ACCUM_STEPS}")"
HORIZON_HOURS="${HORIZON_HOURS:-8}"
MAX_STEPS_MAIN="${MAX_STEPS_MAIN:-${DEFAULT_MAX_STEPS}}"
MAX_STEPS_EXP="${MAX_STEPS_EXP:-${DEFAULT_MAX_STEPS}}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DRY_RUN="${DRY_RUN:-0}"

START_TS="$(date +%s)"
HORIZON_SEC="$(( HORIZON_HOURS * 3600 ))"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Night Bridge Sweep V2 (8h Horizon)

Sweep ID: ${SWEEP_ID}
Start time: $(date)
Horizon hours: ${HORIZON_HOURS}

Priority updates from latest findings:
- Perceiver-style bridge (depth 3) is current top performer.
- Vision on CPU is supported but significantly slower; keep vision on GPU.
- Throughput sweet spot: bf16, batch_size=192, workers=4.

Runtime knobs:
- MAX_STEPS_MAIN=${MAX_STEPS_MAIN}
- MAX_STEPS_EXP=${MAX_STEPS_EXP}
- EVAL_EVERY=${EVAL_EVERY}
- EVAL_BATCHES=${EVAL_BATCHES}
- BATCH_SIZE=${BATCH_SIZE}
- GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
- NUM_WORKERS=${NUM_WORKERS}
- PREFETCH_FACTOR=${PREFETCH_FACTOR}

Data roots remain unchanged:
- images_root: images
- annotations_root: data/vqav2
EOF

COMMON_ARGS=(
  --precision bf16
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --epochs 300
  --log_every "${LOG_EVERY}"
  --eval_every "${EVAL_EVERY}"
  --eval_batches "${EVAL_BATCHES}"
  --eval_log_every 20
  --ckpt_every "${CKPT_EVERY}"
  --eval_scorer official
  --final_sanity_count 0
  --cuda_empty_cache_after_eval
  --num_visual_tokens 49
  --bridge_token_reduce all
  --bridge_add_2d_pos_emb
  --bridge_num_heads 8
  --prefix_calibration
  --prefix_calib_layernorm
  --prefix_calib_bias
  --prefix_calib_gate_init 1.0
  --prefix_norm_target_ratio 4.0
  --prefix_norm_reg_weight 0.005
  --prefix_batchvar_reg_weight 0.0002
  --freeze_mode bridge_plus_top_lm
  --train_top_lm_layers 2
  --lr 0.0002
  --lr_schedule cosine
  --lr_warmup_steps 600
  --lr_min_ratio 0.15
)

within_horizon() {
  local now elapsed
  now="$(date +%s)"
  elapsed="$((now - START_TS))"
  [[ "${elapsed}" -lt "${HORIZON_SEC}" ]]
}

run_one() {
  local suffix="$1"
  shift
  local run_id="${SWEEP_ID}_${suffix}"
  if ! within_horizon; then
    echo "[$(date)] STOP horizon reached before ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 1
  fi
  echo "[$(date)] START ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
  echo "[$(date)] CMD ./runmm.sh ${run_id} ${COMMON_ARGS[*]} $*" >> "${SWEEP_DIR}/timeline.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date)] DRY_RUN skip ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  {
    ./runmm.sh "${run_id}" "${COMMON_ARGS[@]}" "$@"
  } >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1
  echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
  return 0
}

# A) Re-run current best family first.
run_one "perceiver_d3_pd03_main" \
  --max_steps "${MAX_STEPS_MAIN}" \
  --bridge_type perceiver_resampler \
  --bridge_query_depth 3 \
  --bridge_pre_mixer_type none \
  --prefix_dropout 0.03 || true

# B) Dropout ablation around the winner.
run_one "perceiver_d3_pd00_main" \
  --max_steps "${MAX_STEPS_MAIN}" \
  --bridge_type perceiver_resampler \
  --bridge_query_depth 3 \
  --bridge_pre_mixer_type none \
  --prefix_dropout 0.0 || true

# C) Slightly deeper perceiver.
run_one "perceiver_d4_pd03_main" \
  --max_steps "${MAX_STEPS_MAIN}" \
  --bridge_type perceiver_resampler \
  --bridge_query_depth 4 \
  --bridge_pre_mixer_type none \
  --prefix_dropout 0.03 || true

# D) Hybrid: stabilize perceiver image branch with learned constant component.
run_one "hybrid_tok060_perc_d3_main" \
  --max_steps "${MAX_STEPS_MAIN}" \
  --bridge_type hybrid_const_image \
  --bridge_hybrid_image_bridge_type perceiver_resampler \
  --bridge_hybrid_alpha_mode token \
  --bridge_hybrid_alpha_init 0.60 \
  --bridge_query_depth 3 \
  --bridge_pre_mixer_type none \
  --prefix_dropout 0.03 || true

# E) Hybrid alpha sensitivity.
run_one "hybrid_tok075_perc_d3_main" \
  --max_steps "${MAX_STEPS_MAIN}" \
  --bridge_type hybrid_const_image \
  --bridge_hybrid_image_bridge_type perceiver_resampler \
  --bridge_hybrid_alpha_mode token \
  --bridge_hybrid_alpha_init 0.75 \
  --bridge_query_depth 3 \
  --bridge_pre_mixer_type none \
  --prefix_dropout 0.03 || true

# F) Spatial mixer check on perceiver.
run_one "perceiver_d3_sa1_main" \
  --max_steps "${MAX_STEPS_MAIN}" \
  --bridge_type perceiver_resampler \
  --bridge_query_depth 3 \
  --bridge_pre_mixer_type self_attn \
  --bridge_pre_mixer_layers 1 \
  --prefix_dropout 0.03 || true

# G) Exploratory: qformer-lite depth 3.
run_one "qformer_d3_exp" \
  --max_steps "${MAX_STEPS_EXP}" \
  --bridge_type qformer_lite \
  --bridge_query_depth 3 \
  --bridge_pre_mixer_type none \
  --prefix_dropout 0.02 || true

# H) Exploratory: learned-query depth bump.
run_one "lq_ref2_sa1_exp" \
  --max_steps "${MAX_STEPS_EXP}" \
  --bridge_type learned_query \
  --bridge_refine_layers 2 \
  --bridge_pre_mixer_type self_attn \
  --bridge_pre_mixer_layers 1 \
  --prefix_dropout 0.0 || true

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
