#!/bin/bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmnight_bridge_v1_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"
ln -sfn "${SWEEP_ID}" "logs/mmnight_bridge_v1_latest"

MAX_STEPS="${MAX_STEPS:-3000}"
EVAL_EVERY="${EVAL_EVERY:-750}"
EVAL_BATCHES="${EVAL_BATCHES:-80}"
LOG_EVERY="${LOG_EVERY:-20}"
CKPT_EVERY="${CKPT_EVERY:-750}"
BATCH_SIZE="${BATCH_SIZE:-192}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DRY_RUN="${DRY_RUN:-0}"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Night Bridge Sweep V1

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Purpose:
- Explore next-generation bridge architectures overnight.
- Keep train/val data roots unchanged from runmm defaults:
  - images_root: images
  - annotations_root: data/vqav2

Runtime knobs:
- MAX_STEPS=${MAX_STEPS}
- EVAL_EVERY=${EVAL_EVERY}
- EVAL_BATCHES=${EVAL_BATCHES}
- BATCH_SIZE=${BATCH_SIZE}
- GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
- NUM_WORKERS=${NUM_WORKERS}
- PREFETCH_FACTOR=${PREFETCH_FACTOR}

This sweep is sequential and Docker-only via runmm.sh.
EOF

COMMON_ARGS=(
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --precision bf16
  --max_steps "${MAX_STEPS}"
  --epochs 200
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
  --prefix_dropout 0.0
  --freeze_mode bridge_plus_top_lm
  --train_top_lm_layers 2
  --lr 0.0002
  --lr_schedule cosine
  --lr_warmup_steps 500
  --lr_min_ratio 0.15
)

run_one() {
  local suffix="$1"
  shift
  local run_id="${SWEEP_ID}_${suffix}"
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
}

# 1) Main next-gen candidate.
run_one "lq_base" \
  --bridge_type learned_query \
  --bridge_refine_layers 1 \
  --bridge_pre_mixer_type none

# 2) Spatial mixer before learned-query reduction.
run_one "lq_spmix_sa1" \
  --bridge_type learned_query \
  --bridge_refine_layers 1 \
  --bridge_pre_mixer_type self_attn \
  --bridge_pre_mixer_layers 1

# 3) Hybrid constant + image prefix.
run_one "hybrid_tok065_lqimg" \
  --bridge_type hybrid_const_image \
  --bridge_hybrid_image_bridge_type learned_query \
  --bridge_hybrid_alpha_mode token \
  --bridge_hybrid_alpha_init 0.65 \
  --bridge_refine_layers 1 \
  --bridge_pre_mixer_type none

# 4) Spatial conv mixer before learned-query reduction.
run_one "lq_spmix_conv1d1" \
  --bridge_type learned_query \
  --bridge_refine_layers 1 \
  --bridge_pre_mixer_type conv1d \
  --bridge_pre_mixer_layers 1

# 5) Perceiver-style resampler (2 rounds).
run_one "perceiver_d2_sa1" \
  --bridge_type perceiver_resampler \
  --bridge_query_depth 2 \
  --bridge_pre_mixer_type none

# 6) Q-former-lite (2 blocks).
run_one "qformer_d2_sa1" \
  --bridge_type qformer_lite \
  --bridge_query_depth 2 \
  --bridge_pre_mixer_type none

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
