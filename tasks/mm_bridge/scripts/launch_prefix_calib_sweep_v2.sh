#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/mm_run_budget.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmcal_sweep_v2_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"

BATCH_SIZE="${BATCH_SIZE:-256}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-$(mm_budget_steps_for_bs_ga "${BATCH_SIZE}" "${GRAD_ACCUM_STEPS}")}"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Prefix Calibration Sweep V2

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Data roots remain unchanged from runmm defaults:
- images_root: images
- annotations_root: data/vqav2

Runs are sequential via runmm + Docker.
EOF

COMMON_ARGS=(
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_steps "${MAX_STEPS}"
  --epochs 100
  --log_every 20
  --eval_every 1000
  --eval_batches 0
  --eval_log_every 20
  --ckpt_every 1000
  --eval_scorer official
  --final_sanity_count 0
  --num_visual_tokens 49
  --bridge_token_reduce all
  --bridge_type mlp
  --bridge_add_2d_pos_emb
  --prefix_calibration
  --prefix_calib_layernorm
  --prefix_calib_bias
  --prefix_calib_gate_init 1.0
  --freeze_mode bridge_plus_top_lm
)

run_one() {
  local run_id="$1"
  shift
  if [[ -f "logs/${run_id}/step_${MAX_STEPS}.tar" ]]; then
    echo "[$(date)] SKIP  ${run_id} (found logs/${run_id}/step_${MAX_STEPS}.tar)" | tee -a "${SWEEP_DIR}/timeline.log"
    return 0
  fi
  echo "[$(date)] START ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
  {
    echo "[$(date)] CMD ./runmm.sh ${run_id} ${COMMON_ARGS[*]} $*"
    ./runmm.sh "${run_id}" "${COMMON_ARGS[@]}" "$@"
  } >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1
  echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
}

# A) Strong prior baseline from v1, extended horizon.
run_one "mmcal2_top1_const_ext1" \
  --train_top_lm_layers 1 \
  --lr 0.0002 \
  --lr_schedule constant \
  --prefix_norm_target_ratio 4.0 \
  --prefix_norm_reg_weight 0.005 \
  --prefix_batchvar_reg_weight 0.0002 \
  --prefix_dropout 0.0

# B) Cosine decay + warmup around top1.
run_one "mmcal2_top1_cos_ext1" \
  --train_top_lm_layers 1 \
  --lr 0.00025 \
  --lr_schedule cosine \
  --lr_warmup_steps 400 \
  --lr_min_ratio 0.15 \
  --prefix_norm_target_ratio 4.0 \
  --prefix_norm_reg_weight 0.005 \
  --prefix_batchvar_reg_weight 0.0002 \
  --prefix_dropout 0.0

# C) Cosine + mild prefix dropout + softer regularization.
run_one "mmcal2_top1_cos_pd05_ext1" \
  --train_top_lm_layers 1 \
  --lr 0.00025 \
  --lr_schedule cosine \
  --lr_warmup_steps 500 \
  --lr_min_ratio 0.15 \
  --prefix_norm_target_ratio 3.8 \
  --prefix_norm_reg_weight 0.003 \
  --prefix_batchvar_reg_weight 0.00015 \
  --prefix_dropout 0.05

# D) Top2 LM layers with cosine.
run_one "mmcal2_top2_cos_ext1" \
  --train_top_lm_layers 2 \
  --lr 0.0002 \
  --lr_schedule cosine \
  --lr_warmup_steps 500 \
  --lr_min_ratio 0.15 \
  --prefix_norm_target_ratio 4.0 \
  --prefix_norm_reg_weight 0.005 \
  --prefix_batchvar_reg_weight 0.0002 \
  --prefix_dropout 0.0

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
