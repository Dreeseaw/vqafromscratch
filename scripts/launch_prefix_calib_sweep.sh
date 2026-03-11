#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/mm_run_budget.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="mmcal_sweep_${STAMP}"
SWEEP_DIR="logs/${SWEEP_ID}"
mkdir -pv "${SWEEP_DIR}"

BATCH_SIZE="${BATCH_SIZE:-256}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-$(mm_budget_steps_for_bs_ga "${BATCH_SIZE}" "${GRAD_ACCUM_STEPS}")}"

cat > "${SWEEP_DIR}/README.md" <<EOF
# Prefix Calibration Sweep

Sweep ID: ${SWEEP_ID}
Start time: $(date)

Runs are executed sequentially via \`runmm.sh\` with Docker.
Each run writes canonical training logs to \`logs/<run_id>/logfile.txt\`.
This sweep also captures stdout to \`logs/${SWEEP_ID}/<run_id>.stdout.log\`.
EOF

COMMON_ARGS=(
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_steps "${MAX_STEPS}"
  --epochs 50
  --log_every 20
  --eval_every 500
  --eval_batches 0
  --eval_log_every 20
  --ckpt_every 1000
  --eval_scorer official
  --final_sanity_count 0
  --num_visual_tokens 49
  --bridge_token_reduce all
)

run_one() {
  local run_id="$1"
  shift
  echo "[$(date)] START ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
  {
    echo "[$(date)] CMD ./runmm.sh ${run_id} ${COMMON_ARGS[*]} $*"
    ./runmm.sh "${run_id}" "${COMMON_ARGS[@]}" "$@"
  } >> "${SWEEP_DIR}/${run_id}.stdout.log" 2>&1
  echo "[$(date)] END   ${run_id}" | tee -a "${SWEEP_DIR}/timeline.log"
}

# 1) New fix: calibrated MLP bridge, bridge-only training.
run_one "mmcal_mlp49_calib_bonly_v1" \
  --bridge_type mlp \
  --bridge_add_2d_pos_emb \
  --freeze_mode bridge_only \
  --lr 0.001 \
  --prefix_calibration \
  --prefix_calib_layernorm \
  --prefix_calib_bias \
  --prefix_calib_gate_init 1.0 \
  --prefix_norm_target_ratio 4.0 \
  --prefix_norm_reg_weight 0.01 \
  --prefix_batchvar_reg_weight 0.0005

# 2) Calibrated MLP + unfreeze top LM layer.
run_one "mmcal_mlp49_calib_top1_v1" \
  --bridge_type mlp \
  --bridge_add_2d_pos_emb \
  --freeze_mode bridge_plus_top_lm \
  --train_top_lm_layers 1 \
  --lr 0.0002 \
  --prefix_calibration \
  --prefix_calib_layernorm \
  --prefix_calib_bias \
  --prefix_calib_gate_init 1.0 \
  --prefix_norm_target_ratio 4.0 \
  --prefix_norm_reg_weight 0.005 \
  --prefix_batchvar_reg_weight 0.0002

# 3) Calibrated MLP + unfreeze top 2 LM layers.
run_one "mmcal_mlp49_calib_top2_v1" \
  --bridge_type mlp \
  --bridge_add_2d_pos_emb \
  --freeze_mode bridge_plus_top_lm \
  --train_top_lm_layers 2 \
  --lr 0.00015 \
  --prefix_calibration \
  --prefix_calib_layernorm \
  --prefix_calib_bias \
  --prefix_calib_gate_init 1.0 \
  --prefix_norm_target_ratio 4.0 \
  --prefix_norm_reg_weight 0.005 \
  --prefix_batchvar_reg_weight 0.0002

# 4) Learned-token + top-LM run as an optimistic control.
run_one "mmcal_lt49_top1_v1" \
  --bridge_type learned_tokens \
  --freeze_mode bridge_plus_top_lm \
  --train_top_lm_layers 1 \
  --lr 0.00015

echo "[$(date)] SWEEP COMPLETE ${SWEEP_ID}" | tee -a "${SWEEP_DIR}/timeline.log"
