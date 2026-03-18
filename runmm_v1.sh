#!/bin/bash
set -e

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id> [checkpoint_step] [extra mm args...]"
  echo
  echo "Example:"
  echo "  $0 mm_dinovit_v2_cement_qonly_s42"
  exit 1
fi
shift || true

CKPT_STEP=""
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  CKPT_STEP="$1"
  shift || true
fi
EXTRA_ARGS=("$@")
LAUNCH_LOG="logs/$RUN_ID/launcher.log"

mkdir -pv "logs/$RUN_ID"

echo "[$(date)] START run_id=${RUN_ID} checkpoint=${CKPT_STEP:-none}" >> "${LAUNCH_LOG}"

BASE_ARGS=(
  --precision bf16
  --num_workers 4
  --prefetch_factor 2
  --epochs 400
  --max_steps 9000
  --manual_max_steps
  --log_every 20
  --eval_every 1000
  --eval_batches 100
  --final_eval_batches 0
  --eval_log_every 20
  --eval_fraction 1.0
  --ckpt_every 1000
  --eval_scorer official
  --final_sanity_count 0
  --cuda_empty_cache_after_eval
  --eval_use_kv_cache
  --eval_kv_cache_mode batched
  --vision_model siglip_base
  --vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224
  --vision_feature_source encoder
  --vision_feature_mode auto
  --batch_size 96
  --grad_accum_steps 2
  --eval_batch_size 96
  --num_visual_tokens 49
  --bridge_type perceiver_resampler
  --bridge_query_depth 3
  --bridge_num_heads 8
  --bridge_token_reduce adaptive_pool
  --bridge_add_2d_pos_emb
  --bridge_pre_mixer_type none
  --bridge_question_conditioning
  --bridge_query_bank_mode question_hidden_attn
  --bridge_question_context_mode question_only
  --bridge_qquery_scale 1.0
  --bridge_qcond_scale 0.5
  --bridge_token_selector_type none
  --bridge_token_select_k 0
  --prefix_calibration
  --prefix_calib_layernorm
  --prefix_calib_bias
  --prefix_calib_gate_init 1.0
  --prefix_geom_mlp_ratio 0.5
  --prefix_geom_token_mixer_layers 1
  --prefix_norm_target_ratio 4.0
  --prefix_norm_reg_weight 0.005
  --prefix_batchvar_reg_weight 0.0002
  --prefix_dropout 0.03
  --freeze_mode bridge_plus_top_lm
  --train_top_lm_layers 2
  --lm_visual_adapter_type cross_attn
  --lm_visual_adapter_layers 3
  --lm_visual_adapter_num_heads 8
  --lm_visual_adapter_dropout 0.0
  --lm_visual_adapter_gate_init 0.5
  --lr 0.0002
  --lr_schedule cosine
  --lr_warmup_steps 600
  --lr_min_ratio 0.15
  --seed 35
)

echo "[$(date)] CMD ./runmm.sh ${RUN_ID} ${CKPT_STEP:+$CKPT_STEP }${BASE_ARGS[*]} ${EXTRA_ARGS[*]}" >> "${LAUNCH_LOG}"

status=0
if [[ -n "${CKPT_STEP}" ]]; then
  ./runmm.sh "${RUN_ID}" "${CKPT_STEP}" "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}" || status=$?
else
  ./runmm.sh "${RUN_ID}" "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}" || status=$?
fi

echo "[$(date)] EXIT status=${status}" >> "${LAUNCH_LOG}"
exit "${status}"
