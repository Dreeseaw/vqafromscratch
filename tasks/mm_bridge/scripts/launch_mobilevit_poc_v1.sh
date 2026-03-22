#!/bin/bash
set -euo pipefail

RUN_ID="${1:-mm_mobilevit_poc_v1}"
MODEL_DIR="${MODEL_DIR:-logs/hf_vision/apple_mobilevit_small}"

./runmm.sh "${RUN_ID}" \
  --precision bf16 \
  --vision_model mobilevit_hf \
  --vision_checkpoint "${MODEL_DIR}" \
  --vision_feature_source encoder \
  --vision_feature_mode auto \
  --bridge_type perceiver_resampler \
  --bridge_token_reduce adaptive_pool \
  --num_visual_tokens 49 \
  --bridge_add_2d_pos_emb \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.5 \
  --bridge_query_bank_mode question_mix \
  --bridge_qquery_basis_count 4 \
  --bridge_qquery_scale 1.0 \
  --prefix_calibration \
  --prefix_calib_layernorm \
  --prefix_calib_bias \
  --prefix_calib_gate_init 1.0 \
  --prefix_geom_mlp_ratio 0.5 \
  --prefix_geom_token_mixer_layers 1 \
  --prefix_norm_target_ratio 4 \
  --prefix_norm_reg_weight 0.005 \
  --prefix_batchvar_reg_weight 0.0002 \
  --prefix_dropout 0.03 \
  --freeze_mode bridge_plus_top_lm \
  --train_top_lm_layers 2 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 0 \
  --max_steps 1 \
  --manual_max_steps \
  --limit_train 1 \
  --eval_every 0 \
  --eval_batches 0 \
  --final_eval_batches 0 \
  --debug_shapes
