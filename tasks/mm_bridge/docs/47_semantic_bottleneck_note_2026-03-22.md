# Semantic Bottleneck Note

## What changed

An optional late semantic bottleneck now sits **after** the perceiver evidence retrieval path and **before** the LM-facing visual prefix export.

Current flow when enabled:

```text
visual tokens
  -> perceiver evidence latents [B, K, D]
  -> semantic bottleneck queries [B, M, D]
  -> semantic latent projection [B, M, Z]
  -> decode back to LM width [B, M, D]
  -> prefix calibrator
  -> LM
```

When disabled, the baseline path is unchanged:

```text
visual tokens
  -> perceiver evidence latents [B, K, D]
  -> prefix calibrator
  -> LM
```

## Why post-perceiver

This is intentionally **not** early dynbudget-style pruning.

The perceiver still sees the full dense visual evidence and performs retrieval first. The bottleneck only compresses the already-retrieved evidence latents that would otherwise be exported directly to the LM. That preserves the current winning modeling bias:

- dense visual evidence goes into retrieval
- semantic compression happens only at the LM interface

## Losses added

Two optional auxiliary losses were added on the semantic export path:

- `semantic_recon_loss`
  - MSE between decoded semantic tokens and an **adaptive-pooled target** derived from the pre-bottleneck evidence latents
- `semantic_consistency_loss`
  - cosine consistency against that same pooled target

The target choice is deliberately simple:

- target = adaptive average pooling of the pre-bottleneck evidence latents from `K -> M`
- the pooled target is detached

So the first implementation is a deterministic late compression head supervised against a pooled summary of the dense evidence it is replacing.

## Config flags

New flags:

- `--semantic_bottleneck`
- `--semantic_tokens`
- `--semantic_latent_dim`
- `--semantic_recon_loss_weight`
- `--semantic_consistency_loss_weight`
- `--semantic_token_schedule`

Notes:

- `M = semantic_tokens` is clamped to `<= K`
- `Z = semantic_latent_dim` is clamped to `<= D`
- this is currently wired for `perceiver_resampler` and `multiscale_perceiver`

## Example command

Example on the current Cement-style winning scaffold:

```bash
./runmm.sh mm_semantic_bottleneck_siglip_poc \
  --vision_model siglip_base \
  --vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224 \
  --bridge_type perceiver_resampler \
  --num_visual_tokens 49 \
  --bridge_query_depth 3 \
  --bridge_add_2d_pos_emb \
  --bridge_question_conditioning \
  --bridge_qcond_scale 0.5 \
  --bridge_query_bank_mode question_hidden_attn \
  --bridge_question_context_mode question_only \
  --bridge_token_selector_type none \
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
  --lm_visual_adapter_type cross_attn \
  --lm_visual_adapter_layers 3 \
  --lm_visual_adapter_num_heads 8 \
  --lm_visual_adapter_gate_init 0.5 \
  --semantic_bottleneck \
  --semantic_tokens 16 \
  --semantic_latent_dim 256 \
  --semantic_recon_loss_weight 0.1 \
  --semantic_consistency_loss_weight 0.1
```

## Files changed

- `models/bridge.py`
- `train/mm.py`
