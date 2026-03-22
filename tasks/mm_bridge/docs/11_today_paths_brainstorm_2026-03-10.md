# Today Path Brainstorm (2026-03-10)

## Current Frontier

- Best official val accuracy so far: `0.4544` (`perceiver_d3_pd03_main`).
- Top cluster is very tight: `0.4527` to `0.4544`.
- Existing spread is small enough that seed variance may be masking true ranking.

## Major Paths

## Path A: Ranking Stability / Seed Variance

Hypothesis:
- Current deltas (`~0.0002` to `0.0017`) may be within random-seed noise.

Runs:
- Re-run top perceiver and top hybrid with alternate seeds.
- Keep architecture and optimization fixed.

Decision rule:
- If variance is larger than architecture gap, prioritize robustness-first and avoid overfitting to single-seed winners.

## Path B: Interface Geometry Retune Around Winner

Hypothesis:
- Interface is still the dominant bottleneck; ratio target `4.0` may not be optimal for perceiver/hybrid.

Runs:
- Sweep `prefix_norm_target_ratio` and regularizer weights around current defaults.

Decision rule:
- Keep settings that improve `overall` without regressing `number`.

## Path C: Visual Feature Source Ablation (Untested Axis)

Hypothesis:
- `posterior_mu` may discard useful structure; `encoder` features may preserve information helpful for counting/attributes.

Runs:
- Hold perceiver/hybrid fixed, switch `--vision_feature_source posterior_mu -> encoder`.

Decision rule:
- If `encoder` helps or is neutral on overall while helping `number`, promote this axis immediately.

## Path D: Hybrid Gating Behavior

Hypothesis:
- Hybrid works because constant-prefix stability anchors LM conditioning; gate parameterization may still be suboptimal.

Runs:
- Compare `token` vs `scalar` alpha modes and alpha init values (e.g., `0.75`, `0.85`).

Decision rule:
- Prefer variants that preserve top-line while reducing seed sensitivity.

## Path E: LM Adaptation Depth

Hypothesis:
- `train_top_lm_layers=2` is good, but might not be optimal for perceiver/hybrid.

Runs:
- Compare top-1 / top-2 / top-3 LM layers with same bridge.

Decision rule:
- Take extra LM unfreezing only if gain exceeds expected seed noise.

## Recommended 5-Hour Queue (Concrete)

Priority logic:
- First lock signal quality (Path A), then test one high-upside new axis (Path C).
- Keep all runs Docker-only via `./runmm.sh`.

Assumed runtime:
- `max_steps=6000`, `eval_every=1000`, `eval_batches=160` should fit about `45-55 min` per run on current settings.
- Total for 5 runs: roughly `4.0-4.75 h`.

### Common arguments

```bash
--precision bf16 \
--batch_size 192 --grad_accum_steps 1 \
--num_workers 4 --prefetch_factor 2 \
--epochs 300 --max_steps 6000 \
--log_every 20 --eval_every 1000 --eval_batches 160 --eval_log_every 20 \
--ckpt_every 1000 --eval_scorer official --final_sanity_count 0 \
--cuda_empty_cache_after_eval \
--num_visual_tokens 49 --bridge_token_reduce all --bridge_add_2d_pos_emb --bridge_num_heads 8 \
--prefix_calibration --prefix_calib_layernorm --prefix_calib_bias --prefix_calib_gate_init 1.0 \
--prefix_norm_target_ratio 4.0 --prefix_norm_reg_weight 0.005 --prefix_batchvar_reg_weight 0.0002 \
--freeze_mode bridge_plus_top_lm --train_top_lm_layers 2 \
--lr 0.0002 --lr_schedule cosine --lr_warmup_steps 600 --lr_min_ratio 0.15
```

### Queue

1. `mmday_20260310_perc_d3_pd03_s11`
- `--seed 11 --bridge_type perceiver_resampler --bridge_query_depth 3 --bridge_pre_mixer_type none --prefix_dropout 0.03`

2. `mmday_20260310_perc_d3_pd03_s73`
- `--seed 73 --bridge_type perceiver_resampler --bridge_query_depth 3 --bridge_pre_mixer_type none --prefix_dropout 0.03`

3. `mmday_20260310_hybrid_tok075_perc_d3_s11`
- `--seed 11 --bridge_type hybrid_const_image --bridge_hybrid_image_bridge_type perceiver_resampler --bridge_hybrid_alpha_mode token --bridge_hybrid_alpha_init 0.75 --bridge_query_depth 3 --bridge_pre_mixer_type none --prefix_dropout 0.03`

4. `mmday_20260310_perc_d3_pd03_encoder_s11`
- `--seed 11 --vision_feature_source encoder --bridge_type perceiver_resampler --bridge_query_depth 3 --bridge_pre_mixer_type none --prefix_dropout 0.03`

5. `mmday_20260310_hybrid_tok075_perc_d3_encoder_s11`
- `--seed 11 --vision_feature_source encoder --bridge_type hybrid_const_image --bridge_hybrid_image_bridge_type perceiver_resampler --bridge_hybrid_alpha_mode token --bridge_hybrid_alpha_init 0.75 --bridge_query_depth 3 --bridge_pre_mixer_type none --prefix_dropout 0.03`

## Backup Queue (If time remains)

1. `mmday_20260310_hybrid_scalar085_perc_d3_s11`
- `--bridge_hybrid_alpha_mode scalar --bridge_hybrid_alpha_init 0.85`

2. `mmday_20260310_perc_d3_pd03_top3_s11`
- `--train_top_lm_layers 3`

3. `mmdiag_mmday_best_sofar`
- Run `./tasks/mm_bridge/scripts/run_mm_diag.sh` on the best new checkpoint to compare clean vs perturbation sensitivity and prefix geometry.
