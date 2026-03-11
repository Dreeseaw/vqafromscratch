# Final Architecture Run Queue V1 - 2026-03-10

## Purpose

Concrete queue for the next architecture cycle under the new permanent policy:

- fixed training sample budget across runs
- no periodic evals
- one final eval only
- final eval uses half the eval split by default
- restart-safe skip/resume behavior

This queue is prepared but not started.

## Global Policy

- `eval_every=0`
- `eval_batches=0`
- `eval_fraction=0.5`
- `ckpt_every=1000`
- `precision=bf16`
- `freeze_mode=bridge_plus_top_lm`
- `train_top_lm_layers=2`
- `lr=0.0002`
- `lr_schedule=cosine`
- `lr_warmup_steps=600`
- `lr_min_ratio=0.15`
- `prefix_calibration=on`
- `prefix_dropout=0.03`

## Ordered Queue

1. `safe qcond perceiver`
- run id suffix: `safeqcond_d3_main`
- bridge: `perceiver_resampler`
- feature source: `posterior_mu`
- batching: `BS=192`, `GA=1`, effective `192`
- target steps: `6000`
- key args:
  - `--bridge_question_conditioning`
  - `--bridge_question_context_mode prompt_only`

2. `multiscale perceiver`
- run id suffix: `multiscale_d3_main`
- bridge: `multiscale_perceiver`
- feature source: `encoder_plus_posterior_mu`
- batching: `BS=128`, `GA=2`, effective `256`
- target steps: `4500`
- key args:
  - `--bridge_token_reduce adaptive_pool`

3. `early-layer perceiver`
- run id suffix: `earlylayer_encoder_d3_main`
- bridge: `perceiver_resampler`
- feature source: `encoder`
- batching: `BS=192`, `GA=1`, effective `192`
- target steps: `6000`
- key args:
  - `--bridge_token_reduce adaptive_pool`

4. `oracle196 + adaptive selection`
- run id suffix: `oracle196_topk64_main`
- bridge: `perceiver_resampler`
- feature source: `posterior_mu`
- batching: `BS=64`, `GA=3`, effective `192`
- target steps: `6000`
- key args:
  - `--num_visual_tokens 196`
  - `--bridge_token_reduce adaptive_pool`
  - `--bridge_token_selector_type topk`
  - `--bridge_token_select_k 64`

5. `geometry-aware calibration`
- run id suffix: `geomcal_d3_main`
- bridge: `perceiver_resampler`
- feature source: `posterior_mu`
- batching: `BS=192`, `GA=1`, effective `192`
- target steps: `6000`
- key args:
  - `--prefix_geom_mlp_ratio 0.5`
  - `--prefix_geom_token_mixer_layers 1`

6. `adaptive selection v2`
- run id suffix: `topk32_d3_main`
- bridge: `perceiver_resampler`
- feature source: `posterior_mu`
- batching: `BS=192`, `GA=1`, effective `192`
- target steps: `6000`
- key args:
  - `--bridge_token_selector_type topk`
  - `--bridge_token_select_k 32`

7. `structured roles`
- run id suffix: `structuredroles_d3_exp`
- bridge: `structured_roles`
- feature source: `posterior_mu`
- batching: `BS=192`, `GA=1`, effective `192`
- target steps: `6000`
- key args:
  - `--bridge_num_roles 4`

8. `evidence sparse`
- run id suffix: `evidencesparse_d3_exp`
- bridge: `evidence_sparse`
- feature source: `posterior_mu`
- batching: `BS=192`, `GA=1`, effective `192`
- target steps: `6000`
- key args:
  - `--bridge_evidence_topk 24`

## Notes

- The queue intentionally excludes periodic validation to avoid eval-induced training degradation.
- `safe qcond` uses prompt-only question context to avoid the previously identified answer leakage path.
- `oracle196 + topk64` keeps the conservative `64x3` pairing because it is already field-tested on this GPU.
- `multiscale` keeps `128x2` because it was the heaviest new branch in the probe pass.

## Launcher

- script: `scripts/launch_final_arch_run_queue_v1.sh`
- recommended command:

```bash
RUN_PREFIX=mmarch_final_v1_20260310 ./scripts/launch_final_arch_run_queue_v1.sh
```

- rerun/resume command:

```bash
RUN_PREFIX=mmarch_final_v1_20260310 ./scripts/launch_final_arch_run_queue_v1.sh
```
