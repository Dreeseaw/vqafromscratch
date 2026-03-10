# Night Sweep Plan V1 (Architecture-Focused)

## Context Snapshot

- Current best validated baseline family: prefix-calibrated MLP bridge with partial LM unfreeze.
- Best score so far: **0.4345** (`logs/mmcal2_top2_cos_ext1/step_5000.tar`).
- Practical bottleneck from prior analysis: image-conditioned prefixes underperform when interface stability is weak.

This night plan targets architecture upgrades for the bridge itself, while keeping data roots unchanged:

- images: `images`
- annotations: `data/vqav2`

## Goals for the Night

1. Test the manager-priority bridge (`learned_query`) as the primary candidate.
2. Compare extraction quality from richer reducers (spatial mixer, perceiver, q-former-lite).
3. Test a hybrid constant+image prefix to combine stability + visual variation.
4. Keep run length in an **~8 hour** envelope (sequential, unattended).

## Quick Performance Findings (Before Launch)

Measured on short Docker runs with the learned-query setup:

- `fp32`, batch 256, workers 2: ~`0.83` steps/s (`bench_lq_speed`)
- `bf16`, batch 256, workers 2: ~`1.72` steps/s (`bench_lq_bf16`)
- `bf16`, batch 192, workers 2: ~`2.36` steps/s (`bench_lq_bf16_b192`)
- `bf16`, batch 128, accum 2: ~`1.76` steps/s (`bench_lq_bf16_b128a2`)
- `bf16`, batch 192, workers 4: up to ~`3.87` steps/s in short run (`bench_lq_bf16_b192_w4`)

Vision-on-CPU test:

- `bf16`, batch 192, `--vision_device cpu`: ~`0.22` steps/s (`bench_lq_bf16_b192_viscpu2`)
- conclusion: **possible but much slower** in this pipeline (GPU->CPU image transfer + CPU encode + CPU->GPU feature transfer).

## Prepared Run Script

- Launcher: `scripts/launch_night_bridge_sweep_v1.sh`
- Default schedule: 5 sequential runs
- Default runtime knobs:
  - `MAX_STEPS=3000`
  - `EVAL_EVERY=750`
  - `EVAL_BATCHES=80`
  - `BATCH_SIZE=192`
  - `GRAD_ACCUM_STEPS=1`
  - `NUM_WORKERS=4`
  - `PREFETCH_FACTOR=2`
  - `precision=bf16`

Expected wall-clock envelope is approximately **6-9 hours** depending on bridge compute cost and eval speed.

## Run Order and Rationale

1. `lq_base`
   - `bridge_type=learned_query`
   - baseline for learned-query cross-attention reducer.
2. `lq_spmix_sa1`
   - learned-query + one spatial self-attn mixer block before reduction.
   - isolates benefit of pre-compression spatial interaction.
3. `hybrid_tok065_lqimg`
   - `bridge_type=hybrid_const_image` with token-wise gate (`alpha` init 0.65).
   - tests stable learned prefix blended with learned-query image prefix.
4. `lq_spmix_conv1d1`
   - learned-query + lightweight conv1d token mixer.
   - low-overhead spatial mixing ablation.
5. `perceiver_d2_sa1`
   - perceiver-style latent resampler with 2 cross-attn rounds.
6. `qformer_d2_sa1`
   - q-former-lite with 2 alternating self/cross blocks.

All runs use:

- prefix calibration enabled
- top-2 LM layers trainable (`freeze_mode=bridge_plus_top_lm`, `train_top_lm_layers=2`)
- cosine LR schedule (`lr=2e-4`, warmup 500, min ratio 0.15)

## How To Launch

```bash
./scripts/launch_night_bridge_sweep_v1.sh
```

Optional safer-memory override (if needed):

```bash
BATCH_SIZE=128 GRAD_ACCUM_STEPS=2 ./scripts/launch_night_bridge_sweep_v1.sh
```

## Monitoring Commands

```bash
tail -f logs/mmnight_bridge_v1_latest/timeline.log
```

```bash
ls -1dt logs/mmnight_bridge_v1_* | head -n 1
```

```bash
for d in $(ls -1dt logs/mmnight_bridge_v1_*_* 2>/dev/null | head -n 5); do
  f="$d/logfile.txt"
  [[ -f "$f" ]] || continue
  echo "=== $d ==="
  rg "overall_accuracy=" "$f" | tail -n 1
done
```
