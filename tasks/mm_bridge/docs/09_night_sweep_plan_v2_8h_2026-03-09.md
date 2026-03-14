# Night Sweep Plan V2 (Final, 8h Horizon)

## Why This Update

This plan supersedes V1 based on the latest two runs plus throughput profiling:

- `mmdinner_perceiver_d3_notnight_20260309_221422` reached **0.4415** (current best).
- `mmdinner_lq_deeper_sa2_ref2_clean_20260309_213605` reached **0.4257**.
- Throughput sweet spot remained:
  - `bf16`
  - `batch_size=192`
  - `num_workers=4`
  - `prefetch_factor=2`
- Vision-on-CPU remained much slower and is not used for night runs.

## Launcher

- `tasks/mm_bridge/scripts/launch_night_bridge_sweep_v2_8h.sh`
- Sequential queue with an **8-hour time guard** (`HORIZON_HOURS=8` default).
- If horizon is reached, remaining runs are skipped automatically.

## Runtime Defaults

- `MAX_STEPS_MAIN=9000`
- `MAX_STEPS_EXP=5000`
- `EVAL_EVERY=1000`
- `EVAL_BATCHES=200`
- `BATCH_SIZE=192`
- `GRAD_ACCUM_STEPS=1`
- `NUM_WORKERS=4`
- `PREFETCH_FACTOR=2`
- `precision=bf16`

## Queue Order (Highest Expected Value First)

| Order | Model Variant | Max Steps | Purpose |
|---|---|---:|---|
| 1 | `perceiver_d3_pd03_main` | 9000 | Re-run current best family to confirm stability and ceiling |
| 2 | `perceiver_d3_pd00_main` | 9000 | Dropout ablation around winner |
| 3 | `perceiver_d4_pd03_main` | 9000 | Depth scaling test |
| 4 | `hybrid_tok060_perc_d3_main` | 9000 | Hybrid constant + perceiver image branch |
| 5 | `hybrid_tok075_perc_d3_main` | 9000 | Hybrid gate sensitivity |
| 6 | `perceiver_d3_sa1_main` | 9000 | Spatial mixer interaction test |
| 7 | `qformer_d3_exp` | 5000 | Secondary architecture exploratory |
| 8 | `lq_ref2_sa1_exp` | 5000 | Learned-query sanity comparator |

## Expected Night Envelope

Given observed `~3.0 steps/s` on recent best runs and eval overhead at 80 batches:

- main run: roughly 55-70 minutes each
- exploratory run: roughly 30-45 minutes each

The queue is intended to consume approximately an 8-hour overnight window, with horizon clipping if runtime varies.

## Launch

```bash
./tasks/mm_bridge/scripts/launch_night_bridge_sweep_v2_8h.sh
```

Optional tighter memory mode:

```bash
BATCH_SIZE=128 GRAD_ACCUM_STEPS=2 ./tasks/mm_bridge/scripts/launch_night_bridge_sweep_v2_8h.sh
```

## Monitor

```bash
tail -f logs/mmnight_bridge_v2_8h_latest/timeline.log
```
