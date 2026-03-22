# Architecture Coverage Big Sweep V1 (6h) - 2026-03-10

## Purpose

Run a large architecture-focused queue that prioritizes:

1. Question-conditioned perceiver bridge
2. Early-layer VM feature-source variants
3. Large-token oracle checks
4. Adaptive token selection variants

This is designed for exploration-first progress before heavy ablation stabilization.

## Launcher

- Script: `tasks/mm_bridge/scripts/launch_arch_coverage_big_sweep_v1_6h.sh`
- Docker path: uses `./runmm.sh` for every run (Docker-backed).
- Horizon guard: default `HORIZON_HOURS=6`.

## Restart Safety (Skip + Resume)

The launcher is restart-safe when `RUN_PREFIX` is kept unchanged.

- Complete-run skip:
  - If `logs/<run_id>/step_<target>.tar` exists, the run is skipped.
- Partial-run resume:
  - If incomplete checkpoints exist, launcher resumes from latest `step_<N>.tar`.
- Practical rule:
  - Re-run launcher with same `RUN_PREFIX` to continue without redoing finished runs.

## Default Runtime Knobs

- `MAX_STEPS_MAIN=7000`
- `MAX_STEPS_EXP=5000`
- `MAX_STEPS_HEAVY=4000`
- `batch_size=192` (`96` + accum `2` on 196-token oracle runs)
- `precision=bf16`
- `eval_every=1000`
- `eval_batches=160`
- `freeze_mode=bridge_plus_top_lm`
- `train_top_lm_layers=2`

## Queue Order

1. `perceiver_d3_anchor`
2. `perceiver_d3_qcond`
3. `perceiver_d3_qcond_encoder`
4. `perceiver_d3_topk24`
5. `perceiver_d3_qcond_topk24`
6. `perceiver_d3_qcond_topk24_encoder`
7. `perceiver_oracle196`
8. `perceiver_oracle196_qcond`
9. `perceiver_oracle196_qcond_encoder`
10. `perceiver_d4_qcond`

All run IDs are prefixed by:
- `RUN_PREFIX` (default: `mmarch_cov_v1_20260310`)

## Launch / Relaunch

```bash
./tasks/mm_bridge/scripts/launch_arch_coverage_big_sweep_v1_6h.sh
```

Relaunch same queue without redoing completed runs:

```bash
RUN_PREFIX=mmarch_cov_v1_20260310 ./tasks/mm_bridge/scripts/launch_arch_coverage_big_sweep_v1_6h.sh
```

Optional dry-run command print:

```bash
DRY_RUN=1 ./tasks/mm_bridge/scripts/launch_arch_coverage_big_sweep_v1_6h.sh
```
