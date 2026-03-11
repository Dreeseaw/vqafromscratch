# New Architecture Memory Probes V1 - 2026-03-10

## Purpose

Benchmark the newly added architecture branches with short Docker-backed runs to determine safe `batch_size` / `grad_accum_steps` pairs before launching longer jobs.

## Covered Branches

1. leakage-safe qcond perceiver
2. multi-scale perceiver
3. geometry-aware prefix calibration
4. structured token roles bridge
5. evidence-focused sparse bridge

## Probe Policy

- short training window (`max_steps=60`)
- no periodic evals (`eval_every=0`)
- tiny final eval only (`eval_batches=1`, `limit_eval=64`)
- aggressive BS/GA attempt first
- safe fallback if the first attempt fails

## Launcher

- Script: `scripts/launch_new_arch_memory_probes_v1.sh`
- Results table: `logs/mmarch_memprobe_v1_latest/probe_results.tsv`

## Initial Candidate Pairs

- leakage-safe qcond: `192x1` then `128x2`
- multi-scale: `128x2` then `64x3`
- geometry-aware calibration: `192x1` then `128x2`
- structured roles: `192x1` then `128x2`
- evidence sparse: `192x1` then `128x2`

## Results

All first-pass probes completed successfully in Docker.

| Branch | Tested BS | Tested GA | Status | Recommended next long-run pair |
|---|---:|---:|---|---|
| leakage-safe qcond perceiver | 192 | 1 | success | `192x1` |
| multi-scale perceiver | 128 | 2 | success | `128x2` |
| geometry-aware prefix calibration | 192 | 1 | success | `192x1` |
| structured token roles bridge | 192 | 1 | success | `192x1` |
| evidence-focused sparse bridge | 192 | 1 | success | `192x1` |

## Notes

- `multi-scale` was intentionally probed at `128x2` rather than `192x1` because it is the heaviest new branch in the current set.
- The other newly added branches all cleared the more aggressive `192x1` target on the current GPU.
- For this project stage, these should be treated as practical defaults for the first real training runs, not absolute maxima.
- If we want to squeeze more throughput later, the next obvious follow-up is a dedicated `multi-scale 192x1` fit check.

## Artifacts

- Probe launcher: `scripts/launch_new_arch_memory_probes_v1.sh`
- Timeline log: `logs/mmarch_memprobe_v1_latest/timeline.log`
- Result table: `logs/mmarch_memprobe_v1_latest/probe_results.tsv`
