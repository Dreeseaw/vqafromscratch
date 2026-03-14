# Hammer KV-Cache Correctness Report

Date:
- 2026-03-12

Artifacts:
- launcher: `tasks/mm_bridge/scripts/launch_hammer_kvcache_correctness_v1.sh`
- probe script: `tasks/mm_bridge/scripts/mm_kvcache_correctness_probe.py`
- sweep dir: `logs/mmhammer_kvcorrect_v1_20260312_193655`
- latest symlink: `logs/mmhammer_kvcorrect_v1_latest`

Purpose:
- verify that the new batched eval KV-cache continuation path reproduces the old serial fallback exactly on real checkpoints
- check both historical canaries and Hammer-family bridge-only runs

Protocol:
- compare `--eval_use_kv_cache --eval_kv_cache_mode serial` vs `--eval_use_kv_cache --eval_kv_cache_mode batched`
- same checkpoint, same split, same batch size, same `10` eval batches
- batch size `64`, so each comparison covered `640` eval samples
- pass rule: zero prediction mismatches and zero missing question ids

## Result

All five probed checkpoints passed exactly.

Table:

| Label | Checkpoint | Samples | Serial Acc | Batched Acc | Exact Match | Serial samp/s | Batched samp/s | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `structuredroles_frontier_step9000` | `logs/mmarch_high_entropy_v1_20260311_structuredroles_frontier/step_9000.tar` | 640 | 0.463750 | 0.463750 | 1.000000 | 30.317 | 173.869 | 5.74x |
| `safeqcond_earlylayer_geomcal_frontier_step9000` | `logs/mmarch_high_entropy_v1_20260311_safeqcond_earlylayer_geomcal_frontier/step_9000.tar` | 640 | 0.503281 | 0.503281 | 1.000000 | 31.006 | 174.032 | 5.61x |
| `hammer_qquery_step2000` | `logs/mmhammer_v1_qquery_earlylayer_geomcal/step_2000.tar` | 640 | 0.453906 | 0.453906 | 1.000000 | 33.030 | 174.423 | 5.28x |
| `perf_dynbudget_qscore_step40` | `logs/mmhammer_perf_v1_20260312_dynbudget_qscore_earlylayer_geomcal_b64a3/step_40.tar` | 640 | 0.305469 | 0.305469 | 1.000000 | 30.861 | 169.056 | 5.48x |
| `perf_qquery_dynbudget_step40` | `logs/mmhammer_perf_v1_20260312_qquery_dynbudget_earlylayer_geomcal_b64a3/step_40.tar` | 640 | 0.333594 | 0.333594 | 1.000000 | 29.036 | 180.616 | 6.22x |

## Read

What this establishes:

- the batched continuation path matched the serial fallback exactly on all probed checkpoints
- this includes the historical non-qcond canary (`structuredroles_frontier`)
- this includes the best old qcond frontier checkpoint
- this includes the new Hammer qquery and dynbudget bridge-only families

What it does not establish:

- this is not yet a full-eval proof across every checkpoint family
- if a future checkpoint shows disagreement, `serial` should still be treated as the fallback reference mode

Operational conclusion:

- `--eval_use_kv_cache --eval_kv_cache_mode batched` is now supported by real-checkpoint evidence, not just toy-model checks
- for the tested families, the speedup over serial fallback was about `5.3x` to `6.2x` in eval samples/sec
