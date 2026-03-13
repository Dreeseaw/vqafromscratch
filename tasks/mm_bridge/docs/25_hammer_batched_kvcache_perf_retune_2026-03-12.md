# Hammer Batched KV-Cache Perf Retune

Date:
- 2026-03-12

Purpose:
- refresh Hammer launcher settings after the eval KV-cache continuation path was changed from serial fallback to batched mode
- replace the earlier bridge-only tuning assumptions that were dominated by the broken serial eval path

Source:
- launcher: `tasks/mm_bridge/scripts/launch_hammer_perf_probes_v1.sh`
- probe dir: `logs/mmhammer_perfprobe_v1_20260312_201307`
- run prefix: `mmhammer_perf_v2_20260312`

Probe settings:
- `max_steps=40`
- `final_eval_batches=4`
- `eval_use_kv_cache`
- `eval_kv_cache_mode=batched`
- effective batch `192`

## Results

All seven Hammer families passed the existing throughput floors at `192 x 1`.

| Variant | Layout | Train steps/s | Eval steps/s | Outcome |
|---|---:|---:|---:|---|
| `qquery_earlylayer_geomcal` | `192 x 1` | `5.22` | `1.52` | pass |
| `adapter_safeqcond_earlylayer_geomcal` | `192 x 1` | `4.93` | `1.55` | pass |
| `dynbudget_qscore_earlylayer_geomcal` | `192 x 1` | `5.16` | `1.37` | pass |
| `qquery_adapter_earlylayer_geomcal` | `192 x 1` | `4.93` | `1.30` | pass |
| `qquery_dynbudget_earlylayer_geomcal` | `192 x 1` | `5.15` | `1.26` | pass |
| `dynbudget_adapter_earlylayer_geomcal` | `192 x 1` | `4.70` | `1.36` | pass |
| `qquery_dynbudget_adapter_earlylayer_geomcal` | `192 x 1` | `4.77` | `1.36` | pass |

## Read

What changed:

- the earlier `64 x 3` / optional-omit recommendations were artifacts of the old serial KV-cache eval path
- once the batched continuation path was both corrected and revalidated on real checkpoints, the bridge-only families stopped being eval-bound

Launcher consequence:

- all Hammer families should now run by default
- all Hammer families can use `batch_size=192`, `grad_accum_steps=1`
- bridge-only families should use `eval_use_kv_cache` with `eval_kv_cache_mode=batched`
- the old “noncompliant dynbudget-only” label is no longer appropriate under the retuned path

Supersession note:

- this note supersedes the performance-layout recommendations in `23_hammer_perf_tuning_report_2026-03-12.md`
