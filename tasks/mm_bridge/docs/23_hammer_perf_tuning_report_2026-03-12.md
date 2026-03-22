# Hammer Performance Tuning Report - 2026-03-12

## Scope

This note records the short Hammer throughput probes used to choose
`batch_size x grad_accum_steps` layouts at effective batch `192`.

Probe source:

- `tasks/mm_bridge/scripts/launch_hammer_perf_probes_v1.sh`
- latest probe log dir: `logs/mmhammer_perfprobe_v1_latest`

Probe policy:

- `max_steps=40`
- `final_eval_batches=4`
- thresholds:
  - train `steps_per_s > 3.0`
  - eval `steps_per_s > 0.8`

## Results

| Variant | Best passing layout | Train steps/s | Eval steps/s | Outcome |
|---|---:|---:|---:|---|
| `qquery_earlylayer_geomcal` | `64 x 3` | `3.52` | `0.83` | pass |
| `adapter_safeqcond_earlylayer_geomcal` | `192 x 1` | `4.51` | `1.38` | pass |
| `dynbudget_qscore_earlylayer_geomcal` | none | n/a | n/a | no pass |
| `qquery_adapter_earlylayer_geomcal` | `192 x 1` | `4.49` | `1.30` | pass |
| `qquery_dynbudget_earlylayer_geomcal` | none | n/a | n/a | no pass |
| `dynbudget_adapter_earlylayer_geomcal` | `192 x 1` | `4.46` | `1.28` | pass |
| `qquery_dynbudget_adapter_earlylayer_geomcal` | `192 x 1` | `4.26` | `7.44` | pass |

## Failure Boundary

The two bridge-only dynbudget variants did not find a layout that cleared both
floors while keeping effective batch `192`.

### `dynbudget_qscore_earlylayer_geomcal`

- best train-compliant layout: `48 x 4` at `3.22 / 0.65`
- best eval-compliant layout: `32 x 6` at `2.15 / 0.96`
- fastest projected overnight layout under
  `9000 train steps + (100*9 + 1120) eval steps`: `64 x 3`
  at `3.99 / 0.64`

Conclusion:

- no effective-batch-192 layout satisfied both constraints
- if throughput floors are relaxed and pure wall-clock is the target, use `64 x 3`

### `qquery_dynbudget_earlylayer_geomcal`

- best train-compliant layout: `48 x 4` at `3.05 / 0.60`
- best eval-compliant layout: `32 x 6` at `2.05 / 0.92`
- fastest projected overnight layout under
  `9000 train steps + (100*9 + 1120) eval steps`: `64 x 3`
  at `3.68 / 0.60`

Conclusion:

- no effective-batch-192 layout satisfied both constraints
- if throughput floors are relaxed and pure wall-clock is the target, use `64 x 3`

## Launcher Decision

Recommended default Hammer queue:

1. `anchor_safeqcond_earlylayer_geomcal`
2. `qquery_earlylayer_geomcal`
3. `adapter_safeqcond_earlylayer_geomcal`
4. `qquery_adapter_earlylayer_geomcal`
5. `dynbudget_adapter_earlylayer_geomcal`
6. `qquery_dynbudget_adapter_earlylayer_geomcal`

Recommended layouts:

- bridge-only qquery family: `64 x 3`
- adapter family: `192 x 1`

Anchor note:

- `anchor_safeqcond_earlylayer_geomcal` was not re-probed separately
- the high-entropy frontier run already demonstrated that the same stack clears
  the throughput bar at `192 x 1`
- recorded high-entropy frontier evidence:
  - train `steps_per_s=4.88`
  - periodic mini-eval `100 / 54.1s = 1.85 eval steps/s`
  - full eval `1100 / 618.5s = 1.78 eval steps/s`
- the final launcher therefore keeps the anchor at `192 x 1`

Default omission:

- `dynbudget_qscore_earlylayer_geomcal`
- `qquery_dynbudget_earlylayer_geomcal`

Those two runs remain available as optional noncompliant launches, but they are
not suitable for the default Hammer queue under the current throughput
constraints.

Overnight note:

- if the objective changes from "clear both floors" to "finish fastest
  overnight," the optional noncompliant dynbudget-only runs should use `64 x 3`
