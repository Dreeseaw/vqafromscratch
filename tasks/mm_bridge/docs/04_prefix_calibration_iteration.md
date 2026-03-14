# Prefix Calibration Iteration (In Progress)

## Objective
Test the new calibrated bridge interface under shortened but indicative training schedules, then escalate toward >40% by unfreezing top LM layers.

## Code Added
- Prefix calibration and regularizers in multimodal trainer:
  - `--prefix_calibration`
  - `--prefix_calib_layernorm/--no-prefix_calib_layernorm`
  - `--prefix_calib_bias/--no-prefix_calib_bias`
  - `--prefix_calib_gate_init`
  - `--prefix_norm_target_ratio`
  - `--prefix_norm_reg_weight`
  - `--prefix_batchvar_reg_weight`

## Active Sweep
- Launcher: `tasks/mm_bridge/scripts/launch_prefix_calib_sweep.sh`
- Active sweep dir: `logs/mmcal_sweep_20260309_105004`
- Planned sequence:
  1. `mmcal_mlp49_calib_bonly_v1`
  2. `mmcal_mlp49_calib_top1_v1`
  3. `mmcal_mlp49_calib_top2_v1`
  4. `mmcal_lt49_top1_v1`

## Final Sweep Results (all runs completed)

Eval setup for all values below:
- official scorer
- 120 eval batches (`30,720` val samples)
- final checkpoint eval after `max_steps=2500`

| run_id | config summary | final overall acc |
|---|---|---:|
| `mmcal_mlp49_calib_bonly_v1` | MLP+calibration, bridge-only | `0.3402` |
| `mmcal_mlp49_calib_top1_v1` | MLP+calibration, top-1 LM layers trainable | `0.4160` |
| `mmcal_mlp49_calib_top2_v1` | MLP+calibration, top-2 LM layers trainable | `0.4144` |
| `mmcal_lt49_top1_v1` | learned tokens, top-1 LM layers trainable | `0.3893` |

## Trajectory Notes (Run 1: `mmcal_mlp49_calib_bonly_v1`)
- At step 500 (eval on 120 batches = 30,720 val samples):
  - `overall_accuracy=0.3180`
- At step 1000 (same eval budget):
  - `overall_accuracy=0.3300`
- Calibration dynamics are active:
  - prefix/text norm ratio reduced from ~`11.2` (step 20) to ~`5.17` (step 1000)
  - regularization terms steadily decline
- Current interpretation:
  - bridge-only + calibration is stabilizing geometry,
  - but still not enough for large gains on its own.
  - top-LM unfreezing with calibrated image bridge is the high-leverage path and crossed the 40% target in this shortened schedule.
