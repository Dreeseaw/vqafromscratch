# Prefix Calibration Sweep V2 (Launched)

## Sweep
- Script: `scripts/launch_prefix_calib_sweep_v2.sh`
- Sweep dir: `logs/mmcal_sweep_v2_20260309_140137`
- Data roots unchanged:
  - `images_root=images`
  - `annotations_root=data/vqav2`

## Planned Runs
1. `mmcal2_top1_const_ext1`
2. `mmcal2_top1_cos_ext1`
3. `mmcal2_top1_cos_pd05_ext1`
4. `mmcal2_top2_cos_ext1`

All runs:
- `max_steps=5000`
- `eval_every=1000`
- `eval_batches=160` (indicative larger slice than prior 120)
- calibrated MLP bridge (`K=49`, `bridge_add_2d_pos_emb`)
- `freeze_mode=bridge_plus_top_lm`

## Live Status Snapshot
- Active run: `mmcal2_top1_const_ext1`
- Early metrics:
  - step 20 loss `2.7438`
  - step 40 loss `2.3710`
  - step 60 loss `2.2342`

Monitoring:
- `tail -f logs/mmcal_sweep_v2_20260309_140137/timeline.log`
- `tail -f logs/mmcal2_top1_const_ext1/logfile.txt`
