# All Runs Structured Report (2026-03-10)

## Scope

This report consolidates the bridge-focused multimodal runs used in this investigation arc:

- historical gap baselines (`mmbr_basesweep_*`)
- prefix-calibration sweeps (`mmcal_*`, `mmcal2_*`)
- dinner follow-up probes (`mmdinner_*`)
- completed night sweep (`mmnight_bridge_v2_8h_20260309_234936_*`)

All values below are taken from each run's final logged official evaluation in `logs/<run_id>/logfile.txt`.

## Model -> Overall Accuracy

| Model | Overall Accuracy |
|---|---:|
| `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_pd03_main` | `0.4544` |
| `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_sa1_main` | `0.4542` |
| `mmnight_bridge_v2_8h_20260309_234936_hybrid_tok075_perc_d3_main` | `0.4538` |
| `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_pd00_main` | `0.4531` |
| `mmnight_bridge_v2_8h_20260309_234936_perceiver_d4_pd03_main` | `0.4529` |
| `mmnight_bridge_v2_8h_20260309_234936_hybrid_tok060_perc_d3_main` | `0.4527` |
| `mmdinner_perceiver_d3_notnight_20260309_221422` | `0.4415` |
| `mmnight_bridge_v2_8h_20260309_234936_lq_ref2_sa1_exp` | `0.4388` |
| `mmnight_bridge_v2_8h_20260309_234936_qformer_d3_exp` | `0.4383` |
| `mmcal2_top2_cos_ext1` | `0.4345` |
| `mmcal2_top1_cos_pd05_ext1` | `0.4325` |
| `mmcal2_top1_cos_ext1` | `0.4323` |
| `mmcal2_top1_const_ext1` | `0.4319` |
| `mmdinner_lq_deeper_sa2_ref2_clean_20260309_213605` | `0.4257` |
| `mmcal_mlp49_calib_top1_v1` | `0.4160` |
| `mmcal_mlp49_calib_top2_v1` | `0.4144` |
| `mmcal_lt49_top1_v1` | `0.3893` |
| `mmbr_basesweep_lt1` | `0.3540` |
| `mmcal_mlp49_calib_bonly_v1` | `0.3402` |
| `mmbr_basesweep_on_high` | `0.3429` |
| `mmbr_basesweep_off_high` | `0.3368` |

## Chronological Run Ledger

| Phase | Run ID | Bridge | Final Step | Overall | Yes/No | Number | Other | Last steps_per_s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| historical baseline | `mmbr_basesweep_lt1` | `learned_tokens` | 17330 | 0.3540 | 0.6392 | 0.2724 | 0.1585 | 1.65 |
| historical baseline | `mmbr_basesweep_on_high` | `mlp` | 3466 | 0.3429 | 0.6356 | 0.2807 | 0.1364 | 1.68 |
| historical baseline | `mmbr_basesweep_off_high` | `mlp` | 3466 | 0.3368 | 0.6409 | 0.2729 | 0.1220 | 1.61 |
| prefix-calib sweep v1 | `mmcal_mlp49_calib_bonly_v1` | `mlp` | 2500 | 0.3402 | 0.6467 | 0.2713 | 0.1243 | 1.15 |
| prefix-calib sweep v1 | `mmcal_mlp49_calib_top1_v1` | `mlp` | 2500 | 0.4160 | 0.6644 | 0.2986 | 0.2575 | 1.19 |
| prefix-calib sweep v1 | `mmcal_mlp49_calib_top2_v1` | `mlp` | 2500 | 0.4144 | 0.6617 | 0.2971 | 0.2568 | 1.13 |
| prefix-calib sweep v1 | `mmcal_lt49_top1_v1` | `learned_tokens` | 2500 | 0.3893 | 0.6429 | 0.2833 | 0.2237 | 1.25 |
| prefix-calib sweep v2 | `mmcal2_top1_const_ext1` | `mlp` | 5000 | 0.4319 | 0.6791 | 0.2943 | 0.2805 | 1.32 |
| prefix-calib sweep v2 | `mmcal2_top1_cos_ext1` | `mlp` | 5000 | 0.4323 | 0.6790 | 0.3009 | 0.2798 | 1.24 |
| prefix-calib sweep v2 | `mmcal2_top1_cos_pd05_ext1` | `mlp` | 5000 | 0.4325 | 0.6788 | 0.2999 | 0.2807 | 1.30 |
| prefix-calib sweep v2 | `mmcal2_top2_cos_ext1` | `mlp` | 5000 | 0.4345 | 0.6803 | 0.2988 | 0.2838 | 1.30 |
| dinner follow-up | `mmdinner_lq_deeper_sa2_ref2_clean_20260309_213605` | `learned_query` | 4200 | 0.4257 | 0.6806 | 0.2937 | 0.2701 | 2.92 |
| dinner follow-up | `mmdinner_perceiver_d3_notnight_20260309_221422` | `perceiver_resampler` | 7000 | 0.4415 | 0.6861 | 0.3085 | 0.2939 | 3.14 |
| night sweep v2 | `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_pd03_main` | `perceiver_resampler` | 9000 | 0.4544 | 0.6889 | 0.3125 | 0.3134 | 2.31 |
| night sweep v2 | `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_pd00_main` | `perceiver_resampler` | 9000 | 0.4531 | 0.6872 | 0.3106 | 0.3125 | 2.35 |
| night sweep v2 | `mmnight_bridge_v2_8h_20260309_234936_perceiver_d4_pd03_main` | `perceiver_resampler` | 9000 | 0.4529 | 0.6895 | 0.3095 | 0.3107 | 2.29 |
| night sweep v2 | `mmnight_bridge_v2_8h_20260309_234936_hybrid_tok060_perc_d3_main` | `hybrid_const_image` | 9000 | 0.4527 | 0.6874 | 0.3080 | 0.3123 | 2.34 |
| night sweep v2 | `mmnight_bridge_v2_8h_20260309_234936_hybrid_tok075_perc_d3_main` | `hybrid_const_image` | 9000 | 0.4538 | 0.6892 | 0.3098 | 0.3127 | 2.35 |
| night sweep v2 | `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_sa1_main` | `perceiver_resampler` | 9000 | 0.4542 | 0.6868 | 0.3064 | 0.3161 | 2.33 |
| night sweep v2 (explore) | `mmnight_bridge_v2_8h_20260309_234936_qformer_d3_exp` | `qformer_lite` | 5000 | 0.4383 | 0.6836 | 0.2983 | 0.2886 | 2.52 |
| night sweep v2 (explore) | `mmnight_bridge_v2_8h_20260309_234936_lq_ref2_sa1_exp` | `learned_query` | 5000 | 0.4388 | 0.6781 | 0.3051 | 0.2919 | 2.50 |

## Main Outcome

- Best overall run in this cycle: `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_pd03_main` at **0.4544**.
- The strongest cluster is now `perceiver_resampler` and `hybrid_const_image` around `0.4527` to `0.4544`.
- This materially outperforms earlier calibrated MLP runs (`~0.432` to `0.435`) and historical baselines (`~0.337` to `0.354`).

