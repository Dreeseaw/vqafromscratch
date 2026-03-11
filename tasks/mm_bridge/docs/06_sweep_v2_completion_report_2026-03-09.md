# Sweep V2 Completion Report (2026-03-09)

## Scope

This document summarizes the **completed** run set from:

- `logs/mmcal_sweep_v2_20260309_153504`

No future-run planning is included here by design.

Primary target for this loop: exceed **40% overall VQAv2 val accuracy** with the same train/val data roots.

## Sweep Timeline

From `logs/mmcal_sweep_v2_20260309_153504/timeline.log`:

- 2026-03-09 15:35:04 EDT: skip `mmcal2_top1_const_ext1` (already had `step_5000.tar`)
- 2026-03-09 15:35:04 EDT: start `mmcal2_top1_cos_ext1`
- 2026-03-09 16:49:37 EDT: end `mmcal2_top1_cos_ext1`
- 2026-03-09 16:49:37 EDT: start `mmcal2_top1_cos_pd05_ext1`
- 2026-03-09 18:01:02 EDT: end `mmcal2_top1_cos_pd05_ext1`
- 2026-03-09 18:01:02 EDT: start `mmcal2_top2_cos_ext1`
- 2026-03-09 19:13:00 EDT: end `mmcal2_top2_cos_ext1`
- 2026-03-09 19:13:00 EDT: sweep complete

## Final Results (Official Scorer)

All runs use `eval_batches=160` and report final eval at step 5000.

| Run ID | Variant Summary | Final Overall Acc | Yes/No | Number | Other | Final Checkpoint |
|---|---|---:|---:|---:|---:|---|
| `mmcal2_top1_const_ext1` | top-1 LM layer trainable, constant LR, no prefix dropout | **0.4319** | 0.6791 | 0.2943 | 0.2805 | `logs/mmcal2_top1_const_ext1/step_5000.tar` |
| `mmcal2_top1_cos_ext1` | top-1 LM layer trainable, cosine + warmup, no prefix dropout | **0.4323** | 0.6790 | 0.3009 | 0.2798 | `logs/mmcal2_top1_cos_ext1/step_5000.tar` |
| `mmcal2_top1_cos_pd05_ext1` | top-1 LM layer trainable, cosine + warmup, prefix dropout 0.05 | **0.4325** | 0.6788 | 0.2999 | 0.2807 | `logs/mmcal2_top1_cos_pd05_ext1/step_5000.tar` |
| `mmcal2_top2_cos_ext1` | top-2 LM layers trainable, cosine + warmup, no prefix dropout | **0.4345** | 0.6803 | 0.2988 | 0.2838 | `logs/mmcal2_top2_cos_ext1/step_5000.tar` |

## Headline Findings

1. The loop target was met and exceeded.
   - Best run reached **43.45%** overall (`mmcal2_top2_cos_ext1`), clearly above 40%.

2. The best setting in this sweep was unfreezing two top LM blocks (`train_top_lm_layers=2`).
   - Gain vs strongest top-1 run: `0.4345 - 0.4325 = +0.0020` (0.20 accuracy points).

3. Prefix dropout (`0.05`) was mildly helpful in top-1 comparisons.
   - `top1_cos_pd05` (0.4325) slightly outperformed `top1_cos` (0.4323) and `top1_const` (0.4319).

4. Accuracy profile remains consistent with prior behavior.
   - `yes/no` is strongest, `number` remains the weakest answer type, `other` improves with stronger interface settings.

## Runtime Notes (Eval Cost)

Observed `batch=160/160` eval elapsed time:

- `mmcal2_top1_const_ext1`: ~189s to ~206s
- `mmcal2_top1_cos_ext1`: ~180s to ~194s
- `mmcal2_top1_cos_pd05_ext1`: ~191s to ~193s
- `mmcal2_top2_cos_ext1`: ~191s to ~203s

Eval remains the main periodic slowdown, but runs completed cleanly and consistently.

## Current Best Artifact

- Best checkpoint from this sweep family:
  - `logs/mmcal2_top2_cos_ext1/step_5000.tar`
- Best reported official val score:
  - **0.4345**

