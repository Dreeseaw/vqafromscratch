# Dinner Follow-Up Runs Report (2026-03-09)

## Scope

This report summarizes the two latest follow-up runs launched after the night-plan preparation:

1. `mmdinner_lq_deeper_sa2_ref2_clean_20260309_213605`
2. `mmdinner_perceiver_d3_notnight_20260309_221422`

Both use official VQAv2 scoring and fixed eval slices (`eval_batches=80`).

## Final Results

| Model | Overall Accuracy |
|---|---:|
| `mmdinner_lq_deeper_sa2_ref2_clean_20260309_213605` | `0.4257` |
| `mmdinner_perceiver_d3_notnight_20260309_221422` | `0.4415` |

## Run Notes

### 1) `mmdinner_lq_deeper_sa2_ref2_clean_20260309_213605`
- Bridge: `learned_query`
- Distinguishing change: deeper learned-query stack (`pre_mixer=self_attn x2`, `refine_layers=2`)
- Final: `0.4257`
- Interpretation: stable training, but under the best prior calibrated top2 baseline (`0.4345`).

### 2) `mmdinner_perceiver_d3_notnight_20260309_221422`
- Bridge: `perceiver_resampler`
- Distinguishing change: deeper resampler (`bridge_query_depth=3`) with mild prefix dropout (`0.03`)
- Final: `0.4415`
- Interpretation: strongest score so far in this branch; beats prior best (`0.4345`) by `+0.0070`.

## Practical Takeaway

For the immediate next cycle, `perceiver_resampler` with deeper latent updates appears to be the most promising non-night-plan architecture explored so far.

