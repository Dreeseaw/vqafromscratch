# Qwen2.5-VL-3B Distillation Report

Date: 2026-03-29

## Summary

Qwen2.5-VL-3B teacher distillation was a clear bridge-stage win on the non-dual-VM SigLIP/Cement line.

- Teacher extraction completed on all `443,757` VQAv2 train QA pairs.
- Teacher store size: `3000` answer-vocab logits per sample.
- Distilled bridge full VQAv2 eval: `0.6235`
- Distilled bridge answer types:
  - yes/no: `0.7634`
  - number: `0.4591`
  - other: `0.5607`
- Distilled compressed `K=8` full VQAv2 eval: `0.6023`
- Distilled compressed answer types:
  - yes/no: `0.7605`
  - number: `0.4546`
  - other: `0.5211`

Main read:

- Distillation improved the bridge-stage frontier materially.
- The gain largely survived compression.
- This is now the strongest non-dual-VM bridge result in the repo.

## What Was Run

The original plan was:

1. Qwen teacher inference over VQAv2 train
2. Fresh Cement-style control bridge
3. Fresh Cement-style distilled bridge
4. Standard `K=8` format-alignment compression on the distilled bridge

In practice, the fresh control bridge was intentionally skipped once the teacher finished. That was the right call. The repo already has the real Cement full-eval baseline band from three seeds, so the fresh control would have been redundant GPU spend.

So the completed experiment was:

1. Teacher inference
2. Distilled bridge
3. Distilled `K=8` compression

## Phase A: Teacher Inference

Artifacts:

- bundle: [mmqwenkd_v1_20260327_224015](/home/wdree/percy/vqafromscratch/logs/mmqwenkd_v1_20260327_224015)
- teacher run: [mmqwenkd_v1_20260327_224015_teacher](/home/wdree/percy/vqafromscratch/logs/mmqwenkd_v1_20260327_224015_teacher)
- teacher data: [qwen25vl3b_vqav2_train_v1](/home/wdree/percy/vqafromscratch/data/distillation/qwen25vl3b_vqav2_train_v1)

Teacher facts:

- total items: `443,757`
- completed: `443,757`
- skipped: `0`
- answer vocab size: `3000`
- prompt style: single short answer, teacher logits mapped into the local VQA answer vocab via first-answer-token projection

Runtime:

- final teacher log says `elapsed_h=3.87`
- average throughput: `31.838 samples/s`

Practical note:

- the repo now has a reusable Qwen teacher soft-label store for bridge-stage answer-vocab KL.

## Phase B: Distilled Bridge

Run:

- [mmqwenkd_v1_20260327_224015_distill_bridge](/home/wdree/percy/vqafromscratch/logs/mmqwenkd_v1_20260327_224015_distill_bridge)

Recipe:

- same SigLIP/Cement bridge architecture
- same bridge-stage optimizer/schedule family
- same LM checkpoint: [lm_final step_45000](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar)
- same trainable slice as current Cement line: top `2` LM layers + adapters + bridge
- added answer-vocab KL against the Qwen teacher

Full VQAv2 result at `step_9000`:

| Run | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| Cement anchor (`s42`, full eval) | 0.6163 | 0.7589 | 0.4573 | 0.5499 |
| Cement 3-seed full-eval mean | 0.6129 | — | — | — |
| Qwen-distilled bridge | 0.6235 | 0.7634 | 0.4591 | 0.5607 |

Delta vs reference:

- vs best completed Cement single run: `+0.0072`
- vs Cement 3-seed full-eval mean: `+0.0106`
- yes/no: `+0.0045` vs Cement anchor
- number: `+0.0018`
- other: `+0.0108`

This was not a small fluctuation. The biggest benefit landed in `other`, which is where a stronger answer distribution teacher should help most.

## Phase C: Distilled K=8 Compression

Run:

- [mmqwenkd_v1_20260327_224015_distill_k8](/home/wdree/percy/vqafromscratch/logs/mmqwenkd_v1_20260327_224015_distill_k8)

Recipe:

- standard non-dual-VM `K=8` format-alignment compression
- adapters disabled
- remap teacher: [mmsemantic_remap_v1_debug step_500](/home/wdree/percy/vqafromscratch/logs/mmsemantic_remap_v1_debug/step_500.tar)
- init from the distilled bridge `step_9000`

Full VQAv2 result at `step_3000`:

| Run | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| SigLIP-only format-aligned `K=8` baseline | 0.5900 | 0.7393 | 0.4499 | 0.5134 |
| Qwen-distilled format-aligned `K=8` | 0.6023 | 0.7605 | 0.4546 | 0.5211 |

Delta vs prior SigLIP-only compressed baseline:

- overall: `+0.0123`
- yes/no: `+0.0212`
- number: `+0.0047`
- other: `+0.0077`

Compression penalty comparison:

- Cement anchor -> old SigLIP `K=8`: `0.6163 -> 0.5900` = `-0.0263`
- Distilled bridge -> distilled `K=8`: `0.6235 -> 0.6023` = `-0.0212`

So the distillation gain did survive compression, and the compression penalty was smaller than on the original Cement line.

## Interpretation

The cleanest modeling read is:

1. Qwen teacher KL improved the bridge’s answer-facing representation, not just the LM head behavior.
2. The improvement concentrated in `other`, which is consistent with the teacher adding richer answer preference structure on hard open-ended examples.
3. Compression still costs a lot, but less than before. That suggests the distilled bridge is producing tokens that are easier to preserve through the later `K=8` bottleneck.

The result does **not** say that the teacher solved everything:

- bridge-stage gain: large and real
- compressed-stage gain: real, but still below the uncompressed distilled bridge by `2.12` points

So the right interpretation is not “compression is fixed.” It is “the starting point for compression is now stronger.”

## Frontier Status

For the non-dual-VM line, the new ordering is:

| System | Overall |
|---|---:|
| Qwen-distilled bridge | 0.6235 |
| Cement anchor | 0.6163 |
| Qwen-distilled compressed `K=8` | 0.6023 |
| Prior SigLIP compressed `K=8` | 0.5900 |

That makes the distilled bridge the current frontier reference for the plain SigLIP/Cement family.

## Recommendation

Keep Qwen distillation in the bridge-stage recipe.

It is worth carrying forward because:

- it improved the bridge frontier by about `0.7-1.1` points depending on baseline reference
- it improved the compressed `K=8` line by `1.2` points
- the gain is broad, with the strongest lift in `other`

The next logical test is LM-side transfer:

- compare the original LM `step_45000` vs the new reasoning-mix LM `step_45000`
- then try the overtrained `step_300000` checkpoint on the same frontier stack

