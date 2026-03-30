# Variant A Co-Train Report

Primary bundle: [mmgrid_cotrain_v1_20260327_194350](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350)  
Launcher: [launch_variant_a_cotrain_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_variant_a_cotrain_v1.sh)

## Summary

This experiment was negative.

- Fresh end-to-end bridge training with the co-trained Variant A bottleneck finished at `0.5888` full VQAv2 val, well below the Cement bridge anchor `0.6163`.
- Carrying the trained bottleneck directly into stage-2 compression, without reinitialization as requested, finished at `0.5587`.
- That is worse than both the standard SigLIP-only format-aligned `K=8` line (`0.5900`) and the frozen-perceiver Variant A posthoc result (`0.5951`).
- The tiny-head probe also dropped sharply to `0.4452`, which means the co-trained compressed tokens were less linearly decodable, not more.

The useful conclusion is that Variant A is mildly helpful as a late bottleneck patch on top of a strong frozen perceiver, but it is not a good bridge-stage primitive in this form.

## Objective

Train the current Cement-style bridge from scratch with the Variant A semantic bottleneck in the forward path from step zero:

```text
SigLIP grid [196, D]
-> perceiver [49, D]
-> Variant A bottleneck K=8 attends over cat(perceiver [49], grid [196]) = [245, D]
-> LM prefix
```

Then run stage-2 format-alignment compression on top of that bridge, with one important user-directed change:

- stage 2 did **not** reinitialize the bottleneck weights

## Execution History

### Aborted first attempt

First bridge attempt:
- run: [mmgrid_cotrain_v1_20260327_193946_bridge](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_193946_bridge)
- loader: `64 x 3`, `num_workers=1`, `prefetch_factor=1`, `--no-pin_memory`
- observed throughput: `1.18-1.27 steps/s`
- stopped intentionally after the throughput complaint

This attempt is only provenance. It is not used for any result.

### Authoritative run

Authoritative bundle:
- [mmgrid_cotrain_v1_20260327_194350](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350)

Timeline:
- bridge start: `2026-03-27 19:43:50 EDT`
- bridge end: `2026-03-27 20:56:39 EDT`
- compression start: `2026-03-27 20:56:39 EDT`
- compression end: `2026-03-27 21:43:08 EDT`

Durations:
- bridge: about `1h 12m 49s`
- compression: about `46m 29s`
- total authoritative bundle: about `1h 59m 18s`

Recorded in:
- [progress.md](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/progress.md)
- [timeline.log](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/timeline.log)

## Stage 1: Fresh Bridge Training

Stage-1 run:
- [mmgrid_cotrain_v1_20260327_194350_bridge](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350_bridge)

Resolved runtime config from [logfile.txt](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350_bridge/logfile.txt):

- VM: frozen `siglip_base`
- LM checkpoint: `logs/lm_final/step_45000.tar`
- bridge: `perceiver_resampler`, depth `3`, `49` perceiver tokens
- query path: `question_hidden_attn`, `question_only`
- Variant A enabled:
  - `semantic_bottleneck=1`
  - `semantic_tokens=8`
  - `semantic_latent_dim=256`
  - `semantic_grid_access=1`
  - `semantic_query_derivation=0`
- losses:
  - `loss_vqa` only for the semantic path
  - `semantic_recon_w=0`
  - `semantic_consistency_w=0`
  - `semantic_format_loss_weight=0`
- freeze recipe:
  - `freeze_mode=bridge_plus_top_lm`
  - `train_top_lm_layers=2`
  - LM visual adapters enabled, `3` layers
- dataloader:
  - `batch_size=96`
  - `grad_accum_steps=2`
  - effective batch `192`
  - `eval_batch_size=128`
  - `num_workers=4`
  - `prefetch_factor=1`
  - `pin_memory=0`
- schedule:
  - `max_steps=9000`
  - `lr=2e-4`
  - cosine
  - warmup `600`
  - `lr_min_ratio=0.15`

Parameter breakdown from the same log:

| Component | Total | Trainable |
|---|---:|---:|
| Vision | 92,884,224 | 0 |
| Bridge | 24,338,617 | 24,338,617 |
| Semantic bottleneck | 3,424,441 | 3,424,441 |
| LM | 39,859,712 | 13,588,992 |
| LM visual adapters | 6,314,496 | 6,314,496 |
| Total | 163,927,225 | 44,772,281 |

Observed bridge-stage throughput:
- early steady-state after restart: about `2.8-3.1 steps/s`
- final phase remained around `2.8-3.0 steps/s`
- `grid_attn` settled around `0.63-0.65`, so the co-trained bottleneck really was using the direct-grid path

### Bridge mini-eval curve

| Step | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| 1000 | 0.4416 | 0.6313 | 0.3141 | 0.3341 |
| 2000 | 0.4789 | 0.6604 | 0.3481 | 0.3786 |
| 3000 | 0.5108 | 0.6692 | 0.3723 | 0.4298 |
| 4000 | 0.5243 | 0.6749 | 0.3875 | 0.4487 |
| 5000 | 0.5494 | 0.6925 | 0.3999 | 0.4829 |
| 6000 | 0.5599 | 0.7037 | 0.4093 | 0.4931 |
| 7000 | 0.5695 | 0.7139 | 0.4023 | 0.5068 |
| 8000 | 0.5763 | 0.7171 | 0.4133 | 0.5152 |
| 9000 | 0.5842 | 0.7280 | 0.4175 | 0.5220 |

### Bridge full eval

Artifact:
- [bridge_step_9000_full.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/bridge_step_9000_full.json)

Full VQAv2 val:

| Condition | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| Cement anchor bridge | 0.6163 | 0.7589 | 0.4573 | 0.5499 |
| Variant A co-trained bridge | 0.5888 | 0.7223 | 0.4337 | 0.5284 |

Bridge-stage delta vs Cement:
- overall: `-0.0275`
- yes/no: `-0.0366`
- number: `-0.0236`
- other: `-0.0215`

So the bridge already underperformed before stage-2 compression started.

## Stage 2: Compression On Top Of The Co-Trained Bridge

Stage-2 run:
- [mmgrid_cotrain_v1_20260327_194350_compression](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350_compression)

This stage honored the user correction: the bottleneck was **not** reinitialized. It was loaded from the bridge `step_9000` checkpoint through `--init_from_mm_checkpoint`.

Resolved runtime config from [logfile.txt](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350_compression/logfile.txt):

- init checkpoint:
  - [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350_bridge/step_9000.tar)
- freeze recipe:
  - `freeze_mode=semantic_bottleneck_only`
  - perceiver frozen
  - LM frozen
  - adapters disabled in forward pass
- bottleneck kept as Variant A:
  - `semantic_bottleneck=1`
  - `semantic_tokens=8`
  - `semantic_latent_dim=256`
  - `semantic_grid_access=1`
- remap teacher:
  - `prefix_remap_checkpoint=logs/mmsemantic_remap_v1_debug/step_500.tar`
  - loaded as frozen teacher, not applied in forward
- losses:
  - `L_vqa`
  - `0.1 * L_distill`
  - `L_format` from `0.3 -> 0.0`
  - no consistency loss
- dataloader:
  - `batch_size=96`
  - `grad_accum_steps=2`
  - effective batch `192`
  - `eval_batch_size=96`
  - `num_workers=2`
  - `prefetch_factor=1`
  - `pin_memory=0`
- schedule:
  - `max_steps=3000`
  - `lr=2e-4`
  - cosine
  - warmup `200`
  - format anneal `2250 -> 3000`

Parameter breakdown:

| Component | Total | Trainable |
|---|---:|---:|
| Vision | 92,884,224 | 0 |
| Bridge | 24,338,617 | 3,424,441 |
| Semantic bottleneck | 3,424,441 | 3,424,441 |
| Prefix remap | 262,656 | 0 |
| LM | 39,859,712 | 0 |
| LM visual adapters | 6,314,496 | 0 |
| Total | 164,189,881 | 3,424,441 |

Observed compression-stage throughput:
- training mostly `2.2-3.1 steps/s`
- periodic eval about `3.5-3.7 steps/s`
- `grid_attn` dropped from bridge-stage `~0.64` to compression-stage `~0.33-0.35`

That attention shift is important. Once the perceiver and LM were frozen and adapters removed, the bottleneck did not continue to lean heavily on raw-grid retrieval.

### Compression periodic evals

Artifacts:
- [compression_periodic_step_500.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/compression_periodic_step_500.json)
- [compression_periodic_step_1000.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/compression_periodic_step_1000.json)
- [compression_periodic_step_1500.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/compression_periodic_step_1500.json)
- [compression_periodic_step_2000.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/compression_periodic_step_2000.json)
- [compression_periodic_step_2500.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/compression_periodic_step_2500.json)
- [compression_periodic_step_3000.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/compression_periodic_step_3000.json)

| Step | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| 500 | 0.5318 | 0.6978 | 0.3907 | 0.4469 |
| 1000 | 0.5388 | 0.7009 | 0.4094 | 0.4536 |
| 1500 | 0.5399 | 0.6938 | 0.3936 | 0.4653 |
| 2000 | 0.5502 | 0.7061 | 0.4066 | 0.4735 |
| 2500 | 0.5502 | 0.7021 | 0.4112 | 0.4752 |
| 3000 | 0.5524 | 0.6999 | 0.4109 | 0.4814 |

Mini-eval best checkpoint:
- `step_3000`

### Compression full eval

Artifact:
- [compression_best_full.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/compression_best_full.json)

Full VQAv2 val:

| Condition | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| SigLIP K=8 format-aligned | 0.5900 | 0.7393 | 0.4499 | 0.5134 |
| SigLIP Variant A K=8, frozen perceiver | 0.5951 | 0.7499 | 0.4505 | 0.5157 |
| Variant A co-trained bridge -> compression | 0.5587 | 0.7001 | 0.4252 | 0.4865 |

Compression deltas:
- vs standard SigLIP K=8: `-0.0313`
- vs frozen-perceiver Variant A: `-0.0364`
- vs its own bridge stage: `-0.0301`

### Tiny-head probe

Artifact:
- [compression_probe.json](/home/wdree/percy/vqafromscratch/logs/mmgrid_cotrain_v1_20260327_194350/compression_probe.json)

Probe setup:
- checkpoint: `step_3000`
- feature pool: `flatten`
- train subset: `9999`
- val subset: `4319`
- epochs: `10`

Best probe:

| Condition | Probe Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| SigLIP K=8 format-aligned | 0.5103 | 0.6273 | 0.3511 | 0.4316 |
| SigLIP Variant A K=8, frozen perceiver | 0.5117 | not logged here | not logged here | not logged here |
| Variant A co-trained bridge -> compression | 0.4452 | 0.5564 | 0.2874 | 0.3813 |

This is a large probe drop. The compressed tokens were not just less LM-compatible; they were less semantically linearly decodable too.

## Interpretation

This result argues against the idea that Variant A should be introduced at bridge stage and co-trained end-to-end.

The main points:

1. The bridge underperformed before compression.
   Cement bridge training with the co-trained Variant A bottleneck topped out at `0.5888`, so the problem is not only in stage 2.

2. Carrying the bottleneck weights into stage 2 did not rescue the system.
   The stage-2 run respected the “do not re-init” instruction, but that path converged to `0.5587`, not toward the strong frozen-perceiver Variant A result.

3. The compressed tokens were weaker on both full eval and probe.
   That combination matters. If the probe had improved while the LM score fell, the story could have been “better tokens, worse frozen LM consumption.” That did not happen here.

4. Direct-grid access was being used, but not productively enough.
   Bridge-stage `grid_attn` was about `0.64`. Compression-stage `grid_attn` fell to about `0.34`. So the system did not collapse into ignoring the extra path entirely, but it also did not turn that extra access into stronger compressed semantics.

My read is that Variant A works as a late retrieval patch precisely because it is bolted onto a strong existing perceiver. When the same mechanism is made part of the bridge training objective from step zero, it appears to disrupt the cleaner evidence-filtering role that the perceiver normally provides.

## Bottom Line

This line should not replace the current SigLIP bridge recipe.

Recommended status:
- keep the frozen-perceiver Variant A result as an interesting late-stage bottleneck trick
- do not promote co-trained Variant A as a new bridge default
- if this direction is revisited, the next defensible variant would be:
  - co-trained bridge with Variant A but **fresh** stage-2 bottleneck reinit
  - or a weaker grid-access path that only turns on after the perceiver has already stabilized

As executed here, the end-to-end co-trained Variant A system is clearly below the current frontier.
