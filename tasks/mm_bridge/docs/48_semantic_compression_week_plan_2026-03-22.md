# Semantic Compression Week Plan

## Goal

Test the "late semantic bottleneck" thesis against the completed Cement full-eval reference without changing the winning evidence-retrieval scaffold.

Fixed anchor:
- checkpoint: `logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar`
- single-run full-eval score: `0.6163`

Important Cement note:
- `0.6203` at `s53 step_8000` was a **periodic mini-eval peak**, not the completed full-val benchmark
- the completed full-eval single-run reference is `s42 step_9000 = 0.6163`
- the 3-seed final full-eval question-only mean is `0.6129`
- the 3-seed best-checkpoint peak mean is `0.6174`
- exported prefix before this experiment: `49 x D`

The comparison question is narrow:

`49 dense evidence latents -> K exported semantic latents -> LM`

How much answer-relevant information survives when only the exported LM-facing token budget is reduced?

## Architecture

Baseline Cement path:

```text
SigLIP-B/16 (frozen)
  -> dense visual tokens
  -> perceiver_resampler depth 3
  -> 49 evidence latents
  -> prefix calibration
  -> LM + visual adapters
```

Compression path for this sweep:

```text
SigLIP-B/16 (frozen)
  -> dense visual tokens
  -> perceiver_resampler depth 3
  -> 49 evidence latents
  -> semantic bottleneck:
       K learned semantic queries cross-attend over 49 evidence latents
       -> K semantic export tokens for LM
       -> linear token-axis decoder reconstructs 49-token evidence target
  -> prefix calibration
  -> LM visual adapters only
```

Important boundary:
- no early pruning before the perceiver
- no dynbudget
- no VM retraining
- no perceiver retraining
- no top-LM retraining in the main compression runs

## Training Design

Each compression run:
- initializes from the Cement full-eval reference weights
- uses the same Cement bridge family and SigLIP VM
- enables the semantic bottleneck after the perceiver
- freezes everything except:
  - semantic bottleneck module
  - LM visual adapters
- uses the Cement full-eval reference as a frozen teacher

Loss:
- primary: existing VQA cross-entropy
- auxiliary: semantic reconstruction + consistency against the frozen teacher's **49 pre-bottleneck evidence latents**

Distillation target:
- target source: raw teacher perceiver evidence latents from the Cement full-eval anchor
- reconstruction head: token-axis linear decoder from `K -> 49`
- aux weights: `semantic_recon_loss_weight=0.1`, `semantic_consistency_loss_weight=0.1`

This isolates compression quality instead of letting the perceiver reconfigure itself around the new budget.

## Base Sweep

Pilot first:

1. `K=16`

Reason:
- this is the most likely near-lossless compression point
- it should reveal convergence behavior quickly
- there is no reason to commit GPU time to all four budgets before seeing whether the new module stabilizes fast or drifts

If the pilot is healthy, expand to:

2. `K=32`
3. `K=8`
4. `K=4`

Common settings:
- init checkpoint: Cement full-eval reference `s42 step_9000`
- teacher checkpoint: same Cement full-eval reference
- seed: `53`
- SigLIP-B/16 frozen
- `question_only`
- `question_hidden_attn`
- `perceiver_resampler`
- `bridge_query_depth=3`
- no dynbudget / no token selector
- LM visual adapters: depth `3`
- freeze mode: `semantic_adapter_only`
- max steps: `4000`
- periodic mini-eval every `500`
- checkpoint every `500`

Why `4000`, not `9000`:
- the trainable delta is much smaller than Cement
- the new module is learning compression of already-good 49-token evidence, not extraction from raw visual tokens
- the LM adapters are warm-started and only need to re-align to a new exported-token distribution
- peak/terminal divergence is already known to be real in this codebase, so shorter runs with denser eval are more informative than another blind 9k default

Prepared launcher:
- [launch_semantic_compression_sweep_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_semantic_compression_sweep_v1.sh)

Launch command:

```bash
./tasks/mm_bridge/scripts/launch_semantic_compression_sweep_v1.sh
```

Full sweep after the pilot:

```bash
RUN_FULL_SWEEP=1 ./tasks/mm_bridge/scripts/launch_semantic_compression_sweep_v1.sh
```

## Analysis Order

### 1. Primary metric: VQAv2 val accuracy vs K

Target readout:
- overall VQAv2 val accuracy
- `yes/no`, `number`, `other`
- degradation curve from `49 -> 32 -> 16 -> 8 -> 4`

Interpretation:
- flat or mild drop through `K=16` means the LM-facing bridge bandwidth is redundant
- sharp collapse from `16 -> 8` means the exported token budget is now the real bottleneck
- category-specific collapse first in `other` or `number` suggests semantic compression hurts compositional or counting-heavy evidence before simpler binary decisions

### 2. Tiny-head probe on exported semantic tokens

Question:
- how much answer information is already present in the exported tokens before LM reasoning?

Prepared utility:
- [mm_semantic_probe.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_probe.py)

Example:

```bash
./.venv_local/bin/python -m tasks.mm_bridge.scripts.mm_semantic_probe \
  --checkpoint logs/<run_id>/step_9000.tar \
  --device cuda \
  --limit_train 10000 \
  --limit_val 5000 \
  --answer_top_k 3000 \
  --epochs 10 \
  --output_json logs/<run_id>/semantic_probe.json
```

Readout:
- probe accuracy overall
- probe accuracy by answer type
- compare probe accuracy to full-system accuracy at the same `K`

Interpretation:
- small probe/full-system gap means exported tokens are already semantically strong
- large gap means the LM is still doing most of the work after compression

### 3. LM visual-adapter ablation

Question:
- after compression, is the LM rescuing weak tokens, or are the tokens carrying more of the answer content themselves?

Prepared utility:
- [mm_adapter_ablation.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_adapter_ablation.py)

Example:

```bash
./.venv_local/bin/python -m tasks.mm_bridge.scripts.mm_adapter_ablation \
  --checkpoint logs/<run_id>/step_9000.tar \
  --device cuda \
  --keep_counts 3,2,1,0 \
  --output_json logs/<run_id>/adapter_ablation.json
```

Required comparison:
- run on the Cement full-eval anchor
- run on best compressed checkpoint, expected first target `K=16`

Interpretation:
- if compressed `K=16` is **less** sensitive to adapter removal than the 49-token Cement full-eval anchor, that supports the semantic-bottleneck thesis
- if compressed `K=16` is **more** sensitive, the bottleneck is forcing the LM to reconstruct lost evidence instead of receiving cleaner semantics

### 4. Grounding / spatial binding

Question:
- does compression preserve object-level spatial binding, or does it collapse the evidence into diffuse semantic averages?

Prepared path:
- [mm_grounding_inspection.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_grounding_inspection.py)

Important implementation note:
- when semantic bottleneck is enabled, the grounding script now sees the **semantic query attention over the 49 evidence latents** as the last attention map
- this is the right map for `K=16` / `K=8` grounding inspection

Priority order:
- do this after primary accuracy, probe, and adapter ablation stabilize
- first compare the Cement full-eval anchor vs best compressed run
- then extend to GQA-based grounding slices

## GQA Use

Do **not** mix GQA into round one.

Round one:
- VQAv2-only training for direct comparability to Cement

Round two:
- take the best-performing compressed `K`
- add GQA mix pressure
- re-run:
  - full val score
  - probe
  - adapter ablation
  - grounding

This keeps the first week interpretable:
- architecture effect first
- data effect second

## Success Criteria

Strong success at `K=16` means:
- VQAv2 within about `1` point of the `0.6163` full-eval anchor
- probe accuracy meaningfully above trivial chance and not catastrophically below the full model
- lower adapter-ablation sensitivity than the 49-token Cement full-eval anchor
- grounding still concentrated rather than diffuse

Failure modes and what they mean:
- accuracy drops early but probe stays decent:
  LM integration is the bottleneck, not token semantics
- probe collapses early but full model holds:
  tokens are not semantic; LM is rescuing them
- grounding collapses first:
  compression is breaking spatial binding before answer accuracy fully reveals it
- `number` / `other` degrade first:
  compression is preferentially hurting detail-heavy reasoning

## Implemented Support

Training/runtime changes:
- frozen-teacher semantic distillation target from external MM checkpoint
- new `semantic_adapter_only` freeze mode
- new `init_from_mm_checkpoint` path for new runs initialized from the Cement full-eval anchor
- semantic bottleneck now reconstructs back to the full 49-token target width

Relevant files:
- [train/mm.py](/home/wdree/percy/vqafromscratch/train/mm.py)
- [models/bridge.py](/home/wdree/percy/vqafromscratch/models/bridge.py)
- [mm_semantic_probe.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_probe.py)
- [mm_adapter_ablation.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_adapter_ablation.py)
- [mm_grounding_inspection.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_grounding_inspection.py)
- [launch_semantic_compression_sweep_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_semantic_compression_sweep_v1.sh)
