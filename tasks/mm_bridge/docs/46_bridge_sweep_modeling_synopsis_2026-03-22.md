# 46 Bridge Sweep Modeling Synopsis (2026-03-22)

## Purpose

This note summarizes the bridge line from the late fixed-VM bridge sweeps to the current Cement frontier, with emphasis on modeling tradeoffs rather than raw chronology.

The question is:

- what architectural ideas actually paid rent?
- what ideas looked plausible but were mostly complexity?
- why did the frontier move when it moved?

## Executive Summary

The project’s best path was not:

- bigger bridge
- more routing tricks
- more role structure
- more aggressive token pruning

It was:

1. stabilize question-conditioned evidence extraction
2. improve query formation
3. improve the vision model
4. preserve more visual evidence for the perceiver
5. clean up the bridge conditioning context

That sequence produced the current frontier:

- High-Entropy: `0.4568`
- Hammer: `0.4608`
- Nail: `0.4653`
- Plank: `0.5240`
- Crane: `0.5762`
- Cement: `0.6129` mean final full eval, `0.6163` best completed single run

The architectural moral is that the current system is best understood as a **visual evidence retrieval and compression stack**. Once that became clear, the successful changes were the ones that improved either:

- the quality of the evidence
- the quality of the query
- or the amount of evidence that survived long enough to matter

## Core Scaffold

The stable scaffold that survived all these sweeps is:

```text
Image
  -> Frozen Vision Tower
  -> Visual Token Grid
  -> Question-Conditioned Perceiver Bridge
  -> Calibrated Visual Prefix
  -> LM (+ top-layer visual adapters)
  -> Answer
```

The current Cement frontier family instantiates that as:

```text
SigLIP-B/16
  -> 49 patch tokens
  -> attnqquery perceiver bridge (depth 3)
  -> prefix calibration
  -> top-3 LM visual adapters
  -> VQA decoding
```

This is important because the system is **not** a unified multimodal transformer. It is a frozen-VM / bridge / partially-trainable-LM stack. So every improvement had to work through that interface.

## Frontier Table

| Sweep | Best Run | Best Score | Main Architectural Lesson |
|---|---|---:|---|
| High-Entropy | `safeqcond_earlylayer_geomcal_frontier` | `0.4568` | stable question conditioning + calibration stack |
| Hammer | `qquery_dynbudget_adapter_earlylayer_geomcal` | `0.4608` | LM-side visual reuse matters |
| Nail | `lmmeanqquery_dynbudget_adapter_d3_cap64` | `0.4653` | better query formation beats bigger budget |
| Plank | `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` | `0.5240` | better VM changes the entire score regime |
| Crane | `dinov2s_attnqquery_nodynbudget_adapter_d3` | `0.5762` | dense visual memory beats hard pruning |
| Cement | `question_only` SigLIP family | `0.6129` mean final full eval / `0.6163` best completed single run | `question_only` is the cleanest bridge context |

## Phase 1: Stabilizing The Bridge Interface

High-Entropy was still mostly a bridge-internal sweep. The best result:

- `safeqcond_earlylayer_geomcal_frontier` -> `0.4568`

What it established:

- leakage-safe question conditioning was directionally correct
- early-layer interaction and calibration stacked positively
- more ornate alternatives like structured roles and multiscale did not dominate

Why this makes sense:

- a frozen LM is highly sensitive to the geometry of the incoming prefix
- if the bridge produces unstable or poorly scaled prefixes, the LM never gets a clean chance to use the image

Diagram:

```text
Question text
  -> pooled conditioning
  -> bridge extraction
  -> calibrated prefix
  -> LM
```

The key modeling lesson was simple:

- before a bridge becomes clever, it has to become legible to the frozen LM

That is why calibration and safe conditioning beat more speculative structure at this stage.

## Phase 2: Query Quality Matters More Than Bridge Size

Hammer and Nail shifted attention from “bridge family” to “query mechanism.”

Hammer result:

- `qquery_dynbudget_adapter_earlylayer_geomcal` -> `0.4608`

Important Hammer facts:

- all top 4 runs used LM visual adapters
- the best bridge-only variant was not enough by itself
- `qquery + dynbudget` stacked, but LM-side access mattered more

Nail result:

- `lmmeanqquery_dynbudget_adapter_d3_cap64` -> `0.4653`

What Nail proved:

- stronger qquery formation produced the cleanest gain
- adapter depth helped only a little
- larger token budgets did not help
- role specialization did not help

This is a classic retrieval story:

```text
weak query + richer machinery  <  stronger query + moderate machinery
```

The query path evolved like this:

```text
safeqcond
  -> qquery
  -> lmmeanqquery
  -> attnqquery
```

The key modeling interpretation was:

- the bridge was not primarily starved for parameters
- it was starved for **good retrieval requests**

That is why budget scaling (`cap64 -> cap96`) did nothing. More slots do not help if the wrong evidence is being requested.

## Phase 3: VM Quality Dominates Bridge Complexity

Plank was the first sweep that made the bridge look secondary to the vision model.

Best old-VM Plank run:

- `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64` -> `0.4699`

Best MobileViT Plank run:

- `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` -> `0.5240`

That is roughly a five-point jump from a drop-in VM swap.

Why this is fundamental:

- the bridge can only compress and route the evidence it receives
- if the vision tokens get semantically cleaner, the same bridge can suddenly support better answers

Plank also changed the qquery ordering:

- weaker VM regime: `lmmeanqquery` strongest
- stronger VM regime: `attnqquery` strongest

This suggests an important interaction:

- when visual tokens are weak, a strong pooled query summary helps stabilize extraction
- when visual tokens are richer, token-level attention-derived querying becomes more valuable

So the “best query” is not global. It depends on the quality and richness of the VM token space.

## Phase 4: Dense Visual Memory Beats Hard Pruning

Crane was the biggest conceptual leap.

Key results:

- `mobileclip_attnqquery_dynbudget_adapter_d3_cap64` -> `0.5603`
- `dinov2s_attnqquery_dynbudget_adapter_d3_cap64` -> `0.5323`
- `dinov2s_attnqquery_nodynbudget_adapter_d3` -> `0.5762`

The decisive comparison was:

```text
DINOv2 + dynbudget     -> 0.5323
DINOv2 + nodynbudget   -> 0.5762
delta                  -> +0.0439
```

This was not a small tuning effect. It changed the modeling story.

Why dynbudget failed here:

```text
Dense visual token grid
  -> hard pre-pruning
  -> perceiver
```

But the perceiver itself is already an evidence-selection mechanism. Hard pruning before the perceiver:

- deletes tokens before relevance has been computed
- hurts most on `other`, where distributed evidence matters
- turns a soft attention bottleneck into an information bottleneck

Crane also clarified the VM tradeoff:

- MobileCLIP showed that language-aligned features help at fixed token count
- DINOv2 showed that token bandwidth can matter even more if the bridge is allowed to keep it

So the project’s bottleneck had shifted again:

- from query quality
- to **visual evidence retention**

## Phase 5: Cement Cleaned Up The Champion Family

Cement was intentionally narrower. It asked one question:

- in the stabilized SigLIP champion family, should the bridge condition on `prompt_only` or `question_only`?

Result:

- `question_only` beat `prompt_only` at all 3 matched seeds
- mean final full eval: `0.6129`
- mean best-checkpoint peak: `0.6174`
- best completed single run: `0.6163` at `s42 step_9000`
- higher periodic mini-eval peak: `0.6203` at `s53 step_8000`, which should not be treated as the final benchmark anchor

Why `question_only` wins is easy to justify:

- the bridge should condition on the semantically relevant span
- prompt wrappers and answer stubs are noise for retrieval
- fixed question-only conditioning is cleaner and cheaper than answer-conditioned autoregressive retrieval

Diagram:

```text
prompt_only:
  "Question: ... Answer:"
    -> query context

question_only:
  actual question span only
    -> query context
```

That small contextual cleanup became worth several tenths of a point once the rest of the stack was already strong.

## Why The Current Champion Looks The Way It Does

Current champion:

- frozen SigLIP-B/16
- `attnqquery`
- `question_only`
- perceiver bridge depth `3`
- no dynbudget
- top-3 LM visual adapters
- full-eval reference `0.6129` mean, `0.6163` best completed single run

This is a coherent system:

- strong language-aligned VM
- no early hard evidence deletion
- attention-based question-conditioned querying
- moderate bridge depth
- moderate LM-side multimodal reuse

It is also notable for what it does **not** rely on:

- no VM finetuning
- no role specialization
- no multistage query loops
- no broader architectural rewrite

That restraint is not accidental. The sweeps repeatedly showed that the next bottleneck was usually at a more basic level than the most exotic proposed change.

## Behavioral Read Of The Champion

The Cement diagnostics matter because they show what kind of model we actually built.

Strong facts:

- visual utilization is real: diagnostics clean `0.6174` vs zero-image `0.3146`, while the completed `s42 step_9000` full eval is `0.6163`
- the bridge is most valuable on `number` and `other`
- the model remains overconfident but reasonably calibratable (`ECE ≈ 0.049`)

Weaknesses remain:

- OCR-like categories are still weak
- “why” reasoning is still weak
- temporal questions are weak
- yes/no still shows a strong language prior

These weaknesses line up with fundamentals:

- frozen image-text VMs are not OCR-specialists
- shallow prefix-conditioning is not deep symbolic reasoning
- a single-frame bridge stack has no temporal semantics

So the current frontier is not “how do we get a little more from bridge choreography?”
It is more likely:

- stronger task-aligned visual semantics on hard categories
- or deeper multimodal reasoning without reintroducing the evidence-pruning mistakes Crane exposed

## Condensed Architecture Story

```text
High-Entropy:
  stable bridge conditioning
  -> 0.4568

Hammer / Nail:
  better query formation + LM-side reuse
  -> 0.4653

Plank:
  better VM
  -> 0.5240

Crane:
  keep dense evidence, stop pruning early
  -> 0.5762

Cement:
  clean up bridge context in the best family
  -> 0.6129 mean final full eval
```

## Bottom Line

From a modeling standpoint, the work so far supports one durable thesis:

- VQA performance in this frozen-VM / bridge / partially-trainable-LM setup improves when the LM asks a clean question of a strong visual token set, the bridge preserves evidence long enough for soft selection to work, and the LM gets a modest but real chance to revisit that evidence in-layer.

Everything that violated that thesis tended to flatten out:

- bigger caps without better queries
- role structure without stronger evidence
- hard pruning before the perceiver
- prompt clutter in the bridge context

Everything that aligned with that thesis moved the frontier:

- better conditioning
- better query formation
- better VM quality
- more evidence retention
- cleaner question-only retrieval context

That is the architectural spine that produced the current champion.
