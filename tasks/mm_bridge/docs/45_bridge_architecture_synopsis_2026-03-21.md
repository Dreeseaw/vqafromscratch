# 45 Bridge Architecture Synopsis (2026-03-21)

## Purpose

This note summarizes the bridge line from the late fixed-VM bridge sweeps through the current Cement frontier, with emphasis on:

- what changed architecturally
- why those changes helped or failed
- what the score deltas mean from a modeling standpoint
- how the project’s bottleneck moved over time

This is not a full experiment log. It is the architectural story of how the current best system emerged.

## One-Screen Thesis

The central arc of the project was:

1. stabilize the bridge/question interface
2. improve how the LM asks for visual evidence
3. improve the quality of the visual evidence itself
4. stop deleting visual evidence too early
5. settle on the cleanest question-conditioned bridge context

The current best system was not produced by one giant invention. It was produced by repeatedly finding the next dominant bottleneck and refusing to overcomplicate the wrong layer.

## Frontier Progression

| Sweep | Best Run | Best Score | Main Lesson |
|---|---|---:|---|
| High-Entropy | `safeqcond_earlylayer_geomcal_frontier` | `0.4568` | safe question conditioning and calibration stack cleanly |
| Hammer | `qquery_dynbudget_adapter_earlylayer_geomcal` | `0.4608` | LM adapters matter more than bridge-only tweaks |
| Nail | `lmmeanqquery_dynbudget_adapter_d3_cap64` | `0.4653` | query quality matters more than bigger bridge budget |
| Plank | `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` | `0.5240` | better VM lifts everything more than extra bridge cleverness |
| Crane | `dinov2s_attnqquery_nodynbudget_adapter_d3` | `0.5762` | dense visual memory beats hard token pruning |
| Cement | `question_only` SigLIP full-eval reference | `0.6129` mean final / `0.6163` best completed single run | `question_only` beats `prompt_only` in the stabilized SigLIP family |

The important point is that each frontier shift changed the interpretation of the bottleneck:

- early: bridge conditioning
- middle: query formation
- later: visual representation quality
- latest: visual evidence retention and clean bridge context

## Core Architecture

The stable bridge family that survived this process looks like:

```text
Image
  -> Frozen Vision Model
  -> Visual Token Grid
  -> Perceiver-Style Bridge
       conditioned by question text
  -> Calibrated Visual Prefix
  -> LM
       with top-layer residual visual adapters
  -> Answer tokens
```

Current champion shape:

```text
SigLIP-B/16
  -> 49 visual tokens
  -> attnqquery perceiver bridge (depth 3)
  -> prefix calibration
  -> LM with cross-attn visual adapters in top 3 blocks
  -> VQA decoding
```

Parameterization of the current champion:

```text
VM      :  92.9M total,  0 trainable
Bridge  :  21.4M total, 21.4M trainable
LM side :  46.2M total, 19.9M trainable
Total   : 160.5M total, 41.3M trainable
```

The bridge is therefore not a tiny glue layer anymore. It is a substantial learned interface between a large frozen VM and a partially trainable LM.

## Phase 1: Stabilizing The Question Interface

High-Entropy established the first durable bridge lesson:

- `safeqcond + earlylayer + geomcal` reached `0.4568`
- the best cluster was dominated by `safeqcond`
- more structurally exotic variants like multiscale or structured roles did not win

Why this mattered:

- a bridge must first become a reliable conditional extractor before more elaborate routing matters
- leakage-safe question conditioning was a cleaner inductive bias than looser prompt-conditioned variants
- calibration mattered because the frozen LM is sensitive to prefix geometry, not just prefix content

Architecture intuition:

```text
Question text
  -> pooled conditioning signal
  -> modulates bridge extraction
  -> calibrated prefix distribution
  -> frozen LM can actually use the prefix
```

Modeling tradeoff:

- adding some structure to the bridge was useful
- adding too much structure before the interface was stable was not

This is a classic representation-learning pattern: when the interface distribution is wrong, “more architecture” often loses to “better-conditioned architecture.”

## Phase 2: Query Quality Beats Generic Bridge Growth

Hammer and Nail reframed the project around query formation.

Hammer:

- `qquery_dynbudget_adapter_earlylayer_geomcal` -> `0.4608`
- top 4 Hammer runs all used LM visual adapters
- bridge-only gains were real but smaller than LM-side multimodal reuse

Nail:

- `lmmeanqquery_dynbudget_adapter_d3_cap64` -> `0.4653`
- `qquery_dynbudget_adapter_d3_cap64` -> `0.4617`
- delta from stronger qquery formation: `+0.0036`
- cap `64 -> 96` did nothing
- explicit role specialization lost

The modeling read was:

- the problem was not “the bridge is too small”
- the problem was “the LM is not asking the right visual question”

Query evolution:

```text
Static learned queries
  -> safe question conditioning
  -> qquery
  -> LM-mean qquery
  -> attn-derived qquery
```

This progression follows a coherent principle:

- if the vision side is frozen, the cheapest way to improve evidence extraction is to improve the query, not the memory slot count

This is why Nail killed two tempting but weak directions:

- larger dynbudget caps
- role-specialized token families

Both added machinery without improving the retrieval signal.

## Phase 3: VM Quality Dominates More Than Expected

Plank was the first sweep that made the bridge look secondary to the VM.

Best original-VM Plank run:

- `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64` -> `0.4699`

Best MobileViT Plank run:

- `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` -> `0.5240`

That is roughly a five-point jump from a drop-in VM swap under the same general bridge family.

Why this matters from first principles:

- the bridge cannot recover visual distinctions that are not present in the token grid
- if the VM makes object/attribute/relation evidence cleaner, the same bridge can suddenly look much smarter

Plank also changed the qquery ordering:

- old VM regime: `lmmeanqquery` strongest
- stronger VM regime: `attnqquery` strongest

Interpretation:

- when visual tokens are weak, a strong global summary query helps
- when visual tokens are richer, attention-derived querying can exploit token-level structure better

So the “best query mechanism” is not absolute. It depends on the richness of the visual token set.

## Phase 4: Dense Visual Memory Beats Hard Pruning

Crane was the largest conceptual jump.

Key results:

- `mobileclip_attnqquery_dynbudget_adapter_d3_cap64` -> `0.5603`
- `dinov2s_attnqquery_dynbudget_adapter_d3_cap64` -> `0.5323`
- `dinov2s_attnqquery_nodynbudget_adapter_d3` -> `0.5762`

This sweep settled three major facts:

1. `attnqquery` is the default query path in the strong-VM regime.
2. language alignment helps at fixed token count.
3. dense evidence bandwidth matters even more than language alignment if the bridge can keep it.

The most important comparison was:

```text
DINOv2 + dynbudget     -> 0.5323
DINOv2 + nodynbudget   -> 0.5762
delta                  -> +0.0439
```

That is not a tuning detail. It is a mechanistic result.

Why dynbudget failed in this regime:

```text
Visual grid (256 tokens)
  -> dynbudget
       hard deletes tokens early
  -> perceiver
       can only attend over survivors
```

But the perceiver itself already performs soft evidence selection through attention. So with a rich VM:

- hard pruning throws away potentially useful evidence before the model can judge relevance
- the loss is worst on `other`, where diverse evidence matters most

Crane therefore shifted the project from “better querying” to “better visual memory.”

That is a different modeling problem.

## Phase 5: Cement Settled The Cleanest Champion Family

Cement stopped doing wide exploration and instead asked a narrow, high-value question:

- in the stabilized SigLIP family, should the bridge condition on `prompt_only` or `question_only`?

Result:

- `question_only` beat `prompt_only` at all 3 matched seeds
- mean final full eval: `0.6129` vs `0.6099`
- mean best-checkpoint peak: `0.6174` vs `0.6119`
- best completed single run: `0.6163` at `s42 step_9000`
- higher periodic mini-eval peak: `0.6203` at `s53 step_8000` (not the final benchmark anchor)

Why `question_only` wins is straightforward:

- it removes answer-stub and prompt-wrapper clutter from query formation
- it keeps the bridge conditioned on the most semantically relevant span
- it preserves a fixed, non-autoregressive conditioning path

Diagram:

```text
prompt_only:
  "Question: ... Answer:"
     -> bridge context

question_only:
  actual question span only
     -> bridge context
```

In a system where the bridge is trying to retrieve evidence relevant to a question, `question_only` is the cleaner inductive bias. Cement showed that this cleanliness survives contact with the strongest current family.

## Why The Current Champion Looks The Way It Does

Current champion:

- SigLIP-B/16 VM
- `attnqquery`
- `question_only`
- perceiver bridge depth `3`
- no dynbudget
- LM adapters depth `3`
- full-eval reference `0.6129` mean, `0.6163` best completed single run

This is a balanced system:

- strong semantic VM
- enough bridge capacity to distill visual evidence into a prefix
- enough LM-side visual reuse to exploit the prefix
- no unnecessary hard routing before attention can work

It is also notable for what it does **not** do:

- no VM finetuning
- no role-specialized bridge
- no large cap sweeps
- no iterative retrieval loop
- no large-scale bridge pretraining

That restraint is part of why it works. Each surviving component has now paid rent experimentally.

## Remaining Weaknesses

Even the champion has a clear error profile:

- OCR and reading remain weak
- “why” and causal reasoning remain weak
- temporal questions remain weak
- yes/no still carries a large language prior

These weaknesses line up with modeling fundamentals:

- frozen VMs pretrained for image-text similarity are not OCR specialists
- prefix-conditioning plus shallow adapters is not a deep reasoning engine
- single-frame systems are weak on temporal semantics

So the current frontier is not “how do we get one more point from qquery?”
It is more likely:

- stronger VM/task alignment for hard categories
- or stronger multimodal reasoning depth without throwing away the good parts of the current interface

## Minimal Diagram Of The Full Story

```text
Stable conditioning
  High-Entropy
    -> 0.4568

Better querying + LM-side reuse
  Hammer / Nail
    -> 0.4653

Better vision
  Plank
    -> 0.5240

Better visual memory retention
  Crane
    -> 0.5762

Cleaner question context in the best family
  Cement
    -> 0.6129 mean final full eval
```

## Bottom Line

From a modeling standpoint, the project evolved along the correct hierarchy:

1. fix the bridge interface
2. improve the query signal
3. improve the VM
4. stop deleting evidence too early
5. clean up the conditioning context

The current best system is therefore not an arbitrary winner. It is the product of a progressively sharpened architectural thesis:

- multimodal VQA improves when the LM asks a clean question of a strong visual token set, the bridge preserves rather than prematurely prunes evidence, and the LM gets a small but meaningful chance to revisit that evidence in-layer.

That is the durable lesson carried forward by the sweeps so far.
