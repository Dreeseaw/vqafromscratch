# 36codex Crane Sweep Report - 2026-03-15

## Scope

This is my read of the completed Crane results so far, written after reviewing:

- `tasks/mm_bridge/docs/34_crane_extended_sweep_plan_2026-03-14.md`
- `tasks/mm_bridge/docs/36_crane_part1_sweep_report_2026-03-15.md`
- the authoritative Crane bundles under `logs/mmcrane_v1_*`
- the per-run logs for the completed MobileCLIP, DINOv2, and caption-align runs

This document is not meant to replace the existing part-1 report. It is the more opinionated interpretation:

- what the runs actually taught us
- which conclusions I would now treat as real
- what I think the next runs should be
- what Crane means for the larger trajectory of the project

## Executive Read

Crane changed the project again, and more sharply than I expected.

The headline is not just that the frontier moved from `0.5240` to `0.5762`.

It is that the move came from a very specific architectural lesson:

- this project was still bottlenecked more by visual evidence bandwidth than by bridge cleverness or LM capacity

The winning run:

- `dinov2s_attnqquery_nodynbudget_adapter_d3` -> `0.5762`

beats:

- the Plank frontier by `+0.0522`
- MobileCLIP by `+0.0159`
- DINOv2 with dynbudget by `+0.0439`

So the most important Crane result is not "DINOv2 is best."

It is:

- when the VM exposes a large dense token grid, the perceiver wants to see all of it
- hard pre-selection is actively harmful

That is a deeper conclusion than any single run name.

## What I Think Crane Settled

### 1. `attnqquery` is now the default query path unless proven otherwise

This now held across qualitatively different VMs:

- MobileViT in Plank
- DINOv2 in Crane

On DINOv2 with dynbudget:

- `attnqquery`: `0.5323`
- `lmmeanqquery`: `0.5248`

That is not a tiny edge. It is enough to stop treating `lmmeanqquery` as co-frontier.

My read:

- `lmmeanqquery` was the right bridge improvement for the weaker visual regime
- `attnqquery` is the right bridge improvement once the visual tokens become richer and more numerous

So for current work, I would treat:

- `attnqquery`

as the live default, and:

- `lmmeanqquery`

as a useful control rather than a mainline.

### 2. Dynbudget is not a universal good; it is specifically wrong for high-token perceiver setups

This is the cleanest mechanistic conclusion in Crane.

DINOv2 results:

- nodynbudget: `0.5762`
- cap128: `0.5311`
- cap64: `0.5323`
- cap32: `0.5160`

That is a brutal result.

The reason is coherent:

- the perceiver already performs soft evidence selection through cross-attention
- dynbudget adds hard evidence deletion before the perceiver can decide what matters

With a 256-token visual grid, this becomes destructive rather than efficient.

The category breakdown supports that story:

- yes/no: `+0.0300` for nodynbudget over cap64
- number: `+0.0236`
- other: `+0.0599`

The biggest gain is on `other`, which is exactly where over-pruning diverse evidence should hurt most.

My conclusion:

- dynbudget should be considered a low-token helper, not a general bridge principle
- for dense-token VMs feeding a perceiver, default to `nodynbudget`

### 3. MobileCLIP proved that language alignment helps, but Crane also showed it is not the first-order term

At roughly matched low token count:

- MobileCLIP `49` tokens: `0.5603`
- MobileViT `49` tokens: `0.5240`

That is a big gain. So yes:

- language-aligned visual features matter

But the DINOv2 result is the more important correction:

- DINOv2 has no language alignment
- DINOv2 with 256 tokens and no hard filtering still wins at `0.5762`

So the actual hierarchy appears to be:

1. enough dense visual evidence
2. then per-token semantic efficiency / language alignment

I would phrase it this way:

- MobileCLIP showed that better semantic priors help when token bandwidth is fixed
- DINOv2 showed that evidence bandwidth can dominate semantic alignment when the bridge is allowed to use it

That is a strong systems-level result, not just a VM bakeoff.

### 4. The current LM was not "maxed out" in the way the project had feared

This is one of the most important grand-scheme consequences.

Same LM family, same general bridge family, but:

- Plank frontier: `0.5240`
- Crane frontier: `0.5762`

That is a `+0.0522` jump without replacing the LM.

So the prior story:

- "the LM is probably the next hard ceiling"

was too early.

Crane showed that the LM still had a lot of unused headroom once the visual side stopped starving it.

That does not mean LM scaling is irrelevant. It means:

- the bridge/VM interface was still hiding a major chunk of usable performance

I would now say:

- the current LM is no longer the first bottleneck
- but it will probably become the next serious bottleneck somewhere in the high `0.58` to low `0.60` range

### 5. Adapter depth is weak, at least in the dynbudget regime

The d4/d5 story was underwhelming:

- MobileCLIP d3: `0.5603`
- MobileCLIP d4: `0.5578`
- DINOv2 dyn d3: `0.5323`
- DINOv2 dyn d5: `0.5338`

That is basically flat.

My read is not that deeper LM-side visual reasoning is dead.

It is:

- when the bridge is already losing evidence through dynbudget, extra LM-side depth cannot rescue much

So I would not carry forward:

- more adapter depth under dynbudget

as a serious axis.

The only depth test I still care about is:

- deeper adapters on DINOv2 nodynbudget

because that is the first regime where enough visual information actually reaches the LM-facing side.

### 6. Caption-align is not refuted, but it is nowhere near the most urgent lever

I agree with the part-1 report that the executed caption-align runs are not clean comparators.

The step-count and LR-schedule issue means the final numbers are contaminated.

The early matched-step deltas do suggest:

- a real early convergence benefit

But even if that positive signal is real, it is small relative to what Crane already found:

- caption-align early benefit: around `+0.01` at matched early steps
- nodynbudget benefit on DINOv2: `+0.0439`

So my position is:

- caption-align remains alive as a medium-priority cleanup experiment
- but it should not displace the dense-token DINOv2 mainline

## What I Think the Data Really Means

### The project is shifting from "better querying" to "better visual memory"

Hammer and Nail taught:

- better querying mattered

Plank taught:

- better VMs mattered even more than expected

Crane now adds:

- once the VM gives the bridge a dense enough token grid, the right move is often to stop deleting tokens and let the perceiver do its job

So the project story is no longer primarily:

- "what question-conditioned query should we form?"

It is now:

- "how much visual evidence should the LM-facing stack be allowed to retain and revisit?"

That is a more consequential shift.

I think Crane is the first sweep that makes the project feel less like "bridge tinkering" and more like a real multimodal memory design problem.

### The ideal next VM is probably "MobileCLIP semantics with DINOv2 token richness"

Crane gives a clear conceptual target:

- MobileCLIP wins at low token count because its tokens are more language-compatible
- DINOv2 wins overall because it provides far more dense evidence

So the ideal frontier VM for this project is not either one alone. It is:

- a language-aligned model that exposes a large dense token grid

That points naturally toward:

- CLIP / SigLIP / similar VMs with 196+ patch tokens

more than toward additional small mobile backbones.

I would treat that as the most important VM-side strategic implication of Crane.

### The perceiver resampler itself looks stronger than dynbudget gave it credit for

A subtle but important point:

- the winning Crane run is still a perceiver resampler setup

The big change was not abandoning the perceiver. It was:

- letting it cross-attend over all 256 DINOv2 tokens

That means the perceiver family did not fail.

What failed was:

- overconstraining the perceiver with pre-filtered evidence

So I would not interpret Crane as "throw away the bridge."

I would interpret it as:

- the bridge was better than we thought once we stopped starving it

## Runs I Would Do Next

If the goal is to keep momentum and answer the highest-value remaining questions with minimal new engineering, this is the order I would use.

### 1. `dinov2s_attnqquery_nodynbudget_adapter_d3_seed2`

Why first:

- `0.5762` is now the project headline
- it needs a second seed before too much architecture work is built on top of it

This is not glamorous, but it is the right calibration move.

### 2. `dinov2s_questiononly_attnqquery_nodynbudget_adapter_d3`

Why second:

- question-only already showed a mild gain on the dynbudget DINOv2 path
- the question-only idea is cheap
- it is exactly the sort of query refinement that might still matter once the dense-token baseline is stabilized

This is the cleanest low-risk chance to beat `0.5762`.

### 3. `dinov2s_attnqquery_nodynbudget_adapter_d3_18k`

Why third:

- the nodynbudget frontier looked healthy through 9k
- longer training is the cheapest remaining way to test whether the current family has another real slope

I would do this before broader new-family exploration.

### 4. `dinov2s_attnqquery_nodynbudget_adapter_d4`

Why fourth:

- adapter depth was flat under dynbudget
- but nodynbudget is the first place where enough dense evidence reaches the LM to make a depth test meaningful

This is a much better depth test than the ones already run.

## Runs I Would Deprioritize

### Corrected caption-align

Worth doing later, not now.

Reason:

- real but smaller signal
- requires cleanup
- not competitive with the immediate nodynbudget mainline questions

### More dynbudget cap sweeps

Not worth it.

Crane already answered that question hard enough.

### More MobileViT Tier-1 cleanup

Also not worth centering now.

MobileViT is no longer the frontier VM, and Crane already gave a more important direction.

## Grand-Scheme Interpretation

Crane meaningfully changes the grand strategy of the project.

### Before Crane

The project looked like:

- improve bridge querying
- maybe deepen LM adapters
- maybe pretrain the bridge later

### After Crane

The better strategy looks like:

1. dense high-token VMs are now a first-class lever
2. attnqquery is the default bridge-side query mechanism
3. pre-filtering tokens before a perceiver is often the wrong abstraction
4. the next serious frontier is dense visual memory, not more evidence pruning

That is a cleaner and more promising picture.

### My honest view on the BLIP-2 gap after Crane

Before Crane, the gap to `65.2` felt like a different world.

After Crane, it still feels large, but no longer absurd.

The remaining gap from `0.5762` is about:

- `0.652 - 0.5762 = 0.0758`

That is still too large for minor sweeps, but it is now in a range where a few real phase changes could matter:

- dense language-aligned VM tokens
- stronger LM-side multimodal memory
- possibly corrected pretraining or stronger LM priors

So I would say:

- Crane did not get the project to BLIP-2 territory
- but it did show a plausible route toward it

That route is not:

- more cap tuning
- more role structure
- more dynbudget cleverness

It is:

- better dense-token vision backbones
- keeping that dense evidence alive through the bridge
- then only afterward increasing LM-side multimodal depth or LM strength

## One-Line Summary

Crane showed that the project’s next real frontier is dense visual memory rather than more aggressive visual pruning: `attnqquery` is now the default bridge query, DINOv2 without dynbudget reset the frontier to `0.5762`, MobileCLIP proved language alignment matters at fixed token count, and the most important next steps are seed-checking and extending the DINOv2 nodynbudget line before spending more energy on caption pretraining or additional bridge-side tricks.
