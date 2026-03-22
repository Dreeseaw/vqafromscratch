# Nail Sweep Plan - 2026-03-13

## Codename

`nail`

## Purpose

Nail is the first post-Hammer sweep designed to do three things at once:

1. verify that the new `0.4608` mainline is real
2. verify that it is still genuinely image-grounded
3. push the new adapter-centered family upward without falling back into a broad bridge-only architecture zoo

This sweep should not be about:

- more plain bridge-only sightseeing
- more micro-retunes of dropout, norm ratio, or tiny bridge shape knobs
- treating a single `0.4608` run as settled truth before it survives stability and corruption checks

The right Nail cycle is therefore a consolidation-and-scaling sweep, not another open-ended family search.

## Entry State

Current best evidence entering Nail:

- best observed run: `qquery_dynbudget_adapter_earlylayer_geomcal` at `0.4608`
- second-best run: `dynbudget_adapter_earlylayer_geomcal` at `0.4602`
- top 4 Hammer runs all included LM visual adapters
- best bridge-only Hammer result: `qquery_dynbudget_earlylayer_geomcal` at `0.4576`

What Hammer v2 already taught us:

- LM-side visual adapters are the strongest new lever so far
- `qquery` and `dynbudget` are both useful, but mostly become frontier-positive when combined with deeper LM-side fusion
- bridge-only improvements are still real, but they appear to plateau in the high `0.457x` band
- the main unresolved uncertainty is no longer "can the bridge work?" but rather "how stable and scalable is the adapter-centered mainline?"

So the project now looks bottlenecked less by discovering another bridge family and more by answering:

1. is the new best run stable across seeds
2. is it still truly using the image
3. how much more headroom exists in LM-side visual integration depth
4. whether richer qquery generation can move the new adapter mainline farther

## Nail Thesis

If Hammer answered "where is the remaining headroom?", Nail should answer:

1. is the new mainline statistically real
2. is it visually grounded
3. is the next gain mostly from deeper LM-side interaction, richer qquery generation, or a slightly looser visual token cap

That means Nail should treat:

- `qquery_dynbudget_adapter_earlylayer_geomcal`

as the active baseline, not the old `safeqcond` anchor.

Nail should organize work into three lanes:

1. robustness and diagnostics
2. local scaling around the current best family
3. one higher-upside architecture jump on top of the validated family

## Main Research Questions

### 1. Seed Stability

Are the best adapter-family results actually stable, or did Hammer over-read one favorable seed?

More concretely:

- does `qquery_dynbudget_adapter_earlylayer_geomcal` stay on top over 3 total seeds
- does `dynbudget_adapter_earlylayer_geomcal` remain a real second family, or collapse under reseeding

### 2. Image Dependence

Does the current best checkpoint still rely materially on visual evidence?

More concretely:

- how much does accuracy fall under image shuffle
- how much does accuracy fall under image zero
- how much does accuracy fall under random image swap

### 3. Adapter Depth

How many LM layers actually benefit from visual adapters?

More concretely:

- is the current depth-2 setting underpowered
- does depth-3 help
- does depth-1 retain most of the gain more cheaply

### 4. Token Budget Sensitivity

Is visual compression still limiting the new adapter family?

More concretely:

- does the dynbudget cap at `64` still look like the right operating point
- is `49` too restrictive
- does `96` buy real accuracy or just overhead

### 5. Stronger Question-Conditioned Query Generation

Is the current qquery mechanism still too weak, even inside the best adapter stack?

More concretely:

- does deriving queries from LM question hidden states beat the current simple question-mix basis
- does query generation from richer question-token attention improve targeted evidence extraction

## Architecture Directions

### 1. Stronger Question-Conditioned Query Generation

Current qquery helped, but it was not yet the dominant single-direction win.

Nail version:

- generate queries from richer question-side representations
- prefer LM question hidden states over a single pooled question embedding
- test both simple mean-pool projection and token-attention-derived query generation

Conceptually:

`question hidden states -> query generator -> visual cross-attend -> bridge tokens -> LM adapters`

Why this matters now:

- Hammer suggests the current qquery is useful but not yet strong enough to dominate on its own
- if qquery improves inside the adapter family, that points to extraction quality still mattering after LM-side fusion improved

### 2. Visual Token Routing / Role Specialization

Current dynbudget preserves more detail, but it still treats tokens fairly generically.

Nail version:

- split bridge token roles or routed slots into a small structured set
- examples: object, attribute, spatial, and global tokens
- keep this inside the winning `qquery + dynbudget + adapter` family instead of testing it as a standalone bridge novelty

Conceptually:

`visual grid -> routed/typed token groups -> bridge tokens -> LM adapters`

Why this matters now:

- the remaining misses likely include counting, spatial reasoning, and multi-object composition
- role specialization is the most plausible next step after generic adaptive token preservation

### 3. Bridge Pretraining

Bridge pretraining is still strategically interesting, but it is a bigger phase-change investment than the other Nail ideas.

Candidate directions:

- image to bridge-token to caption-embedding alignment
- image to bridge-token to VM-latent reconstruction/alignment

Why it matters:

- it could improve token quality and optimization stability before VQA supervision

Why it should not dominate the first Nail cycle:

- it needs new training machinery
- Nail first needs to establish whether the cheaper adapter/qquery depth path already has easy headroom left

So pretraining should be treated as a Nail breakout branch, not the first-line Nail queue.

## Shared Nail Baseline

Unless a run is explicitly defined otherwise, Nail runs should inherit the current best family:

- `qquery_dynbudget_adapter_earlylayer_geomcal`
- `bridge_query_bank_mode=question_mix`
- `bridge_token_selector_type=qadaptive`
- dynbudget cap `64`
- LM visual adapters enabled
- adapter depth `2`
- `vision_feature_source=encoder`
- `bridge_token_reduce=adaptive_pool`
- `bridge_question_context_mode=prompt_only`
- effective batch `192`
- target step `9000`
- full final eval with official scorer
- batched KV-cache eval path

Reason:

- this keeps Nail centered on the new frontier family
- it avoids wasting budget relearning Hammer’s main conclusion

## What Nail Should Not Spend Budget On

Do not spend the first Nail cycle on:

- more new bridge-only families without adapters
- more safeqcond-only carry-forward retunes
- dropout or norm-ratio micro-ablations
- broad token-budget grids beyond `{49, 64, 96}`
- bridge pretraining before seed stability and image dependence are known

Those may all become useful later, but they are not the highest-entropy questions right now.

## Proposed Run Set

Recommended run prefix:

- `mmnail_v1_20260313`

### Lane A: Robustness and Diagnostics

These are part of the core Nail question set, not optional cleanup.

Assumption:

- the existing Hammer v2 checkpoints count as seed 1 for the two target families

#### 1. `qquery_dynbudget_adapter_seed2`

What it is:

- exact reseed of `qquery_dynbudget_adapter_earlylayer_geomcal`

Why it exists:

- starts the 3-seed stability check for the current best run

#### 2. `qquery_dynbudget_adapter_seed3`

What it is:

- second reseed of the current best run

Why it exists:

- completes the 3-total-seed picture for the current best family

#### 3. `dynbudget_adapter_seed2`

What it is:

- exact reseed of `dynbudget_adapter_earlylayer_geomcal`

Why it exists:

- tests whether the second-best Hammer family is also real under reseeding

#### 4. `dynbudget_adapter_seed3`

What it is:

- second reseed of the second-best Hammer family

Why it exists:

- completes the 3-total-seed picture for the main non-qquery adapter family

#### 5. `best_ckpt_image_corruptions`

What it is:

- eval-only corruption suite on the current `0.4608` checkpoint
- test `image_shuffle`
- test `image_zero`
- test `random_image_swap`

Why it exists:

- prevents the project from over-celebrating a run that might be weakly image-grounded

### Lane B: Local Sensitivity Around the Best Family

This lane maps the nearest credible gains around the current best stack.

#### 6. `qquery_dynbudget_adapter_d1_cap64`

What it is:

- best family with LM adapter depth `1`

Why it exists:

- lower-bound check on how much of the adapter gain requires depth

#### 7. `qquery_dynbudget_adapter_d3_cap64`

What it is:

- best family with LM adapter depth `3`

Why it exists:

- direct test of whether deeper LM-side fusion is the next easy scaling axis

#### 8. `qquery_dynbudget_adapter_d2_cap49`

What it is:

- best family with dynbudget cap `49`

Why it exists:

- checks whether the current `64` cap is already doing important work

#### 9. `qquery_dynbudget_adapter_d2_cap96`

What it is:

- best family with dynbudget cap `96`

Why it exists:

- checks whether compression is still limiting the best family enough to justify a larger cap

### Lane C: High-Upside Architecture Probes

These are the real "next jump" experiments, but they should be built on the already-winning family.

#### 10. `lmmeanqquery_dynbudget_adapter_d3_cap64`

What it is:

- base: `qquery_dynbudget_adapter_earlylayer_geomcal`
- replace simple qquery generation with projection of the mean LM question hidden states
- adapter depth `3`
- dynbudget cap `64`

Why it exists:

- this is the strongest direct test of richer question-driven extraction inside the best overall family

#### 11. `attnqquery_dynbudget_adapter_d3_cap64`

What it is:

- base: `qquery_dynbudget_adapter_earlylayer_geomcal`
- generate queries from attention over question tokens or LM question hidden states
- adapter depth `3`
- dynbudget cap `64`

Why it exists:

- tests whether a richer token-aware query generator beats simple mean-pool projection

#### 12. `rolespecial_dynbudget_adapter_d3_cap64`

What it is:

- keep adapters and dynbudget
- add routed or role-specialized bridge tokens
- keep depth `3`, cap `64`

Why it exists:

- probes whether the next gain comes more from structured visual roles than from richer qquery generation alone

## Optional Breakout Runs

These should not be in the first Nail launcher unless the main Nail evidence comes back clean.

### `bridgepretrain_latentalign_qquery_dynbudget_adapter`

Purpose:

- test VM-latent or caption-embedding alignment pretraining before VQA finetuning

Why it is optional:

- this is a new training regime, not just a new run config

### `bridgepretrain_captionalign_qquery_dynbudget_adapter`

Purpose:

- alternate pretraining route if latent reconstruction looks too tied to current VM geometry

Why it is optional:

- same engineering-phase caveat as above

## Recommended Single Overnight Experiment

If Nail gets exactly one aggressive overnight shot before the full queue is built, it should be:

- `lmmeanqquery_dynbudget_adapter_d3_cap64`

Configuration:

- base: `qquery_dynbudget_adapter_earlylayer_geomcal`
- query generation: `queries = projection(mean(LM_question_hidden_states))`
- adapter depth: `3`
- dynbudget cap: `64`

Purpose:

- strongest near-term chance to improve targeted extraction while leaning further into the adapter-centered story that Hammer already validated

Expected value:

- best single-run chance to move the frontier without changing the VM

## Projected Execution Priority

The best priority order for the first Nail cycle is:

1. `best_ckpt_image_corruptions`
2. `qquery_dynbudget_adapter_seed2`
3. `qquery_dynbudget_adapter_seed3`
4. `dynbudget_adapter_seed2`
5. `dynbudget_adapter_seed3`
6. `qquery_dynbudget_adapter_d3_cap64`
7. `qquery_dynbudget_adapter_d2_cap96`
8. `lmmeanqquery_dynbudget_adapter_d3_cap64`
9. `attnqquery_dynbudget_adapter_d3_cap64`
10. `rolespecial_dynbudget_adapter_d3_cap64`
11. `qquery_dynbudget_adapter_d1_cap64`
12. `qquery_dynbudget_adapter_d2_cap49`

Why this order:

- corruption and seeds come first because frontier claims should be validated before being elaborated
- depth and token-cap checks come next because they are the cheapest local map of remaining headroom
- stronger qquery and role-specialization probes come after the baseline is validated
- the weakening controls (`d1`, `cap49`) go later because they are useful, but lower upside

## Projected Score Ordering

If the new architectural ideas work, the expected score ordering is roughly:

1. `lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `attnqquery_dynbudget_adapter_d3_cap64`
3. `qquery_dynbudget_adapter_d3_cap64`
4. `rolespecial_dynbudget_adapter_d3_cap64`
5. `qquery_dynbudget_adapter_d2_cap96`
6. `qquery_dynbudget_adapter` reseeds
7. `dynbudget_adapter` reseeds
8. `qquery_dynbudget_adapter_d1_cap64`
9. `qquery_dynbudget_adapter_d2_cap49`

Important caveat:

- this is the expected score ranking, not the execution order
- the corruption suite is diagnostic only and not part of score ranking

## Expected Outcome Bands

Best-case:

- the `0.4608` family is stable across seeds
- corruption tests show clear image dependence
- depth `3` helps
- stronger qquery generation adds another real gain

Middle-case:

- the adapter family is stable
- corruption tests look healthy
- depth and cap tuning matter a little
- richer qquery helps modestly but not dramatically

Worst-case that is still useful:

- the best run is seed-fragile
- corruption tests show weak visual dependence
- deeper adapters and richer qquery do not help

Even that outcome would still be high-value, because it would force the project to stop assuming the new frontier is fully trustworthy and shift attention toward grounding, evaluation, or pretraining.

## Short Version

Hammer said the project's best remaining lever is deeper LM-side visual interaction.

Nail should now verify that result, stress it, and then take one well-aimed swing at a stronger qquery mechanism on top of the adapter-centered mainline, rather than reopening the whole bridge search space.

## Updated Run Draft

This revision reflects three practical constraints:

- keep Nail focused on iteration, not seed accounting
- do not make bridge pretraining part of the first Nail cycle
- spend low-priority slots on stronger frontier probes, not weakened controls

### Revised Main Queue

#### 1. `best_ckpt_image_corruptions`

Why it stays:

- this is still the fastest high-value trust check on the `0.4608` checkpoint

#### 2. `qquery_dynbudget_adapter_d3_cap64`

Why it moves up:

- adapter depth is the cleanest local scaling axis inside the current best family

#### 3. `qquery_dynbudget_adapter_d2_cap96`

Why it stays:

- this is still the best direct test of whether compression is limiting the current mainline

#### 4. `lmmeanqquery_dynbudget_adapter_d3_cap64`

Why it stays:

- this is still the strongest single higher-upside run in the current family

#### 5. `attnqquery_dynbudget_adapter_d3_cap64`

Why it stays:

- this is the natural richer-qquery follow-up if mean-pooled LM question states help

#### 6. `rolespecial_dynbudget_adapter_d3_cap64`

Why it stays:

- this is still the best structured frontier probe for counting, spatial, and multi-object reasoning

#### 7. `lmmeanqquery_dynbudget_adapter_d3_cap96`

What it is:

- combine richer LM-hidden-state qquery generation with adapter depth `3` and cap `96`

Why it is in:

- this is a true frontier probe, not a weakening control
- if the next jump needs both better extraction and a less restrictive token cap, this is where it should show up

#### 8. `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64`

What it is:

- combine role-specialized routing with LM-mean qquery generation at depth `3`

Why it is in:

- this is the most ambitious architecture stack that still stays inside the adapter-centered Nail thesis

### Revised Optional Runs

These move out of the first-line Nail queue.

#### Optional: seed stability

- `qquery_dynbudget_adapter_seed2`
- `qquery_dynbudget_adapter_seed3`
- `dynbudget_adapter_seed2`
- `dynbudget_adapter_seed3`

Why optional now:

- useful later, but not the best use of the next iteration cycle
- there is not yet a broader seed-sweep baseline in the project to compare against

#### Optional: weaker local controls

- `qquery_dynbudget_adapter_d1_cap64`
- `qquery_dynbudget_adapter_d2_cap49`

Why optional now:

- both are still informative
- neither is as valuable right now as another frontier-pushing architecture probe

#### Deferred: bridge pretraining

- `bridgepretrain_latentalign_qquery_dynbudget_adapter`
- `bridgepretrain_captionalign_qquery_dynbudget_adapter`

Why deferred:

- still strategically interesting
- too much engineering surface area right after stabilizing the current runtime
- too high a bug-exposure cost for the next immediate cycle

### Revised Execution Priority

1. `best_ckpt_image_corruptions`
2. `qquery_dynbudget_adapter_d3_cap64`
3. `qquery_dynbudget_adapter_d2_cap96`
4. `lmmeanqquery_dynbudget_adapter_d3_cap64`
5. `attnqquery_dynbudget_adapter_d3_cap64`
6. `rolespecial_dynbudget_adapter_d3_cap64`
7. `lmmeanqquery_dynbudget_adapter_d3_cap96`
8. `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64`

### Revised Short Version

For the next Nail cycle, the right emphasis is:

- trust-check the best checkpoint with corruption eval
- keep pushing the adapter-centered mainline
- prefer richer qquery, deeper adapters, and stronger routed-token stacks
- defer seeds and bridge pretraining until there is more room to spend iteration budget on validation rather than frontier movement
