# Hardhat Sweep Plan - 2026-03-15

## Codename

`hardhat`

## Purpose

Hardhat is the sweep that consolidates all unfinished Crane questions into one clean next step.

It should do three things:

1. stabilize and finish the new DINOv2 nodynbudget frontier
2. test the last cheap bridge-side refinements that still plausibly matter in that regime
3. begin the next VM line with a sharper language-alignment experiment rather than spinning up a vague "Crane Part 2"

So Hardhat is not:

- more dynbudget exploration
- more old-VM cleanup
- a caption-align-centric sweep
- a diffuse bag of carry-over runs

It is:

- a dense-visual-memory sweep anchored on the new `0.5762` frontier

## Entry State

Authoritative frontier entering Hardhat:

- best run: `dinov2s_attnqquery_nodynbudget_adapter_d3` at `0.5762`

Most important supporting Crane results:

- `mobileclip_attnqquery_dynbudget_adapter_d3_cap64`: `0.5603`
- `dinov2s_attnqquery_dynbudget_adapter_d3_cap64`: `0.5323`
- `dinov2s_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.5248`
- `dinov2s_questiononly_attnqquery_dynbudget_adapter_d3_cap64`: `0.5355`

Key conclusions inherited from Crane:

1. `attnqquery` is now the default query family.
2. DINOv2 nodynbudget is the strongest current in-family line.
3. Dynbudget is harmful for dense-token perceiver setups.
4. MobileCLIP proved language alignment helps at fixed token count.
5. The next ideal VM target is likely language-aligned and high-token-count.

Key unresolved questions inherited from Crane:

1. Is `0.5762` stable across seeds?
2. Does question-only context help in the nodynbudget regime?
3. Does longer training still buy real slope on the new frontier?
4. Does deeper LM-side multimodal reasoning matter once all 256 DINOv2 tokens reach the perceiver?
5. Does perceiver depth itself become a live lever in the dense-token regime?
6. Is caption-align worth keeping alive once corrected?
7. Does language alignment still help when token count is matched against DINOv2 rather than starved at `49` tokens?

## Hardhat Thesis

Crane already answered the broad question:

- dense visual evidence beats hard pre-filtering

Hardhat should now answer the narrower but more useful question:

- once dense evidence is preserved, what is the next real bottleneck?

My current answer is:

- first confirm the DINOv2 nodynbudget line is real and still rising
- then probe whether the remaining headroom is mostly in query cleanup, longer optimization, or slightly deeper multimodal extraction
- then start the first matched-capacity language-alignment test

That leads to one practical Hardhat queue and one stretch branch.

## What Hardhat Should Treat As Settled

Do not spend Hardhat slots on:

- any new DINOv2 dynbudget cap sweeps
- more `lmmeanqquery` as a mainline
- more MobileViT completion runs
- more adapter-depth sweeps under dynbudget

Reason:

- Crane already priced those directions hard enough

## Shared Hardhat Baseline

Unless a run explicitly changes something, Hardhat should inherit:

- `vision_model=dinov2_small`
- dense token path: `nodynbudget`
- `bridge_query_bank_mode=question_hidden_attn`
- `bridge_question_context_mode=prompt_only`
- `bridge_type=perceiver_resampler`
- `bridge_query_depth=3`
- `lm_visual_adapter_type=cross_attn`
- `lm_visual_adapter_layers=3`
- effective batch `192` via `96x2`
- standard `9000` steps unless explicitly extended
- full-val final eval

Reference run:

- `dinov2s_attnqquery_nodynbudget_adapter_d3`

## Core Questions

### 1. Stability

Before stacking more ideas onto `0.5762`, is that run stable enough to act as the real project anchor?

### 2. Query Cleanliness

Does question-only context help once the model has access to the full DINOv2 token grid?

### 3. Optimization Headroom

Did the Crane frontier stop because the architecture saturated, or because `9000` steps is simply too short now?

### 4. LM-Side Depth

Once all 256 tokens inform the prefix, do deeper LM visual adapters finally matter?

### 5. Bridge Extraction Depth

If the perceiver is now the main distillation module over a dense token grid, does `query_depth=4` start paying off?

### 6. Language Alignment At Matched Capacity

If we compare DINOv2-sized dense-token vision against a similarly sized language-aligned dense-token VM, does alignment still help once token count is no longer confounded?

## Hardhat Main Queue

### Run 1. `dinov2s_attnqquery_nodynbudget_adapter_d3_seed2`

Purpose:

- stabilize the new frontier

Why first:

- everything else depends on whether `0.5762` is a durable result or a lucky one

What it answers:

- whether the DINOv2 nodynbudget line is robust enough to treat as the true baseline

Expected outcome:

- likely still frontier-adjacent
- if it collapses badly, Hardhat should pause and reassess before stacking more changes

### Run 2. `dinov2s_questiononly_attnqquery_nodynbudget_adapter_d3`

Purpose:

- test the cleanest remaining query-side refinement on the actual frontier path

Why this is still live:

- `questiononly` was the best old-VM bridge refinement
- it was also mildly positive on DINOv2 under dynbudget
- it is cheap and directly interpretable

Modeling rationale:

- once the VM exposes dense evidence, a diffuse question context becomes more expensive because the bridge has more real options to misallocate attention over

What it answers:

- whether the current frontier is still partly query-polluted rather than purely capacity-limited

Expected outcome:

- best cheap chance to beat `0.5762` without changing the overall family

### Run 3. `dinov2s_attnqquery_nodynbudget_adapter_d3_18k`

Purpose:

- test whether the nodynbudget frontier is optimization-limited

Why it matters:

- Crane showed the DINOv2 nodynbudget curve separating early and staying healthy through 9k
- if 18k gives another real gain, it changes the default budget for any future top-line run

Modeling rationale:

- the stronger the visual front-end becomes, the less likely it is that the previous `9000`-step budget remains adequate

What it answers:

- whether the next easy gain is just more training rather than more architecture

Expected outcome:

- high-value regardless of the result
- if flat, keep future comparison runs at 9k
- if clearly positive, future frontier stacks should use 18k

### Run 4. `dinov2s_attnqquery_nodynbudget_adapter_d4`

Purpose:

- retest LM-side depth in the only regime where it currently has a real chance to matter

Why this is different from the Crane d4/d5 probes:

- those were all dynbudget-constrained
- nodynbudget passes a much richer visual summary into the LM side

Modeling rationale:

- if the remaining bottleneck has shifted from evidence availability to repeated use of evidence, this is where it should show up

What it answers:

- whether deeper LM-side multimodal reasoning is finally becoming first-order

Expected outcome:

- modest upside, but important diagnostic value

### Run 5. `dinov2s_attnqquery_nodynbudget_bridge_d4_adapter_d3`

Purpose:

- probe perceiver depth directly

Meaning:

- keep the frontier family
- raise bridge/perceiver `query_depth` from `3` to `4`
- keep adapter depth at `3`

Why this run exists:

- the bridge is now doing the real dense-token distillation work
- if any bridge-compute increase still matters, this is the right place to look

Modeling rationale:

- once we stopped deleting tokens, the perceiver became the central compression bottleneck again
- a slightly deeper perceiver is more coherent than returning to token-selection tricks

My skepticism:

- I still think this is less likely to pay than seed2 / questiononly / 18k
- but it is now a legitimate Hardhat run, not a side thought

What it answers:

- whether the next bridge lever is more careful dense extraction rather than more LM-side depth

### Run 6. `dinov2s_captionalign_attnqquery_nodynbudget_adapter_d3_fixed`

Purpose:

- give caption-align one clean, corrected trial on the actual frontier family

What must be fixed first:

1. full 3k pretrain + 9k VQA accounting
2. fresh VQA LR schedule
3. clean optimizer init/transfer handling

Why it is last in the core queue:

- caption-align showed an early positive signal
- but Crane also exposed much larger unambiguous gains elsewhere

What it answers:

- whether caption pretraining produces a real final-score benefit once its implementation stops handicapping the VQA phase

## Hardhat Expansion Branch

These are still part of the Hardhat doc, but should be treated as the first expansion branch after the core queue rather than mixed into it.

### Expansion Hypothesis

Crane strongly suggests the ideal next VM is:

- language-aligned
- dense-token
- not simply another small mobile backbone

### Expansion Run A. `siglips_attnqquery_nodynbudget_adapter_d3`

Purpose:

- the cleanest matched-capacity test of language alignment versus DINOv2

Why SigLIP-S first:

- much cleaner attribution than CLIP ViT-B/16
- closer to DINOv2-small in scale
- dense token grid with language-aligned training objective

What it answers:

- whether language alignment helps when token count and model scale are no longer badly mismatched

### Expansion Run B. `clipvitb16_attnqquery_nodynbudget_adapter_d3`

Purpose:

- the bigger systems bet: dense tokens plus much stronger language-aligned features

Why second:

- higher upside than SigLIP-S
- but less controlled as a scientific comparison

What it answers:

- whether the project is ready for another MobileViT-to-DINOv2 style jump using a better high-token aligned VM

## Recommended Execution Priority

### Core

1. `dinov2s_attnqquery_nodynbudget_adapter_d3_seed2`
2. `dinov2s_questiononly_attnqquery_nodynbudget_adapter_d3`
3. `dinov2s_attnqquery_nodynbudget_adapter_d3_18k`
4. `dinov2s_attnqquery_nodynbudget_adapter_d4`
5. `dinov2s_attnqquery_nodynbudget_bridge_d4_adapter_d3`
6. `dinov2s_captionalign_attnqquery_nodynbudget_adapter_d3_fixed`

### Expansion

7. `siglips_attnqquery_nodynbudget_adapter_d3`
8. `clipvitb16_attnqquery_nodynbudget_adapter_d3`

## Decision Rules

### After Run 1

- if seed2 is far below `0.5762`, stop stacking and reassess before trusting the new baseline

### After Run 3

- if 18k gives a meaningful gain, make longer training the default for any new top-line attempt
- if it is flat, keep subsequent comparative work at 9k

### After Runs 4 and 5

- if adapter depth helps but bridge depth does not, the bottleneck is LM-side use of dense evidence
- if bridge depth helps but adapter depth does not, the bottleneck is still dense-token distillation
- if neither helps, the next serious slope likely lies more in VM quality than in extra bridge/LM compute

### After Run 6

- if corrected caption-align is still flat, move it to archival/deferred status
- if corrected caption-align is positive, keep it as a cheap booster rather than a mainline identity shift

### After Expansion Run A

- if SigLIP-S beats DINOv2-small, language alignment has now won at matched capacity
- if it does not, dense self-supervised token quality may still be the better trade in this project

## Project-Level Interpretation

Hardhat should operationalize the following grand-scheme read:

1. query quality mattered
2. then VM quality mattered
3. Crane showed dense visual evidence mattered even more than token pruning tricks
4. the next phase is about preserving and exploiting dense evidence, then testing aligned dense-token VMs

So the main question is no longer:

- "what additional bridge novelty should we try?"

It is:

- "how do we best preserve and distill dense visual memory into the current LM stack?"

That is why Hardhat should stay narrow and disciplined.

## Recommended Single Run

If only one Hardhat run should go first:

- `dinov2s_attnqquery_nodynbudget_adapter_d3_seed2`

Reason:

- everything depends on the new baseline being real

If only one new-idea run should go first after the seed check:

- `dinov2s_questiononly_attnqquery_nodynbudget_adapter_d3`

Reason:

- it is the cheapest clean refinement of the actual winning family

## One-Line Summary

Hardhat should replace any vague "Crane Part 2" idea with one disciplined sweep anchored on the DINOv2 nodynbudget frontier: first stabilize and extend that line with seed2, question-only context, longer training, LM-depth and perceiver-depth probes, then run one corrected caption-align test, and only after that open the next VM branch with a matched-capacity dense-token language-aligned model such as SigLIP-S.
