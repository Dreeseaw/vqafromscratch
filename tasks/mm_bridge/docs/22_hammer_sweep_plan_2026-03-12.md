# Hammer Sweep Plan - 2026-03-12

## Codename

`hammer`

## Purpose

Hammer is the first post-high-entropy sweep designed to attack the remaining structural bottlenecks directly.

This sweep should not be about:

- tiny frontier retunes
- more dropout or norm-ratio micro-ablations
- another round of bridge-family sightseeing

This sweep should be about answering three hard questions:

1. Is the remaining ceiling mostly an extraction problem?
2. Is it mostly a compression problem?
3. Is prefix-only fusion now the main limitation?

The right Hammer cycle is therefore a bottleneck-breaking sweep, not a frontier-polishing sweep.

## Entry State

Current best evidence entering Hammer:

- best observed run: `safeqcond_earlylayer_geomcal_frontier` at `0.4568`
- strongest two-way stack: `safeqcond_earlylayer_frontier` at `0.4561`
- top cluster is tightly packed and overwhelmingly `safeqcond`-centered
- `earlylayer` and `geomcal` both look real, but their marginal gains are now smaller than the original qcond breakthrough

What the high-entropy sweep already taught us:

- leakage-safe qcond is real
- early-layer evidence is real
- geometry-aware calibration is a useful modifier, not the main story
- multiscale and hybrid stay competitive, but they did not beat the strongest qcond+earlylayer path

So the project is no longer bottlenecked on "can the bridge work at all?"

The likely remaining limits are now:

1. fixed-budget compression of the visual grid into `K=49`
2. incomplete question-guided evidence extraction
3. prefix-only LM fusion depth

## Hammer Thesis

Hammer should treat the current best stack as the stable carry-forward control:

- `safeqcond + earlylayer + geomcal`

Then it should attack exactly three new structural directions:

1. question-conditioned perceiver queries
2. adaptive visual token budget
3. residual visual adapters inside the LM

The sweep logic should be:

1. hold the current best path fixed as the anchor
2. add one major new mechanism at a time
3. only then test the highest-value pairwise and full stacks

That keeps the sweep high-information instead of turning it into an uninterpretable pile of simultaneous changes.

## Main Research Questions

### 1. Extraction

Does the bridge still summarize the image too generically even when qcond is enabled?

More concretely:

- is current `safeqcond` still mostly modulating a static query bank
- would question-derived query tokens extract more relevant evidence than the current latent FiLM-style conditioning

### 2. Compression

How much accuracy is still being lost by forcing the image into a fixed token budget before LM use?

More concretely:

- does a question-conditioned token budget help preserve the right evidence
- are counting, spatial, and multi-object questions still bottlenecked by uniform compression

### 3. Fusion Depth

Has prefix-only fusion become the dominant remaining limiter?

More concretely:

- would letting LM hidden states attend to visual tokens inside the LM beat the current prefix-only path
- are we now bottlenecked more by reasoning depth than by bridge token quality alone

## Architecture Directions

### 1. Question-Conditioned Perceiver Queries

Current `safeqcond` is already strong, but it does not fully replace the static learned query bank with question-derived queries.

Hammer version:

- pooled question/prompt context produces or modulates the actual perceiver query tokens
- those question-derived queries attend into the visual grid directly
- the bridge then refines those extracted tokens in the normal perceiver path

Conceptually:

`question context -> query tokens -> cross-attend into visual grid -> perceiver refinement -> prefix`

Why this is distinct from current qcond:

- current qcond mostly conditions extraction
- Hammer qquery changes what the extractor queries with

Expected value:

- strongest direct attack on incomplete question-guided evidence extraction
- most likely to improve `number` and `other`

### 2. Adaptive Visual Token Budget

The current bridge still compresses into a fixed token count.

Earlier fixed top-k selection did not solve this, but that is not the same as a real adaptive-budget design.

Hammer version:

- score visual tokens before compression
- make the scorer question-conditioned
- route a variable or softly gated subset of tokens into the bridge
- preserve more detail when the question appears evidence-hungry

Conceptually:

`visual grid -> question-conditioned token scoring -> adaptive keep/routing -> bridge extraction`

Expected value:

- direct attack on compression loss
- most relevant for counting, spatial relations, and multi-object scenes

### 3. Residual Visual Adapters Inside the LM

The current best system still relies on prefix-only visual conditioning.

Hammer version:

- keep the bridge tokens
- add lightweight residual cross-attention adapters inside the top LM blocks
- let LM hidden states revisit visual evidence during decoding

Conceptually:

`LM hidden state -> cross-attend into bridge visual tokens -> residual merge back into LM`

Expected value:

- direct test of whether multimodal reasoning now needs in-layer interaction rather than a stronger prefix alone
- best architectural probe for the "reasoning depth vs extraction" question

## Shared Hammer Baseline

Unless a run is explicitly defined otherwise, Hammer runs should inherit the best current bridge stack:

- `safeqcond`
- `bridge_question_context_mode=prompt_only`
- `vision_feature_source=encoder`
- `bridge_token_reduce=adaptive_pool`
- `prefix_geom_mlp_ratio=0.5`
- `prefix_geom_token_mixer_layers=1`
- effective batch `192`
- target step `9000`
- full final eval with official scorer

Reason:

- this keeps Hammer focused on new bottlenecks
- it avoids relearning solved earlier choices inside the same sweep

## What Hammer Should Not Spend Budget On

Do not spend this sweep on:

- `K=49` vs `K=56` vs `K=64` micro-sweeps
- prefix dropout nudges
- more hybrid alpha retunes
- top-LM-layer count micro-ablations
- another structured-roles side branch

Those may be useful later, but they are low-entropy relative to the current project questions.

## Proposed Run Set

Recommended run prefix:

- `mmhammer_v1_20260312`

### 1. `anchor_safeqcond_earlylayer_geomcal`

What it is:

- exact carry-forward of the current best stack

Why it exists:

- gives Hammer a same-sweep control
- makes every new branch comparable against the strongest known baseline

What it answers:

- does the current best path hold up cleanly under the new sweep setup

### 2. `qquery_earlylayer_geomcal`

What it is:

- replace static perceiver queries with question-conditioned query tokens
- keep `safeqcond + earlylayer + geomcal` otherwise fixed

Why it exists:

- highest-value direct test of better question-guided extraction

What it answers:

- does changing the extractor query bank itself beat current qcond modulation
- is extraction quality still the main remaining limiter

### 3. `adapter_safeqcond_earlylayer_geomcal`

What it is:

- keep the current best bridge
- add residual visual cross-attention adapters inside the top LM blocks

Why it exists:

- cleanest direct test of the prefix-only bottleneck

What it answers:

- does in-layer visual access improve reasoning beyond prefix-only fusion
- has the bottleneck moved from extraction into LM interaction depth

### 4. `dynbudget_qscore_earlylayer_geomcal`

What it is:

- keep the current best bridge
- add question-conditioned visual token scoring / routing before compression

Why it exists:

- cleanest direct test of the fixed-token compression bottleneck

What it answers:

- is fixed-budget compression still discarding useful evidence
- do harder questions benefit from preserving more visual detail

### 5. `qquery_adapter_earlylayer_geomcal`

What it is:

- combine question-conditioned perceiver queries with residual LM visual adapters

Why it exists:

- highest-value extraction-plus-reasoning stack

What it answers:

- if better evidence is extracted, can the LM use it more effectively only when it can revisit visual tokens in-layer

### 6. `qquery_dynbudget_earlylayer_geomcal`

What it is:

- combine question-conditioned queries with adaptive token budget

Why it exists:

- highest-value extraction-plus-compression stack

What it answers:

- does better question guidance mostly matter because it helps preserve the right evidence before compression

### 7. `dynbudget_adapter_earlylayer_geomcal`

What it is:

- combine adaptive token budget with residual LM visual adapters

Why it exists:

- probes whether preserved detail matters mainly when the LM can revisit visual tokens during reasoning

What it answers:

- is the project bottleneck currently "preserve more detail" or "let the LM actually use preserved detail"

### 8. `qquery_dynbudget_adapter_earlylayer_geomcal`

What it is:

- the full Hammer stack
- question-conditioned query tokens
- adaptive token budget
- residual LM visual adapters

Why it exists:

- this is the densest direct attack on all three remaining bottlenecks at once

What it answers:

- if extraction, compression, and reasoning depth are all partially limiting, does the full stack become the new mainline

## Optional Backup Runs

If the main 8-run set completes cleanly and there is still budget left, the best backup diagnostics are:

### `dynbudget_blindscore_earlylayer_geomcal`

Purpose:

- control for the adaptive-budget direction
- separate "dynamic budget helps" from "question-guided scoring helps"

### `qquery_multiscale_geomcal`

Purpose:

- follow up only if qquery helps but early-layer-only evidence still looks too narrow
- tests whether better extraction wants richer dual-scale evidence rather than just better queries

### `adapter_top4_safeqcond_earlylayer_geomcal`

Purpose:

- only if the basic adapter run is clearly positive
- maps whether fusion depth scaling matters beyond the initial in-layer adapter test

## Projected Final Ordering

Projected Hammer queue order:

1. `anchor_safeqcond_earlylayer_geomcal`
2. `qquery_earlylayer_geomcal`
3. `adapter_safeqcond_earlylayer_geomcal`
4. `dynbudget_qscore_earlylayer_geomcal`
5. `qquery_adapter_earlylayer_geomcal`
6. `qquery_dynbudget_earlylayer_geomcal`
7. `dynbudget_adapter_earlylayer_geomcal`
8. `qquery_dynbudget_adapter_earlylayer_geomcal`

Why this order:

- the anchor goes first because Hammer needs a clean same-sweep control on the current best path
- `qquery` goes second because extraction quality is the single highest-value remaining question
- `adapter` goes third because prefix-only fusion is the next most structural bottleneck candidate
- `dynbudget` goes fourth because compression is still important, but slightly less foundational than the extraction and fusion-depth questions
- pairwise stacks come after the three single-direction tests so their interpretation is clean
- the full Hammer stack goes last because it has the highest ceiling but the lowest diagnostic clarity if run too early

If the sweep is cut short, the keep set should be:

1. `anchor_safeqcond_earlylayer_geomcal`
2. `qquery_earlylayer_geomcal`
3. `adapter_safeqcond_earlylayer_geomcal`
4. `dynbudget_qscore_earlylayer_geomcal`
5. `qquery_adapter_earlylayer_geomcal`

## Why Hammer Is High-Entropy

Every main run changes one major project belief:

- whether question-guided extraction still has major headroom
- whether fixed compression is now the dominant bottleneck
- whether prefix-only fusion is now too shallow
- whether these gains stack or mostly overlap

That is exactly the kind of sweep the project should run now.

A low-entropy alternative would mostly answer:

- maybe `K=56` is slightly better than `K=49`
- maybe one more prefix regularizer value is a bit better

That is not the right move after the high-entropy sweep already pushed the frontier.

## Expected Outcome Bands

Best-case:

- `qquery` is clearly positive
- one of the adapter stacks is also clearly positive
- Hammer identifies a new mainline above the current `0.4568` anchor

Middle-case:

- one of the three new directions is clearly real
- one is neutral
- one is too unstable or too costly
- the next phase still becomes sharply more focused

Worst-case that is still useful:

- none of the new directions beat the anchor
- that would imply the current ceiling is less about bridge structure and more about VM representation quality, training data, or broader LM adaptation limits

Even that would still be high-value information.

## Short Version

If Hammer has to be summarized in one line:

- keep `safeqcond + earlylayer + geomcal` as the control, then test question-conditioned queries, adaptive budget compression, and in-layer visual adapters as the three main attacks on the remaining ceiling
