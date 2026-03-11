# High-Entropy Sweep Plan - 2026-03-11

## Purpose

Define the next bridge sweep as a high-information cycle, not a low-entropy ablation pass.

The goal of this sweep is not:

- tiny local parameter nudges
- cleanup ablations around already weak branches
- more token-count-only probing

The goal is:

- confirm which new branches actually scale
- test whether the strongest new gains stack
- spend the next `6+` hours on runs that materially change project direction

## Current Read

The recent sweep should be treated as successful architecture triage under a shortened horizon.

Most important facts:

- `safeqcond_d3_main` finished at `0.4460`
- the previous eventual-best perceiver run was only about `0.4410` at its own `step=6000`
- `structuredroles_d3_exp` and `earlylayer_encoder_d3_main` were also in that competitive `6k` band

So the current read is:

- the best new branches are not just "interesting"
- they are already pacing competitively
- the highest-value next move is to validate slope under longer training

That means the right principle for this sweep is:

- validate slope first
- then test one or two high-value combinations
- do not scatter budget across many novelty-only branches yet

## Sweep Philosophy

This sweep should be a mixed cycle:

1. harden the best new single branches at the old frontier horizon
2. include a small number of genuinely informative combination runs
3. avoid low-entropy variants that only tweak one scalar without changing project understanding

Concretely, I would not spend this sweep on:

- top-k size sweeps
- dropout micro-ablations
- more standalone sparse-selection variants
- more pure oracle token-count scaling

## Recommended Run Set

### 1. `safe qcond frontier harden`

Why:

- strongest new result
- directly validates that the qcond fix is real
- highest chance of overtaking the old frontier under full budget

What it answers:

- does safe qcond keep its advantage past `6k`
- is question-guided extraction now the main project branch

Why high-entropy:

- this is not a cleanup rerun
- it decides whether the central research direction changes

### 2. `structured roles frontier harden`

Why:

- strongest novelty-positive branch
- already competitive at the shorter horizon
- if it keeps pace at full budget, it becomes a serious new family

What it answers:

- was structured roles just a fast starter
- or is it a real frontier-capable architecture

Why high-entropy:

- this run determines whether a novel token-organization idea belongs in the mainline

### 3. `early-layer encoder frontier harden`

Why:

- strong enough at `6k` to remain live
- directly probes whether earlier visual features matter beyond warm-start effects

What it answers:

- is the latent bottleneck partly caused by over-compressed final VM features
- should the project shift toward earlier or multi-scale visual sources

Why high-entropy:

- this is a core representational question, not a surface ablation

### 4. `safe qcond + geometry-aware calibration`

Why:

- safe qcond is the strongest new core
- geometry-aware calibration looked decent as a modifier but not a standalone winner
- this is a plausible stacking path

What it answers:

- do better question-guided extraction and better LM-interface shaping compound
- or are they mostly solving the same failure mode

Why high-entropy:

- this directly tests gain compositionality, which is much more valuable than another standalone rerun

### 5. `safe qcond + early-layer encoder`

Why:

- if question-conditioning wants better raw evidence, earlier features are one of the best candidates
- this is probably the most meaningful two-way combination in the current search space

What it answers:

- does qcond become more powerful when fed less-compressed visual evidence
- is the real path "question-guided selection over earlier visual detail"

Why high-entropy:

- this is one of the most informative architectural conjunctions available right now

### 6. `structured roles + geometry-aware calibration`

Why:

- structured roles was strong enough to deserve one serious stack test
- geometry-aware calibration is a reasonable modifier for any nontrivial tokenized bridge

What it answers:

- can structured token semantics benefit from better interface geometry
- or is structured roles already doing enough internal organization that extra calibration adds little

Why high-entropy:

- this tests whether structured roles can become a real mainline family instead of a one-off curiosity

## Proposed Sweep Order

If the horizon is only moderately above `6` hours, I would prioritize in this order:

1. `safe qcond frontier harden`
2. `structured roles frontier harden`
3. `early-layer encoder frontier harden`
4. `safe qcond + geometry-aware calibration`
5. `safe qcond + early-layer encoder`
6. `structured roles + geometry-aware calibration`

Reason:

- first confirm the three strongest single branches
- then spend the remaining time on the highest-value stacking tests

## Why This Sweep Is High-Entropy

Every run in this set changes one of the major project beliefs:

- whether qcond is the new mainline
- whether structured roles is truly frontier-capable
- whether earlier features are part of the real solution
- whether gains stack across extraction and interface geometry

That is the kind of sweep that teaches the project something even if only one run wins.

By contrast, a low-entropy sweep would mostly tell us:

- maybe `k=24` is a little better than `k=32`
- maybe `dropout=0.02` is a little better than `0.03`

That is not the right use of the next cycle.

## Main Expected Outcomes

Best-case:

- `safe qcond` becomes the new clear mainline
- one of the combination runs shows additive gains
- the next cycle becomes very focused

Middle-case:

- `safe qcond`, `structured roles`, and `early-layer` all remain competitive
- stacking is mixed
- we still get a clean narrowed frontier family

Worst-case that is still useful:

- the single-branch reruns flatten out
- the combinations do not stack
- we learn that the `6k` pace advantage was mostly transient

Even that would still be high-value information.

## Current Recommendation

If I had to pick the single best next sweep theme in one line:

- run a slope-validation sweep around `safe qcond`, with `structured roles` and `early-layer` as the two main challengers, then spend the remaining budget on one or two stack tests

That is the highest-value next move for the project.
