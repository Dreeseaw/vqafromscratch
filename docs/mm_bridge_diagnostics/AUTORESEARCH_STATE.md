# AUTORESEARCH_STATE

This file is the standing scratchpad for current project interpretation, current beliefs, and current directional bias for the bridge auto-research effort.

It is intentionally more opinionated than the formal reports.

## Current Project State

The project is in a much better state than it was during the earlier "is the bridge even helping?" phase.

That question is largely answered.

Current belief:

- bridge design matters
- the perceiver-style family is real
- image-conditioned bridges can compete
- the remaining problem is no longer basic viability
- the remaining problem is finding the best fusion path and training budget for frontier performance

## Current Strongest Beliefs

### 1. Safe qcond is real

This is the most important recent update.

The original qcond collapse now looks like an implementation/pathology issue, not a dead research direction.

Current belief:

- question-conditioned extraction is one of the highest-value paths in the whole project
- it should stay near the center of the research program

### 2. Structured roles is unusually promising

Structured roles performed well enough to matter and is more novel than the standard perceiver tweaks.

Current belief:

- this may be the best novelty-to-performance tradeoff in the current queue
- it deserves serious follow-up, not just a footnote

### 3. Early-layer features are still alive

The encoder-feature run was not a blowout, but it was clearly credible.

Current belief:

- final visual latents are probably not the full story
- the project should keep an open path toward earlier or multi-scale visual features

### 4. Sparse evidence ideas are not ready by themselves

Top-k and evidence-sparse variants did not justify aggressive promotion yet.

Current belief:

- sparse extraction probably needs better guidance
- question guidance, better saliency, or stronger structure will likely be required

## Current Directional Bias

If I had to bias the next chunk of project time right now, I would bias toward:

1. safe qcond follow-ups
2. structured roles follow-ups
3. early-layer or multiscale follow-ups
4. combinations of those with the strongest existing perceiver settings

I would currently de-bias away from:

1. naive token-pruning-only work
2. pure oracle token-count scaling as a main frontier path
3. very large novelty jumps before consolidating the stronger new signals

## Current Interpretation of the Frontier

The current old frontier still matters, but I do not think it should dominate decision-making too rigidly.

Reason:

- some new arches were on a better pace at the comparable `6k` point
- several new branches were trained under lighter budget than the old endpoint runs

Current belief:

- the next best run is likely to come from a new arch trained longer
- not from endlessly re-running the exact old best config unchanged

## Current Research Mood

The project no longer feels confused.

It feels narrowed.

The search space is still large, but the high-value region is smaller than before.

That is good.

The current vibe is:

- less "do bridges work?"
- more "which bridge family actually deserves scaling and consolidation?"

## Current Practical Rule

Before spending large budget on ablations:

- first identify which new arch family has genuine pace advantage
- then ablate inside that family

Right now, the best candidates for that treatment are:

1. safe qcond
2. structured roles
3. early-layer features

## Current Risk

The main risk now is not total failure.

The main risk is wasting cycles on branches that are merely interesting instead of branches that are both interesting and plausibly frontier-capable.

So the project should stay disciplined:

- keep novelty where it teaches something
- but keep the main budget concentrated on arches that already showed pace

## Update Intent

This file should be updated whenever one of these changes materially:

- what I think the leading branch is
- what I think the main bottleneck is
- what I think should be deprioritized
- how I interpret the overall direction of the project
