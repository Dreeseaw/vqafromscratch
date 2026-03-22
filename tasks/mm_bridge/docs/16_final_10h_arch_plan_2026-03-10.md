# Final 10h Architecture Plan - 2026-03-10

## Objective

Finalize a 10-hour architecture program that balances:

1. projected overall accuracy
2. project insight for future runs
3. a controlled amount of novelty

This plan supersedes earlier brainstorming and is the current reference for the next architecture cycle.

## Final Order

1. **Leakage-safe question-conditioned perceiver**
- Highest near-term upside if conditioning is restricted to true question-prefix tokens only.
- Best direct test of whether question-guided evidence extraction is the missing capability.

2. **Multi-scale bridge**
- Strongest score/insight blend after q-conditioned extraction.
- Tests whether combining detail-heavy and semantic visual signals improves VQA without changing the LM fusion location.

3. **Early-layer feature bridge**
- High diagnostic value with moderate implementation risk.
- If it wins, it changes how we think about the VM bottleneck and the value of final latent features.

4. **Oracle196 + adaptive token selection**
- Best compression/evidence-efficiency probe.
- Tests whether larger evidence banks are useful when paired with selective sparsification.

5. **Geometry-aware prefix calibration on a strong non-qcond bridge**
- Direct continuation of the strongest established project finding: LM interface sensitivity.
- Novel enough to be interesting without losing contact with known bottlenecks.

6. **Adaptive token selection v2**
- Important follow-up once oracle+selection results are available.
- Focuses on whether sparse evidence extraction, not just token count, drives gains.

7. **Structured token roles bridge**
- Good novelty/interpretability tradeoff.
- Useful if we want a more semantic explanation for what bridge tokens are learning.

8. **Evidence-focused sparse bridge**
- Most speculative of the selected set.
- Valuable for novelty and future research direction, but lower immediate confidence than the runs above.

## Explicitly Deferred

- **Residual LM visual adapter**
  - Very high score ceiling, but too large a branch point for the current mixed architecture cycle.
- **Dynamic token budgets**
  - Novel and interesting, but lower confidence than the selected evidence/compression runs.
- **Slot attention / token routing / bridge pretraining**
  - Not the best use of this 10-hour block.

## Practical Sweep Shape

Recommended split:

- `main`: items `1`, `2`, `3`
- `medium`: items `4`, `5`, `6`
- `explore`: items `7`, `8`

## Current Implementation Priorities

The existing codebase already partially supports:

- early-layer feature probing (`vision_feature_source=encoder`)
- baseline adaptive token selection (`topk`)

The missing work to unlock the full plan is:

1. leakage-safe qcond
2. multi-scale bridge support
3. geometry-aware prefix calibration
4. structured token roles bridge
5. evidence-focused sparse bridge

## Operational Rule

Before running any long training jobs with these new branches:

- benchmark each new branch with short Docker-backed runs
- determine a safe `batch_size` / `grad_accum_steps` pair
- keep `eval_every=0` during those memory/performance probes

