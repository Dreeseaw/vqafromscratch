# MM Bridge Global Task Context

This document is the standing context and comparison policy for the ongoing multimodal bridge auto-research task.

It is intended to prevent apples-to-oranges comparisons across bridge runs and to preserve stable operating assumptions across future work.

Unless a future note explicitly overrides it for a narrow reason, this document should be treated as the default policy for bridge-modeling experiments.

## Task Scope

This policy applies to:

- multimodal bridge modeling experiments
- bridge architecture sweeps
- bridge ablations
- bridge diagnostics that are intended to compare against the main research line

This policy does not automatically apply to:

- short memory probes
- crash-repro runs
- quick one-off debugging jobs
- special-purpose diagnostics whose goal is not score comparison

Those exception runs should be labeled clearly as non-comparable.

## Comparison Standard

For the remainder of this bridge auto-research task, the comparison-standard training setup is:

- effective batch size: `192`
- progress eval cadence: every `1000` steps
- progress eval size: `100` eval batches
- final eval: entire eval split

This policy exists so that:

- intermediate learning curves are comparable
- run pacing is easier to judge
- final reported scores are full-eval scores suitable for external sharing

## Batch Policy

The standing batch policy is:

- comparison-standard effective batch size must be `192`
- actual in-memory batch size may vary
- gradient accumulation may vary
- but the product of `batch_size * grad_accum_steps` should equal `192`

Valid examples:

- `192 x 1`
- `96 x 2`
- `64 x 3`
- `48 x 4`

Use the largest in-memory `batch_size` that is stable on the current arch, then adjust `grad_accum_steps` to preserve effective batch `192`.

This means future bridge comparisons should not use effective batch `256` as the default comparison regime.

## Evaluation Policy

### Progress Eval

The standard progress-eval configuration is:

- `eval_every=1000`
- `eval_batches=100`

Purpose:

- preserve meaningful progress checkpoints
- allow direct comparison to other long runs
- maintain a consistent partial-eval signal during training

### Final Eval

The standard final-eval configuration is:

- evaluate on the entire eval split

Purpose:

- produce a score that is suitable for reporting externally
- avoid ambiguity caused by half-eval or small-eval comparisons
- ensure final headline numbers are presentation-ready

## Interpretation Rule

When comparing future bridge runs:

- prefer runs that used effective batch `192`
- prefer runs that used `100`-batch periodic eval every `1000` steps
- prefer final full-eval scores for headline ranking

If a run violates any of those standards, it should be explicitly labeled as one of:

- `non-standard batch regime`
- `non-standard progress eval`
- `non-standard final eval`
- `diagnostic only`

## Historical Note

Some earlier bridge runs used different comparison regimes, including:

- different effective batch sizes
- no periodic evals
- half-eval final scoring

Those runs are still useful, but they should be interpreted with care and should not be treated as perfect apples-to-apples comparisons against the new standard policy.

## Recommended Default Run Template

For standard long bridge experiments, prefer the following policy:

- effective batch `192`
- `eval_every=1000`
- `eval_batches=100`
- final eval on the full eval split
- checkpoint every `1000` steps

If memory is tight:

- reduce raw `batch_size`
- increase `grad_accum_steps`
- keep effective batch fixed at `192`

## Exception Handling

Allowed exceptions:

1. memory probes
2. early architecture smoke tests
3. debugging runs
4. diagnostic-only investigations

For those runs:

- deviations are allowed
- but they must be called out in the run note or report
- they should not replace the standard-comparison runs

## Operational Intent

The intent of this policy is simple:

- future bridge runs should be directly comparable by default
- intermediate progress should be visible on the same cadence
- final best runs should end with full-eval scores that can be shown publicly without caveat
