# Prefix Remap Diagnostic Report

Date: 2026-03-24  
Plan: [54_prefix_remap_diagnostic_plan_2026-03-23.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/54_prefix_remap_diagnostic_plan_2026-03-23.md)  
Run: [mmsemantic_remap_v1_debug](/home/wdree/percy/vqafromscratch/logs/mmsemantic_remap_v1_debug)  
Analyzer output: [remap_diagnostic.md](/home/wdree/percy/vqafromscratch/logs/mmsemantic_remap_v1_debug/remap_diagnostic.md)

## Setup

This experiment loaded the completed `K=8` semantic bottleneck checkpoint from:

- [step_4000.tar](/home/wdree/percy/vqafromscratch/logs/mmsemantic_v1_20260322_k8/step_4000.tar)

Then it froze everything except a single new linear prefix remap:

```text
Frozen SigLIP VM
-> Frozen perceiver
-> Frozen K=8 semantic bottleneck
-> trainable PrefixRemap: Linear(512, 512)
-> frozen LM prefix path
-> LM visual adapters disabled entirely
```

Trainable count from the run log:

- `262,656 / 164,189,881` params

This is a true keep-0 diagnostic. The LM adapters were not merely frozen; they were disabled in the forward path.

## Main Result

The linear remap recovered most of the lost K=8 keep-0 performance.

| Condition | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| `K=8 keep-3` full system | 0.6154 | 0.7611 | 0.4583 | 0.5462 |
| `K=8 keep-0` baseline | 0.3744 | 0.4510 | 0.2757 | 0.3423 |
| `K=8` linear probe | 0.5031 | 0.6199 | 0.3511 | 0.4316 |
| `K=8 + PrefixRemap keep-0` final eval | 0.5762 | 0.7456 | 0.4377 | 0.4839 |

The most important number is `yes/no = 0.7456`.

That is:

- `+0.2946` over the raw keep-0 baseline
- only `-0.0155` below the full `keep-3` system
- `95.0%` recovery of the adapter contribution on yes/no

This matches Outcome A from the experiment plan.

## Training Trace

The run improved quickly and mostly monotonically on the 100-batch checkpoint evals:

| Step | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| 100 | 0.5494 | 0.7455 | 0.4015 | 0.4442 |
| 200 | 0.5605 | 0.7474 | 0.4246 | 0.4587 |
| 300 | 0.5682 | 0.7420 | 0.4322 | 0.4763 |
| 400 | 0.5695 | 0.7440 | 0.4342 | 0.4767 |
| 500 mini-eval | 0.5774 | 0.7545 | 0.4414 | 0.4830 |
| 500 final full eval | 0.5762 | 0.7456 | 0.4377 | 0.4839 |

Two things matter here:

- recovery happened very early; yes/no was already `0.7455` by step `100`
- the final full eval stayed aligned with the step-500 mini-eval, so this was not a noisy checkpoint illusion

## Interpretation

The evidence now strongly favors a format-mismatch story over a deep routing-failure story.

Why:

- the `K=8` tokens already contained usable signal, which the probe had suggested
- a single linear map was enough to let the frozen LM consume that signal much more effectively
- the recovery is broad, not yes/no-only:
  - `overall`: `83.7%` of adapter work recovered
  - `yes/no`: `95.0%`
  - `number`: `88.7%`
  - `other`: `69.5%`

The cleanest modeling read is:

1. The `K=8` bottleneck is not primarily failing by deleting answer-relevant content.
2. It is exporting that content in a geometry the frozen LM prefix pathway does not naturally expect.
3. The LM visual adapters were compensating for that mismatch, at least in large part, by acting as format translators.

This is especially important for the earlier fragility result. The old interpretation was that `K=8` needed strong LM-side help. The refined interpretation is narrower and better:

- `K=8` needs strong alignment to the LM input format
- it does not necessarily need deep in-network adapter reasoning to the same degree

## What This Means

For the semantic bottleneck line, this is a strong positive.

The bottleneck thesis now looks like:

- late compression to `K=8` preserves far more task signal than the raw keep-0 collapse implied
- the main remaining issue is interface alignment, not semantic emptiness

That pushes the next design pressure toward:

- training the bottleneck to emit LM-compatible tokens directly
- or adding explicit format-alignment losses during compression tuning

It pushes away from:

- immediately adding more LM-side adapter depth
- treating the `K=8` failure as evidence that aggressive semantic compression is fundamentally broken

## Recommended Next Experiment

The next clean follow-up should be bottleneck-side format alignment, not bigger LM machinery.

Best next move:

- add an explicit alignment objective that encourages bottleneck outputs to match the prefix-space geometry that the frozen LM already likes

Concrete versions:

- distill from the uncompressed champion prefix after prefix calibration
- or keep the linear remap during compression training as a temporary teacher head, then fold that pressure back into the bottleneck outputs

Short version:

- `K=8` is viable
- the adapters were mostly rescuing interface mismatch
- the frontier is now LM-prefix compatibility, not raw semantic capacity
