# Quarter-LM Format Refinement Report

Bundle: [lmshrink_quarter_format_v1_20260325_191730](/home/wdree/percy/vqafromscratch/logs/lmshrink_quarter_format_v1_20260325_191730)

## Outcome

Phase 1 failed the gate, so the experiment correctly stopped before the long `5k` quarter-specific format retrain.

Quarter reference from the LM shrink sweep:

- bridge: `0.6044`
- compressed: `0.5705`
- compression delta: `-0.0339`

Quarter remap diagnostic:

- best `100`-batch remap eval: `0.5621`
- gain over compressed baseline: `-0.0084`
- full-val terminal remap eval from the training run: `0.5575`

So the quarter-specific linear remap did not recover the quarter gap. It underperformed the existing quarter compression line on both the mini-eval gate and the full-val terminal eval.

## Interpretation

This is a negative result, but it is scientifically clean.

The quarter LM shrink result already told us that smaller LMs are more format-sensitive than the large LM. This experiment asked a sharper question: is the remaining quarter gap mostly a simple linear interface mismatch? The answer looks like no.

If the quarter compressed model were mainly suffering from a bad LM-facing basis, a quarter-specific remap should have added back at least some of the missing accuracy. Instead:

- overall went down
- `other` stayed clearly below the original quarter compressed line
- the remap never established a better frontier than the existing bottleneck output

That means the original quarter compression is already close to the best linear prefix geometry available to this stack.

## What This Means

The remaining quarter gap is more plausibly one of these:

1. Genuine LM-capacity loss.
2. A deeper non-linear consumption/routing issue inside the smaller LM.
3. A bottleneck-learning issue that cannot be repaired by a thin linear remap after the fact.

What it is not, based on this result, is a cheap linear-format bug waiting to be fixed.

## Practical Read

For deployment-style decision-making:

- `Half` remains the strongest small-LM frontier.
- `Quarter` is still viable, but the current gap should be treated as real, not obviously recoverable by a lightweight remap trick.

So the updated recommendation is:

- keep `24.1M` as the safe deployment frontier
- treat `12.2M` as the aggressive option only if the extra accuracy loss is acceptable

## Next Move

If quarter work continues, the next useful experiment should not be “another linear remap.”

It should be one of:

1. Quarter LM + stronger bottleneck-side supervision during compression, not posthoc remap.
2. Quarter LM + minimal adapter/routing capacity restored at compression time.
3. Quarter LM + architecture change that improves token consumption, not just token basis.

The cheap linear-fix path has been tested and came back negative.
