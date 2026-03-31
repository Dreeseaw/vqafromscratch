# Variable-K Semantic Budgeting Plan

## Goal

Test whether the late LM-facing semantic bottleneck should support a variable token budget per sample instead of a fixed `K`, and measure the oracle upside before spending time on any heuristic or learned budget scheduler.

This is an LM-side experiment only. The perceiver still sees the normal visual inputs and still retrieves the same dense evidence latents. The only change is how many ordered semantic tokens are exposed to the LM interface.

## Base Line

Start from the strongest current fresh-bridge SigLIP2 source that is directly compatible with the existing semantic-compression path:

- source bridge checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar)
- source family: SigLIP2 + `lm_final` + Qwen-KD + fresh-init top-2-block VM finetune
- source bridge full eval: `0.6470`

The new run is a stage-2-style compression experiment on top of that full-evaled bridge checkpoint, but with a max semantic budget of `16` and variable prefix exposure during training.

## Training Change

- keep the current late semantic bottleneck architecture
- set max semantic token count to `16`
- define supported budgets as `K in {4, 8, 12, 16}`
- during training, sample one budget per batch and expose only the first `K` semantic tokens to the LM
- keep the full `K=16` bottleneck path internally so this stays a prefix-robustness test, not a new bottleneck architecture
- keep the existing format-alignment setup and broader recipe otherwise unchanged

## Eval Bundle

At the completed checkpoint, run the same model at:

1. fixed `K=4`
2. fixed `K=8`
3. fixed `K=12`
4. fixed `K=16`
5. oracle best-per-sample over `{4, 8, 12, 16}`

Report:

- overall VQAv2 accuracy for each fixed `K`
- yes/no, number, other breakdowns
- oracle overall accuracy
- oracle answer-type breakdowns
- oracle average selected `K`
- accuracy vs average budget tradeoff

## Run Shape

- one new training run: variable-K late semantic bottleneck
- downstream eval-only passes on the resulting checkpoint for each supported `K`
- one oracle aggregation pass over the saved prediction records

Bundle family:

- `logs/mmsemantic_varbudget_v1_*`

Primary outputs:

- `progress.md`
- `timeline.log`
- per-`K` eval JSON and predictions
- oracle summary JSON
- final report markdown

## Success Read

- strong: oracle clearly beats the best fixed `K` and the average oracle budget stays materially below `16`
- moderate: oracle gain exists but is small (`< 0.005`) or needs nearly full budget on most samples
- weak: oracle is nearly flat relative to the best fixed `K`, which means scheduler work is probably not worth near-term engineering time
