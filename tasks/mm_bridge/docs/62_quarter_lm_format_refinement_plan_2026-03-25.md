# Quarter-LM Format-Alignment Refinement Plan

## Goal

Test whether the `12.2M` quarter LM's extra K=8 compression penalty is mostly a format-compatibility problem rather than a hard LM-capacity limit.

Reference from the completed LM shrink sweep:

- quarter bridge: `0.6044`
- quarter compressed: `0.5705`
- compression delta: `-0.0339`

The experiment is split into two phases so we do not spend a 5k-step retrain if a quarter-specific linear remap already shows that the existing quarter bottleneck is close to format-optimal.

## Phase 1: Quarter-Specific Remap Diagnostic

Start from the completed quarter compression checkpoint:

- bridge source: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmsemantic_lmshrink_v1_20260324_221950_bridge_quarter/step_9000.tar)
- compression source: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmsemantic_lmshrink_v1_20260324_221950_compression_quarter/step_3000.tar)
- quarter LM checkpoint: [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/mmsemantic_lmshrink_v1_20260324_221950_pretrain_quarter/step_45000.tar)

Setup:

- freeze VM, perceiver, bottleneck, LM
- disable LM visual adapters
- train only `PrefixRemap`
- 500 steps
- effective batch `192`
- eval every `100`

Decision rule:

- if remap gain over the quarter compressed baseline is `< 0.005` overall, stop and treat the original quarter compression as already well-tuned
- if remap gain is `>= 0.005`, proceed to Phase 2

## Phase 2: Quarter-Specific Format Retraining

Start fresh from the quarter bridge checkpoint, not from the old compressed checkpoint.

Setup:

- frozen SigLIP VM
- frozen quarter bridge/perceiver from the quarter bridge `step_9000`
- randomly initialized `K=8` semantic bottleneck, trainable
- frozen quarter LM
- LM visual adapters disabled in forward pass
- quarter-specific remap from Phase 1 used only as a frozen teacher via `L_format`

Loss:

- `L_total = L_vqa + 0.1 * L_distill + 0.3 * L_format`
- `L_format` annealed `0.3 -> 0.0` over steps `3750-5000`

Training:

- `5000` steps
- effective batch `192`
- eval every `500`
- checkpoint every `500`

## Output

The launcher writes into a single bundle:

- `logs/lmshrink_quarter_format_v1_<timestamp>/`

It records:

- phase 1 remap evals and decision
- phase 2 checkpoint evals
- best checkpoint full eval
- tiny-head probe
- final experiment report

## Success Read

- `>= 0.58` overall: strong result; quarter becomes a serious deployment frontier
- `0.575 - 0.58`: moderate; format work helps, but quarter still trails half meaningfully
- near `0.5705`: weak/no improvement; the quarter gap is mostly true LM-capacity cost
