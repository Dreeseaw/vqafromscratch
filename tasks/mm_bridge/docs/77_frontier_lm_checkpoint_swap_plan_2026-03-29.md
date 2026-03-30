# Frontier LM Checkpoint Swap Plan

Date: 2026-03-29

## Goal

Test whether the new reasoning-mix LM checkpoints improve the current non-dual-VM frontier when dropped into the existing Cement-style bridge stack.

The comparison points we care about are:

- original LM baseline: [lm_final step_45000](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar)
- new reasoning-mix LM at the real comparison checkpoint: [lm_reasoningmix_v1_20260328_1618 step_45000](/home/wdree/percy/vqafromscratch/logs/lm_reasoningmix_v1_20260328_1618/step_45000.tar)
- new reasoning-mix LM at the “for shits and giggles” checkpoint: [lm_reasoningmix_v1_20260328_1618 step_300000](/home/wdree/percy/vqafromscratch/logs/lm_reasoningmix_v1_20260328_1618/step_300000.tar)

## Current Frontier to Use

Use the current non-dual-VM frontier only.

That means:

- frozen SigLIP-B/16 vision tower
- question_only bridge context
- question_hidden_attn / attnqquery
- perceiver resampler depth `3`
- no dynbudget
- current LM adapter setup unchanged

Do **not** use dual-VM.

The two relevant bridge recipes are:

1. Cement champion bridge recipe
2. Qwen-distilled bridge recipe, because it beat Cement cleanly

Current references:

| System | LM checkpoint | Overall | Yes/No | Number | Other |
|---|---|---:|---:|---:|---:|
| Cement champion | old `45k` | 0.6163 | 0.7589 | 0.4573 | 0.5499 |
| Qwen-distilled bridge | old `45k` | 0.6235 | 0.7634 | 0.4591 | 0.5607 |

So the distilled bridge is the true non-dual-VM frontier. Cement still matters as the undistilled anchor.

## Main Question

Does the new reasoning-mix LM help the existing frontier bridge stack, or is the frontier gain mostly from bridge-side distillation rather than LM pretraining?

There are really two subquestions:

1. `45k vs 45k`: apples-to-apples comparison against the old canonical LM
2. `300k vs 45k`: what happens if we use a much later reasoning-mix checkpoint that is almost certainly overtrained but might still help bridge consumption?

## Recommended Run Matrix

### Tier 1: Real comparison

These are the two must-run experiments.

| Run | Bridge recipe | LM checkpoint | Purpose |
|---|---|---|---|
| A | Cement champion recipe | new reasoning LM `45k` | direct replacement test on undistilled anchor |
| B | Qwen-distilled bridge recipe | new reasoning LM `45k` | direct replacement test on current best non-dual-VM recipe |

Interpretation:

- Run A answers whether the LM alone helps the plain Cement line.
- Run B answers whether the new LM stacks with the stronger Qwen-distilled bridge.

### Tier 2: High-curiosity run

| Run | Bridge recipe | LM checkpoint | Purpose |
|---|---|---|---|
| C | Qwen-distilled bridge recipe | new reasoning LM `300k` | test whether the very late LM checkpoint helps or hurts the current best bridge recipe |

This is the right place for the `300k` test. It should **not** be run first on the weaker Cement line.

If the `300k` checkpoint is better, interesting. If it is worse, that is also useful, because it tells us the bridge prefers the earlier LM geometry.

### Optional Tier 3

Only run if Tier 1 is clearly positive.

| Run | Bridge recipe | LM checkpoint | Purpose |
|---|---|---|---|
| D | Cement champion recipe | new reasoning LM `300k` | curiosity-only symmetry check |

## Training Strategy

Do **not** warm-start from old MM checkpoints.

Reason:

- the perceiver/adapters in the current Cement and distilled runs were trained against the old LM checkpoint
- if we load those bridge weights and only swap the LM, we confound “LM quality” with “bridge-LM geometry mismatch”

So each LM swap should be a fresh bridge-stage run, using the same recipe family but with the new LM checkpoint loaded from the start.

That means:

- fresh bridge init
- frozen SigLIP
- same current bridge hyperparameters
- LM loaded from the selected checkpoint
- train top `2` LM layers + adapters + bridge as usual

## Exact Experimental Order

### Stage 1: Bridge-only runs

Priority order:

1. Run B: distilled bridge recipe + new LM `45k`
2. Run A: Cement recipe + new LM `45k`
3. Run C: distilled bridge recipe + new LM `300k`
4. Run D only if wanted

Why this order:

- Run B is the highest-value result because it tests the new LM on the actual frontier recipe.
- Run A anchors whether the gain is generic or only visible under bridge distillation.
- Run C is the curiosity test once the real `45k` answer is known.

### Stage 2: Compression follow-up

Only do this if at least one Stage 1 run beats its corresponding old-LM baseline.

Compression follow-up should be run on the best Stage 1 bridge result only.

That gives one clean question:

- does the LM swap gain survive `K=8` compression?

No reason to spend GPU on compression for a losing bridge checkpoint.

## Config Guidance

### A. Cement recipe + new LM `45k`

Use the standard current Cement bridge setup:

- SigLIP-B/16
- question_only
- attnqquery
- perceiver depth `3`
- adapters on
- standard `96x2`
- standard full `9000` steps

Only change:

- `--lm_checkpoint logs/lm_reasoningmix_v1_20260328_1618/step_45000.tar`

### B. Distilled bridge recipe + new LM `45k`

Use the same Qwen distillation recipe that produced `0.6235`, but replace the LM checkpoint:

- answer-vocab KL stays on
- same Qwen teacher labels
- same bridge config
- same trainer settings

Only change:

- `--lm_checkpoint logs/lm_reasoningmix_v1_20260328_1618/step_45000.tar`

### C. Distilled bridge recipe + new LM `300k`

Same as B, but:

- `--lm_checkpoint logs/lm_reasoningmix_v1_20260328_1618/step_300000.tar`

## Evaluation

At bridge `step_9000`, run full VQAv2 val with per-category breakdown:

| Run | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| Cement + old LM `45k` | 0.6163 | 0.7589 | 0.4573 | 0.5499 |
| Distilled + old LM `45k` | 0.6235 | 0.7634 | 0.4591 | 0.5607 |
| Cement + new LM `45k` | ? | ? | ? | ? |
| Distilled + new LM `45k` | ? | ? | ? | ? |
| Distilled + new LM `300k` | ? | ? | ? | ? |

The main comparison columns are:

- overall
- other

`other` matters most because:

- the Qwen-distilled bridge win was strongest there
- the reasoning-mix LM should, in principle, help more on open-ended/compositional answers than on yes/no

## Success Criteria

### Strong success

- Distilled + new LM `45k` beats `0.6235` by `>= 0.5` points

Interpretation:

- the new LM is a real frontier upgrade, not just a lateral swap

### Mild success

- Distilled + new LM `45k` is `+0.2` to `+0.5`

Interpretation:

- keep it, but bridge-side distillation is still the bigger lever

### Neutral

- within roughly `±0.2`

Interpretation:

- the bridge is doing most of the heavy lifting, and LM-side reasoning pretraining is not strongly transferring into VQA here

### Negative

- clearly below the old LM baseline

Interpretation:

- the new LM geometry is less compatible with the current bridge recipe
- or the long reasoning-mix training pushed the LM in a direction that hurts this VQA adaptation setting

### 300k interpretation

- if `300k > 45k`: late LM training helped the bridge-consumption regime
- if `300k < 45k`: the bridge prefers the earlier checkpoint geometry and `45k` should stay the practical default

## Recommendation

Run this as a tight, bridge-only checkpoint swap study first.

Recommended first three runs:

1. Distilled bridge + new LM `45k`
2. Cement bridge + new LM `45k`
3. Distilled bridge + new LM `300k`

Then:

- compress only the winner

That keeps the study cheap, clear, and directly tied to the current real frontier.

