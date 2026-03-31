## Scope

This report closes Tier 1 experiment bundle 3 + 4 from [81_variable_k_oracle_plan_2026-03-30.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/81_variable_k_oracle_plan_2026-03-30.md). The executed bundle is [mmsemantic_varbudget_v1_20260330_180744](/home/wdree/percy/vqafromscratch/logs/mmsemantic_varbudget_v1_20260330_180744), trained from the fresh-bridge VM-finetuned frontier checkpoint [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar).

The hypothesis was narrow: keep the current SigLIP2-era late semantic-token path and bridge recipe stable, but train one compressor that is prefix-robust across multiple LM-side token budgets. Then measure the oracle upper bound from per-sample best-budget selection before spending time on any learned or heuristic scheduler.

This is not perceiver-side dynbudget. The visual backbone, bridge evidence path, and latent capacity stay fixed. The only variable is how many ordered question-conditioned semantic tokens are exposed to the LM interface at runtime.

## What Changed

Training path:

- semantic bottleneck now supports a max exported sequence with a smaller active LM-facing prefix budget
- supported training budgets were `K in {4, 8, 12, 16}`
- the run sampled one budget per batch during training and exposed only the first `K` tokens to the LM
- the full max-token bottleneck stayed intact internally so this remained a variable-prefix experiment, not a new architecture

Eval path:

- fixed-budget eval can now override the active semantic budget at inference time
- a new oracle sweep script runs one checkpoint at each supported `K`
- the oracle reducer computes best-per-sample achievable accuracy across budgets, the average selected budget, and the budget histogram under a smallest-`K` tie break

Primary code paths touched:

- [models/bridge.py](/home/wdree/percy/vqafromscratch/models/bridge.py)
- [train/mm.py](/home/wdree/percy/vqafromscratch/train/mm.py)
- [mm_format_alignment_eval.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_format_alignment_eval.py)
- [mm_semantic_budget_oracle_eval.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_budget_oracle_eval.py)
- [launch_variable_budget_oracle_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_variable_budget_oracle_v1.sh)

## What Ran

Source bridge:

- fresh bridge frontier: `SigLIP2 + lm_final + Qwen-KD + fresh bridge top-2-block VM finetune`
- checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar)
- full-eval VQAv2 baseline: `0.6470`
- source GQA: `0.5180`
- source OCR subset: `0.3118`

Variable-budget compression run:

- run dir: [mmsemantic_varbudget_v1_20260330_180744_varprefix_k16](/home/wdree/percy/vqafromscratch/logs/mmsemantic_varbudget_v1_20260330_180744_varprefix_k16)
- max semantic width: `16`
- sampled train budgets: `4, 8, 12, 16`
- freeze mode: `semantic_bottleneck_only`
- training steps: `3000`
- source init: fresh-bridge VM-finetune frontier checkpoint above

Oracle eval bundle:

- bundle dir: [mmsemantic_varbudget_v1_20260330_180744](/home/wdree/percy/vqafromscratch/logs/mmsemantic_varbudget_v1_20260330_180744)
- eval budgets: `4, 8, 12, 16`
- output summary: [oracle_eval_summary.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_varbudget_v1_20260330_180744/oracle_eval_summary.json)

## Fixed-K Results

| Budget K | Overall | Yes/No | Number | Other | Delta vs source bridge |
|---|---:|---:|---:|---:|---:|
| 4 | 0.6278 | 0.7846 | 0.4831 | 0.5469 | -0.0192 |
| 8 | 0.6289 | 0.7857 | 0.4833 | 0.5482 | -0.0181 |
| 12 | 0.6285 | 0.7856 | 0.4839 | 0.5472 | -0.0185 |
| 16 | 0.6282 | 0.7852 | 0.4832 | 0.5472 | -0.0188 |

Read:

- the fixed-budget curve is almost flat across `K in {4, 8, 12, 16}`
- the best fixed point is `K=8`, but only by `+0.0011` over `K=4`
- larger fixed budgets do not help on average; `K=12` and `K=16` both trail `K=8`
- so the ordered-prefix training mostly worked: early tokens carry almost all of the average utility
- however, fixed compression still remains about `1.8` to `1.9` points below the uncompressed source bridge

## Oracle Read

Oracle over `{4, 8, 12, 16}`:

- oracle overall: `0.6521`
- oracle yes/no: `0.8117`
- oracle number: `0.5088`
- oracle other: `0.5686`
- oracle average selected `K`: `4.2214`
- oracle delta vs best fixed budget: `+0.0232`
- oracle delta vs source uncompressed bridge: `+0.0051`

Selected-budget distribution with smallest-`K` tie break:

- `K=4`: `207,022 / 214,354` samples (`96.58%`)
- `K=8`: `4,205` samples (`1.96%`)
- `K=12`: `1,719` samples (`0.80%`)
- `K=16`: `1,408` samples (`0.66%`)

Accuracy vs average budget:

| Mode | Average K | Overall |
|---|---:|---:|
| fixed `K=4` | 4.0000 | 0.6278 |
| fixed `K=8` | 8.0000 | 0.6289 |
| fixed `K=12` | 12.0000 | 0.6285 |
| fixed `K=16` | 16.0000 | 0.6282 |
| oracle | 4.2214 | 0.6521 |

## Interpretation

The important result is not the fixed-`K` table; it is the combination of a nearly flat fixed-`K` curve and a very strong oracle upper bound.

First, fixed `K` barely moves. That means the prefix-robust training did what it was supposed to do: the first few semantic tokens now carry most of the information the LM needs on average. In that sense, the ordered-token experiment is a success.

Second, more tokens are not monotonically better as a global policy. The best fixed budget is `K=8`, not `K=16`, and the spread from worst to best fixed budget is only about `0.11` points. That suggests later tokens are useful only selectively and can otherwise add noise or at least fail to help.

Third, the oracle upside is large. Oracle reaches `0.6521`, which is `+2.32` points over the best fixed compressed setting and even `+0.51` over the uncompressed fresh-bridge source. The average oracle budget is only `4.22`, with `96.6%` of samples choosing `K=4` under smallest-`K` tie break. So the story is not "use more tokens everywhere." The story is "most samples are already solved by a tiny prefix, but a small minority really does want extra semantic capacity."

That is exactly the pattern that justifies scheduler work. If the oracle had only gained a few tenths while needing near-full average budget, it would not be worth it. Instead, the upper bound says there is meaningful heterogeneity in required LM-side semantic budget.

The caveat is that this oracle is intentionally optimistic. It gets to pick the best budget after seeing which prediction is correct. A practical heuristic or learned scheduler will not recover all `+2.32`. So the right conclusion is not "we have already solved dynamic budgeting." The right conclusion is "there is real headroom here, and it is large enough to justify the next scheduler experiment."

## Recommendation

Dynamic LM-side semantic budgeting looks real enough to pursue.

Recommended next experiment:

- keep this exact checkpoint and budget set
- default to `K=4`
- add a cheap post-hoc escalation policy that only sends a small subset of questions to `K=8` or `K=16`
- start with non-learned uncertainty signals from the `K=4` answer path before building a learned budget predictor

Why this next step fits the evidence:

- oracle says almost all samples should stay at `K=4`
- the gain comes from identifying a small hard tail, not from globally raising budget
- since fixed `K=16` is not better than fixed `K=8`, the scheduler should be selective rather than monotonic

So the answer to the original question is yes: variable per-sample LM-side semantic budgeting appears promising enough to justify later heuristic or learned schedulers. The clean next move is a simple `K=4`-first escalation experiment on top of this same variable-prefix checkpoint, because that tests whether even a crude scheduler can convert part of the oracle gap into real accuracy while keeping the average budget near `4`.
