## Scope

This report closes the corrected clean-`9k` overnight dynamic-budget bundle from [83_dynamic_budget_overnight_bundle_plan_2026-03-30.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/83_dynamic_budget_overnight_bundle_plan_2026-03-30.md). The finished bundle is [mmdynbudget_clean9k_overnight_v1_20260330_232048](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048), built on the clean matched-`9k` frontier source checkpoint [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar).

The comparison here is intentionally source-matched and compute-comparable. The earlier higher-compute stacked branch was excluded from scientific comparison after the plan correction. So this report answers the practical scheduler and extreme-compression questions on the clean `9k` SigLIP2 frontier line, not on a mixed-compute stacked line.

The hypothesis was:

- train one prefix-robust semantic-token compressor with budgets `K in {2,4,8,16}`
- compare it against the existing clean-source variable-prefix checkpoint trained on `{4,8,12,16}`
- measure fixed-`K`, oracle, cheap scheduler, and hard-tail behavior
- decide whether dynamic budgeting is practically exploitable or still mostly an oracle curiosity

## What Changed

Training path:

- the semantic bottleneck already supports ordered-prefix LM-side token truncation from one max-width sequence
- a new clean-source compression checkpoint was trained with sampled budgets `K in {2,4,8,16}`
- the architecture stayed the same: semantic bottleneck only, exported width `16`, `3000` compression-stage steps

Eval and analysis path:

- full fixed-budget eval ran at `K=1,2,4,8,16`
- oracle eval reduced over `{2,4,8,16}`
- post-hoc scheduler sweep evaluated cheap uncertainty rules from the `K=4` path and compact side probes from the `K=2` path
- thresholds were tuned on `qid % 5 == 0` and reported on the disjoint remaining `80%`

Bundle/runtime support:

- eval-only stages were kept as separate runs
- the scheduler sweep was patched to stream JSONL artifacts instead of loading giant payloads into RAM
- the overnight launcher was patched to resume from existing completed artifacts after the scheduler OOM, instead of rerunning completed stages

Primary code paths touched in this bundle family:

- [models/bridge.py](/home/wdree/percy/vqafromscratch/models/bridge.py)
- [train/mm.py](/home/wdree/percy/vqafromscratch/train/mm.py)
- [mm_semantic_budget_eval_suite.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_budget_eval_suite.py)
- [mm_semantic_budget_scheduler_sweep.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_budget_scheduler_sweep.py)
- [launch_dynamic_budget_overnight_bundle_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_dynamic_budget_overnight_bundle_v1.sh)

## What Ran

Source bridge:

- clean `9k` frontier source: [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2)
- checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar)
- source full eval: `0.6470`

Existing clean-source baseline variable-prefix checkpoint:

- run: [mmsemantic_varbudget_v1_20260330_180744_varprefix_k16](/home/wdree/percy/vqafromscratch/logs/mmsemantic_varbudget_v1_20260330_180744_varprefix_k16)
- checkpoint: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmsemantic_varbudget_v1_20260330_180744_varprefix_k16/step_3000.tar)
- train budgets: `{4,8,12,16}`

New clean-source variable-prefix checkpoint:

- run: [mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_train](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_train)
- checkpoint: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_train/step_3000.tar)
- train budgets: `{2,4,8,16}`

Completed bundle stages:

- fixed/oracle eval for the new checkpoint: [mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_fixed_eval](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_fixed_eval)
- scheduler sweep for the new checkpoint, resumed after the OOM: [mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_scheduler_resume1](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_scheduler_resume1)
- fixed/oracle eval for the baseline checkpoint: [mmdynbudget_clean9k_overnight_v1_20260330_232048_baseline416_fixed_eval_resume1](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_baseline416_fixed_eval_resume1)
- scheduler sweep for the baseline checkpoint: [mmdynbudget_clean9k_overnight_v1_20260330_232048_baseline416_scheduler_resume1](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_baseline416_scheduler_resume1)
- finished timeline: [timeline.log](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048/timeline.log)

Structured outputs:

- new checkpoint eval summary: [summary.json](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048/frontier24_eval/summary.json)
- new checkpoint scheduler summary: [frontier24_eval_scheduler_summary.json](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048/frontier24_eval_scheduler_summary.json)
- baseline eval summary: [summary.json](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048/baseline416_eval/summary.json)
- baseline scheduler summary: [baseline416_eval_scheduler_summary.json](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048/baseline416_eval_scheduler_summary.json)

Note on requested OCR / GQA tail breakdowns:

- this bundle did not run separate OCR-subset or GQA-subset evals for the new checkpoints
- the hard-tail section therefore uses the saved val-set answer-type breakdowns plus the scheduler sweep's OCR-like lexical proxy, not dedicated OCR or GQA eval sets

## Fixed-K And Oracle Results

### New clean-source `{2,4,8,16}` checkpoint

| Policy | Avg K | Overall | Yes/No | Number | Other | Delta vs source bridge |
|---|---:|---:|---:|---:|---:|---:|
| fixed `K=1` | 1.0000 | `0.3033` | `0.0025` | `0.3916` | `0.5090` | `-0.3437` |
| fixed `K=2` | 2.0000 | `0.6287` | `0.7865` | `0.4819` | `0.5475` | `-0.0183` |
| fixed `K=4` | 4.0000 | `0.6285` | `0.7858` | `0.4819` | `0.5476` | `-0.0185` |
| fixed `K=8` | 8.0000 | `0.6293` | `0.7866` | `0.4828` | `0.5484` | `-0.0177` |
| fixed `K=16` | 16.0000 | `0.6287` | `0.7872` | `0.4832` | `0.5466` | `-0.0183` |
| oracle over `{2,4,8,16}` | 2.2509 | `0.6545` | `0.8174` | `0.5062` | `0.5699` | `+0.0075` |

Oracle notes:

- best fixed budget: `K=8` at `0.6293`
- oracle delta vs best fixed: `+0.0252`
- oracle selected-budget histogram: `K=2` `206,577`, `K=4` `2,950`, `K=8` `2,462`, `K=16` `2,365`
- oracle selected-budget fractions: `K=2` `96.37%`, `K=4` `1.38%`, `K=8` `1.15%`, `K=16` `1.10%`

### Existing clean-source `{4,8,12,16}` checkpoint re-evaled on `{2,4,8,16}`

| Policy | Avg K | Overall | Yes/No | Number | Other | Delta vs source bridge |
|---|---:|---:|---:|---:|---:|---:|
| fixed `K=1` | 1.0000 | `0.3015` | `0.0030` | `0.3965` | `0.5036` | `-0.3455` |
| fixed `K=2` | 2.0000 | `0.6264` | `0.7816` | `0.4825` | `0.5463` | `-0.0206` |
| fixed `K=4` | 4.0000 | `0.6278` | `0.7846` | `0.4831` | `0.5469` | `-0.0192` |
| fixed `K=8` | 8.0000 | `0.6289` | `0.7857` | `0.4833` | `0.5482` | `-0.0181` |
| fixed `K=16` | 16.0000 | `0.6282` | `0.7852` | `0.4832` | `0.5472` | `-0.0188` |
| oracle over `{2,4,8,16}` | 2.2350 | `0.6544` | `0.8151` | `0.5093` | `0.5706` | `+0.0074` |

Oracle notes:

- best fixed budget: `K=8` at `0.6289`
- oracle delta vs best fixed: `+0.0255`
- oracle selected-budget histogram: `K=2` `205,945`, `K=4` `3,758`, `K=8` `2,782`, `K=16` `1,869`
- oracle selected-budget fractions: `K=2` `96.08%`, `K=4` `1.75%`, `K=8` `1.30%`, `K=16` `0.87%`

Read across both checkpoints:

- adding `K=2` into training made `K=2` much stronger: `0.6287` vs `0.6264`, a `+0.0023` gain
- it also slightly improved every fixed setting: `+0.0004` to `+0.0007` over the older clean-source variable-prefix checkpoint
- the oracle upper bound barely changed: `0.6545` vs `0.6544`
- `K=1` is still basically degenerate as a real policy, even though it retains some non-trivial utility on non-yes/no categories

## Scheduler Pareto Tables

The scheduler tables below use the disjoint report split, not the tuning slice. That is the honest number to use for the "can a cheap scheduler work?" question.

### New clean-source `{2,4,8,16}` checkpoint

| Policy | Overall | Yes/No | Number | Other | Avg K | Escalated | Delta vs fixed `K=4` | Gap to oracle recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed `K=4` report baseline | `0.6299` | `0.7873` | `0.4836` | `0.5479` | `4.0000` | `0.0%` | `+0.0000` | `0.00%` |
| fixed `K=8` report best fixed | `0.6305` | n/a | n/a | n/a | `8.0000` | `100.0%` | `+0.0006` | n/a |
| recommended cheap main: `K=4 -> K=8`, margin, top `10%` | `0.6301` | `0.7874` | `0.4839` | `0.5482` | `4.4000` | `10.0%` | `+0.0003` | `1.08%` |
| best-accuracy cheap main: `K=4 -> K=8`, margin, top `20%` | `0.6304` | `0.7874` | `0.4842` | `0.5487` | `4.8000` | `20.0%` | `+0.0005` | `2.11%` |
| recommended compact side probe: `K=2 -> K=4`, hybrid, top `20%` | `0.6298` | `0.7868` | `0.4840` | `0.5481` | `2.4000` | `20.0%` | `+0.0001` vs fixed `K=2` | `0.52%` |
| best-accuracy side probe: `K=2 -> K=8`, entropy, top `20%` | `0.6300` | `0.7868` | `0.4840` | `0.5484` | `3.2000` | `20.0%` | `+0.0013` vs fixed `K=2` | `1.09%` |

Relevant upper bounds from the same checkpoint:

- oracle over `4 -> 8` only: `0.6414` at average `K=4.0733`
- oracle over `4 -> 8 -> 16`: `0.6506` at average `K=4.2298`
- oracle over `2 -> 4` only: `0.6386` at average `K=2.0277`
- oracle over `2 -> 4 -> 8`: `0.6467` at average `K=2.0969`

### Existing clean-source `{4,8,12,16}` checkpoint

| Policy | Overall | Yes/No | Number | Other | Avg K | Escalated | Delta vs fixed `K=4` | Gap to oracle recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed `K=4` report baseline | `0.6286` | `0.7847` | `0.4844` | `0.5471` | `4.0000` | `0.0%` | `+0.0000` | `0.00%` |
| fixed `K=8` report best fixed | `0.6297` | n/a | n/a | n/a | `8.0000` | `100.0%` | `+0.0011` | n/a |
| recommended cheap main: `K=4 -> K=8`, confidence, top `10%` | `0.6289` | `0.7847` | `0.4847` | `0.5477` | `4.4000` | `10.0%` | `+0.0003` | `1.24%` |
| best-accuracy cheap main: `K=4 -> K=8`, confidence, top `20%` | `0.6291` | `0.7847` | `0.4846` | `0.5480` | `4.8000` | `20.0%` | `+0.0005` | `1.67%` |
| recommended compact side probe: `K=2 -> K=4 -> K=8`, margin, `10%` total | `0.6272` | `0.7811` | `0.4840` | `0.5470` | `2.4000` | `10.0%` | `-0.0000` vs fixed `K=2` | `-0.07%` |
| best-accuracy side probe: `K=2 -> K=8`, confidence, top `20%` | `0.6278` | `0.7816` | `0.4844` | `0.5479` | `3.2000` | `20.0%` | `+0.0014` vs fixed `K=2` | `2.22%` |

Relevant upper bounds from the same checkpoint:

- oracle over `4 -> 8` only: `0.6419` at average `K=4.0787`
- oracle over `4 -> 8 -> 16`: `0.6494` at average `K=4.2085`
- oracle over `2 -> 4` only: `0.6392` at average `K=2.0353`
- oracle over `2 -> 4 -> 8`: `0.6484` at average `K=2.1133`

Read:

- cheap schedulers improved over fixed `K=4`, but only by `+0.0003` to `+0.0005` on the honest report split
- no cheap scheduler beat the fixed `K=8` report baseline
- the recovered fraction of the available oracle gap stayed tiny: roughly `1%` to `2%`, not `20%` to `50%`
- the newer `{2,4,8,16}` training made the `K=2` side branch much more viable than before

## Where The Hard Tail Lives

### Oracle tail beyond `K=4`

New checkpoint:

- oracle-selected tail size beyond `K=4`: `4,827 / 214,354` samples (`2.25%`)
- split inside that tail: `K=8` `2,462`, `K=16` `2,365`
- answer-type mix: `46.1%` other, `41.8%` yes/no, `12.1%` number
- OCR-like fraction: `1.76%`
- top prefixes: `what is`, `how many`, `is this`, `what color`, `is there`, `where is`, `does this`, `what kind`

Baseline checkpoint:

- oracle-selected tail size beyond `K=4`: `4,651 / 214,354` samples (`2.17%`)
- split inside that tail: `K=8` `2,782`, `K=16` `1,869`
- answer-type mix: `52.4%` other, `32.7%` yes/no, `14.9%` number
- OCR-like fraction: `1.72%`
- top prefixes: `what is`, `how many`, `is this`, `what color`, `where is`, `what kind`, `what are`, `is there`

Main takeaways:

- the hard tail is not mainly OCR by the lexical proxy used here
- it is also not mainly number questions
- the true oracle tail contains a lot more yes/no and broad "other" questions than the cheap schedulers guessed
- `K=16` matters for a tiny special subset: about `1.10%` of all samples on the new checkpoint and `0.87%` on the baseline

### Oracle tail beyond `K=2`

New checkpoint:

- oracle-selected tail beyond `K=2`: `7,777` samples (`3.63%`)
- tail budget split: `K=4` `2,950`, `K=8` `2,462`, `K=16` `2,365`

Baseline checkpoint:

- oracle-selected tail beyond `K=2`: `8,409` samples (`3.92%`)
- tail budget split: `K=4` `3,758`, `K=8` `2,782`, `K=16` `1,869`

Interpretation:

- `K=2` is surprisingly viable after variable-prefix training
- under oracle selection, about `96%` of all samples stay at `K=2`
- including `K=2` during training improves `K=2` materially without collapsing the higher-budget path

### What The Cheap Scheduler Actually Escalates

New checkpoint recommended main (`K=4 -> K=8`, top `10%` by margin):

- escalated tail answer-type mix: `47.5%` other, `44.3%` number, `8.2%` yes/no
- OCR-like fraction: `7.33%`

Baseline checkpoint recommended main (`K=4 -> K=8`, top `10%` by confidence):

- escalated tail answer-type mix: `62.6%` other, `37.2%` number, `0.2%` yes/no
- OCR-like fraction: `10.71%`

This mismatch matters. The cheap uncertainty rules over-target number/OCR-ish cases and miss much of the true yes/no-plus-other oracle tail.

## Interpretation

### 1. A cheap scheduler does not recover a meaningful chunk of the oracle gap

This is the main practical answer from the bundle. The oracle gap is real and large, roughly `+2.5` points over the best fixed compressed setting. But the cheap post-hoc schedulers recovered only about `+0.0003` to `+0.0005` absolute over fixed `K=4` on held-out reporting, which is only about `1%` to `2%` of the available oracle headroom.

So the answer to "is dynamic budgeting already practically exploitable with cheap hand rules?" is mostly no.

### 2. The oracle story is still real

This is not a negative result for dynamic budgeting itself. Both checkpoints reached essentially the same oracle:

- new `{2,4,8,16}` checkpoint: `0.6545`
- older `{4,8,12,16}` checkpoint re-evaled on `{2,4,8,16}`: `0.6544`

That oracle is `+0.0074` to `+0.0075` over the uncompressed clean `9k` source bridge while using average budget only around `2.24` to `2.25`. So the headroom is real. The problem is scheduler quality, not absence of per-sample heterogeneity.

### 3. `K=4` is no longer the only interesting tiny-prefix regime

Before this bundle, `K=4` looked like the obvious tiny-prefix default. After training with `K=2` in the budget set, that is no longer clearly true.

On the new checkpoint:

- fixed `K=2`: `0.6287`
- fixed `K=4`: `0.6285`
- fixed `K=8`: `0.6293`

So `K=2` is only `0.0006` behind the best fixed point and actually edges out fixed `K=4`. That makes `K=2` scientifically interesting and potentially practical when budget matters more than the last few basis points of accuracy.

### 4. `K=8` matters in practice; `K=16` matters only for a tiny tail

For cheap schedulers, every competitive main policy was really a `K=4 -> K=8` story. The `K=16` options were not worth their cost under these simple signals.

But the oracle says `K=16` is not useless:

- new checkpoint oracle tail beyond `K=4`: `2,365` samples chose `K=16`, almost as many as the `2,462` that chose `K=8`
- baseline oracle tail beyond `K=4`: `1,869` samples chose `K=16`

So `K=16` matters for a very small subset, but cheap rules could not find that subset reliably.

### 5. The hard tail is broader than "OCR"

The oracle tail was mostly non-OCR by the available proxy and had substantial yes/no content. That argues against an OCR-specific next step as the primary follow-up. OCR may still matter for a slice of the escalated set, but it is not the dominant explanation for the oracle gap.

### 6. `K=1` is still too collapsed

`K=1` kept some non-zero utility on non-yes/no classes, but overall accuracy around `0.30` is far too low to treat as a serious practical regime right now.

## Recommendation

The next single experiment should be a learned budget predictor on top of the new clean-source `{2,4,8,16}` variable-prefix checkpoint.

Why this is the right next step:

- the oracle headroom is large and stable across both checkpoints
- cheap hand-built schedulers clearly underperform
- the hard tail is not simple enough to capture with OCR-ish or raw uncertainty thresholds alone
- `K=2` is now strong enough that a learned predictor could choose among `2`, `4`, `8`, and maybe `16` in a genuinely meaningful way

What not to do next:

- do not spend another full bundle on hand-built scheduler rules; this bundle already showed that simple confidence/margin/entropy heuristics are weak
- do not prioritize an OCR-only recovery line first; the oracle tail is too broad for that to be the main story
- do not prioritize a standalone extreme-compression family before testing learned scheduling, because this bundle already answered the core scientific question about `K=2`

Concrete next experiment:

- use the new checkpoint [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_train/step_3000.tar)
- treat oracle-selected budget over `{2,4,8,16}` as supervision
- train a lightweight budget predictor from question features plus cheap base-path uncertainty features
- evaluate against:
  - fixed `K=2`
  - fixed `K=8`
  - the best cheap `K=4 -> K=8` rule from this report
  - the best cheap `K=2`-side rule from this report

Bottom line:

- dynamic LM-side semantic budgeting is not just an oracle curiosity
- but the practical lever is not unlocked by simple hand-built schedulers
- `K=2` is the most interesting new scientific result from this bundle
- the next real test is whether a learned predictor can convert the very large oracle headroom into a deployable policy on the clean `9k` frontier line
