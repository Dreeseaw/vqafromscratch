# Dynamic Budget Overnight Bundle Plan

Date:
- `2026-03-30`

Goal:
- answer whether cheap LM-side semantic-budget schedulers can recover a meaningful part of the oracle gap on the clean matched-`9k` SigLIP2 frontier line
- measure whether `K=4` remains the right practical default
- quantify whether `K=8` is enough for most of the tail or whether `K=16` matters
- probe whether extreme LM-side semantic compression (`K=2`, `K=1`) is merely destructive or scientifically interesting

Revision note:

- the first launch of this bundle incorrectly used the higher-compute `18k` stacked bridge as the new training stem
- that launch was aborted and excluded from scientific comparison
- this corrected overnight bundle uses only the clean `9k` frontier bridge as the new training source so all new comparisons stay source-matched and compute-comparable

## Starting Points

Clean-source variable-prefix checkpoint already available:
- checkpoint: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmsemantic_varbudget_v1_20260330_180744_varprefix_k16/step_3000.tar)
- source line: clean `9k` frontier bridge
- trained budgets: `{4,8,12,16}`

New clean-source variable-prefix training source:
- checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar)
- source line: current clean `9k` frontier bridge
- new train budgets: `{2,4,8,16}`

## Planned Bundle

1. Train one new clean-`9k`-source variable-prefix compressor.
- keep the existing SigLIP2 semantic-bottleneck compression recipe stable
- semantic bottleneck only
- exported width `16`
- sampled training budgets `{2,4,8,16}`
- `3000` compression-stage steps

2. Run full fixed-K plus oracle eval on both clean-source variable-prefix checkpoints.
- existing baseline checkpoint: clean `9k` source, train budgets `{4,8,12,16}`
- new checkpoint: clean `9k` source, train budgets `{2,4,8,16}`
- fixed `K in {2,4,8,16}`
- `K=1` as an eval-only probe
- oracle over `{2,4,8,16}`
- save per-budget JSON summaries and per-sample JSONL records
- attach uncertainty stats only on `K=4` and `K=2`, since those are the scheduler base paths

3. Run offline scheduler sweeps on both checkpoints from saved eval artifacts.
- main policy family:
  - `K=4 -> K=8`
  - `K=4 -> K=16`
  - `K=4 -> K=8 -> K=16`
- side probe:
  - `K=2 -> K=4`
  - `K=2 -> K=8`
  - `K=2 -> K=4 -> K=8`
- signals:
  - `K`-base mean confidence
  - mean top-1 vs top-2 margin
  - mean entropy
  - one cheap OCR/count-aware hybrid
- thresholding:
  - fixed percentile grids
  - tune slice = `question_id % 5 == 0`
  - report slice = remaining `80%`

4. Produce hard-tail analysis.
- oracle-selected tail and recommended simple scheduler tail
- answer-type breakdown
- OCR-like question fraction
- question-length buckets
- top question types and lexical prefixes
- budget histograms for the tail

Intentionally excluded from this corrected overnight bundle:

- the higher-compute stacked `18k` transfer question
- that is scientifically separate because it changes the source stem compute regime and would confound any claim about dynamic-budget behavior itself
- if needed later, run it as a clearly labeled higher-compute curiosity bundle after the clean `9k` story is settled

## Runtime / Perf Choices

Compression training settings:
- follow the existing stable semantic-bottleneck compression recipe already used in the prior clean varprefix run, keeping the broader recipe fixed apart from the new budget set
- `batch_size=96`, `grad_accum_steps=2`, `num_workers=2`, `prefetch_factor=1`, `pin_memory=0`

Eval-only settings:
- use `eval_use_kv_cache=1`
- use `eval_kv_cache_mode=batched`
- use conservative posthoc eval loader settings to avoid overnight fragility on the `16 GB` GPU

## Tracker Contract

- create the bundle directory, timeline, and first run identity before launch
- keep train and every meaningful eval/analysis stage as separate runs
- rebuild DuckDB after run starts and after each stage boundary so the tracker can tail the active bundle cleanly

## Deliverables

- code changes for the eval suite, scheduler sweep, and overnight launcher
- trackable corrected overnight experiment bundle
- final report at [84_dynamic_budget_overnight_bundle_report_2026-03-30.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/84_dynamic_budget_overnight_bundle_report_2026-03-30.md)
