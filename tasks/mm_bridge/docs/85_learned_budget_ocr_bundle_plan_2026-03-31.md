# Clean-Regime OCR-Aware Learned Budget Bundle Plan

Date:
- `2026-03-31`

Primary question:
- can OCR/chart-aware compression plus a learned semantic-budget predictor turn the clean-`9k` dynamic-budget oracle headroom into a practical win on the clean matched-compute frontier line

Scientific comparison rule:

- source checkpoint must be the clean matched-`9k` bridge champion
- no stacked `18k/21k` continuation source
- no fresh from-init bridge pretraining branch
- keep the regime comparable:
  - clean `9k` bridge source
  - `3000` compression-stage steps
  - learned budget predictor trained on top of that compression family

## Starting Points

Source bridge:
- checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar)
- full eval: `0.6470`

Current clean-source variable-prefix references:
- existing clean varprefix baseline: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmsemantic_varbudget_v1_20260330_180744_varprefix_k16/step_3000.tar)
- latest clean `{2,4,8,16}` varprefix checkpoint: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_train/step_3000.tar)

## Planned Bundle

1. Train one new OCR/chart-aware clean-source variable-prefix compressor.
- same semantic bottleneck architecture
- source = clean `9k` bridge checkpoint above
- `3000` compression-stage steps
- budgets `{2,4,8,16}`
- max exported width `16`
- random budget per batch
- training mix target:
  - `65%` VQAv2
  - `25%` ChartQA
  - `10%` TextOCR readout QA

2. Run structured fixed-budget plus oracle evals.
- VQAv2 fixed `K in {2,4,8,16}` plus `K=1` probe and oracle over `{2,4,8,16}`
- direct VQAv2 comparison against the existing clean-source variable-prefix baseline
- panel evals on the new checkpoint:
  - ChartQA
  - TextOCR readout
  - GQA with slices
  - heuristic OCR subset on VQAv2
  - semantic probe

3. Export low-budget features for learned prediction.
- export pooled `K=2`, `K=4`, and `K=8` semantic-prefix features
- export pooled question features
- reuse low-budget answer-path uncertainty stats from fixed eval artifacts
- oracle-selected budgets are the labels

4. Train one lightweight learned cascade predictor.
- node 1: stay at `K=2` vs escalate
- node 2: stay at `K=4` vs escalate
- node 3: choose `K=8` vs `K=16`
- inputs:
  - pooled semantic-prefix features
  - pooled question features
  - low-budget uncertainty stats
  - lexical OCR/chart/count cues
  - dataset id
- split policy:
  - train / validation / report split by deterministic question-id hash modulo
  - keep report disjoint from threshold/model selection

5. Compare against fixed and cheap baselines.
- fixed `K=2`
- fixed `K=4`
- fixed `K=8`
- best cheap scheduler from prior clean varprefix work
- cheap scheduler rerun on the new VQAv2 checkpoint if inexpensive

## Implementation Notes

Keep code changes local:

- add ChartQA eval/train loader support from the existing DuckDB QA registry
- add TextOCR readout QA support as crop-based OCR QA, not a new captioning task
- add a dataset-aware exact scorer that preserves punctuation and decimals
- add a generic multi-dataset QA batch sampler for the compression-stage mix
- add feature export and learned predictor scripts as post-hoc tools on top of the existing eval artifacts

## Runtime / Perf Defaults

Use the strongest known stable clean-`9k` family settings unless the new mix forces a safer fallback:

- train: `batch_size=96`, `grad_accum_steps=2`
- eval cache: `eval_use_kv_cache=1`, `eval_kv_cache_mode=batched`
- start conservative on data workers for the new mixed dataset path
- every meaningful eval or predictor stage gets its own run id and logfile

## Tracker Contract

- create the bundle directory, first run dir, logfile, and timeline markers before launch
- keep train, eval-only, predictor-train, and panel eval stages as separate runs
- rebuild DuckDB after launch and after stage boundaries

## Expected Deliverables

- code changes for OCR/chart QA loading, exact scoring, feature export, learned predictor, and launcher
- one completed clean-regime bundle
- final report at [86_learned_budget_ocr_bundle_report_2026-03-31.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/86_learned_budget_ocr_bundle_report_2026-03-31.md)

## Decision Rule

The bundle is successful if it answers all of:

- does a learned predictor beat the prior cheap scheduler by a meaningful amount
- is `K=2` now the best practical tiny-prefix default
- does OCR/chart-aware compression materially improve the hard tail
- is the remaining budget tail mostly OCR/chart, or still broader than that
