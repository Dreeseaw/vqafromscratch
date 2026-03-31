## Scope

This report closes the clean-regime OCR-aware learned-budget bundle from [85_learned_budget_ocr_bundle_plan_2026-03-31.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/85_learned_budget_ocr_bundle_plan_2026-03-31.md). The finished bundle is [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335), trained from the clean matched-compute `9k` frontier bridge [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar).

This was a scientifically matched downstream bundle:

- source bridge = clean `9k` frontier champion
- compression stage = `3000` steps
- semantic budgets = `{2,4,8,16}`
- learned predictor trained only on top of that compression family

The question was not whether dynamic budgeting has oracle headroom. We already knew it did. The question here was whether an OCR/chart-aware compression run plus a lightweight learned scheduler could turn that headroom into a practical win without leaving the clean `9k` comparison regime.

## What Changed

Compression-stage training:

- trained one new semantic-bottleneck-only variable-prefix checkpoint from the clean `9k` source
- kept the architecture stable and max exported width at `16`
- sampled one LM-side semantic budget per batch from `{2,4,8,16}`
- changed the training mix to `65%` VQAv2, `25%` ChartQA, `10%` TextOCR readout QA

Learned-budget path:

- added feature export over low-budget paths
- trained a lightweight cascade predictor
- cascade structure: `K=2` stay vs escalate
- cascade structure: `K=4` stay vs escalate
- cascade structure: `K=8` vs `K=16`
- inputs combined pooled prefix features, question features, low-budget uncertainty features, lexical OCR/chart cues, and dataset id

Panel / analysis support:

- full fixed-`K` plus oracle evals for VQAv2, ChartQA, and TextOCR
- GQA slice readouts at `K=2` and `K=8`
- semantic probe readouts at `K=2` and `K=8`
- heuristic OCR-subset readout at `K=2`
- matched baseline evals for the prior clean `{2,4,8,16}` variable-prefix checkpoint

Main code paths in this bundle:

- [train/vqa_data.py](/home/wdree/percy/vqafromscratch/train/vqa_data.py)
- [train/mm.py](/home/wdree/percy/vqafromscratch/train/mm.py)
- [evals/vqa.py](/home/wdree/percy/vqafromscratch/evals/vqa.py)
- [mm_semantic_budget_eval_suite.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_budget_eval_suite.py)
- [mm_semantic_budget_feature_export.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_budget_feature_export.py)
- [mm_semantic_budget_learned_predictor.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_budget_learned_predictor.py)
- [mm_gqa_eval.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_gqa_eval.py)
- [mm_ocr_subset_eval.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_ocr_subset_eval.py)
- [mm_semantic_probe.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_probe.py)
- [launch_learned_budget_ocr_bundle_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_learned_budget_ocr_bundle_v1.sh)

## What Ran

Bundle / train:

- bundle: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335)
- OCR-aware train run: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_train](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_train)
- trained checkpoint: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_train/step_3000.tar)

New-checkpoint eval-only runs:

- VQAv2 fixed/oracle: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_vqa_eval_resume3](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_vqa_eval_resume3)
- ChartQA fixed/oracle: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_chartqa_eval](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_chartqa_eval)
- TextOCR fixed/oracle: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_textocr_eval](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_textocr_eval)
- cheap schedulers: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_vqa_scheduler](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_vqa_scheduler), [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_chartqa_scheduler](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_chartqa_scheduler), [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_textocr_scheduler](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_textocr_scheduler)
- feature export: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_vqa_features](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_vqa_features), [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_chartqa_features](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_chartqa_features), [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_textocr_features](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_textocr_features)
- learned predictor: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_learned_budget](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_learned_budget)
- panel readouts: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_gqa_slices_k2](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_gqa_slices_k2), [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_gqa_slices_k8](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_gqa_slices_k8), [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_probe_k2](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_probe_k2), [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_probe_k8](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_probe_k8), [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_ocrsubset_k2](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_ocrsubset_k2)

Matched baseline runs:

- reused identical prior clean VQAv2 eval: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_baseline_vqa_eval](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_baseline_vqa_eval)
- baseline ChartQA fixed/oracle: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_baseline_chartqa_eval](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_baseline_chartqa_eval)
- baseline TextOCR fixed/oracle: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_baseline_textocr_eval](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_baseline_textocr_eval)
- baseline OCR subset: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_baseline_ocrsubset_k2](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_baseline_ocrsubset_k2)

Structured summaries:

- new VQAv2: [summary.json](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335/ocrmix_vqa_eval/summary.json)
- new ChartQA: [summary.json](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335/ocrmix_chartqa_eval/summary.json)
- new TextOCR: [summary.json](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335/ocrmix_textocr_eval/summary.json)
- learned predictor: [learned_budget_predictor_summary.json](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335/learned_budget_predictor_summary.json)
- bundle timeline: [timeline.log](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335/timeline.log)

## Fixed-K And Oracle Tables

### New OCR-aware variable-prefix checkpoint

| Benchmark | `K=2` | `K=4` | `K=8` | `K=16` | Best fixed | Oracle | Oracle avg `K` |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQAv2 | `0.6233` | `0.6232` | `0.6239` | `0.6232` | `0.6239 @ K=8` | `0.6485` | `2.2204` |
| ChartQA | `0.0380` | `0.0365` | `0.0385` | `0.0391` | `0.0391 @ K=16` | `0.0464` | `2.0604` |
| TextOCR | `0.0192` | `0.0217` | `0.0232` | `0.0210` | `0.0232 @ K=8` | `0.0311` | `2.0785` |

### Existing clean-source variable-prefix baseline

| Benchmark | `K=2` | `K=4` | `K=8` | `K=16` | Best fixed | Oracle | Oracle avg `K` |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQAv2 | `0.6287` | `0.6285` | `0.6293` | `0.6287` | `0.6293 @ K=8` | `0.6545` | `2.2509` |
| ChartQA | `0.0333` | `0.0323` | `0.0349` | `0.0312` | `0.0349 @ K=8` | `0.0391` | `2.0281` |
| TextOCR | `0.0016` | `0.0015` | `0.0016` | `0.0015` | `0.0016 @ K=2` | `0.0020` | `2.0025` |

### New OCR-aware line vs prior clean baseline

| Benchmark | Best fixed delta | Oracle delta | Read |
|---|---:|---:|---|
| VQAv2 | `-0.0054` | `-0.0060` | OCR-aware mix hurt the clean core task |
| ChartQA | `+0.0042` | `+0.0073` | real gain on chart reading |
| TextOCR | `+0.0215` | `+0.0291` | huge gain on direct text readout |

The compression-stage mix did exactly what it was asked to do for ChartQA and TextOCR. It did not preserve the clean VQAv2 frontier line while doing it.

## Learned Predictor Results

The learned predictor used the OCR-aware checkpoint above. The full-split combined panel summary is:

- overall weighted panel accuracy: `0.4464`
- avg selected budget: `4.9854`
- escalated fraction: `30.82%`
- selected-budget histogram: `K=2` `213012`, `K=4` `9857`, `K=8` `36398`, `K=16` `48655`
- weighted oracle-gap recovered from fixed `K=2`: `30.94%`

That overall number is a pooled VQAv2 + ChartQA + TextOCR metric, so it is dominated by VQAv2 sample count. The clearer view is per benchmark:

| Benchmark | Learned predictor | Delta vs `K=2` | Delta vs best fixed | Delta vs best cheap scheduler | Oracle-gap recovered |
|---|---:|---:|---:|---:|---:|
| VQAv2 | `0.6308` | `+0.0076` | `+0.0069` | `+0.0067` | `29.92%` |
| ChartQA | `0.0396` | `+0.0016` | `+0.0005` | `+0.0026` | `18.75%` |
| TextOCR | `0.0235` | `+0.0043` | `+0.0003` | `+0.0009` | `36.13%` |

Best cheap schedulers on the same OCR-aware checkpoint:

- VQAv2: `0.6241` at avg `K=4.9600`
- ChartQA: `0.0370` at avg `K=5.2000`
- TextOCR: `0.0226` at avg `K=4.9599`

So the learned predictor is the first real practical dynamic-budget win in this project line. The cheap schedulers barely moved. The learned predictor materially beat them on every benchmark in the compact panel.

Important caveat:

- this does not mean the new OCR-aware compression checkpoint is the new default compressed model
- it means the learned-budget idea works; the OCR-heavy compression mix is the part that remains questionable

## Compact Panel Readout

### Hard-tail allocation

From the predictor full split:

- extra-budget tail fraction: `30.82%`
- tail dataset mix: `92.82%` VQAv2, `7.13%` TextOCR, `0.05%` ChartQA
- tail answer-type mix: `58.08%` other, `29.75%` yes/no, `12.17%` number
- top prefixes: `what is`, `how many`, `what text`, `is this`, `what color`, `where is`

This is the key negative result for the OCR-aware story. Even after training on more chart/text-heavy data, the learned scheduler still spends most extra budget on the broader VQAv2 hard tail, not mainly on OCR/chart cases.

### GQA slices

| Budget | Overall | Spatial | Attribute | Exist | Count |
|---|---:|---:|---:|---:|---:|
| `K=2` | `0.5087` | `0.4494` | `0.5064` | `0.5208` | `0.4000` |
| `K=8` | `0.5061` | `0.4509` | `0.5085` | `0.5147` | `0.5000` |

Read:

- no compelling GQA gain from larger LM-side semantic budget here
- the `count` slice only had `10` samples under the current capped slice path, so treat that row as a weak diagnostic rather than a real conclusion

### Probe / semantic readout

| Probe budget | Best val accuracy |
|---|---:|
| `K=2` | `0.5791` |
| `K=8` | `0.5573` |

This is scientifically interesting. The extreme bottleneck is not just surviving; the `K=2` prefix is actually more linearly readable than the wider `K=8` prefix on this probe setup.

### OCR heuristic subset at `K=2`

| Checkpoint | OCR subset accuracy |
|---|---:|
| New OCR-aware line | `0.2197` |
| Prior clean baseline | `0.2322` |

This is another cautionary result. The OCR-aware mix improved TextOCR massively, but it did not improve the repo's heuristic OCR subset. So the gain is real but narrow, and does not automatically transfer to every text-heavy slice we care about.

## Interpretation

### 1. The learned budget predictor is a real practical unlock

This is the main positive result. The project finally has a learned dynamic-budget policy that materially beats both fixed low-budget baselines and the prior cheap heuristics.

The strongest evidence is VQAv2:

- learned predictor: `0.6308`
- OCR-aware checkpoint best fixed: `0.6239`
- best cheap scheduler: `0.6241`
- recovered about `29.9%` of the available oracle gap

That is qualitatively different from the earlier cheap-rule result, which recovered only a few percent of the gap.

### 2. The OCR/chart-aware compression-stage mix is not the right clean-regime default

This is the main negative result. The new compression checkpoint improved the dedicated text/chart benchmarks, but it damaged the core clean VQAv2 line:

- VQAv2 best fixed fell from `0.6293` to `0.6239`
- VQAv2 oracle fell from `0.6545` to `0.6485`
- the heuristic OCR subset also fell from `0.2322` to `0.2197`

So the compression-stage mix over-rotated toward ChartQA/TextOCR. It bought specialized recovery by spending too much of the clean VQAv2 budget.

### 3. `K=2` is now a serious default tiny-prefix regime

On the new OCR-aware checkpoint:

- VQAv2 `K=2` is only `0.0006` behind best fixed `K=8`
- ChartQA oracle average budget is only `2.06`
- TextOCR oracle average budget is only `2.08`
- the probe actually prefers `K=2` over `K=8`

So the extreme bottleneck branch remains scientifically real. The bundle strengthens, rather than weakens, the claim that `K=2` is the right default starting point for later dynamic-budget work.

### 4. The hard tail is still broader than OCR/chart

The extra-budget tail is still overwhelmingly a broad VQAv2 tail. It is not mostly OCR/chart, and not mostly number.

That means the next scheduler should not be framed as "just learn OCR-ness." It needs to combine uncertainty with general question semantics and broad sample difficulty.

### 5. `K=16` remains a rare oracle need, even though the learned predictor still overuses it

Oracle average budgets stay near `2`, which means the true task does not need much `K=16`. The current cascade still selects `K=16` too often. That is not a reason to drop learned scheduling. It is a reason to add stronger cost-aware training or calibration on the next pass.

## Recommendation

The answer to the bundle question is:

- learned budget prediction: `yes`
- OCR/chart-aware compression as the new clean default: `no`

Single next experiment:

- train the same learned cascade on top of the stronger non-OCR-mixed clean `{2,4,8,16}` variable-prefix checkpoint, with explicit budget-cost regularization or threshold calibration to suppress unnecessary `K=16` use

Why this is the right next move:

- the learned scheduler idea now looks real
- the OCR-aware compression mix clearly trades away too much clean VQAv2 quality
- the hard tail is still mostly the broad clean VQAv2 tail, so the next gain is more likely to come from better scheduling on the stronger clean checkpoint than from pushing even harder into OCR-heavy compression data

So the next bundle should be a clean-regime learned-budget follow-up, not another OCR-heavy compression retrain.
