## Scope

This is the recurring champion ledger for the current MM frontier in `vqafromscratch`.

Use this file first when choosing a default source checkpoint or recipe. Update it whenever a new full-evaled line becomes the project champion in a meaningful regime. Dated reports like [80_frontier_champions_by_compute_2026-03-30.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/80_frontier_champions_by_compute_2026-03-30.md) and [82_variable_k_oracle_report_2026-03-30.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/82_variable_k_oracle_report_2026-03-30.md) remain the audit trail; this file is the rolling summary.

Method rules:

- full-evaled checkpoints only for deployable champion claims
- oracle rows are allowed, but must be labeled as upper bounds rather than deployable champions
- compute regime stays explicit so `9k`, `12k`, `18k`, and `21k` results are not mixed carelessly

Last updated:

- `2026-03-31`

## Default Start Point

Unless the user explicitly says otherwise, the default source model/setup for new bridge or compression work is the current clean `9k` frontier champion:

- run: [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2)
- checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar)
- recipe: `SigLIP2 + lm_final + Qwen-KD + fresh-init top-2-block VM finetune`
- full eval: `0.6470`

Why this is the default:

- it is the strongest clean matched-`9k` bridge line
- it is the current best base source for downstream compression work
- it avoids silently drifting onto higher-compute stacked lines unless the user asks for that

## Performance Notes

The champion metrics above come from the real full run, not from a later perf-retuned rerun. For future reruns of the same clean `9k` SigLIP2 frontier family, use the stronger verified training profile that was benchmarked afterward:

- train: `batch_size=96`, `grad_accum_steps=2`
- eval: `eval_batch_size=160`
- loader: `num_workers=4`, `prefetch_factor=2`, `pin_memory=1`
- attention backend: `mm_sdp_backend=math`
- KD path: use the current `train/mm.py` hot-path optimization; it is already in-tree and does not require an extra flag

Measured speed notes for that family:

- original real champion run averaged about `1.8875 steps/s`:
  - [winner_ftclean_bridge_v2 logfile](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/logfile.txt)
- tuned loader/profile `200`-step smoke averaged about `2.510 steps/s` over steps `100-200`:
  - [math tuned 200-step smoke](/home/wdree/percy/vqafromscratch/logs/mmsmoke_perf_ftclean_math_tuned_200_base/logfile.txt)
- tuned loader/profile plus KD hot-path fix averaged about `2.6183 steps/s` over steps `100-200`:
  - [KD-optimized 200-step smoke](/home/wdree/percy/vqafromscratch/logs/mmsmoke_perf_ftclean_kdopt_200/logfile.txt)

Rejected perf paths for this SigLIP2 champion family:

- `num_workers=6`
- `sdpa auto`
- GPU-resident KD logits
- unit-range VM-side renorm image path
- `torchvision` tensor decode path

Those were benchmarked and did not beat the kept profile on this machine. The relevant comments live in:

- [launch_tier0_vm_bundle_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_tier0_vm_bundle_v1.sh)
- [mm.py](/home/wdree/percy/vqafromscratch/train/mm.py)
- [vqa_data.py](/home/wdree/percy/vqafromscratch/train/vqa_data.py)

## Current Champions

| Regime / Category | Champion line | Overall | Yes/No | Number | Other | GQA | OCR | Probe |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean `9k` bridge default | SigLIP2 + `lm_final` + Qwen-KD + fresh-init top-2-block VM finetune | `0.6470` | `0.7853` | `0.4878` | `0.5840` | `0.5180` | `0.3118` | `0.5267` |
| Clean `12k` compressed | SigLIP2 + `lm_final` + Qwen-KD + `K=8` compression | `0.6239` | `0.7797` | `0.4778` | `0.5440` | n/a | n/a | `0.5485` |
| Best raw bridge regardless of compute | SigLIP2 + `lm_final` + Qwen-KD, then top-2-block VM finetune | `0.6650` | `0.8165` | `0.5016` | `0.5931` | `0.5354` | `0.3448` | `0.5487` |
| Best compressed line regardless of compute | SigLIP2 + `lm_final` + Qwen-KD, then top-2-block VM finetune, then `K=8` compression | `0.6421` | `0.8137` | `0.4933` | `0.5509` | `0.5476` | `0.2372` | `0.5814` |
| Dynamic-budget oracle upper bound | Variable-prefix `K in {2,4,8,16}` compressor on top of the clean `9k` champ | `0.6545` | `0.8174` | `0.5062` | `0.5699` | n/a | n/a | n/a |

Oracle note:

- the dynamic-budget oracle row is an upper bound, not a deployable model selection policy
- average oracle selected budget was `2.2509`
- oracle selected `K=2` on `96.37%` of samples

## Exact Artifacts

### Clean `9k` bridge default

- run: [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2)
- full eval: [winner_ftclean_bridge_peak_full.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftclean_bridge_peak_full.json)
- GQA: [winner_ftclean_bridge_gqa.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftclean_bridge_gqa.json)
- OCR: [winner_ftclean_bridge_ocr.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftclean_bridge_ocr.json)
- probe: [winner_ftclean_bridge_probe.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftclean_bridge_probe.json)

### Clean `12k` compressed

- bridge source: [mmtier0_vm_v1_20260330_001641_winner_kd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_kd_bridge)
- compression run: [mmtier0_vm_v1_20260330_001641_winner_stacked_k8](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_stacked_k8)
- full eval: [winner_stacked_k8_peak_full.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_stacked_k8_peak_full.json)
- probe: [winner_stacked_k8_probe.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_stacked_k8_probe.json)

### Best raw bridge regardless of compute

- run: [mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge)
- full eval: [winner_ftkd_bridge_peak_full.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_bridge_peak_full.json)
- GQA: [winner_ftkd_bridge_gqa.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_bridge_gqa.json)
- OCR: [winner_ftkd_bridge_ocr.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_bridge_ocr.json)
- probe: [winner_ftkd_bridge_probe.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_bridge_probe.json)

### Best compressed line regardless of compute

- run: [mmtier0_vm_v1_20260330_001641_winner_ftkd_k8](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8)
- full eval: [winner_ftkd_k8_peak_full.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8_peak_full.json)
- GQA: [winner_ftkd_k8_gqa.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8_gqa.json)
- OCR: [winner_ftkd_k8_ocr.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8_ocr.json)
- probe: [winner_ftkd_k8_probe.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8_probe.json)

### Dynamic-budget oracle upper bound

- bundle: [mmdynbudget_clean9k_overnight_v1_20260330_232048](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048)
- train run: [mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_train](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048_frontier24_train)
- source checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar)
- oracle summary: [summary.json](/home/wdree/percy/vqafromscratch/logs/mmdynbudget_clean9k_overnight_v1_20260330_232048/frontier24_eval/summary.json)
- report: [84_dynamic_budget_overnight_bundle_report_2026-03-30.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/84_dynamic_budget_overnight_bundle_report_2026-03-30.md)

### Learned-budget Research Reference

- bundle: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335)
- train run: [mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_train](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335_ocrmix_train)
- learned predictor summary: [learned_budget_predictor_summary.json](/home/wdree/percy/vqafromscratch/logs/mmlearnbudget_ocrmix_clean9k_v1_20260331_112335/learned_budget_predictor_summary.json)
- report: [86_learned_budget_ocr_bundle_report_2026-03-31.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/86_learned_budget_ocr_bundle_report_2026-03-31.md)

Read:

- this is the first clean-regime learned scheduler that materially beats the cheap-rule baselines
- it is a research reference, not a new default source checkpoint
- the OCR/chart-aware compression mix improved ChartQA and TextOCR, but it regressed clean VQAv2 too much to replace the current clean compressed default

## Practical Defaults

- New bridge or compression work: start from the clean `9k` champion unless the user says otherwise.
- Absolute raw VQAv2 frontier: use the `18k` stacked bridge line if compute mismatch is acceptable.
- Absolute compressed frontier: use the `21k` stacked compressed line if compute mismatch is acceptable.
- Dynamic-budget scheduling work: use the variable-prefix oracle bundle as the headroom reference, but do not treat oracle numbers as deployable metrics.
- Learned-budget follow-up work: start from the stronger non-OCR-mixed clean `{2,4,8,16}` variable-prefix checkpoint, not from the OCR-aware compression checkpoint, unless the goal is specifically OCR/chart specialization.

## Update Checklist

When a new candidate lands, update this file only if all of the following are true:

- the checkpoint has a corresponding full eval or explicitly labeled oracle/upper-bound eval
- the compute regime is clear
- the exact run dir and checkpoint path are known
- the change matters enough to alter what should be used by default
