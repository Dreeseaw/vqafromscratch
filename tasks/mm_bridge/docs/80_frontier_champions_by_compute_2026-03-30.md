## Scope

This note is the compute-regime champion ledger for the current non-dual-VM frontier after the Tier 0 VM bundle and the corrected fresh-init SigLIP2 finetune run. It is intentionally organized by **MM training steps** so `9k`, `12k`, `18k`, and `21k` lines are not mixed together.

Step count is the primary regime axis here because that is how the project has been comparing bridge-stage and bridge+compression pipelines. It is still worth remembering that `9k` finetune steps are more expensive than `9k` frozen-VM steps, but this note is explicitly about the current **step-regime champions**.

Bundle:
- [mmtier0_vm_v1_20260330_001641](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641)

Method rule:
- only full-evaled checkpoints are used for champion claims
- the broken first `winner_ftclean_bridge` attempt is excluded
- the valid fresh-init finetune run is [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2)

## Champions

| Regime | Champion line | Overall | Yes/No | Number | Other | GQA | OCR | Probe |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `9k` bridge | SigLIP2 + `lm_final` + Qwen-KD + fresh-init top-2-block VM finetune | `0.6470` | `0.7853` | `0.4878` | `0.5840` | `0.5180` | `0.3118` | `0.5267` |
| `12k` bridge+K=8 | SigLIP2 + `lm_final` + Qwen-KD + K=8 compression | `0.6239` | `0.7797` | `0.4778` | `0.5440` | n/a | n/a | `0.5485` |
| `18k` stacked bridge | SigLIP2 + `lm_final` + Qwen-KD, then top-2-block VM finetune | `0.6650` | `0.8165` | `0.5016` | `0.5931` | `0.5354` | `0.3448` | `0.5487` |
| `21k` stacked bridge+K=8 | SigLIP2 + `lm_final` + Qwen-KD, then top-2-block VM finetune, then K=8 compression | `0.6421` | `0.8137` | `0.4933` | `0.5509` | `0.5476` | `0.2372` | `0.5814` |

## Exact Setups

### `9k` champion

Run:
- [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2)

Full eval:
- [winner_ftclean_bridge_peak_full.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftclean_bridge_peak_full.json)

Diagnostics:
- [winner_ftclean_bridge_gqa.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftclean_bridge_gqa.json)
- [winner_ftclean_bridge_ocr.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftclean_bridge_ocr.json)
- [winner_ftclean_bridge_probe.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftclean_bridge_probe.json)

Recipe:
- VM: `siglip2_b16`
- VM init: frozen pretrained SigLIP2 weights, but top 2 blocks trainable from step zero
- LM checkpoint: [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar)
- Bridge style: Cement champion recipe
- KD: Qwen soft labels on, weight `0.3`, temp `4.0`
- Train length: `9000` steps
- Full-evaled MM checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar)
- Training log dir: [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2)
- Important verification: logfile shows `module=vision ... trainable_params=14,175,744`

Interpretation:
- this is the strongest **clean** bridge-stage line at matched `9k` steps
- it beats the frozen SigLIP2 + KD bridge by `+0.0056`
- it also beats the reasoning-LM-v2 SigLIP2 + KD bridge by `+0.0072`

### `12k` champion

Run:
- bridge source: [mmtier0_vm_v1_20260330_001641_winner_kd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_kd_bridge)
- compression: [mmtier0_vm_v1_20260330_001641_winner_stacked_k8](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_stacked_k8)

Full eval:
- [winner_stacked_k8_peak_full.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_stacked_k8_peak_full.json)

Probe:
- [winner_stacked_k8_probe.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_stacked_k8_probe.json)

Recipe:
- Stage 1 bridge LM checkpoint: [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar)
- Stage 1 bridge full-evaled MM checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_kd_bridge/step_9000.tar)
- Stage 2 compression: standard K=8 format-alignment compression from the bridge’s full-evaled `step_9000`
- Compression full-evaled MM checkpoint: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_stacked_k8/step_3000.tar)
- Training log dirs:
  - [mmtier0_vm_v1_20260330_001641_winner_kd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_kd_bridge)
  - [mmtier0_vm_v1_20260330_001641_winner_stacked_k8](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_stacked_k8)
- Total MM steps: `9000 + 3000 = 12000`

Interpretation:
- this is the current best **clean compressed** line by step regime
- it is the strongest `12k` result now on the board
- no clean compressed hard-eval suite was run for the new `9k` fresh-init finetune line, so this remains the champion for `12k`

### `18k` champion

Run:
- [mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge)

Full eval:
- [winner_ftkd_bridge_peak_full.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_bridge_peak_full.json)

Diagnostics:
- [winner_ftkd_bridge_gqa.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_bridge_gqa.json)
- [winner_ftkd_bridge_ocr.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_bridge_ocr.json)
- [winner_ftkd_bridge_probe.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_bridge_probe.json)

Recipe:
- Stage 1 bridge LM checkpoint: [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar)
- Stage 1 bridge full-evaled MM checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_kd_bridge/step_9000.tar)
- Stage 2 bridge continuation: top-2-block VM finetune enabled
- Stage 2 full-evaled MM checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge/step_9000.tar)
- Training log dirs:
  - [mmtier0_vm_v1_20260330_001641_winner_kd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_kd_bridge)
  - [mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge)
- Total MM steps: `9000 + 9000 = 18000`

Interpretation:
- this is the best **absolute raw VQAv2** bridge line so far
- it is not a clean `9k` comparator
- OCR is best here too: `0.3448`

### `21k` champion

Run:
- [mmtier0_vm_v1_20260330_001641_winner_ftkd_k8](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_k8)

Full eval:
- [winner_ftkd_k8_peak_full.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8_peak_full.json)

Diagnostics:
- [winner_ftkd_k8_gqa.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8_gqa.json)
- [winner_ftkd_k8_ocr.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8_ocr.json)
- [winner_ftkd_k8_probe.json](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641/winner_ftkd_k8_probe.json)

Recipe:
- Stage 1 bridge LM checkpoint: [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar)
- Stage 1 bridge full-evaled MM checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_kd_bridge/step_9000.tar)
- Stage 2 bridge continuation: top-2-block VM finetune enabled
- Stage 2 bridge full-evaled MM checkpoint: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge/step_9000.tar)
- Stage 3 compression: K=8 from the full-evaled `18k` bridge checkpoint
- Stage 3 compression full-evaled MM checkpoint: [step_3000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_k8/step_3000.tar)
- Training log dirs:
  - [mmtier0_vm_v1_20260330_001641_winner_kd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_kd_bridge)
  - [mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_bridge)
  - [mmtier0_vm_v1_20260330_001641_winner_ftkd_k8](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftkd_k8)
- Total MM steps: `9000 + 9000 + 3000 = 21000`

Interpretation:
- this is the strongest compressed line overall
- it is also the strongest probe line overall
- it is the best available GQA exact-match line in the family

## Sub-Metric Champions By Regime

### `9k`

- best raw VQAv2: `winner_ftclean_bridge` at `0.6470`
- best OCR: `winner_kd_bridge` at `0.3204`
- best GQA: `winner_kd_bridge` at `0.5392`
- best probe: `winner_ftclean_bridge` at `0.5267`

### `12k`

- best raw VQAv2: `winner_stacked_k8` at `0.6239`
- best probe: `winner_stacked_k8` at `0.5485`

### `18k`

- best raw VQAv2: `winner_ftkd_bridge` at `0.6650`
- best OCR: `winner_ftkd_bridge` at `0.3448`

### `21k`

- best raw VQAv2: `winner_ftkd_k8` at `0.6421`
- best GQA: `winner_ftkd_k8` at `0.5476`
- best probe: `winner_ftkd_k8` at `0.5814`

## Current Practical Takeaways

- If you care about the strongest clean bridge line at matched `9k`, the new champion is the fresh-init SigLIP2 + `lm_final` + KD + top-layer finetune run.
- If you care about the strongest clean compressed line at matched `12k`, the champion is still SigLIP2 + `lm_final` + KD + K=8.
- If you care about the absolute frontier regardless of compute, the `18k` and `21k` stacked finetune lines still hold the top raw and compressed scores.
- If you care specifically about OCR at bridge stage, the `18k` stacked bridge is best overall, but among clean `9k` lines the plain KD bridge still edges the fresh-init finetune line.
