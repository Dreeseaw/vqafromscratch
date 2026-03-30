# Bottleneck Format-Alignment Experiment Plan

## Goal

Retrain the `K=8` semantic bottleneck from the Cement SigLIP checkpoint with LM adapters disabled from the start, while using the trained linear PrefixRemap only as a frozen alignment teacher. The question is whether the bottleneck can internalize the remap's LM-format correction and remove the need for both adapters and remap at inference.

## Fixed Setup

- start from [step_8000.tar](/home/wdree/percy/vqafromscratch/logs/mmcement_v1_20260316_siglip_cement_questiononly_s53/step_8000.tar)
- frozen SigLIP VM
- frozen perceiver
- `question_only` context
- `attnqquery`
- no dynbudget
- semantic bottleneck enabled at `K=8`
- LM frozen
- LM visual adapters disabled in the forward path from step 1

Teacher remap source:

- [step_500.tar](/home/wdree/percy/vqafromscratch/logs/mmsemantic_remap_v1_debug/step_500.tar)

## Trainable Parameters

- semantic bottleneck only
- PrefixRemap present as a frozen teacher module, but not applied in the forward path during training

## Loss

- `L_vqa`: standard VQA cross-entropy
- `L_distill`: existing semantic bottleneck reconstruction MSE back to the frozen perceiver evidence latents
- `L_format`: MSE from bottleneck export tokens to frozen `PrefixRemap(export_tokens)`

Weights:

- `L_distill`: `0.1`
- `L_format`: `0.3 -> 0.0` linearly annealed over steps `2250 -> 3000`

## Training

- VQAv2 train only
- `batch_size=96`, `grad_accum_steps=2`
- `max_steps=3000`
- `lr=2e-4`
- cosine schedule
- `lr_warmup_steps=200`
- checkpoint every `500`
- log every step

## Eval Plan

Quick checkpoint evals at steps:

- `500, 1000, 1500, 2000, 2500, 3000`

Each checkpoint gets two 100-batch evals:

1. no remap, no adapters
2. with remap, no adapters

Then run full-val eval on the best checkpoint in both modes.

## Secondary Diagnostics

On the best checkpoint:

- adapter ablation with adapters re-enabled: keep `3,2,1`
- tiny-head probe on the aligned `K=8` tokens

## Success Tiers

- strong: no-remap full eval reaches about `0.57+ overall` and `0.74+ yes/no`
- moderate: no-remap lands in `0.50-0.57`
- weak: no-remap stays below `0.50`

## Deliverables

- run bundle: `logs/mmsemantic_format_v1_<timestamp>/`
- run-local report: `format_alignment_report.md`
- task-doc report after completion
