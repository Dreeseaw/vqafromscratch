# Qwen2.5-VL Distillation Plan

## Goal

Run a fresh SigLIP/Cement-style bridge with answer-vocab knowledge distillation from `Qwen/Qwen2.5-VL-3B-Instruct`, then compress it with the existing format-alignment recipe.

## Repo-Specific Adaptation

The current MM bridge is generative, not a direct answer classifier. So the KD signal is applied in the existing LM token space:

- Build a fixed VQAv2 answer vocabulary from train-set canonical answers.
- Extract Qwen first-answer-token logits and project them onto that answer vocabulary.
- During bridge training, take the student LM logits at the first answer token position and project them onto the same answer vocabulary.
- Apply `KL(student/T || teacher/T) * T^2` there.

This keeps the champion architecture intact and avoids adding a new answer head.

## Teacher Artifact

Output directory:

- `data/distillation/qwen25vl3b_vqav2_train_v1`

Stored format:

- `meta.json`
- `progress.json`
- `shard_*.pt`

Each shard contains:

- `question_ids`
- `teacher_logits` in `float16`, shaped `[N, answer_vocab_size]`

The teacher dataset is keyed by `question_id` and is resumable by shard and dataset cursor.

## Runtime Choices

Safety choices for the overnight batch:

- Teacher inference starts at `batch_size=4` with automatic CUDA OOM backoff.
- Bridge/control and compression use:
  - `batch_size=96`
  - `grad_accum_steps=2`
  - `eval_batch_size=128`
  - `num_workers=2`
  - `prefetch_factor=1`
  - `--no-pin_memory`

These are intentionally below the repo’s fastest settings to reduce WSL host-memory risk.

## Experiment Phases

1. Teacher inference on all VQAv2 train `(image, question)` pairs.
2. Fresh control bridge training with the current Cement recipe.
3. Fresh distilled bridge training with the same recipe plus answer-vocab KL.
4. Standard K=8 format-alignment compression on the distilled bridge.

## Current-Recipe Notes

- The current Cement bridge recipe in this repo trains the top **2** LM layers, not 3.
- The VM is frozen SigLIP-B/16.
- The bridge is `perceiver_resampler`, `question_only`, `question_hidden_attn`, no dynbudget.
- Compression remains the existing semantic-bottleneck format-alignment recipe.

## Outputs

Bundle family:

- `logs/mmqwenkd_v1_*`

Expected runs:

- teacher extraction
- fresh control bridge
- distilled bridge
- distilled K=8 compression

Bundle files:

- `progress.md`
- `timeline.log`
- `qwen_distill_report.md`
