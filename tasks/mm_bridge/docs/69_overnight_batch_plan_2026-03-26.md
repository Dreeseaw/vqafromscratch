# Overnight Batch Plan

Bundle: `logs/mmovernight_batch_v1_<timestamp>/`

Priority order:

1. Full VQAv2 + GQA eval of the finished Cement+GQA bridge checkpoint.
2. `K=8` format-alignment compression on the Cement+GQA bridge perceiver.
3. Dual-VM `K=8` compression with `VQAv2 + GQA` mix only, no grounding loss.
4. Dual-VM `K=12` compression on VQAv2 only.
5. `K=12` compression on the Cement+GQA bridge perceiver if time remains.
6. Tiny-head probes for the new compressed checkpoints if time remains.

Safety choices:

- prefer resumable sub-runs over one-shot monoliths
- use `64 x 3` training as the default compression profile
- use `num_workers=1`, `prefetch_factor=1`, `--no-pin_memory` for VQAv2-only compression
- use `num_workers=0`, `prefetch_factor=1`, `--no-pin_memory`, `gqa_train_fraction=0.1` for GQA-mixed compression
- use `eval_batch_size=128` for VQAv2 full evals unless a run-specific lower value is safer
- use `batch_size=64` for GQA exact-match evals

Key checkpoints:

- Cement+GQA bridge: `logs/mmcement_gqa7030_v1_96x2/step_9000.tar`
- Dual-VM warm-start: `logs/mmdualvm_v1_20260324_rerun/step_9000.tar`
- SigLIP remap teacher: `logs/mmsemantic_remap_v1_debug/step_500.tar`
- Dual-VM remap teacher: `logs/dualvm_compressed_v1_20260325_234134_phase1_remap/step_500.tar`

Outputs:

- rolling status in `progress.md`
- per-run JSON eval artifacts in the bundle dir
- final overnight bundle summary in `overnight_batch_report.md`
