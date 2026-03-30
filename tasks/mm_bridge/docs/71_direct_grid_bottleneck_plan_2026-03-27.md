# Direct Grid Bottleneck Plan

Question: does the current `49 -> 8` semantic bottleneck plateau because the bottleneck only sees perceiver latents, not the full projected visual grid?

This experiment keeps the existing format-alignment recipe and changes only the bottleneck attention target.

Variants:
- `Variant A`: bottleneck cross-attends over `cat(perceiver_out, grid_tokens)`
- `Variant B`: bottleneck derives `K` queries from `perceiver_out`, then cross-attends over `grid_tokens` only

Execution order:
1. `Variant A` on the frozen Cement SigLIP perceiver
2. `Variant B` on the frozen Cement SigLIP perceiver
3. If either SigLIP run beats the SigLIP `K=8` reference (`0.5900`), run the better variant on the frozen dual-VM perceiver

Conservative runtime choices:
- SigLIP runs: `batch_size=64`, `grad_accum_steps=3`, `eval_batch_size=96`, `num_workers=1`, `prefetch_factor=1`, `--no-pin_memory`
- dual-VM run: `batch_size=48`, `grad_accum_steps=4`, `eval_batch_size=64`, `num_workers=0`, `prefetch_factor=1`, `--no-pin_memory`
- `3000` train steps, checkpoint/eval every `500`

Controls used:
- SigLIP `K=8` format-aligned reference from [57_format_alignment_training_report_2026-03-24.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/57_format_alignment_training_report_2026-03-24.md)
- dual-VM `K=8` reference from [65_dualvm_compression_report_2026-03-26.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/65_dualvm_compression_report_2026-03-26.md)

Primary readouts:
- VQAv2 full-val overall + `yes/no`, `number`, `other`
- probe accuracy on exported compressed tokens
- `compression_grid_attn_fraction` for `Variant A`
- OCR subset only for the dual-VM winner, if a dual-VM run is warranted
