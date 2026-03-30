# Dual-VM + K=8 Compression Report

Bundle: [dualvm_compressed_v1_20260325_234134](/home/wdree/percy/vqafromscratch/logs/dualvm_compressed_v1_20260325_234134)

## Setup

This experiment combined the best warm-start dual-VM system with the existing post-perceiver semantic bottleneck path:

- start point: warm-start dual-VM checkpoint at [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmdualvm_v1_20260324_rerun/step_9000.tar)
- frozen upstream stack during compression:
  - frozen SigLIP-B/16 VM
  - frozen ViTSTR-Tiny VM
  - frozen dual-VM perceiver
  - frozen LM, adapters disabled
- trainable compression module:
  - new `K=8` semantic bottleneck only
- teacher:
  - dual-VM-specific prefix remap, trained first in Phase 1

The core question was not whether dual-VM helps in general. That was already known from the warm-start run. The question here was whether the OCR-aware dual-VM perceiver had already baked useful OCR evidence into its 49 exported latents strongly enough for a later `49 -> 8` bottleneck to preserve it.

## Results

### Phase 1: Remap Diagnostic

Reference files:
- keep-0 baseline: [phase1_dual_keep0_full.json](/home/wdree/percy/vqafromscratch/logs/dualvm_compressed_v1_20260325_234134/phase1_dual_keep0_full.json)
- remap full eval: [phase1_best_step_500_with_remap_full.json](/home/wdree/percy/vqafromscratch/logs/dualvm_compressed_v1_20260325_234134/phase1_best_step_500_with_remap_full.json)

| Condition | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| Warm-start dual-VM keep-3 | 0.6307 | 0.7918 | 0.4699 | 0.5508 |
| Dual-VM keep-0 | 0.4700 | 0.6259 | 0.2927 | 0.3985 |
| Dual-VM + remap keep-0 | 0.6013 | 0.7839 | 0.4554 | 0.5009 |

Read:
- adapter removal cost the warm-start dual-VM system `-0.1607`
- the learned remap recovered `+0.1312`, or about `81.7%` of that drop
- this is the same structural story as the earlier Cement remap result: a large share of the adapter contribution is prefix-format translation, not deep new reasoning

So Phase 2 was justified.

### Phase 2: K=8 Format-Alignment

Reference files:
- best full eval: [phase2_best_step_3000_full.json](/home/wdree/percy/vqafromscratch/logs/dualvm_compressed_v1_20260325_234134/phase2_best_step_3000_full.json)
- probe: [phase2_tiny_head_probe_best.json](/home/wdree/percy/vqafromscratch/logs/dualvm_compressed_v1_20260325_234134/phase2_tiny_head_probe_best.json)
- OCR subset: [phase2_ocr_analysis.json](/home/wdree/percy/vqafromscratch/logs/dualvm_compressed_v1_20260325_234134/phase2_ocr_analysis.json)
- bundle summary: [dualvm_compression_report.md](/home/wdree/percy/vqafromscratch/logs/dualvm_compressed_v1_20260325_234134/dualvm_compression_report.md)

Best checkpoint by mini-eval and confirmed by full eval:
- `step_3000`

Full-eval comparison:

| Condition | Overall | Yes/No | Number | Other | OCR subset |
|---|---:|---:|---:|---:|---:|
| Dual-VM warm-start | 0.6307 | 0.7918 | 0.4699 | 0.5508 | 0.2930 |
| Cement anchor | 0.6163 | 0.7589 | 0.4573 | 0.5499 | 0.2510 |
| SigLIP-only K=8 | 0.5900 | 0.7393 | 0.4499 | 0.5134 | - |
| Dual-VM K=8 | 0.6109 | 0.7866 | 0.4611 | 0.5168 | 0.2106 |

Probe:
- dual-VM K=8 probe = `0.5337`
- earlier SigLIP-only K=8 probe = `0.5103`

Phase-2 training trace on periodic eval:

| Step | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| 500 | 0.5751 | 0.7760 | 0.4467 | 0.4610 |
| 1000 | 0.5965 | 0.7937 | 0.4477 | 0.4906 |
| 1500 | 0.5962 | 0.7821 | 0.4344 | 0.5020 |
| 2000 | 0.6068 | 0.7987 | 0.4507 | 0.5068 |
| 2500 | 0.6124 | 0.7985 | 0.4629 | 0.5149 |
| 3000 | 0.6137 | 0.7966 | 0.4600 | 0.5197 |

## Main Interpretation

This result splits cleanly in two directions.

First, for general VQAv2, the combination works. The compressed dual-VM system beat the compressed SigLIP-only system by `+0.0209` overall (`0.6109` vs `0.5900`) and stayed close to the uncompressed Cement anchor (`-0.0054`). So the dual-VM perceiver is clearly putting useful extra evidence into the 49-token latent set, and the semantic bottleneck can preserve enough of that evidence to improve general compressed VQA.

Second, for OCR specifically, the combination failed. The OCR subset fell to `0.2106`, which is:

- `-0.0824` below the warm-start dual-VM OCR score `0.2930`
- `-0.0404` below the Cement anchor OCR score `0.2510`

So the OCR-specific gain from the uncompressed dual-VM run does **not** survive the `49 -> 8` bottleneck in the current training recipe.

That means the compressed dual-VM line is not preserving OCR evidence in the narrow sense the experiment was designed to test, even though it is improving general compressed VQA.

## Modeling Read

The strongest coherent read is:

1. The dual-VM perceiver does inject extra useful evidence into the shared 49-latent representation.
   That is why compressed dual-VM still beats compressed SigLIP-only overall.

2. But the bottleneck keeps mostly the *globally useful* part of that added evidence, not the OCR-specialized part.
   The probe increase (`0.5337` vs `0.5103`) supports this. The compressed tokens are more linearly decodable overall, so information is not vanishing wholesale.

3. OCR evidence appears to be the part most vulnerable to late compression.
   The OCR subset regression plus the low ViTSTR attention fractions in the OCR buckets suggest the bottleneck is preserving broad semantic/context features more readily than the narrow text-reading cues that helped the uncompressed dual-VM line.

4. So this is not “dual-VM and compression are incompatible.”
   It is narrower than that:
   - dual-VM helps compressed *general* VQA
   - current K=8 compression does not preserve the *OCR-specific* win

## Recommendation

Do not treat this as the new default OCR path.

Recommended project stance:
- keep the uncompressed warm-start dual-VM line as the OCR-specialized frontier
- keep the SigLIP-only format-aligned K=8 line as the simplest compressed line
- keep this dual-VM K=8 result as a useful general-compressed variant, but not as proof that OCR survives compression

If you want to push this line further, the next useful experiments should bias the bottleneck toward OCR retention explicitly, for example:
- OCR-aware compression loss or subset weighting
- text-question-conditioned bottleneck queries
- larger compressed token budget for dual-VM only, such as `K=12` or `K=16`
- OCR-specific probe/teacher pressure instead of only generic VQA + distill + format losses

Bottom line:
- general compressed performance: positive
- OCR preservation: negative
- default recommendation: do not merge dual-VM and K=8 compression as one canonical path yet
