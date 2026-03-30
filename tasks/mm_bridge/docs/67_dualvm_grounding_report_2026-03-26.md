# Dual-VM Grounding + GQA Report

## Scope

This report covers the `dualvm_grounding_v1` experiment from [66_dualvm_grounding_plan_2026-03-26.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/66_dualvm_grounding_plan_2026-03-26.md):

- control: fresh dual-VM `K=8` bottleneck, VQAv2-only
- mixed: fresh dual-VM `K=8` bottleneck, VQAv2 + GQA + pointing with `L_ground`

Primary artifacts:

- bundle: [dualvm_grounding_v1_20260326_092234](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234)
- control run: [dualvm_grounding_v1_20260326_092234_control](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234_control)
- completed mixed run: [dualvm_grounding_v1_20260326_092234_groundgqa_safer](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234_groundgqa_safer)
- bundle summary: [grounding_report.md](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/grounding_report.md)

## Ops

This experiment hit WSL host-memory failure twice before the final mixed run completed.

- previous boot OOM #1: `2026-03-26 10:29:22`
- previous boot OOM #2: `2026-03-26 11:38:06`
- both were host RAM / swap exhaustion, not CUDA VRAM OOM

The mixed arm only became stable after moving to a conservative loader profile:

- `batch_size=64`
- `grad_accum_steps=3`
- `eval_batch_size=32`
- `num_workers=0`
- `--no-pin_memory`
- `eval_every=250`
- `ckpt_every=250`
- `eval_batches=50`
- `final_eval_batches=50`

Originally, the mixed arm only had a successful official-score `50`-batch eval from the safe run. I later reran a true full VQAv2 val pass on the finished checkpoint. That full-val result supersedes the earlier `50`-batch read and materially changes the conclusion.

## Main Results

### Training-time comparison

These were the live in-run numbers before the posthoc full-val correction.

| Condition | Eval regime | Overall | Yes/No | Number | Other |
|---|---|---:|---:|---:|---:|
| Prior dual-VM `K=8` | completed run reference | 0.6109 | 0.7866 | 0.4611 | 0.5168 |
| Control best | periodic eval, step `2500` | 0.6112 | 0.7971 | 0.4642 | 0.5132 |
| Mixed final | final eval, step `3000`, `50` batches | 0.6331 | 0.7860 | 0.5457 | 0.5263 |

The control reproduced the prior dual-VM `K=8` baseline closely. That mattered, because it showed the mixed line was not just drifting randomly. But the later full-val pass showed that the `50`-batch mixed eval substantially overstated the final effect.

Relative to the reproduced control:

- overall: `+0.0219`
- yes/no: `-0.0111`
- number: `+0.0815`
- other: `+0.0131`

Relative to the prior dual-VM `K=8` reference:

- overall: `+0.0222`
- yes/no: `-0.0006`
- number: `+0.0846`
- other: `+0.0095`

As a live training signal, the mixed arm looked promising and the gain pattern was dominated by `number`. That signal did not survive the full-val check.

### Posthoc bundle evals

The bundle-level posthoc summary files are:

- control full eval: [control_best_step_2500_full.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/control_best_step_2500_full.json)
- mixed best-step eval: [mixed_best_step_3000_full.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/mixed_best_step_3000_full.json)
- direct full-val artifact: [mixed_best_step_3000_fullval.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/mixed_best_step_3000_fullval.json)

Those now report:

- control: `0.6086 / 0.7858 / 0.4601 / 0.5131`
- mixed: `0.6066 / 0.7818 / 0.4615 / 0.5118`

The earlier `50`-batch mixed eval (`0.6331`) was optimistic. The true full-val rerun on the exact same checkpoint came back at `0.6066`, which is:

- slightly below the reproduced control full eval `0.6086`
- below the prior dual-VM `K=8` reference `0.6109`
- well below the provisional `50`-batch read

## GQA

GQA exact-match evaluation is now working and non-degenerate:

- sanity: [gqa_exact_sanity.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/gqa_exact_sanity.json) -> `0.4492`

Best-step 5k-sample GQA slices:

- control: [control_best_step_2500_gqa.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/control_best_step_2500_gqa.json)
- mixed: [mixed_best_step_3000_gqa.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/mixed_best_step_3000_gqa.json)

| Condition | Overall | Spatial | Attribute | Exist | Count |
|---|---:|---:|---:|---:|---:|
| Control | 0.5046 | 0.4416 | 0.5036 | 0.5144 | 0.3000 |
| Mixed | 0.6056 | 0.4876 | 0.5206 | 0.6660 | 0.3000 |

This is the strongest positive secondary result in the whole experiment. The mixed bottleneck is much better on GQA-style compositional evaluation, especially `exist`, with smaller but real gains in `spatial` and `attribute`.

But it does not line up with the full VQAv2 result. The cleanest interpretation is:

- the mixed training signal improves GQA-style exact-match behavior
- those gains do not transfer cleanly to the main full VQAv2 objective
- the earlier `50`-batch VQAv2 read overstated generalization

## OCR, Probe, Grounding

### OCR subset

- control OCR: [control_ocr_analysis.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/control_ocr_analysis.json)
- mixed OCR: [mixed_ocr_analysis.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/mixed_ocr_analysis.json)

| Condition | OCR subset overall |
|---|---:|
| Cement anchor | 0.2510 |
| Prior dual-VM `K=8` | 0.2106 |
| Control | 0.1998 |
| Mixed | 0.2062 |

Grounding+GQA did not rescue OCR. The mixed arm is slightly above the control, but still below the prior dual-VM `K=8`, and well below the uncompressed warm-start OCR result (`0.2930` from the earlier dual-VM report).

### Tiny-head probe

- control probe: [control_probe.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/control_probe.json)
- mixed probe: [mixed_probe.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/mixed_probe.json)

| Condition | Probe overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| Prior dual-VM `K=8` | 0.5337 | - | - | - |
| Control | 0.5346 | 0.6665 | 0.3735 | 0.4505 |
| Mixed | 0.5362 | 0.6707 | 0.3907 | 0.4446 |

Probe gains are real but small. That suggests the main improvement is not “the tokens suddenly became much more linearly decodable.” A better read is:

- the bottleneck learned a slightly better representation
- the biggest benefit shows up in the frozen-LM consumption path, not just in probe separability

### Grounding mass

- control grounding mass: [control_grounding_mass.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/control_grounding_mass.json)
- mixed grounding mass: [mixed_grounding_mass.json](/home/wdree/percy/vqafromscratch/logs/dualvm_grounding_v1_20260326_092234/mixed_grounding_mass.json)

| Condition | Mean target mass |
|---|---:|
| Control | 0.007024 |
| Mixed | 0.007023 |

By this metric, `L_ground` did essentially nothing measurable.

That does not mean the extra supervision had no effect. It clearly changed VQAv2 and GQA behavior. But it does mean the current aggregate mass-in-target metric is not detecting a useful grounding shift, or the training signal is acting more as a general shaping prior than as a direct attention-localization improvement.

## Modeling Read

The experiment does not support the original narrow thesis, which was:

- close the compressed dual-VM `other` gap primarily through grounding + compositional supervision
- preserve or improve OCR-heavy behavior

What it actually did:

1. It strongly improved GQA exact-match behavior.
2. It slightly improved OCR relative to the fresh control, but not relative to the prior dual-VM `K=8`.
3. It did not improve the final full VQAv2 result.
4. It did not improve the current grounding-mass metric.
5. Its promising `50`-batch VQAv2 eval did not survive a full-val rerun.

So the best interpretation is:

- GQA training pressure is useful in-distribution for GQA-style exact-match behavior.
- The pointing loss, as currently wired, is not showing a clean independent win.
- The combined supervision does not currently improve the main VQAv2 objective after a full-val check.

In other words, this is neither a clean grounding win nor a reliable VQAv2 improvement. Right now it looks like a domain-shifted specialization effect.

## Success Call

Against the plan's original target, this is negative:

- it did not close the `other` gap enough to hit the stated strong criterion
- it did not improve OCR subset behavior

And against the practical project frontier, the full-val result is also not positive:

- compressed dual-VM `K=8` moved from `0.6109` to `0.6066`
- the reproduced fresh control was `0.6086`
- so the mixed recipe did not beat either the prior baseline or the control under full evaluation

So I would not call this a grounding success, and I would no longer call it a positive compressed-training result overall. The strongest thing it showed was improved GQA behavior, not improved VQAv2.

## Recommendation

Short term:

- do not adopt the grounding+GQA recipe as the default compressed dual-VM path
- keep the control recipe as the main compressed reference
- keep the mixed result only as evidence that GQA exact-match behavior can move independently of VQAv2 full-val quality

More precise next-step read:

- if the goal is overall compressed VQAv2 quality, this exact recipe is not worth continuing unchanged
- if the goal is OCR retention, this experiment did not solve it
- if the goal is measurable spatial grounding, the current `L_ground` formulation or metric needs revision before more sweeping claims are justified
- if this line is revisited, I would ablate `GQA` and `L_ground` separately instead of keeping them coupled

Operationally, any future mixed GQA/pointing run on this machine should keep the safer loader regime that finally completed.
