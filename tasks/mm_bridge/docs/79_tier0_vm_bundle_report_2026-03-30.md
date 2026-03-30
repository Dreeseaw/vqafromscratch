## Scope

This report closes the Tier 0 VM frontier-de-risking bundle from [78_tier0_vm_bundle_plan_2026-03-29.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/78_tier0_vm_bundle_plan_2026-03-29.md). The executed bundle is [mmtier0_vm_v1_20260330_001641](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641).

One methodological correction matters up front: downstream stages in the final executed bundle were anchored to full-evaled checkpoints, not periodic mini-eval peaks. Mini-evals are treated here only as training-curve context.

Second correction: training compute must be treated as a real comparison axis. The tables below now include a simple MM-step compute proxy so stacked `18k` and `21k` lines do not get silently mixed with plain `9k` or `12k` lines. This is a step-count proxy, not exact FLOPs, and it slightly understates the true cost of the VM-finetuned line because those later steps are more expensive than fully frozen-VM steps.

Third correction: the first attempted fresh-init `lm_final + KD + top-layer VM finetune` run was invalid and is excluded. The SigLIP2 tail-block unfreeze resolver was broken for `OpenCLIPBackbone`, so the VM stayed frozen. The corrected and verified run is [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2), trained from LM checkpoint [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar), and its logfile explicitly shows `module=vision ... trainable_params=14,175,744`.

## Ranked Results

Raw VQAv2 frontier, full eval:

| Line | MM train compute | Overall | Yes/No | Number | Other | Delta vs Cement |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2 + KD + top-layer VM finetune | 18k | 0.6650 | 0.8165 | 0.5016 | 0.5931 | +0.0487 |
| SigLIP2 + fresh-init KD + top-layer VM finetune | 9k | 0.6470 | 0.7853 | 0.4878 | 0.5840 | +0.0307 |
| SigLIP2 + KD | 9k | 0.6414 | 0.7754 | 0.4856 | 0.5807 | +0.0251 |
| SigLIP2 + reasoning-LM v2 + KD | 9k | 0.6398 | 0.7736 | 0.4819 | 0.5798 | +0.0235 |
| SigLIP2 frozen bridge | 9k | 0.6330 | 0.7715 | 0.4735 | 0.5700 | +0.0167 |
| PE-Core frozen bridge | 9k | 0.6190 | 0.7575 | 0.4568 | 0.5565 | +0.0027 |
| Cement reference | 9k | 0.6163 | 0.7589 | 0.4573 | 0.5499 | baseline |

Compressed `K=8` frontier, full eval:

| Line | MM train compute | Overall | Yes/No | Number | Other | Delta vs Ref K=8 |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2 + KD + top-layer VM finetune + K=8 | 21k | 0.6421 | 0.8137 | 0.4933 | 0.5509 | +0.0398 |
| SigLIP2 + KD + K=8 | 12k | 0.6239 | 0.7797 | 0.4778 | 0.5440 | +0.0216 |
| SigLIP2 frozen + K=8 | 12k | 0.6107 | 0.7611 | 0.4694 | 0.5337 | +0.0084 |
| Current reference K=8 | 12k | 0.6023 | 0.7605 | 0.4546 | 0.5211 | baseline |

Compute-aware read:

- At matched `9k` bridge compute, the VM swap is the first real frontier move: `0.6163 -> 0.6330`.
- At matched `9k` bridge compute on the stronger VM, KD adds a second clean move: `0.6330 -> 0.6414`.
- At matched `9k`, fresh-init top-layer VM finetuning on top of `lm_final + KD` adds another clean move: `0.6414 -> 0.6470`.
- At matched `9k`, swapping the LM from `lm_final` to reasoning-LM v2 inside the same fresh-init SigLIP2 + KD bridge recipe does not improve the frontier: `0.6414 -> 0.6398`.
- The `0.6650` finetuned line is the highest score in the bundle, but it is an `18k` stacked line, not a direct `9k` comparator.
- At matched `12k` compressed compute, KD on SigLIP2 is the strongest clean compressed comparison: `0.6107 -> 0.6239`.
- The `0.6421` stacked compressed line is real and exciting, but it is a `21k` result and should be read as a higher-compute frontier, not a like-for-like replacement for the `12k` lines.

## Readout

The frozen VM swap already moved the frontier. SigLIP2 was clearly stronger than both the old Cement VM line and PE-Core, gaining `+1.67` points over Cement and `+1.40` over PE-Core at bridge stage. PE-Core was not competitive enough to justify deeper stacking.

KD still added cleanly on top of the stronger VM. On SigLIP2, Qwen answer-KD improved the frozen bridge from `0.6330` to `0.6414`, with the largest bridge-stage category lift in `other` (`0.5700 -> 0.5807`) and a solid gain in `number` (`0.4735 -> 0.4856`).

The fresh-init SigLIP2 + reasoning-LM v2 + KD bridge did not beat that line. It finished at `0.6398`, which is effectively tied but still below the older `lm_final`-based SigLIP2 + KD bridge. The reasoning-LM checkpoint here was [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_reasoningmix_v2_20260329_133946/step_45000.tar), while the stronger baseline used [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar). Its supporting diagnostics were also slightly weaker overall: `GQA 0.5372` vs `0.5392`, `OCR 0.2950` vs `0.3204`, and probe `0.5122` vs `0.5159`.

The new clean fresh-init finetune result matters. With LM checkpoint [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar), SigLIP2, Qwen KD, and top-2-block VM finetuning all trained from step zero for `9k` steps, the corrected run [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2) reached `0.6470`. That makes it the strongest clean `9k` bridge line in the bundle. The lift over the frozen SigLIP2 + KD bridge is modest but real: `0.6414 -> 0.6470`, with the biggest category gain in `yes/no` and a smaller gain in `other`.

The biggest executed gain still came from partial VM finetuning, but there are now two distinct reads instead of one muddy one. The clean `9k` fresh-init finetune line shows that top-layer VM tuning helps even without an extra `9k` KD warm-start stage. Separately, the KD-initialized stacked finetune line reached `0.6650`, which is the strongest raw VQAv2 result in the bundle by a large margin, but should still be interpreted as "KD plus top-layer VM finetuning compounds well at 18k compute" rather than as a clean isolated estimate of finetuning alone.

Compression held up unusually well on the stronger lines. At matched `12k` compute, SigLIP2 + KD `K=8` reached `0.6239`, which is `+2.16` points over the current reference `K=8` line. The stacked `21k` SigLIP2 `K=8` run reached `0.6421`, which is `+3.98` over the current reference `K=8` line and still `+2.58` over the old uncompressed Cement bridge. That is the clearest Tier 0 signal that the plateau was not purely a bottleneck-compression ceiling.

## Hard Eval Read

OCR / text reading:

| Line | OCR subset |
|---|---:|
| SigLIP2 + KD + top-layer VM finetune | 0.3448 |
| SigLIP2 + KD | 0.3204 |
| SigLIP2 + fresh-init KD + top-layer VM finetune | 0.3118 |
| SigLIP2 + reasoning-LM v2 + KD | 0.2950 |
| SigLIP2 frozen bridge | 0.3000 |
| Cement reference | 0.2510 |
| SigLIP2 + KD + top-layer VM finetune + K=8 | 0.2372 |
| Current reference K=8 | 0.1982 |

Raw OCR clearly improved from the VM swap and improved again from KD and stacked finetuning. Compression still harms OCR badly, but the stacked SigLIP2 `K=8` line recovered a useful part of that loss relative to the old compressed reference.

GQA exact-match / hard reasoning:

| Line | GQA overall | Spatial | Attribute | Exist |
|---|---:|---:|---:|---:|
| SigLIP2 + KD | 0.5392 | 0.4424 | 0.5070 | 0.5642 |
| SigLIP2 + reasoning-LM v2 + KD | 0.5372 | 0.4506 | 0.5048 | 0.5750 |
| SigLIP2 + fresh-init KD + top-layer VM finetune | 0.5180 | 0.4466 | 0.5082 | 0.5358 |
| SigLIP2 + KD + top-layer VM finetune | 0.5354 | 0.4472 | 0.5308 | 0.5506 |
| SigLIP2 + KD + top-layer VM finetune + K=8 | 0.5476 | 0.4660 | 0.5264 | 0.5696 |
| Current reference K=8 | 0.5066 | 0.4404 | 0.4818 | 0.5148 |
| Cement reference | 0.4916 | 0.4408 | 0.5090 | 0.4836 |

The strongest hard-reasoning story is that the new SigLIP2 family materially improved exact-match GQA over both old references, and the stacked `K=8` line preserved that well enough to become the best GQA result in the bundle.

Retrieval / semantic token usefulness via probe:

| Line | Probe |
|---|---:|
| SigLIP2 + KD + top-layer VM finetune + K=8 | 0.5814 |
| SigLIP2 + KD + top-layer VM finetune | 0.5487 |
| SigLIP2 + KD + K=8 | 0.5485 |
| SigLIP2 + fresh-init KD + top-layer VM finetune | 0.5267 |
| SigLIP2 + KD | 0.5159 |
| SigLIP2 + reasoning-LM v2 + KD | 0.5122 |
| SigLIP2 frozen + K=8 | 0.5274 |
| Current reference K=8 | 0.5091 |
| Cement reference | 0.4779 |

This is the strongest evidence that the frontier moved on semantic token quality, not just surface VQAv2 fitting. The stacked `K=8` line is substantially denser and more linearly usable than the old compressed reference.

## Answer To The Main Question

The frontier did not move from a single source, and the answer depends on whether you care about matched compute or absolute best score.

- At matched `9k` bridge compute, the frontier moved most from swapping the VM.
- At matched `9k` on the stronger VM, KD added a smaller but still clean second move.
- At matched `9k`, fresh-init top-layer VM finetuning added a third clean move and produced the best bridge endpoint in that compute regime.
- At matched `9k`, the newer reasoning-LM v2 did not move the SigLIP2 + KD frontier further.
- The largest absolute score gain came from partial VM finetuning on top of KD, but that result is both stacked and higher-compute, so it is not the clean answer to "what moved the frontier most per unit of like-for-like training."

So the best current interpretation is:

- the project was not yet at a hard VM ceiling on the old line
- a better VM moved the frontier first
- training signal still mattered after the VM swap
- partial VM finetuning appears to matter a lot on top of the stronger line, but this needs a clean compute-matched control

## Recommendation

The single best clean `9k` bridge line to continue tomorrow is the fresh-init SigLIP2 + `lm_final` + Qwen-KD + top-layer VM finetune run.

If you want the cleanest scientifically defensible bridge frontier, continue from [mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2) and its full-evaled checkpoint [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2/step_9000.tar), which uses LM checkpoint [step_45000.tar](/home/wdree/percy/vqafromscratch/logs/lm_final/step_45000.tar).

If you want the best clean compressed line available now, continue from the `12k` SigLIP2 + KD `K=8` path.

If you want the highest raw upside regardless of compute, continue the stacked SigLIP2 + KD + top-layer finetuning line at `18k/21k`.
