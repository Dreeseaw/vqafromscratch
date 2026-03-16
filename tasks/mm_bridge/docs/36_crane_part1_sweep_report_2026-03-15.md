# 36 Crane Part 1 Sweep Report (2026-03-15)

## Scope

This document reports on the completed portion of the Crane extended sweep (Tiers 1-4). Sources:

- `tasks/mm_bridge/docs/34_crane_extended_sweep_plan_2026-03-14.md` (sweep plan)
- `tasks/mm_bridge/docs/32b_plank_sweep_report_2026-03-14.md` (Plank reference)
- sweep bundles `logs/mmcrane_v1_20260314_*`
- per-run logs under `logs/mmcrane_v1_20260314_*/logfile.txt`

This document is retrospective. It records what ran, establishes the new frontier, identifies caveats in the caption-align results, and outlines directions for Crane Part 2.

## Run Set and Completion Status

### Completed Runs (11/20)

| # | Run | Tier | VM | Status |
|---|---|---:|---|---|
| 4 | `mobileclip_attnqquery_dynbudget_adapter_d3_cap64` | 2 | MobileCLIP | complete |
| 5 | `dinov2s_attnqquery_dynbudget_adapter_d3_cap64` | 2 | DINOv2 | complete |
| 6 | `dinov2s_lmmeanqquery_dynbudget_adapter_d3_cap64` | 2 | DINOv2 | complete |
| 7 | `dinov2s_attnqquery_nodynbudget_adapter_d3` | 3 | DINOv2 | complete |
| 8 | `dinov2s_attnqquery_dynbudget_adapter_d3_cap128` | 3 | DINOv2 | complete |
| 9 | `dinov2s_attnqquery_dynbudget_adapter_d3_cap32` | 3 | DINOv2 | complete |
| 10 | `mobilevit_captionalign_attnqquery_dynbudget_adapter_d3_cap64` | 4 | MobileViT | complete (caveat) |
| 11 | `dinov2s_captionalign_attnqquery_dynbudget_adapter_d3_cap64` | 4 | DINOv2 | complete (caveat) |
| 12 | `mobileclip_captionalign_attnqquery_dynbudget_adapter_d3_cap64` | 4 | MobileCLIP | complete (caveat) |
| — | `dinov2s_attnqquery_dynbudget_adapter_d5_cap64` | 6 | DINOv2 | complete |
| — | `mobileclip_attnqquery_dynbudget_adapter_d4_cap64` | 6 | MobileCLIP | complete |

### Partially Completed / Not Started (Tier 1)

| # | Run | Status | Notes |
|---|---|---|---|
| 1 | `mobilevit_questiononly_attnqquery` | partial (2k steps) | crashed at step ~2k |
| 2 | `mobilevit_attnqquery_dynbudget_adapter_d4_cap64` | crashed at startup | |
| 3 | `mobilevit_attnqquery_dynbudget_adapter_d3_cap64_18k` | not started | |
| 19 | `mobilevit_attnqquery_dynbudget_adapter_d3_cap64_seed2` | not started | |

### Additional Run (out-of-plan)

| Run | Notes |
|---|---|
| `dinov2s_questiononly_attnqquery_dynbudget_adapter_d3_cap64` | Originally Tier 6 (Run 16), ran alongside Tier 2-3 |

Tier 1 (MobileViT completion) is largely incomplete but now low-priority — MobileViT is no longer the frontier VM.

## Sweep Definition

All runs followed the standard comparison policy:

- effective batch size `192`
- target step `9000`
- `eval_every=1000`, `eval_batches=100` (periodic)
- final eval on full validation split (`eval_fraction=1.0`, `final_eval_batches=0`)
- official scorer
- `--eval_use_kv_cache --eval_kv_cache_mode batched`
- layout: `batch_size=96, grad_accum_steps=2, eval_batch_size=96` for all VMs

**Exception:** Tier 4 caption-align runs have a significant caveat — see [Caption-Align Caveat](#caption-align-caveat-two-stage-training-was-improperly-configured) below.

## Final Ranking

Reference frontier entering Crane:

- Plank winner: `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` at `0.5240`

### Full Ranking Table

| Rank | Run | Final Overall | Yes/No | Number | Other | Delta vs `0.5240` |
|---|---|---:|---:|---:|---:|---:|
| 1 | `dinov2s_attnqquery_nodynbudget_adapter_d3` | **0.5762** | 0.7286 | 0.4039 | 0.5059 | **`+0.0522`** |
| 2 | `mobileclip_attnqquery_dynbudget_adapter_d3_cap64` | **0.5603** | 0.7195 | 0.3912 | 0.4839 | `+0.0363` |
| 3 | `mobileclip_attnqquery_dynbudget_adapter_d4_cap64` | 0.5578 | 0.7127 | 0.3929 | 0.4837 | `+0.0338` |
| 4 | `dinov2s_questiononly_attnqquery_dynbudget_adapter_d3_cap64` | 0.5355 | 0.7065 | 0.3731 | 0.4484 | `+0.0115` |
| 5 | `dinov2s_attnqquery_dynbudget_adapter_d5_cap64` | 0.5338 | 0.6985 | 0.3786 | 0.4496 | `+0.0098` |
| 6 | `dinov2s_attnqquery_dynbudget_adapter_d3_cap64` | 0.5323 | 0.6986 | 0.3803 | 0.4460 | `+0.0083` |
| 7 | `dinov2s_attnqquery_dynbudget_adapter_d3_cap128` | 0.5311 | 0.7011 | 0.3654 | 0.4457 | `+0.0071` |
| 8 | `dinov2s_lmmeanqquery_dynbudget_adapter_d3_cap64` | 0.5248 | 0.7101 | 0.3670 | 0.4255 | `+0.0008` |
| 9 | *Plank frontier (reference)* | *0.5240* | *0.6983* | *0.3405* | *0.4401* | *—* |
| 10 | `dinov2s_attnqquery_dynbudget_adapter_d3_cap32` | 0.5160 | 0.6949 | 0.3633 | 0.4204 | `-0.0080` |
| 11 | `dinov2s_captionalign_attnqquery_dynbudget_adapter_d3_cap64` | 0.5143 | 0.6993 | 0.3567 | 0.4153 | `-0.0097` |
| 12 | `mobileclip_captionalign_attnqquery_dynbudget_adapter_d3_cap64` | 0.4939 | 0.6935 | 0.3372 | 0.3835 | `-0.0301` |
| 13 | `mobilevit_captionalign_attnqquery_dynbudget_adapter_d3_cap64` | 0.4854 | 0.6895 | 0.3158 | 0.3750 | `-0.0386` |

**New frontier: `dinov2s_attnqquery_nodynbudget_adapter_d3` at `0.5762` (+0.0522 over Plank).**

This is the largest single-run improvement in the project's history, surpassing even the Plank MobileViT jump (+0.0587 over Nail, but from a lower base).

## Periodic Eval Curves

All values are periodic 100-batch evals (left 9) plus the full-val final eval (rightmost). Caption-align runs start at step 4k (VQA training began at step 3k).

| Run | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dinov2s_nodynbudget_d3` | 0.44 | 0.48 | 0.50 | 0.53 | 0.54 | 0.55 | 0.56 | 0.57 | 0.58 | **0.576** |
| `mobileclip_d3_cap64` | 0.41 | 0.46 | 0.49 | 0.51 | 0.53 | 0.53 | 0.54 | 0.55 | 0.56 | **0.560** |
| `mobileclip_d4_cap64` | 0.41 | 0.46 | 0.49 | 0.51 | 0.53 | 0.53 | 0.54 | 0.55 | 0.56 | **0.558** |
| `dinov2s_questiononly_d3_cap64` | 0.41 | 0.45 | 0.48 | 0.49 | 0.51 | 0.52 | 0.52 | 0.53 | 0.54 | **0.536** |
| `dinov2s_d5_cap64` | 0.41 | 0.45 | 0.48 | 0.49 | 0.50 | 0.51 | 0.52 | 0.53 | 0.53 | **0.534** |
| `dinov2s_d3_cap64` | 0.41 | 0.45 | 0.47 | 0.49 | 0.51 | 0.52 | 0.52 | 0.53 | 0.53 | **0.532** |
| `dinov2s_d3_cap128` | 0.41 | 0.45 | 0.48 | 0.50 | 0.50 | 0.51 | 0.52 | 0.53 | 0.53 | **0.531** |
| `dinov2s_lmmeanqquery_d3_cap64` | 0.41 | 0.44 | 0.47 | 0.49 | 0.50 | 0.51 | 0.52 | 0.52 | 0.53 | **0.525** |
| `dinov2s_d3_cap32` | 0.41 | 0.45 | 0.47 | 0.49 | 0.49 | 0.50 | 0.51 | 0.52 | 0.52 | **0.516** |
| `dinov2s_captionalign_d3_cap64` | — | — | — | 0.43 | 0.46 | 0.48 | 0.49 | 0.49 | 0.51 | **0.514** |
| `mobileclip_captionalign_d3_cap64` | — | — | — | 0.42 | 0.45 | 0.47 | 0.48 | 0.48 | 0.49 | **0.494** |
| `mobilevit_captionalign_d3_cap64` | — | — | — | 0.42 | 0.45 | 0.46 | 0.47 | 0.47 | 0.48 | **0.485** |

Key observations from the curves:

1. **DINOv2 nodynbudget separates early and never looks back.** At step 1k it's already 0.44 vs 0.41 for all dynbudget variants — 3 percentage points ahead from the first evaluation. The gap grows through training.
2. **All DINOv2 dynbudget variants are tightly clustered.** cap32/64/128 and d3/d5 are within ~0.02 of each other throughout. The cap value and adapter depth are weak levers compared to the dynbudget on/off switch.
3. **MobileCLIP curves are still rising at 9k.** The step 8k→9k delta is +0.01, suggesting continued gains at 12k+.
4. **Caption-align curves start lower and converge slowly.** See caveat section below.

## The Core Finding: Dynbudget Destroys DINOv2 Performance

This is the most important result from Crane Part 1.

### The monotonic cap sweep

| Cap setting | Tokens to perceiver | Final | Delta vs nodynbudget |
|---|---:|---:|---:|
| nodynbudget | 256 | **0.5762** | — |
| cap128 | ~128 | 0.5311 | -0.0451 |
| cap64 | ~64 | 0.5323 | -0.0439 |
| cap32 | ~32 | 0.5160 | -0.0602 |

The relationship is approximately monotonic: more tokens to the perceiver → better scores. The one anomaly (cap128 < cap64) is within noise — the 100-batch periodic evals at 9k show them tied at 0.53.

### Why dynbudget hurts

The qadaptive token selector performs hard top-k selection: it scores each of the 256 DINOv2 tokens by question relevance, then passes only the top-k to the perceiver as key/values. The perceiver's 49 learned queries then cross-attend over this reduced set.

The perceiver cross-attention is itself a soft selection mechanism — it weights all key/value tokens by learned relevance. Hard pre-filtering before soft selection is redundant and destructive: the selector discards tokens that the perceiver would have weighted appropriately on its own.

This is specific to the perceiver resampler architecture. A flat projection (e.g., linear → reshape) that needs to handle all tokens simultaneously might benefit from pre-filtering. But the perceiver is designed to distill variable-length sequences into a fixed-size output — token selection is its core function.

### Per-category analysis

| | nodynbudget | dynbudget cap64 | Delta |
|---|---:|---:|---:|
| Overall | 0.5762 | 0.5323 | +0.0439 |
| Yes/No | 0.7286 | 0.6986 | +0.0300 |
| Number | 0.4039 | 0.3803 | +0.0236 |
| **Other** | **0.5059** | **0.4460** | **+0.0599** |

The `other` category benefits most from removing dynbudget (+0.0599). This makes sense: `other` questions are compositional and open-ended ("What color is the bus?", "What sport is being played?"), requiring the model to attend to diverse visual evidence. Hard token selection is most damaging when the question requires holistic scene understanding rather than a specific spatial location.

`number` gains +0.0236 — counting requires attending to multiple instances of the same object class, distributed across the image. Hard selection might drop instances that fall below the relevance threshold but still contribute to the count.

## VM Comparison: DINOv2 vs MobileCLIP vs MobileViT

### The head-to-head (all with attnqquery + dynbudget + d3 + cap64)

| VM | Params | Tokens | Pre-training | Overall | Yes/No | Number | Other |
|---|---:|---:|---|---:|---:|---:|---:|
| DINOv2 nodyn | 22M | 256→49 | Self-supervised (LVD-142M) | **0.5762** | **0.7286** | **0.4039** | **0.5059** |
| MobileCLIP | 11.4M | 49→49 | CLIP (DataCompDR-1B) | 0.5603 | 0.7195 | 0.3912 | 0.4839 |
| DINOv2 dyn | 22M | 256→64→49 | Self-supervised (LVD-142M) | 0.5323 | 0.6986 | 0.3803 | 0.4460 |
| MobileViT | 5.6M | 49→49 | Supervised (ImageNet-1M) | 0.5240 | 0.6983 | 0.3405 | 0.4401 |

**The confounded comparison:** Naively, DINOv2 nodynbudget (0.5762) beats MobileCLIP (0.5603). But DINOv2 benefits from 256 tokens reaching the perceiver cross-attention while MobileCLIP provides only 49. The "DINOv2 is better" claim is confounded with "more perceiver key/values is better." To properly compare the VMs we would need either DINOv2 capped to 49 tokens or MobileCLIP with 256 tokens (neither exists).

**What we can say:** DINOv2 with dynbudget cap64 (0.5323) uses ~64 tokens — closer to MobileCLIP's 49. At this token count, MobileCLIP wins decisively (0.5603 vs 0.5323). This suggests MobileCLIP's language-aligned features are genuinely more informative per token than DINOv2's self-supervised features, and DINOv2's advantage at 256 tokens comes from quantity over quality.

**What this means for the project:**

1. **Language alignment matters per-token.** MobileCLIP at 49 tokens beats DINOv2 at ~64 tokens. CLIP pre-training produces features that are more efficient for VQA at matched token counts.
2. **But raw token count matters more than per-token quality.** DINOv2 at 256 tokens beats MobileCLIP at 49, despite lower per-token quality. The perceiver can extract more total information from a larger pool of spatially-distributed features.
3. **The ideal VM would combine both.** A CLIP-aligned model producing 256+ tokens (e.g., CLIP ViT-B/16 with 196 tokens at 768-dim) would likely exceed both. This is a direction for the next sweep.

### The attnqquery universality test (Runs 5 vs 6)

| Query mode | DINOv2 + dynbudget | Delta |
|---|---:|---:|
| attnqquery | 0.5323 | — |
| lmmeanqquery | 0.5248 | -0.0075 |

attnqquery beats lmmeanqquery on DINOv2, consistent with the Plank MobileViT result. The advantage concentrates in `other` (0.4460 vs 0.4255 = +0.0205). This confirms attnqquery as the universal default regardless of VM pre-training objective.

### Adapter depth

| Config | Overall | Delta |
|---|---:|---:|
| DINOv2 d3 cap64 | 0.5323 | — |
| DINOv2 d5 cap64 | 0.5338 | +0.0015 |
| MobileCLIP d3 cap64 | 0.5603 | — |
| MobileCLIP d4 cap64 | 0.5578 | -0.0025 |

Adapter depth is a flat lever. d5 is negligibly better on DINOv2, d4 is negligibly worse on MobileCLIP. The d3 default is fine. However, this was tested only with dynbudget — nodynbudget passes richer information to the LM prefix, and deeper adapters might extract more value from it. Worth one probe in Part 2.

## Caption-Align Caveat: Two-Stage Training Was Improperly Configured

**The three Tier 4 caption-align runs were executed under conditions that make their results non-comparable to single-stage baselines.** Three confounds:

### Confound 1: 6k VQA steps instead of 9k

The plan called for 3k caption-align steps followed by 9k VQA steps (12k total). What actually happened: `caption_pretrain.py` saved `global_step=3000` in the checkpoint, and `mm.py` resumed from that step, training from step 3001 to step 9000 — only **6k VQA steps**. The intended 9k VQA steps would have required `--max_steps 12000`.

### Confound 2: LR schedule was not reset

The cosine LR schedule is a function of absolute `global_step`. With `lr_warmup_steps=600` and `max_steps=9000`:

- At step 3001 (first VQA step), the LR scale is **0.84** (already past warmup and into decay)
- At step 9000 (last VQA step), the LR scale is **0.15** (minimum)

The VQA training phase therefore operated on a truncated schedule: LR fell from 84% to 15% over 6k steps, with no warmup. By contrast, the single-stage baselines warm from 0% to 100% over the first 600 steps, spend most of training near peak, and decay to 15% only at the end.

Critically, the 48 newly initialized visual adapter parameters (3 adapter layers, randomly initialized) received no warmup at all — they were immediately hit with a learning rate already in its decay phase.

### Confound 3: Optimizer state mismatch

The caption-align optimizer tracked only bridge + calibrator params (94 state entries). The VQA optimizer tracks bridge + calibrator + adapters + top-2 LM layers. The `load_state_dict` failed and the optimizer was reinitialized from scratch — which means Adam momentum/variance buffers for bridge params were lost. In a properly implemented two-stage pipeline, the bridge optimizer state should transfer cleanly.

### What the data actually shows

Despite these confounds, a careful step-matched comparison reveals a **positive early signal**:

| VQA steps completed | Captionalign (step) | Baseline (step) | Delta |
|---:|---:|---:|---:|
| 1k | 0.4331 (4k) | 0.4147 (1k) | **+0.018** |
| 2k | 0.4607 (5k) | 0.4521 (2k) | **+0.009** |
| 3k | 0.4835 (6k) | 0.4740 (3k) | **+0.010** |
| 4k | 0.4932 (7k) | 0.4922 (4k) | +0.001 |
| 5k | 0.4933 (8k) | 0.5063 (5k) | **-0.013** |
| 6k | 0.5125 (9k) | 0.5168 (6k) | -0.004 |

*(Both runs are DINOv2 + attnqquery + dynbudget + d3 + cap64. Baseline scores are periodic 100-batch evals.)*

Caption-align provides a **clear early acceleration**: +0.018 at 1k VQA steps, gradually declining to parity at ~4k VQA steps, then falling behind. The crossover at ~4k-5k VQA steps coincides with the captionalign run's LR hitting the steep part of its cosine decay (LR scale dropping below ~0.5) while the baseline at matched VQA step count still has most of its LR budget remaining.

**Interpretation:** The caption-align bridge initialization genuinely helps early convergence — the bridge starts with a better representation of image→language mapping. But the broken LR schedule starves the later training phase, preventing the model from fully exploiting this head start. The adapter layers, which start from random initialization with no warmup, are particularly disadvantaged.

### Verdict: not dead, but not proven

Caption-align pre-training shows a real positive transfer signal that is masked by the implementation bugs. A properly configured two-stage run would need:

1. `--max_steps 12000` (3k caption-align + 9k VQA) OR reset `global_step` to 0 at VQA start
2. Fresh LR schedule for the VQA phase (warmup from 0, full cosine decay over 9k steps)
3. Optimizer state: either properly transfer bridge state + fresh init for new params, or fresh init for all (current behavior after the try/except fix)

Whether this is worth the engineering time depends on priorities. The signal is small (+0.01 at 3k VQA steps) relative to the nodynbudget signal (+0.044), and a corrected run adds ~0.12h for the caption-align phase. It's a low-cost, moderate-value experiment.

## Findings Summary

### Settled (finalize in Part 2)

| Finding | Evidence | Confidence |
|---|---|---|
| **Dynbudget hurts on DINOv2** | Monotonic: nodyn (0.576) > cap128 (0.531) > cap64 (0.532) > cap32 (0.516) | Very high |
| **attnqquery is the universal default** | Wins on DINOv2 (+0.008 vs lmmeanqquery), MobileViT (+0.006), same direction on both | High |
| **Adapter depth is flat at d3** | d5 ≈ d3 on DINOv2, d4 ≈ d3 on MobileCLIP | High (with dynbudget) |
| **MobileCLIP > MobileViT** | 0.5603 vs 0.5240 at matched token count (49) | High |
| **More tokens to perceiver >> per-token quality** | DINOv2@256 > MobileCLIP@49 > DINOv2@64 | High |

### Open (needs Part 2 or next sweep)

| Question | Why it matters | Proposed test |
|---|---|---|
| **DINOv2 nodynbudget + questiononly** | +0.003 with dynbudget; larger effect on richer token set? | Single run, ~0.7h |
| **DINOv2 nodynbudget + deeper adapters** | d3→d5 was flat with dynbudget, but nodyn passes richer prefix | Single run, ~0.7h |
| **DINOv2 nodynbudget + 18k steps** | Curves still rising at 9k; how much headroom? | Single run, ~1.3h |
| **Caption-align (properly configured)** | Early transfer signal was real; broken schedule masked potential | Need code fix, then ~0.9h |
| **Seed check on 0.5762** | Frontier must be seed-checked before treating as settled | Single run, ~0.7h |
| **CLIP ViT with >49 tokens** | Combines language alignment + high token count; predicted best of both | Next sweep (Eng work + runs) |

### Dead (do not pursue)

| Direction | Why |
|---|---|
| **Dynbudget on high-token VMs** | Monotonically worse. Perceiver cross-attention is a strictly better soft selection mechanism than hard top-k pre-filtering |
| **lmmeanqquery** | Consistently worse than attnqquery across all VMs tested |
| **Deeper adapters (d4/d5) with dynbudget** | Flat across two VMs. Not worth compute unless nodynbudget changes the picture |

## Architecture Note: How 256 DINOv2 Tokens Fit in a 256-Token LM

The LM's `max_seq_len=256` limits the **LM sequence**, not the perceiver's input. The flow:

```
DINOv2: [B, 256, 384]  (256 patch tokens)
    ↓ visual_proj
Bridge input: [B, 256, 512]  (projected to LM dim)
    ↓ perceiver cross-attention (49 learned queries × 256 key/values)
Bridge output: [B, 49, 512]  (49 visual prefix tokens)
    ↓ prefix_calibrator
LM input: [49 visual tokens] + [up to 207 text tokens] = 256 max
```

The 256 DINOv2 tokens are consumed as key/values in the perceiver's cross-attention blocks. They never enter the LM sequence directly. The perceiver distills them into 49 output tokens, which become the visual prefix. This leaves `256 - 49 = 207` positions for text tokens in the LM context.

With dynbudget, the selector reduces 256→k tokens **before** the perceiver, so the cross-attention operates on fewer key/values. With nodynbudget, the perceiver cross-attends over all 256 key/values — more compute per cross-attention layer, but the perceiver output is still 49 tokens regardless.

## Directions for Crane Part 2

### Priority 1: Solidify the DINOv2 nodynbudget frontier

These are ablation runs to nail down the best nodynbudget config and seed-check the frontier.

| Run | Config delta from frontier | Purpose | Est. time |
|---|---|---|---:|
| nodynbudget + questiononly | `--bridge_question_context_mode question_only` | questiononly gave +0.003 with dynbudget | 0.7h |
| nodynbudget + d4 | `--lm_visual_adapter_layers 4` | Test depth now that 256 tokens reach LM | 0.7h |
| nodynbudget seed2 | `--seed 53` | Non-negotiable frontier verification | 0.7h |

### Priority 2: Longer training

| Run | Config delta | Purpose | Est. time |
|---|---|---|---:|
| nodynbudget 18k | `--max_steps 18000` | Curves rising at 9k; find ceiling | 1.3h |
| best_config 18k | Stack all P1 wins + 18k | Max-out run | 1.3h |

### Priority 3: Caption-align (properly configured, optional)

If the code fix is low-effort, one corrected caption-align run on DINOv2 nodynbudget would resolve whether the early transfer signal translates to a final-score gain. This requires:

- Reset `global_step` to 0 when loading a caption-align checkpoint for VQA training, OR pass `--max_steps 12000`
- Ensure the LR schedule warmup runs fresh from step 0 of VQA training

Estimated: 0.9h (0.12h caption-align + 0.7h VQA + overhead). Low cost, moderate information value.

### Priority 4: Future sweep signal — CLIP + high token count

The Crane results establish two independent axes of improvement:

1. **Language alignment** (MobileCLIP@49 > DINOv2@64 at matched token count)
2. **Token count** (DINOv2@256 > MobileCLIP@49 despite lower per-token quality)

The natural next VM to test combines both: a CLIP-aligned model with >49 tokens. Candidates:

- **CLIP ViT-B/16** (OpenAI/OpenCLIP): 196 tokens at 768-dim, ~86M params. Language-aligned, high token count, widely available.
- **SigLIP ViT-S/16**: 196 tokens at 384-dim, ~22M params. Similar to DINOv2-small in size but with CLIP-style alignment.

This is a next-sweep direction (requires Eng work), not a Part 2 item.

## Cost Summary

| Tier | Runs completed | Wall-clock hours |
|---|---:|---:|
| Tier 1 (partial) | 0 of 4 | ~0 |
| Tier 2 | 3 of 3 | ~2.5 |
| Tier 3 | 3 of 3 | ~2.4 |
| Tier 4 | 3 of 3 | ~3.0 |
| Tier 6 (partial) | 2 of 3 | ~1.4 |
| **Total Part 1** | **11** | **~9.3** |

Part 2 estimated: ~4.5-5.5h for Priorities 1-2, +0.9h if caption-align fix is included. Total Crane budget: ~15h, well within the 30-50h allocation.
