# 37 Hardhat Sweep Plan (2026-03-15)

## Ancestry

This plan supersedes "Crane Part 2." After reviewing the Crane Part 1 results ([36_crane_part1_sweep_report_2026-03-15.md](36_crane_part1_sweep_report_2026-03-15.md), [36codex_crane_sweep_report_2026-03-15.md](36codex_crane_sweep_report_2026-03-15.md)), two clear workstreams emerged:

1. **Solidify the DINOv2 nodynbudget frontier** (seed check, ablations, longer training)
2. **Test a language-aligned high-token VM** (the strongest remaining lever)

Rather than treat (1) as a "Part 2" appendix to Crane and (2) as a separate future sweep, Hardhat combines both into a single execution plan with shared infrastructure and decision gates.

Sources:
- `tasks/mm_bridge/docs/36_crane_part1_sweep_report_2026-03-15.md`
- `tasks/mm_bridge/docs/36codex_crane_sweep_report_2026-03-15.md`
- `tasks/mm_bridge/docs/COWORKER_CHATTER.md` (Claude + Codex discussion)
- `tasks/mm_bridge/docs/34_crane_extended_sweep_plan_2026-03-14.md`
- Crane run bundles `logs/mmcrane_v1_20260314_*`

## Goal

Two goals, in priority order:

1. **Establish the best score achievable on this LM with the DINOv2-small nodynbudget family.** This means seed-checking 0.5762, testing the remaining ablation axes (questiononly, adapter depth, perceiver depth, longer training), and determining whether caption-align pre-training adds value when properly configured.

2. **Test whether language-aligned visual features improve per-token quality at scale.** Crane showed that MobileCLIP (CLIP-aligned, 49 tokens) beat DINOv2 (self-supervised, ~64 tokens) at matched token count. The natural follow-up is a CLIP-family model with a large token grid — combining language alignment with the token quantity that made DINOv2 nodynbudget win overall.

## Entering State

**Current frontier:** `dinov2s_attnqquery_nodynbudget_adapter_d3` at **0.5762** (single seed).

**Settled from Crane:**
- attnqquery is the default query mechanism (beats lmmeanqquery on DINOv2 and MobileViT)
- Dynbudget is destructive on high-token VMs feeding a perceiver resampler
- Adapter depth (d3 vs d4 vs d5) is flat under dynbudget — untested under nodynbudget
- Caption-align pre-training showed positive early transfer but was tested under broken conditions (see Crane Part 1 report, "Caption-Align Caveat")

**Bridge config baseline (Crane frontier):**
```
--bridge_type perceiver_resampler
--bridge_query_depth 3
--bridge_num_heads 8
--num_visual_tokens 49
--bridge_token_reduce adaptive_pool
--bridge_add_2d_pos_emb
--bridge_question_conditioning
--bridge_question_context_mode prompt_only
--bridge_query_bank_mode question_hidden_attn
--bridge_qquery_scale 1.0
--bridge_token_selector_type none    # nodynbudget
--bridge_token_select_k 0           # nodynbudget
--lm_visual_adapter_type cross_attn
--lm_visual_adapter_layers 3
--lm_visual_adapter_num_heads 8
--lm_visual_adapter_dropout 0.0
--lm_visual_adapter_gate_init 0.5
```

## Budget

**Remaining from Crane allocation:** ~21-41h (original 30-50h minus ~9.3h Crane Part 1).

Hardhat estimated total: **12-16h** depending on new-VM perf profiles and conditional runs. Well within budget.

## Engineering Prerequisites

### Eng-1: Fix Two-Stage Caption-Align Pipeline

**Problem:** The Crane caption-align runs had three confounds (detailed in [36_crane_part1_sweep_report_2026-03-15.md](36_crane_part1_sweep_report_2026-03-15.md)):
1. `global_step=3000` from the caption checkpoint caused VQA training to run steps 3001-9000 (6k VQA steps) instead of the intended 9k
2. The cosine LR schedule continued from step 3000 (already at 84% of peak, no warmup) instead of resetting
3. Optimizer state mismatch (bridge-only state vs full model) — already handled by the try/except fix

**Fix:** Add a `--reset_schedule` flag to `train/mm.py` that, when set:
```python
if args.reset_schedule:
    global_step = 0
    start_epoch = 0
    resume_batch_in_epoch = 0
    # Don't restore optimizer state (already handled by try/except)
```

This loads the model weights from the checkpoint but restarts training metadata. The LR schedule will warm up fresh from step 0.

**Code changes:**
1. Add `--reset_schedule` arg to `train/mm.py` argparser
2. Add 3-line conditional after checkpoint loading (lines ~2922-2925)
3. Update `launch_hardhat_sweep_v1.sh` `run_twostage()` to pass `--reset_schedule` and `--max_steps 9000` (not 12000 — clean 9k VQA steps)

**Smoke test:** Load a Crane caption-align checkpoint with `--reset_schedule`, verify `global_step` starts at 0, LR warms from 0, training runs full 9k steps.

**Effort:** ~30 minutes including smoke test.

### Eng-2: SigLIP-B/16 Integration

**Correction from Crane discussion:** SigLIP ViT-S/16 does not exist. The smallest SigLIP patch-16 model is **SigLIP ViT-B/16** (`google/siglip-base-patch16-224`), with:
- **196 patch tokens** (14x14 at 224x224 input) at **768-dim**
- **~86M** vision encoder params
- 12 layers, 12 heads
- Sigmoid contrastive loss on WebLI (language-aligned)
- **No CLS token** — SigLIP uses mean pooling, so all 196 tokens are patch tokens (no stripping needed)

This means the "clean capacity-matched comparison" with DINOv2-small (22M, 384-dim) is **not possible** — SigLIP-B is 4x larger with 2x wider features. Any comparison between SigLIP-B and DINOv2-small is confounded by model capacity.

**What we can still learn:** SigLIP-B/16 at 196 tokens with nodynbudget tests whether a language-aligned high-token VM exceeds the DINOv2-small frontier. If it does, we learn that the combination of alignment + tokens matters. If it doesn't (despite 4x more params), that's strong evidence that DINOv2's self-supervised spatial features are genuinely better for VQA than CLIP-style alignment, at least through our bridge.

**For a cleaner capacity comparison**, we would need DINOv2-B/14 (`facebook/dinov2-base`, ~86M, 256 tokens at 768-dim). SigLIP-B/16 (86M, 196 tokens, 768-dim, language-aligned) vs DINOv2-B/14 (86M, 256 tokens, 768-dim, self-supervised) is the closest we can get. Same param count, same feature dim, similar token count (196 vs 256), only pre-training differs. This is Eng-3.

**Code changes:** Same pattern as DINOv2 wrapper.
1. Add `HFSigLIPBasePatch16Backbone` to `models/hf_vision.py` (~60 lines)
2. Add `download_siglip_base_patch16` function
3. Add `"siglip_base"` to `--vision_model` choices
4. Add elif branch in `build_vision_model_from_args()`
5. Download model to `logs/hf_vision/google_siglip_base_patch16_224/`

Key wrapper notes:
- No CLS token stripping needed (SigLIP doesn't use CLS)
- Standard ImageNet normalization
- RGB order, 224x224 target size
- Output: `[B, 196, 768]`

**Memory concern:** 86M frozen params + 768-dim features are both significantly larger than DINOv2-small's 22M + 384-dim. The bridge `visual_proj` (LazyLinear from 768 to 512) will have 2x the input, and the perceiver cross-attention will operate on 196 key/values at 512-dim (after projection). Perf probes required — b96a2 may not fit. Fallback layouts: b64a3, b48a4, b32a6.

**Effort:** ~1 hour including download and smoke test.

### Eng-3: DINOv2-B/14 Integration (Optional, Capacity-Matched Comparison)

**Purpose:** Provide a capacity-matched comparison for SigLIP-B/16. Same param count (~86M), same feature dim (768), similar token count (256 vs 196), only pre-training differs.

**Model:** `facebook/dinov2-base` — ViT-B/14, 86M params, 256 tokens at 768-dim at 224x224.

**Code changes:** Minimal — reuse `HFDINOv2SmallBackbone` pattern with different model ID and constants. The existing CLS-stripping logic applies.

**Effort:** ~30 minutes (mostly download time).

**Decision:** Only build if Tier 4 (SigLIP-B) results are interesting enough to warrant the controlled comparison. If SigLIP-B massively beats or clearly loses to DINOv2-small, the DINOv2-B run may not be needed.

## Tier 1: DINOv2 Nodynbudget Solidification

No new code required. All runs use the Crane frontier config as baseline. Can start immediately.

All runs: DINOv2-small, attnqquery, nodynbudget, b96a2.

### Run 1: `dinov2s_attnqquery_nodynbudget_adapter_d3_seed2`

**Purpose:** Seed check of the 0.5762 frontier. Non-negotiable.

**Config delta:** `--seed 53`

**Expected range:** 0.565–0.585 (Plank seed variance was 0.005 on MobileViT).

**Est. time:** 0.7h

### Run 2: `dinov2s_questiononly_attnqquery_nodynbudget_adapter_d3`

**Purpose:** questiononly gave +0.003 on DINOv2 with dynbudget (0.5355 vs 0.5323). With nodynbudget, the perceiver cross-attends over all 256 tokens — a richer visual set that may benefit more from a sharper question signal. The attention-derived query focuses retrieval on question-relevant visual content; with more tokens to select from, the sharpening might matter more.

**Config delta:** `--bridge_question_context_mode question_only`

**Expected range:** 0.576–0.585.

**Information value:** HIGH. Cheapest shot at beating the frontier.

**Est. time:** 0.7h

### Run 3: `dinov2s_attnqquery_nodynbudget_adapter_d4`

**Purpose:** Adapter depth was flat under dynbudget (d5 ≈ d3), but that test had a hard information ceiling — the selector discarded tokens before the perceiver could extract them. With nodynbudget, 256 tokens are distilled to 49 prefix tokens containing strictly more information. Deeper adapters give the LM more opportunities to re-access this richer prefix during generation.

With d4, adapters are placed at roughly every 3 layers in the 12-layer LM (layers 8, 9, 10, 11 with 2 frozen top layers + adapters). Each adapter layer performs cross-attention from LM hidden states to the 49-token visual prefix.

**Config delta:** `--lm_visual_adapter_layers 4`

**Expected range:** 0.575–0.585.

**Information value:** MEDIUM-HIGH. If d4 helps here, it's the first evidence that adapter depth matters when the prefix is rich enough.

**Est. time:** 0.7h

### Run 4: `dinov2s_attnqquery_nodynbudget_adapter_d3_qdepth4`

**Purpose:** The perceiver has `query_depth=3` with 8 heads. With 256 key/values in nodynbudget mode, each head attends over 32 positions per query per layer — 96 total attention "looks" per query across all 3 layers. A 4th cross-attention layer gives each query 128 total looks, a 33% increase in extraction capacity.

The perceiver depth hasn't been swept since Nail. In the Nail regime (49 tokens from the old VM, dynbudget), depth didn't matter because there wasn't much to extract. With 256 dense DINOv2 tokens, the perceiver is now the bottleneck between a rich visual input and a fixed 49-token prefix.

**Config delta:** `--bridge_query_depth 4`

**Expected range:** 0.575–0.590. Wider range reflects genuine uncertainty — perceiver depth is unexplored territory in the dense-token regime.

**Information value:** MEDIUM-HIGH. If depth helps, it informs perceiver config for all future high-token VMs.

**Est. time:** 0.7h (slightly more compute per step due to extra cross-attention layer, but still fast).

### Tier 1 Subtotal: ~2.8h (4 runs)

**Decision gate after Tier 1:**
- If seed2 drops below 0.565: the frontier is noisy, temper expectations.
- If any of Runs 2-4 beat 0.5762: incorporate the winning delta into the Tier 2 stacking run and the Tier 4 new-VM config.
- If all of Runs 2-4 are flat: the DINOv2-small nodynbudget d3 config is near-optimal for this VM. Adapter depth, perceiver depth, and question mode are weak levers. The next gain must come from the VM itself (Tier 4).

## Tier 2: Longer Training

### Run 5: `dinov2s_attnqquery_nodynbudget_adapter_d3_18k`

**Purpose:** The Crane nodynbudget learning curve was still rising at 9k (periodic evals: 0.57 → 0.58 in the last 2k steps). 18k tests whether the current family has another real slope or is approaching plateau.

**Config delta:** `--max_steps 18000`

**Expected range:** 0.585–0.600.

**Information value:** HIGH. If the curve is still steep at 18k, the current LM has more headroom than expected and we should consider even longer runs. If it flattens, the next gain requires a different lever (VM, LM, or architectural change).

**Est. time:** 1.3h

### Run 6: `dinov2s_[best_tier1_config]_18k`

**Conditional on Tier 1:** If any Tier 1 run beat the frontier, stack the winning delta(s) with 18k training. If Tier 1 was flat, skip this run (Run 5 already covers the long-training question).

**Est. time:** 1.3h (if run)

### Tier 2 Subtotal: 1.3–2.6h

**Decision gate after Tier 2:**
- The 18k result tells us whether the LM is saturating. If 18k barely improves over 9k (delta < 0.003), the LM ceiling is near. If 18k gives > 0.008, the ceiling is higher than expected.
- This informs whether Tier 4 new-VM runs should use 9k or 18k.

## Tier 3: Corrected Caption-Align (Requires Eng-1)

### Run 7: `dinov2s_captionalign_attnqquery_nodynbudget_adapter_d3`

**Purpose:** The definitive caption-align test. Crane's caption-align runs were executed under three confounds (broken LR schedule, 6k instead of 9k VQA steps, lost optimizer state). Despite this, step-matched comparison showed +0.018 early transfer at 1k VQA steps (see [36_crane_part1_sweep_report_2026-03-15.md](36_crane_part1_sweep_report_2026-03-15.md), "Caption-Align Caveat").

This run uses the Eng-1 fix (`--reset_schedule`) for a clean measurement:
- Stage 1: 3k caption-align steps on DINOv2 features (bridge + calibrator only, ~7 min)
- Stage 2: 9k VQA steps from the pre-trained bridge, with fresh LR schedule (warmup from 0, full cosine decay)

**Config:** Crane frontier (DINOv2-S nodynbudget d3) + two-stage training with `--reset_schedule`.

**Expected range:** 0.575–0.590. If the early transfer signal holds with a correct schedule, caption-align should match or exceed the single-stage baseline. If it still underperforms, the pre-training objective is genuinely unhelpful for this bridge architecture.

**Information value:** MEDIUM. Both Codex and I agree this should run after the nodynbudget solidification runs, but it's cheap (0.9h) and resolves a question that has been hanging since Crane.

**Est. time:** 0.9h (0.12h caption-align + 0.7h VQA + overhead)

### Tier 3 Subtotal: 0.9h

## Tier 4: SigLIP-B/16 — Language-Aligned High-Token VM (Requires Eng-2)

This is the "CLIP semantics + large token grid" hypothesis that both Crane reports and the coworker discussion converged on.

### Why SigLIP-B/16

Crane established two independent axes of improvement:
1. Language alignment helps per-token (MobileCLIP@49 > DINOv2@~64 at matched token count)
2. Token quantity overwhelms per-token quality (DINOv2@256 > MobileCLIP@49)

SigLIP-B/16 combines both: 196 language-aligned tokens at 768-dim. The question is whether the combination exceeds DINOv2-small's 256 self-supervised tokens.

**Capacity caveat:** SigLIP-B is ~86M params vs DINOv2-small's ~22M. This means a SigLIP-B win could be attributed to capacity rather than alignment. We acknowledge this confound. If SigLIP-B wins, the DINOv2-B comparison (Eng-3, Tier 5) would be needed to isolate the alignment contribution. If SigLIP-B loses despite 4x more params, that's an even stronger signal — language alignment actively hurts when the perceiver can access dense spatial features.

### Perf Probes (pre-Tier 4)

Before running real training, probe SigLIP-B/16 at the standard batch layouts:
- 192x1, 96x2, 64x3, 48x4, 32x6

The larger VM (86M frozen) and wider features (768-dim) may require smaller batches. If b96a2 doesn't fit, use whatever the probe recommends. Effective batch 192 must be maintained.

**Est. time:** 0.5h for probes.

### Run 8: `siglip_attnqquery_nodynbudget_adapter_d3`

**Purpose:** SigLIP-B/16 baseline with the Crane frontier bridge config (nodynbudget, attnqquery, d3). Direct comparison to the DINOv2-small frontier.

**Key architectural note:** SigLIP-B produces 196 tokens (14x14). With `num_visual_tokens=49` and `bridge_token_reduce=adaptive_pool`, the bridge first spatially pools 196→49 tokens before the perceiver sees them. Alternatively, with nodynbudget, we could let all 196 tokens pass to the perceiver as key/values (same as DINOv2's 256). The `adaptive_pool` step happens before the perceiver for token_reduce, so with 196 input tokens and 49 output queries, the perceiver cross-attends over either 49 pooled tokens or 196 raw tokens depending on configuration.

**Decision: run nodynbudget with all 196 tokens reaching the perceiver.** The Crane result is clear: more key/values to the perceiver is better. Pooling 196→49 before the perceiver would discard information. The `bridge_token_reduce=adaptive_pool` should apply to the perceiver output (already 49 queries), not the input.

Need to verify: does `adaptive_pool` reduce the perceiver's input tokens or its output? If it reduces input, we may need `bridge_token_reduce=none` for the nodynbudget SigLIP runs.

**Expected range:** 0.570–0.600. Wide range: SigLIP's language alignment could provide a large boost over DINOv2-small, or SigLIP's 196 tokens (vs DINOv2's 256) could offset the alignment advantage.

**Information value:** CRITICAL. This is the central experiment of Hardhat.

**Est. time:** 0.7–1.1h (depending on batch layout)

### Run 9: `siglip_[best_bridge_config]_nodynbudget`

**Conditional:** If any Tier 1 ablation (questiononly, d4, qdepth4) helped on DINOv2, apply the same delta to SigLIP. Tests whether bridge improvements transfer across VMs.

**Est. time:** 0.7–1.1h (if run)

### Tier 4 Subtotal: 1.9–2.7h (probes + 1-2 runs)

**Decision gate after Tier 4:**
- **If SigLIP-B > DINOv2-S frontier:** Language-aligned high-token VMs are the path forward. Build Eng-3 (DINOv2-B) to isolate alignment vs capacity. SigLIP becomes the new frontier VM. Consider SigLIP + 18k as the max-out run.
- **If SigLIP-B ≈ DINOv2-S:** Alignment helps per-token (Crane showed this) but DINOv2's extra 60 tokens compensate. Try DINOv2-B (256 tokens at 768-dim) — it has both more tokens AND more capacity.
- **If SigLIP-B < DINOv2-S despite 4x params:** Self-supervised spatial features are genuinely better for VQA through this bridge. The project's VM strategy should prioritize DINOv2 scaling (ViT-B, ViT-L) over CLIP-family models.

## Tier 5: Capacity-Matched Comparison (Requires Eng-3, Conditional)

**Only run if Tier 4 SigLIP-B result is ambiguous** (within ±0.01 of DINOv2-S frontier) or if SigLIP-B wins and we need to isolate why.

### Run 10: `dinov2b_attnqquery_nodynbudget_adapter_d3`

**Purpose:** DINOv2-B/14 is ~86M params with 256 tokens at 768-dim. Compared to SigLIP-B/16 (~86M, 196 tokens, 768-dim), this isolates pre-training objective (self-supervised vs language-aligned) at matched model capacity. The confound is token count (256 vs 196), but the Crane evidence suggests more tokens is better, so DINOv2-B has a structural advantage here. If SigLIP-B still wins despite fewer tokens, language alignment is a strong signal.

**Config:** Same as Run 8 but with DINOv2-B.

**Expected range:** 0.580–0.610. Larger model + more tokens should exceed DINOv2-small.

**Est. time:** 0.7–1.1h + perf probes

### Tier 5 Subtotal: 1.2–1.6h (if run)

## Tier 6: Final Frontier

### Run 11: `[best_vm]_[best_config]_18k`

**Purpose:** Max-out run. Stack all positive signals:
- Best VM from Tiers 4-5 (or DINOv2-S if new VMs disappoint)
- Best bridge config from Tier 1 (questiononly/d4/qdepth4 if any helped)
- 18k training (if Tier 2 showed continued slope)
- Caption-align pre-training (if Tier 3 showed positive signal)

This is the "how far can the current LM go" run.

**Est. time:** 1.3–2.2h (depending on VM and whether caption-align is included)

### Run 12: `[best_config]_seed2`

**Purpose:** Seed check of the new frontier. Non-negotiable.

**Est. time:** 0.7–1.1h

### Tier 6 Subtotal: 2.0–3.3h

## Comparison Policy

Same as Crane, unchanged:

- Effective batch size: 192
- Target step: 9000 (or 18000 for long runs)
- `--eval_every 1000`, `--eval_batches 100` (periodic)
- Final eval: full validation split (`--eval_fraction 1.0 --final_eval_batches 0`)
- Official scorer
- `--eval_use_kv_cache --eval_kv_cache_mode batched`
- `--precision bf16`

**Per-VM layouts (confirmed or pending probes):**

| VM | batch_size | grad_accum | eval_batch | Status |
|---|---:|---:|---:|---|
| DINOv2-S (256 tok, 384d, 22M) | 96 | 2 | 96 | Confirmed (Crane probes) |
| SigLIP-B (196 tok, 768d, 86M) | 192 | 1 | 96 | Confirmed (Hardhat probes, train=2.90 sps @ b192, eval=233 samples/s @ b96) |
| DINOv2-B (256 tok, 768d, 86M) | TBD | TBD | TBD | Needs perf probes (if Tier 5) |

## Common Flag Groups

**COMMON_ARGS** (all runs, same as Crane):
```
--precision bf16
--epochs 400
--max_steps 9000
--manual_max_steps
--log_every 20
--eval_every 1000
--eval_batches 100
--final_eval_batches 0
--eval_log_every 20
--eval_fraction 1.0
--ckpt_every 1000
--eval_scorer official
--final_sanity_count 0
--cuda_empty_cache_after_eval
--eval_use_kv_cache
--eval_kv_cache_mode batched
--vision_feature_source encoder
--num_visual_tokens 49
--bridge_token_reduce adaptive_pool
--bridge_add_2d_pos_emb
--bridge_num_heads 8
--bridge_type perceiver_resampler
--bridge_query_depth 3
--bridge_pre_mixer_type none
--bridge_question_conditioning
--bridge_question_context_mode prompt_only
--prefix_calibration
--prefix_calib_layernorm
--prefix_calib_bias
--prefix_calib_gate_init 1.0
--prefix_geom_mlp_ratio 0.5
--prefix_geom_token_mixer_layers 1
--prefix_norm_target_ratio 4.0
--prefix_norm_reg_weight 0.005
--prefix_batchvar_reg_weight 0.0002
--prefix_dropout 0.03
--freeze_mode bridge_plus_top_lm
--train_top_lm_layers 2
--lr 0.0002
--lr_schedule cosine
--lr_warmup_steps 600
--lr_min_ratio 0.15
--min_train_steps_per_s 1.0
--min_train_steps_window 100
```

**NODYNBUDGET_ATTNQQUERY_ADAPTER_ARGS** (Hardhat default — no token selection):
```
--bridge_query_bank_mode question_hidden_attn
--bridge_qquery_scale 1.0
--bridge_token_selector_type none
--bridge_token_select_k 0
--lm_visual_adapter_type cross_attn
--lm_visual_adapter_layers 3
--lm_visual_adapter_num_heads 8
--lm_visual_adapter_dropout 0.0
--lm_visual_adapter_gate_init 0.5
```

**DINOV2S_ARGS:**
```
--vision_model dinov2_small
--vision_checkpoint logs/hf_vision/facebook_dinov2_small
--vision_feature_mode auto
--batch_size 96
--grad_accum_steps 2
--eval_batch_size 96
```

**SIGLIP_ARGS** (confirmed via perf probes, b192a1 train + b96 eval):
```
--vision_model siglip_base
--vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224
--vision_feature_mode auto
--batch_size 192
--grad_accum_steps 1
--eval_batch_size 96
```

## Execution Schedule

### Phase A: Tier 1 + Eng-1 (Parallel)

Tier 1 runs need no new code. Start immediately while building Eng-1 (caption-align fix) and Eng-2 (SigLIP wrapper).

```
Run 1: seed2                    (0.7h)
Run 2: questiononly             (0.7h)
Run 3: d4 adapters              (0.7h)
Run 4: query_depth=4            (0.7h)
  → Tier 1 decision gate
Build Eng-1 (caption-align fix)
Build Eng-2 (SigLIP wrapper) + download model
```

Training hours: 2.8h. Running total: **2.8h**.

### Phase B: Tier 2 + SigLIP Perf Probes

```
Run 5: 18k training             (1.3h)
Run 6: best_config_18k          (1.3h, conditional)
SigLIP perf probes              (0.5h)
  → Tier 2 decision gate
```

Training hours: 1.8-3.1h. Running total: **4.6-5.9h**.

### Phase C: Tier 3 + Tier 4

```
Run 7: corrected caption-align  (0.9h)
Run 8: SigLIP-B nodynbudget     (0.7-1.1h)
Run 9: SigLIP-B best_config     (0.7-1.1h, conditional)
  → Tier 4 decision gate
```

Training hours: 2.3-3.1h. Running total: **6.9-9.0h**.

### Phase D: Tier 5 + Tier 6 (Conditional)

```
Run 10: DINOv2-B (conditional)  (1.2-1.6h)
Run 11: best_vm_best_config_18k (1.3-2.2h)
Run 12: seed2 of best           (0.7-1.1h)
  → Write Hardhat sweep report
```

Training hours: 3.2-4.9h. Running total: **10.1-13.9h**.

## Restart Safety

Same as Crane. The launcher uses skip-if-done, auto-resume from latest checkpoint, and low-throughput watchdog restart. See [34_crane_extended_sweep_plan_2026-03-14.md](34_crane_extended_sweep_plan_2026-03-14.md), "Restart Safety" section.

## Run Index

| # | Run | Tier | VM | Key delta | Est. h | Priority |
|---|---|---:|---|---|---:|---|
| 1 | seed2 (frontier) | 1 | DINOv2-S | seed=53 | 0.7 | NON-NEGOTIABLE |
| 2 | questiononly nodynbudget | 1 | DINOv2-S | question_only | 0.7 | HIGH |
| 3 | d4 nodynbudget | 1 | DINOv2-S | d4 adapters | 0.7 | MEDIUM-HIGH |
| 4 | qdepth4 nodynbudget | 1 | DINOv2-S | query_depth=4 | 0.7 | MEDIUM-HIGH |
| 5 | 18k nodynbudget | 2 | DINOv2-S | 18k steps | 1.3 | HIGH |
| 6 | best_config_18k | 2 | DINOv2-S | stack + 18k | 1.3 | CONDITIONAL |
| 7 | corrected caption-align | 3 | DINOv2-S | reset_schedule | 0.9 | MEDIUM |
| 8 | SigLIP-B nodynbudget | 4 | SigLIP-B | new VM | 0.7-1.1 | CRITICAL |
| 9 | SigLIP-B best_config | 4 | SigLIP-B | bridge transfer | 0.7-1.1 | CONDITIONAL |
| 10 | DINOv2-B nodynbudget | 5 | DINOv2-B | capacity match | 0.7-1.1 | CONDITIONAL |
| 11 | max-out run | 6 | best | everything | 1.3-2.2 | HIGH |
| 12 | seed2 (final) | 6 | best | seed=53 | 0.7-1.1 | NON-NEGOTIABLE |

**Maximum training hours: ~14h. Minimum (skip conditionals): ~8h.**

## What This Sweep Answers

| Question | Runs | Diagnostic |
|---|---|---|
| Is 0.5762 stable across seeds? | 1 | Seed variance |
| Does questiononly help on nodynbudget? | 2 vs frontier | Overall + `other` |
| Does adapter depth matter with rich prefix? | 3 vs frontier | Overall |
| Does perceiver depth matter at 256 tokens? | 4 vs frontier | Overall |
| Is there headroom beyond 9k steps? | 5, 6 | Learning curve shape |
| Does properly configured caption-align help? | 7 vs frontier | Step-matched comparison |
| Does language alignment + high tokens beat DINOv2-S? | 8 vs frontier | Overall + per-category |
| Can bridge improvements transfer across VMs? | 9 vs 8 | Delta consistency |
| Is alignment or capacity driving SigLIP? | 10 vs 8 | Capacity-matched comparison |
| How far can the current LM go? | 11 | Final frontier number |

The sweep answers whether the project is ready to graduate to a larger LM. If the best Hardhat run exceeds ~0.60, the bridge architecture is mature enough for LM scaling. If it plateaus below ~0.58, there's still bridge or VM work to do.
