# 34 Crane Extended Sweep Plan — "Max Out the LM" (2026-03-14)

## Ancestry

This plan supersedes doc 33 (original Crane plan). It incorporates direction from the research lead: replace the single large CLIP ViT-B/16 run with two smaller-footprint VMs (MobileCLIP-S0, DINOv2-small) that answer sharper questions about what matters in the vision encoder, add caption-align pre-training infrastructure, and test longer training.

Sources:
- `tasks/mm_bridge/docs/33_crane_sweep_plan_2026-03-14.md` (original Crane)
- `tasks/mm_bridge/docs/32a_plank_sweep_report_2026-03-14.md`
- `tasks/mm_bridge/docs/32b_plank_sweep_report_2026-03-14.md`
- Research lead input: "CRANE EXTENDED SWEEP PLAN — MAX OUT THE LM"
- Existing launcher pattern: `tasks/mm_bridge/scripts/launch_plank_sweep_v1.sh`

## Goal

Squeeze maximum VQA performance from the current ~46M-param LM before moving to advanced LM training. Answer the key remaining bridge questions:

1. Does language alignment in the VM matter at our scale?
2. Does attnqquery generalize beyond language-aligned VMs?
3. Does dynbudget help when it must actually filter?
4. Can caption-align pre-training substitute for VM-level language alignment?
5. Is longer training a free lunch?
6. How far can the current LM go?

## Budget

**Training time target: 30–50 hours.**

This machine doubles as a gaming PC. The sweep must tolerate arbitrary mid-run kills and restarts. Checkpoint-based resume at 1000-step granularity handles this (see Restart Safety below).

Estimated per-run times (inclusive of periodic evals, final eval, overhead):

| Run type | Est. hours |
|---|---:|
| MobileViT 9k (96×2) | ~3.5 |
| MobileViT 18k (96×2) | ~7 |
| MobileCLIP 9k (96×2, 49 tokens) | ~3.5 |
| DINOv2 9k (96×2, 256 tokens) | ~3.5 |
| DINOv2 nodynbudget 9k (96×2) | ~3.5 |
| Caption pre-train 3k steps | ~0.12 |
| Two-stage (3k pre + 9k VQA) | base + 0.12 |

These are wall-clock estimates based on Plank timing data (~3.5h per MobileViT 9k run from 32b). DINOv2 estimated 40% slower due to heavier perceiver cross-attention over 256 tokens and lower batch size.

## Engineering Prerequisites

All three engineering tasks must be built and smoke-tested before their respective tier runs begin. Eng-1 and Eng-2 can be built in parallel with Tier 1 training.

### Eng-1: MobileCLIP-S0 Integration

**Model:** MobileCLIP-S0 (~11.4M params), trained with CLIP objective on DataCompDR-1B (~1.28B image-text pairs). Language-aligned vision features at mobile scale.

**Output shape (to confirm):** Likely 7×7 = 49 tokens at 512-dim. This must be verified from the actual model config after download. If different, document the actual shape and adjust `--num_visual_tokens` accordingly.

**Availability:** MobileCLIP-S0 may not be directly on HuggingFace transformers. Check:
1. `apple/mobileclip-s0` on HuggingFace (if available, use `CLIPVisionModel`)
2. `timm` library (`timm.create_model('mobileclip_s0')`)
3. Apple's `ml-mobileclip` GitHub repo (direct weight loading)

The wrapper must follow the `HFMobileViTSmallBackbone` pattern in `models/hf_vision.py`:
- `__init__(model_dir, *, device)` — load from local directory
- `_prepare_inputs(images)` — normalize (ImageNet mean/std for MobileCLIP), resize to 224×224, RGB order (no BGR flip)
- `forward(images)` → `[B, Nv, Dv]` float32 token sequence
- `_encoder(images)` → delegates to `forward()`
- Freeze all params (`requires_grad_(False)` or rely on `freeze_mode` flag)

**Code changes:**
1. Add wrapper class to `models/hf_vision.py` (~60 lines)
2. Add `"mobileclip_s0"` to `--vision_model` choices at `train/mm.py:2405`
3. Add elif branch in `build_vision_model_from_args()` at `train/mm.py:338`
4. Download model to `logs/hf_vision/apple_mobileclip_s0/`

**Smoke test:** 100-step training run, verify:
- Output shape matches expectation
- No gradients flow into VM
- Loss decreases
- Memory fits at estimated batch size

**Memory estimate (MEASURED):** 49 tokens at 1024-dim, 11.4M params. All batch layouts from b192a1 through b32a6 fit without OOM. Best layout: **b96a2** (train=3.53 sps, eval=2.50 sps). Similar throughput to MobileViT.

### Eng-2: DINOv2-small Integration

**Model:** DINOv2 ViT-S/14 (~22M params), self-supervised on LVD-142M. Strong spatial features, NO language alignment.

**Output shape:** 16×16 = 256 patch tokens at 384-dim on 224×224 input. The HF model (`facebook/dinov2-small`) returns `last_hidden_state` with shape `[B, 257, 384]` (256 patches + 1 CLS token). **Strip the CLS token** — return only the 256 patch tokens.

**Critical: use 224×224 input, not DINOv2's native 518×518.** DINOv2 at 518px gives (518/14)² = 1369 tokens, far too many for our bridge and memory budget. At 224px we get 256 tokens — already 5× more than MobileViT.

**Availability:** `facebook/dinov2-small` on HuggingFace. Straightforward `Dinov2Model.from_pretrained()` or `AutoModel.from_pretrained()`.

**Code changes:** Same pattern as Eng-1.
1. Add `HFDINOv2SmallBackbone` to `models/hf_vision.py` (~60 lines)
2. Add `"dinov2_small"` to `--vision_model` choices
3. Add elif branch in `build_vision_model_from_args()`
4. Download model to `logs/hf_vision/facebook_dinov2_small/`

Key wrapper differences from MobileViT:
- ImageNet normalization (same constants)
- RGB order (no BGR flip)
- Target size 224×224 (not 518)
- Strip CLS token: `hidden = outputs.last_hidden_state[:, 1:, :]` (patch tokens only)

**Token handling in bridge:** The bridge handles arbitrary input token counts. The perceiver cross-attends its 49 learned queries over whatever key/value count it receives. With 256 input tokens, the perceiver cross-attention matrix is 49×256 per head per layer — 5× more than MobileViT's 49×49 but still modest.

**dynbudget at 256 tokens:** With `--bridge_token_selector_type qadaptive --bridge_token_select_k 64`, the selector chooses ~64 of 256 tokens. This is the first genuine filtering regime in the project. The selector code (`models/bridge.py:318`) already handles arbitrary input counts: `max_k = min(max(1, int(self._selector_k)), n)`.

**Memory estimate:** 256 tokens at 384-dim with 22M VM params. The main cost increase is:
- VM forward: 22M vs 5.6M params (~4× more)
- Perceiver cross-attention: 49×256 vs 49×49 key/values (~5× more per layer)
- Token selector: operates on 256 tokens

**Memory estimate (MEASURED):** All batch layouts from b192a1 through b32a6 fit without OOM. Best layout: **b96a2** for both dynbudget (train=4.31 sps, eval=13.19 sps) and nodynbudget (train=4.20 sps, eval=2.45 sps). DINOv2+dynbudget eval is 5× faster because the selector reduces 256→64 tokens before perceiver cross-attention. Surprisingly, DINOv2 is *faster* than MobileCLIP at the same batch layout due to smaller feature dim (384 vs 1024).

**Smoke test (PASSED):**
- Output shape: `[B, 256, 384]` after CLS stripping -- confirmed
- dynbudget with cap=64: selector operates on 256 tokens, selects 64 -- confirmed
- Memory: no OOM at any batch layout through b192a1 -- confirmed
- `--num_visual_tokens 49`: perceiver compresses 256→49 -- confirmed

### Eng-3: Caption-Align Pre-Training Infrastructure

**Purpose:** Two-stage training pipeline. Stage 1: align bridge output with LM caption encoding via cosine similarity. Stage 2: standard VQA fine-tuning from pre-trained bridge.

**Components:**

1. **COCO caption dataloader** (~60 lines, following `train/vqa_data.py`):
   - Load `captions_train2014.json` (~83k images, ~414k captions)
   - Per epoch: one random caption per image
   - Return `(image_tensor, caption_text)` batches
   - Images already available (VQAv2 uses COCO images)
   - Annotation file download: `captions_train2014.json` (~20MB)

2. **Caption encoding** (~30 lines):
   - Tokenize caption with existing BPE tokenizer
   - Forward through frozen LM
   - Mean-pool hidden states over caption tokens
   - L2-normalize → target vector `t`
   - Option to precompute and cache if I/O bound

3. **Pre-training loop** (~80 lines):
   - Forward: image → frozen VM → bridge → mean-pool perceiver output → L2-normalize → vector `v`
   - Loss: `1 - cosine_similarity(v, t)` averaged over batch
   - Optimizer: same optimizer config (Adam, cosine schedule) but only bridge params
   - Schedule: ~3k steps (tune based on convergence)
   - **Collapse monitoring:** Log `std(v)` per batch. If `std < 0.01` for 100 consecutive steps, halt and flag. This catches the trivial solution where the bridge produces near-constant output.

4. **Two-stage launcher** (~30 lines in sweep script):
   - Stage 1: run pre-training, save bridge checkpoint
   - Stage 2: load bridge checkpoint into standard VQA training (via `--checkpoint` pointing to the pre-trained step)

**Total new code:** ~200 lines.

**Data dependency:** Download `captions_train2014.json` from COCO website.

**Smoke test (PASSED on all 3 VMs):**
- MobileViT: loss 1.03 → 0.47 in 20 steps, cos_sim 0.53, bridge_std 0.044
- MobileCLIP: loss 1.01 → 0.78 in 10 steps, cos_sim 0.22, bridge_std 0.044
- DINOv2: loss 1.00 → 0.84 in 10 steps, cos_sim 0.16, bridge_std 0.044
- No collapse, bridge weights diverge, checkpoints loadable by VQA training
- Throughput: ~7.5 sps at b96 — 3000 steps takes ~7 minutes

**Implementation:**
- `train/caption_pretrain.py` — pre-training loop (~210 lines)
- `train/caption_data.py` — COCO 2014 caption dataset (~125 lines)
- `runcapalign.sh` — Docker launcher wrapper

## Performance Probe Results (2026-03-14)

All probes completed without OOM at any batch layout. Best layout for all variants: **b96a2** (effective batch 192).

| Variant | train sps | eval sps | Notes |
|---|---:|---:|---|
| MobileCLIP b96a2 | 3.53 | 2.50 | Similar to MobileViT |
| DINOv2+dynbudget b96a2 | 4.31 | 13.19 | Eval 5x fast (64/256 tokens after selector) |
| DINOv2 nodynbudget b96a2 | 4.20 | 2.45 | All 256 tokens to perceiver |

DINOv2 is *faster* than MobileCLIP despite having 2x params (22M vs 11.4M) — the smaller feature dim (384 vs 1024) dominates. This means DINOv2 runs are NOT slower than MobileCLIP runs, updating the time estimates above.

## Tier 1: MobileViT Completion (Flag Changes Only)

These runs complete the MobileViT story. No new code required. Can run immediately.

### Run 1: `mobilevit_questiononly_attnqquery_dynbudget_adapter_d3_cap64`

**Hypothesis:** Stacking question-only pooling with attnqquery on MobileViT improves the frontier.

**Modeling reasoning:** The frontier run uses `--bridge_question_context_mode prompt_only`. The `question_only` mode restricts the attention pool to question tokens, removing prompt-template tokens ("Question:", "Answer:") that dilute the query signal. On the original VM, `questiononly` gained +0.0046 (best Stage B result). On MobileViT, where richer features reward more precise queries, the sharpening should have at least as much value and possibly more — attnqquery's attention head can spend all capacity differentiating between question-relevant aspects rather than wasting representational budget on learning to ignore template tokens.

**Config delta from frontier:** `--bridge_question_context_mode question_only`

**Expected range:** 0.520–0.530. Information value: **HIGH** (sets MobileViT ceiling).

**Est. time:** 3.5h

### Run 2: `mobilevit_attnqquery_dynbudget_adapter_d4_cap64`

**Hypothesis:** Deeper adapters (d4) extract more value from MobileViT features.

**Modeling reasoning:** d2→d3 gave +0.0009 on the old VM. With MobileViT's richer features, more frequent LM re-access to visual tokens during generation should be more productive — each adapter layer can extract different aspects of the 640-dim evidence at different stages of answer formation. d4 places adapters at roughly every 2.5 layers in the 12-layer LM, shrinking the "visual memory gap" between access points.

**Config delta from frontier:** `--lm_visual_adapter_layers 4`

**Expected range:** 0.522–0.532. Information value: **MEDIUM** (if d4 helps here, run d4 on new VMs too).

**Est. time:** 3.5h

### Run 3: `mobilevit_attnqquery_dynbudget_adapter_d3_cap64_18k`

**Hypothesis:** MobileViT attnqquery curves were still rising at 9k. Doubling training budget yields free performance.

**Modeling reasoning:** The periodic eval curves in 32b show linear improvement with no plateau at 9k steps for all MobileViT runs. The 8k→9k periodic delta for attnqquery was +0.003, suggesting substantial remaining slope. At 18k, we should see continued gains at least through 12k–15k before diminishing returns.

This run is strategically important: if 18k barely improves over 9k (delta < 0.003), we keep all subsequent runs at 9k and save 3.5h per run across the entire sweep. If 18k helps substantially (delta > 0.008), the final stacking runs in Tier 5 should use 18k.

**Config delta from frontier:** `--max_steps 18000`

**Expected range:** 0.530–0.545. Information value: **HIGH** (informs step count for all subsequent runs).

**Est. time:** 7h

### Run 19: `mobilevit_attnqquery_dynbudget_adapter_d3_cap64_seed2`

**Hypothesis:** The frontier score (0.5240) needs a second seed before being treated as a settled number.

**Modeling reasoning:** Plank's seed check on lmmeanqquery showed 0.0051 variance between seeds. The attnqquery frontier has never been seed-checked. The attnqquery-vs-lmmeanqquery gap is 0.0059 — within range of seed noise. If seed2 drops below lmmeanqquery seed1 (0.5181), the attnqquery dominance on MobileViT is not yet settled, and all Tier 2+ runs that build on attnqquery need to be reconsidered.

**Config delta from frontier:** `--seed 53`

**Expected range:** 0.518–0.528. Information value: **HIGH** (frontier stability).

**Est. time:** 3.5h

### Tier 1 Subtotal: ~17.5h

**Decision gate after Tier 1:**
- If Run 3 (18k) gains < 0.003 over 9k: keep all subsequent runs at 9k.
- If Run 3 gains > 0.008: consider 18k for the final Tier 5 stacking run.
- If Run 19 drops below 0.518: re-evaluate attnqquery as the default; consider lmmeanqquery for Tier 2 VM comparison.
- If Run 1 (questiononly) beats the frontier: it becomes the new base config for Tier 5.
- If Run 2 (d4) beats the frontier: d4 becomes the adapter depth for Tier 5.

## Tier 2: New VM Baselines (Requires Eng-1, Eng-2)

The central experimental battery. Direct comparison between MobileViT, MobileCLIP, and DINOv2 on the same bridge config.

### Run 4: `mobileclip_attnqquery_dynbudget_adapter_d3_cap64`

**Hypothesis:** Language-aligned features from MobileCLIP match or beat MobileViT, especially on `other` where language grounding helps with compositional questions.

**Modeling reasoning:** MobileCLIP-S0 is trained via CLIP objective on DataCompDR-1B (1.28B image-text pairs vs MobileViT's 1M ImageNet images). The CLIP objective forces visual features to be predictive of text — the feature space encodes "what in this image can be described in language." For VQA, where the bridge must translate visual evidence into language-compatible representations, pre-aligned features should provide a stronger starting point.

The key comparison is MobileCLIP (~11.4M params, language-aligned, DataCompDR-1B) vs MobileViT (~5.6M params, no alignment, ImageNet). If MobileCLIP wins, language alignment in the VM is the dominant factor. If MobileViT wins despite smaller pre-training data, raw feature quality and architecture matter more than alignment at this model scale.

If MobileCLIP-S0 produces 49 tokens (same as MobileViT), this is the cleanest possible comparison: same token count, same bridge, only the VM and its pre-training differ.

**Config:** Same as frontier but with MobileCLIP VM. Batch/accum TBD from Eng-1 memory profile.

**Expected range:** 0.520–0.555. Wide range reflects genuine uncertainty about how much DataCompDR-1B pre-training helps vs ImageNet at ~11M params.

**Information value: CRITICAL.** Core "does language alignment matter" experiment.

**Est. time:** ~3.5h (if 49 tokens, similar to MobileViT)

### Run 5: `dinov2s_attnqquery_dynbudget_adapter_d3_cap64`

**Hypothesis:** DINOv2's strong spatial features + 256 tokens produce a large jump, but lack of language alignment may leave `other` behind MobileCLIP.

**Modeling reasoning:** DINOv2-small has more params (22M vs 5.6M/11.4M), more tokens (256 vs 49), and stronger pre-training data (LVD-142M, ~142M curated images). But it has zero language alignment — the self-supervised objective (DINO + iBOT) optimizes for visual self-consistency, not text prediction.

This creates a clean contrast with MobileCLIP: DINOv2 has more spatial information (256 tokens, 22M params) but no language prior. The `other` category is the sharpest diagnostic: MobileCLIP should win on `other` (language-grounded compositional questions) while DINOv2 should win on `number` (fine-grained spatial counting benefits from 256 tokens).

With 256 tokens and `--bridge_token_select_k 64`, dynbudget performs genuine filtering for the first time: selecting ~64 of 256 tokens based on question relevance. The perceiver then cross-attends its 49 queries over these 64 selected tokens.

**Config:** Batch/accum TBD from Eng-2 profile. `--num_visual_tokens 49` (same perceiver output as MobileViT for fair LM-side comparison). `--bridge_token_select_k 64 --bridge_token_select_k_min 24`.

**Expected range:** 0.530–0.570.

**Information value: CRITICAL.** Paired with Run 4, this answers the central question.

**Est. time:** ~5h

### Run 6: `dinov2s_lmmeanqquery_dynbudget_adapter_d3_cap64`

**Hypothesis:** The attnqquery advantage may be specific to language-aligned VMs. Without language alignment, the relationship between attnqquery and lmmeanqquery might revert.

**Modeling reasoning:** Under MobileViT (ImageNet-only, no language alignment), attnqquery reversed its Nail-era loss to lmmeanqquery. The hypothesis from 32b was that richer features amplify the attention-derived query advantage. But there's an alternative explanation: attnqquery works best when visual features and LM attention patterns share some structural compatibility (even if indirect, through ImageNet → similar object categories → attention pattern regularity).

DINOv2 features are self-supervised, with very different internal structure from MobileViT's classification features. If attnqquery still wins on DINOv2, the mechanism is genuinely about spatial selectivity in the query — the attention head is learning to focus on the right question tokens regardless of visual feature style. If lmmeanqquery wins, attnqquery's advantage is VM-specific and bridge design should be re-evaluated per VM.

**Config:** Same as Run 5 but with `--bridge_query_bank_mode question_hidden_mean`.

**Expected range:** 0.525–0.560.

**Information value: HIGH.** Establishes whether query mechanism choice transfers across VM families.

**Est. time:** ~5h

### Tier 2 Subtotal: ~13.5h

**Decision gate after Tier 2 — write VM comparison analysis doc before proceeding:**

This is the most important decision point in the sweep. The analysis should cover:

1. **VM ranking:** MobileCLIP vs DINOv2 vs MobileViT overall and per answer type
2. **Language alignment value:** Run 4 vs Run 5 (same bridge, different VM pre-training philosophy)
3. **attnqquery universality:** Run 5 vs Run 6 (attnqquery on non-language-aligned VM)
4. **Token count value:** Do DINOv2's 256 tokens (filtered to ~64 by dynbudget) outperform MobileCLIP's ~49 tokens?
5. **Winner selection:** Choose the VM for Tier 3–5

**Decision rules:**
- If DINOv2 wins overall: proceed to Tier 3 (dynbudget sweep) + Tier 4 (caption-align on DINOv2).
- If MobileCLIP wins: skip Tier 3 (dynbudget sweep irrelevant with 49 tokens), proceed to Tier 4 (caption-align on MobileCLIP — test whether additional alignment helps).
- If MobileViT still wins: both new VMs failed to deliver. Re-evaluate — likely move to caption-align on MobileViT as the next lever.
- If Run 5 ≈ Run 6 (attnqquery ≈ lmmeanqquery on DINOv2): the query mechanism choice is VM-dependent. Use whichever won for subsequent DINOv2 runs.

## Tier 3: DINOv2 Dynbudget Sweep (Requires Eng-2, Tier 2 Results)

**Only run if DINOv2 is the VM winner or co-winner from Tier 2.**

DINOv2's 256 tokens are the first regime where dynbudget must genuinely select. These three runs bracket the token budget design space.

### Run 7: `dinov2s_attnqquery_nodynbudget_adapter_d3_cap64`

**Hypothesis:** Ablating dynbudget tells us whether question-conditioned filtering of 256→64 tokens helps or hurts.

With 256 input tokens and no filtering, all 256 go to the perceiver as key/values. Cross-attention cost: 49 queries × 256 keys per head per layer. This is 4× more work than the filtered case (49×64) and may hit memory limits — if so, reduce batch size further.

If dynbudget helps (Run 5 > Run 7): the selector is genuinely identifying question-relevant tokens from a large pool. Keep dynbudget for all high-token-count VMs.

If dynbudget hurts (Run 7 > Run 5): the filtering is too aggressive and drops good tokens. The perceiver cross-attention is a better mechanism for soft selection than hard top-k filtering.

**Config delta from Run 5:** `--bridge_token_selector_type none --bridge_token_select_k 0`

**Expected range:** Hard to call. ±0.015 from Run 5.

**Information value: HIGH.** First real dynbudget signal in the project.

**Est. time:** ~5h (potentially slower due to 256 tokens in perceiver)

### Run 8: `dinov2s_attnqquery_dynbudget_adapter_d3_cap128`

**Hypothesis:** cap64 may be too aggressive for 256 input tokens. cap128 passes ~half, a gentler filtering regime.

**Config delta from Run 5:** `--bridge_token_select_k 128`

**Expected range:** Within ±0.005 of Run 5.

**Information value: MEDIUM.** Brackets the budget design space.

**Est. time:** ~5h

### Run 9: `dinov2s_attnqquery_dynbudget_adapter_d3_cap32`

**Hypothesis:** Aggressive filtering to 32 of 256 tokens. If this doesn't collapse, the selector is genuinely identifying the most informative tokens — and we get a throughput win.

**Config delta from Run 5:** `--bridge_token_select_k 32 --bridge_token_select_k_min 12`

**Expected range:** 0.510–0.550.

**Information value: MEDIUM-HIGH.** If cap32 ≈ cap64, we get ~2× faster perceiver cross-attention for free in all future scaling.

**Est. time:** ~5h

### Tier 3 Subtotal: ~15h

**Decision gate after Tier 3:** Select the optimal cap value for Tier 5 stacking.

## Tier 4: Caption-Align Pre-Training (Requires Eng-3, Benefits from Tier 2)

### Run 10: `mobilevit_captionalign_attnqquery_dynbudget_adapter_d3_cap64`

**Hypothesis:** Caption-align pre-training on COCO gives the bridge a better starting point before VQA fine-tuning.

**Modeling reasoning:** MobileViT has no language alignment in the VM. Caption-align pre-training provides alignment at the bridge level: the bridge learns to produce representations that match the LM's encoding of image descriptions before seeing any VQA supervision. This is the cheapest test of whether bridge-level alignment helps.

**Config:** Stage 1: 3k steps caption-align. Stage 2: standard 9k VQA loading pre-trained bridge.

**Expected range:** 0.530–0.545.

**Information value: HIGH.** If it helps on MobileViT (no VM alignment), it should help even more on DINOv2.

**Est. time:** ~5h (1.5h pre-train + 3.5h VQA)

### Run 11: `dinov2s_captionalign_attnqquery_dynbudget_adapter_d3_cap64`

**Hypothesis:** DINOv2 features are spatially strong but not language-aligned. Caption-align pre-training provides the language alignment that the VM lacks — at the bridge level instead of the VM level.

**Modeling reasoning:** This is the "can we get language alignment cheaply through the bridge instead of needing it in the VM" experiment. If Run 11 beats Run 5 by a meaningful margin (>0.005), caption-align is providing genuine alignment value. If it also closes the gap to MobileCLIP (Run 4), the result would say: "language alignment matters, but you can inject it at the bridge layer rather than requiring it in the VM." That would be a strong finding for the project — it means the VM choice is primarily about spatial feature quality, with language alignment handled by the bridge.

**Config:** Stage 1: 3k steps caption-align on DINOv2 features. Stage 2: standard 9k VQA.

**Expected range:** 0.540–0.580.

**Information value: HIGH.**

**Est. time:** ~6.5h (1.5h pre-train + 5h VQA)

### Run 12: `mobileclip_captionalign_attnqquery_dynbudget_adapter_d3_cap64`

**Only run if MobileCLIP won Tier 2 and compute budget remains.**

**Hypothesis:** MobileCLIP already has CLIP alignment. Does additional caption-align help or is it redundant?

**Expected range:** Within ±0.005 of Run 4 (likely redundant).

**Information value: MEDIUM.** Likely cut.

**Est. time:** ~5h

### Tier 4 Subtotal: ~11.5–16.5h

## Tier 5: Stacking Winners (After Tiers 1–4)

These runs combine the best signals. The configs are NOT pre-committed — the agent must select based on what actually won.

### Run 13: `[best_vm]_[best_qctx]_attnqquery_dynbudget_adapter_[d3|d4]_cap[best]`

The "everything we know works, all at once" run. Stack:
- Best VM from Tier 2
- Best question context mode (question_only if Run 1 helped, prompt_only otherwise)
- Best adapter depth (d4 if Run 2 helped, d3 otherwise)
- Best dynbudget cap from Tier 3 (or default cap64 if DINOv2 lost)
- Caption-align if Tier 4 showed positive signal

**Est. time:** 3.5–5h depending on VM.

### Run 14: `[best_config]_18k`

Same as Run 13 but trained for 18k steps. **Only run if Run 3 showed that longer training helps (delta > 0.005).**

**Est. time:** 7–10h depending on VM.

### Run 15: `[best_config]_[captionalign]_18k`

The full stack: best VM + caption-align + all positive signals + longest training. **Only run if both caption-align (Tier 4) and 18k (Run 3) showed positive signal.** This is the "max out the LM" run.

**Est. time:** 8.5–11.5h depending on VM.

### Run 20: `[best_config]_seed2`

Second seed of the final frontier run. **Non-negotiable before declaring a final number.**

**Est. time:** 3.5–5h depending on VM.

### Tier 5 Subtotal: variable, ~10–20h

## Tier 6: Diagnostic / Curiosity (Low Priority, Fill Compute Gaps)

| Run | Purpose | VM | Est. time |
|---|---|---|---:|
| 16 | questiononly on DINOv2 | DINOv2 | ~5h |
| 17 | d4 adapters on MobileCLIP | MobileCLIP | ~3.5h |
| 18 | d5 adapters on DINOv2 | DINOv2 | ~5h |

These run only if compute gaps exist. None are required for the sweep conclusions.

## Execution Schedule

The budget constraint (30–50h training) means not all runs execute. The schedule is designed as a priority-ordered queue with decision gates that may cut later runs.

### Phase A: Engineering + Tier 1 (Parallel)

**Wall clock: ~3 days at ~6h training/day**

Build Eng-1 and Eng-2 while Tier 1 runs execute sequentially:

```
[Day 1]  Build Eng-1 (MobileCLIP) → smoke test
         Run 19: seed2               (3.5h training)
         Run 1: questiononly          (3.5h training)

[Day 2]  Build Eng-2 (DINOv2) → smoke test
         Run 2: adapter d4           (3.5h training)
         Run 3: 18k training         (7h training — may span gaming breaks)

[Day 3]  Build Eng-3 (caption-align) → smoke test
```

Training hours: 17.5h. Running total: **17.5h**.

### Phase B: Tier 2 (New VM Comparison)

**Wall clock: ~2 days**

```
         Run 4: mobileclip_attnqquery  (3.5h)
         Run 5: dinov2s_attnqquery     (5h)
         Run 6: dinov2s_lmmeanqquery   (5h)

         → Write VM comparison analysis doc
         → Select VM winner and decide Tier 3 vs Tier 4 priority
```

Training hours: 13.5h. Running total: **31h**.

At this point we're at 31h — within the 30–50h window. The remaining budget is 0–19h.

### Phase C: Selective Deep-Dives (Budget-Dependent)

Based on Tier 2 results, pick the highest-value runs from Tiers 3–5:

**If DINOv2 wins (15–19h remaining):**
```
         Run 7: dinov2s_nodynbudget    (5h)     — dynbudget ablation
         Run 10: mobilevit_captionalign (5h)    — caption-align baseline
         Run 13: best_vm_stacked       (5h)     — stacking run
```

**If MobileCLIP wins (10–15h remaining):**
```
         Run 10: mobilevit_captionalign (5h)    — caption-align baseline
         Run 13: best_vm_stacked       (3.5h)   — stacking run
         Run 20: best_config_seed2     (3.5h)   — seed check
```

**If MobileViT still wins (5–10h remaining):**
```
         Run 10: mobilevit_captionalign (5h)    — only lever left
         Run 20: best_config_seed2     (3.5h)   — seed check
```

### Phase D: Final Frontier

```
         Run 20: seed2 of best config (if not already run)
         → Write Crane sweep report
```

**Maximum training hours across all phases: ~50h.**
**Minimum (if MobileViT wins Tier 2): ~37h.**

## Restart Safety

This sweep is designed for arbitrary mid-run interruption. The user can kill the Docker container at any time to game, then restart the launcher.

### How It Works

The existing launcher infrastructure provides full restart safety:

1. **Checkpoints every 1000 steps.** At ~3.0 steps/s, this is one checkpoint every ~5.5 minutes. Maximum lost work on kill: 5.5 minutes.

2. **Skip-if-done.** `has_completed_eval()` checks for `final_eval` tag in the answers JSONL. Completed runs are skipped instantly on restart.

3. **Auto-resume.** `latest_ckpt_step()` finds the highest checkpoint. The launcher passes it as `--checkpoint <step>` to resume training from that point. Optimizer state, RNG state, epoch, and batch position are all restored.

4. **Eval-only mode.** If training is complete (step_9000.tar exists) but final eval hasn't run, the launcher enters `--eval_only` mode — cheaper than re-training.

5. **Low-throughput watchdog.** `--min_train_steps_per_s` detects when training is slow (e.g., after a GPU context switch from gaming). On exit code 86, the launcher auto-restarts from the latest checkpoint, up to `MAX_LOW_SPS_RESTARTS=8` times.

### User Workflow

```bash
# Start the sweep (runs until complete or killed)
bash tasks/mm_bridge/scripts/launch_crane_sweep_v1.sh

# Want to game? Just kill it:
# Ctrl+C, or kill the Docker container, or close the terminal

# Done gaming? Re-run the same command:
bash tasks/mm_bridge/scripts/launch_crane_sweep_v1.sh
# → automatically skips completed runs, resumes partial runs
```

### Skip Controls

The launcher script exposes environment variables for selective execution:

```bash
SKIP_TIER1=1          # skip all MobileViT completion runs
SKIP_TIER2=1          # skip new VM runs
SKIP_TIER3=1          # skip dynbudget sweep
SKIP_TIER4=1          # skip caption-align runs
SKIP_TIER5=1          # skip stacking runs
SKIP_TIER6=1          # skip diagnostics
```

Individual runs can also be skipped by name:
```bash
SKIP_RUN3_18K=1       # skip the 18k run to save 7h
```

## Comparison Policy

All runs follow the standing comparison policy from `MM_BRIDGE_GLOBAL_TASK_CONTEXT.md`:

- Effective batch size: 192
- Target step: 9000 (or 18000 for explicitly long runs)
- `--eval_every 1000`
- `--eval_batches 100` (periodic)
- Final eval: full validation split (`--eval_fraction 1.0 --final_eval_batches 0`)
- Official scorer
- `--eval_use_kv_cache --eval_kv_cache_mode batched`
- `--precision bf16`

**Per-VM layouts to maintain effective batch 192:**

| VM | batch_size | grad_accum_steps | eval_batch_size |
|---|---:|---:|---:|
| MobileViT (49 tokens, 640-dim) | 96 | 2 | 96 |
| MobileCLIP (49 tokens, 1024-dim) | 96 | 2 | 96 |
| DINOv2 (256 tokens, 384-dim) | 96 | 2 | 96 |

All batch layouts confirmed via perf probes (2026-03-14). All VMs fit at b96a2 without OOM.

## Common Flag Groups

For reference and launcher scripting. These are the flag groups shared across runs.

**COMMON_ARGS** (all runs):
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

**DYN_ADAPTER_ARGS** (adapter + dynbudget defaults):
```
--bridge_token_selector_type qadaptive
--bridge_token_select_k 64
--bridge_token_select_k_min 24
--lm_visual_adapter_type cross_attn
--lm_visual_adapter_layers 3
--lm_visual_adapter_num_heads 8
--lm_visual_adapter_dropout 0.0
--lm_visual_adapter_gate_init 0.5
```

**ATTNQQUERY_ARGS** (attnqquery bridge defaults):
```
--bridge_query_bank_mode question_hidden_attn
--bridge_qquery_scale 1.0
```

**MOBILEVIT_ARGS:**
```
--vision_model mobilevit_hf
--vision_checkpoint logs/hf_vision/apple_mobilevit_small
--vision_feature_mode auto
--num_workers 2
--prefetch_factor 1
--no-pin_memory
```

**MOBILECLIP_ARGS** (confirmed after Eng-1, b96a2):
```
--vision_model mobileclip_s0
--vision_checkpoint logs/hf_vision/apple_mobileclip_s0
--vision_feature_mode auto
--batch_size 96
--grad_accum_steps 2
--eval_batch_size 96
```

**DINOV2_ARGS** (confirmed after Eng-2, b96a2):
```
--vision_model dinov2_small
--vision_checkpoint logs/hf_vision/facebook_dinov2_small
--vision_feature_mode auto
--batch_size 96
--grad_accum_steps 2
--eval_batch_size 96
```

## What This Sweep Answers

If all core runs complete (Tiers 1–2 plus selective Tiers 3–5), Crane answers:

| Question | Runs | Diagnostic |
|---|---|---|
| Does language alignment in the VM matter? | 4 vs 5 | `other` category split |
| Does attnqquery generalize across VMs? | 5 vs 6 | Overall + `other` |
| Does dynbudget help when it filters? | 5 vs 7 | Overall (first real signal) |
| What's the right token budget? | 7 vs 8 vs 9 | Overall + throughput |
| Can caption-align substitute for VM alignment? | 11 vs 5, 10 vs 4 | `other` category |
| Is longer training free performance? | 3 vs frontier | Learning curve shape |
| MobileViT ceiling? | Runs 1, 2, 19 | Stacking + seed |
| How far can the current LM go? | Run 15 | Final frontier number |

The sweep establishes whether the bridge research is ready to graduate to a larger-LM regime. If the best Crane run exceeds ~0.57, the bridge architecture is strong enough that the next step is an LM upgrade. If it plateaus below ~0.54, there's still bridge-level work to do before scaling the LM.

## Run Index

| # | Run | Tier | VM | Key delta | Est. hours | Priority |
|---|---|---|---|---|---:|---|
| 1 | questiononly_attnqquery | 1 | MobileViT | question_only | 3.5 | HIGH |
| 2 | adapter_d4 | 1 | MobileViT | d4 adapters | 3.5 | MEDIUM |
| 3 | 18k | 1 | MobileViT | 18k steps | 7.0 | HIGH |
| 4 | mobileclip_attnqquery | 2 | MobileCLIP | new VM | 3.5 | CRITICAL |
| 5 | dinov2s_attnqquery | 2 | DINOv2 | new VM | 5.0 | CRITICAL |
| 6 | dinov2s_lmmeanqquery | 2 | DINOv2 | lmmeanqquery | 5.0 | HIGH |
| 7 | dinov2s_nodynbudget | 3 | DINOv2 | no filtering | 5.0 | HIGH |
| 8 | dinov2s_cap128 | 3 | DINOv2 | cap128 | 5.0 | MEDIUM |
| 9 | dinov2s_cap32 | 3 | DINOv2 | cap32 | 5.0 | MEDIUM-HIGH |
| 10 | mobilevit_captionalign | 4 | MobileViT | caption-align | 5.0 | HIGH |
| 11 | dinov2s_captionalign | 4 | DINOv2 | caption-align | 6.5 | HIGH |
| 12 | mobileclip_captionalign | 4 | MobileCLIP | caption-align | 5.0 | LOW (cut) |
| 13 | stacked winner | 5 | best | all positives | 3.5–5.0 | HIGH |
| 14 | stacked_18k | 5 | best | +18k | 7–10 | CONDITIONAL |
| 15 | full_stack_18k | 5 | best | everything | 8.5–11.5 | CONDITIONAL |
| 16 | dinov2s_questiononly | 6 | DINOv2 | questiononly | 5.0 | LOW |
| 17 | mobileclip_d4 | 6 | MobileCLIP | d4 adapters | 3.5 | LOW |
| 18 | dinov2s_d5 | 6 | DINOv2 | d5 adapters | 5.0 | LOW |
| 19 | seed2 (frontier) | 1 | MobileViT | seed=53 | 3.5 | HIGH |
| 20 | seed2 (final) | 5 | best | seed=53 | 3.5–5.0 | NON-NEGOTIABLE |
