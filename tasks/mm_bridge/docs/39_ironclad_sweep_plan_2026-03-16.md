# 39 Ironclad Sweep Plan (2026-03-16)

## Ancestry

Ironclad follows Hardhat. Where Hardhat solidified the DINOv2 nodynbudget frontier and proved SigLIP-B is the new top VM, Ironclad opens the next chapter: training methodology and bridge architecture changes motivated by the "Novel Directions for an 80M-Parameter Multimodal VQA System" roadmap.

Sources:

- `tasks/mm_bridge/docs/37claude_hardhat_sweep_plan_2026-03-15.md`
- `tasks/mm_bridge/docs/36_crane_part1_sweep_report_2026-03-15.md`
- `tasks/mm_bridge/docs/Novel Directions for an 80M-Parameter Multimodal VQA System_ Architecture, Training, and Diagnostic Roadmap.pdf`
- `tasks/mm_bridge/docs/38_sports_imbalance_audit_2026-03-15.md`
- Hardhat Phase 1 + Phase 2 run bundles

## Entering State

### Hardhat Phase 1 Results (complete)

| Run | Final | Yes/No | Number | Other | Key finding |
|---|---:|---:|---:|---:|---|
| dinov2s_nodynbudget_d3 (Crane frontier) | 0.5762 | 0.7286 | 0.4039 | 0.5059 | reference |
| dinov2s_seed2 | 0.5658 | 0.7039 | 0.3900 | 0.5073 | seed variance ~0.01 |
| dinov2s_questiononly | **0.5803** | **0.7332** | 0.4097 | 0.5091 | **+0.004, new DINOv2-S winner** |
| dinov2s_d4 | 0.5508 | 0.7049 | 0.3629 | 0.4873 | deeper adapters hurt |
| dinov2s_qdepth4 | 0.5762 | 0.7272 | 0.4018 | 0.5076 | perceiver depth flat |
| dinov2s_questiononly_18k | **0.5915** | **0.7646** | 0.4092 | 0.5081 | **+0.015 over 9k, still climbing** |
| dinov2s_captionalign | 0.5421 | — | — | — | dead (below single-stage baseline) |
| **siglip_nodynbudget_d3** | **0.6095** | **0.7446** | **0.4532** | **0.5482** | **+0.033 over DINOv2-S, new overall frontier** |

Peak periodic eval on SigLIP-B was 0.6130 at step 8k (final full-val settled at 0.6095).

### Hardhat Phase 2 Status (in progress)

- `siglip_questiononly_18k`: **currently running**, at step ~1760 as of writing. This is the max-out run.
- `siglip_questiononly_18k_seed2`: not yet started (queued after run 11).
- `dinov2b_nodynbudget_d3`: not yet started (capacity-matched comparison).
- `siglip_questiononly_d3` @ 9k: failed at launch (likely a naming/config issue in the phase 2 launcher).
- `siglip_questiononly_18k_maxout` + `maxout_seed2`: also failed at launch.

### What Hardhat Settled

| Finding | Confidence |
|---|---|
| SigLIP-B is the new frontier VM (0.6095 vs 0.5762 DINOv2-S) | Very high |
| questiononly is the best question context mode | High (won on both VMs) |
| 18k training gives real gains on DINOv2-S (+0.015 over 9k, curve still rising) | High |
| Adapter depth d4 hurts (0.5508 vs 0.5762) | High |
| Perceiver qdepth4 is flat (0.5762 vs 0.5762) | High |
| Caption-align is dead under current bridge architecture | High |
| Seed variance on DINOv2-S nodynbudget is ~0.01 | Medium |

### What Hardhat Left Open (pending Phase 2 completion)

1. **SigLIP + questiononly + 18k**: will the DINOv2-S 18k gain pattern replicate on SigLIP?
2. **DINOv2-B vs SigLIP-B**: is SigLIP's win due to language alignment or model capacity?
3. **SigLIP seed stability**: is 0.6095 robust?

These will be answered by the currently running Phase 2. Ironclad should be planned assuming Phase 2 delivers a frontier in the **0.61–0.63** range (SigLIP questiononly 18k).

### Current Frontier Config (Hardhat Phase 1 best)

```
VM: SigLIP-B/16 (86M frozen, 196 tokens, 768-dim, language-aligned)
Bridge: perceiver_resampler, query_depth=3, 8 heads, 49 output queries
Token path: nodynbudget (all 196 SigLIP tokens → perceiver cross-attention)
Query mode: question_hidden_attn + question_only (questiononly pending confirmation on SigLIP)
Adapters: cross_attn d3, gate_init=0.5
Training: 9k steps, b192a1, cosine LR, bf16
```

## Ironclad Thesis

Hardhat answered the "which VM" and "which bridge config" questions. The architecture is now:
- a strong language-aligned frozen VM (SigLIP-B)
- a perceiver resampler that sees all tokens
- question-conditioned query generation
- cross-attention adapters in the top LM layers

The remaining accuracy gap to 0.70+ is not going to close by tuning adapter depth or perceiver depth. Hardhat proved those are dead levers. The next gains must come from one of:

1. **Training methodology** — how we train the existing architecture
2. **Bridge computation** — fundamentally different bridge primitives
3. **Diagnostic-driven fixes** — finding and fixing specific information loss points

The Novel Directions PDF prescribes concrete moves in all three categories. Ironclad selects the ones that are high-value at our scale and feasible within the existing infra.

## What Ironclad Does NOT Do

- No LM scaling (stays at the current ~27M LM)
- No new VM integration (SigLIP-B is the frontier; DINOv2-B comparison is a Hardhat Phase 2 deliverable)
- No data pipeline changes (dataset composition is a separate task: `tasks/datasetting`)
- No caption-align (dead)
- No dynbudget exploration (dead)

## Direction Selection From The PDF

The PDF proposes ~15 directions. Here's the triage for our specific situation (80M total params, SigLIP-B frozen VM, perceiver bridge, 16GB GPU):

### HIGH value, LOW risk — do these

| Direction | PDF section | Why it fits | Est. effort |
|---|---|---|---|
| **Contrastive auxiliary loss** | "Contrastive auxiliary losses" | Zero new params at inference, adds signal to bridge training. ConClaT showed +0.78% on VQAv2 at full scale. | ~2h eng |
| **Answer-type auxiliary head** | "Answer-type prediction" | Trivially cheap (3-way linear, ~0.01M). Forces bridge toward question-type sensitivity. | ~30min eng |
| **Task-progressive curriculum (TPCL)** | "Task-progressive curriculum learning" | Zero-cost, model-agnostic. Literature shows >5% on VQA-CP, meaningful on VQAv2. Only requires sorting training data by question type and a phased schedule. | ~2h eng |
| **Oracle experiment pipeline** | "Probing the bridge" | Inference-only, definitively locates the bottleneck. Tells us whether to invest in bridge vs LM vs VM. | ~3h eng |

### MEDIUM value, MEDIUM risk — selective

| Direction | PDF section | Why maybe | Concern |
|---|---|---|---|
| **Deformable cross-attention bridge** | "Deformable attention bridges" | Honeybee D-Abstractor scored massively higher than resampler. Spatially-adaptive extraction is a genuine new inductive bias. | Significant code, unclear if the gain transfers to our small LM. |
| **MoE bridge routing** | "Mixture-of-experts routing" | Question-type-specific experts could specialize without FLOP increase. | Engineering complexity high for 4-expert routing + load balancing. |
| **Knowledge distillation from BLIP-2** | "Knowledge distillation" | Highest absolute gain potential per the PDF. | Requires offline teacher inference on all VQAv2 images. One-time cost ~6-8h GPU. |

### LOW value or WRONG for us — skip

| Direction | Why skip |
|---|---|
| **Mamba bridge** | Novel but unproven at our scale, high implementation risk, SSM bidirectional scan is complex |
| **Hypernetwork-generated bridge weights** | "Highest novelty, more implementation risk" — the PDF itself flags this |
| **GRPO post-training** | Requires sampling multiple answers per question at temperature. Our LM is small enough that RL signal will be noisy |
| **Visual token reconstruction (ViCToR)** | Adds a reconstruction branch; our bridge is already well-regularized via prefix calibration |
| **Energy-based adaptive token selection** | We just proved hard token selection is destructive (dynbudget). Spectral selection is softer but the lesson still applies. |

## Ironclad Sweep Structure

### Tier 0: Oracle Diagnostics (no training, inference-only)

Before spending GPU hours on new training ideas, run the oracle experiments to locate the actual bottleneck. These cost only inference time and permanently inform all future decisions.

**Oracle 0A: Bridge bypass**

Feed all 196 SigLIP-B tokens directly to the LM via extended cross-attention, skipping the perceiver entirely. This requires temporarily expanding the LM visual prefix to 196 tokens.

- If accuracy jumps significantly (>0.03): the perceiver is compressing too aggressively. The right move is either more output queries (64 or 96 instead of 49) or a different bridge primitive.
- If accuracy is similar or worse: the perceiver is doing its job. The bottleneck is downstream.

**Config:** Load the SigLIP frontier checkpoint. Replace perceiver output with the raw SigLIP tokens (projected to LM dim). Eval-only, full val.

**Risk:** 196 visual prefix tokens + text may exceed max_seq_len=256. If so, truncate text to 60 tokens (enough for most VQA questions) as a diagnostic compromise.

**Est. time:** 20min (just forward passes)

**Oracle 0B: Visual sufficiency test**

Feed blank/random images through the full pipeline. The gap between random-image accuracy (~0.42 question-only baseline) and real-image accuracy (0.6095) is the model's visual utilization. At 0.19 effective visual signal, there's significant room to improve how well the bridge extracts and passes visual information.

**Config:** Replace real images with zeros or Gaussian noise. Eval-only.

**Est. time:** 20min

**Oracle 0C: EmbedLens-style token analysis**

Cluster the 49 bridge output tokens by activation pattern across the val set. Classify into alive (carry image-specific semantics), dead (no image-specific meaning), and sink (attention sinks) categories.

This requires:
1. Forward pass on ~1000 val images, collecting bridge output token activations
2. K-means or DBSCAN clustering on the token representations
3. Measure per-cluster variance across images (alive tokens vary, dead/sink tokens don't)

**Est. time:** 1-2h eng + 30min inference

**Decision gate after Tier 0:**

- If bridge bypass shows large gain → prioritize Tier 2 (bridge architecture changes)
- If token analysis shows many dead tokens → the perceiver is wasting capacity, increase output queries or add token-level diversity loss
- If visual sufficiency gap is small → the bridge is already extracting most available signal, focus on training methodology (Tier 1)
- If visual sufficiency gap is large → there's signal being lost somewhere, oracles 0A and 0C will tell you where

### Tier 1: Training Methodology (zero new architecture)

These changes modify how the existing SigLIP + perceiver + adapter architecture is trained. No new modules at inference time (the aux heads are discarded).

**Run 1: `siglip_questiononly_contrastive_aux`**

Add a supervised contrastive loss (InfoNCE) on the bridge output representation:
- Small projection head (bridge_dim → 128, 2-layer MLP, ~0.13M params, discarded at inference)
- Within each mini-batch, group (image, question) pairs by answer class
- L_total = L_CE + 0.3 * L_contrastive

This is the ConClaT approach. It pulls together bridge representations of pairs sharing the same answer while pushing apart different-answer pairs. The bridge learns to separate visual evidence by answer class, not just minimize cross-entropy.

**Eng work:**
1. Add `ContrastiveHead` module (~30 lines): Linear(512, 256) → ReLU → Linear(256, 128) → L2-normalize
2. Add `supervised_contrastive_loss()` function (~40 lines): group by answer index in batch, compute InfoNCE
3. Add `--aux_contrastive_weight` arg (default 0.3)
4. Attach loss after bridge output in the training loop
5. Discard head at eval

**Config:** SigLIP frontier + `--aux_contrastive_weight 0.3`

**Est. time:** 0.7h training + 2h eng

**Run 2: `siglip_questiononly_answertype_aux`**

Add a 3-way answer-type classifier head on the bridge output:
- Pool bridge output tokens → single vector
- Linear(512, 3) predicting yes/no vs number vs other
- L_total = L_CE + 0.1 * L_type

This is trivially cheap and pushes the bridge toward representations that distinguish question types early, before the LM has to figure it out.

**Eng work:** ~30 lines. Pool mean of bridge output, 1 linear layer, cross-entropy against the answer type label (derivable from existing VQAv2 annotations).

**Config:** SigLIP frontier + `--aux_answertype_weight 0.1`

**Est. time:** 0.7h training + 30min eng

**Run 3: `siglip_questiononly_tpcl`**

Task-progressive curriculum learning:
- Phase 1 (steps 0–3000): train only on yes/no questions
- Phase 2 (steps 3000–6000): add number questions
- Phase 3 (steps 6000–9000): train on full dataset

This requires a dataset wrapper that filters by question type per phase. The transition should be smooth (not a hard cutover) — linearly ramp the inclusion probability for each new type over 500 steps.

**Eng work:**
1. Add `question_type` field to the VQA dataset (derivable from the annotation's `answer_type`)
2. Add `CurriculumSampler` that filters/weights samples by training phase
3. Add `--curriculum_schedule` arg: `"yesno:0-3000,number:3000-6000,full:6000-9000"`

**Config:** SigLIP frontier + TPCL schedule

**Est. time:** 0.7h training + 2h eng

**Run 4: `siglip_questiononly_combined_aux`**

Stack contrastive + answertype aux losses together. If both Runs 1 and 2 showed positive signal individually, test whether they compose.

**Config:** SigLIP frontier + both aux heads

**Conditional:** only if Run 1 or Run 2 showed ≥+0.003

**Est. time:** 0.7h

**Run 5: `siglip_questiononly_[best_tier1]_18k`**

Take the best Tier 1 config and run for 18k steps. This stacks the training methodology win with the longer-training win from Hardhat.

**Conditional:** only if any Tier 1 run beat the SigLIP frontier

**Est. time:** 1.3h

### Tier 1 Subtotal: 3.8–5.1h (4-5 runs) + ~5h eng

**Decision gate after Tier 1:**

- If contrastive aux helps: the bridge was undertrained on answer-class separation. Consider increasing weight or exploring harder negatives.
- If TPCL helps: the model struggles with curriculum difficulty. Consider whether `number` questions need special handling (counting is typically catastrophic for small models per the PDF).
- If nothing helps: the training methodology is not the bottleneck. The architecture itself needs to change (Tier 2).

### Tier 2: Bridge Architecture Changes (new computation at inference)

These are more invasive changes. Only pursue after Tier 0 oracles and Tier 1 methodology runs inform where the actual bottleneck is.

**Run 6: `siglip_questiononly_deformable_bridge`**

Replace perceiver cross-attention with deformable cross-attention:
- Each of 49 query tokens attends to K=4 learned spatial sampling points (not all 196)
- Offsets predicted by a 2-layer MLP per head
- O(49 * 4 * num_heads) instead of O(49 * 196 * num_heads)

This is the Honeybee D-Abstractor approach. The key difference from the perceiver: deformable attention provides **spatially-adaptive, locality-aware** extraction. The perceiver does global soft averaging; deformable attention lets each query token focus on a specific spatial region with subpixel precision.

**Eng work:** This is substantial (~150 lines for the deformable attention module, ~50 lines for integration). The deformable attention implementation needs:
1. `DeformableAttention` module with offset prediction MLPs
2. Bilinear interpolation for sub-grid sampling
3. Integration into the bridge as a drop-in replacement for perceiver cross-attention layers

**Config:** SigLIP-B + deformable bridge (query_depth=3, 8 heads, K=4 sampling points)

**Est. time:** 0.7h training + 4-6h eng

**Run 7: `siglip_questiononly_queries_96`**

Before building a whole new bridge type, test whether the perceiver just needs more output queries. If Oracle 0A showed that bypassing the perceiver helps, this is the cheaper fix: increase perceiver output from 49 to 96 tokens.

This doubles the visual prefix in the LM from 49 to 96 tokens, leaving 160 positions for text (still enough for VQA questions which average ~12 tokens).

**Eng work:** Minimal — change `--num_visual_tokens 96`. May need to verify memory fits.

**Config:** SigLIP frontier + `--num_visual_tokens 96`

**Conditional:** only if Oracle 0A showed a gain from bypassing the perceiver

**Est. time:** 0.7h training + perf probe

### Tier 2 Subtotal: 1.4h training + 4-6h eng (conditional)

### Tier 3: Knowledge Distillation (highest ceiling, highest cost)

**Run 8: `siglip_questiononly_kd_blip2`**

This requires a one-time offline step: run BLIP-2 (or LLaVA-1.5-7B) on all VQAv2 training images to extract soft answer distributions. Then train with:

L = L_VQA + 0.3 * KL(softmax(student_logits/T), softmax(teacher_logits/T))

at temperature T=4.

The PDF calls this the highest absolute gain potential. VL2Lite showed distillation from CLIP-scale VLMs retains 98.4% of teacher performance at MobileNet-V2 scale.

**Eng work:**
1. Teacher inference script: run BLIP-2 on VQAv2 train split, save logits (~4-6h GPU, one-time)
2. `DistillationLoss` module (~40 lines)
3. DataLoader modification to load teacher logits alongside VQA samples

**Config:** SigLIP frontier + KD loss

**Conditional:** only if Tier 0 + Tier 1 suggest LM-side reasoning is the bottleneck rather than bridge extraction

**Est. time:** 0.7h training + 4-6h teacher inference + 3h eng

### Tier 4: Diagnostic Benchmarks (eval-only)

Run after the best Ironclad config is established. These don't produce accuracy gains but reveal specific failure patterns for future sweeps.

**Eval 4A: Per-question-type deep dive**

Beyond the standard yes/no / number / other split, break down accuracy by:
- Counting questions (TallyQA-style: "how many X?")
- Color questions ("what color is?")
- Spatial questions ("where is?", "which side?")
- Action questions ("what is X doing?")

This requires parsing question text with regex patterns. No model changes.

**Eval 4B: Confidence calibration**

Compute Expected Calibration Error (ECE) over 10 confidence bins. Plot reliability diagrams. At 0.61 accuracy, expect significant overconfidence. This informs whether post-processing (temperature scaling) would help.

**Eval 4C: Visual grounding quality**

Extract perceiver cross-attention weights, overlay on images, and qualitatively assess whether the model is attending to question-relevant regions. Sample 100 correct and 100 incorrect predictions.

## Recommended Execution Order

```
Phase A: Oracle diagnostics (Tier 0)              ~1-2h (inference only)
  → decision gate: where is the bottleneck?

Phase B: Eng work for Tier 1                       ~5h
  contrastive head + loss
  answer-type head + loss
  curriculum sampler

Phase C: Tier 1 training runs                      ~3.8h
  Run 1: contrastive aux
  Run 2: answer-type aux
  Run 3: TPCL
  Run 4: combined (conditional)
  Run 5: best + 18k (conditional)
  → decision gate: did methodology help?

Phase D: Tier 2 (conditional)                      ~5-7h eng + 1.4h training
  Run 6: deformable bridge (if oracles say bridge is bottleneck)
  Run 7: more queries (if oracle 0A was positive)

Phase E: Diagnostics (Tier 4)                      ~2h
  Per-type breakdown, calibration, grounding
```

**Total estimated: ~12-18h eng + 7-10h GPU**

This is larger than Hardhat because the directions are more engineering-heavy. But Ironclad is also more selective — the oracle diagnostics in Tier 0 gate whether Tiers 2 and 3 are worth pursuing, potentially saving 10+ hours.

## Decision Rules

### After Tier 0 Oracles

| Oracle result | Implication | Action |
|---|---|---|
| Bridge bypass >> perceiver (>0.03 gain) | Perceiver is the bottleneck | Prioritize Tier 2 (deformable bridge, more queries) |
| Bridge bypass ≈ perceiver | Perceiver is fine | Skip Tier 2, focus on Tier 1 methodology |
| Many dead bridge tokens (>30%) | Perceiver wastes capacity | Add token diversity loss or increase query count |
| Visual sufficiency gap < 0.15 | Bridge extracts most signal | Focus on LM-side improvements (aux losses, KD) |
| Visual sufficiency gap > 0.20 | Significant signal loss | Focus on bridge extraction (Tier 2) |

### After Tier 1

| Result | Action |
|---|---|
| Contrastive aux ≥ +0.005 | Keep, stack with other wins |
| TPCL ≥ +0.005 | Keep, investigate per-phase accuracy patterns |
| Answer-type aux ≥ +0.003 | Keep (it's free at inference) |
| All Tier 1 flat | Architecture is the bottleneck, proceed to Tier 2 |

### Project-Level Read

Ironclad answers the question: **"Now that we have the right VM and bridge config, what's the next category of improvement?"**

The answer will be one of:
- **Training methodology** (if Tier 1 works): cheap, composable, no new inference cost
- **Bridge architecture** (if Tier 2 works): the perceiver has reached its limits, need a different extraction primitive
- **Teacher knowledge** (if Tier 3 works): the small LM needs external guidance to reason better
- **None of the above** (if everything is flat): the LM itself is the bottleneck, and the project should graduate to a larger LM

That last outcome is important. If a 27M LM with a well-tuned bridge + SigLIP-B saturates around 0.61-0.63 regardless of training tricks and bridge architecture, then the next real move is LM scaling — and that's a different project entirely.

## What This Sweep Answers

| Question | Runs | Diagnostic |
|---|---|---|
| Where does information die in the pipeline? | Oracles 0A-0C | Bottleneck location |
| Does contrastive bridge training help? | Run 1 vs frontier | Bridge representation quality |
| Does question-type awareness help? | Run 2 vs frontier | Bridge specialization |
| Does curriculum scheduling help? | Run 3 vs frontier | Training dynamics |
| Do training tricks compose? | Run 4 vs Runs 1-3 | Composability |
| Is the perceiver the limiting factor? | Run 6, Run 7, Oracle 0A | Bridge architecture |
| What specific question types fail? | Eval 4A | Per-type diagnosis |
| Does the model know when it's wrong? | Eval 4B | Calibration |
| Is the model looking at the right regions? | Eval 4C | Grounding quality |

## One-Line Summary

Ironclad should answer "what category of improvement comes next" by first diagnosing the bottleneck with inference-only oracles, then testing zero-parameter training methodology changes (contrastive aux, answer-type head, curriculum), and only if those are flat, moving to bridge architecture changes (deformable attention, more queries) or knowledge distillation.
