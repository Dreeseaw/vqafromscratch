# 33 Crane Sweep Plan (2026-03-14)

## Context

This plan accounts for:

- Report 32a (coworker's Plank sweep report)
- Report 32b (my Plank sweep report)
- The full codebase, including `train/mm.py`, `models/bridge.py`, `models/hf_vision.py`
- All prior sweep history and AUTORESEARCH_STATE
- The project's deferred ideas list

The Plank sweep established a new frontier at `0.5240` from `mobilevit_attnqquery_dynbudget_adapter_d3_cap64`. Both reports agree on all material findings. Crane builds on that consensus.

## The Gap to Close

Current frontier: `0.5240` (single seed, MobileViT + attnqquery + dynbudget + adapter d3 + cap64).

BLIP-2 reference: `0.6520` on VQAv2.

Gap: `0.1280`.

BLIP-2's resource envelope:

- ViT-g/14: 1B params, 257 tokens at 1408-dim (pre-trained on 129M image-text pairs via EVA-CLIP)
- Q-Former: 188M trainable params, 32 learned queries, pre-trained on 129M image-text pairs in two stages
- OPT-2.7B: 2.7B params, strong language model

Our resource envelope:

- MobileViT-small: 5.6M params, 49 tokens at 640-dim (ImageNet-1k only)
- Perceiver resampler bridge: ~2M trainable params
- LM: ~26M frozen + ~20M trainable (adapters + bridge), 12 layers, 512-dim
- Training data: VQAv2 only (~440k train pairs)

The honest reading of the gap is that most of it comes from three sources, roughly in order of magnitude:

1. **Vision model capacity and pre-training data** (~0.05–0.08 of the gap). BLIP-2 uses ViT-g pre-trained on 129M image-text pairs. We use MobileViT-small pre-trained on ImageNet-1k. The Plank result showed a +0.058 jump from upgrading vitvae2 (49 tokens × 256-dim, ImageNet) to MobileViT-small (49 tokens × 640-dim, ImageNet). A further upgrade to a language-aligned vision encoder with more tokens, more parameters, and image-text pre-training should yield another large step.

2. **Language model capacity** (~0.03–0.05 of the gap). OPT-2.7B has roughly 100× our LM parameter count. More capacity means better answer generation, especially for complex open-ended questions. Our `other` category at 0.4401 vs BLIP-2's substantially higher `other` reflects this. We cannot close this gap without a larger LM, but we can narrow it with better visual grounding.

3. **Bridge pre-training and alignment quality** (~0.01–0.03 of the gap). BLIP-2's Q-Former is pre-trained on 129M image-text pairs in two stages (image-text contrastive + image-grounded text generation). Our bridge trains from scratch on VQA supervision alone. The bridge has never seen a visual concept outside of the VQA answer distribution.

The first source is the one Crane can most directly address. The second is a hard structural constraint. The third is addressable but requires new infrastructure.

## What Plank Told Us That Crane Must Respect

### 1. VM quality is the dominant lever

The +0.058 jump from the VM switch was larger than all prior gains combined. This is not ambiguous. The strongest next move is a better VM.

### 2. attnqquery benefits most from richer vision

Under MobileViT, `attnqquery` reversed its Nail-era loss to `lmmeanqquery`, driven by a +0.0118 gain on `other`. The mechanism is clear: attention-derived queries can focus on specific question-relevant LM positions, and this precision matters more when each visual token carries more information. A further VM upgrade with even richer tokens should amplify this effect.

### 3. The original-VM family is dead for frontier work

All five Stage B runs clustered in [0.4637, 0.4699]. No architectural change to the bridge moved the needle meaningfully without changing the VM. Crane should not spend any budget on original-VM experiments.

### 4. dynbudget has not yet been tested in a regime where it matters

With 49 tokens and cap=64, the `qadaptive` selector trivially passes all tokens. dynbudget has never operated in a regime where it must actually filter. A VM with more tokens (e.g. CLIP ViT-B/16 at 196 tokens) would be the first real test of whether question-conditioned token selection has value.

### 5. The training curves are still rising

Both reports note that MobileViT attnqquery shows no plateau at 9k steps. The periodic eval curve is linear and rising through the end of training. There is likely free performance from longer runs, though diminishing returns will set in.

### 6. Seed variance is real

The 0.0051 gap between two seeds of MobileViT lmmeanqquery is large relative to the between-variant differences within the MobileViT family (attnqquery vs lmmeanqquery is 0.0059). The attnqquery frontier has not been seed-checked at all. Any Crane result must be interpreted against this variance floor.

## Tier 1: Low Engineering Risk, Mid-Entropy Frontier Runs

These runs use existing infrastructure with flag changes only. Each tests a specific hypothesis about stacking known-positive signals on the MobileViT frontier.

### Run 1: `mobilevit_questiononly_attnqquery_dynbudget_adapter_d3_cap64`

**Hypothesis:** Combining question-only pooling with attention-derived queries on MobileViT features will improve the frontier.

**Modeling reasoning:**

The `attnqquery` path works by attending over LM hidden states to form a question-conditioned query for the perceiver. The attention weights are computed over all text tokens in the input sequence—which includes both the question and the prompt template ("Question: ... Answer:").

The `questiononly` modification (`--bridge_question_context_mode question_only`) restricts this pooling to the question span only. On the original VM, this was the cleanest Stage B win (+0.0046).

The interaction between these two changes is not trivially additive. Here is why it should be positive:

The attention mechanism in `attnqquery` computes a weighted sum over LM hidden states. When the attention pool includes prompt-template tokens ("Question:", "Answer:"), those tokens carry generic structural information that dilutes the question-specific signal. The attention weights can partially compensate by downweighting template tokens, but with a shallow attention head, this wastes representational capacity on learning to ignore irrelevant tokens rather than focusing on question semantics.

By restricting the attention pool to question tokens only, we remove the distractor tokens entirely. The attention head can now spend all of its capacity on differentiating between question-relevant aspects: the subject entity, the question type (color, location, count, action), and the key attribute being asked about.

This effect should be amplified on MobileViT because:

1. MobileViT features carry finer-grained spatial/attribute information than the old VM. A sharper query extracts more of this information.
2. The `other` category—where `attnqquery` showed its largest gain—is exactly where question-specificity matters most. Open-ended questions like "What color is the cat on the left?" require the query to capture both the entity and the attribute, not just a diffuse average of the input.
3. On the old VM, `questiononly` improved `other` by +0.0056 over the Nail winner. On MobileViT, where the `other` baseline is already 0.4401 (not 0.3298), there is more headroom for question-focused queries to exploit.

The risk is that `attnqquery` already implicitly downweights template tokens via attention, making the explicit restriction redundant. But the +0.0046 gain from `questiononly` on the old VM—where the base query mechanism was `lmmeanqquery`, which has no such implicit mechanism—suggests the template tokens do carry real noise.

**Expected range:** 0.520–0.530. The lower bound assumes redundancy with attnqquery's implicit focusing. The upper bound assumes the gains stack with modest interaction.

**Config delta from frontier:**
```
--bridge_question_context_mode question_only
```
Everything else identical to the frontier run.

### Run 2: `mobilevit_attnqquery_dynbudget_adapter_d4_cap64`

**Hypothesis:** Deeper LM visual cross-attention adapters will extract more value from MobileViT features.

**Modeling reasoning:**

The LM visual adapter stack inserts cross-attention layers at evenly-spaced positions in the LM's 12 transformer layers. Currently:

- d2 = adapters at layers 4, 8 (every 4 layers)
- d3 = adapters at layers 3, 6, 9 (every 3 layers)
- d4 = adapters at layers 2, 4, 7, 9 (every ~2.5 layers)

The adapter at each position allows the LM to re-attend to visual prefix tokens during generation. With d3, visual information enters the LM reasoning chain at three discrete points. Between those points, the LM processes text-only for 3 layers before getting another chance to look at the image.

The case for d4 rests on the interaction between adapter depth and visual feature quality:

**Under the old VM (Nail):** d2 → d3 gave +0.0009 (0.4644 → 0.4653). This is a marginal gain. The old VM produces 49 tokens at 256-dim—relatively coarse features with limited spatial and attribute detail. Adding more re-access points yields diminishing returns because the visual features themselves don't have much more to give at each access.

**Under MobileViT:** The features are 640-dim and encode finer-grained spatial relationships and attribute information. The `other` category improvement (+0.093 from old VM to MobileViT) demonstrates that MobileViT features contain richer evidence that was not available before. With richer features available, more frequent re-access during LM reasoning should be more valuable: each adapter layer can extract different aspects of the visual evidence at different stages of answer generation.

The specific mechanism: in a 12-layer LM generating a multi-token answer, different layers handle different levels of abstraction. Early layers establish entity grounding, middle layers handle attribute binding, and later layers refine the answer token distribution. With d3, there are 3-layer gaps where the LM must reason about visual evidence "from memory." With d4, these gaps shrink to ~2.5 layers, allowing tighter visual grounding throughout the generation process.

The risk is parameter overhead and optimization difficulty. d4 adds one more cross-attention module (~0.5M params), and distributing adapters more densely in a 12-layer LM may cause optimization interference. But with 9k steps and an effective batch of 192, the training budget should be sufficient for one additional adapter layer.

There is also the question of whether d3 → d4 follows the same diminishing-returns pattern as d2 → d3 on the old VM. Under the old VM, d2 → d3 = +0.0009. Under MobileViT with richer features, the marginal value of an additional adapter layer could be higher. The key test is whether the d3 → d4 delta on MobileViT exceeds the d2 → d3 delta on the old VM.

**Expected range:** 0.522–0.532. The lower bound assumes the same diminishing pattern as the old VM. The upper bound assumes richer features make denser visual access more productive.

**Config delta from frontier:**
```
--lm_visual_adapter_layers 4
```
Everything else identical to the frontier run.

### Run 3: `mobilevit_attnqquery_adapter_d3_cap64_nodynbudget`

**Hypothesis:** With 49 tokens and cap=64, the `qadaptive` token selector passes all tokens and its scoring signal has no filtering effect—but it may still act as a learned attention bias.

**Modeling reasoning:**

The `qadaptive` selector (`models/bridge.py` lines 258–277) works as follows:

1. Projects the question context into a query via `token_selector_qproj`
2. Computes per-token relevance scores via a small MLP (`token_selector`)
3. Applies tanh gating to the scores
4. Selects the top-k tokens, where k is determined by a learned `token_budget` network

With 49 visual tokens and cap=64, the budget network can request up to 64 tokens. Since only 49 exist, all tokens are always selected. The selector never filters.

But the scoring signal is still computed and the tanh-gated scores still modulate the token representations before they enter the perceiver. This means dynbudget, in the current configuration, is operating purely as a learned attention-like re-weighting of visual tokens conditioned on the question. It is not a sparsity mechanism—it is a soft attention mask.

This ablation tests whether that re-weighting helps or hurts. Two plausible outcomes:

**Positive (nodynbudget wins):** The scoring signal adds noise. Since all tokens pass anyway, the tanh gating may distort token magnitudes without meaningful filtering benefit. Removing it gives the perceiver cross-attention a cleaner signal. If this is the case, we should drop dynbudget from the frontier config until we have enough tokens for it to actually filter.

**Negative (dynbudget still wins even with no filtering):** The question-conditioned scoring acts as a useful soft attention prior that helps the perceiver focus on question-relevant tokens. Even though no tokens are dropped, the magnitude modulation provides useful gradient signal during training. If this is the case, the mechanism has value beyond sparsity and should be kept.

This ablation has strategic importance beyond the immediate score: if we move to CLIP ViT-B/16 (196 tokens) in Tier 2, understanding whether dynbudget's value comes from filtering vs. re-weighting will inform whether to raise the cap to match the larger token count or keep it at 64 for genuine sparsity.

**Expected range:** 0.518–0.526. The tight range reflects genuine uncertainty about the direction. If the re-weighting is purely noise, we recover the attnqquery + adapter d3 baseline on MobileViT (roughly 0.522 from the signal without dynbudget distortion). If the re-weighting is useful, removing it costs ~0.002–0.005.

**Config delta from frontier:**
```
--bridge_token_selector_type none
--bridge_token_select_k 0
```
Everything else identical to the frontier run.

## Tier 2: Constraint-Breaking Approaches

These require new code and make structural changes to the system. They target the primary bottleneck identified by Plank: vision model quality.

### Run 4: `clip_vit_attnqquery_dynbudget_adapter_d3_cap64`

**Hypothesis:** CLIP ViT-B/16 as a frozen VM will produce a step-change improvement comparable to or larger than the vitvae2 → MobileViT jump, because it provides language-aligned features, 4× more tokens, and substantially richer representations.

**The BLIP-2 decomposition argument:**

BLIP-2 uses ViT-g/14 (1B params, EVA-CLIP pre-trained). We cannot run ViT-g on our hardware. But we can ask: how much of BLIP-2's vision advantage comes from language alignment vs. raw scale?

CLIP ViT-B/16 (86M params) was pre-trained on 400M image-text pairs via contrastive learning. It produces 196 tokens at 768-dim for 224×224 input. It is:

- **Language-aligned.** The contrastive objective forces the visual features to be predictive of text descriptions. This is qualitatively different from ImageNet classification features. For VQA, where the bridge must translate visual evidence into language-compatible representations, language-aligned features should provide a much better starting point. The bridge no longer needs to learn the concept of "what aspects of images relate to language" from VQA supervision alone—CLIP has already encoded this.

- **Spatially richer.** 196 tokens vs 49 tokens means 4× more spatial resolution. Fine-grained questions about small objects, spatial relationships, and counting tasks all benefit from more visual tokens. This is also the first configuration where `dynbudget` with cap=64 would be required to actually filter—selecting 64 of 196 tokens is genuine question-conditioned sparsity.

- **Feature-dense.** 768-dim vs 640-dim, with representations shaped by 400M diverse image-text pairs rather than 1M ImageNet images. The feature space encodes a much broader range of visual concepts.

**Quantitative estimate of the gain:**

The vitvae2 → MobileViT jump was +0.058 on overall accuracy. That jump came from:
- More params: 2.7M → 5.6M (2× increase)
- More feature dim: 256 → 640 (2.5× increase)
- Same token count: 49 → 49 (no change)
- Same pre-training: ImageNet → ImageNet (no change)

MobileViT → CLIP ViT-B/16 would provide:
- More params: 5.6M → 86M (15× increase)
- More feature dim: 640 → 768 (1.2× increase)
- More tokens: 49 → 196 (4× increase)
- Better pre-training: ImageNet → 400M image-text pairs (qualitative change)

The token count increase and the pre-training change are both qualitatively new factors that were not present in the first VM jump. The 15× parameter increase is also much larger than the 2× of the first jump.

Conservative estimate: +0.04 over current frontier → `0.564`
Optimistic estimate: +0.08 over current frontier → `0.604`

This would not reach BLIP-2 (0.652), but it would narrow the gap from 0.128 to somewhere in 0.048–0.088. The remaining gap would be attributable to LM capacity (our 46M total vs OPT-2.7B) and bridge pre-training.

**Why this is the right constraint to break first:**

The LM capacity gap is the hardest to close and the most expensive in engineering terms (new tokenizer, new LM checkpoints, potential training instability). Bridge pre-training is high-upside but requires new data infrastructure and a multi-stage training pipeline. Swapping the frozen VM is the cheapest structural change that addresses the largest single bottleneck.

**Engineering plan:**

The code change is well-scoped. `models/hf_vision.py` provides the exact template:

1. Write `HFCLIPViTBackbone` class (~80 lines) following the `HFMobileViTSmallBackbone` pattern. Key differences:
   - Import `CLIPVisionModel` from `transformers` (not `CLIPModel`—we want the vision tower only)
   - The CLIP image processor uses different normalization constants: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
   - No BGR flip needed (CLIP uses RGB)
   - `_encoder()` returns `outputs.last_hidden_state` — same convention as MobileViT
   - Output shape: [B, 197, 768] (196 patch tokens + 1 CLS token). We should return all 197 or strip the CLS token — the perceiver bridge will handle whatever token count it receives

2. Add `clip_vit` to `--vision_model` choices in `train/mm.py` line 2405

3. Add elif branch in `build_vision_model_from_args()` at line 338 (alongside the `mobilevit_hf` branch)

4. Download `openai/clip-vit-base-patch16` to `logs/hf_vision/openai_clip_vit_base_patch16`

5. No bridge changes needed. The perceiver resampler's `visual_proj` linear layer will adapt from 768-dim to `lm_hidden_size` (512). The cross-attention mechanism handles arbitrary token counts.

6. Batch layout: CLIP ViT-B/16 is heavier than MobileViT-small. With 196 tokens, the bridge cross-attention cost scales ~4× for the visual side. We may need to drop to `batch_size=48, grad_accum_steps=4` to maintain effective batch 192 on 16GB. The eval batch size may also need reduction to 48.

**What this run tests beyond the score:**

- Whether language-aligned features change the qquery ordering again (attnqquery vs lmmeanqquery could shift again under CLIP features, since CLIP features are already partially "language-shaped")
- Whether dynbudget at cap=64 with 196 tokens actually helps (the first real filtering regime)
- Whether our small LM and bridge architecture can absorb the richer features or whether we hit a bridge/LM capacity ceiling
- Whether the adapter d3 configuration is still appropriate with 4× more visual tokens

**Risks:**

- Memory. CLIP ViT-B/16 with 86M params and 196 output tokens on 16GB may require aggressive batch size reduction. If `batch_size=48, grad_accum_steps=4` doesn't fit, we fall back to 32×6.
- Training speed. More tokens × more params = slower. May need to extend beyond 9k steps if convergence is slower with the richer feature space.
- Feature scale mismatch. CLIP features are normalized differently from MobileViT. The `visual_proj` in the bridge should handle this, but if the feature norms are very different, the bridge learning dynamics may be affected.

### Run 5: `mobilevit_attnqquery_captionalign_adapter_d3_cap64`

**Hypothesis:** Pre-training the bridge to align its output with caption semantics will improve the bridge's starting point for VQA fine-tuning, because the bridge will enter VQA training already knowing how to extract language-relevant visual summaries.

**The bridge pre-training argument:**

Currently, the bridge learns everything from VQA supervision: how to extract visual features, how to compress them into LM-compatible tokens, and which visual aspects are relevant to questions. VQA supervision is narrow—it only rewards getting the correct answer token. The bridge receives gradient signal through the LM, which means it must learn visual extraction quality indirectly, through the lens of what helps the LM predict answer tokens.

BLIP-2's Q-Former pre-training addresses this with two explicit alignment stages:

1. **Image-Text Contrastive (ITC):** align the Q-Former output with text embeddings via contrastive loss
2. **Image-grounded Text Generation (ITG):** use the Q-Former output to generate captions

We cannot replicate the full BLIP-2 pre-training pipeline (no 129M pairs, no contrastive infrastructure). But we can implement a lightweight version of stage 2: given an image, train the bridge to produce outputs whose mean representation aligns with the LM's encoding of a description of that image.

**CaptionAlign mechanism:**

Stage 1 (pre-training, ~3k steps):
- Input: COCO image + COCO caption pair
- Forward: image → frozen VM → bridge → mean-pool bridge output → representation `v`
- Target: caption → frozen LM encode → mean-pool hidden states → representation `t`
- Loss: `1 - cosine_similarity(v, t)` (+ optional L2 regularization on bridge params)

Stage 2 (VQA fine-tuning, standard 9k steps):
- Normal VQA training from the pre-trained bridge checkpoint

**Why this should help:**

The bridge currently starts from random initialization. Its queries have no prior on what visual information is language-relevant. The perceiver must learn from scratch—through noisy VQA gradients—which visual features to extract and how to format them for the LM.

CaptionAlign gives the bridge a warm start: before seeing any VQA examples, it has already learned to produce representations that, when averaged, match the LM's encoding of a description of the image content. This means:

1. The bridge's visual extraction has a prior toward "language-relevant visual content" rather than starting from a random point in representation space.
2. The perceiver queries are initialized in a part of the space where the LM can already interpret them, rather than requiring the first thousands of VQA steps to establish basic bridge-LM alignment.
3. The pre-training uses a direct alignment signal (cosine similarity with LM caption encoding) rather than an indirect one (VQA answer prediction gradient backpropagated through the frozen LM).

**Why this is high-entropy / high-upside:**

Bridge pre-training has been deferred since the project's first sweep. The AUTORESEARCH_STATE journal (2026-03-13) called it "a phase-change investment, not the next cheap high-information move." That was correct when the bridge was still being debugged and the VM was the old vitvae2.

Now the situation is different:
- The bridge architecture is stable (perceiver resampler with qquery + dynbudget + adapters)
- The VM is stronger (MobileViT provides richer features worth aligning to)
- The bridge-only architectural modifications have mostly saturated (Plank Stage B showed diminishing returns)

This means the bridge's random initialization is now more likely to be a real bottleneck than before. With the architecture stabilized and the VM upgraded, the bridge starting point is the next natural constraint to address.

**Quantitative estimate:**

The gain from bridge pre-training is hard to predict because it is a qualitative change in the training procedure, not a parameter or architecture modification. Analogies from the literature:

- BLIP-2's Q-Former pre-training is essential to its performance, but it uses 129M pairs and a much larger model, so the magnitude is not directly comparable.
- In our setting with COCO captions (~120k training images, ~5 captions each), the pre-training signal is much smaller. The expected gain is modest: +0.01 to +0.04.

Conservative estimate: +0.010 → `0.534`
Optimistic estimate: +0.040 → `0.564`

The wide range reflects genuine uncertainty about whether a lightweight alignment step provides meaningful benefit at our scale, or whether the VQA fine-tuning stage can already learn the necessary alignment from scratch in 9k steps.

**Engineering plan:**

This requires more new code than Run 4:

1. **Caption dataset class** (~60 lines). Following `train/vqa_data.py` as template. Load COCO captions, pair with images, return `(image, caption_text)` batches. COCO 2014 train has ~83k images and ~414k captions. We use one caption per image per epoch (random selection).

2. **Caption encoding utility** (~20 lines). Use the frozen LM to encode each caption: tokenize → forward through frozen LM → mean-pool hidden states → normalize. This can be precomputed and cached if needed.

3. **Pre-training loop** (~80 lines). Follows the structure of the VQA training loop in `train/mm.py` but with:
   - Different data loader (caption pairs instead of VQA triplets)
   - Different loss function (cosine similarity instead of cross-entropy)
   - Only bridge parameters are optimized (VM and LM both frozen)
   - Shorter schedule (~3k steps)

4. **Two-stage launcher** (~30 lines shell script). Stage 1: pre-training with cosine loss. Stage 2: load pre-trained bridge weights into standard VQA training.

Total new code: ~200 lines.

**Data dependency:** This requires COCO 2014 caption annotations. The images are already available (VQAv2 uses COCO images). The caption annotation file (`captions_train2014.json`) needs to be downloaded (~20MB).

**Risks:**

- Pre-training collapse. Mean-pooled cosine alignment is a simple objective. The bridge might find trivial solutions (e.g., producing near-constant outputs that match the average caption embedding). Regularization and monitoring are needed.
- Stage transition. The bridge parameters shift during pre-training. If the pre-trained representations are too far from what the VQA fine-tuning expects, the first phase of VQA training may be unstable. A learning rate warmup at the VQA stage should mitigate this.
- The pre-training signal may be too weak at 3k steps with ~83k images to provide meaningful alignment. We may need to tune the step count.

## Tier 1+2 Priority Queue

| Priority | Run | Tier | Engineering | Expected Range |
|---|---|---|---|---|
| 1 | `mobilevit_questiononly_attnqquery_dynbudget_adapter_d3_cap64` | 1 | Flag change only | 0.520–0.530 |
| 2 | `clip_vit_attnqquery_dynbudget_adapter_d3_cap64` | 2 | ~100 lines new code | 0.564–0.604 |
| 3 | `mobilevit_attnqquery_dynbudget_adapter_d4_cap64` | 1 | Flag change only | 0.522–0.532 |
| 4 | `mobilevit_attnqquery_adapter_d3_cap64_nodynbudget` | 1 | Flag change only | 0.518–0.526 |
| 5 | `mobilevit_attnqquery_captionalign_adapter_d3_cap64` | 2 | ~200 lines new code | 0.534–0.564 |

The ordering reflects both expected value and information value:

- **Run 1 first** because it is the cheapest way to test whether the two cleanest Plank wins (questiononly + attnqquery) stack on MobileViT. It sets a baseline for "how good can Tier 1 get" within the current infrastructure.
- **Run 2 second** because the VM upgrade is the highest-expected-value single change and the engineering is well-scoped. If CLIP ViT-B/16 produces a large jump, it redefines the frontier for all subsequent runs (and potentially makes the MobileViT Tier 1 results moot for frontier work, though still informative for ablation).
- **Run 3 third** because adapter depth is the second most promising architectural lever after VM quality, and the test is free (flag change only).
- **Run 4 fourth** because the dynbudget ablation is informative for future planning (especially for the CLIP regime where dynbudget would finally do real filtering) but has the widest uncertainty range.
- **Run 5 last** because it has the highest engineering cost and the most uncertainty. If Run 2 (CLIP) produces a large jump, we may want to do CaptionAlign on CLIP features instead of MobileViT features.

## Seed Check

The Crane plan assumes `mobilevit_attnqquery_dynbudget_adapter_d3_cap64_seed2` from the Plank priority list runs alongside or before the Crane queue. If the seed check shows the attnqquery frontier is unstable (gap > 0.008 from seed1), the Tier 1 runs should be re-evaluated—they all build on the assumption that attnqquery is the correct base configuration on MobileViT.

## Standard Config (MobileViT family)

All MobileViT-based runs use the established MobileViT layout:

```
--vision_model mobilevit_hf
--vision_checkpoint logs/hf_vision/apple_mobilevit_small
--batch_size 96
--grad_accum_steps 2
--eval_batch_size 96
--max_steps 9000
--eval_every 1000
--eval_batches 100
--eval_use_kv_cache
--eval_kv_cache_mode batched
--precision bf16
```

Final eval: full validation split with official scorer.

## Standard Config (CLIP family)

Estimated. May need adjustment based on memory profiling:

```
--vision_model clip_vit
--vision_checkpoint logs/hf_vision/openai_clip_vit_base_patch16
--batch_size 48
--grad_accum_steps 4
--eval_batch_size 48
--max_steps 9000
--eval_every 1000
--eval_batches 100
--eval_use_kv_cache
--eval_kv_cache_mode batched
--precision bf16
```

If `batch_size=48` doesn't fit with CLIP's 196 tokens on 16GB, fall back to `batch_size=32, grad_accum_steps=6`. The effective batch must remain 192 for comparison policy compliance.

## Longer-Term View

If Crane's Tier 2 runs deliver as modeled:

- CLIP ViT-B/16 at 0.56–0.60 would put us within 0.05–0.09 of BLIP-2.
- The remaining gap would be dominated by LM capacity.
- The next constraint to break after CLIP would be the LM. Possible paths: load a larger pre-trained LM (GPT-2 117M or GPT-2 345M), accept the tokenizer change cost, and run the full bridge + adapter stack on top. This is a major engineering change but would address the largest remaining gap.
- CaptionAlign (or a stronger variant) on CLIP features could provide an additional +0.01–0.03 by improving bridge initialization quality.
- DINOv2 ViT-B/14 is an alternative to CLIP ViT-B/16 with stronger spatial features but no language alignment. If CLIP's language alignment proves less important than expected, DINOv2 becomes the comparison.

The realistic ceiling for the current LM (46M params) with the strongest available frozen VM and optimal bridge alignment is somewhere in the range of 0.58–0.62. Closing the last 0.03–0.07 to BLIP-2 will almost certainly require a larger LM.
