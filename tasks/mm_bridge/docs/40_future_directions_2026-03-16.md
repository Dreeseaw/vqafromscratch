# 40 Future Directions (2026-03-16)

## Purpose

This document steps back from the current sweep cadence to assess where the mm_bridge project is heading, what the Hardhat results actually mean for the project's long-term trajectory, and what comes after bridge architecture research.

This is a forward-looking document, not a sweep plan. It overlaps with the Ironclad sweep plan (doc 39) in places but addresses a wider scope: project identity, VM strategy, and the path toward a from-scratch multi-benchmark VLM.

## Where We Are After Hardhat

### The Complete Hardhat Scoreboard

| Run | Steps | Final | Y/N | Num | Other | Status |
|---|---:|---:|---:|---:|---:|---|
| **SigLIP-B questiononly 18k** | **10k/18k** | **0.6173** (periodic) | **0.7774** | **0.4384** | **0.5469** | **crashed (mem leak at step ~10.4k)** |
| SigLIP-B baseline | 9k | 0.6095 | 0.7446 | 0.4532 | 0.5482 | complete |
| DINOv2-B baseline | 9k | 0.5953 | 0.7388 | 0.4208 | 0.5323 | complete |
| DINOv2-S questiononly 18k | 18k | 0.5915 | 0.7646 | 0.4092 | 0.5081 | complete |
| DINOv2-S questiononly | 9k | 0.5803 | 0.7332 | 0.4097 | 0.5091 | complete |
| DINOv2-S qdepth4 | 9k | 0.5762 | 0.7272 | 0.4018 | 0.5076 | complete |
| DINOv2-S baseline (Crane frontier) | 9k | 0.5762 | 0.7286 | 0.4039 | 0.5059 | complete |
| DINOv2-S seed2 | 9k | 0.5658 | 0.7039 | 0.3900 | 0.5073 | complete |
| DINOv2-S d4 adapters | 9k | 0.5508 | 0.7049 | 0.3629 | 0.4873 | complete |
| DINOv2-S captionalign | 9k | 0.5421 | 0.7152 | 0.3847 | 0.4521 | complete |
| SigLIP-B questiononly 18k seed2 | 1.4k/18k | 0.4467 (1k eval) | — | — | — | crashed (mem leak) |

### Phase 2 Completion Notes

Phase 2 was shortened. The 18k SigLIP runs died from a GPU memory leak — the b192a1 batch layout passed short probes but leaked memory over hours, causing steps/s to degrade from ~3.0 to ~0.2 before the watchdog killed the process. The correct layout for SigLIP-B and DINOv2-B on 16GB is b96a2.

The siglip_questiononly 18k run reached step 10,480 before dying. Its last periodic eval at step 10k was **0.6173** — still climbing at +0.007/eval. Had it completed to 18k, extrapolating the DINOv2-S 18k curve shape suggests a final in the **0.63–0.64** range.

The 18k seed2 and 9k siglip_questiononly runs failed at launch or died early. Not worth re-running given the project direction shift below.

### What Hardhat Proved

**1. Language alignment wins decisively.**

The DINOv2-B vs SigLIP-B comparison is the cleanest test:

| VM | Params | Tokens | Pre-training | Final |
|---|---:|---:|---|---:|
| SigLIP-B/16 | 86M | 196 | Language-aligned (WebLI sigmoid contrastive) | **0.6095** |
| DINOv2-B/14 | 86M | 256 | Self-supervised (LVD-142M DINO+iBOT) | 0.5953 |

Same param count, same feature dim (768), SigLIP has *fewer* tokens (196 vs 256). SigLIP still wins by +0.014. Since Crane proved more tokens is better, SigLIP winning with fewer tokens means the per-token quality advantage from language alignment more than compensates for 60 fewer tokens. This is definitive: **for VQA through a perceiver bridge, language-aligned features are worth more per token than self-supervised spatial features.**

The per-category breakdown reinforces this:

| Category | SigLIP-B | DINOv2-B | Delta |
|---|---:|---:|---:|
| Yes/No | 0.7446 | 0.7388 | +0.006 |
| Number | **0.4532** | **0.4208** | **+0.032** |
| Other | **0.5482** | **0.5323** | **+0.016** |

SigLIP's advantage is concentrated in `number` (+3.2%) and `other` (+1.6%). Language-aligned features help most on questions that require compositional understanding — counting objects requires knowing what the objects *are* (language concepts), and open-ended questions require relating visual evidence to semantic categories.

**2. questiononly is confirmed on SigLIP.**

SigLIP questiononly at step 9k (periodic eval): 0.6105 vs SigLIP baseline final: 0.6095. The effect is small but consistent with DINOv2-S where questiononly gave +0.004. The 18k questiononly curve was at 0.6173 at step 10k and still climbing.

**3. 18k training has real headroom.**

On DINOv2-S: 0.5915 at 18k vs 0.5762 at 9k = +0.015.
On SigLIP: 0.6173 at 10k (periodic) vs 0.6095 at 9k (final) = +0.008 minimum, likely +0.03–0.05 if completed.

**4. Adapter depth and perceiver depth are dead levers.**

d4 adapters: 0.5508 (worse). qdepth4: 0.5762 (flat). These won't be revisited.

**5. Caption-align is dead.**

0.5421 vs 0.5762 baseline. Properly configured (with --reset_schedule) and still 3.4 points below. The bridge does not benefit from caption pre-training in this architecture.

## The Strategic Read

### What the bridge architecture work has established

Over Nail → Plank → Crane → Hardhat, the project has converged on a clear picture:

1. **The perceiver resampler works.** It distills variable-length token sequences into a fixed 49-token prefix. Dynbudget (hard pre-filtering) hurts. The perceiver should see all tokens.

2. **Question conditioning works.** Attention-derived query banks (attnqquery) with question_only context mode is the best configuration. The bridge should know what's being asked before extracting visual evidence.

3. **The VM matters more than the bridge.** The single largest accuracy jump in the project came from switching VMs (MobileViT → DINOv2 → SigLIP), not from bridge innovations. Bridge config changes (query mode, adapter depth, perceiver depth) produce ±0.005 deltas. VM changes produce ±0.03–0.05 deltas.

4. **Language-aligned VMs are better for VQA.** SigLIP-B beats DINOv2-B at matched capacity with fewer tokens. The perceiver extracts more useful information from features that already encode language-relevant concepts.

5. **The LM is probably the next bottleneck.** The ~27M LM has been held constant throughout. As the bridge improves, the question becomes whether the LM can reason well enough to exploit better visual evidence. The Ironclad oracle experiments (doc 39) will test this directly.

### What this means for the project's direction

The bridge architecture is approaching maturity for this LM scale. There are still potential gains from training methodology (Ironclad Tier 1: contrastive aux, curriculum, answer-type head) and from bridge primitives (Ironclad Tier 2: deformable attention, more queries). But the 80/20 story is clear: **the project has been VM-limited more than bridge-limited, and will soon be LM-limited.**

This motivates two major direction shifts.

## Direction 1: Custom Vision Transformer

### The problem with off-the-shelf VMs

Every VM we've tested is either:
- **Too small and wrong pre-training** (MobileViT: 5.6M, supervised ImageNet — low ceiling)
- **Right pre-training but wrong shape** (MobileCLIP: 11.4M, CLIP — only 49 tokens)
- **Right pre-training and enough tokens but too big** (SigLIP-B: 86M, DINOv2-B: 86M — frozen 86M params is a lot of dead weight in an 80M total-param model)

The ideal VM for this project would be:
- **10–25M params** (similar to DINOv2-S/MobileCLIP, keeps total model well under 100M)
- **Language-aligned pre-training** (SigLIP/CLIP-style, proven to produce better VQA features)
- **High token count** (196+ tokens from a ViT patch grid, not collapsed to 49)
- **384-dim features** (matches the bridge projection without wasteful 768→512 downprojection)

This VM does not exist as a published checkpoint. It needs to be trained.

### What a custom ViT would look like

A ViT-S/16 architecture with CLIP-style contrastive pre-training:
- 12 layers, 6 heads, d=384, MLP ratio=4 → ~22M params
- Patch size 16 at 224×224 → 196 tokens (14×14 grid)
- Pre-trained with sigmoid contrastive loss (SigLIP-style) on a curated image-text corpus
- No CLS token (mean pooling, following SigLIP)

This gives the project a VM that is:
- Capacity-matched to DINOv2-S (both ~22M) for clean comparisons
- Language-aligned like SigLIP-B (but 4x smaller)
- High-token like DINOv2 (196 tokens at native resolution)
- Natively 384-dim (no wasteful projection)

### Pre-training considerations

Training a ViT-S from scratch with contrastive learning is feasible but non-trivial:
- **Data:** CC3M + CC12M + a curated subset of LAION (filtered for quality). Total ~15-20M image-text pairs. Need to handle licensing.
- **Compute:** ViT-S contrastive training converges in ~30-50 GPU-hours on a single A100. On a 16GB consumer GPU, this is 2-4x longer due to smaller batches and no TF32 benefit. Realistic estimate: **60-100 GPU-hours** (~3-4 days continuous, or ~1 week with interruptions).
- **Text encoder:** Use a frozen text encoder (e.g., the text tower from a published SigLIP or CLIP model) as the teacher side of the contrastive pair. Only the vision tower is trained from scratch.
- **Modern training strategies:** Patch dropout (MAE-style), progressive resolution, EMA, gradient checkpointing. These are well-documented for ViT-S scale.

### The payoff

If a custom ViT-S/16 with CLIP-style pre-training achieves per-token quality comparable to SigLIP-B (which is plausible — the pre-training objective matters more than model size for feature quality at sufficient training), the bridge would get:
- SigLIP-quality features at 196 tokens
- 22M frozen params instead of 86M
- 384-dim native features (no downprojection)
- Room to increase total model params (bigger LM or unfreezing VM layers)

This is the "step back to step forward" move. It delays the next VQA accuracy push by 1-2 weeks of VM pre-training, but it removes the fundamental constraint that has shaped the entire project: relying on off-the-shelf VMs that are never the right size.

### Relationship to the datasetting task

The new `tasks/datasetting` task is directly relevant here. The pre-training corpus for a custom ViT determines its feature quality. If the datasetting task produces a high-quality, well-filtered image-text corpus, that corpus serves double duty:
1. ViT pre-training data
2. Bridge caption-align / auxiliary training data (if caption-align is ever revisited with a better approach)

These two tasks should be planned jointly.

## Direction 2: Multi-Benchmark Target

### Moving beyond VQAv2-only

The project has been optimized for VQAv2 accuracy. This is a useful forcing function — it's well-understood, has an official scorer, and rewards genuine visual understanding. But a single-benchmark focus creates blind spots:
- VQAv2 has known language biases (question-only baselines score ~42%)
- The "yes/no" category is disproportionately easy and inflates overall scores
- Counting is systematically underweighted
- Spatial reasoning is barely tested
- No robustness or compositionality measurement

### Recommended benchmark suite

For a from-scratch multi-benchmark VLM, the evaluation should include:

| Benchmark | Tests | Why it matters |
|---|---|---|
| VQAv2 | General VQA (legacy comparability) | The project's history lives here |
| VQA-CP v2 | VQA under shifted answer priors | Measures language bias exploitation |
| GQA | Compositional scene graph questions | Tests structured reasoning |
| TextVQA | OCR-dependent questions | Tests whether the VM extracts text |
| POPE / RePOPE | Object hallucination | Tests whether the model makes things up |
| SugarCrepe | Hard compositional negatives | Tests attribute/object/relation binding |
| VSR | Spatial relationship judgments | Tests spatial reasoning specifically |
| TallyQA | Counting (simple + complex) | Tests the weakest category directly |

Not all of these need to be primary optimization targets. VQAv2 + GQA + POPE + TallyQA would be a strong initial suite. The others serve as diagnostics.

### Architectural implications of multi-benchmark

Some benchmarks require capabilities the current architecture lacks:
- **TextVQA** needs OCR-quality features. Neither SigLIP nor DINOv2 are OCR-specialized. A custom ViT pre-trained with OCR-augmented data (e.g., including document images) would help.
- **GQA** rewards structured scene understanding. The bridge may need explicit spatial encoding beyond 2D positional embeddings.
- **TallyQA** counting requires attending to multiple instances. This is where the perceiver's global attention is actually an advantage over deformable attention (which focuses on K=4 local points per query).

These considerations should feed into the custom ViT design and the Ironclad diagnostic results.

## Direction 3: Ironclad — Completing Bridge Research

Before shifting to custom VMs and multi-benchmark evaluation, the Ironclad sweep (doc 39) should run to answer the question: **"Is the bridge architecture itself still improvable, or has it converged?"**

The key Ironclad deliverables:
1. Oracle diagnostics to locate the bottleneck (bridge vs LM vs VM)
2. Training methodology experiments (contrastive aux, TPCL, answer-type head)
3. Optional bridge architecture changes (deformable attention, query count)

If Ironclad shows the bridge is saturated — i.e., training tricks and architecture changes produce <0.005 gains — that's the definitive signal to cap off bridge research and redirect effort to VM pre-training and LM scaling.

If Ironclad shows the bridge still has headroom — i.e., contrastive aux or deformable attention gives +0.01 — then those improvements should be incorporated into the final bridge config before freezing it.

Either way, Ironclad produces a **terminal bridge configuration** that becomes the fixed architecture for all future work.

## Direction 4: LM Scaling (Deferred)

The current ~27M LM has been held constant throughout bridge research. At some point, the LM becomes the bottleneck — the bridge can provide excellent visual evidence but the LM can't reason over it well enough.

The Ironclad oracle experiments will partially test this (replacing the LM with a larger one to measure the reasoning gap). But full LM scaling is a separate project:
- Train a larger LM from scratch (55M? 110M?) on the existing wiki corpus
- Or adopt a pre-trained small LM (GPT-2 small at 124M, or a modern equivalent)
- Re-tune the bridge for the new LM (the perceiver output token count, adapter placement, and calibrator all need to be re-optimized)

This is explicitly deferred. LM scaling only makes sense after the bridge architecture is finalized and the VM is settled.

## Recommended Sequencing

```
NOW:
  ├── Complete Ironclad oracles + Tier 1           (1-2 weeks)
  ├── Begin datasetting task planning               (parallel)
  └── Begin custom ViT architecture design          (parallel)

AFTER IRONCLAD:
  ├── Write terminal bridge architecture doc         (caps bridge research)
  ├── Start ViT-S/16 contrastive pre-training        (1-2 weeks)
  └── Set up multi-benchmark eval harness            (parallel)

AFTER CUSTOM VM:
  ├── Bridge re-tune with custom ViT                 (1 week)
  ├── Multi-benchmark baseline evaluation            (2-3 days)
  └── Decide on LM scaling vs further VM work        (based on results)
```

## The Project Identity Shift

The project started as "VQA from scratch" with the implicit question "how good can a tiny model get on VQAv2?" That question is mostly answered: around 0.60–0.64 with an 80M-param model using a strong frozen VM and an optimized bridge.

The project is shifting toward: **"Can we build a competitive multi-benchmark VLM from scratch, including the vision encoder, with modern training strategies at sub-100M scale?"**

This is a harder and more interesting question. It requires:
- Owning the full stack (VM + bridge + LM), not just the bridge
- Training methodology that generalizes across benchmarks
- Data curation as a first-class research concern (hence the datasetting task)
- A clear parameter budget and efficiency target

The bridge research done so far is not wasted — the perceiver architecture, question conditioning, and adapter placement are all reusable knowledge. But the bridge is now one component of a larger system, not the entire research surface.

## Open Questions

1. **Can a 22M ViT-S with CLIP-style pre-training match SigLIP-B's per-token quality?** This is the central bet of the custom VM direction. SigLIP-B has 4x more params and was trained on WebLI (4B image-text pairs). A 22M model trained on 15-20M pairs might not reach the same feature quality. But it only needs to match per-token quality — the bridge handles the rest.

2. **Is the ~27M LM already the bottleneck?** If the Ironclad oracle shows that replacing the LM dramatically improves accuracy, then LM scaling should jump ahead of custom VM work in priority.

3. **What's the right pre-training corpus for a custom ViT?** CC3M+CC12M is the obvious starting point, but license cleanliness and data quality filtering matter enormously (per SmolVLM's finding that data quality >> quantity for small models).

4. **Should the custom ViT be trained with MAE-style self-supervised objectives in addition to contrastive learning?** DINOv2's spatial features are better for certain tasks (grounding, spatial reasoning). A hybrid objective (contrastive + masked reconstruction) might give the best of both.

5. **When do we cap off bridge research?** The answer depends on Ironclad results. If Ironclad Tier 1 produces meaningful gains, there may be one more methodology sweep worth running. If everything is flat, the bridge is done.

## One-Line Summary

The project is evolving from "optimize a bridge between frozen components" to "build a complete from-scratch VLM," and the next major moves are: finish Ironclad to cap off bridge research, train a custom language-aligned ViT-S that fits our parameter budget, expand to multi-benchmark evaluation, and defer LM scaling until the vision pipeline is settled.
