# Datasetting Manifesto

**Date:** 2026-03-15
**Scope:** Design a training corpus for a ~270M parameter VLM targeting strong score/param and score/FLOP on VQAv2, GQA, TextVQA, ChartQA, Winoground, and POPE.

---

## 1. Core Thesis

Small models are *disproportionately sensitive* to data quality. Every wasted token costs more at 270M params than at 7B. The goal is not "more data" but "denser signal per token."

But we do not *know* which data is densest for our model. The manifesto's job is not to prescribe a recipe — it is to define a **research methodology** for discovering the right dataset through controlled experimentation, semantic tracking of findings, and progressive refinement.

---

## 2. Constraints

| Resource | Budget |
|----------|--------|
| Parameters | ~270M |
| Text pretraining tokens | 5–6B |
| Total project storage | ~1 TB (shared across all tasks: datasetting, mm_bridge, models, logs) |
| GPU | 16 GB RTX 5080 (shared with mm_bridge) |
| Distillation models | >= 1B params provides solid annotation value (FineWeb finding) — Qwen2.5-1.5B/3B via Ollama on-device |

**Licensing model: Separable Tiers**

Not a single licensing constraint — a two-tier system:

| Tier | License | Purpose |
|------|---------|---------|
| **Core (commercial-safe)** | CC0 / CC-BY / MIT / Apache 2.0 | The fallback dataset. A model trained only on this tier is fully commercializable. |
| **Research boost** | Research-only, GPT-distilled, etc. | Separable add-on. Used during research to measure ceiling. Can be swapped out before any commercialization. |

All data must be tagged with its tier. Training configs must record which tiers were active. A model trained on Core-only must remain competitive — if the Research tier is doing all the work, that's a red flag, not a success.

---

## 3. The Experiment Loop — Research Methodology

### 3a. The Evaluation Instrument

Every dataset experiment runs through the **same fixed model** to isolate data effects from architecture effects:

- **Architecture:** lm_final + current best bridge (from mm_bridge frontier)
- **Vision backbone:** MobileViT (frozen, from mm_bridge standard config)
- **Training budget per experiment:** Fixed and small — enough to measure signal, not enough to fully converge. The exact budget is calibrated in the first experiment (see §3d).

The evaluation instrument changes **only** when mm_bridge advances its frontier and we explicitly re-baseline.

### 3b. The Hypothesis Loop

Each experiment follows this cycle:

```
    ┌─ Hypothesis: "Property X of the data mix improves benchmark Y"
    │
    ├─ Design: Fix everything except X. Carry forward prior-best mix as control.
    │
    ├─ Run: Short training on fixed model. 2–4 variants + 1 control.
    │
    ├─ Rank: Score all 6 benchmarks. Break down by benchmark type.
    │
    ├─ Combination Evidence: Does X stack with prior improvements?
    │         Does X help across benchmarks or only on one?
    │
    ├─ Mechanism: WHY did it help? Which benchmark categories moved?
    │         What does this imply about what the model is learning?
    │
    ├─ Reliability Audit: Were runs comparable? Same budget, same eval?
    │         Any confounds (data leakage, sampling bias)?
    │
    ├─ Takeaways: 2–3 sentences. What's settled. What's opened.
    │
    ├─ Update DATASETTING_STATE.md: New frontier mix. Updated bias.
    │
    └─ Design next experiment: What axis does this evidence point toward?
```

**Discipline rules:**
- **One axis per experiment.** Don't change source AND filtering AND mix ratio simultaneously.
- **Always carry forward control.** The prior-best mix runs alongside every new variant. No re-baselining without explicit justification.
- **Break down by benchmark.** A +0.02 overall that comes entirely from VQAv2 (which we train on directly) means something different from +0.02 spread across GQA and ChartQA.
- **Record cost.** Wall-clock time, GPU hours, storage used. An improvement that 3x's the data pipeline cost needs to justify itself.
- **Defer speculation.** List ideas that emerge but don't pursue them in the same experiment. They become candidates for the next one.

### 3c. Semantic Progression — The Axes

Experiments are organized into progressive **stages**. Each stage builds on settled findings from the prior stage. You don't optimize mix ratios until you know which sources are good. You don't optimize curriculum until you have a good mix.

**Stage 1: Source Selection**
- Which raw sources give the best signal per token? (PixelProse vs CC3M re-captioned vs raw alt-text)
- What's the floor — how does the current data (VQAv2-only, COCO captions) perform?
- What's the ceiling — what do the best available sources achieve?

**Stage 2: Quality Filtering**
- Does FineWeb-style educational quality scoring help for multimodal data?
- How aggressive should filtering be? (top 5% vs 10% vs 25% vs unfiltered)
- SoftDedup vs hard dedup vs none?
- CLIP score threshold for image-text alignment?

**Stage 3: Mix Composition**
- Optimal ratio of captioning vs VQA vs text-only?
- Per-source caps: 100K vs 250K vs uncapped? (Cambrian finding: 250K–350K cap)
- Domain diversity: does adding charts, documents, natural images help or hurt at small scale?
- Core-tier-only vs Core+Research: how much does the research tier add?

**Stage 4: Synthetic Augmentation**
- Does synthetic VQA data add signal beyond real data, or is it redundant?
- Re-captioning quality: 1.5B model vs 3B model?
- Hard-negative mining for Winoground: how much is enough?
- Anti-hallucination data for POPE: preference tuning vs simple filtering?

**Stage 5: Curriculum & Training Tactics**
- Does order matter? (broad → narrow vs narrow → broad vs random)
- Multi-phase vs single-phase training?
- Annealing into high-quality data in final 20%?
- Interleaving text-only and multimodal data vs separate phases?
- Code inclusion: does structured code data help ChartQA/structured reasoning?

Each stage produces a **stage report** (numbered doc, like mm_bridge sweeps) and updates the frontier mix in DATASETTING_STATE.md.

### 3d. Calibration Experiment (Experiment 0)

Before any dataset research, we need to calibrate the evaluation instrument:

1. Train the fixed model on VQAv2-only (our existing data) at 3 different budgets: 500K samples, 1M samples, 2M samples
2. Eval all 6 benchmarks at each budget
3. Identify: at what budget does signal stabilize? (i.e., where do relative rankings stop changing)
4. That budget becomes the **standard experiment budget** for all subsequent runs
5. This also establishes the **baseline scores** — the floor that every dataset variant must beat

This is analogous to mm_bridge establishing its eval policy and comparison standard before running sweeps.

---

## 4. The Source Inventory

### 4a. Core Tier (Commercial-Safe)

| Source | License | Type | Size Available | Estimated Storage | Status |
|--------|---------|------|----------------|-------------------|--------|
| **PixelProse** | CC-BY-4.0 | Image-caption pairs (dense, Gemini-generated) | 16.9M pairs | ~80 GB subset | Need to download |
| **FineWeb-Edu** | CC-BY-4.0 | High-quality web text (educational score >= 3) | 1.3T tokens | ~5 GB for 500M tokens | Accessible |
| **VQAv2 annotations** | CC-BY-4.0 | VQA pairs on COCO images | ~440K | ~25 GB (existing) | Already have |
| **GQA** | CC-BY-4.0 | Compositional VQA (scene graphs) | ~22M Q&A | ~20 GB | Need to download |
| **Docmatix** | MIT | Document/chart VQA from PDFs | 2.4M images, 9.5M Q&A | ~50 GB subset | Need to subsample |
| **RLAIF-V** | CC-BY-4.0 | Preference pairs for anti-hallucination | ~83K | ~1 GB | Need to download |
| **Self-generated synthetic** | Ours | Charts, hard-negatives, distilled QA | Variable | ~10–20 GB | To build |
| **COCO images** | Flickr per-image | Natural images | ~330K | ~25 GB (existing) | Already have |

**Nemotron-CC v2.1:** Access not yet approved. Would be primary text source (organic High/Medium-High tiers). Until approved, FineWeb-Edu is the text pretraining source.

### 4b. Research Tier (Separable, Non-Commercial)

| Source | License | Type | Size | Why include |
|--------|---------|------|------|-------------|
| **LLaVA-Instruct-150K/665K** | Research only (GPT-4 generated) | Multimodal instruction data | 150K–665K | Gold-standard instruction format. Ceiling reference. |
| **ShareGPT4V** | Research only | Dense image descriptions + VQA | ~100K | Very high quality captions. Tests ceiling of caption density. |
| **ALLaVA** | Research only (GPT-4V) | Fine-grained annotations + reasoning VQA | ~1.3M | Shown to match 7B models at 4B scale with this data alone. |

These are labeled, tracked separately, and used to measure how much the Core tier loses vs. a no-holds-barred data mix. If the gap is small, Core is strong enough. If the gap is large, we need to invest more in Core-tier synthetic generation.

### 4c. Storage Budget

| Category | Estimated Size |
|----------|---------------|
| Existing (COCO, VQAv2, models, logs) | ~200 GB |
| PixelProse subset (1M images + captions) | ~80 GB |
| FineWeb-Edu (500M–1B tokens) | ~5–10 GB |
| GQA | ~20 GB |
| Docmatix (subsampled ~200K images) | ~50 GB |
| Research tier datasets | ~30 GB |
| Synthetic pipelines output | ~20 GB |
| Experiment checkpoints & ablation data | ~100 GB |
| **Headroom** | **~490 GB** |
| **Total** | **~1 TB** |

---

## 5. Quality Filtering — The Classifier Stack

Adapting FineWeb-Edu's approach to our scale. Key insight from FineWeb: **any model >= 1B params provides solid distillation/annotation value.** We don't need a 70B model — Qwen2.5-1.5B or 3B via Ollama on the 5080 is sufficient.

### 5a. Text Quality Classifier

1. **Annotate** ~10K–50K text samples with informational quality scores (0–5 scale) using Qwen2.5-3B via Ollama
2. **Train classifier** on Snowflake-arctic-embed-m embeddings + linear regression head (~minutes on CPU)
3. **Score** all candidate text data
4. **Threshold sweep** — test score >= 2, 3, 4 as separate experiment variants in Stage 2
5. **SoftDedup** — assign lower sampling weights to high-commonness text

### 5b. Image-Text Quality Signals

- **CLIP score** — alignment between image and caption. Threshold TBD via experiment.
- **Caption length/density** — longer, more specific captions tend to carry more signal
- **Aesthetic score** — low-quality images add noise. Use LAION aesthetic predictor or similar.
- **Perceptual dedup** — remove near-duplicate images (pHash or CLIP embedding clustering)

### 5c. Multimodal Quality Classifier (Experiment-Driven)

Whether to build a multimodal quality classifier (scoring image-text *pairs* jointly) is itself an experiment question. It might not be worth the effort if CLIP score + text quality independently are sufficient. Test in Stage 2.

---

## 6. Synthetic Data Pipelines

All synthetic generation uses Ollama on-device with >= 1B models. No cloud compute required.

### 6a. Caption Re-generation
- Take PixelProse / CC3M images, re-caption with open VLM (Qwen2-VL or InternVL2)
- Produce both short (1-sentence) and detailed (3–5 sentence) captions per image
- **Experiment question (Stage 4):** Does re-captioning with a 3B model beat PixelProse's original Gemini captions? If not, skip this pipeline.

### 6b. VQA Pair Generation
- Given (image, detailed_caption), use Qwen2.5-3B to generate 3–5 QA pairs
- Filter: answers must be grounded in the caption (substring or paraphrase match)
- Existing `distill_qa_ollama.py` pipeline can be adapted

### 6c. Chart/Document Synthesis
- matplotlib/plotly charts from public datasets (UCI, government stats)
- Programmatic QA pairs about chart content
- Covers ChartQA benchmark gap with zero license risk

### 6d. Compositional Hard-Negatives (Winoground)
- Take (image, caption) pairs, generate adversarial variants:
  - Swap subject/object, change attributes, alter spatial relationships
  - Binary discrimination task: "Does this image match caption A or B?"

### 6e. Anti-Hallucination Data (POPE)
- RLAIF-V approach: generate paired (correct, hallucinated) answers
- Train with DPO/preference loss
- **Experiment question (Stage 4):** Is RLAIF-V preference tuning better than simply including more grounded VQA data?

---

## 7. Tracking Infrastructure

### 7a. DATASETTING_STATE.md

Analogous to mm_bridge's AUTORESEARCH_STATE.md. Contains:
- **Current frontier mix:** The best-performing data configuration found so far, with scores.
- **Settled findings:** What's been tested and conclusively shown to help/hurt.
- **Dead ends:** Axes that didn't pan out (and why — so we don't revisit them).
- **Current bias:** What the evidence points toward next.
- **Live-but-deferred ideas:** Things worth testing later but not now.

Updated after every experiment.

### 7b. Experiment Reports (Numbered Docs)

Each experiment gets a doc in `tasks/datasetting/docs/`:

```
NN_<experiment_name>_<date>.md
```

Structure:
1. **Hypothesis** — What are we testing and why?
2. **Setup** — What varied, what was held fixed, exact data configs
3. **Results** — Ranking table across all 6 benchmarks + breakdown
4. **Combination evidence** — Does this stack with prior improvements?
5. **Mechanism** — Which benchmarks moved and what does that imply?
6. **Reliability** — Any confounds, caveats, or comparability issues?
7. **Takeaways** — 2–3 sentences. Settled vs. opened.

### 7c. Data Mix Versioning

Each experiment's data configuration is recorded as a reproducible spec:

```json
{
  "mix_id": "mix_003",
  "parent": "mix_002",
  "change": "added GQA 200K subset, capped VQAv2 at 250K",
  "tier": "core",
  "sources": {
    "vqav2": {"samples": 250000, "filter": "none"},
    "gqa": {"samples": 200000, "filter": "balanced_subset"},
    "pixelprose": {"samples": 500000, "filter": "quality_score >= 0.7"}
  },
  "total_samples": 950000,
  "storage_gb": 45
}
```

This makes every experiment reproducible and every comparison explicit about what changed.

---

## 8. Key Research Findings Informing This Plan

1. **FineWeb-Edu (HuggingFace, 2024):** Filtering 15T → 1.3T tokens (top 8%) gave +4 MMLU, +11 ARC at 1.82B scale. Any model >= 1B provides solid annotation value for quality scoring. Quality >> quantity for small models.

2. **SoftDedup (ACL 2024):** Soft deduplication (reweighting, not removing) achieves same perplexity in 26% fewer steps. Complementary to hard dedup.

3. **ALLaVA (2024):** 4B models with high-quality synthetic VQA data match 7B/13B models on 17 benchmarks. Data quality is the primary lever for small VLMs.

4. **SynthVLM (2024):** 100K high-quality synthetic image-text pairs outperform 558K real pairs for alignment pretraining.

5. **Cambrian-1 (NeurIPS 2024):** Per-source cap of 250K–350K samples prevents single-source domination.

6. **Bunny (2024):** Coreset selection (fewer duplicates, more informative) beats random sampling for alignment pretraining.

7. **Nemotron-CC (NVIDIA, 2024):** Ensembling 3 quality classifiers recovers 2.5x more high-quality data than any single classifier. Heuristic filters hurt at the top quality tier.

---

## 9. Initial Experiment Plan

This is not the final plan. This is the first few experiments derived from the methodology. The actual sequence will evolve based on findings.

**Experiment 0 — Calibration**
- Train fixed model on VQAv2-only at 500K / 1M / 2M samples
- Establish baseline scores and standard experiment budget
- Establish eval reliability (variance across seeds)

**Experiment 1 — Source Floor/Ceiling (Stage 1)**
- Compare: VQAv2-only vs VQAv2 + COCO captions vs VQAv2 + PixelProse subset
- Core tier only
- Measures: does richer captioning data lift non-VQAv2 benchmarks?

**Experiment 2 — Research Tier Delta (Stage 1)**
- Compare: best Core-tier mix from Exp 1 vs same + LLaVA-Instruct-150K (research tier)
- Measures: how large is the research-tier advantage? Calibrates how much synthetic effort is needed to close the gap.

**Experiment 3 — Quality Filtering (Stage 2)**
- Take best source mix, apply quality filtering at 3 thresholds
- Measures: is aggressive filtering worth it, or does our source selection already handle quality?

Subsequent experiments designed based on findings.

---

## 10. Open Questions (Resolved and Remaining)

| # | Question | Status |
|---|----------|--------|
| 1 | Nemotron-CC access | **Not approved.** FineWeb-Edu is primary text source until further notice. |
| 2 | Docmatix storage | **Resolved:** Subsample to ~200K images (~50 GB). Total project budget ~1 TB. |
| 3 | Distillation model size | **Resolved:** >= 1B is sufficient (FineWeb finding). Use Qwen2.5-1.5B/3B via Ollama. No cloud GPU needed. |
| 4 | COCO image licensing | Covered by separable tier model. COCO is in Core tier for annotations; Flickr image licenses are per-image but standard research use is accepted practice. |
| 5 | Code in text mix | **Deferred to Stage 5 experiment.** Evidence is mixed (SmolVLM uses 20% code; unclear if it helps at 270M scale). Will test as a curriculum question after mix composition is settled. |
