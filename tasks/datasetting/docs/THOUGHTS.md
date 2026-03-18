# Datasetting Thoughts

This file is the scratchpad for raw notes, links, hunches, and half-formed ideas.

Use this for:
- candidate datasets
- possible experiment directions
- qualitative project pivots
- "this might matter later" notes

Do not treat this as canonical state. Promote distilled conclusions into `DATASETTING_STATE.md` only after they survive some scrutiny.

---

## 2026-03-16

### Project Direction Shift

- The project is now more explicitly about building a **param-efficient VLM**, not just running open-ended dataset research.
- Strong current belief: the self-trained VM is the weakest part of the stack by a wide margin.
- Immediate implication: datasetting should not be only "better LM text mix" work. It should explicitly support **vision model improvement** too.

### Two-Part Datasetting Structure

- Part 1 should focus on **vision-model datasets**:
  - stronger ViT-style visual backbones
  - more text-aligned vision training
  - same general experiment loop, but the VM is the component under study
- Part 2 should focus on **LM datasets**:
  - best available VM held fixed
  - LM corpus/mix/filtering becomes the primary variable
- There is heavy overlap between the two parts, so the source inventory / filtering / synthetic pipelines should stay shared when possible.

### Part 1 Starting Baseline

- Starting baseline for the VM-focused track is the current **basic VQA dataset/captions** setup.
- This should act as the floor before adding richer image-text sources or more text-aligned visual training data.

### Candidate Dataset: Molmo2 Multi-Image Pointing

- Candidate source: `https://huggingface.co/datasets/allenai/Molmo2-MultiImagePoint`
- Initial thought: this looks more useful as **downstream VLM grounding/alignment data** than as a core pretraining source.
- Likely best fit:
  - Part 1 first
  - later-stage or targeted add-on
  - useful for reference grounding / multi-image alignment / visually anchored pointing behavior
- Tentative classification:
  - good deferred candidate
  - probably not part of the earliest baseline mix

### Current Meta Rule

- Raw ideas go here first.
- Only move things into `DATASETTING_STATE.md` once they are either:
  - active priorities
  - settled beliefs
  - clearly tracked deferred candidates
