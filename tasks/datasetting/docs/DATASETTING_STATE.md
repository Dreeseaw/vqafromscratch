# Datasetting Research State

## Current Frontier Mix

- Part 1 VM benchmark methodology is now established as a **three-stage VM recipe plus paired MM benchmark**:
  - Stage 1: SSL-only DINO on image data
  - Stage 2: SSL + SigLIP-style image-text cross-training with a small from-scratch bidirectional text encoder
  - Stage 3: SigLIP-only alignment on the same text encoder
  - Stage 4: plug the resulting VM checkpoint into the fixed multimodal evaluation chassis (`mm_*`)
- The frontier object is no longer just "dataset". It is the **VM pretraining recipe**:
  - image-only corpora
  - image-text corpora
  - objective family
  - objective schedule
  - checkpoint policy
- The current main entrypoint is deliberately **one family at a time**:
  - `tasks/datasetting/scripts/launch_vm_recipe_ratio_v1.sh`
  - broad unattended sweep launchers are deferred
- Current benchmarked image mixes:
  - `coco_local:train2014` only
  - `coco_local:train2014 + coco_text:train + textocr:train`
  - `coco_local:train2014 + coco_text:train + textocr:train + inat2021:train_mini(10%)`
- No clear frontier winner yet. The mixed OCR-heavy image recipe was essentially flat versus the COCO-only baseline on the first paired test.
- `mixed2` is slightly below baseline on first pass (`0.5055` final), but the pilot runs are still confounded by unequal VM epoch budgets. The first formal sweep should correct that by using matched total image exposure.

## Baseline

For the Part 1 VM track, the current baseline is:

- `vm_dinovit_v2` trained on `coco_local:train2014` only
- evaluated by `mm_dinovit_v2` using the fixed MM bridge chassis and `logs/vm_dinovit_v2/epoch_100.tar`
- final VQAv2 val accuracy: `0.5077`

## Settled Findings

- VM pretraining-recipe experiments should be evaluated with a **paired VM→MM protocol**, not by inspecting SSL loss alone.
- The MM-side evaluation chassis should stay fixed while comparing VM dataset recipes:
  - `perceiver_resampler`
  - `question_hidden_attn`
  - `question_only`
  - LM visual adapters on
  - `9000` MM training steps
  - official VQAv2 val scoring
- The current naming convention is good and should be preserved:
  - `vm_<family>` for the SSL VM training run
  - `mm_<family>` for the downstream bridge/VQA evaluation using that VM
- Data and objective are coupled enough that Part 1 should be allowed to study both together:
  - `SSL-only`
  - `SSL + SigLIP-style cross-training`
  - `SigLIP-only` alignment tails
  - later image-text objectives if they can be compared under the same fixed downstream eval
- The current default starter recipe shape is:
  - `80/20/0`
  - stage 2 auxiliary DINO weight starts at `0.5`
- The first OCR-heavy mixed recipe did **not** produce a meaningful gain over the COCO-only baseline:
  - `mm_dinovit_v2` final eval: `0.5077`
  - `mm_dinovit_mixed1` final eval: `0.5080`
  - interpretation: effectively flat for decision-making purposes
- The first OCR + iNat pilot also did not improve over baseline:
  - `mm_dinovit_mixed2` final eval: `0.5055`
  - interpretation: adding `inat2021` at `10%` is not an immediate win under the current pilot setup

## Dead Ends

(None yet.)

## Current Bias

Use the paired `vm_*` / `mm_*` benchmark as the default Part 1 test methodology and keep the downstream MM side fixed while varying only the **VM pretraining recipe**:

- VM image source mix
- VM image-text source mix
- VM pretraining objective schedule
- checkpoint selection policy when there is evidence the terminal checkpoint is suboptimal

Near-term comparison policy:

- keep the MM-side chassis frozen
- prefer one primary VM recipe question at a time
- treat final MM eval as the authoritative ranking metric
- record both the VM stage run and the downstream MM stage run together as one experiment family
- enforce a **high-entropy run policy**:
  - VM runs cost about `~6 hours` at the current benchmark budget
  - practical throughput is only about `3` experiment families/day max
  - both VM and MM curves are still climbing at stop time
  - therefore early experiments must be decisive source screens, not broad low-information grids
- next formal program: `LIMBO`
  - serial, manual, one-family-at-a-time
  - `limbo_coco`
  - `limbo_ocrmix`
  - `limbo_ocrtail`
  - first test the three-stage shape itself, then OCR-heavy pressure, then a pure alignment tail
- `GQA` should stay prominent in evaluation and later multimodal supervision, but it is **not** currently the preferred first new VM SSL source:
  - its main value is reasoning structure
  - the more urgent SSL gap is missing pixel distribution
- the strongest likely missing early image gaps are:
  - scene-centric / layout-heavy imagery
  - cluttered relation-rich scenes
  - broader street-sign/storefront OCR-in-context imagery
- if only one foundational source is added before the wider mixing/filtering phase, prefer a dataset that expands one of those pixel regimes directly rather than adding `GQA` first as raw VM pretraining data

## Live-but-Deferred Ideas

- Nemotron-CC v2.1 as primary text source (blocked on access approval)
- Code inclusion in text mix (deferred to Stage 5)
- Multimodal quality classifier (deferred to Stage 2 — may not be needed)
- SynthVLM-style synthetic image generation (deferred to Stage 4)
- Bridge pretraining on caption data (interface with mm_bridge task)
- Bridge-style semi-automation for VM dataset experiments:
  - chain `vm_*` training into `mm_*` evaluation
  - handle the internal three-stage VM shape explicitly instead of pretending these are single-stage runs
