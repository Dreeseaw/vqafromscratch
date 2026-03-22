# VM Recipe Engineering Status (2026-03-18)

## Purpose

Part 1 is no longer just "which image dataset helps the VM?".

The correct frontier object is now a **VM pretraining recipe**:

- image-only source mix
- image-text source mix
- objective family
- objective schedule
- checkpoint policy

This note defines the engineering surface needed to support that widened scope without turning the task into undisciplined VM tinkering.

## Guiding Constraint

VM runs are expensive:

- roughly `~6 hours` minimum at the current benchmark budget
- practical throughput is only about `3` experiment families/day max
- both VM and MM curves are still improving when the run stops

So the engineering goal is not maximum flexibility. It is to support a **small number of high-entropy, well-instrumented recipe comparisons**.

## Eng-1: Image-Text Pair Registry in DuckDB

### Purpose

The VM corpus can no longer be represented as only a table of images. We need a first-class layer for:

- captions
- dense narratives
- OCR-heavy captions
- later question/answer or grounding text if needed

### Requirements

- one canonical `image_text_pairs` table linked to `images.image_id`
- source-level metadata:
  - license
  - tier
  - local path
  - notes
- support multiple pair types:
  - `caption`
  - `narrative`
  - `qa`
  - `ocr_text` if ever needed
- easy SQL export of:
  - image-only samples
  - image-text samples
  - tier-filtered subsets
  - source-balanced subsets

### Immediate target sources

- `coco_captions_2014`
- `coco_text_captions`
- `TextCaps`
- `Flickr30k`
- later: `Localized Narratives (COCO-first)`

### Current status

This is now landed enough for real use:

- the DuckDB layer tracks first-class image-text pairs
- `coco_captions_2014` and `coco_text_captions` are registered and usable
- `TextCaps` and `Flickr30k` are on disk and partially integrated

This remains the base dependency for every later image-text phase.

## Eng-2: Recipe-Aware Data Loader Layer

### Purpose

The current VM path can no longer assume image-only SSL. The recipe surface needs a loader that can serve:

- image-only batches for DINO-style training
- image-text batches for SigLIP-style alignment
- mixed image-text batches that also expose DINO crops for cross-training

### Current surface

- `train/vm_recipe_data.py`
- SQL-driven subset selection from DuckDB
- recipe manifest inputs such as:
  - `image_sources`
  - `pair_sources`
  - `tier_mask`
  - `per_source_caps`
  - `sampling_weights`

### Output modes

- `image_only`
  - returns `image_tensor`
- `image_text`
  - returns `image_tensor`, `text`
- `image_text_cross`
  - returns `siglip_image`, DINO crops, and text for stage 2

This layer is now real enough to power the current launcher.

## Eng-3: Three-Stage VM Recipe Training

### Purpose

Add the actual three-stage VM-training surface that turns learned visual structure into more language-aligned visual tokens.

This is motivated by the `mm_bridge` finding that pretrained `SigLIP-B/16` was the strongest off-the-shelf VM tested there. The project should be allowed to ask whether a smaller self-trained VM benefits from a local image-text phase after DINO-style structure learning.

### Current implementation

Use separate entrypoints rather than mutating `train/dino_ssl.py` into a kitchen sink:

- stage 1: `rundino.sh`
- stage 2: `runvmcrosssiglip.sh`
- stage 3: `runvmsiglip.sh`
- shared trainer surface: `train/vm_siglip_align.py`
- text encoder: `models/vm_text_encoder.py`

Core recipe abstraction:

- `phase 1`: DINO-style image-only SSL
- `phase 2`: SSL + SigLIP-style image-text cross-training
- `phase 3`: SigLIP-style image-text alignment only

### Current properties

- small from-scratch bidirectional text encoder
- learnable visual projection head for the alignment stage
- sigmoid image-text contrastive objective
- configurable stage-2 auxiliary DINO schedule
- phase-boundary checkpointing:
  - end of DINO phase
  - periodic stage-2 / stage-3 checkpoints
  - final checkpoint
- recipe metadata written to logs:
  - image sources
  - pair sources
  - phase durations
  - phase start/end markers

The current default is deliberately simple:

- starter ratio: `80/20/0`
- stage-2 DINO weight schedule: `0.5@0.0`

The next likely schedule variant is:

- `0.5@0.0,0.1@0.5`

## Eng-4: Recipe Launcher

### Purpose

The launcher surface now needs to represent:

- VM stage 1
- VM stage 2
- VM stage 3
- paired MM eval

### Current launcher

- `tasks/datasetting/scripts/launch_vm_recipe_ratio_v1.sh`

### Current properties

- family manifest per run
- phase-resume support
- clear run naming:
  - `vm_recipev1_*`
  - `mm_recipev1_*`
- phase-boundary checkpoint tracking
- `timeline.log` entries for:
  - stage 1 start/end
  - stage 2 start/end
  - stage 3 start/end
  - MM start/end
- one-family-at-a-time operation

This one-family launcher is the correct main entrypoint for now. A broader sweep launcher can be added later if the operating style changes.

## Eng-5: Tracker / FE Support

### Purpose

The research tracker needs to understand that VM runs now have **internal phases**, not just one monolithic training trace.

### Needed tracker support

- run metadata fields:
  - `vmObjectiveFamily`
  - `vmPhaseSchedule`
  - `pairSources`
  - `phaseBoundaryCheckpoint`
- run-detail rendering:
  - phase timeline
  - DINO metrics
  - SigLIP alignment metrics
  - selected checkpoint policy
- family-level grouping:
  - VM recipe run
  - downstream MM eval run

## Eng-6: Remaining Gaps

### Purpose

Because these runs are underconverged, the terminal checkpoint is not guaranteed to be the best downstream VM.

Still worth building:

- stronger checkpoint-probe policy
- clearer family-level FE rendering for internal phases
- later, a real multi-family sweep launcher if the manual one-family mode becomes a bottleneck

## Recommended Build Order

1. finish saturating the pair registry with the downloaded caption sources
2. harden tracker/FE phase rendering
3. run the first real serial recipe program
4. only then widen the schedule search if the initial shape is alive

## Working Conclusion

The task should widen, but only in the way the theory demands.

The correct expansion is:

- from datasetting
- to **VM pretraining recipe design**

That means keeping image-text data infrastructure, a three-stage VM recipe surface, and one controlled new axis at a time.
