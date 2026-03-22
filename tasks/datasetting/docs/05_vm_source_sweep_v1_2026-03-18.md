# VM Source Sweep V1 (Historical / Superseded)

This document is a **historical DINO-only sweep plan** from before the current three-stage VM recipe surface existed.

It is still useful as a record of the earlier source-isolation thinking, but it is **not** the current Part 1 plan.

Current live plan:

- see `LIMBO Program V1`
- use the three-stage `launch_vm_recipe_ratio_v1.sh` entrypoint
- treat this source-only plan as archived context, not active instruction

## Historical Purpose

This is the first **formal** Part 1 sweep for the datasetting task.

Its job is not to search architecture. Its job is to answer a narrower and more important question:

> Which image-source families help a small self-trained ViT produce the **highest-value visual tokens** for the frozen bridge evaluation chassis?

The bridge only gets `49` visual tokens. So the dataset does not win by making the VM broadly "better" in an abstract sense. It wins by teaching the VM to compress the right evidence into the most transferable token geometry before projection and selection.

## Operational Note

This sweep is conceptually six families, but it should be **executed like a tournament**, not like a leisurely full-grid research job.

Reason:

- the VM stage costs roughly `~6 hours` at the current budget
- practical throughput is only about `3` experiment families/day max
- the VM and MM curves are still climbing when these runs stop

So the intended use is:

- launch a small decisive wave first
- inspect the result
- then decide which second-wave families are worth paying for

The point is to maximize information gained per day, not to mechanically exhaust every branch.

## Core Question

What kind of extra image data helps the VM most beyond plain COCO natural-image supervision?

The first candidate source families already in hand are:

- **OCR-heavy natural images**
  - `textocr:train`
  - `coco_text:train`
- **Long-tail natural detail / biodiversity**
  - `inat2021:train_mini`
- **Baseline natural web-photo distribution**
  - `coco_local:train2014`

This sweep isolates those families before we add broader scenery/street sources later.

## Why This Sweep Exists

The pilot runs were useful, but they are not clean enough to anchor the whole task:

| Family | VM run | MM run | Mix | VM budget | Final MM acc |
|---|---|---|---|---:|---:|
| COCO baseline | `vm_dinovit_v2` | `mm_dinovit_v2` | `coco_local` only | 100 epochs | `0.5077` |
| OCR mix pilot | `vm_dinovit_mixed1` | `mm_dinovit_mixed1` | `coco_local + textocr + coco_text` | 75 epochs | `0.5080` |
| OCR + iNat pilot | `vm_dinovit_mixed2` | `mm_dinovit_mixed2` | `coco_local + textocr + coco_text + 10% iNat` | 40 epochs | `0.5055` |

These runs already tell us two useful things:

1. OCR-heavy data is not an automatic breakthrough.
2. Adding `inat2021` at `10%` did not produce an obvious gain in the first pass.

But they are still confounded by **different VM training budgets**. This sweep fixes that.

## Sweep Principle

The VM stage will be compared at a **matched total image-exposure budget**, not a matched epoch count.

That matters because the datasets have different sizes. If we hold epochs fixed, larger mixtures get many more total training examples than the control. If we hold image-exposure roughly fixed, we get a cleaner answer about **source quality per unit budget**.

### Target Exposure Budget

Use approximately **6.6M image presentations** per VM run.

That yields the following matched schedules:

| Family | Effective images | VM epochs | Total image presentations |
|---|---:|---:|---:|
| COCO only | `82,783` | `80` | `6.62M` |
| COCO + TextOCR | `101,377` | `65` | `6.59M` |
| COCO + COCO-Text | `98,954` | `67` | `6.63M` |
| COCO + both OCR sources | `117,548` | `56` | `6.58M` |
| COCO + iNat10 | `132,783` | `50` | `6.64M` |
| COCO + both OCR + iNat10 | `167,548` | `40` | `6.70M` |

This is the main methodological correction over the pilot work.

## Fixed Evaluation Instrument

### Stage A: VM Training

- DINO-style SSL
- ViT-S/16-ish small VM (`dim=192`, `depth=12`, `patch_size=16`)
- BF16
- batch size `128`
- same optimizer/schedule as current `train.dino_ssl`
- no architecture changes during this sweep

### Stage B: Downstream MM Benchmark

Hold the MM side fixed for all families:

- `runmm.sh`
- `9000` training steps
- official VQAv2 val scorer
- `perceiver_resampler`
- `question_hidden_attn`
- `question_only`
- LM visual adapters on
- `49` visual tokens
- seed `35`

The downstream score is the authority. SSL loss is only supporting evidence.

## Run Families

This sweep should use the following paired run ids.

### Family A: Control

- VM: `vm_dinovit_srcsweep1_cocoonly`
- MM: `mm_dinovit_srcsweep1_cocoonly`
- VM data: `coco_local:train2014`
- Epochs: `80`

Purpose:

- clean matched-budget control
- replaces the older `vm_dinovit_v2` pilot as the formal reference point

### Family B: TextOCR only

- VM: `vm_dinovit_srcsweep1_textocr`
- MM: `mm_dinovit_srcsweep1_textocr`
- VM data:
  - `coco_local:train2014 = 100`
  - `textocr:train = 100`
- Epochs: `65`

Purpose:

- isolate scene-text supervision from the more generic COCO-Text source
- test whether explicit OCR-natural-image pressure helps token usefulness

### Family C: COCO-Text only

- VM: `vm_dinovit_srcsweep1_cocotext`
- MM: `mm_dinovit_srcsweep1_cocotext`
- VM data:
  - `coco_local:train2014 = 100`
  - `coco_text:train = 100`
- Epochs: `67`

Purpose:

- isolate the denser-but-smaller COCO-derived OCR source
- test whether this source alone is cleaner/more useful than TextOCR

### Family D: Both OCR sources

- VM: `vm_dinovit_srcsweep1_ocrboth`
- MM: `mm_dinovit_srcsweep1_ocrboth`
- VM data:
  - `coco_local:train2014 = 100`
  - `textocr:train = 100`
  - `coco_text:train = 100`
- Epochs: `56`

Purpose:

- clean matched-budget rerun of the OCR-heavy idea
- removes the old `mixed1` epoch-budget confound

### Family E: iNat only

- VM: `vm_dinovit_srcsweep1_inat10`
- MM: `mm_dinovit_srcsweep1_inat10`
- VM data:
  - `coco_local:train2014 = 100`
  - `inat2021:train_mini = 10`
- Epochs: `50`

Purpose:

- isolate long-tail natural detail / biodiversity without OCR
- test whether extra visual diversity helps token abstraction more than text-rich images do

### Family F: OCR + iNat

- VM: `vm_dinovit_srcsweep1_ocrboth_inat10`
- MM: `mm_dinovit_srcsweep1_ocrboth_inat10`
- VM data:
  - `coco_local:train2014 = 100`
  - `textocr:train = 100`
  - `coco_text:train = 100`
  - `inat2021:train_mini = 10`
- Epochs: `40`

Purpose:

- full currently-available mixed-source hypothesis
- formal matched-budget version of the existing `mixed2` idea

## What The Sweep Is Testing

There are really three hypotheses here:

### H1: OCR helps token abstraction

If `textocr`, `cocotext`, or `ocrboth` beats control, the small VM benefits from text-bearing natural scenes because those images pressure it to encode:

- localized symbolic cues
- fine edge detail
- object-text co-reference
- spatially precise evidence

Those are exactly the kinds of features that should survive projection and token selection well.

### H2: Long-tail natural diversity helps token abstraction

If `inat10` beats control, the VM benefits more from fine-grained natural diversity than from OCR pressure. That would suggest the bottleneck is not reading/local symbol grounding but richer object/detail abstractions.

### H3: The best result is a mix, not a pure source

If `ocrboth_inat10` wins, then the right answer is not "OCR" or "nature" alone but a corpus with complementary pressures:

- precise local symbolic evidence
- broad/fine-grained natural detail

That would be a promising direction for later adding scenery/street-view sources.

## Primary Metric

Primary ranking metric:

- **final VQAv2 val accuracy** from the paired `mm_*` run

Secondary metrics:

- periodic MM peak accuracy
- VM terminal loss
- VM terminal steps/s
- whether gains concentrate in likely visually grounded question types

## Decision Rules

### Clear win

A family is a real winner if it beats control by at least **`+0.004` final accuracy**.

Reason:

- `mixed1` versus the older baseline was only `+0.0003`
- that is too small to justify changing project direction
- the first formal sweep should only promote changes with visible signal

### Weak / ambiguous signal

If the best family is within `±0.003` of control:

- call the sweep inconclusive on source mix alone
- do **not** overfit to noise
- move the next sweep toward:
  - dataset balancing ratios
  - checkpoint selection policy
  - broader scenery/street-view source additions

### Interpretation branches

If `textocr` or `cocotext` wins:

- prioritize OCR-natural-image additions
- later add street/sign/storefront data

If `inat10` wins:

- prioritize diversity/nature/scenery pressure
- later add Places365 / Open Images / Mapillary-scale sources

If `ocrboth_inat10` wins:

- treat the future corpus as a deliberately mixed pressure system
- continue with ratio tuning, not source elimination

If control wins:

- suspect that current DINO SSL recipe is the bottleneck more than the source mix
- shift the next sweep toward SSL recipe or checkpoint-policy changes

## Runtime / Cost

This sweep is not tiny, but it is still affordable because the model is small.

Approximate structure:

- `6` VM runs
- `6` MM runs
- sequential queue, restart-safe

The long pole is the VM stage. The `ocrboth_inat10` arm is already effectively done in pilot form (`vm_dinovit_mixed2` / `mm_dinovit_mixed2`), so one arm may be partially reusable for sanity checks even if the formal run ids are new.

## Why This Is The Right First Sweep

It does three necessary things at once:

1. It converts the current pilots into a **clean matched-budget comparison**.
2. It isolates the currently available source families instead of continuing to throw mixtures together.
3. It optimizes for the actual project question:
   what data teaches a small VM to produce the most useful visual tokens for the bridge?

That is the right first sweep before scaling to bigger image inventories or moving on to LM-side dataset work.
