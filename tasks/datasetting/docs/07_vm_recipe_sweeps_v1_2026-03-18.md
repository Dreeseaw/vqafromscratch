# LIMBO Program V1 (2026-03-18)

## Purpose

This is the first real Part 1 program after widening the task from dataset-only work to full **VM pretraining recipe design**.

The name is `LIMBO`.

Reason:

- stage 1 hardens the VM under pure SSL pressure
- stage 2 hits it with both SSL and language-alignment pressure at once
- stage 3 leaves only language-alignment pressure

So the VM is not being politely tuned. It is being shaped under successive, deliberately different kinds of stress.

## Operating Style

`LIMBO` is not a giant unattended sweep.

It is a **serial manual program** built around the current one-family launcher:

- `tasks/datasetting/scripts/launch_vm_recipe_ratio_v1.sh`

That is the right operating mode for now because:

- each family costs about `~6 hours` minimum and often wants more
- the VM and MM curves are still climbing when we stop
- this task only gets a few real shots per day
- the user wants to inspect and intervene between runs

So `LIMBO` should be run one family at a time, with a decision after each family.

## Fixed Instrument

All families use the same downstream instrument:

- one `vm_*` recipe family
- one paired `mm_*` benchmark run
- final VQAv2 val from `mm_*` is the authoritative rank

The MM side stays fixed.

## Canonical VM Shape

The canonical VM recipe shape for `LIMBO` is:

1. **Stage 1**
   - SSL-only DINO
2. **Stage 2**
   - SSL + SigLIP-style image-text cross-training
   - same VM backbone
   - small from-scratch bidirectional text encoder
   - auxiliary DINO weight starts at `0.5`
3. **Stage 3**
   - SigLIP-only image-text alignment
   - same text encoder carried forward from stage 2

Rules:

- stage 1 always happens
- stage 2 or stage 3 may be `0`
- stage 2 and stage 3 may not both be `0`
- if stage 2 is `0`, stage 3 initializes its text encoder from scratch

## Default Starter Split

The clean starter split is:

- stage 1: `80`
- stage 2: `20`
- stage 3: `0`

Written as:

- `80/20/0`

This is the default `LIMBO` opening move, not a permanent truth.

## Main Question

The first serious question is no longer just:

> Which images help?

It is:

> Which compact three-stage VM recipe most efficiently creates bridge-useful visual tokens?

That question couples:

- image-only source mix
- image-text source mix
- stage allocation
- auxiliary SSL pressure during cross-training

So the first program should be small, explicit, and high-entropy.

## LIMBO V1 Families

`LIMBO V1` is a **three-family serial program**.

The goal is to learn from:

- one clean control
- one stronger OCR-aware data recipe
- one objective-tail variant

without exploding the search space.

### Family 1: `limbo_coco`

Purpose:

- clean three-stage control
- answer whether the current recipe surface itself already helps on plain COCO

Config:

- VM run:
  - `vm_recipev1_s1_80_s2_20_s3_0_limbo_coco`
- MM run:
  - `mm_recipev1_s1_80_s2_20_s3_0_limbo_coco`
- stage split:
  - `80/20/0`
- DINO image mix:
  - `coco_local:train2014 = 100`
- pair mix:
  - `coco_captions_2014:train2014 = 100`
- cross DINO weight schedule:
  - `0.5@0.0`

Why it exists:

- it upgrades the old COCO-only VM baseline into the current three-stage recipe surface
- it gives a clean control before adding OCR-heavy pressure

### Family 2: `limbo_ocrmix`

Purpose:

- test whether the OCR-heavy image regime plus OCR-aware caption pairs creates more bridge-useful visual tokens than plain COCO

Config:

- VM run:
  - `vm_recipev1_s1_80_s2_20_s3_0_limbo_ocrmix`
- MM run:
  - `mm_recipev1_s1_80_s2_20_s3_0_limbo_ocrmix`
- stage split:
  - `80/20/0`
- DINO image mix:
  - `coco_local:train2014 = 100`
  - `textocr:train = 100`
  - `coco_text:train = 100`
- pair mix:
  - `coco_captions_2014:train2014 = 100`
  - `coco_text_captions:train = 100`
- cross DINO weight schedule:
  - `0.5@0.0`

Why it exists:

- it is the cleanest currently-available mixed regime that pressures:
  - local text
  - text/object co-reference
  - tighter local evidence
- it re-tests the old `mixed1` idea, but now under the real recipe surface rather than a plain DINO-only pilot

### Family 3: `limbo_ocrtail`

Purpose:

- test whether a short pure alignment tail is worth paying for once the OCR-heavy mixed regime is alive

Config:

- VM run:
  - `vm_recipev1_s1_70_s2_20_s3_10_limbo_ocrtail`
- MM run:
  - `mm_recipev1_s1_70_s2_20_s3_10_limbo_ocrtail`
- stage split:
  - `70/20/10`
- DINO image mix:
  - `coco_local:train2014 = 100`
  - `textocr:train = 100`
  - `coco_text:train = 100`
- pair mix:
  - `coco_captions_2014:train2014 = 100`
  - `coco_text_captions:train = 100`
- cross DINO weight schedule:
  - `0.5@0.0`

Why it exists:

- it is the smallest clean test of whether stage 3 matters at all
- it does not open a new source axis
- it asks whether the VM benefits from a final period of pure alignment after the mixed-pressure cross phase

## Run Order

Run `LIMBO V1` in this order:

1. `limbo_coco`
2. `limbo_ocrmix`
3. `limbo_ocrtail`

Do not reorder them casually.

Reason:

- family 1 establishes the recipe-shape control
- family 2 tests the highest-value currently available data-pressure hypothesis
- family 3 only makes sense once family 2 tells us the OCR-heavy branch is still worth pushing

## Decision Gates

### After `limbo_coco`

Questions:

- does the three-stage recipe on plain COCO beat or at least match the old `vm_dinovit_v2 -> mm_dinovit_v2` floor?
- does the stage-2 cross phase look numerically healthy?

If the answer is no:

- fix recipe mechanics before spending more runs on source comparisons

### After `limbo_ocrmix`

Questions:

- does OCR-heavy pressure produce a visible downstream gain over `limbo_coco`?
- do we see a better final score, not just prettier internal losses?

If the answer is no:

- do not keep pouring runs into OCR-heavy variants
- next branch should likely move toward scenery/layout/street-view pressure instead

### After `limbo_ocrtail`

Questions:

- does a pure alignment tail help beyond `80/20/0` on the same data?
- does stage 3 improve final MM eval enough to justify the extra complexity and reduced stage-1 budget?

If the answer is no:

- keep `80/20/0` as the default shape
- stage 3 becomes a later specialist tool, not the default

If the answer is yes:

- promote `70/20/10` or another small-tail shape into the next wave

## Promotion Rule

Promote only if a family produces a **visible downstream win**.

For now, "visible" means:

- around `+0.004` final accuracy or better over the relevant control
- or a smaller gain that is clearly corroborated by checkpoint sanity and repeated behavior

Do not promote on:

- prettier SSL curves
- prettier alignment curves
- tiny `~0.001` wiggles with no real confidence

## Launcher Commands

### Family 1

```bash
FAMILY=limbo_coco \
DINO_DATASET_MIX='{"coco_local:train2014":100}' \
PAIR_MIX='{"coco_captions_2014:train2014":100}' \
STAGE1_RATIO_PERCENT=80 \
STAGE2_RATIO_PERCENT=20 \
STAGE3_RATIO_PERCENT=0 \
CROSS_DINO_WEIGHT_SCHEDULE='0.5@0.0' \
bash tasks/datasetting/scripts/launch_vm_recipe_ratio_v1.sh
```

### Family 2

```bash
FAMILY=limbo_ocrmix \
DINO_DATASET_MIX='{"coco_local:train2014":100,"textocr:train":100,"coco_text:train":100}' \
PAIR_MIX='{"coco_captions_2014:train2014":100,"coco_text_captions:train":100}' \
STAGE1_RATIO_PERCENT=80 \
STAGE2_RATIO_PERCENT=20 \
STAGE3_RATIO_PERCENT=0 \
CROSS_DINO_WEIGHT_SCHEDULE='0.5@0.0' \
bash tasks/datasetting/scripts/launch_vm_recipe_ratio_v1.sh
```

### Family 3

```bash
FAMILY=limbo_ocrtail \
DINO_DATASET_MIX='{"coco_local:train2014":100,"textocr:train":100,"coco_text:train":100}' \
PAIR_MIX='{"coco_captions_2014:train2014":100,"coco_text_captions:train":100}' \
STAGE1_RATIO_PERCENT=70 \
STAGE2_RATIO_PERCENT=20 \
STAGE3_RATIO_PERCENT=10 \
CROSS_DINO_WEIGHT_SCHEDULE='0.5@0.0' \
bash tasks/datasetting/scripts/launch_vm_recipe_ratio_v1.sh
```

## What `LIMBO` Is Not Testing Yet

`LIMBO V1` deliberately does **not** try to answer:

- scenery/layout gap sources like `ADE20K` / `Places365`
- broader street-view / storefront sources
- more aggressive stage-2 DINO-weight schedules like `0.5@0.0,0.1@0.5`
- text-tower size sweeps
- checkpoint-selection micro-policy

Those are real next-wave questions, but they are not the right first three shots.

## Working Conclusion

`LIMBO` is the right first program because it respects the real throughput constraint and the new theory of the task.

It uses one simple launcher, one family at a time, and asks three concrete questions:

1. does the new three-stage recipe shape work cleanly at all?
2. does OCR-heavy data pressure help under that shape?
3. does a pure alignment tail help enough to become part of the default recipe?
