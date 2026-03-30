# Dual-VM Grounding + GQA Plan

Goal: test whether structured supervision changes what the dual-VM `K=8` bottleneck preserves.

Reference point:
- prior dual-VM `K=8` full eval: `0.6109`
- prior dual-VM `K=8` category split: `yes/no 0.7866 / number 0.4611 / other 0.5168`
- target: improve `other` without giving back the strong `yes/no` and `number` behavior

## Design

Two sequential runs:

1. Control
- same dual-VM `K=8` compression recipe as before
- fresh bottleneck init
- VQAv2-only
- no grounding loss

2. Grounding + GQA
- same frozen dual-VM upstream stack
- same fresh `K=8` bottleneck init
- dual-VM-specific remap teacher
- mixed training:
  - VQAv2
  - GQA train
  - pointing index supervision
- losses:
  - `L_vqa`
  - `0.1 * L_distill`
  - `L_format` annealed `0.3 -> 0.0`
  - `0.05 * L_ground`

## Engineering Notes

- GQA scoring uses exact-match, not the VQA official scorer.
- Dual-VM grounding supervision pads `196` SigLIP targets to the perceiver’s `392`-token attention length by zero-filling the ViTSTR half.
- For runtime simplicity, GQA evaluation is done posthoc on saved checkpoints rather than inside the training loop.
- Grounding comparison uses the pointing index as the available supervision-aligned proxy grounding set.

## Eval Bundle

For both runs:
- VQAv2 checkpoint trace every `500` steps
- GQA exact-match checkpoint trace every `500` steps
- full VQAv2 eval at best checkpoint
- full GQA exact-match eval at best checkpoint
- OCR subset analysis
- tiny-head probe
- grounding-target mass eval

## Decision Rule

Primary comparison is `grounding+GQA - control`.

Success tiers:
- strong: reaches or exceeds Cement-level compressed quality
- moderate: improves `other` and/or overall meaningfully over control
- weak: stays flat or regresses relative to the control
