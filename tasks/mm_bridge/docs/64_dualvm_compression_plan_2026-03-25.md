# Dual-VM + K=8 Compression Plan

## Goal

Test whether the OCR-aware dual-VM perceiver can survive late K=8 semantic compression.

Reference system:

- warm-start dual VM: [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmdualvm_v1_20260324_rerun/step_9000.tar)
- full score: `0.6307`
- OCR subset: `0.2930`

Reference compressed systems:

- Cement anchor: `0.6163`
- SigLIP-only K=8 format-aligned: `0.5900`

The key question is whether the OCR gain survives the bottleneck. Since the bottleneck sits after the perceiver, this is really a test of whether the dual-VM perceiver already baked OCR evidence into its 49 exported latents.

## Phase 1

Train a dual-VM-specific remap first.

Measurements:

1. dual-VM keep-3 full score
2. dual-VM keep-0 full score
3. dual-VM + remap keep-0 full score

Proceed gate:

- continue only if remap improves the keep-0 full score by at least `+0.01`

This is stricter than the quarter gate because the prompt explicitly frames “meaningful” as about a point.

## Phase 2

If Phase 1 clears the gate:

- start from the dual-VM warm-start checkpoint
- freeze both VMs, perceiver, LM
- disable adapters
- train only a fresh `K=8` semantic bottleneck
- use the dual-VM remap as a frozen format teacher

Train:

- `3000` steps
- effective batch `192`
- eval every `500`
- checkpoint every `500`

## Phase 3

At the best compressed checkpoint:

1. full-val eval
2. OCR subset eval via the existing dual-VM OCR script
3. tiny-head probe

## Success Read

- strong: compressed dual-VM beats SigLIP-only compressed (`> 0.5900`) and OCR subset stays above `0.27`
- moderate: overall near SigLIP-only compressed but OCR subset still improved
- weak: OCR subset falls back near Cement / overall underperforms the SigLIP-only compressed line
