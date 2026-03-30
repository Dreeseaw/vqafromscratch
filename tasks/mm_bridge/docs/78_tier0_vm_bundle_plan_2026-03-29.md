# Tier 0 VM Frontier De-Risking Plan

Date: 2026-03-29

## Purpose

Run a clean overnight sweep to answer one question:

- is the current frontier mostly limited by the frozen VM, or is there still large headroom in bridge-stage training signals and light VM adaptation?

This is a Tier 0 pass, so the stack stays as close as possible to the proven Cement bridge recipe. The only staged interventions are:

1. frozen VM swap
2. Qwen answer-KD on the strongest new frozen VM
3. top-layer VM finetuning on the same strongest new VM
4. K=8 compression on the strongest frozen and strongest stacked lines if time allows
5. hard-capability / semantic-token evals on the strongest checkpoints

## Reference Choice

I am treating the classic Cement bridge stack as the fixed recipe for this bundle:

- bridge: `perceiver_resampler`
- query path: `question_hidden_attn`
- question context: `question_only`
- token selector: `none`
- LM-side adapters: same Cement residual cross-attn path
- LM checkpoint: `logs/lm_final/step_45000.tar`

Reason for this choice:

- the user asked for the Cement-style champion recipe as the fixed comparison surface
- the bundle itself explicitly tests KD as a separate intervention, so KD is not part of the frozen-VM baseline
- keeping the old LM fixed makes the “VM swap vs KD vs partial VM finetuning” attribution cleaner

External reference lines will still be mentioned in the final report:

- Cement anchor full eval: `0.6163`
- Cement mini-eval peak: `0.6203` at `s53 step_8000`
- old SigLIP Qwen-KD bridge: `0.6235`

## New VM Targets

Two new frozen VM baselines:

1. `ViT-B-16-SigLIP2`
2. `PE-Core-B-16`

Engineering choice:

- both are integrated through the existing OpenCLIP-style token wrapper path
- local checkpoints are materialized under `logs/hf_vision/`
- PE-Core emits a CLS token; the wrapper strips it so both new VMs present a clean `196 x 768` patch-token interface to the existing bridge

This keeps the bridge interface stable and preserves the same downstream geometry as much as possible.

## Run Order

Priority order for the overnight bundle:

1. download + verify the two new VMs
2. frozen `siglip2_b16` bridge
3. frozen `pe_core_b16` bridge
4. choose stronger frozen VM by peak mini-eval, then full-eval that peak checkpoint
5. K=8 compression on the stronger frozen VM
6. Qwen answer-KD bridge on the stronger frozen VM
7. top-layer VM finetune bridge on the stronger frozen VM
8. if time remains, K=8 compression on the stronger stacked line
9. hard-capability eval suite on:
   - Cement reference
   - best new frozen VM
   - KD-on-best-new-VM
   - finetuned-best-new-VM
   - best compressed new-VM line(s)

If one frozen VM is clearly losing, it will not receive KD or VM finetune follow-up.

## Runtime and Safety

Conservative defaults for plain bridge training:

- train batch: `96 x 2`
- eval batch: `128`
- workers: `2`
- prefetch: `1`
- `--no-pin_memory`

Conservative defaults for partial VM finetuning:

- start at `64 x 3`
- same safe loader profile
- VM LR scaled down relative to bridge LR

Compression:

- start at `96 x 2` on single-stream VMs
- if VRAM or host RAM looks unstable, fall back to `64 x 3`

Babysitting:

- strict foreground `sleep 60 -> check stuff! -> inspect`
- watch:
  - process liveness
  - latest logfile step
  - latest checkpoint
  - `steps_per_s`
  - GPU memory/utilization
  - host RAM staying safely below the WSL danger zone

## Peak Metric Policy

This bundle will not rank runs by terminal step alone.

For each training run:

- periodic mini-eval metrics are parsed from the logfile
- the peak periodic checkpoint is identified
- if the peak step is not already the terminal full-eval step, a single posthoc full eval is run on that peak checkpoint

That gives:

- peak mini-eval ranking for early frontier triage
- peak full-eval ranking for final comparison

## Hard-Eval Suite

The hard-eval / semantic-usefulness pass uses existing repo tools:

- VQAv2 full eval with category breakdown
- GQA exact-match with `spatial`, `attribute`, `exist`, `count`
- OCR subset analysis
- tiny-head probe as the semantic / retrieval usefulness readout of exported tokens

Interpretation target:

- VQAv2 answers “raw frontier”
- GQA answers “compositional / hard reasoning”
- OCR subset answers “text reading”
- probe answers “semantic token usefulness”

## Success Logic

The final report should cleanly answer:

- which new frozen VM is strongest
- whether KD still stacks on that stronger VM
- whether top-layer VM finetuning beats KD or merely matches it
- whether compression remains viable on the winning VM line
- whether the main frontier shift came from the VM, KD, or partial VM finetuning

## Deliverables

At the end of the overnight bundle:

- every stage run is registered in the experiment DB via a bundle timeline
- peak and full metrics are materialized in bundle JSON artifacts
- the final report ranks:
  - raw VQAv2 frontier
  - compressed K=8 frontier
  - OCR / hard reasoning
  - probe / semantic-token usefulness
  - single best line to continue tomorrow
