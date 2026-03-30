# Prefix Remap Diagnostic Plan

## Goal

Test whether the `K=8` semantic bottleneck's keep-0 failure is mainly a linear format mismatch at the LM prefix interface.

Question:

`if a single learned linear remap is inserted on top of the frozen K=8 exported tokens, can the no-adapter system recover yes/no accuracy?`

This is a micro-diagnostic, not a new training regime.

## Fixed Reference

Checkpoint:
- `logs/mmsemantic_v1_20260322_k8/step_4000.tar`

Reference numbers:

| Condition | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| K=8 keep-3 | `0.6154` | `0.7611` | `0.4583` | `0.5462` |
| K=8 keep-0 | `0.3744` | `0.4510` | `0.2757` | `0.3423` |
| K=8 probe | `0.5031` | `0.6199` | `0.3511` | `0.4316` |

## Architecture

Frozen stack:

```text
SigLIP-B/16
  -> perceiver
  -> semantic bottleneck (K=8)
  -> prefix calibration
  -> PrefixRemap (new, trainable)
  -> LM prefix ingestion
```

Important:
- LM visual adapters are disabled entirely at forward time
- only the new linear remap is trainable

## Training Setup

- data: VQAv2 train
- trainable params: `PrefixRemap.proj` only
- batch size: `192`
- steps: `500`
- checkpoints: every `100`
- periodic eval: every `100` using the standard quick eval (`100` batches)
- final eval: standard final full eval at step `500`
- optimizer: AdamW
- LR: `1e-3`
- schedule: `constant`

## Decision Rule

Outcome A:
- yes/no recovers toward `0.60+`
- interpretation: mostly linear format mismatch

Outcome B:
- yes/no reaches roughly `0.50 - 0.55`
- interpretation: part format, part deeper routing

Outcome C:
- yes/no stays near the keep-0 baseline
- interpretation: adapters are doing real in-network routing/reasoning, not just format translation

## Implemented Support

Runtime changes:
- [mm.py](/home/wdree/percy/vqafromscratch/train/mm.py)

New scripts:
- [launch_prefix_remap_diagnostic_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_prefix_remap_diagnostic_v1.sh)
- [analyze_prefix_remap_diagnostic.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/analyze_prefix_remap_diagnostic.py)

## Command

```bash
./tasks/mm_bridge/scripts/launch_prefix_remap_diagnostic_v1.sh
```
