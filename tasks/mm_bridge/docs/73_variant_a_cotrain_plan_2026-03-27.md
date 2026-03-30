# 73 Variant A Co-Training Plan (2026-03-27)

## Goal

Test whether Variant A works better when it is **co-trained from bridge step zero** instead of added later on top of a frozen perceiver.

The architecture under test is:

```text
SigLIP grid [196, D]
  -> perceiver [49, D]
  -> Variant A bottleneck queries attend over [perceiver_out ; grid] = [245, D]
  -> export K=8 tokens to LM
```

## Stage 1: Fresh bridge training

Start from the normal pretrained LM checkpoint, not from a Cement MM checkpoint.

Trainable:
- perceiver / bridge
- Variant A bottleneck
- top `2` LM layers
- LM visual adapters

Frozen:
- SigLIP VM
- lower LM

Recipe:
- VQAv2 only
- `9000` steps
- Cement LR schedule: cosine, warmup `600`, peak `2e-4`
- conservative runtime profile:
  - train `64 x 3`
  - eval batch `128`
  - `num_workers=1`
  - `prefetch_factor=1`
  - `--no-pin_memory`

Important choice:
- use the **current** Cement recipe (`train_top_lm_layers=2`), not the older stale “top 3 layers” phrasing
- bridge-stage loss is pure `L_vqa`
- semantic recon / consistency / format losses are all disabled in stage 1

## Stage 2: Compression tuning

Start from the stage-1 bridge checkpoint.

Trainable:
- Variant A bottleneck only

Frozen:
- VM
- perceiver
- LM
- LM adapters disabled in forward

Recipe:
- VQAv2 only
- `3000` steps
- `L_vqa + 0.1 * L_distill + 0.3 * L_format`, with format annealed to `0` over steps `2250-3000`
- same SigLIP remap teacher as the earlier SigLIP compression line:
  - `logs/mmsemantic_remap_v1_debug/step_500.tar`
- conservative runtime profile:
  - train `64 x 3`
  - eval batch `96`
  - `num_workers=1`
  - `prefetch_factor=1`
  - `--no-pin_memory`

User clarification applied:
- **do not reinitialize the bottleneck at stage 2**
- stage 2 continues from the co-trained stage-1 Variant A bottleneck weights

## Comparisons

Primary bridge comparison:

| Condition | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| Cement anchor | 0.6163 | 0.7589 | 0.4573 | 0.5499 |
| Variant A co-trained bridge | ? | ? | ? | ? |

Primary compressed comparison:

| Condition | Overall | Yes/No | Number | Other | Probe |
|---|---:|---:|---:|---:|---:|
| SigLIP K=8 format-aligned | 0.5900 | 0.7393 | 0.4499 | 0.5134 | 0.5103 |
| Variant A K=8 frozen-perceiver | 0.5951 | 0.7499 | 0.4505 | 0.5157 | 0.5117 |
| Variant A K=8 co-trained | ? | ? | ? | ? | ? |

## Decision Rule

The key question is whether co-training closes the remaining gap between:
- the posthoc Variant A result (`0.5951`)
- and the uncompressed Cement bridge (`0.6163`)

Interpretation:
- if co-trained Variant A clearly beats `0.5951`, then the frozen-perceiver setup was leaving value on the table
- if it stays near `0.595`, then direct grid access is mostly a small posthoc gain, not a new training regime
