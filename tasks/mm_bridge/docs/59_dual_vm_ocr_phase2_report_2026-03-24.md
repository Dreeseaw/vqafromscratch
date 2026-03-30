# Dual-VM OCR Phase 2 Report

## Scope

This report covers the completed dual-VM OCR experiment line built on top of the Cement champion family:

- warm-start dual VM: [mmdualvm_v1_20260324_rerun](/home/wdree/percy/vqafromscratch/logs/mmdualvm_v1_20260324_rerun)
- fresh-bridge dual VM: [mmdualvm_v1_20260324_freshbridge](/home/wdree/percy/vqafromscratch/logs/mmdualvm_v1_20260324_freshbridge)
- anchor reference: [mmcement_v1_20260316_siglip_cement_questiononly_s42](/home/wdree/percy/vqafromscratch/logs/mmcement_v1_20260316_siglip_cement_questiononly_s42)

Architecture in both cases:

```text
image
-> frozen SigLIP-B/16 tokens (196 x 768)
-> frozen ViTSTR-Tiny tokens (196 x 192)
-> learned per-stream projections to bridge width
-> concatenated token grid
-> perceiver resampler (49 latents)
-> question_only + attnqquery bridge path
-> LM + LM visual adapters
-> VQA decoding
```

The only difference between the two main runs was initialization:

- warm-start: perceiver/bridge/LM trainable slice initialized from Cement
- fresh-bridge: same trainable path initialized fresh

## ViTSTR Source

The OCR tower used the released ViTSTR-Tiny checkpoint from `roatienza/deep-text-recognition-benchmark`, loaded as a frozen DeiT-Tiny-style encoder backbone with:

- input resolution: `224 x 224`
- patch size: `16`
- token count: `196` patch tokens
- hidden dim: `192`
- depth: `12`
- heads: `3`

It was used as an encoder only; the recognition head/decoder was stripped.

## Main Results

| Condition | Overall | Yes/No | Number | Other |
|---|---:|---:|---:|---:|
| Cement anchor | 0.6163 | 0.7589 | 0.4573 | 0.5499 |
| Dual-VM warm-start | 0.6307 | 0.7918 | 0.4699 | 0.5508 |
| Dual-VM fresh-bridge | 0.6160 | 0.7581 | 0.4572 | 0.5501 |

OCR subset comparison:

| Condition | OCR subset overall | Delta vs anchor |
|---|---:|---:|
| Cement anchor | 0.2510 | - |
| Dual-VM warm-start | 0.2930 | +0.0420 |
| Dual-VM fresh-bridge | 0.2884 | +0.0374 |

The top-line conclusion is simple:

- dual VM can help
- but the gain was almost entirely a warm-start effect
- the fresh-bridge run did preserve the OCR subset lift, but it did not convert that into a general VQAv2 win

## Training Dynamics

Warm-start behavior:

- final full eval: `0.6307`
- ViTSTR attention fraction started high and collapsed hard:
  - start `0.3320`
  - end `0.0202`
- despite low average OCR routing, the model still improved both global score and OCR subset score

Fresh-bridge behavior:

- final full eval: `0.6160`
- periodic curve climbed steadily:
  - `1000 -> 0.4617`
  - `5000 -> 0.5879`
  - `8000 -> 0.6144`
  - `9000 periodic -> 0.6173`
  - `9000 final -> 0.6160`
- ViTSTR attention stayed materially higher:
  - start `0.4700`
  - end `0.0583`

That contrast matters. The fresh run used the OCR stream more consistently, but still only matched the SigLIP-only anchor. The warm-start run used the OCR stream sparsely on average yet achieved the best result.

## OCR Routing Read

Warm-start OCR routing from [ocr_analysis.json](/home/wdree/percy/vqafromscratch/logs/mmdualvm_v1_20260324_rerun/ocr_analysis.json):

- `brand_logo`: `0.0124`
- `name_label`: `0.0180`
- `sign_written`: `0.0069`
- `word_text`: `0.0162`
- control: `0.0155`

Fresh-bridge OCR routing from [ocr_analysis.json](/home/wdree/percy/vqafromscratch/logs/mmdualvm_v1_20260324_freshbridge/ocr_analysis.json):

- `brand_logo`: `0.0163`
- `name_label`: `0.0075`
- `sign_written`: `0.0068`
- `word_text`: `0.0178`
- control: `0.0804`

So the fresh-bridge run did not learn a clean OCR-selective routing signature. In fact, the sampled control set drew more ViTSTR attention than the heuristic OCR subset. That strongly suggests the raw OCR tower is being used as a general auxiliary texture stream, not as a disciplined text reader.

## Modeling Interpretation

The dual-VM result splits into two different claims:

1. OCR-like information is genuinely helpful.
The OCR subset improved in both runs, including the fresh-bridge run. So adding ViTSTR did expose useful evidence that the anchor lacked.

2. The current bridge does not naturally learn a clean OCR-specialized retrieval policy from scratch.
The fresh-bridge run matched the anchor overall instead of surpassing it, and its routing statistics were not OCR-selective. The perceiver learned to ingest the extra stream, but not to separate "text questions should use OCR tokens" from "generic questions should mostly ignore them."

The warm-start gain is therefore best read as:

- Cement already provided a strong SigLIP-centered decision surface
- adding ViTSTR gave the existing bridge a modest extra evidence source
- because the bridge was already competent, it could harvest the useful OCR bits without needing to relearn multimodal routing from zero

The fresh-bridge run shows the harder truth:

- once the bridge has to learn both normal VQA extraction and OCR routing jointly from scratch, the OCR stream does not automatically create a new frontier

## Recommendation

Keep the dual-VM line alive, but not as "drop in OCR tower and expect a free win."

Most defensible next steps:

1. Treat the warm-start dual-VM result as the real positive result.
It is a genuine empirical gain over Cement and over the OCR subset.

2. Do not interpret the fresh-bridge neutrality as "OCR does not help."
It means the current perceiver/qquery stack does not discover OCR-specialized routing robustly from scratch.

3. If this line continues, bias the next experiment toward explicit OCR routing pressure rather than more generic training time.
The evidence says the missing piece is not raw OCR features. It is selective use.

Short version:

- warm-start dual VM: real win
- fresh-bridge dual VM: roughly neutral overall, still OCR-positive
- project-level meaning: OCR information is useful, but the bridge needs help learning when to trust it
