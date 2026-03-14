# MobileViT Performance Tuning

Date: `2026-03-13`

## Purpose

Make the Hugging Face `apple/mobilevit-small` vision path cheap enough to support full-family MM bridge runs on the local `16 GB` GPU.

## Engineering Change

The original `mobilevit_hf` wrapper used the Hugging Face image processor in the forward path, which forced a CPU preprocessing round-trip on every batch.

The runtime was changed to:

- preprocess image tensors directly on-device
- resize with `torch.nn.functional.interpolate`
- apply ImageNet de-normalization and MobileViT BGR channel order directly in PyTorch
- run the MobileViT backbone under CUDA `bf16` autocast internally
- cast returned visual features back to `float32` before the bridge

## Validation

PoC shape check after the rewrite:

- run: `mm_mobilevit_poc_v1_fast2`
- visual features: `(1, 49, 640)`
- visual prefix: `(1, 49, 512)`

The optimized PoC eval path improved from roughly `10` to roughly `18` samples/s at `batch_size=1`.

## Representative Heavy-Family Probe

Probe run:

- `mm_mobilevit_probe_lmmean_d3_b192`
- `mobilevit_hf + question_hidden_mean + qadaptive + cross_attn adapters (3 layers)`
- `batch_size=192`
- `grad_accum_steps=1`
- `precision=bf16`
- `max_steps=80`
- `final_eval_batches=4`

Observed training throughput:

- step `20`: `3.84` train steps/s
- step `40`: `4.22`
- step `60`: `4.19`
- step `80`: `4.19`

Observed GPU memory during the `60`-step window:

- about `13.95 GB / 16.30 GB`

Observed eval throughput at `eval_batch_size=192`:

- `0.77` eval steps/s
- about `148` eval samples/s

## Recommendation

For future MobileViT bridge-family runs on the current machine:

- use `--precision bf16`
- use `batch_size=192`
- use `grad_accum_steps=1`
- use `eval_batch_size=192`
- use `--eval_use_kv_cache --eval_kv_cache_mode batched`

Interpretation:

- the optimized `mobilevit_hf` path is now fast enough for standard-comparison bridge runs at the normal `192 x 1` layout
- the remaining cost is dominated by real model compute, not the old preprocessing overhead
