# Image Signal Sensitivity

## Goal
Measure how much each trained checkpoint actually depends on image content at inference time.

## Method
- Script: `scripts/mm_bridge_diagnostics.py`
- Docker wrapper: `run_mm_diag.sh`
- Eval budget per mode: `80` batches at `batch_size=256` (`20,480` samples)
- Perturbation modes:
  - `clean`: original image
  - `shuffle`: image reassigned across questions in batch
  - `zero`: all-zero image tensor
  - `noise`: additive Gaussian noise (`std=0.2`)
  - `fixed_image`: first image repeated across full batch

## Executed Runs
- `logs/mmdiag_lt1/diag_report.json` (learned tokens, K=49)
- `logs/mmdiag_mlp_onhigh/diag_report.json` (MLP, K=49, +2D pos)
- `logs/mmdiag_mlp_offhigh/diag_report.json` (MLP, K=49, no 2D pos)
- `logs/mmdiag_mlp_k8/diag_report.json` (MLP, K=8)

## Results

| run | bridge | clean acc | shuffle delta | zero delta | fixed-image delta | agreement(clean vs shuffle) |
|---|---|---:|---:|---:|---:|---:|
| `mmdiag_lt1` | learned tokens K=49 | 0.3533 | +0.0000 | +0.0000 | +0.0000 | 1.0000 |
| `mmdiag_mlp_onhigh` | MLP K=49 (+2D pos) | 0.3408 | -0.0147 | -0.0114 | -0.0204 | 0.5847 |
| `mmdiag_mlp_offhigh` | MLP K=49 | 0.3377 | -0.0180 | -0.0391 | -0.0179 | 0.5421 |
| `mmdiag_mlp_k8` | MLP K=8 | 0.3259 | -0.0151 | -0.0232 | -0.0205 | 0.5384 |

## Interpretation
- Learned-token model is perfectly image-invariant (by design), yet strongest on accuracy.
- Image-conditioned models do use image content:
  - shuffling/zeroing/fixing images reduces accuracy and changes many predictions.
- But image-conditioned models remain partly language-prior driven:
  - even after strong image corruption, accuracy remains relatively high (roughly `0.299` to `0.329`).
- Additive noise at `std=0.2` barely affects accuracy, suggesting dependence is more on coarse/global signal than fine-grained detail.
