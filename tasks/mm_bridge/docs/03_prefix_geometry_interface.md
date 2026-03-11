# Prefix Geometry and LM Interface

## Goal
Diagnose whether the frozen LM is receiving visually-conditioned prefixes with statistics that are hard to use, compared with stable learned constants.

## Metrics
From `diag_report.json` prefix stats:
- `prefix_batch_variance_mean`: variance across samples (batch axis)
- `prefix_pairwise_cos_mean`: cosine similarity between flattened prefixes from different samples
- `prefix_text_norm_ratio`: average visual-prefix token norm / text-token norm

## Clean-Mode Geometry

| run | bridge | prefix_batch_variance_mean | prefix_pairwise_cos_mean | prefix_text_norm_ratio |
|---|---|---:|---:|---:|
| `mmdiag_lt1` | learned tokens K=49 | 0.0000 | 1.0000 | 1.3383 |
| `mmdiag_mlp_onhigh` | MLP K=49 (+2D pos) | 1.2342 | 0.7958 | 26.6001 |
| `mmdiag_mlp_offhigh` | MLP K=49 | 8.2850 | 0.1968 | 31.5029 |
| `mmdiag_mlp_k8` | MLP K=8 | 0.6010 | 0.6205 | 12.4189 |

## Observations
- Learned tokens present perfectly stable prefixes to the LM:
  - zero batch variance, pairwise cosine 1.0, moderate norm ratio.
- Image-conditioned prefixes are high-variance and often very high-norm relative to text embeddings:
  - norm ratios from ~12x up to ~31x text-token norm.
- Better MLP variant (`onhigh`) is notably more self-similar and lower variance than `offhigh`, and also has better accuracy.

## Interpretation
- Evidence supports LM-interface sensitivity:
  - frozen LM likely benefits from stable, consistent prefix geometry.
  - noisy/high-amplitude visual prefixes create a conditioning distribution shift the LM handles less effectively.
- This points to interface calibration as a central bottleneck, not only visual-feature absence.
