# Final Diagnosis: Frozen-Bridge VQA Failure Mode

## Executive Conclusion
Primary bottleneck is the **visual-to-LM interface**, not a complete lack of visual signal in VM latents.

The image-conditioned bridge does encode and use image information (accuracy drops under image corruption), but the frozen LM performs better with a stable learned constant prefix than with high-variance visual prefixes. This points to interface/geometry mismatch and conditioning stability as the dominant failure mode.

## Evidence Summary

1. Historical gap is real and stable
- Learned constant prefix (`mmbr_basesweep_lt1`) reaches `0.3540`.
- Best image-conditioned MLP (`mmbr_basesweep_on_high`) reaches `0.3429`.

2. Image-conditioned checkpoints are image-sensitive, but not enough to win
- For MLP K=49 (`mmdiag_mlp_onhigh`), clean `0.3408` drops to:
  - shuffle `-0.0147`
  - zero `-0.0114`
  - fixed-image `-0.0204`
- For MLP K=49 no-pos (`mmdiag_mlp_offhigh`), zero drop is larger (`-0.0391`), confirming image dependence.

3. Learned constant prefix is fully image-invariant and still best
- `mmdiag_lt1`: all perturbations produce identical predictions and accuracy.

4. Prefix geometry strongly differs by bridge type
- Learned tokens: zero sample variance, pairwise cosine `1.0`, norm ratio `~1.34`.
- MLP bridges: substantial sample variance and very high prefix/text norm ratios (`~12x` to `~31x`).
- Better MLP variant also has more stable geometry than weaker MLP variant.

5. Compression contributes but does not explain the whole gap
- MLP K=8 (`mmdiag_mlp_k8`) underperforms K=49.
- So H5 (compression bottleneck) is real, but K=49 MLP still trails learned constant tokens.

## Hypothesis Ranking (H1-H6)

- **Most supported**
  - `H4` LM interface sensitivity
  - `H3` representation geometry mismatch
- **Supported, secondary**
  - `H5` compression bottleneck
  - `H2` bridge extraction failure (current bridge is likely under-calibrated)
- **Not primary from current evidence**
  - `H1` total visual representation deficiency
  - `H6` VM objective mismatch as sole explanation

## Strategic Decision Target
Prioritize **bridge/interface redesign and calibration** before retraining VM from scratch.

## Recommended Next Steps

1. Prefix calibration layer (low risk, high signal)
- Add trainable LayerNorm + gated scale + bias after bridge output and before LM concat.
- Add explicit norm target regularizer so visual prefix norms track text embedding norms.

2. Stability regularization
- Penalize per-batch prefix variance for semantically similar questions or random pair consistency.
- Add mild token-wise dropout/noise during training to reduce over-fragile conditioning.

3. Bridge architecture upgrades
- Replace pure MLP mapping with cross-attention adapter from learned query tokens into visual features.
- Keep K fixed, but learn query bank initialized from LM embedding stats.

4. Interface curriculum
- Warm start from learned constant tokens, then progressively blend in image-conditioned component:
  - `prefix = alpha * learned_const + (1 - alpha) * image_prefix`
  - anneal `alpha` from 1.0 to 0.0.

5. Later-stage VM investigations only if needed
- If calibrated bridge still plateaus below learned-token baseline, run VM-side semantic probes and VM objective changes.

## Repro Commands

```bash
# learned-token baseline sensitivity
./tasks/mm_bridge/scripts/run_mm_diag.sh mmdiag_lt1 \
  --checkpoint logs/mmbr_basesweep_lt1/step_17330.tar \
  --max_batches 80 --stats_batches 40 --batch_size 256 \
  --modes clean,shuffle,zero,noise,fixed_image --noise_std 0.2

# image-conditioned K=49 (+2D pos)
./tasks/mm_bridge/scripts/run_mm_diag.sh mmdiag_mlp_onhigh \
  --checkpoint logs/mmbr_basesweep_on_high/step_3466.tar \
  --max_batches 80 --stats_batches 40 --batch_size 256 \
  --modes clean,shuffle,zero,noise,fixed_image --noise_std 0.2

# image-conditioned K=49 (no 2D pos)
./tasks/mm_bridge/scripts/run_mm_diag.sh mmdiag_mlp_offhigh \
  --checkpoint logs/mmbr_basesweep_off_high/step_3466.tar \
  --max_batches 80 --stats_batches 40 --batch_size 256 \
  --modes clean,shuffle,zero,noise,fixed_image --noise_std 0.2
```

## Model -> Overall Accuracy (Official)

| Model | Overall Accuracy |
|---|---:|
| `mmbr_basesweep_lt1` | `0.3540` |
| `mmbr_basesweep_on_high` | `0.3429` |
| `mmcal2_top1_const_ext1` | `0.4319` |
| `mmcal2_top1_cos_ext1` | `0.4323` |
| `mmcal2_top1_cos_pd05_ext1` | `0.4325` |
| `mmcal2_top2_cos_ext1` | `0.4345` |
| `mmdinner_lq_deeper_sa2_ref2_clean_20260309_213605` | `0.4257` |
| `mmdinner_perceiver_d3_notnight_20260309_221422` | `0.4415` |

## Artifacts Produced
- Intermediate direction notes:
  - `tasks/mm_bridge/docs/01_historical_gap_audit.md`
  - `tasks/mm_bridge/docs/02_image_signal_sensitivity.md`
  - `tasks/mm_bridge/docs/03_prefix_geometry_interface.md`
  - `tasks/mm_bridge/docs/08_dinner_followup_runs_report_2026-03-09.md`
- Per-run diagnostic outputs:
  - `logs/mmdiag_*/diag_report.json`
  - `logs/mmdiag_*/diag_report.md`
