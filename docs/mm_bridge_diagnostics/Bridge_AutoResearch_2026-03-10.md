# Bridge_AutoResearch (Handoff) - 2026-03-10

## Purpose

This document is a continuation packet for another agent to resume bridge research quickly without re-deriving context.

Primary objective remains: explain and close the gap where image-conditioned prefixes underperform or barely match stable learned prefixes in frozen-bridge VQA, while keeping fixed train/val data roots.

## Fixed System Setup

- Dataset: VQAv2 (`images_root=images`, `annotations_root=data/vqav2`)
- VM: frozen VAE visual encoder (`logs/vm_base2/step_15001.tar`)
- LM: decoder LM (`logs/lm_boom2/step_45000.tar`)
- Default multimodal launch path: `./runmm.sh <run_id> ...` (Docker-backed)
- Core training mode during most successful runs: `freeze_mode=bridge_plus_top_lm` with `train_top_lm_layers=2`

Notes:
- `train/mm.py` always enables `model.lm._unembed` trainability for `bridge_plus_top_lm`.
- LM embed/unembed weights are tied in this codebase; updating `_unembed` also updates embeddings through shared weights.

## What Was Added (Code-Level)

### Bridge families

In `models/bridge.py`:
- `learned_query` (query cross-attention reducer)
- `perceiver_resampler` (multi-round latent query updates)
- `qformer_lite` (alternating query self-attn + cross-attn)
- `hybrid_const_image` (mix learned constant prefix with image prefix)

### Interface calibration

In `train/mm.py`:
- prefix calibrator (post-bridge, pre-LM) with options:
  - `--prefix_calibration`
  - `--prefix_calib_layernorm`
  - `--prefix_calib_bias`
  - `--prefix_calib_gate_init`
- regularizers:
  - `--prefix_norm_target_ratio`
  - `--prefix_norm_reg_weight`
  - `--prefix_batchvar_reg_weight`

### Throughput and stability controls

In `train/mm.py` and run scripts:
- `--precision bf16`
- loader knobs (`--num_workers`, `--prefetch_factor`, `--pin_memory`)
- `--grad_accum_steps`
- `--cuda_empty_cache_after_eval`
- optional VM device routing: `--vision_device auto|cpu|cuda`

## Architecture Mechanics (Exact)

Notation:
- VM tokens: `V in R^(B x N x Dv)` (here usually `N=49`)
- bridge outputs visual prefix: `P in R^(B x K x D)` (here usually `K=49`)
- LM text embeddings: `T in R^(B x L x D)`
- LM input is `[P ; T]`

Bridge implementations in `models/bridge.py`:

1. `mlp`
- Per visual token projection: `v -> Linear(Dv,h) -> GELU -> Linear(h,D)`.
- If token count differs from `K`, reduction/expansion is controlled by `token_reduce`; most successful runs used `token_reduce=all` with `N=K=49`.
- Optional 2D sin-cos positional embedding added before projection (`--bridge_add_2d_pos_emb`).

2. `learned_tokens`
- No image dependence.
- Learned parameter `P0 in R^(1 x K x D)` expanded across batch.

3. `learned_query`
- Learned queries `Q0 in R^(1 x K x D)`.
- One cross-attention pass into projected visual tokens:
  - `Q1 = CrossAttnFFN(Q0, Vproj)`.
- Optional query refinement blocks (`--bridge_refine_layers`) with self-attn FFN.
- Optional visual pre-mixer (`none | self_attn | conv1d`) before cross-attn.

4. `perceiver_resampler`
- Learned latents `Z0 in R^(1 x K x D)`.
- Repeated rounds (`--bridge_query_depth`) of:
  - `Z <- CrossAttnFFN(Z, Vproj)`
  - `Z <- SelfAttnFFN(Z)`
- This is iterative compression rather than one-shot extraction.

5. `qformer_lite`
- Learned queries with depth `d`.
- Each block alternates:
  - query self-attn
  - cross-attn into `Vproj`
  - FFN residual

6. `hybrid_const_image`
- Mix of constant learned prefix and image bridge output:
  - `P = alpha * P_const + (1 - alpha) * P_img`
- `alpha` can be scalar or token-wise (`--bridge_hybrid_alpha_mode`).
- In the successful night sweep, `P_img` used `perceiver_resampler`.

Prefix calibration in `train/mm.py`:
- post-bridge calibration before LM concat:
  - optional LayerNorm
  - learnable gain gate
  - optional learnable bias
- regularizers:
  - norm-ratio target (`prefix_norm_target_ratio`)
  - batch variance penalty (`prefix_batchvar_reg_weight`)

## Experiment Classes (Why, What, Params)

Trainable and total parameter counts below are copied from each run's final log line:
`[mm] trainable_params=... / total_params=...`.

| Class | Why This Was Tried | Targeted Hypotheses | Representative Runs | Trainable / Total Params | Result |
|---|---|---|---|---|---|
| C0: historical controls (`learned_tokens` vs image `mlp`) | Reproduce the core paradox and establish baseline gap before adding complexity. | H1 vs H4 boundary check | `mmbr_basesweep_lt1`, `mmbr_basesweep_on_high`, `mmbr_basesweep_off_high` | `25,088 / 41,852,451` (learned tokens), `542,208 / 42,369,571` (MLP) | Constant prefix outperformed image MLP; confirmed problem is real. |
| C1: image-signal diagnostics | Verify whether image-conditioned models truly use visual info or collapse to language priors. | H1, H2 | `mmdiag_*` runs via `run_mm_diag.sh` | N/A (eval-only diagnostics) | Image bridge models were image-sensitive, but still underperformed constant prefix. |
| C2: prefix calibration on MLP (bridge-only) | Test if interface geometry mismatch (scale/variance) is the blocker even with same bridge. | H3, H4 | `mmcal_mlp49_calib_bonly_v1` | `543,232 / 42,370,595` | Calibration alone stabilized but did not produce large gains (`0.3402`). |
| C3: partial LM adaptation (`bridge_plus_top_lm`) | If LM interface is brittle, allow minimal LM adaptation at the top to absorb prefix shift. | H4 primarily | `mmcal_mlp49_calib_top1_v1`, `mmcal_mlp49_calib_top2_v1`, `mmcal2_top*` | top1: `11,505,152 / 42,370,595`; top2: `14,132,224 / 42,370,595` | Crossed 40%; best calibrated MLP reached `0.4345`. |
| C4: learned-query reducer | Replace one MLP projection with learned-query cross-attn extraction from spatial tokens. | H2, H5 | `mmdinner_lq_*`, `mmnight_*_lq_ref2_sa1_exp` | `26,234,368 / 54,472,739` | Improved over MLP in some settings but below perceiver/hybrid (`0.4388`). |
| C5: perceiver resampler | Test iterative latent extraction/compression instead of one-shot query pass. | H2, H5 | `mmdinner_perceiver_d3_notnight_*`, `mmnight_*_perceiver_*` | d3: `32,541,184 / 60,779,555`; d4: `38,846,976 / 67,085,347` | Best family; current best `0.4544` at d3 + pd0.03. |
| C6: spatial pre-mixer before reduction | Allow visual tokens to interact before query extraction/compression. | H2, H5 | `mmnight_*_perceiver_d3_sa1_main`, `mmnight_*_lq_ref2_sa1_exp` | perceiver d3+sa1: `35,693,568 / 63,931,939` | Competitive (`0.4542`) but not better than best non-mixer d3 run. |
| C7: hybrid constant+image | Blend stable LM conditioning from constant prefix with image variation from learned image bridge. | H4 directly, also H3 | `mmnight_*_hybrid_tok060_perc_d3_main`, `mmnight_*_hybrid_tok075_perc_d3_main` | `32,566,321 / 60,804,692` | Very strong (`0.4527-0.4538`), validating stability+image blend idea. |
| C8: qformer-lite | Test deeper query reasoning with alternating self/cross blocks. | H2 | `mmnight_*_qformer_d3_exp` | `26,238,976 / 54,477,347` | Under perceiver/hybrid in this cycle (`0.4383`). |
| C9: scheduler/dropout micro-ablations | Improve optimization stability and generalization without changing architecture class. | Optimization support for H4/H3 | `mmcal2_top1_const_ext1`, `mmcal2_top1_cos_ext1`, `mmcal2_top1_cos_pd05_ext1` | `11,505,152 / 42,370,595` | Small but consistent gains; best top1 variant used cosine + dropout 0.05 (`0.4325`). |

## Why The Search Progressed In This Order

1. Reproduce and bound the failure mode first.
- We started with constant vs image MLP controls and corruption diagnostics to avoid optimizing against a possibly false premise.

2. Attack the lowest-effort/highest-leverage bottleneck next.
- Diagnostics suggested LM-interface mismatch (prefix geometry) more than total absence of image signal.
- That motivated prefix calibration and modest LM unfreezing before large bridge redesign.

3. Move to architecture changes only after crossing 40% with calibration.
- Once calibrated MLP+top-LM proved the interface could work (`~0.434`), we focused on extraction/compression quality (learned-query, perceiver, qformer).

4. Combine stability and image information after perceiver success.
- Hybrid bridge was explicitly tested because constant prefixes remained strong; results confirmed this was a valid design axis.

5. Keep throughput optimized to increase nightly search breadth.
- Benchmarks were used to choose `bf16 + bs192 + workers4`, avoid VM-on-CPU, and maintain more completed runs per horizon.

## High-Signal Findings (Current)

1. Historical diagnosis
- Learned constant prefix baseline beat MLP image bridge in the early frozen-bridge setup.
- Diagnostic perturbations showed image bridges are image-sensitive, but geometry/stability mismatch at LM interface was a central bottleneck.

2. Prefix calibration + top-LM unfreeze crossed 40%
- Calibrated MLP with top LM layers trainable reached up to `0.4345` (`mmcal2_top2_cos_ext1`).

3. New bridge family wins
- `perceiver_resampler` and `hybrid_const_image` reached `~0.453-0.454`.
- Current best: `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_pd03_main` at **0.4544** overall.

4. Q-former-lite and deeper learned-query were competitive but below perceiver/hybrid in this sweep
- `qformer_d3_exp`: `0.4383`
- `lq_ref2_sa1_exp`: `0.4388`

## Run Status and Artifacts

- Structured run ledger: `docs/mm_bridge_diagnostics/10_all_runs_structured_2026-03-10.md`
- Night sweep timeline: `logs/mmnight_bridge_v2_8h_20260309_234936/timeline.log`
- Night sweep launcher: `scripts/launch_night_bridge_sweep_v2_8h.sh`
- Prior diagnostics narrative:
  - `docs/mm_bridge_diagnostics/01_historical_gap_audit.md`
  - `docs/mm_bridge_diagnostics/02_image_signal_sensitivity.md`
  - `docs/mm_bridge_diagnostics/03_prefix_geometry_interface.md`
  - `docs/mm_bridge_diagnostics/04_prefix_calibration_iteration.md`
  - `docs/mm_bridge_diagnostics/05_prefix_calibration_sweep_v2.md`
  - `docs/mm_bridge_diagnostics/06_sweep_v2_completion_report_2026-03-09.md`
  - `docs/mm_bridge_diagnostics/08_dinner_followup_runs_report_2026-03-09.md`
  - `docs/mm_bridge_diagnostics/09_night_sweep_plan_v2_8h_2026-03-09.md`

## Performance Engineering Summary

Quick benchmark signal from `logs/bench_*`:
- `bf16` is materially better than earlier fp32-style throughput.
- Loader workers matter: `batch=192, workers=4` showed strong speedup versus workers=2.
- Very large batch (`320`) degraded throughput badly.
- Vision on CPU (`--vision_device cpu`) was significantly slower (roughly an order-of-magnitude step-rate drop in observed probe runs) and was not used for successful sweeps.
- Night sweep settings that worked well:
  - `batch_size=192`
  - `grad_accum_steps=1`
  - `num_workers=4`
  - `prefetch_factor=2`
  - `eval_batches=200`
  - `--cuda_empty_cache_after_eval`

## Interpretation of Bottleneck (Current Ranking)

Most supported:
- LM interface sensitivity to prefix distribution (stability, norm scale, variance)
- one-shot extraction/compression weakness in older bridges

Supported:
- compression bottlenecks
- partial feature/interface mismatch between VM latent geometry and LM embedding geometry

Not ruled out but lower priority right now:
- total VM semantic deficiency as the primary blocker
- VM pretraining objective mismatch as the only blocker

## Recommended Next Research Thread (If Resuming)

Given current best results already come from perceiver/hybrid, next work should bias toward:

1. Perceiver/hybrid local refinements
- alpha schedule (static vs learned vs annealed)
- selective pre-mixer (1 layer self-attn only when it improves number-category)
- depth/regularization sweeps around current best (`d3`, `pd 0.00-0.05`)

2. Category-targeted tuning
- number-category remains lowest among answer types; prioritize losses or curriculum that help counting/attributes.

3. Robustness checks
- repeat top 2-3 configs with different seeds to confirm ranking stability.
- hold settings fixed; verify gains are not eval-slice artifacts.

## Operational Notes for Another Agent

- Use Docker launch path only (`./runmm.sh` or sweep scripts).
- Keep data roots unchanged unless explicitly requested.
- Do not delete/move project files; add docs/scripts only.
- Prefer short indicative runs for pruning, then longer confirmatory runs on top candidates.
- Keep writing one markdown per direction plus periodic consolidated summaries.

## Dashboard / Tracking

- Progress dashboard server: `tracker/research/researchtrackerapp.ts`
- Run:
  - `bun run tracker/research/researchtrackerapp.ts -p 4090`
- UI:
  - dashboard: `http://localhost:4090`
  - markdown viewer: `http://localhost:4090/doc?file=<doc_name>.md`
