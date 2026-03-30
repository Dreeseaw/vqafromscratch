# LM Shrink Sweep Plan

This sweep asks a narrower question than the pasted prompt: how small can the LM get in this codebase while preserving useful bridge behavior, using the actual current training stack rather than an idealized one.

## Grounded Starting Point

The canonical LM in this repo is not `~46M`; it is `39,859,712` params at:
- `embed_size=512`
- `num_heads=8`
- `layers=12`
- `mlp_ratio=2`
- tied vocab over `16279` tokens

The current best bridge line is the Cement family:
- frozen `SigLIP-B/16`
- `question_only`
- `question_hidden_attn`
- perceiver depth `3`
- no dynbudget
- LM adapter depth `3`
- top `2` LM layers trainable during bridge training

The current format-alignment line is also already defined in-code:
- `K=8` semantic bottleneck
- bottleneck-only training
- adapters disabled in forward
- `L_vqa + 0.1 * L_distill`
- optional `L_format` when LM geometry is compatible with the existing remap teacher

## Variant Definitions

To maximize signal tonight without rewriting the LM core, the sweep uses variants the current code can instantiate directly:

| Variant | LM shape | Params | Notes |
|---|---|---:|---|
| `half` | `d=512, h=8, L=6` | `24.1M` | Clean half-depth variant, same 512-wide interface as canonical |
| `quarter` | `d=384, h=6, L=4` | `12.2M` | True quarter-class LM, different interface width |
| `tiny` | `d=256, h=4, L=3` | `6.1M` | Smallest still-plausible decoder-only LM |
| `randinit` | `d=512, h=8, L=12` | `39.9M` | No LM pretraining control |

Why this differs from the pasted prompt:
- the current LM stack does not already have an internal "same embed width, smaller hidden body" path
- adding that tonight would create avoidable architecture risk
- the current MM trainer already supports different LM widths cleanly

## Actual Three-Stage Pipeline

For `half`, `quarter`, and `tiny`:
1. LM pretraining on the canonical mixed wiki + distill corpus
2. Fresh bridge training with frozen SigLIP and fresh perceiver
3. K=8 format-alignment compression on top of that variant's own bridge checkpoint

For `randinit`:
1. skip LM pretraining
2. fresh bridge training from random-init LM
3. K=8 format-alignment compression

## Important Recipe Adjustments

These are intentional, codebase-aware deviations from the pasted prompt:

1. Bridge training uses the real current Cement freeze recipe: top `2` LM layers, not top `3`.
2. The remap teacher is only reused where the LM geometry is still `512`-wide.
   - `half`: yes
   - `quarter`: no
   - `tiny`: no
   - `randinit`: no
3. The run order is:
   - `half`
   - `randinit`
   - `quarter`
   - `tiny`

That order is better for a single overnight GPU because:
- `half` answers the main product question first
- `randinit` is the shortest high-value control
- `quarter` and `tiny` are lower-cost follow-ons after the main answer exists

## Expected Timing

Using actual repo timings:
- canonical LM pretrain to `step_45000`: about `2h17m`
- canonical 9k Cement bridge block: about `64m`
- canonical 3k format-alignment block: about `14m`

Conservative per-variant estimate:
- `half`: about `2.9h`
- `randinit`: about `1.6h`
- `quarter`: about `2.3h`
- `tiny`: about `1.9h`

Total expected overnight runtime with post-phase eval/probe overhead:
- about `8.5h` to `10h`, assuming no stalls

So unlike the pasted prompt, this full sweep is actually plausible tonight on one GPU if the runner is resume-safe and the machine stays healthy.

## Deliverables

The runner will write:
- `codebase_inspection.md`
- `progress.md`
- `lm_shrink_report.md`

under a stamped bundle dir:
- `logs/mmsemantic_lmshrink_v1_<timestamp>/`

The runner is:
- [run_lm_shrink_sweep.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/run_lm_shrink_sweep.py)
