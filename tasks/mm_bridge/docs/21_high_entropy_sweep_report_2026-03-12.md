# High-Entropy Sweep Report - 2026-03-12

## Scope

This note compiles the completed high-entropy architecture sweep from:

- `tasks/mm_bridge/docs/20_high_entropy_sweep_plan_2026-03-11.md`
- `tasks/mm_bridge/scripts/launch_high_entropy_sweep_v1.sh`
- sweep bundles under `logs/mmarch_high_entropy_v1_20260311_140532` through `logs/mmarch_high_entropy_v1_20260312_093925`
- per-run logs under `logs/mmarch_high_entropy_v1_20260311_*`
- `tasks/mm_bridge/docs/MM_BRIDGE_RUN_STABILITY_POSTMORTEM.md`
- prior comparison context from `tasks/mm_bridge/docs/19_final_arch_report_2026-03-11.md` and `tasks/mm_bridge/docs/10_all_runs_structured_2026-03-10.md`

The purpose here is retrospective only:

- record what actually ran
- record which scores are authoritative
- preserve the evidence that matters when choosing later sweeps

This is not a forward plan.

## Sweep Definition

Common sweep policy from the launcher and bundle READMEs:

- effective batch size `192`
- target step `9000`
- `eval_every=1000`
- periodic evals on `100` val batches
- final eval on full val (`final_eval_batches=0`, `eval_fraction=1.0`)
- official scorer
- common LM/bridge training policy inherited from the final architecture queue

Important comparison caveat:

- periodic evals are partial-val checks only
- final scores below are the full-val official numbers
- use periodic curves directionally, not as exact ranking evidence

## Planned vs Executed Set

The plan document proposed 6 runs.

The launcher actually executed 9 runs:

1. `safeqcond_frontier`
2. `structuredroles_frontier`
3. `earlylayer_encoder_frontier`
4. `safeqcond_earlylayer_frontier`
5. `safeqcond_geomcal_frontier`
6. `structuredroles_geomcal_frontier`
7. `safeqcond_earlylayer_geomcal_frontier`
8. `safeqcond_multiscale_frontier`
9. `safeqcond_hybrid_tok075_frontier`

So the realized sweep was broader than the written plan by three extra stack tests:

- `safeqcond + earlylayer + geomcal`
- `safeqcond + multiscale`
- `safeqcond + hybrid tok075`

## Completion and Provenance

All 9 run IDs eventually reached a `step_9000.tar` checkpoint and an authoritative final score.

Completion required several relaunch bundles:

- `logs/mmarch_high_entropy_v1_20260311_140532`
- `logs/mmarch_high_entropy_v1_20260312_002859`
- `logs/mmarch_high_entropy_v1_20260312_004020`
- `logs/mmarch_high_entropy_v1_20260312_020559`
- `logs/mmarch_high_entropy_v1_20260312_090953`
- `logs/mmarch_high_entropy_v1_20260312_093925`

The stability postmortem identifies the main interruption causes:

- launcher wall-clock timeout kills
- qcond eval semantic regression
- non-qcond KV-cache eval regression
- host OOM from persistent train/val workers during long eval

Run-status caveats that matter for interpretation:

- `safeqcond_frontier`, `structuredroles_frontier`, `safeqcond_earlylayer_frontier`, and `safeqcond_geomcal_frontier` finished with authoritative post-hoc `eval_only` scoring from the `9000` checkpoint
- `earlylayer_encoder_frontier` and `structuredroles_geomcal_frontier` required resumed training
- resume restores checkpoint state, but not exact sampler/DataLoader position

## Final Ranking

Reference frontier from prior work:

- previous best official full-val run: `0.4544`
- source: `mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_pd03_main`
- recorded in `tasks/mm_bridge/docs/10_all_runs_structured_2026-03-10.md`

Final high-entropy ranking:

| Rank | Run | Final tag | Overall | Yes/No | Number | Other | Delta vs `0.4544` |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | `safeqcond_earlylayer_geomcal_frontier` | `final_eval` | `0.4568` | 0.6855 | 0.3202 | 0.3189 | `+0.0024` |
| 2 | `safeqcond_earlylayer_frontier` | `eval_only` | `0.4561` | 0.6846 | 0.3195 | 0.3184 | `+0.0017` |
| 3 | `safeqcond_hybrid_tok075_frontier` | `final_eval` | `0.4547` | 0.6871 | 0.3171 | 0.3141 | `+0.0003` |
| 4 | `safeqcond_geomcal_frontier` | `eval_only` | `0.4544` | 0.6860 | 0.3187 | 0.3140 | `+0.0000` |
| 5 | `earlylayer_encoder_frontier` | `final_eval` | `0.4543` | 0.6909 | 0.3160 | 0.3108 | `-0.0001` |
| 6 | `safeqcond_frontier` | `eval_only` | `0.4541` | 0.6842 | 0.3169 | 0.3152 | `-0.0003` |
| 7 | `safeqcond_multiscale_frontier` | `final_eval` | `0.4533` | 0.6808 | 0.3190 | 0.3157 | `-0.0011` |
| 8 | `structuredroles_geomcal_frontier` | `final_eval` | `0.4522` | 0.6885 | 0.3149 | 0.3087 | `-0.0022` |
| 9 | `structuredroles_frontier` | `eval_only` | `0.4507` | 0.6838 | 0.3163 | 0.3088 | `-0.0037` |

High-level read from the final table:

- the sweep did produce a new best observed score: `0.4568`
- the top cluster is dominated by `safeqcond`-centered stacks
- all 9 final runs landed in a tight `0.4507` to `0.4568` band
- the gap from rank 1 to rank 9 is only `0.0061`

## Relation to the Prior 6k Architecture Sweep

The prior architecture sweep in `tasks/mm_bridge/docs/19_final_arch_report_2026-03-11.md` used:

- shorter training horizons
- half-val final evaluation

So the deltas below are directional, not exact apples-to-apples.

Still, the extension picture is useful:

| Branch | Prior 6k result | High-entropy 9k result | Directional delta |
|---|---:|---:|---:|
| `safeqcond` | `0.4460` | `0.4541` | `+0.0081` |
| `structuredroles` | `0.4435` | `0.4507` | `+0.0072` |
| `earlylayer_encoder` | `0.4429` | `0.4543` | `+0.0114` |
| `multiscale` | `0.4398` | `0.4533` | `+0.0135` |

What this preserves:

- the strong 6k branches did in fact convert into frontier-range 9k runs
- early-layer and multiscale families gained the most from longer training
- structured roles remained positive, but it did not close the gap to the qcond-led cluster

## Combination Evidence

This sweep was supposed to test stacking. The final results do preserve that evidence.

### 1. `safeqcond + earlylayer`

- `safeqcond_frontier`: `0.4541`
- `earlylayer_encoder_frontier`: `0.4543`
- `safeqcond_earlylayer_frontier`: `0.4561`

Observed combination delta:

- `+0.0018` over the better single-branch comparator (`0.4543`)

This is the clearest two-way positive stack in the sweep.

### 2. `safeqcond + geomcal`

- `safeqcond_frontier`: `0.4541`
- `safeqcond_geomcal_frontier`: `0.4544`

Observed combination delta:

- `+0.0003` over `safeqcond_frontier`

This is positive, but small.

### 3. `structuredroles + geomcal`

- `structuredroles_frontier`: `0.4507`
- `structuredroles_geomcal_frontier`: `0.4522`

Observed combination delta:

- `+0.0015` over `structuredroles_frontier`

This is a real lift, though it still leaves the family below the best qcond-led branches.

### 4. `safeqcond + earlylayer + geomcal`

- `safeqcond_earlylayer_frontier`: `0.4561`
- `safeqcond_earlylayer_geomcal_frontier`: `0.4568`

Observed combination delta:

- `+0.0007` over the already-strong `safeqcond + earlylayer` stack

This is the best score in the sweep, but the extra gain over the two-way stack is modest.

### 5. `safeqcond + multiscale`

- `safeqcond_frontier`: `0.4541`
- `safeqcond_multiscale_frontier`: `0.4533`

Observed combination delta:

- `-0.0008` relative to `safeqcond_frontier`

This kept the run competitive, but it did not beat the simpler qcond single.

### 6. `safeqcond + hybrid tok075`

- `safeqcond_frontier`: `0.4541`
- `safeqcond_hybrid_tok075_frontier`: `0.4547`

Observed combination delta:

- `+0.0006` over `safeqcond_frontier`

Relative to the best prior non-qcond hybrid run from `tasks/mm_bridge/docs/10_all_runs_structured_2026-03-10.md`:

- old `hybrid_tok075_perc_d3_main`: `0.4538`
- new `safeqcond_hybrid_tok075_frontier`: `0.4547`
- directional gain: `+0.0009`

## Answer-Type Patterns

The answer-type splits are useful because the best overall runs are not winning in exactly the same way.

Best `yes/no`:

- `earlylayer_encoder_frontier`: `0.6909`
- `structuredroles_geomcal_frontier`: `0.6885`
- `safeqcond_hybrid_tok075_frontier`: `0.6871`

Best `number`:

- `safeqcond_earlylayer_geomcal_frontier`: `0.3202`
- `safeqcond_earlylayer_frontier`: `0.3195`
- `safeqcond_multiscale_frontier`: `0.3190`

Best `other`:

- `safeqcond_earlylayer_geomcal_frontier`: `0.3189`
- `safeqcond_earlylayer_frontier`: `0.3184`
- `safeqcond_multiscale_frontier`: `0.3157`

Interpretation preserved by these splits:

- `earlylayer` is especially strong on `yes/no`
- the best qcond-led stacks pull ahead mainly by lifting `number` and `other`
- `structuredroles + geomcal` improves `yes/no` a lot more than it improves `number/other`

## Throughput and Cost Signal

Last logged train throughput at the end of training:

| Run | Final overall | Last logged steps/s |
|---|---:|---:|
| `safeqcond_earlylayer_frontier` | `0.4561` | `5.13` |
| `earlylayer_encoder_frontier` | `0.4543` | `5.08` |
| `safeqcond_geomcal_frontier` | `0.4544` | `5.05` |
| `structuredroles_frontier` | `0.4507` | `5.02` |
| `safeqcond_frontier` | `0.4541` | `4.94` |
| `safeqcond_earlylayer_geomcal_frontier` | `0.4568` | `4.88` |
| `structuredroles_geomcal_frontier` | `0.4522` | `4.86` |
| `safeqcond_multiscale_frontier` | `0.4533` | `4.80` |
| `safeqcond_hybrid_tok075_frontier` | `0.4547` | `4.73` |

Cost read:

- none of the winning qcond-led stacks paid a catastrophic throughput penalty
- `safeqcond + earlylayer` was both the fastest and one of the strongest
- `hybrid` and `multiscale` were the slowest of the group, though still within a fairly tight band

## Reliability Notes

### 1. Periodic evals are approximate

Periodic checks are only `100` validation batches, while final scores are full-val official evals.

Use periodic curves for:

- collapse detection
- rough slope shape

Do not use them for:

- precise branch ranking
- sub-basis-point combination decisions

### 2. `earlylayer_encoder_frontier` periodic scores are not trustworthy

Its logged periodic evals from `1000` through `8000` are collapse-like and conflict with the final full-val `0.4543`.

Given the documented non-qcond eval regression in `tasks/mm_bridge/docs/MM_BRIDGE_RUN_STABILITY_POSTMORTEM.md`, the right read is:

- final full eval is authoritative
- early periodic evals should not be used as real slope evidence for this branch

### 3. Some authoritative finals are `eval_only` completions

That applies to:

- `safeqcond_frontier`
- `structuredroles_frontier`
- `safeqcond_earlylayer_frontier`
- `safeqcond_geomcal_frontier`

These should still be treated as authoritative because they score the saved `step_9000.tar` checkpoint under the corrected full-val path.

## Compact Takeaways

What this sweep established, without projecting beyond the evidence:

- `safeqcond` was confirmed as a real frontier family under longer training
- `safeqcond + earlylayer` was the strongest two-way stack in the sweep
- `safeqcond + earlylayer + geomcal` produced the best observed result at `0.4568`
- `geomcal` behaves more like a small positive modifier than a large standalone swing
- `structuredroles` stayed positive but did not match the qcond-led cluster
- `multiscale` and `hybrid` remained competitive, but only `hybrid` slightly exceeded the plain `safeqcond` single
- answer-type splits suggest the qcond-led wins come mostly from `number` and `other`, while `earlylayer` is strongest on `yes/no`

