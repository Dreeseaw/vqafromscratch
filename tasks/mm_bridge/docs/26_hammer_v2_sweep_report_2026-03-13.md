# Hammer v2 Sweep Report - 2026-03-13

## Scope

This note compiles the completed Hammer v2 sweep from:

- `tasks/mm_bridge/docs/22_hammer_sweep_plan_2026-03-12.md`
- `tasks/mm_bridge/scripts/launch_hammer_sweep_v1.sh`
- `tasks/mm_bridge/docs/25_hammer_batched_kvcache_perf_retune_2026-03-12.md`
- `tasks/mm_bridge/docs/24_hammer_kvcache_correctness_report_2026-03-12.md`
- sweep bundles `logs/mmhammer_v1_20260312_213054` and `logs/mmhammer_v1_20260312_234646`
- per-run logs under `logs/mmhammer_v2_*`
- prior comparison context from `tasks/mm_bridge/docs/21_high_entropy_sweep_report_2026-03-12.md`

The purpose here is retrospective only:

- record what actually ran
- record which scores are authoritative
- preserve the evidence that matters when choosing later sweeps

This is not a forward plan.

## Naming and Provenance

There is one naming mismatch that matters:

- the launcher bundle IDs still use `SWEEP_ID=mmhammer_v1_<stamp>`
- the actual run IDs for this completed sweep use `RUN_PREFIX=mmhammer_v2`

So the completed Hammer v2 results live under `logs/mmhammer_v2_*`, while the authoritative sweep timelines live under `logs/mmhammer_v1_*`.

Authoritative completed bundles for this report:

- `logs/mmhammer_v1_20260312_213054`
- `logs/mmhammer_v1_20260312_234646`

Non-authoritative Hammer launcher artifacts that should not be used for ranking:

- failed early bundles `logs/mmhammer_v1_20260312_181541` through `logs/mmhammer_v1_20260312_183400`
- dry-run bundle `logs/mmhammer_v1_20260312_213239`

Important execution details:

- `SKIP_ANCHOR=1` was used, so the carry-forward control did not run inside this sweep
- `qquery_earlylayer_geomcal` completed first in `logs/mmhammer_v1_20260312_213054`
- the later bundle `logs/mmhammer_v1_20260312_234646` skipped that completed run and finished the remaining six

All 7 realized Hammer v2 runs reached:

- `step_9000.tar`
- `fixed_eval_val_answers.jsonl` with `tag=final_eval`

Unlike the earlier high-entropy sweep, this completed set does not rely on:

- post-hoc `eval_only` finals
- resumed training for ranking runs

## Sweep Definition

Common sweep policy from the launcher and bundle READMEs:

- effective batch size `192`
- train layout `192 x 1`
- eval batch size `192`
- target step `9000`
- `eval_every=1000`
- periodic evals on `100` val batches
- final eval on full val (`final_eval_batches=0`, `eval_fraction=1.0`)
- official scorer
- `--eval_use_kv_cache --eval_kv_cache_mode batched`

Important comparison caveats:

- periodic evals are partial-val checks only
- final scores below are the full-val official numbers
- the batched KV-cache eval path was separately correctness-checked in `tasks/mm_bridge/docs/24_hammer_kvcache_correctness_report_2026-03-12.md`

## Planned vs Executed Set

The plan document proposed 8 runs:

1. `anchor_safeqcond_earlylayer_geomcal`
2. `qquery_earlylayer_geomcal`
3. `adapter_safeqcond_earlylayer_geomcal`
4. `dynbudget_qscore_earlylayer_geomcal`
5. `qquery_adapter_earlylayer_geomcal`
6. `qquery_dynbudget_earlylayer_geomcal`
7. `dynbudget_adapter_earlylayer_geomcal`
8. `qquery_dynbudget_adapter_earlylayer_geomcal`

The realized Hammer v2 sweep executed 7 of them:

1. `qquery_earlylayer_geomcal`
2. `adapter_safeqcond_earlylayer_geomcal`
3. `dynbudget_qscore_earlylayer_geomcal`
4. `qquery_adapter_earlylayer_geomcal`
5. `qquery_dynbudget_earlylayer_geomcal`
6. `dynbudget_adapter_earlylayer_geomcal`
7. `qquery_dynbudget_adapter_earlylayer_geomcal`

The only missing planned run was the anchor, because it was intentionally skipped.

## Final Ranking

Reference frontier entering Hammer:

- previous best official full-val run: `safeqcond_earlylayer_geomcal_frontier`
- score: `0.4568`
- source: `tasks/mm_bridge/docs/21_high_entropy_sweep_report_2026-03-12.md`

Final Hammer v2 ranking:

| Rank | Run | Final tag | Overall | Yes/No | Number | Other | Delta vs `0.4568` |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | `qquery_dynbudget_adapter_earlylayer_geomcal` | `final_eval` | `0.4608` | 0.6895 | 0.3182 | 0.3244 | `+0.0040` |
| 2 | `dynbudget_adapter_earlylayer_geomcal` | `final_eval` | `0.4602` | 0.6897 | 0.3164 | 0.3237 | `+0.0034` |
| 3 | `qquery_adapter_earlylayer_geomcal` | `final_eval` | `0.4594` | 0.6868 | 0.3207 | 0.3230 | `+0.0026` |
| 4 | `adapter_safeqcond_earlylayer_geomcal` | `final_eval` | `0.4591` | 0.6837 | 0.3229 | 0.3242 | `+0.0023` |
| 5 | `qquery_dynbudget_earlylayer_geomcal` | `final_eval` | `0.4576` | 0.6886 | 0.3202 | 0.3181 | `+0.0008` |
| 6 | `dynbudget_qscore_earlylayer_geomcal` | `final_eval` | `0.4563` | 0.6831 | 0.3213 | 0.3194 | `-0.0005` |
| 7 | `qquery_earlylayer_geomcal` | `final_eval` | `0.4561` | 0.6850 | 0.3171 | 0.3186 | `-0.0007` |

High-level read from the final table:

- Hammer v2 produced a new best observed score: `0.4608`
- 5 of the 7 realized runs beat the prior `0.4568` frontier
- all top 4 runs use LM visual adapters
- the best bridge-only result was `qquery_dynbudget_earlylayer_geomcal` at `0.4576`

## Combination Evidence

Because the anchor was skipped, the clean control for this report is the prior best official frontier:

- `safeqcond_earlylayer_geomcal_frontier`: `0.4568`

That makes the Hammer deltas slightly less exact than a same-sweep control would have been, but still fully usable.

### 1. `qquery` alone

- prior best carry-forward control: `0.4568`
- `qquery_earlylayer_geomcal`: `0.4561`

Observed delta:

- `-0.0007` versus the prior best control

This means question-derived queries alone were competitive, but not enough to beat the established best bridge.

### 2. `dynbudget` alone

- prior best carry-forward control: `0.4568`
- `dynbudget_qscore_earlylayer_geomcal`: `0.4563`

Observed delta:

- `-0.0005` versus the prior best control

This means adaptive token scoring alone also landed close, but still did not clear the prior frontier.

### 3. `qquery + dynbudget`

- `qquery_earlylayer_geomcal`: `0.4561`
- `dynbudget_qscore_earlylayer_geomcal`: `0.4563`
- `qquery_dynbudget_earlylayer_geomcal`: `0.4576`

Observed combination deltas:

- `+0.0015` over `qquery`
- `+0.0013` over `dynbudget`
- `+0.0008` over the prior `0.4568` frontier

This is the clearest Hammer evidence that extraction and compression improvements do stack on the bridge side, even though neither single-direction run won alone.

### 4. `adapters` on the carry-forward bridge

- prior best carry-forward control: `0.4568`
- `adapter_safeqcond_earlylayer_geomcal`: `0.4591`

Observed delta:

- `+0.0023` over the prior best control

This is the strongest single-direction gain in the realized sweep.

### 5. `qquery + adapters`

- `qquery_earlylayer_geomcal`: `0.4561`
- `adapter_safeqcond_earlylayer_geomcal`: `0.4591`
- `qquery_adapter_earlylayer_geomcal`: `0.4594`

Observed combination deltas:

- `+0.0033` over `qquery`
- `+0.0003` over the adapter baseline

This says qquery became mildly positive once the LM could revisit visual tokens in-layer.

### 6. `dynbudget + adapters`

- `dynbudget_qscore_earlylayer_geomcal`: `0.4563`
- `adapter_safeqcond_earlylayer_geomcal`: `0.4591`
- `dynbudget_adapter_earlylayer_geomcal`: `0.4602`

Observed combination deltas:

- `+0.0039` over `dynbudget`
- `+0.0011` over the adapter baseline

This is a stronger pairwise gain than `qquery + adapters`, which suggests preserved detail mattered more than qquery alone once the LM had deeper visual access.

### 7. Full Hammer stack

- `qquery_dynbudget_earlylayer_geomcal`: `0.4576`
- `qquery_adapter_earlylayer_geomcal`: `0.4594`
- `dynbudget_adapter_earlylayer_geomcal`: `0.4602`
- `qquery_dynbudget_adapter_earlylayer_geomcal`: `0.4608`

Observed combination deltas:

- `+0.0032` over `qquery + dynbudget`
- `+0.0014` over `qquery + adapters`
- `+0.0006` over `dynbudget + adapters`
- `+0.0040` over the pre-Hammer best frontier

This is the best score in the sweep and the clearest sign that Hammer’s three targeted bottlenecks were not fully redundant.

## Answer-Type Patterns

Best `yes/no`:

- `dynbudget_adapter_earlylayer_geomcal`: `0.6897`
- `qquery_dynbudget_adapter_earlylayer_geomcal`: `0.6895`
- `qquery_dynbudget_earlylayer_geomcal`: `0.6886`

Best `number`:

- `adapter_safeqcond_earlylayer_geomcal`: `0.3229`
- `dynbudget_qscore_earlylayer_geomcal`: `0.3213`
- `qquery_adapter_earlylayer_geomcal`: `0.3207`

Best `other`:

- `qquery_dynbudget_adapter_earlylayer_geomcal`: `0.3244`
- `adapter_safeqcond_earlylayer_geomcal`: `0.3242`
- `dynbudget_adapter_earlylayer_geomcal`: `0.3237`

Interpretation preserved by these splits:

- the adapter family owns the overall ranking mainly by lifting `yes/no` and `other`
- the full Hammer stack wins overall because it combines near-best `yes/no` with the best `other`
- the best `number` score came from the simpler adapter-on-anchor branch, not from the full stack
- bridge-only Hammer variants stayed competitive, but their wins were too small on `other` to take the frontier

## Throughput and Cost Signal

All realized Hammer v2 runs used the same layout:

- train `batch_size=192`, `grad_accum_steps=1`
- eval `batch_size=192`

So `steps/s` is directly comparable across the whole set.

| Run | Final overall | Last logged train steps/s | Full-eval steps/s |
|---|---:|---:|---:|
| `qquery_earlylayer_geomcal` | `0.4561` | `5.13` | `2.41` |
| `dynbudget_qscore_earlylayer_geomcal` | `0.4563` | `5.08` | `2.57` |
| `qquery_dynbudget_earlylayer_geomcal` | `0.4576` | `5.05` | `2.57` |
| `qquery_adapter_earlylayer_geomcal` | `0.4594` | `4.78` | `1.89` |
| `qquery_dynbudget_adapter_earlylayer_geomcal` | `0.4608` | `4.73` | `1.97` |
| `dynbudget_adapter_earlylayer_geomcal` | `0.4602` | `4.72` | `1.92` |
| `adapter_safeqcond_earlylayer_geomcal` | `0.4591` | `4.64` | `1.85` |

Cost read:

- the bridge-only families were the fastest to train and evaluate after batched KV-cache landed
- the adapter families paid a real but not catastrophic speed cost
- that speed cost bought the entire top 4 of the ranking
- among the top 3 runs, `qquery_dynbudget_adapter_earlylayer_geomcal` was both the most accurate and the fastest evaluator

## Reliability Notes

### 1. Periodic evals are approximate

Periodic checks are only `100` validation batches, while final scores are full-val official evals.

Use periodic curves for:

- collapse detection
- rough slope shape

Do not use them for:

- precise branch ranking
- small combination decisions

Within this clean Hammer v2 set, the `9000`-step periodic-to-final drift still ranged from:

- `0.0000` (`dynbudget_qscore_earlylayer_geomcal`)
- to `+0.0023` (`dynbudget_adapter_earlylayer_geomcal`)
- to `-0.0017` (`qquery_dynbudget_earlylayer_geomcal`)

### 2. Batched KV-cache is part of the authoritative setup

The completed Hammer v2 bridge-only results use:

- `--eval_use_kv_cache`
- `--eval_kv_cache_mode batched`

That path was separately checked in `tasks/mm_bridge/docs/24_hammer_kvcache_correctness_report_2026-03-12.md` on real checkpoints before this sweep was launched.

### 3. This sweep is cleaner than high-entropy

For the realized Hammer v2 ranking set:

- no `eval_only` finals were needed
- no resumed-training caveat applies to the reported 7 runs
- every ranking number here comes from an in-run `final_eval` on the full validation set

## Compact Takeaways

What this sweep established, without projecting beyond the evidence:

- Hammer v2 produced a new best observed result at `0.4608`
- the full stack `qquery + dynbudget + adapters` won the sweep
- LM visual adapters were the strongest single new direction
- `qquery` and `dynbudget` alone were near-frontier but not enough individually
- `qquery + dynbudget` was the strongest bridge-only Hammer stack at `0.4576`
- 5 of the 7 realized Hammer runs beat the old `0.4568` frontier
- the top 4 runs were all adapter-based, which shifts the strongest remaining evidence toward deeper LM-side fusion rather than bridge-only extraction/compression changes alone
