# Nail Sweep Report - 2026-03-13

## Scope

This note compiles the completed Nail sweep from:

- `tasks/mm_bridge/docs/27_nail_sweep_plan_2026-03-13.md`
- `tasks/mm_bridge/scripts/launch_nail_sweep_v1.sh`
- sweep bundle `logs/mmnail_v1_20260313_112243`
- per-run logs under `logs/mmnail_v1_*`
- comparison context from `tasks/mm_bridge/docs/26_hammer_v2_sweep_report_2026-03-13.md`

The purpose here is retrospective only:

- record what actually ran
- record which scores are authoritative
- preserve the evidence that matters when choosing later sweeps

This is not a forward plan.

## Naming and Provenance

Nail is cleaner than Hammer on naming:

- the launcher bundle uses `SWEEP_ID=mmnail_v1_<stamp>`
- the actual run IDs also use `RUN_PREFIX=mmnail_v1`

Authoritative completed bundle for this report:

- `logs/mmnail_v1_20260313_112243`

Non-authoritative Nail launcher artifact:

- dry-run bundle `logs/mmnail_v1_20260313_095755`

The authoritative run set contains 10 realized run IDs:

- 3 eval-only corruption aliases of the Hammer-best checkpoint
- 7 train-and-final-eval Nail architecture runs

All 7 Nail architecture runs reached:

- `step_9000.tar`
- `fixed_eval_val_answers.jsonl` with `tag=final_eval`

The 3 corruption runs reached:

- `step_9000.tar` as a checkpoint alias to the Hammer-best source run
- `fixed_eval_val_answers.jsonl` with `tag=eval_only`

## Sweep Definition

The authoritative Nail launcher followed the revised run draft appended to the Nail plan.

Common policy for the 7 train runs:

- effective batch size `192`
- train layout `192 x 1`
- eval batch size `192`
- target step `9000`
- `eval_every=1000`
- periodic evals on `100` val batches
- final eval on full val (`final_eval_batches=0`, `eval_fraction=1.0`)
- official scorer
- `--eval_use_kv_cache --eval_kv_cache_mode batched`

Common policy for the corruption suite:

- no training
- eval-only scoring of the existing `mmhammer_v2_qquery_dynbudget_adapter_earlylayer_geomcal` `step_9000.tar`
- corruption modes: `shuffle`, `zero`, `random_swap`

Important comparison caveats:

- periodic evals are still partial-val checks only
- final scores below are the full-val official numbers
- the corruption suite uses `eval_only` alias runs and should be read as a diagnostic of the Hammer-best checkpoint, not as new training results

## Planned vs Executed Set

The revised Nail main queue in the launcher had 8 priority items:

1. `best_ckpt_image_corruptions`
2. `qquery_dynbudget_adapter_d3_cap64`
3. `qquery_dynbudget_adapter_d2_cap96`
4. `lmmeanqquery_dynbudget_adapter_d3_cap64`
5. `attnqquery_dynbudget_adapter_d3_cap64`
6. `rolespecial_dynbudget_adapter_d3_cap64`
7. `lmmeanqquery_dynbudget_adapter_d3_cap96`
8. `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64`

The realized sweep matched that revised main queue exactly.

Because the first queue item was a corruption suite, the realized run IDs were:

1. `best_ckpt_image_shuffle`
2. `best_ckpt_image_zero`
3. `best_ckpt_random_image_swap`
4. `qquery_dynbudget_adapter_d3_cap64`
5. `qquery_dynbudget_adapter_d2_cap96`
6. `lmmeanqquery_dynbudget_adapter_d3_cap64`
7. `attnqquery_dynbudget_adapter_d3_cap64`
8. `rolespecial_dynbudget_adapter_d3_cap64`
9. `lmmeanqquery_dynbudget_adapter_d3_cap96`
10. `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64`

Optional seed runs and bridge-pretraining runs were not part of the executed Nail queue.

## Final Ranking

Reference frontier entering Nail:

- previous best official full-val run: `qquery_dynbudget_adapter_earlylayer_geomcal`
- score: `0.4608`
- source: `tasks/mm_bridge/docs/26_hammer_v2_sweep_report_2026-03-13.md`

Final Nail architecture ranking:

| Rank | Run | Final tag | Overall | Yes/No | Number | Other | Delta vs `0.4608` |
|---|---|---|---:|---:|---:|---:|---:|
| 1T | `lmmeanqquery_dynbudget_adapter_d3_cap64` | `final_eval` | `0.4653` | 0.6927 | 0.3230 | 0.3298 | `+0.0045` |
| 1T | `lmmeanqquery_dynbudget_adapter_d3_cap96` | `final_eval` | `0.4653` | 0.6927 | 0.3230 | 0.3298 | `+0.0045` |
| 3 | `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64` | `final_eval` | `0.4643` | 0.6900 | 0.3200 | 0.3306 | `+0.0035` |
| 4 | `attnqquery_dynbudget_adapter_d3_cap64` | `final_eval` | `0.4624` | 0.6847 | 0.3204 | 0.3307 | `+0.0016` |
| 5 | `qquery_dynbudget_adapter_d3_cap64` | `final_eval` | `0.4617` | 0.6892 | 0.3203 | 0.3260 | `+0.0009` |
| 6 | `qquery_dynbudget_adapter_d2_cap96` | `final_eval` | `0.4608` | 0.6895 | 0.3182 | 0.3244 | `+0.0000` |
| 7 | `rolespecial_dynbudget_adapter_d3_cap64` | `final_eval` | `0.4602` | 0.6858 | 0.3218 | 0.3251 | `-0.0006` |

High-level read from the final table:

- Nail produced a new best observed score: `0.4653`
- the top two runs tied at four-decimal precision
- the biggest new gain came from stronger qquery generation, not from cap increases alone
- role specialization helped only when combined with the stronger LM-mean qquery path

## Corruption Suite

The corruption suite re-scored the Hammer-best checkpoint:

- source checkpoint: `mmhammer_v2_qquery_dynbudget_adapter_earlylayer_geomcal`
- clean reference score: `0.4608`

Corruption results:

| Corruption run | Final tag | Overall | Yes/No | Number | Other | Delta vs clean `0.4608` |
|---|---|---:|---:|---:|---:|---:|
| `best_ckpt_image_shuffle` | `eval_only` | `0.4514` | 0.6873 | 0.3159 | 0.3078 | `-0.0094` |
| `best_ckpt_random_image_swap` | `eval_only` | `0.4019` | 0.6590 | 0.2944 | 0.2345 | `-0.0589` |
| `best_ckpt_image_zero` | `eval_only` | `0.3813` | 0.6453 | 0.2534 | 0.2141 | `-0.0795` |

What this preserves:

- the model is materially image-dependent
- zeroing the image causes a large collapse
- random image swap also causes a large collapse
- plain shuffle hurts less than zero/swap, but still drops the model by almost a full point

So Nail did not discover an image-independent frontier artifact.

## Combination Evidence

Because Nail held the family narrower than Hammer, the most important comparisons are local deltas inside the adapter-centered mainline.

### 1. Adapter depth `2 -> 3`

- base Hammer-best reference: `0.4608`
- `qquery_dynbudget_adapter_d3_cap64`: `0.4617`

Observed delta:

- `+0.0009`

This says deeper adapters were positive, but only modestly so by themselves.

### 2. Cap `64 -> 96` without stronger qquery

- base Hammer-best reference: `0.4608`
- `qquery_dynbudget_adapter_d2_cap96`: `0.4608`

Observed delta:

- `+0.0000`

This says a larger dynbudget cap alone did not move the frontier.

### 3. LM-mean qquery on top of deeper adapters

- `qquery_dynbudget_adapter_d3_cap64`: `0.4617`
- `lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4653`

Observed delta:

- `+0.0036`

This is the clearest positive result in the sweep.

### 4. Attention-derived qquery on top of deeper adapters

- `qquery_dynbudget_adapter_d3_cap64`: `0.4617`
- `attnqquery_dynbudget_adapter_d3_cap64`: `0.4624`

Observed delta:

- `+0.0007`

This is positive, but much smaller than the LM-mean qquery jump.

### 5. Role specialization on top of deeper adapters

- `qquery_dynbudget_adapter_d3_cap64`: `0.4617`
- `rolespecial_dynbudget_adapter_d3_cap64`: `0.4602`

Observed delta:

- `-0.0015`

So role specialization alone was not helpful inside the current family.

### 6. Larger cap on top of the LM-mean qquery win

- `lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4653`
- `lmmeanqquery_dynbudget_adapter_d3_cap96`: `0.4653`

Observed delta:

- `+0.0000`

Within logged precision, the larger cap bought nothing once the stronger qquery path was already in place.

### 7. Role specialization on top of the LM-mean qquery win

- `lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4653`
- `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4643`

Observed delta:

- `-0.0010`

So role specialization still did not help, even in the stronger qquery branch.

## Answer-Type Patterns

Best `yes/no`:

- `lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.6927`
- `lmmeanqquery_dynbudget_adapter_d3_cap96`: `0.6927`
- `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.6900`

Best `number`:

- `lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.3230`
- `lmmeanqquery_dynbudget_adapter_d3_cap96`: `0.3230`
- `rolespecial_dynbudget_adapter_d3_cap64`: `0.3218`

Best `other`:

- `attnqquery_dynbudget_adapter_d3_cap64`: `0.3307`
- `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.3306`
- `lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.3298`

Interpretation preserved by these splits:

- the LM-mean qquery win came mainly from stronger `yes/no` and `number`
- the attention-derived qquery branch had the best `other`, but it did not convert that into the best overall score
- role specialization was not a broad win, even though it stayed competitive on `other`

## Throughput and Cost Signal

All 7 Nail train runs used the same layout:

- train `batch_size=192`, `grad_accum_steps=1`
- eval `batch_size=192`

So `steps/s` is directly comparable across the full architecture set.

| Run | Final overall | Last logged train steps/s | Full-eval steps/s |
|---|---:|---:|---:|
| `qquery_dynbudget_adapter_d2_cap96` | `0.4608` | `4.60` | `1.96` |
| `qquery_dynbudget_adapter_d3_cap64` | `0.4617` | `4.50` | `1.94` |
| `lmmeanqquery_dynbudget_adapter_d3_cap64` | `0.4653` | `4.43` | `1.94` |
| `lmmeanqquery_dynbudget_adapter_d3_cap96` | `0.4653` | `4.41` | `1.94` |
| `attnqquery_dynbudget_adapter_d3_cap64` | `0.4624` | `4.37` | `1.84` |
| `rolespecial_dynbudget_adapter_d3_cap64` | `0.4602` | `4.37` | `1.96` |
| `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64` | `0.4643` | `4.33` | `1.96` |

Cost read:

- Nail stayed in a tight throughput band despite the richer qquery work
- the best run was only modestly slower than the Hammer baseline
- the two tied winners did not pay a catastrophic runtime penalty for the score gain
- the attention-derived qquery run was the slowest evaluator without being the best scorer

## Reliability Notes

### 1. Periodic evals are still approximate

Periodic checks are only `100` validation batches, while final scores are full-val official evals.

Use periodic curves for:

- collapse detection
- rough slope shape

Do not use them for:

- precise branch ranking
- small architecture decisions at the top of the table

The largest periodic-to-final drifts in Nail were:

- `lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4684 -> 0.4653` (`-0.0031`)
- `lmmeanqquery_dynbudget_adapter_d3_cap96`: `0.4684 -> 0.4653` (`-0.0031`)
- `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4681 -> 0.4643` (`-0.0038`)

So the Nail top-end was visibly overestimated by partial-val periodic checks.

### 2. The corruption suite is authoritative for diagnostics, not for architecture ranking

Those runs are:

- `eval_only`
- alias loads of the Hammer-best checkpoint
- not new trained checkpoints

They should be used as image-dependence evidence, not as new frontier candidates.

### 3. The architecture sweep itself completed cleanly

For the 7 Nail train runs:

- no dry-run artifacts are used for ranking
- no post-hoc rescue evals were needed
- every ranking number here comes from an in-run `final_eval` on the full validation set

## Compact Takeaways

What this sweep established, without projecting beyond the evidence:

- Nail produced a new best observed result at `0.4653`
- the winning change was stronger LM-mean qquery generation, not larger cap alone
- deeper adapters helped a little on their own, but not nearly as much as the stronger qquery change
- increasing the dynbudget cap from `64` to `96` did not help by itself and did not improve the LM-mean qquery winner
- role specialization was not a clean positive inside this adapter-centered family
- the corruption suite showed that the frontier checkpoint still depends materially on image input
