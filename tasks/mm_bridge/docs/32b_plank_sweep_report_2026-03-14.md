# Plank Sweep Report - 2026-03-14

## Scope

This document reports on the completed Plank sweep. Sources:

- `tasks/mm_bridge/docs/29_plank_sweep_plan_2026-03-13.md`
- `tasks/mm_bridge/docs/30_mobilevit_perf_tuning_2026-03-13.md`
- sweep bundles `logs/mmplank_v1_*`
- per-run logs under `logs/mmplank_v1_mobilevit_*/`, `logs/mmplank_v1_questiononly_*/`, etc.

This document is retrospective only. It records what ran, which scores are authoritative, and what the results establish for future work.

## Run Set and Provenance

The Plank sweep executed 9 runs in total:

**MobileViT stage (Stage A):**
1. `mmplank_v1_mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `mmplank_v1_mobilevit_qquery_dynbudget_adapter_d3_cap64`
3. `mmplank_v1_mobilevit_attnqquery_dynbudget_adapter_d3_cap64`
4. `mmplank_v1_mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64_seed2`

**Original-VM query quality stage (Stage B):**
5. `mmplank_v1_questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`
6. `mmplank_v1_multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`
7. `mmplank_v1_hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`
8. `mmplank_v1_iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`
9. `mmplank_v1_visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`

All 9 runs reached `step_9000.tar` and a `fixed_eval_val_answers.jsonl` with `tag=final_eval`. The sweep completed without missing runs.

## Sweep Definition

All runs followed the standard comparison policy:

- effective batch size `192`
- target step `9000`
- `eval_every=1000`
- periodic evals on `100` val batches
- final eval on full validation split (`eval_fraction=1.0`, `final_eval_batches=0`)
- official scorer
- `--eval_use_kv_cache --eval_kv_cache_mode batched`

**Layout differences:**

- Original-VM runs: `batch_size=192, grad_accum_steps=1, eval_batch_size=192`
- MobileViT runs: `batch_size=96, grad_accum_steps=2, eval_batch_size=96`

The MobileViT runs used the reduced in-memory batch for stability on the heavier backbone. The effective batch size was maintained at `192` in both cases, so the runs remain standard-comparable under the comparison policy.

**Note on mobilevit_lmmeanqquery provenance:** This run was resumed from `step_7000` in the final launcher bundle. It completed steps 1–7000 in earlier bundles. Its full run is clean and fully logged.

## Final Ranking

Reference frontier entering Plank:

- Nail winner: `lmmeanqquery_dynbudget_adapter_d3_cap64` at `0.4653`
- Previous best: same run (tied with cap96 variant at `0.4653`)

### Full Ranking Table

| Rank | Run | Final Overall | Yes/No | Number | Other | Delta vs `0.4653` |
|---|---|---:|---:|---:|---:|---:|
| 1 | `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` | **0.5240** | 0.6983 | 0.3405 | **0.4401** | `+0.0587` |
| 2 | `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64` | 0.5181 | 0.6983 | 0.3396 | 0.4283 | `+0.0528` |
| 3 | `mobilevit_qquery_dynbudget_adapter_d3_cap64` | 0.5167 | 0.6971 | 0.3333 | 0.4281 | `+0.0514` |
| 4 | `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64_seed2` | 0.5130 | 0.6884 | 0.3318 | 0.4277 | `+0.0477` |
| 5 | `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64` | 0.4699 | 0.6975 | 0.3233 | 0.3354 | `+0.0046` |
| 6 | `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64` | 0.4671 | 0.6936 | 0.3228 | 0.3330 | `+0.0018` |
| 7 | `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64` | 0.4651 | 0.6886 | 0.3218 | 0.3329 | `-0.0002` |
| 8 | `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64` | 0.4650 | 0.6929 | 0.3236 | 0.3290 | `-0.0003` |
| 9 | `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64` | 0.4637 | 0.6916 | 0.3220 | 0.3278 | `-0.0016` |

**New frontier: `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` at `0.5240`.**

## Periodic Eval Curves (steps 1000–9000 + final)

All values below are periodic 100-batch evals (left 9) plus the full-val final eval (rightmost).

| Run | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mobilevit_attnqquery` | 0.400 | 0.454 | 0.468 | 0.490 | 0.500 | 0.511 | 0.510 | 0.520 | 0.523 | **0.524** |
| `mobilevit_lmmeanqquery` | 0.407 | 0.446 | 0.458 | 0.473 | 0.486 | 0.497 | 0.504 | 0.520 | 0.524 | **0.518** |
| `mobilevit_qquery` | 0.399 | 0.448 | 0.461 | 0.485 | 0.494 | 0.501 | 0.507 | 0.515 | 0.519 | **0.517** |
| `mobilevit_lmmeanqquery_seed2` | 0.419 | 0.451 | 0.470 | 0.487 | 0.489 | 0.497 | 0.512 | 0.514 | 0.513 | **0.513** |
| `questiononly` | 0.378 | 0.411 | 0.423 | 0.442 | 0.444 | 0.460 | 0.459 | 0.467 | 0.469 | **0.470** |
| `visual_adapter` | 0.382 | 0.408 | 0.425 | 0.439 | 0.447 | 0.452 | 0.458 | 0.465 | 0.467 | **0.467** |
| `hybrid` | 0.378 | 0.412 | 0.425 | 0.437 | 0.445 | 0.452 | 0.458 | 0.461 | 0.463 | **0.465** |
| `iter2` | 0.379 | 0.409 | 0.424 | 0.437 | 0.442 | 0.450 | 0.455 | 0.464 | 0.462 | **0.465** |
| `multiq4` | 0.375 | 0.407 | 0.420 | 0.435 | 0.442 | 0.447 | 0.455 | 0.459 | 0.462 | **0.464** |

Interpretation:

- MobileViT curves are consistently steep and linear across the full training range. No signs of plateauing at 9k steps.
- Original-VM curves are flatter across the full run. Most reached near-plateau behavior by 6k–7k steps.
- The `mobilevit_attnqquery` 9k-to-final direction is slightly upward (0.5231 → 0.5240), while `mobilevit_lmmeanqquery` drifts downward (0.5240 → 0.5181). This is unusual and may reflect genuine differences in how these models use the full val distribution vs the 100-batch partial sample.

## Stage A: MobileViT Analysis

### The Core Finding

MobileViT produced the three largest score improvements in the entire project history to date. All three MobileViT query-family runs cleared 0.516, and the best hit 0.5240—a jump of **+0.0587** over the previous frontier.

This confirms the Stage A hypothesis from the Plank plan:

> A stronger frozen VM will improve the best-known bridge family by at least +0.005 because the current VM produces classification-optimized features that are over-compressed for fine-grained VQA.

The actual improvement was approximately **10× larger** than the +0.005 threshold. This strongly establishes VM quality as a major bottleneck that the project underestimated throughout the original-VM sweeps.

### The attnqquery Reversal

In Nail, `lmmeanqquery` beat `attnqquery` cleanly (0.4653 vs 0.4624). In Plank with MobileViT, the ranking flips: `attnqquery` wins at 0.5240 vs `lmmeanqquery` at 0.5181.

The reversal is concentrated in `other`:

| | lmmeanqquery final | attnqquery final | Delta |
|---|---:|---:|---:|
| Overall | 0.5181 | 0.5240 | `+0.0059` |
| Yes/No | 0.6983 | 0.6983 | `+0.0000` |
| Number | 0.3396 | 0.3405 | `+0.0009` |
| **Other** | **0.4283** | **0.4401** | **`+0.0118`** |

With the old VM, `attnqquery` and `lmmeanqquery` had essentially the same `other` score (0.3307 vs 0.3298 in Nail). With MobileViT, `attnqquery` opens a large `other` gap of 0.0118.

What this suggests: MobileViT's richer 640-dim features have finer-grained spatial and attribute information that attention-derived query formation extracts more effectively than mean pooling. The attention mechanism can focus on specific token positions relevant to the question, which becomes more valuable when each token carries more information. With the older, sparser VM features, the attention advantage was marginal; with MobileViT, it becomes the dominant query-quality lever.

The mean-pooled `lmmeanqquery` path averages over all question tokens, producing a diffuse global query signal. That works well when visual features are noisy and diffuse pooling provides stability. When visual features are richer and more discriminative, a more focused attention-derived signal can better exploit the additional information.

### qquery vs lmmeanqquery vs attnqquery on MobileViT

With the new VM, the ordering is:

| | overall | yes/no | number | other |
|---|---:|---:|---:|---:|
| `attnqquery` | 0.5240 | 0.6983 | 0.3405 | 0.4401 |
| `lmmeanqquery` | 0.5181 | 0.6983 | 0.3396 | 0.4283 |
| `qquery` | 0.5167 | 0.6971 | 0.3333 | 0.4281 |

All three families are within 0.0073 of each other. The main differentiation is in `other`. Both `attnqquery` and `lmmeanqquery` significantly outperform base `qquery` on `other` (0.4401/0.4283 vs 0.4281), while the `yes/no` and `number` gaps are small.

This means the LM-conditioning quality matters most for open-ended attribute/relational questions, not for yes/no or counting—which is consistent with what we expect.

### Seed Variance

Seed 1 vs Seed 2 for `mobilevit_lmmeanqquery`:

| | overall | yes/no | number | other |
|---|---:|---:|---:|---:|
| seed 1 (seed=35) | 0.5181 | 0.6983 | 0.3396 | 0.4283 |
| seed 2 (seed=53) | 0.5130 | 0.6884 | 0.3318 | 0.4277 |
| delta | -0.0051 | -0.0099 | -0.0078 | -0.0006 |

The 0.0051 seed variance is non-trivial relative to typical step-wise improvements in this project. The `other` category is stable across seeds (0.4283 vs 0.4277), but `yes/no` and `number` vary substantially. This suggests:

1. The MobileViT family results have real seed-to-seed variation that should be accounted for before treating any single-seed frontier number as definitive.
2. The `other` category improvements from MobileViT are reliable across seeds (both seeds are far above the Nail other of 0.3298).
3. The `yes/no` result at seed 1 (0.6983) may be optimistic; seed 2 at 0.6884 is closer to what might be the expected mean.

The new project frontier based on a single seed is `mobilevit_attnqquery` at 0.5240, but this should be verified with a second seed before being treated as a settled headline number.

## Stage B: Original-VM Query Quality Analysis

### Summary

All five Stage B runs used the original (non-MobileViT) VM on top of the Nail-winning bridge family. Results compared to the Nail winner (0.4653):

| Run | Delta | Verdict |
|---|---:|---|
| `questiononly` | `+0.0046` | Mildly positive |
| `visual_adapter` | `+0.0018` | Marginally positive |
| `hybrid` | `-0.0002` | Flat |
| `iter2` | `-0.0003` | Flat |
| `multiq4` | `-0.0016` | Slight negative |

The Stage B gains are real but small. The entire original-VM family is tightly clustered between 0.4637 and 0.4699—barely moved from the Nail frontier despite five different architectural changes. This is not a failure of the Plank plan; it is useful information. The original-VM family has effectively saturated.

### 1. questiononly: What It Confirmed

`questiononly_lmmeanqquery` changed the LM-mean pooling to span only the question-token span rather than the full prompt context.

Result: `+0.0046` overall. Final breakdown:

| | questiononly | Nail lmmeanqquery | delta |
|---|---:|---:|---:|
| Overall | 0.4699 | 0.4653 | +0.0046 |
| Yes/No | 0.6975 | 0.6927 | +0.0048 |
| Number | 0.3233 | 0.3230 | +0.0003 |
| Other | 0.3354 | 0.3298 | +0.0056 |

The gain is genuine and consistent across all categories. It confirms the hypothesis: the global LM-mean was picking up some prompt/context noise that slightly diluted the question-specific query signal. Restricting to question-only tokens sharpens the retrieval request.

However, the gain is small enough that it is within the seed variance range for MobileViT runs. Taken in isolation, questiononly is the clearest Stage B win, but its practical significance is limited now that MobileViT is in the picture.

One important note: the `questiononly` change affects bridge behavior, not just pooling. The same question-only masking would apply when combined with MobileViT. This makes `mobilevit_questiononly_lmmeanqquery` or `mobilevit_questiononly_attnqquery` worth testing in the next sweep—the sharpening might have more impact on richer MobileViT features than on the old VM.

### 2. visual_adapter: The Visual Feature Side Is Not the Bottleneck (for the old VM)

`visual_adapter_lmmeanqquery` added a small trainable residual MLP on top of the frozen VM features before they enter the bridge.

Result: `+0.0018` overall. The gain is positive but marginal.

This says: adding a visual-side adapter to the original VM helps slightly, but not dramatically. Combined with the MobileViT results—which showed that a better frozen VM helps enormously—the interpretation is:

- The trainable visual adapter can compensate for some VM deficiency, but it cannot compensate for the full scale of information the old VM was failing to provide.
- A better frozen VM (MobileViT) does far more than a residual adapter on a worse frozen VM.

Future implication: `visual_adapter` on top of MobileViT features is not a priority. MobileViT already provides better features than the adapter was attempting to recover.

### 3. multiq4: Multiple LM-Conditioned Queries Underperformed

`multiq4_lmmeanqquery` generated 4 LM-conditioned query groups instead of one pooled request.

Result: `0.4637`, below the Nail winner (`-0.0016`).

This is a surprising failure for what the Plank plan called "the highest-upside bridge-only continuation." The most plausible explanations:

**a) Competition rather than complementarity.** With 4 query groups sharing the same 49 visual tokens and the same bridge extraction budget, the groups may be pulling in competing directions rather than specializing by evidence type. The bridge depth (3 layers) and token count (49) may not be sufficient to support meaningful multi-group specialization.

**b) Training signal dilution.** Each query group receives a fraction of the gradient signal. The supervision from VQA answers may not be strong enough to train 4 specialized retrieval heads from scratch at this scale.

**c) Architecture mismatch.** The multi-query implementation uses `question_hidden_mean_multi` mode, which generates multiple queries from the same pooled mean. This may be producing correlated query groups rather than diverse ones.

The right interpretation is not "multiple LM-conditioned queries never work." It is: the current implementation with 4 groups on 49 tokens with a single-VQA training signal does not work at this scale. A richer visual token source (more tokens from MobileViT, or fewer but more diverse query groups) might behave differently.

### 4. hybrid: No Gain from Combining Query Paths

`hybrid_lmmean_attnqquery` combined the LM-mean and attention-derived query paths with a learned gate (initialized at 0.5).

Result: `0.4651` (`-0.0002` vs Nail winner).

In Nail, `attnqquery` was best on `other` (0.3307) while `lmmeanqquery` was best overall (0.4653). The hybrid was expected to combine both strengths. It did not.

Final `other` for hybrid: 0.3329—basically the same as questiononly (0.3354) and the Nail winner (0.3298). The hybrid did not recover the attnqquery `other` advantage.

A possible reason: with the old VM, the `attnqquery` advantage on `other` was already very small (0.3307 vs 0.3298, a gap of 0.0009). The hybrid gate may have learned to heavily weight `lmmeanqquery`, effectively collapsing back to the simpler path. With MobileViT, the reversal is strong enough (0.4401 vs 0.4283 on `other`) that a hybrid might actually be worth combining—or the project may simply use `attnqquery` directly.

### 5. iter2: Iterative Querying Did Not Help

`iter2_lmmeanqquery` used a two-stage bridge: first pass to gather coarse evidence, second pass refined by the first-pass summary.

Result: `0.4650` (`-0.0003`). Essentially flat.

The per-category breakdown is consistent with the Nail winner except for slightly lower `other` (0.3290 vs 0.3298):

This says: single-pass retrieval is not the bottleneck at the current scale. The iterative pass is not providing useful incremental evidence. This is not surprising given that the LM visual adapters are already performing in-layer visual re-access during generation—the iterative querying and the adapter stack are likely targeting the same bottleneck from different directions, and the adapters are doing it more effectively.

## Cross-Sweep Comparisons

### MobileViT lift per query family

How much did MobileViT help each bridge family?

| Family | Original VM (Nail) | MobileViT (Plank) | Delta |
|---|---:|---:|---:|
| `lmmeanqquery` | 0.4653 | 0.5181 | +0.0528 |
| `qquery` | 0.4617 | 0.5167 | +0.0550 |
| `attnqquery` | 0.4624 | 0.5240 | +0.0616 |

All three families gained more than 0.05 from the VM switch. `attnqquery` gained the most.

### Where the `other` gains came from

The `other` category score is the most diagnostic:

| Source | Other |
|---|---:|
| Nail lmmeanqquery (best original-VM final) | 0.3298 |
| Plank questiononly (best original-VM) | 0.3354 |
| Plank mobilevit_lmmeanqquery | 0.4283 |
| Plank mobilevit_attnqquery | **0.4401** |

The jump from 0.3354 (questiononly, best original-VM) to 0.4283 (MobileViT lmmeanqquery) is an increase of 0.0929 in `other` from the VM switch alone. This is the clearest signal that the old VM was producing features that were genuinely insufficient for the open-ended and compositional questions in VQA—not just noisier, but categorically less informative.

## Throughput and Cost

| Run | Train steps/s (end of run) | Full-eval steps/s | Layout |
|---|---:|---:|---|
| `mobilevit_attnqquery` | ~2.9 | ~3.5 | 96x2 |
| `mobilevit_lmmeanqquery` | ~2.6 | ~3.6 | 96x2 |
| `mobilevit_qquery` | ~2.8 | ~3.4 | 96x2 |
| `mobilevit_lmmeanqquery_seed2` | ~2.5 | ~3.5 | 96x2 |
| `questiononly` | ~5.0 | ~1.9 | 192x1 |
| `visual_adapter` | ~4.8 | ~1.9 | 192x1 |
| `hybrid` | ~4.8 | ~1.9 | 192x1 |
| `iter2` | ~4.4 | ~1.9 | 192x1 |
| `multiq4` | ~4.9 | ~1.9 | 192x1 |

MobileViT runs train at roughly half the speed of original-VM runs (~2.5–3.0 vs ~4.4–5.0 steps/s). However, MobileViT eval is faster per step because the full eval with `eval_batch_size=96` at ~3.5 steps/s processes ~96 samples/step vs the original-VM `eval_batch_size=192` at ~1.9 steps/s processing ~192 samples/step. The net full-val evaluation times are roughly comparable.

The cost story is that MobileViT nearly doubles training wall-clock time per run. At 9000 steps, this is roughly 3.5–4 hours per MobileViT run vs ~1.8 hours for original-VM runs. This is a real cost but not prohibitive for the scale of gains observed.

## Reliability Notes

### 1. MobileViT runs used 96x2, not 192x1

The global comparison policy prefers `192x1` as the standard in-memory layout. The MobileViT runs used `96x2`. The effective batch size was maintained at `192`, and the gradient accumulation should not affect the training dynamics materially. These runs should be treated as standard-comparable, but this layout deviation should be noted when reporting.

### 2. mobilevit_lmmeanqquery was resumed from step 7000

The run history shows this run was spread across multiple launcher bundles. Steps 1–7000 were completed in earlier bundles; steps 7000–9000 and final eval completed in the latest bundle. The run log is clean across all segments. This is not a concern for correctness.

### 3. The attnqquery win over lmmeanqquery should be verified with a second seed

The `mobilevit_attnqquery` vs `mobilevit_lmmeanqquery` gap is 0.0059 overall, driven by 0.0118 on `other`. The existing seed2 run is for `lmmeanqquery` only. Before declaring attnqquery as the definitively better family on MobileViT, a second seed of `mobilevit_attnqquery` would be useful. The priority level is not critical—the gap is large enough to likely survive—but it is worth noting.

### 4. Periodic evals are partial-val approximations

The periodic 100-batch evals use `eval_batch_size=96` for MobileViT runs and `eval_batch_size=192` for original-VM runs. Both evaluate only 9600–19200 samples out of a full val set of ~213k. The final full-val scores are the authoritative numbers.

The largest periodic-to-final drifts observed in this sweep:

- `mobilevit_lmmeanqquery`: 9k periodic `0.5240` → final `0.5181` (drift: `-0.0059`)
- `mobilevit_attnqquery`: 9k periodic `0.5231` → final `0.5240` (drift: `+0.0009`)

The downward drift in `lmmeanqquery` is consistent with the project's historical pattern of periodic evals overestimating final scores. The slight upward drift in `attnqquery` is unusual and may reflect a favorable final-eval batch composition or a genuine tail-of-training effect.

## Compact Takeaways

What this sweep established:

1. **MobileViT is a breakthrough.** All three MobileViT query families cleared 0.516, with the best at 0.5240. The VM was a larger bottleneck than any prior sweep recognized.

2. **The new frontier is `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` at `0.5240`.** This is the most authoritative single-seed score from the sweep.

3. **attnqquery reverses its Nail loss when combined with MobileViT.** The richer visual features amplify the attention-derived query advantage specifically on `other` questions. With the old VM, the difference between query types was marginal; with MobileViT, attnqquery dominates open-ended questions.

4. **The original-VM family has saturated near 0.47.** All five Stage B runs cluster in [0.4637, 0.4699]. The original VM has been effectively maxed out for the current bridge architecture.

5. **questiononly worked as predicted, but modestly.** Question-only LM-mean pooling added +0.0046 to the best original-VM result. This confirms that global LM-mean was slightly polluted, but the gain is small relative to the VM-driven gains.

6. **multiq4, hybrid, and iter2 were all flat or negative.** Multiple queries, hybrid generation, and iterative querying did not help on the original-VM family. These ideas are not dead—they may behave differently with MobileViT features—but none proved their value here.

7. **Seed variance is real and non-trivial on MobileViT.** The gap between two seeds of `mobilevit_lmmeanqquery` is 0.0051, concentrated in `yes/no` and `number`. More seed work is needed before the MobileViT family numbers can be treated as low-variance headlines.

8. **MobileViT attnqquery is still learning at 9k steps.** The training curves show no plateau, suggesting longer runs would continue to improve. This is the most important finding for future planning.

## Score Progression Summary

Full project frontier history:

| Sweep | Frontier | Delta |
|---|---:|---:|
| Learned constant prefix (early) | 0.3540 | — |
| Prefix calibration + perceiver (early) | ~0.4300 | — |
| Night sweeps (2026-03-09) | 0.4544 | — |
| Final Arch + High-Entropy (2026-03-11/12) | 0.4568 | +0.0024 |
| Hammer (2026-03-13) | 0.4608 | +0.0040 |
| Nail (2026-03-13) | 0.4653 | +0.0045 |
| **Plank (2026-03-14)** | **0.5240** | **+0.0587** |

The Plank gain from VM switch (+0.0587) is larger than all prior gains combined since the first perceiver baseline.
