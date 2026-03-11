# Final Arch Report - 2026-03-11

## Part I: Compiled Background

This section recompiles the planning context from:

- `docs/mm_bridge_diagnostics/16_final_10h_arch_plan_2026-03-10.md`
- `docs/mm_bridge_diagnostics/17_new_arch_memory_probes_v1_2026-03-10.md`
- `docs/mm_bridge_diagnostics/18_final_arch_run_queue_v1_2026-03-10.md`

### 1. Planning Objective

The final architecture cycle was designed to balance three goals:

1. projected overall accuracy
2. project learning value
3. a controlled amount of novelty

The selected run order was:

1. leakage-safe question-conditioned perceiver
2. multi-scale bridge
3. early-layer feature bridge
4. oracle196 + adaptive token selection
5. geometry-aware prefix calibration
6. adaptive token selection v2
7. structured token roles
8. evidence-focused sparse bridge

The explicitly deferred ideas were:

- residual LM visual adapter
- dynamic token budgets
- slot attention
- token routing
- bridge pretraining

### 2. Memory-Probe Outcome

Before long runs, short Docker-backed probes were used to select practical `batch_size / grad_accum_steps` pairs.

Probe conclusions:

- leakage-safe qcond perceiver: `192x1`
- multi-scale perceiver: `128x2`
- geometry-aware calibration: `192x1`
- structured roles: `192x1`
- evidence sparse: `192x1`

Notes carried forward from the probe report:

- `multi-scale` was intentionally kept at `128x2` because it was the heaviest new branch.
- The other new branches all cleared `192x1`.
- These were treated as safe first-run settings, not absolute maxima.

### 3. Queue Policy

The final queue was built under the following run policy:

- no periodic evals
- one final eval only
- half-eval-set final evaluation by default
- restart-safe skip/resume
- fixed training sample budget across runs

Concrete queue:

1. `safeqcond_d3_main`
2. `multiscale_d3_main`
3. `earlylayer_encoder_d3_main`
4. `oracle196_topk64_main`
5. `geomcal_d3_main`
6. `topk32_d3_main`
7. `structuredroles_d3_exp`
8. `evidencesparse_d3_exp`

Run-time defaults used by the queue:

- `precision=bf16`
- `freeze_mode=bridge_plus_top_lm`
- `train_top_lm_layers=2`
- `prefix_calibration=on`
- `prefix_dropout=0.03`
- `eval_every=0`
- `eval_batches=0`
- `eval_fraction=0.5`
- `ckpt_every=1000`

### 4. Training-Budget Caveat

These runs were intentionally trained under the new fixed-sample budget:

- effective batch `192` -> `6000` steps
- effective batch `256` -> `4500` steps

That implies a total training budget of `1,152,000` samples per run.

This is materially smaller than the earlier `9000`-step frontier runs at effective batch `192`, which saw roughly:

- `192 * 9000 = 1,728,000` samples

So this sweep used about `33%` fewer training samples than the earlier best-accuracy frontier runs.

This matters for interpretation: these runs are valid architecture probes, but they are not fully apples-to-apples with the earlier `0.4544` frontier.

### 5. Evaluation Caveat

The final queue evaluated on half of the validation split (`eval_fraction=0.5`).

That means:

- within-sweep comparisons are clean
- direct comparison against earlier reports collected under a different eval regime should be treated as directional, not final

## Part II: Completed Sweep Analysis

### 1. Completion Status

All eight queue runs completed and logged a `final_eval`.

Completed runs:

- `mmarch_final_v1_20260310_safeqcond_d3_main`
- `mmarch_final_v1_20260310_multiscale_d3_main`
- `mmarch_final_v1_20260310_earlylayer_encoder_d3_main`
- `mmarch_final_v1_20260310_oracle196_topk64_main`
- `mmarch_final_v1_20260310_geomcal_d3_main`
- `mmarch_final_v1_20260310_topk32_d3_main`
- `mmarch_final_v1_20260310_structuredroles_d3_exp`
- `mmarch_final_v1_20260310_evidencesparse_d3_exp`

### 2. Final Ranking

Reference frontier from prior work:

- best prior run: `0.4544`
- source: `docs/mm_bridge_diagnostics/10_all_runs_structured_2026-03-10.md`
- same run at `step=6000`: `0.4410`
- source log: `logs/mmnight_bridge_v2_8h_20260309_234936_perceiver_d3_pd03_main/logfile.txt`

New sweep ranking:

| Rank | Run | Arch | Eff. Batch | Steps | Accuracy | Delta vs `0.4544` | Delta vs best-run `6k` pace (`0.4410`) | Yes/No | Number | Other |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `safeqcond_d3_main` | leakage-safe qcond perceiver | 192 | 6000 | `0.4460` | `-0.0084` | `+0.0050` | 0.6808 | 0.3133 | 0.3024 |
| 2 | `structuredroles_d3_exp` | structured roles | 192 | 6000 | `0.4435` | `-0.0109` | `+0.0025` | 0.6836 | 0.3064 | 0.2969 |
| 3 | `earlylayer_encoder_d3_main` | early-layer encoder perceiver | 192 | 6000 | `0.4429` | `-0.0115` | `+0.0019` | 0.6808 | 0.3108 | 0.2967 |
| 4 | `oracle196_topk64_main` | oracle196 + topk64 | 192 | 6000 | `0.4413` | `-0.0131` | `+0.0003` | 0.6838 | 0.3079 | 0.2918 |
| 5 | `geomcal_d3_main` | geometry-aware calibration | 192 | 6000 | `0.4406` | `-0.0138` | `-0.0004` | 0.6814 | 0.3074 | 0.2926 |
| 6 | `multiscale_d3_main` | multiscale perceiver | 256 | 4500 | `0.4398` | `-0.0146` | `-0.0012` | 0.6854 | 0.3053 | 0.2884 |
| 7 | `topk32_d3_main` | adaptive selection v2 | 192 | 6000 | `0.4363` | `-0.0181` | `-0.0047` | 0.6769 | 0.3079 | 0.2872 |
| 8 | `evidencesparse_d3_exp` | evidence sparse | 192 | 6000 | `0.4360` | `-0.0184` | `-0.0050` | 0.6811 | 0.3091 | 0.2828 |

### 3. Pace Comparison

If these runs are judged only against the prior endpoint `0.4544`, they all look short of the frontier.

That is incomplete.

The more relevant pacing comparison is the eventual-best perceiver run at its own `step=6000`, where it was only `0.4410`.

Against that `6k` reference:

- `safeqcond_d3_main` is ahead by `+0.0050`
- `structuredroles_d3_exp` is ahead by `+0.0025`
- `earlylayer_encoder_d3_main` is ahead by `+0.0019`
- `oracle196_topk64_main` is ahead by `+0.0003`
- `geomcal_d3_main` is essentially at parity

This materially changes the interpretation of the sweep.

The correct read is not "none of the new arches are competitive."

The correct read is:

- several new arches are below the old run's final endpoint
- but multiple new arches were on a better or comparable pace at the same training stage
- that leaves substantial alpha in follow-up runs at larger training budgets

### 4. High-Confidence Conclusions

#### A. The qcond leakage diagnosis was correct

The biggest project-level result in this sweep is that `safeqcond_d3_main` did not collapse.

Instead, it finished as the best run in the queue at `0.4460`.

Relative to the eventual-best perceiver run's `6k` checkpoint (`0.4410`), it was ahead by `+0.0050`.

That strongly supports the earlier diagnosis that the original qcond failure was caused by answer leakage from teacher-forced text, not by question-conditioning as a concept.

In practical terms:

- question-conditioned extraction is back on the table
- the implementation fix was meaningful
- qcond deserves follow-up rather than abandonment

#### B. Structured roles is the strongest new non-qcond arch

`structuredroles_d3_exp` reached `0.4435`, second in the sweep, and ahead of the best-run `6k` pace by `+0.0025`.

That is a strong result for a more novel bridge:

- it is competitive immediately
- it gives a more interpretable token story than generic latent tokens
- it appears more promising than naive sparsification

This is the strongest novelty-positive signal from the cycle.

#### C. Early-layer features are probably real, but not decisive yet

`earlylayer_encoder_d3_main` reached `0.4429`, also ahead of the best-run `6k` pace.

That is not enough to beat the earlier frontier, but it is good enough to say:

- the final latent is probably not the only useful feature source
- earlier features remain a viable direction
- the gain is plausible enough to justify a better-tuned rerun or multiscale follow-up

This is a positive result, just not a breakthrough.

#### D. Naive sparse evidence selection is not yet strong enough

The two weakest runs in the sweep were:

- `topk32_d3_main` at `0.4363`
- `evidencesparse_d3_exp` at `0.4360`

That suggests:

- simple token pruning is not enough by itself
- sparse evidence extraction likely needs better scoring, question guidance, or stronger training signal
- a novelty jump into sparse routing without stronger conditioning is premature

### 5. Medium-Confidence Conclusions

#### A. Oracle196 + topk64 is informative but not yet worth the cost

`oracle196_topk64_main` finished at `0.4413`, which is slightly ahead of the best-run `6k` pace.

Interpretation:

- the large-token-bank idea is not dead
- but the current version does not justify its throughput cost
- right now it looks more like a diagnostic tool than a frontier path

This is still useful because it argues against spending the next cycle on pure token-count inflation.

#### B. Geometry-aware calibration looks more combinational than standalone

`geomcal_d3_main` reached `0.4406`.

That is respectable and effectively on pace with the best old run at `6k`, but not enough to make it a new leading family yet.

Most likely interpretation:

- geometry-aware calibration is helping interface quality
- but its gain is not large enough alone
- it is more promising as a modifier on a stronger core arch than as a primary branch

#### C. Multiscale is still under-tested

`multiscale_d3_main` reached `0.4398`.

This result should be interpreted cautiously because it had the heaviest setup and the smallest update count:

- effective batch `256`
- only `4500` steps under the fixed-sample rule
- notably slower throughput

So the result is mediocre, but it is not conclusive evidence against multiscale.

Because it only ran to `4500` steps at effective batch `256`, its pace comparison is less direct than the `6000`-step runs.

### 6. Cross-Run Pattern

Across the whole sweep, the main gap to the earlier frontier appears in `other` answers.

Prior frontier run (`0.4544`) had:

- yes/no: `0.6889`
- number: `0.3125`
- other: `0.3134`

Best new run (`safeqcond_d3_main`) had:

- yes/no: `0.6808`
- number: `0.3133`
- other: `0.3024`

Interpretation:

- the new architectures are already competitive on `number`
- they are close but not leading on `yes/no`
- the main missing gain is still in `other`

That points to a likely bottleneck:

- evidence extraction is improving
- but semantic/open-ended alignment is still weaker than the best perceiver frontier

### 7. Main Interpretation of the Sweep

This sweep did not beat the established perceiver endpoint.

But endpoint comparison is not the whole story.

At the more relevant mid-training pace check, multiple new arches were already at or ahead of the eventual-best run's `6k` state.

So this sweep produced four important wins:

1. it validated leakage-safe qcond as a real path
2. it surfaced structured roles as a serious novel candidate
3. it kept early-layer features alive as a meaningful input-source direction
4. it showed that several new branches were not merely "close" but actually ahead of the old best run's `6k` pace

That is a good outcome for an architecture-probing cycle, especially given:

- `33%` less training budget than the earlier frontier runs
- half-val evaluation under a different comparison regime than the earlier frontier reports

### 8. Recommended Next Moves

If the next goal is score-first:

1. rerun `safeqcond_d3_main` at full frontier budget
2. rerun `structuredroles_d3_exp` at full frontier budget
3. rerun `earlylayer_encoder_d3_main` at full frontier budget

If the next goal is research value:

1. combine `safe qcond` with the stronger prior perceiver frontier settings
2. combine `geometry-aware calibration` with `safe qcond` or `structured roles`
3. revisit `multiscale` with a more permissive optimization budget before ruling it out

Current deprioritization signal:

- `topk32_d3_main`
- `evidencesparse_d3_exp`
- standalone `oracle196 + topk64` as a mainline frontier path

### 9. Bottom Line

The best result in this sweep is not "a new SOTA bridge endpoint."

The best result is that the sweep found believable next branches with real pace:

- `safe qcond` is now validated
- `structured roles` is worth serious follow-up
- `early-layer features` remain promising

And more importantly:

- multiple options were already at or above the eventual-best run's `6k` pace
- so the current scores likely understate the upside of these branches under longer training

That is enough signal to plan the next cycle around those three rather than dispersing effort across all eight directions equally.
