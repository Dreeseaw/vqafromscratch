# Senior Researcher Perspective: Forward Sweep Plan
Date: 2026-03-14

---

## How I Am Reading This Project

I am coming in as a senior researcher after reading the full trail of documents, sweep plans, and reports from the beginning of this work. I have no attachment to any specific architecture, no emotional stake in past results, and no preference for what should win next. What I have is a clear picture of the research arc, a set of strong opinions about what the evidence actually says, and a map of the bets that I think are worth taking.

Here is how I read the trajectory:

**Phase 1** (early days): The team discovered that frozen image-conditioned bridges could not beat a learned constant prefix. This was a foundational and embarrassing failure mode. The diagnosis was correct: the bottleneck was not visual signal deficiency, it was the interface geometry between the bridge and the frozen LM. Calibration layers, norm regularization, and cross-attention perceivers converted this into a working baseline around `0.43–0.45`.

**Phase 2** (arch probing, March 10-11): Eight architectures were swept at reduced budget. The clear winners were leakage-safe qcond perceiver, structured roles, and early-layer encoder features. None beat the previous frontier at full-budget comparison, but all three were ahead of the eventual-best run's 6k pace. The team correctly interpreted this as "alive directions, not dead ends."

**Phase 3** (high-entropy, March 12): Stacking the top Phase 2 ideas at full 9k budget pushed the frontier to `0.4568`. The best stack was `safeqcond + earlylayer + geomcal`. The team correctly concluded that safeqcond + earlylayer was the real engine, geomcal was a small modifier, and multiscale/hybrid were competitive but not decisive.

**Phase 4** (Hammer, March 13): This was the paradigm-shift sweep. LM visual adapters, question-derived queries (qquery), and adaptive token budgets (dynbudget) were introduced. The full three-way stack hit `0.4608`. **The most important finding in the entire project so far**: LM-side adapters—residual cross-attention into the LM's top layers—produced the biggest single-direction gain. The bridge extracted the tokens, but the LM needed to be able to keep asking for visual evidence during reasoning, not just consume a front-loaded prefix.

**Phase 5** (Nail, March 13): Refined within the adapter-centered mainline. The critical finding was that `lmmeanqquery`—deriving the bridge's query bank from mean-pooled LM hidden states over the question—outperformed plain `qquery` by `+0.0036`. Deeper adapters (d3 vs d2) were mildly positive at `+0.0009`. Cap increases and role specialization were both flat or negative. The new frontier is `lmmeanqquery_dynbudget_adapter_d3_cap64` at `0.4653`.

**Now (Plank plan, March 13, not yet executed)**: The previous researcher correctly identified the next question-quality variants—questiononly, multiq4, hybrid, iter2—and also reserved a MobileViT re-baseline. MobileViT was stabilized as a working drop-in VM on the same day. The current git HEAD is "stable mobilevit vm backbone." **Plank was planned but not executed. That is where we are.**

---

## What This Arc Is Actually Telling Us

Before talking about next sweeps, I want to be explicit about what the evidence says at the theoretical level, because that is what should drive sweep prioritization.

### The Center of Gravity Has Moved

The original framing was: "the bridge is the bottleneck." Every sweep through Phase 3 operated in that framing—better extraction, better calibration, better architecture, better feature sources. That framing was productive and produced real gains.

But Hammer broke it. The biggest single win was not a better bridge architecture. It was LM visual adapters: letting the LM revisit visual tokens during its own reasoning chain, not just at the input. This means the bottleneck had already moved. By Phase 3, the bridge was extracting enough. The new bottleneck was whether the LM could use that evidence deeply enough.

And then Nail reinforced this: the gain from `lmmeanqquery` was not about better bridge tokens. It was about a better question signal driving *which* bridge tokens to produce. The bridge was being asked to produce generic useful tokens; the new winner asked it to produce specifically *question-conditioned* tokens derived from actual LM hidden states. The LM was being invited into the extraction process itself.

**My conclusion**: the project has independently rediscovered late-fusion / cross-attention-based multimodal interaction. The prefix acts as a warm start; the in-LM adapters do the real work. This is architecturally similar to Flamingo's perceiver + cross-attention interleaved design, but arrived at from scratch by this team. That is a meaningful validation of the approach.

### The Answer-Type Decomposition Is Diagnostic

Across the full sweep history, the pattern is consistent:

- `yes/no`: improved mostly from better extraction and earlylayer features
- `number`: improved from qcond and multi-scale coverage
- `other`: improved from adapters, lmmeanqquery, and qcond

`other` is the hardest category and has historically been the bottleneck. The current best result on `other` is `0.3298`. The LM adapter family owns the gains there because "other" questions require open-ended compositional reasoning over visual evidence—exactly what in-layer visual access helps with.

**Implication for future sweeps**: If a new architecture improves overall but does it by lifting `yes/no` or `number`, the effect is likely a retrieval improvement. If it improves `other`, it is likely an LM-side fusion improvement. The two have different downstream paths.

---

## Constraint Analysis: What Is Worth Breaking

This project was designed with deliberate constraints. Here is my honest assessment of each.

### Constraint 1: Frozen Vision Model (VM)

**Original motivation**: Study the bridge and LM-side fusion in isolation. Do not entangle bridge learning with VM adaptation.

**Evidence for keeping it**: The project has made real gains under this constraint. The frozen VM has not been the bottleneck until recently. The team is now asking whether MobileViT improves over the original tiny VM, which is the right first question.

**Evidence for breaking it**: The current VM produces `49` tokens at a resolution optimized for classification. VQA often requires fine-grained spatial reasoning, attribute discrimination, and counting—tasks that classification-optimized VMs do not particularly emphasize. Early-layer features helped, suggesting the final-layer features are genuinely over-compressed for VQA.

**My honest assessment**: The frozen VM constraint is correct to maintain *right now*. The priority order should be:

1. First, establish what MobileViT (frozen) can do. If MobileViT delivers a clean `+0.005` or more on the best bridge family, the VM choice is confirmed as a lever.
2. If the MobileViT gain is large, a small visual-side residual adapter (trainable MLP on top of frozen features) is the right next test.
3. Last-block VM finetuning is the right test after that, but with a small LR and careful stability monitoring.
4. Full VM fine-tuning or VM pretraining is a phase-change investment that does not belong in near-term sweeps.

**Breaking value estimate**: A stronger frozen VM could plausibly add `+0.005` to `+0.015` to the frontier score. VM finetuning (last block only) could add `+0.010` to `+0.025`, but at higher engineering risk and a loss of the clean frozen-component research story.

### Constraint 2: Mostly Frozen LM (top 2 layers trainable + adapters)

**Original motivation**: Preserve LM language capability. Avoid catastrophic forgetting.

**Evidence for keeping it**: The LM adapters are already delivering substantial gains. The top-2-layers trainable setup plus 3 cross-attention adapters is a significant amount of trainable LM-side capacity (13.6M trainable parameters in the LM out of ~40M total LM parameters).

**Evidence for breaking it**: The project has not yet tested adapter depth d4 or d5. There may be more gain in going deeper. Also, the current approach trains top LM layers plus adapters, but all from random init (for the adapters). A LoRA-style reparameterization of the trainable LM layers might provide better optimization behavior.

**My honest assessment**: The LM constraint is still the right default. But:

- Adapter depth d4/d5 should be tested before any broader LM unfreezing
- If adapter gains flatten at d3-d4, that is evidence that the LM-side access depth is no longer the bottleneck
- Full LM fine-tuning risks the project's identity as a frozen-component study

**Breaking value estimate**: Going from d3 to d4 or d5 adapters might add `+0.002` to `+0.005`. Larger LM unfreezing is uncertain and risky.

### Constraint 3: No Bridge Pretraining

**Original motivation**: Simplicity and correctness. Avoid introducing a two-stage training setup before the single-stage setup is understood.

**Evidence for keeping it**: Every sweep has produced useful information under the constraint. Bridge pretraining would change the starting point for all weights and make later comparisons harder.

**Evidence for breaking it**: The bridge is being asked to do something hard from random initialization: align visual token representations with LM embedding geometry from scratch using only VQA question-answer supervision. A captionalign or latentalign pretraining stage—align bridge tokens to caption embeddings from a separate text encoder—would dramatically improve the bridge's starting point and might unlock gains that the current VQA-only supervision cannot produce.

**My honest assessment**: Bridge pretraining is the highest-upside deferred idea in the entire project. The right form of pretraining is:

- **captionalign**: train the bridge to produce tokens whose mean embedding matches the LM's encoding of the image caption. This is direct alignment supervision between visual tokens and language space.
- **latentalign**: align visual tokens to frozen LM embeddings of question-relevant phrases.

Both require a caption dataset (COCO captions is fine) and a two-stage training pipeline. The engineering cost is real but not prohibitive. My current estimate is that a well-executed bridge pretraining stage could add `+0.015` to `+0.040` to the final VQA score, simply by giving the bridge a better starting point.

**This is the phase-change investment the project has been deferring.** It should not replace the current sweep program—the current sweeps are still informative—but it should be scheduled as the next major engineering cycle after the Plank-level query-quality variants are exhausted.

**Breaking value estimate**: `+0.015` to `+0.040`. Highest upside of any deferred idea.

### Constraint 4: Single Frozen VM

**Current assumption**: One VM. Either the original tiny one or MobileViT-small.

**Evidence for breaking it**: Ensemble VMs or multi-scale VMs (early + late features combined) have shown up in the architecture history. The multiscale perceiver experiment showed that combining early and late VM features was positive (from `0.4398` to `0.4533` in the high-entropy sweep). However, this has not been revisited with the current adapter-centered mainline.

**My honest assessment**: Multi-VM ensembling is low on the priority list. The more interesting direction is testing `multiscale_lmmeanqquery_dynbudget_adapter_d3` to see if multiscale features still help in the current adapter-centered family. If lmmeanqquery now drives a more question-specific retrieval, multi-scale might be more valuable than it was in the earlier perceiver-only world.

---

## The Plank Plan Assessment

The Plank plan (doc #29) is correct. Its prioritization is sound and its hypotheses are well-formed. My additions and modifications follow.

### What Plank Gets Right

1. **Query quality first, cap/role/width changes second.** The evidence from Nail is clear.
2. **MobileViT as a separate branch, not an add-on.** The question "does better VM amplify qquery sharpening?" is exactly the right framing.
3. **iter2 as a high-upside bet after the cleaner wins.** Single-shot retrieval may still be a bottleneck.
4. **Deprioritizing role specialization and cap increases.** Both were negative in Nail. The current encoder path only provides 49 tokens anyway.

### What Plank Is Missing or Underspecifies

**1. Adapter depth sweep.** Nail tested d2 vs d3. d4 was never tested. If d3 > d2 by `+0.009`, d4 might yield another increment. This is a cheap test and should be in the main Plank queue.

**2. lmmeanqquery on the MobileViT + multiscale path.** The multiscale perceiver was last tested in the pre-adapter era. In the current adapter-centered family, multiscale features + lmmeanqquery + adapters might be a different beast. It is lower priority but should not be permanently off the table.

**3. A "no dynbudget" ablation of the current best.** We know dynbudget helped in Hammer when combined with qquery and adapters. But is it still pulling its weight in the Nail winner (lmmeanqquery_dynbudget_adapter_d3_cap64)? If dynbudget caps at 49 tokens and the upstream encoder only provides 49 tokens, dynbudget may be doing nothing at all in the Nail winner. This should be verified.

**4. Bridge pretraining as an explicit future phase.** The Plank plan mentions it as deprioritized. I want to elevate it explicitly as the next major investment cycle after Plank-tier runs are exhausted.

**5. Seed stability at the right moment.** The project currently has zero seed replication of any sweep winner. This is a real gap, but the right time to do seed work is after the architecture direction has stabilized. After Plank concludes, running 2-3 seeds of the strongest winner would produce a defensible headline number.

---

## Proposed Future Sweep Structure

This is my proposed generalized program for all sweeps following the current state (post-Nail, MobileViT-stable, Plank-not-yet-executed).

Each sweep is named, given a primary hypothesis, a predicted range of outcomes, and a "go/no-go" condition for the next stage.

---

### Plank Stage A: MobileViT Re-Baseline

**Codename suggestion**: Plank (already named), Stage A

**Primary hypothesis**: A stronger frozen VM (MobileViT-small at 640-dim features) will improve the best-known bridge family by at least `+0.005` over the original VM baseline, because the current VM produces classification-optimized features that are over-compressed for fine-grained VQA.

**Specific runs**:

1. `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64` — primary readout
2. `mobilevit_qquery_dynbudget_adapter_d3_cap64` — baseline comparison
3. `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` — qquery-type contrast

**How to read the results**:

- If `mobilevit_lmmeanqquery` beats `lmmeanqquery` by `+0.005` or more: the VM is still a bottleneck and Stage B should also focus on MobileViT + query sharpening variants.
- If `mobilevit_lmmeanqquery` is flat or below `lmmeanqquery`: the VM is no longer the bottleneck and the frozen-VM comparison line has saturated. Stage B should focus on bridge-side query quality and LM-side adapter depth.
- If `mobilevit_qquery` beats `mobilevit_lmmeanqquery`: something weird is happening with the LM-mean signal on MobileViT features; investigate before continuing the family.

**Why this runs first**: It is the cleanest isolated test of one hypothesis. It does not require any new bridge code. It re-uses a stabilized backbone. The answer is either "VM matters" or "VM doesn't matter anymore," and both are high-value.

---

### Plank Stage B: Query Quality Variants

**Primary hypothesis**: The `lmmeanqquery` path is not yet fully exploited. Question-only pooling, multiple LM-conditioned queries, and hybrid mean+attention generation will each add incremental gains because the current single-pooled LM-mean signal loses specificity from non-question context.

**Specific runs** (in priority order):

1. `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64` — pool only question-span LM tokens, not full prompt context
2. `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64` — 4 LM-conditioned query groups
3. `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64` — combined mean+attention query generation

**Conditionally add if Stage A was positive**:

4. `mobilevit_questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`
5. `mobilevit_multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`

**How to read the results**:

- If `questiononly` wins: the current mean-pooling is polluted by non-question context. The fix is simple and has immediate implications for future qquery design.
- If `multiq4` wins: the single-query bottleneck is real. Multiple queries allow the bridge to field parallel evidence requests (object, attribute, spatial, count), which aligns with the observed gaps in answer-type performance.
- If `hybrid` wins: the two query-generation paths are genuinely complementary. `lmmeanqquery` has a better overall signal; `attnqquery` has a better `other`-category signal. Combining them should lift both.
- If all three are flat: the qquery signal is already near-optimal for the current bridge/LM setup, and the next bottleneck is something else (adapter depth, visual features, bridge pretraining).

---

### Plank Stage C: Structural Ablations and Depth

**Primary hypothesis**: The current configuration (d3 adapters, 2 trainable LM layers, geomcal on) is near-optimal on some axes and has room on others. Specifically, adapter depth d4 might yield additional gain, and the dynbudget selector with a 49-token cap may be doing nothing useful.

**Specific runs**:

1. `lmmeanqquery_dynbudget_adapter_d4_cap64` — go deeper on adapters
2. `lmmeanqquery_nodynbudget_adapter_d3` — ablate dynbudget to check whether it still contributes
3. `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64` — two-stage retrieval

**Why `nodynbudget` matters**: The dynbudget selector scores and filters visual tokens before bridge extraction. But in the current encoder path, the upstream VM produces exactly 49 tokens and the cap is set at 64. This means the selector is choosing all 49 tokens every time—it has no filtering pressure. If so, dynbudget is adding complexity without function. This ablation will either confirm that dynbudget is dead weight in the current setup or reveal that the scoring itself provides a useful signal even without filtering.

**Why `iter2` is in Stage C, not B**: It has the highest algorithmic upside but also the highest implementation risk. It belongs after the cleaner Stage B wins have been read out, not competing with them for priority.

---

### Plank Stage D: Visual Adaptation

**Primary hypothesis**: The frozen VM features are now the limiting factor after query quality improvements have been exhausted. A small visual-side adapter (trainable residual MLP on VM features before the bridge) will allow the bridge to receive slightly more malleable visual tokens without destabilizing the frozen VM.

**Specific runs**:

1. `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64` — trainable MLP residual on VM features
2. (conditionally) `vm_lastblock_tune_lmmeanqquery_dynbudget_adapter_d3_cap64` — unfreeze last VM block with reduced LR

**Go/no-go condition**: Only run Stage D if Stage B improvements have flattened. If multiq4 or questiononly produce strong gains, Stage D is not yet needed.

---

### Phase 2 Investment: Bridge Pretraining

**Codename suggestion**: Stone, Scaffold, or Bedrock (something that implies a new foundation)

**This is the most important long-horizon research direction in the project.** It does not belong in the near-term sweep queue, but it should be explicitly scheduled, not perpetually deferred.

**What it is**: A two-stage training setup where Stage 1 trains the bridge (and only the bridge) to align its visual prefix tokens with an LM-accessible representation of the corresponding image caption. Stage 2 then continues on VQA data with the pretrained bridge.

**Why it matters**: Every current sweep trains the bridge from random initialization using only VQA question-answer pairs as supervision. This is a hard task: the bridge must simultaneously learn to extract useful visual information *and* produce tokens in the LM's embedding geometry *and* learn to respond to qquery signals—all at once. A well-aligned bridge that starts from a caption-aligned initialization would likely:

1. Converge faster
2. Produce higher-quality visual tokens for the LM from the beginning
3. Potentially unlock gains in `other`-category questions that require visual grounding the current bridge does not produce

**Two candidate forms**:

- **captionalign**: Bridge tokens → mean LM-encoded caption representation. The bridge learns to produce visual tokens whose mean lies close to the LM's encoding of what the image contains in natural language.
- **latentalign**: Bridge tokens → LM hidden states from caption encoding, conditioned on the bridge's qquery signal. More complex, but more directly aligned with the in-distribution VQA use case.

**My recommendation**: Start with captionalign. It is simpler, has a clear loss function (cosine similarity between mean visual prefix and LM-encoded caption), and uses standard supervision from COCO captions or CC3M.

**Predicted gain**: `+0.015` to `+0.040` after the pretraining stage is added. The high end of this estimate depends on how much of the current VQA training budget is being spent on bridge alignment versus question-answering.

---

## Honest Assessment of the Current Ceiling

I want to be direct about what I think the ceiling looks like for this architecture class.

The current setup is:
- Frozen tiny VM (~2M params) or MobileViT-small
- Trained bridge (~33M params)
- Mostly frozen LM with adapters (~40M total, ~20M trainable)
- ~81M total parameters

VQAv2 performance for comparable frozen-component setups in the literature (CLIP ViT-B/32 frozen + GPT-2-medium frozen + small bridge, typical numbers in 2022-2024 papers): roughly 50–58%. BLIP-2 with a Q-Former and a frozen 2.7B LM: ~65% zero-shot. Full VQA fine-tuning SOTA models: ~80%+.

My estimate for the current system's ceiling without breaking any major constraint: **approximately 0.49 to 0.53**. This assumes:
- Bridge pretraining is added (Phase 2)
- MobileViT or a stronger VM is used
- Query sharpening improvements from Plank are applied
- Adapter depth is pushed to d4-d5

Above that, the frozen LM becomes the dominant bottleneck. A 125M frozen GPT-2 (the current LM is implied to be smaller based on the parameter counts) is not going to reason deeply about fine-grained visual evidence no matter how good the bridge is.

**To break past ~0.53**, the likely requirement is one of:
1. A larger frozen LM (GPT-2-medium or similar)
2. Partial LM fine-tuning beyond the top layers
3. A genuinely different architecture class (more interleaved cross-attention between LM layers, similar to Flamingo)

All of these are within the project's long-term scope as described in the task context. But they are Phase 3 or Phase 4 investments, not near-term.

---

## Specific Recommendations for the Next Sweep Execution

Given that MobileViT is stable and Plank has not yet been executed:

**Start immediately with Stage A (MobileViT re-baseline)**. This is the highest-information run with the least additional implementation work. The runs:

1. `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `mobilevit_qquery_dynbudget_adapter_d3_cap64`
3. `mobilevit_attnqquery_dynbudget_adapter_d3_cap64`

Run these in parallel if compute allows; the MobileViT probe established that the architecture fits at `192 x 1` at about `4.19` train steps/s, which is acceptable.

**While Stage A is running, implement Stage B variants** (questiononly, multiq4, hybrid). These are bridge-side code changes that should not require large refactors. The questiononly variant is the cheapest implementation: mask the pooling operation in `lmmeanqquery` to only include question-span token positions before mean pooling.

**Queue Stage B immediately after Stage A.** The answer to the MobileViT question will inform whether Stage B should also be run in the MobileViT configuration, but the original-VM Stage B runs can proceed in parallel.

**One thing I would do that the prior plans have not done**: instrument answer-type breakdown more systematically across every sweep. The yes/no / number / other split is already collected, but a breakdown by question word (What, How many, Is there, What color, etc.) would add significant diagnostic value. The current data is almost certainly already in the eval output files. A simple post-processing script over `fixed_eval_val_answers.jsonl` files could compute this without any re-running.

---

## What I Would Tell This Team If I Were Their Manager

You have done genuinely good work here. You have independently converged on the correct set of architectural insights that the field has been discovering for the last three years—late fusion via in-layer cross-attention is better than front-loaded prefix injection alone; LM-conditioned visual querying is better than static generic extraction; early-layer visual features carry information the final latent discards—and you have done it systematically, with clean comparison policies and reproducible sweep infrastructure.

The infrastructure is solid. The comparison policy (effective batch 192, full-val final eval, 9k steps) is the right standard. The log stitcher, the tracker, the Docker-based execution discipline—these are not glamorous, but they are what makes the difference between research that produces reliable conclusions and research that produces noise.

The main thing I would push the team on is this: **stop treating bridge pretraining as a perpetual future investment.** Every time a sweep concludes, there is a reason to defer it. That is correct reasoning in the short term—there were always cheaper wins still available. But the project has now moved through four major sweep cycles and is at the point where the next cheap wins are in the `+0.001` to `+0.005` range. The bridge pretraining idea is potentially a `+0.015` to `+0.040` move. At some point, the expected value calculation tips.

My recommendation: schedule bridge pretraining as the explicit goal after Plank Stage B concludes. Do not run it as a side experiment or a contingency. Design it as its own named sweep with a proper plan document, a clear loss function, and a defined comparison baseline. That will produce the most useful research artifact.

---

## Run Standard Reminder

All future comparable sweeps should use:

- effective batch `192`
- target steps `9000`
- `eval_every=1000`, `eval_batches=100`
- final eval on full validation split
- `--eval_use_kv_cache --eval_kv_cache_mode batched`
- official scorer

Non-standard runs (memory probes, smoke tests, quick ablations) should be labeled explicitly as `diagnostic only` or `non-comparable`.

---

## Summary Table

| Stage | Primary Hypothesis | Expected Delta | Risk | Priority |
|---|---|---|---|---|
| Plank-A: MobileViT | Stronger frozen VM amplifies lmmeanqquery | `+0.003` to `+0.015` | Low | Immediate |
| Plank-B: Query quality | Sharper/richer LM-conditioned query formation | `+0.002` to `+0.010` | Low-Med | After A |
| Plank-C: Depth/ablation | Adapter d4 positive; dynbudget may be dead at cap 49 | `+0.001` to `+0.005` | Low | After B |
| Plank-D: Visual adapter | VM features still bottleneck after B/C | `+0.003` to `+0.012` | Med | After B-flat |
| Phase 2: Bridge pretraining | Aligned bridge starting point unlocks new slope | `+0.015` to `+0.040` | High | After Plank |
| Phase 3: Larger LM | Current LM is the ceiling | `+0.030` to `+0.100` | Very High | Later |

Risks are relative to the project's current clean frozen-component research line. "Med" means engineering complexity and potential instability. "High" means requires a fundamentally different training pipeline.

---

*Written 2026-03-14. This is a fresh-eyes perspective on the full body of work from docs/01 through docs/30.*
