# Plank Sweep Plan - 2026-03-13

## Codename

`plank`

## Purpose

Plank is the first sweep that treats Nail as a real architectural narrowing, not just another leaderboard shuffle.

Its job is to answer one main question:

- how do we improve LM-conditioned visual querying inside the current adapter-centered mainline

Plank is not about:

- making the bridge generically larger
- repeating dead axes from Nail
- spending the first cycle on high-risk training-system changes

The right Plank cycle is a query-quality sweep, with one small visual-adaptation branch held in reserve.

## Entry State

Authoritative frontier entering Plank:

- best observed run: `lmmeanqquery_dynbudget_adapter_d3_cap64` at `0.4653`
- tied best observed run: `lmmeanqquery_dynbudget_adapter_d3_cap96` at `0.4653`
- best non-`lmmeanqquery` Nail run: `attnqquery_dynbudget_adapter_d3_cap64` at `0.4624`
- best pre-Nail reference: `qquery_dynbudget_adapter_earlylayer_geomcal` at `0.4608`

Important framing correction:

- this is not just a "tiny VM + tiny LM" setup
- the current winner is an `81.4M` total-parameter system
- approximate split of the winner:
  - VM frozen: `2.0M`
  - bridge trained: `33.3M`
  - LM frozen: `26.3M`
  - LM trainable: `13.6M`
  - LM added: `6.3M`

So the core question is no longer "is the model too small?"

It is:

- which bridge and LM-side querying computations are actually useful

## What Nail Established

### 1. Best gain came from better query formation

- `qquery_dynbudget_adapter_d3_cap64`: `0.4617`
- `lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4653`
- delta: `+0.0036`

This is the cleanest architectural win in Nail.

### 2. Bigger cap alone did not help

- `qquery_dynbudget_adapter_d2_cap96`: `0.4608`
- `lmmeanqquery_dynbudget_adapter_d3_cap96`: `0.4653`

And more importantly:

- the Nail `cap64` and `cap96` `lmmeanqquery` runs were effectively duplicates
- the upstream encoder path only provided `49` visual tokens to the selector
- so the larger cap was not actually exercised

This means Nail did not test "does more than 49 help?"

It only showed:

- cap changes above the current upstream token count are not a meaningful axis

### 3. Adapter depth was mildly positive

- Hammer-best reference: `0.4608`
- `qquery_dynbudget_adapter_d3_cap64`: `0.4617`
- delta: `+0.0009`

Depth helps, but it is not the main story.

### 4. Role specialization is not a live direction right now

- `rolespecial_dynbudget_adapter_d3_cap64` underperformed base `qquery d3`
- `rolespecial_lmmeanqquery_dynbudget_adapter_d3_cap64` underperformed plain `lmmeanqquery`

So explicit roles are currently adding structure without adding accuracy.

### 5. The frontier is image-grounded

Hammer-best corruption suite:

- clean: `0.4608`
- shuffle: `0.4514`
- random image swap: `0.4019`
- zero image: `0.3813`

The model is doing real image-conditioned work.

## Plank Thesis

Plank should treat the bridge as LM-conditioned visual retrieval, not as a generic static visual adapter stack.

The frontier question is now:

- can the LM ask better questions of the visual tokens

The most promising next directions are therefore:

1. sharper LM-conditioned pooling for qquery generation
2. multiple LM-conditioned queries instead of one compressed request
3. hybrid query generation that combines the best parts of `lmmeanqquery` and `attnqquery`
4. iterative querying if one-shot retrieval is still too bottlenecked

Only after those are tested should Plank spend real budget on:

- small visual adaptation

## Main Research Questions

### 1. Question-Only Pooling

Is the current LM-mean qquery too diffuse because it pools over more than the question span?

Test:

- pool only question tokens for qquery generation

Desired outcome:

- cleaner query signal than global LM-state mean

### 2. Multi-Query Generation

Is one LM-conditioned query request too bottlenecked for VQA?

Test:

- generate multiple LM-conditioned qquery groups from the same question state

Desired outcome:

- allow separate retrieval pressure for object, attribute, count, relation, and global evidence

### 3. Hybrid Query Generation

Are `lmmeanqquery` and `attnqquery` complementary rather than competitive?

Test:

- combine the two query-generation paths with concatenation or gating

Desired outcome:

- preserve `lmmeanqquery`'s overall win while borrowing `attnqquery`'s strength on `other`

### 4. Iterative Querying

Is one-shot visual retrieval the remaining bottleneck?

Test:

- use a first query pass to gather coarse evidence
- form a second query pass from LM state plus retrieved visual summary

Desired outcome:

- stronger compositional and relation-heavy retrieval

### 5. Small Visual Adaptation

If query improvements flatten, are frozen visual features the next bottleneck?

Test:

- either unfreeze only the last VM block
- or add a tiny visual-side adapter before the bridge

Desired outcome:

- improve recoverability of attributes, counts, and fine spatial cues without destabilizing the full system

## Shared Plank Baseline

Unless explicitly overridden, Plank runs should inherit:

- base family: `lmmeanqquery_dynbudget_adapter_d3_cap64`
- `vision_feature_source=encoder`
- `bridge_type=perceiver_resampler`
- `bridge_query_bank_mode=question_hidden_mean`
- `bridge_token_selector_type=qadaptive`
- `lm_visual_adapter_type=cross_attn`
- `lm_visual_adapter_layers=3`
- `bridge_question_context_mode=prompt_only`
- effective batch `192`
- target step `9000`
- official full-val final eval
- `--eval_use_kv_cache --eval_kv_cache_mode batched`

Reason:

- this is the actual winning branch from Nail
- it keeps Plank focused on the strongest demonstrated lever

## Axes To Deprioritize

Do not spend the first Plank cycle on:

- larger dynbudget caps above the current upstream token count
- role specialization retests
- generic bridge widening
- another bridge-only family search
- bridge pretraining
- broad seed sweeps as core queue items

Why:

- Nail already showed role specialization is not frontier-positive
- cap sweeps above `49` were not real tests under the current encoder-token path
- broad seed work is useful later, but iteration still matters more right now

## Proposed Run Set

Recommended run prefix:

- `mmplank_v1_20260313`

### Tier 1: Highest-Priority In-Family Frontier Runs

#### 1. `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`

What it changes:

- keep `lmmeanqquery`
- pool only question-span LM tokens for qquery generation

Why it exists:

- cheapest and cleanest follow-up to the strongest Nail result
- directly tests whether the current LM-mean signal is polluted by non-question context

Expected outcome:

- best low-risk chance of a clean improvement

#### 2. `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`

What it changes:

- form `4` learned LM-conditioned query groups instead of one pooled request

Why it exists:

- best direct test of "better querying beats bigger bridge"

Expected outcome:

- highest-upside bridge-only continuation of the Nail lesson

#### 3. `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`

What it changes:

- combine LM-mean and attention-derived query generation with a learned gate or merge

Why it exists:

- `lmmeanqquery` was best overall
- `attnqquery` was strongest on `other`

Expected outcome:

- potential complementary gain without leaving the proven family

#### 4. `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`

What it changes:

- two-stage query/retrieve/refine/requery bridge path

Why it exists:

- tests whether one-shot retrieval is the remaining bottleneck

Expected outcome:

- biggest algorithmic upside among the pure bridge changes, but higher implementation risk

### Tier 2: Small Visual Adaptation Branch

Only queue these if Tier 1 is flat or nearly flat.

#### 5. `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`

What it changes:

- keep the VM frozen
- add a tiny visual-side trainable adapter before the bridge

Why it exists:

- safer than unfreezing the VM
- direct test of whether the bridge needs slightly more adaptable visual features

#### 6. `vm_lastblock_tune_lmmeanqquery_dynbudget_adapter_d3_cap64`

What it changes:

- unfreeze only the last VM block with a smaller LR

Why it exists:

- most plausible "beyond bridge-only" move if qquery quality improvements stall

### Optional Stability Work

These are useful, but should not sit in the main iteration path of early Plank.

#### 7. `seed2_lmmeanqquery_dynbudget_adapter_d3_cap64`

#### 8. `seed3_lmmeanqquery_dynbudget_adapter_d3_cap64`

Why they are optional:

- there is still no project-wide seed baseline to compare against
- current priority is still frontier movement, not variance characterization

## Revised Execution Priority

1. `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`
3. `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`
4. `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`
5. `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`
6. `vm_lastblock_tune_lmmeanqquery_dynbudget_adapter_d3_cap64`
7. optional seed replications

## Recommended Single Run

If only one Plank run should go first:

- `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`

Why:

- it sharpens the exact winning mechanism from Nail
- it is low-risk
- it is cheap to implement
- if it wins, it strongly confirms that qquery quality is the frontier bottleneck

## Biggest Upside Shot

If only one higher-risk Plank run should be taken:

- `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`

Why:

- this is the most direct test of whether the model wants multiple LM-conditioned visual requests instead of a single compressed query
- it aligns with the strongest current interpretation of the frontier

## Projected Ordering

Plausible projected finish order, assuming clean implementations:

1. `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`
3. `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`
4. `lmmeanqquery_dynbudget_adapter_d3_cap64` carry-forward baseline
5. `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`
6. `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`
7. `vm_lastblock_tune_lmmeanqquery_dynbudget_adapter_d3_cap64`

This ordering is only a rough planning prior.

The main point is:

- query-quality variants should outrank cap, role, and generic width changes

## One-Line Summary

Plank should treat Nail as proof that LM-conditioned query quality is the live frontier: question-only LM pooling, multi-query qquery, hybrid LM-mean plus attention qquery, and iterative querying are the right first-line experiments, while cap increases and role specialization should be considered dead axes for now and small visual adaptation should wait behind the top query-quality probes.

## MobileViT Append

Now that `mobilevit_hf` is a working drop-in frozen VM, Plank should also reserve a tight "same bridge, better vision" stage before spending many slots on broader bridge novelty.

### Updated Framing

Nail already established the important bridge-side priors:

- `lmmeanqquery` was the strongest clean gain over plain `qquery`
- deeper LM adapters were mildly positive
- `cap64 -> cap96` was not a real lever in the current encoder regime
- role specialization was negative twice
- the model is genuinely image-dependent

That means the first MobileViT question is not "invent a new bridge family." It is:

- does a better VM strengthen the same winning qquery path enough to create a new slope?

### MobileViT Stage 1: Same Bridge, Better Vision

Keep this stage deliberately narrow:

- same bridge family
- same `dynbudget`
- same adapter depth `d3`
- fixed cap at `64`
- no role specialization

Primary runs:

1. `mobilevit_qquery_dynbudget_adapter_d3_cap64`
2. `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`
3. `mobilevit_attnqquery_dynbudget_adapter_d3_cap64`
4. `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64_seed2`

Why this set:

- it re-baselines the strongest Nail bridge families on the new VM
- it tests whether the VM improvement lifts all boats or mostly amplifies the best LM-conditioned qquery path
- it avoids wasting slots on dead axes already identified by Nail

### MobileViT Stage 1 Priorities

Operating rules:

- keep token cap fixed at `64`
- keep role specialization out
- focus on query quality, not bridge width
- treat `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64` as the primary readout run

Most wanted run:

- `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`

Why:

- it is the cleanest continuation of the strongest Nail result
- it isolates the exact hypothesis we now care about: better VM features plus best-known LM-conditioned query formation

### MobileViT Follow-On If Stage 1 Wins

If `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64` is a clear winner, the next MobileViT-focused queue should be:

1. `mobilevit_questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `mobilevit_multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`
3. `mobilevit_hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`

Interpretation:

- stage 1 asks whether better vision helps the known best bridge family
- stage 2 asks whether better vision makes qquery sharpening even more valuable

### Revised Draft Queue

Bridge-first Plank queue remains valid, but the practical draft queue should now treat the MobileViT re-baseline as an explicit branch:

1. `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `mobilevit_qquery_dynbudget_adapter_d3_cap64`
3. `mobilevit_attnqquery_dynbudget_adapter_d3_cap64`
4. `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64_seed2`
5. `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`
6. `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`
7. `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`
8. `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`
9. `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`
10. `vm_lastblock_tune_lmmeanqquery_dynbudget_adapter_d3_cap64`

### Current Expectation

Best current guess:

- MobileViT should help most on `other` and attribute-heavy questions
- `lmmeanqquery` should remain the best bridge family
- the main question is whether the stronger VM increases the marginal value of better LM-conditioned querying enough to produce a clearer frontier gap
