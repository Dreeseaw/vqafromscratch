# Crane Sweep Plan - 2026-03-14

## Codename

`crane`

## Purpose

Crane should be the first post-Plank sweep that is explicit about there being two different agendas:

1. cheap, low-eng, in-family runs that can still move the current frontier
2. a separate set of higher-order approaches for escaping the current regime entirely

Those should not be mixed together.

Plank made the split clear:

- if the goal is to improve the current family, the right space is now `MobileViT + qquery + dynbudget + LM visual adapters`
- if the goal is to think honestly about reaching BLIP-2 territory (`65.2` on VQAv2), then bridge-only cleverness is not enough by itself

Crane therefore needs one practical queue and one strategic queue.

## Entry State

Authoritative frontier entering Crane:

- best observed run: `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` at `0.5240`
- second best: `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64` at `0.5181`
- third best: `mobilevit_qquery_dynbudget_adapter_d3_cap64` at `0.5167`
- best original-VM Plank run: `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64` at `0.4699`
- best pre-Plank run: Nail winner `lmmeanqquery_dynbudget_adapter_d3_cap64` at `0.4653`

The gap to the user-provided BLIP-2 reference is:

- `0.652 - 0.524 = 0.128`

That is too large to pretend a few more cap or role tweaks will close it.

## What 32a and 32b Together Established

The two Plank reports converge on the same main picture.

### 1. MobileViT changed the project more than any bridge tweak so far

All three MobileViT qquery families landed above `0.516`, with the best at `0.5240`.

This means:

- the old VM was a much larger bottleneck than the project had previously priced in
- current bridge quality was being judged through an artificially weak visual front-end

### 2. Query quality still matters, but its best form changed under the stronger VM

Old VM story:

- `lmmeanqquery` beat `attnqquery`

MobileViT story:

- `attnqquery` beat `lmmeanqquery`

The swing is concentrated in `other`, not in yes/no:

- `mobilevit_attnqquery`: `other=0.4401`
- `mobilevit_lmmeanqquery`: `other=0.4283`

So the best current interpretation is:

- richer visual features increase the value of more selective question-derived querying

### 3. The original-VM bridge family is near saturation

On the old VM, the Plank query-quality runs clustered in:

- `0.4637` to `0.4699`

That is useful because it says Crane should not spend more core queue slots on old-VM bridge refinements unless they are directly diagnostic for the MobileViT path.

### 4. Some old-VM results still matter as clues

Two old-VM findings remain live:

- `questiononly` was the cleanest positive bridge-side refinement
- `visual_adapter` was mildly positive

Three old-VM findings currently look weak:

- `multiq4`
- `hybrid`
- `iter2`

But they are weak specifically under the old VM. Plank does not prove they are dead under MobileViT.

### 5. The current ceiling is not obviously an extraction ceiling anymore

The current stack already has:

- a stronger frozen VM
- adaptive token selection
- LM-conditioned qquery
- LM residual visual adapters

So once MobileViT lifted the family by about `+0.05`, the remaining question changed from:

- "can the bridge see the right evidence at all?"

to:

- "does the LM have enough depth and enough multimodal contact to use that evidence fully?"

That is the key Crane framing.

## Crane Thesis

Crane should treat the current frontier as:

- `better visual features + better selective querying + somewhat deeper LM-side visual reasoning`

not:

- `more generic bridge machinery`

The practical Crane question is:

- can we sharpen the winning MobileViT path with very small, very comparable changes

The strategic Crane question is:

- what must be broken in the current constraints to close a remaining `12.8` points

## Crane Tier 1: Low-Eng, Mid-Entropy Frontier Runs

These should all be directly comparable to the current best line and should require little or no new engineering.

Shared baseline:

- `vision_model=mobilevit_hf`
- `bridge_token_selector_type=qadaptive`
- `bridge_token_select_k=64`
- `lm_visual_adapter_type=cross_attn`
- `lm_visual_adapter_layers=3` unless overridden
- `bridge_type=perceiver_resampler`
- `dynbudget + adapter + MobileViT` family
- effective batch `192`
- standard `9000` step run and full-val eval

### Run 1. `mobilevit_questiononly_attnqquery_dynbudget_adapter_d3_cap64`

What it changes:

- keep the winning `attnqquery` path
- restrict LM query formation to question-span tokens only

Why this is first:

- `attnqquery` is the current best run
- `questiononly` was the cleanest bridge-side win on the old VM
- combining the two is the lowest-risk way to sharpen the actual winning mechanism rather than changing families again

Modeling rationale:

- the MobileViT result implies the model benefits when the query path can exploit richer local detail
- if `prompt_only` still carries instruction/prompt clutter, then attention-derived querying may be spending capacity on non-question tokens
- the stronger the VM gets, the more expensive a diffuse query becomes, because the model now has better evidence to choose among

What this run would tell us:

- if it beats `0.5240`, the new frontier is not merely "attention qquery," but "clean question-only attention qquery"
- if it ties or loses, then the current attn path is already focused enough and query cleanup is not the next lever

Comparator:

- `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` at `0.5240`

Projected outcome:

- plausible best low-risk candidate in the `0.525` to `0.531` band

### Run 2. `mobilevit_hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`

What it changes:

- combine the two best MobileViT qquery paths rather than choosing one

Why it is live now even though old-VM hybrid was flat:

- old-VM hybrid had little reason to work because `lmmean` and `attn` were only weakly separated
- under MobileViT, they are no longer near-degenerate
- `attnqquery` wins strongly on `other`, while `lmmeanqquery` still looks competitive overall and may be the more stable query source

Modeling rationale:

- MobileViT appears to expose enough fine-grained information for attention-derived querying to matter
- but mean pooling may still provide a more global scene prior
- if those paths are complementary, the combination should help most on open-ended questions without giving back yes/no

What this run would tell us:

- whether the current best family wants one sharp query path or a mixture of coarse global and selective local query signals

Comparator:

- `mobilevit_attnqquery...` at `0.5240`
- `mobilevit_lmmeanqquery...` at `0.5181`

Projected outcome:

- medium-upside run; likely either flat or a real step
- plausible band `0.521` to `0.529`

### Run 3. `mobilevit_attnqquery_dynbudget_adapter_d4_cap64`

What it changes:

- keep the current best architecture
- deepen LM visual adapters from `3` to `4` layers

Why this belongs in Crane:

- Hammer and Nail already suggested adapter depth is mildly positive
- Plank suggests the current family may now be reasoning-limited more than extraction-limited
- `mobilevit_attnqquery` still looked healthy through 9k rather than obviously saturated

Modeling rationale:

- once the bridge is retrieving better evidence, the next failure mode is often not "missing evidence" but "the LM cannot keep consulting it deeply enough"
- a fourth adapter layer is a controlled test of deeper multimodal reasoning without changing the bridge family or taking on VM finetuning risk

What this run would tell us:

- whether the present frontier is still improved mostly by better retrieval, or whether reasoning depth has become the next cheap lever

Comparator:

- `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` at `0.5240`

Projected outcome:

- lower upside than Run 1, but still a strong probe
- plausible band `0.523` to `0.528`

## Tier 1 Control / Optional Support

Not part of the core three ideas, but useful if one extra slot is available:

### `mobilevit_attnqquery_dynbudget_adapter_d3_cap64_seed2`

Why:

- the current best run is single-seed
- `lmmeanqquery` already showed about `0.005` seed spread on MobileViT
- a second seed for `attnqquery` makes the Crane frontier less fragile before further specializing around it

This is not the most informative architecture run, so it should remain support work, not the center of Crane.

## Tier 1 Runs To Avoid

Do not center Crane on:

- old-VM repeats
- larger caps above the current effective token source
- role specialization
- generic bridge widening
- another plain `qquery` MobileViT rerun unless needed as a control

Reason:

- Plank already priced those axes
- none look like the highest-value next comparison

## Crane Tier 2: Higher-Order, Constraint-Breaking Approaches

These are not "just another sweep slot." They are separate bets that deliberately break the current project constraints.

They matter because the current gap to the BLIP-2 reference is about `12.8` points, and no evidence from Hammer, Nail, or Plank suggests that in-family bridge tweaks alone can supply that.

### Approach A. BLIP-2-Lite Bridge Pretraining: `mobilevit_qformer_pretrain_then_vqa`

Core move:

- replace the current purely VQA-trained bridge bottleneck with a Q-Former-like query bottleneck
- pretrain that bottleneck before VQA on image-text objectives

Minimal conceptual pipeline:

```text
image -> MobileViT tokens -> learned query transformer / Q-Former -> compact visual queries
                                                   |
                                       pretrain on caption / retrieval / image-text alignment
                                                   |
                                              VQA finetune
```

Why this is the most serious bridge-side moonshot:

- BLIP-2-level systems do not rely on the VQA loss alone to teach the bridge what visual concepts matter
- our current bridge learns extraction from a narrow answer-supervision signal over only `9000` VQA steps
- that is enough to reach the low `0.52`s with better vision, but it is not a plausible recipe for `0.65`

Modeling rationale:

- Plank showed that the project responds strongly when better visual information reaches the bridge
- that suggests the bridge still benefits from richer, more semantically organized visual concepts
- pretraining the query bottleneck is the cleanest way to teach object, attribute, relation, and textural retrieval before the VQA task starts shaping it

Why this is constraint-breaking:

- needs a separate pretraining stage
- needs image-text data beyond the current VQA-only loop
- probably needs a more BLIP-like bridge rather than only the current Perceiver variants

Why it is more plausible than generic widening:

- it targets the exact mechanism the current stack is missing: broad multimodal concept formation before narrow task adaptation

What success would look like:

- move the project from "good VQA specialization on narrow data" toward "general visual retrieval bottleneck with downstream VQA adaptation"

### Approach B. Persistent Visual Memory Into a Stronger LM: `mobilevit_persistent_xattn_stronger_lm`

Core move:

- stop treating visual information mainly as a short prefix plus a few top-layer adapters
- expose a persistent visual memory to a stronger pretrained LM across more layers

Conceptually:

```text
image -> MobileViT tokens -> compact visual memory
text  -> stronger pretrained LM
LM layers -> repeated cross-attn into visual memory across many layers
```

This can be implemented either as:

- a much deeper adapter stack in a stronger pretrained LM
- or a true multimodal decoder path rather than the current mostly frozen top-2-LM setup

Why this is likely necessary for the full BLIP-2 gap:

- the current LM is still relatively small and only partially trainable
- current multimodal interaction depth is limited
- once retrieval becomes good enough, the remaining deficit is often reasoning capacity and the ability to revisit vision repeatedly while composing the answer

Modeling rationale:

- Plank suggests the system is no longer dominated by total visual ignorance
- the frontier is shifting from "get any useful evidence in" to "reason over useful evidence deeply enough"
- BLIP-2-class systems benefit from both stronger language priors and more persistent visual-language interaction

Why this is constraint-breaking:

- likely requires a stronger pretrained LM
- likely requires broader multimodal data
- likely changes the basic parameter budget and training recipe

Why it is honest:

- if the project truly wants a shot at `0.65`, stronger multimodal interaction depth and a stronger LM prior are more plausible than more qquery variants alone

## Recommended Crane Queue

### Practical queue

1. `mobilevit_questiononly_attnqquery_dynbudget_adapter_d3_cap64`
2. `mobilevit_hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`
3. `mobilevit_attnqquery_dynbudget_adapter_d4_cap64`
4. optional: `mobilevit_attnqquery_dynbudget_adapter_d3_cap64_seed2`

This queue is deliberately narrow. It stays on the best current VM, the best current bridge family, and the most interpretable next changes.

### Strategic queue

1. `mobilevit_qformer_pretrain_then_vqa`
2. `mobilevit_persistent_xattn_stronger_lm`

These are not for the next cheap launcher. They are the next serious research branches once the practical Crane queue is read out.

## Recommended Single Run

If only one Crane run should go first:

- `mobilevit_questiononly_attnqquery_dynbudget_adapter_d3_cap64`

Why:

- it is the cleanest refinement of the actual winning Plank path
- it directly combines the strongest MobileViT result with the strongest old-VM refinement clue
- it is cheap, comparable, and high-signal

## Highest-Upside In-Family Shot

If only one higher-variance but still low-eng run should go first:

- `mobilevit_hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`

Why:

- this is the first time the hybrid idea has a real modeling case
- under MobileViT, `lmmean` and `attn` are finally separated enough that complementarity is plausible instead of wishful

## Honest Read On The BLIP-2 Gap

Crane should be explicit about this:

- moving from `0.5240` to about `0.652` is not a normal sweep continuation

The data from Hammer, Nail, and Plank suggest:

1. in-family runs can still move the current frontier
2. MobileViT was the last "cheap giant win"
3. another MobileViT-scale jump is unlikely from cap, role, or mild bridge surgery alone

So the honest path is:

- use Crane Tier 1 to finish squeezing the current family properly
- use Crane Tier 2 to define the first truly different architecture line

## One-Line Summary

Crane should stay narrow and disciplined in its practical queue by refining the winning `MobileViT + attnqquery + dynbudget + adapters` path with question-only querying, hybrid query formation, and slightly deeper LM visual adapters, while separately acknowledging that a serious shot at BLIP-2-level performance likely requires a pretrained Q-Former-style bridge and/or a stronger LM with persistent multimodal cross-attention rather than more local bridge tweaks.
