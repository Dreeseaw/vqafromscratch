# Semantic Compression Diagnostics Report

Source bundle:
- `/home/wdree/percy/vqafromscratch/logs/mmsemantic_diag_v1_20260322_234722`

Checkpoints analyzed:
- Cement full-eval anchor: `logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar`
- Best completed compressed run: `logs/mmsemantic_v1_20260322_k32/step_4000.tar`
- Lower-budget frontier: `logs/mmsemantic_v1_20260322_k8/step_4000.tar`

## Summary

The semantic-compression sweep held up well under downstream diagnostics, but the results are more specific than the strongest version of the original thesis.

What the diagnostics support:
- late LM-facing compression is genuinely low-loss at `K=32` and `K=8`
- the compressed exported tokens are at least as semantically decodable as the uncompressed Cement anchor
- stronger compression can make the exported tokens *more* linearly decodable

What the diagnostics do **not** support:
- compression did **not** make the system less dependent on LM visual adapters
- at `K=8`, the system became substantially *more* sensitive to adapter removal

So the clean read is:

`compression improves token semantic density, but aggressive compression still pushes more reasoning burden onto the LM adapter stack`

## 1. Reference Full-System Scores

These are the completed full-eval reference points for context:

| Model | Full eval |
|---|---:|
| Cement anchor (`49` exported tokens) | `0.6163` |
| Semantic bottleneck `K=32` | `0.6158` |
| Semantic bottleneck `K=8` | `0.6154` |

The full-system curve remained nearly flat through `K=8`, so the exported LM-facing budget is not the immediate performance bottleneck once the perceiver has already retrieved the dense evidence latents.

## 2. Semantic Probe

Probe setup:
- frozen checkpoint features
- flattened exported tokens
- linear answer head only
- `9999` train samples / `4319` val samples

Results:

| Model | Best probe acc | Yes/No | Number | Other | Best epoch |
|---|---:|---:|---:|---:|---:|
| Anchor | `0.4742` | `0.5876` | `0.3150` | `0.4083` | `9` |
| `K=32` | `0.4749` | `0.5956` | `0.3339` | `0.3959` | `9` |
| `K=8` | `0.5031` | `0.6199` | `0.3511` | `0.4316` | `6` |

Interpretation:
- `K=32` is effectively tied with the anchor on probe accuracy.
- `K=8` is clearly better than both.
- That means the exported tokens in the stronger-compression regime are not just smaller; they are more directly decodable by a tiny classifier.

This is the strongest positive result in the diagnostics. The bottleneck is not merely preserving full-system score; it appears to be concentrating answer-relevant information into fewer LM-facing tokens.

## 3. Adapter Ablation

Ablation setup:
- same full checkpoint
- keep only the top `N` LM visual adapters active
- evaluate at `keep_count in {3, 2, 1, 0}`

Results:

| Model | Keep 3 | Keep 2 | Keep 1 | Keep 0 | Drop 3->0 |
|---|---:|---:|---:|---:|---:|
| Anchor | `0.6163` | `0.6094` | `0.5841` | `0.4546` | `-0.1617` |
| `K=32` | `0.6158` | `0.6089` | `0.5814` | `0.4522` | `-0.1636` |
| `K=8` | `0.6154` | `0.6077` | `0.5719` | `0.3744` | `-0.2410` |

Interpretation:
- `K=32` is basically the same as the anchor. Compression to `32` did **not** reduce LM-adapter dependence in any meaningful way.
- `K=8` is much more fragile. The full model score stays high with all adapters present, but once adapter support is removed, it falls much harder than the anchor.

This is the main negative result. It says the strongest version of the semantic-bottleneck claim is false for `K=8`:
- the tokens are more semantically packed
- but the model still relies more heavily on LM-side multimodal reasoning to cash them out

So the current compression module is producing denser LM-facing tokens, not a self-sufficient semantic state.

## 4. Grounding / Attention Structure

Grounding outputs were saved for `50` correct and `50` incorrect examples per checkpoint.

Structural findings:
- anchor attention maps operate over the raw SigLIP patch grid: `196` visual tokens
- compressed runs expose the semantic-bottleneck attention over the `49` perceiver evidence latents, as intended

Saved token-count metadata:
- anchor: `attended_token_count=196`, `visual_token_count=196`
- `K=32`: `attended_token_count=49`, `visual_token_count=196`
- `K=8`: `attended_token_count=49`, `visual_token_count=196`

Simple attention-shape statistics over the saved `.pt` tensors:

| Model | Per-query map count | Mean per-query entropy | Mean per-query max weight |
|---|---:|---:|---:|
| Anchor | `49` | `3.9488` | `0.1199` |
| `K=32` | `32` | `3.2755` | `0.1489` |
| `K=8` | `8` | `3.4857` | `0.1192` |

Interpretation:
- the compressed runs are not just averaging everything uniformly
- `K=32` in particular shows sharper per-query distributions than the anchor over its attended evidence tokens
- `K=8` is still meaningfully selective, but not sharper than `K=32`

Caveat:
- this is an internal attention-structure read, not bbox supervision
- it confirms the bottleneck is attending selectively over the post-perceiver evidence latents
- it does **not** yet prove better object-level grounding against labeled regions

## 5. Overall Read

The diagnostics split the semantic-compression story into two parts:

1. Good news:
   Compression after evidence retrieval is far cheaper than expected. Even `49 -> 8` barely changes end-task accuracy, and the exported tokens become more linearly decodable rather than less.

2. Constraint:
   The compressed system does not automatically become less LM-dependent. At moderate compression (`K=32`) it behaves basically the same as the anchor, and at stronger compression (`K=8`) it becomes more reliant on LM adapters.

That means the semantic bottleneck is currently acting more like:

`evidence packing for the LM`

than:

`a standalone semantic answer state`

## 6. GQA Follow-Up

The original diagnostics run did not produce a meaningful GQA read because the first ablation pass used the VQA `official` scorer on single-answer GQA labels. That was corrected in a follow-up GQA-only rerun:

- `/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720`

Useful high-level result from that rerun:
- `attribute` was the only coarse GQA group where `K=8` became clearly more adapter-fragile than the Cement anchor
- `spatial` was only mildly worse
- `exist` was effectively unchanged
- `count` was too sparse on this local split to treat as reliable

That follow-up makes the original VQAv2 read more specific:
- the broad VQAv2 signal looked like a `yes/no` fragility problem
- the GQA rerun suggests the content underneath that is more likely **attribute verification / attribute binding** than generic relation or existence reasoning

## 7. What This Means Next

The strongest next questions are now:
- can the bottleneck be made more self-sufficient, rather than merely more compact?
- does adding compositional pressure such as GQA training help the compressed tokens carry more answer content before LM rescue?
- is `K=8` the right frontier for "semantic density", while `K=32` is the right frontier for "safe deployment"?

Practical takeaway:
- if the goal is minimal accuracy loss with cheaper LM-facing bandwidth, `K=32` is already enough
- if the goal is to push toward genuinely semantic exported tokens, `K=8` is the more interesting regime, but it needs follow-up work because adapter dependence rises sharply
