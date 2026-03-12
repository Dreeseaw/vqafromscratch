# Arch Overviews

Purpose:
- keep one compact map of the bridge architectures and modifiers used in this task
- show the routing at a glance
- clarify what each idea is trying to fix in the frozen-bridge VQA setup

Scope:
- frozen visual model
- mostly frozen LM
- bridge produces a small visual prefix for the LM

Legend:
- `VM` = visual model
- `LM` = language model
- `K` = number of prefix tokens sent into the LM
- `qctx` = pooled question context

## 1. Perceiver Resampler

Reference family. This is the current main baseline line.

```text
image
  |
  v
VM features -> visual tokens ------------------------------+
                                                          |
learned latent tokens (K) -> cross-attn -> self-attn -----+  x D rounds
                                                          |
                                                          v
                                                   K LM-prefix tokens
                                                          |
                                                          v
                                                          LM
```

Purpose in this task:
- compress a large visual token grid into a small LM-facing prefix
- keep the bridge expressive without changing the LM internals

Novelty in this task:
- low novelty
- this is the main control architecture

## 2. Safe QCond Perceiver

Question-conditioned Perceiver. Same basic family, but the question modulates the latent queries before they extract image evidence.

```text
question/prompt -> LM embed -> pooled qctx -> FiLM / latent modulation
                                                 |
image -> VM features -> visual tokens -----------+----> perceiver rounds ----> K prefix -> LM
```

A slightly more explicit view:

```text
qctx ----> [gamma, beta]
             |
latent(K) -> modulated latent(K) -> cross-attn to visual tokens -> self-attn -> prefix
```

Legacy autoregressive variant discovered during this task:

```text
question/prompt + generated answer tokens so far
                      |
                      v
                  pooled qctx_t
                      |
latent(K) -> modulated latent(K, t) -> cross-attn to visual tokens -> prefix_t -> next answer token
```

Purpose in this task:
- make visual extraction question-aware
- stop asking the bridge to extract "generic useful tokens" for every question
- in the autoregressive variant, let each generated answer token refine which image evidence is extracted next

Novelty in this task:
- moderate
- well-established direction in the literature, but one of the most important upgrades for this project
- project-specific novelty: unlike standard question-conditioned bridges that usually condition only on the question, this task uncovered a stronger variant where generated answer tokens also feed back into the bridge, so visual tokens are re-queried autoregressively during decoding

## 3. Early-Layer Encoder Bridge

This is not a new bridge family by itself. It changes the visual source feeding the bridge.

```text
image
  |
  v
VM encoder grid --------------> bridge ------------------> K prefix -> LM

instead of

image
  |
  v
VM final / posterior_mu grid -> bridge ------------------> K prefix -> LM
```

Purpose in this task:
- test whether the final VM latent is over-compressed for VQA
- preserve more local detail for counting, spatial, and attribute questions

Novelty in this task:
- low-to-moderate
- research-backed feature-source change, not a brand new bridge mechanism

## 4. Multi-Scale Perceiver

Fuses early and late visual features before Perceiver extraction.

```text
image
  |
  v
VM encoder grid --------> proj_low ---+
                                       \
                                        +--> concat --> spatial mixer --> perceiver --> K prefix -> LM
                                       /
VM posterior_mu grid ---> proj_high --+
```

If qcond is enabled:

```text
question/prompt -> qctx ------------------------------+
                                                      |
encoder grid + posterior grid -> fused tokens -> qcond perceiver -> prefix -> LM
```

Purpose in this task:
- combine spatial/detail signal from early layers with semantics from late layers
- test whether the best bridge needs both scales at once

Novelty in this task:
- moderate
- more novel than early-layer-only, but still strongly research-backed

## 5. Structured Roles Bridge

Uses fixed role groups inside the query tokens so the bridge can specialize parts of the prefix.

```text
visual tokens ----------------------------------------------+
                                                            |
query tokens + role embeddings                              |
   [obj] [attr] [spatial] [global] ...                      |
        |        |         |         |                      |
        +--------+---------+---------+--> cross/self blocks x D --> K prefix -> LM
```

Another way to view it:

```text
role embedding table
      |
      v
K role-tagged query slots -> attend to image -> refine together -> prefix
```

Purpose in this task:
- encourage token specialization instead of letting every prefix token be interchangeable
- improve coverage across object, attribute, spatial, and global question types

Novelty in this task:
- fairly novel
- one of the cleaner novelty-positive ideas in this project

## 6. Evidence Sparse Bridge

Builds one global summary token and a smaller set of sparse evidence tokens selected from the image grid.

```text
visual tokens --> scorer --> top-k evidence tokens --+
                                                     |
visual tokens --> mean --> summary token ------------+--> evidence queries -> self refine -> prefix -> LM
```

More explicit:

```text
visual grid
  | \
  |  \-> summary token
  |
  +-> score each token -> select top-k -> cross-attend with evidence queries -> refine -> concat summary
```

Purpose in this task:
- test sparse evidence extraction instead of uniform dense compression
- see whether the bridge should focus on a few salient regions plus one global token

Novelty in this task:
- fairly novel
- more speculative than structured roles, but useful for sparse-evidence exploration

## 7. Hybrid Constant + Image Bridge

Mixes a learned constant prefix with an image-conditioned prefix.

```text
learned constant prefix ------------------+
                                          \
                                           -> alpha-mix -> final prefix -> LM
                                          /
image -> image bridge -> image prefix ----+
```

Token-gated form:

```text
alpha per token:

[learned tok_1] ---\
[image   tok_1] ----> mix

[learned tok_2] ---\
[image   tok_2] ----> mix

...
```

Purpose in this task:
- keep a stable LM-facing prior while still injecting image information
- reduce the chance that the whole prefix becomes too noisy or too image-dependent

Novelty in this task:
- moderate
- not brand new, but a useful alternative family to pure image-only prefix extraction

## 8. Geometry-Aware Prefix Calibration

This is a post-bridge modifier, not a standalone bridge family.

```text
raw bridge prefix
      |
      v
LayerNorm -> gate/bias -> optional geometry MLP / token mixer -> calibrated prefix -> LM
```

Expanded view:

```text
prefix -> LN -> scale/bias ------------------------------+
                                                        |
prefix -> small residual geometry module (optional) ----+--> calibrated prefix -> LM
```

Purpose in this task:
- fix bridge-to-LM interface mismatch
- make the prefix land in a geometry/norm regime the frozen LM can use more reliably

Novelty in this task:
- potentially novel in the context of this project
- closer to "interface engineering" than to a new visual extractor

## 9. Token Selection / Oracle Front-End

This is another front-end modifier, not a full bridge family.

Two common forms used here:

```text
oracle:
image -> large VM token grid (e.g. 196) -> bridge -> prefix -> LM
```

```text
selector:
image -> VM token grid -> score/select top-k -> bridge -> prefix -> LM
```

Combined:

```text
image -> big token grid -> selector -> smaller evidence set -> bridge -> prefix -> LM
```

Purpose in this task:
- test whether the bridge is losing too much evidence during compression
- distinguish "need more raw visual tokens" from "need smarter token choice"

Novelty in this task:
- low-to-moderate
- useful experimentally, but more of a routing/probing tool than a new bridge family

## 10. How These Pieces Relate

A useful mental grouping is:

Core bridge families:
- `perceiver_resampler`
- `multiscale_perceiver`
- `structured_roles`
- `evidence_sparse`
- `hybrid_const_image`

Evidence-source changes:
- early-layer / `encoder`
- multiscale / `encoder_plus_posterior_mu`
- oracle token count increases

Conditioning changes:
- safe qcond / prompt-conditioned latent modulation
- safe qcond autoregressive refinement / answer-token-conditioned visual re-query

Interface changes:
- geometry-aware prefix calibration

Selection changes:
- token selector / top-k evidence routing

## 11. Project Read

Within this task, the main architectural questions have been:

1. Should visual extraction be question-aware?
2. Are final VM latents too compressed for VQA?
3. Should prefix tokens specialize into roles?
4. Is sparse evidence better than dense compression?
5. Is the real bottleneck extraction, or the bridge-to-LM interface geometry?

That is why the most important families here are:
- safe qcond perceiver
- early-layer / multiscale variants
- structured roles
- geometry-aware calibration

Those are the architectures that most directly move project understanding, not just benchmark decimals.
