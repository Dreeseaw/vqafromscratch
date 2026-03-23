# Semantic Adapter Fragility Report

Source bundle:
- `/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_v1_20260323_110933`

Primary analysis artifact:
- [fragility_analysis.md](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_v1_20260323_110933/fragility_analysis.md)

## Summary

This experiment answered the main open question from the semantic-compression diagnostics:

`why does K=8 need the LM adapters so much more than the anchor?`

The answer is not "compression hurts everything equally," and it is not mainly an `other`-category collapse.

The real result is:
- `K=32` is essentially category-matched to the Cement anchor
- the extra `K=8` fragility is concentrated overwhelmingly in `yes/no`
- that extra fragility appears mostly at the final `keep 1 -> keep 0` transition, not gradually across every ablation step

So the strongest current interpretation is:

`K=8 preserves answer information, but the fully frozen LM has trouble cashing out compressed verification-style evidence without at least some adapter support`

## 1. VQAv2 Category Grid

Core table from the completed run:

| Model | Category | Keep 3 | Keep 2 | Keep 1 | Keep 0 | Delta 3->0 |
|---|---|---:|---:|---:|---:|---:|
| Anchor | Yes/No | `0.7589` | `0.7569` | `0.7411` | `0.6243` | `0.1346` |
| Anchor | Number | `0.4573` | `0.4516` | `0.4280` | `0.2957` | `0.1616` |
| Anchor | Other | `0.5499` | `0.5391` | `0.5059` | `0.3676` | `0.1823` |
| K=32 | Yes/No | `0.7606` | `0.7586` | `0.7420` | `0.6274` | `0.1332` |
| K=32 | Number | `0.4582` | `0.4509` | `0.4249` | `0.2883` | `0.1699` |
| K=32 | Other | `0.5474` | `0.5369` | `0.5006` | `0.3622` | `0.1851` |
| K=8 | Yes/No | `0.7611` | `0.7585` | `0.7286` | `0.4510` | `0.3100` |
| K=8 | Number | `0.4583` | `0.4513` | `0.4178` | `0.2757` | `0.1826` |
| K=8 | Other | `0.5462` | `0.5344` | `0.4934` | `0.3423` | `0.2038` |

Main takeaways:
- `K=32` is basically a no-op relative to the anchor. The deltas are nearly unchanged for all three answer types.
- `K=8` is where the real shift happens.
- But the shift is not broad-based. `number` and `other` move only modestly relative to the anchor.
- `yes/no` is the outlier.

## 2. Fragility Ratios

Fragility ratio:

`(delta_3->0 at compressed K) / (delta_3->0 at anchor)`

Results:

| Category | K32 / Anchor | K8 / Anchor |
|---|---:|---:|
| Yes/No | `0.9897` | `2.3035` |
| Number | `1.0513` | `1.1301` |
| Other | `1.0154` | `1.1180` |

This is the central result of the experiment.

The overall `K=8` fragility increase is being driven almost entirely by `yes/no`. The ratio is `2.30x`, while `number` and `other` are both close to `1.1x`.

That rules out the earlier leading guess that the semantic bottleneck was mainly breaking broad `other`-style compositional diversity. It also rules out a strong "count collapse" story on VQAv2 itself.

The compressed system is specifically becoming much worse at binary verification once all adapter support is removed.

## 3. Where the Failure Appears

Stagewise drop decomposition:

### Anchor yes/no
- `3 -> 2`: `0.0020`
- `2 -> 1`: `0.0158`
- `1 -> 0`: `0.1168`

### K=8 yes/no
- `3 -> 2`: `0.0025`
- `2 -> 1`: `0.0299`
- `1 -> 0`: `0.2776`

Interpretation:
- the compressed `K=8` system does **not** degrade much faster than the anchor while at least one adapter is still alive
- the real collapse happens when the last remaining adapter support is removed

So this is not "the compressed model needs all 3 adapters to function." It is closer to:

`K=8 still works with partial adapter support, but a fully frozen LM cannot reliably read the compressed verification signal by itself`

That is a more precise and more useful statement.

## 4. Probe vs Keep-0

Using the completed `K=8` probe from the semantic diagnostics:

| Category | Probe | K8 Keep 0 | Probe - Keep 0 |
|---|---:|---:|---:|
| Yes/No | `0.6199` | `0.4510` | `0.1689` |
| Number | `0.3511` | `0.2757` | `0.0754` |
| Other | `0.4316` | `0.3423` | `0.0892` |

This is the strongest mechanistic clue in the report.

The largest probe-ablation gap is also `yes/no`.

That means:
- the `K=8` tokens still contain a strong yes/no signal
- but the frozen LM without adapters is much worse at using that signal than a tiny direct classifier is

So the problem is not simply "the information is gone."

The more accurate read is:

`the information is present in the compressed tokens, but the frozen LM path cannot decode or route it well without multimodal adapter support`

This points more toward an interface/usage problem than a pure semantic-capacity problem.

## 5. GQA Slice

The first GQA pass in the main fragility bundle was invalid because it accidentally used the VQA `official` scorer on single-answer GQA labels. I reran the GQA subset analysis only with the `proxy` scorer here:

- `/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720`

Relevant outputs:
- [anchor_spatial.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720/anchor_spatial.json)
- [k8_spatial.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720/k8_spatial.json)
- [anchor_attribute.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720/anchor_attribute.json)
- [k8_attribute.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720/k8_attribute.json)
- [anchor_exist.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720/anchor_exist.json)
- [k8_exist.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720/k8_exist.json)
- [anchor_count.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720/anchor_count.json)
- [k8_count.json](/home/wdree/percy/vqafromscratch/logs/mmsemantic_fragility_gqa_v2_20260323_155720/k8_count.json)

Proxy-scored keep-3 vs keep-0 deltas:

| Group | Anchor delta | K8 delta |
|---|---:|---:|
| Spatial | `0.0217` | `0.0343` |
| Attribute | `0.0268` | `0.0827` |
| Exist | `0.0543` | `0.0553` |
| Count | `0.0667` | `0.1000` |

Interpretation:
- `attribute` is the only clearly amplified GQA group. `K=8` is about `3.1x` more fragile there than the anchor.
- `spatial` is only modestly worse at `K=8`.
- `exist` is basically unchanged.
- `count` points in the same direction as "more fragile under compression," but the local slice is only `10` examples and should not be trusted.

This matters because it sharpens the VQAv2 finding:
- VQAv2 answer types said the extra fragility was mostly `yes/no`
- GQA says that within coarse reasoning families, the most likely content behind that `yes/no` failure is **attribute binding / attribute verification**, not generic spatial relations and not generic existence

So the compressed `K=8` bottleneck appears to preserve broad scene semantics reasonably well, but it makes the LM-adapter interface more important for reading out fine attribute evidence after strong compression.

## 6. Modeling Read

This experiment changes the interpretation of the semantic bottleneck quite a bit.

Before this run, the simplest story was:
- stronger compression makes the LM do more rescue work in general

After this run, the more precise story is:
- stronger compression mostly preserves `number` and `other` fragility at roughly anchor-like levels
- the real new weakness is binary visual verification
- the compressed tokens still encode that verification signal strongly enough for a linear probe to use it
- the failure is therefore mostly in LM-side consumption of the signal, not total destruction of the signal

That is a much better outcome than a broad semantic collapse would have been.

It means the next fixes should probably focus on:
- how compressed tokens are exposed to the LM
- how verification-style evidence is preserved/read out

more than on "recovering lost semantics" in a generic sense.

## 7. Bottom Line

Best concise conclusion:

`K=8 is not failing because the semantic bottleneck destroys all answer information. It is failing mainly because fully removing the LM adapters cripples the model's ability to use compressed verification evidence, with the strongest follow-up signal pointing to attribute-heavy questions.`

That is a narrower and more actionable failure mode than the original aggregate diagnostics implied.
