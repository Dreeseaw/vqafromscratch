# Semantic Compression Synopsis

This note summarizes the completed semantic-compression sweep relative to the Cement full-eval anchor and records the next diagnostics path.

Reference anchor:
- Cement full-eval single-run reference: `logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar`
- anchor full-eval score: `0.6163`

## Core Result

The late semantic bottleneck held up much better than expected.

Completed full-eval runs:

| Run | Exported semantic tokens | Final full eval | Delta vs Cement full-eval anchor |
|---|---:|---:|---:|
| `mmsemantic_v1_20260322_k32` | 32 | `0.6158` | `-0.0005` |
| `mmsemantic_v1_20260322_k16` | 16 | `0.6157` | `-0.0006` |
| `mmsemantic_v1_20260322_k8` | 8 | `0.6154` | `-0.0009` |
| `mmsemantic_v1_20260322_k4` | 4 | `0.6136` | `-0.0027` |

`K=32` was completed through a resume-only full eval after `step_4000`; its authoritative full-eval score is logged in [logfile_from_4000.txt](/home/wdree/percy/vqafromscratch/logs/mmsemantic_v1_20260322_k32/logfile_from_4000.txt).

## Interpretation

The main modeling result is that the LM-facing bridge export can be compressed very aggressively without materially hurting VQAv2 accuracy:

- `49 -> 32` is effectively lossless
- `49 -> 16` is effectively lossless
- `49 -> 8` is still effectively lossless at the level of final full-eval accuracy
- even `49 -> 4` only drops about `0.27` points

That strongly supports the semantic-bottleneck thesis:
- the perceiver appears to retrieve richer evidence than the LM actually needs to consume directly
- most answer-relevant information can be packed into a much smaller late token set if compression happens after evidence retrieval rather than before it

The answer-type pattern is also stable:
- `yes/no` remains strongest across all `K`
- `number` and `other` do not collapse sharply at `K=8`
- the first clearly meaningful degradation appears only by `K=4`

## What It Means

This sweep makes the project look less like a raw token-bandwidth problem and more like a late semantic-packing problem.

The bridge does not seem to need `49` LM-facing tokens to preserve performance. The current evidence says:

- dense retrieval before compression matters
- late export budget matters much less than expected
- the real next question is not "can we compress?" but "what semantics survive compression, and how much LM reasoning is still compensating?"

That is why the next diagnostics should focus on:

1. linear probe quality on exported semantic tokens
2. LM visual-adapter ablation sensitivity
3. grounding/spatial-binding preservation

## Prepared Diagnostics

Prepared launcher:
- [launch_semantic_compression_diagnostics_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_semantic_compression_diagnostics_v1.sh)

Default comparison set:
- Cement full-eval anchor
- `K=32` best completed compressed run
- `K=8` frontier compressed run

Prepared tasks:
- semantic linear probe via [mm_semantic_probe.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_semantic_probe.py)
- LM visual-adapter ablation via [mm_adapter_ablation.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_adapter_ablation.py)
- grounding inspection via [mm_grounding_inspection.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_grounding_inspection.py)

Launch command:

```bash
./tasks/mm_bridge/scripts/launch_semantic_compression_diagnostics_v1.sh
```

Recommended read order for the diagnostics:
- anchor vs `K=32` probe gap
- anchor vs `K=32` adapter-ablation drop
- `K=8` as the stress-test point
- grounding last, after the semantic/adapter picture is clear
