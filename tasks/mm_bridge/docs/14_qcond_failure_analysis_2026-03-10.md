# Q-Conditioned Bridge Failure Analysis - 2026-03-10

## Scope

This report explains why the `qcond` runs in the architecture sweep collapsed despite low training loss.

Evaluated runs:
- `mmarch_cov_v1_20260310_perceiver_d3_anchor`
- `mmarch_cov_v1_20260310_perceiver_d3_qcond`
- `mmarch_cov_v1_20260310_perceiver_d3_qcond_encoder`
- `mmarch_cov_v1_20260310_perceiver_d3_topk24`
- `mmarch_cov_v1_20260310_perceiver_d3_qcond_topk24`
- `mmarch_cov_v1_20260310_perceiver_d3_qcond_topk24_encoder`

## Headline

The `qcond` models show classic **train/inference mismatch from answer leakage into conditioning**:

- training CE becomes extremely low (`~0.08-0.13`) while
- validation accuracy collapses (`~0.058-0.092`).

This pattern strongly indicates the question-conditioning signal during training contained information unavailable at inference.

## Results Snapshot

| Run | qcond | Final overall acc | Final answer-type (yes/no, number, other) |
|---|---:|---:|---|
| `perceiver_d3_anchor` | 0 | `0.4464` | `0.6841, 0.3127, 0.3004` |
| `perceiver_d3_topk24` | 0 | `0.4301` | `0.6770, 0.2992, 0.2763` |
| `perceiver_d3_qcond` | 1 | `0.0856` | `0.1602, 0.0301, 0.0434` |
| `perceiver_d3_qcond_encoder` | 1 | `0.0922` | `0.1749, 0.0378, 0.0436` |
| `perceiver_d3_qcond_topk24` | 1 | `0.0576` | `0.0894, 0.0289, 0.0409` |
| `perceiver_d3_qcond_topk24_encoder` | 1 | `0.0629` | `0.1023, 0.0323, 0.0409` |

## Primary Root Cause (High Confidence)

### 1) Conditioning uses full teacher-forced text during training

In multimodal forward:
- `text_emb` is built from `input_ids` (`train/mm.py:518`)
- `question_context` is pooled from that `text_emb` (`train/mm.py:492`, `train/mm.py:521`)
- For training batches, `input_ids` include prompt + answer tokens (`train/mm.py:691-700`).

So `qcond` conditioning sees answer tokens during training.

### 2) Inference does not have answer tokens

During generation:
- model runs from `prompt_ids` only (`train/mm.py:631-656`, `train/mm.py:739`).
- This excludes ground-truth answer tokens.

Net effect: the `qcond` signal distribution at inference is different from training (and missing leaked answer content).

### 3) Behavior matches leakage signature exactly

`qcond` runs:
- very low CE late in training, e.g. `loss_ce~0.084` at step 5000 and `~0.129` at step 7000.
- catastrophic val accuracy (`0.0576-0.0922`).

Non-`qcond` controls:
- normal CE (`~0.9-1.2`) with strong val accuracy (`0.4301-0.4464`).

This inversion (lower CE but far worse val accuracy) is not normal optimization noise; it is consistent with shortcut learning from leaked target information.

## Secondary Contributors (Medium Confidence)

1. **Higher prefix variance in qcond runs**
- `qcond` runs repeatedly show much larger `reg_var` terms (`~0.20-0.31`) vs anchor/topk non-qcond (`~0.03-0.05`).
- This likely worsens LM interface stability on top of the leakage issue.

2. **Top-k + qcond compounds collapse**
- `qcond_topk24` variants are the worst (`0.0576`, `0.0629`), suggesting token selection plus leaked modulation over-specializes to train-time shortcuts.

## Why this is not an eval artifact

- The collapse is consistent across all qcond variants.
- Both periodic and final eval agree within each run.
- Non-qcond runs in the same sweep setup remain strong.

## Confidence Assessment

- **High confidence** that train/inference conditioning mismatch via answer leakage is the dominant cause.
- **Medium confidence** that elevated prefix variance and token-selection interaction amplify the collapse severity.

## Suggested Next Checks (No code changes yet)

1. Run a quick diagnostic comparing train-time and inference-time `question_context` statistics for one qcond checkpoint to quantify shift magnitude.
2. Verify prediction entropy on val for qcond checkpoints to confirm degenerate decoding behavior.
3. Keep qcond experiments paused until conditioning is restricted to true question-prefix tokens only.

