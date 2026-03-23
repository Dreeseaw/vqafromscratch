# 43 Cement Sweep Report (2026-03-17)

Plan: [doc 41](41_cement_sweep_plan_2026-03-16.md).
Run dirs: `logs/mmcement_v1_20260316_siglip_cement_{promptonly,questiononly}_s{42,53,97}/`.
Diagnostics: `logs/mm_diagnostic_test3/`.

---

## Tier 1: Head-to-Head — question_only vs prompt_only

All 6 runs completed at 9k steps, b96a2, warmup=600, SigLIP-B/16. Only `bridge_question_context_mode` and seed varied.

### Peak Accuracy (best checkpoint per seed)

| Arm | s42 | s53 | s97 | Mean | σ |
|---|---:|---:|---:|---:|---:|
| **question_only** | **0.6163** | **0.6203** | **0.6155** | **0.6174** | **0.0026** |
| prompt_only | 0.6130 | 0.6141 | 0.6085 | 0.6119 | 0.0029 |
| **delta (qonly − prompt)** | **+0.0033** | **+0.0062** | **+0.0070** | **+0.0055** | |

### Final Accuracy (step 9000)

| Arm | s42 | s53 | s97 | Mean |
|---|---:|---:|---:|---:|
| question_only | 0.6163 | 0.6082 | 0.6142 | 0.6129 |
| prompt_only | 0.6130 | 0.6083 | 0.6084 | 0.6099 |

### Per-Type Breakdown (step 9000, mean across seeds)

| Arm | Yes/No | Number | Other |
|---|---:|---:|---:|
| question_only | 0.7511 | 0.4548 | 0.5497 |
| prompt_only | 0.7479 | 0.4498 | 0.5474 |
| delta | +0.0032 | +0.0050 | +0.0023 |

### Verdict

**question_only wins.** It beats prompt_only at all 3 matched seeds (peak), has lower seed variance (σ 0.0026 vs 0.0029), and lifts all three answer types uniformly. Per the decision rule from doc 41, tie goes to question_only for its engineering advantage (no autoregressive dependency in query generation) — but this isn't a tie, it's a clean win.

**Peak frontier: 0.6174 (question_only, mean of best-checkpoint-per-seed). Completed full-eval mean: 0.6129. Best completed single run: 0.6163 (s42 step 9000).**

Note on peaks vs finals: s53 and s97 in both arms peak at step 8000, not 9000. The LR schedule is still declining usefully at 9k but some seeds slightly overfit or drift in the last 1k steps. For future sweeps, consider evaluating at both 8k and 9k.

---

## Tier 2: Oracle Diagnostics

All diagnostics run on the question_only s42 checkpoint (step 9000, final full eval `0.6163`).

### 2A: Visual Sufficiency

| Mode | Overall Acc | Δ vs Clean | Agreement w/ Clean |
|---|---:|---:|---:|
| clean | 0.6174 | — | 1.000 |
| zero (visual features zeroed) | 0.3146 | −0.3028 | 0.273 |
| noise (Gaussian noise features) | 0.6036 | −0.0138 | 0.801 |

**Visual utilization = 0.3028** (clean − zero). The model genuinely depends on vision — zeroing features doesn't just change confidence, it changes 72.7% of predictions. Language priors alone get 31.5%.

Noise robustness is high (only 1.4% drop, 80% agreement). The perceiver bridge smooths over moderate input perturbations effectively.

**Per-type visual utilization (clean − zero):**

| Type | Clean | Zero | Δ | Interpretation |
|---|---:|---:|---:|---|
| Yes/No | 0.759 | 0.634 | 0.125 | Low — strong language prior on binary Qs |
| Number | 0.456 | 0.120 | 0.336 | High — counting requires vision |
| Other | 0.552 | 0.123 | 0.430 | Highest — open-ended answers need vision |

**Strongest visual dependence by question type:**

| Question prefix | Clean | Zero | Δ |
|---|---:|---:|---:|
| what sport is | 0.932 | 0.005 | 0.927 |
| what animal is | 0.784 | 0.035 | 0.749 |
| what color is | 0.789 | 0.129 | 0.660 |
| what is this | 0.652 | 0.062 | 0.590 |
| what is the man | 0.647 | 0.060 | 0.587 |
| what are | 0.610 | 0.044 | 0.566 |

**Lowest visual dependence (language prior dominates):**

| Question prefix | Clean | Zero | Δ |
|---|---:|---:|---:|
| is this an | 0.744 | 0.668 | 0.076 |
| is this a | 0.771 | 0.666 | 0.105 |
| has | 0.710 | 0.653 | 0.057 |
| do you | 0.787 | 0.612 | 0.175 |

**Prefix geometry:** Zeroing collapses prefix representations toward a single point (pairwise cosine 0.974, batch variance drops 3.6×). The bridge produces a near-constant "default" prefix without visual signal, confirming the perceiver meaningfully transforms image features into diverse per-sample prefixes.

### 2B: Query Count Probes

**Incomplete.** The k=32 probe directory (`mm_diagnostic_test3_queryprobe_k32/`) contains only a code snapshot — no logfile or results. The probes did not produce usable data. This diagnostic is deferred.

### 2C: Per-Type Deep Breakdown

**Bug.** The fine-grained breakdown script ran on only 5 samples (likely a JSONL parsing issue that only grabbed the last eval entry rather than the full set). Results are meaningless.

However, the visual sufficiency test's `question_type_accuracy` field covers all 96k val samples across 60+ question prefixes and serves as a rich substitute. Highlights from the clean eval:

**Strongest categories (acc > 0.90):**
- what room is: 0.957
- what sport is: 0.932

**Weakest categories (acc < 0.25):**
- why / why is the: 0.153 / 0.166
- what is the name: 0.147
- what number is: 0.186
- what time: 0.235

The model excels at scene-level recognition and struggles with reasoning ("why"), reading/OCR ("what does the say", "what number", "what is the name"), and temporal concepts. This is consistent with a frozen SigLIP VM that was trained for image-text matching, not OCR or temporal reasoning.

### 2D: Calibration

**ECE: 0.0494** (10-bin). Mean confidence 0.666, mean accuracy 0.617. Consistently overconfident.

| Bin | N | Conf | Acc | Gap |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 843 | 0.077 | 0.063 | +0.014 |
| 0.1–0.2 | 4,071 | 0.154 | 0.130 | +0.024 |
| 0.2–0.3 | 5,156 | 0.252 | 0.212 | +0.040 |
| 0.3–0.4 | 5,795 | 0.350 | 0.270 | +0.080 |
| 0.4–0.5 | 6,140 | 0.451 | 0.350 | +0.101 |
| **0.5–0.6** | **12,603** | **0.553** | **0.556** | **−0.003** |
| 0.6–0.7 | 10,900 | 0.648 | 0.612 | +0.036 |
| 0.7–0.8 | 18,578 | 0.748 | 0.676 | +0.072 |
| 0.8–0.9 | 13,235 | 0.849 | 0.775 | +0.074 |
| 0.9–1.0 | 18,679 | 0.963 | 0.932 | +0.031 |

Worst calibration in the 0.3–0.5 range (gap ~0.08–0.10). The 0.5–0.6 bin is the only well-calibrated zone. The 0.9+ bin is well-calibrated (gap 0.031) — the model knows when it's very confident. Temperature scaling with T ≈ 1.3 would likely halve the ECE.

### 2E: Grounding Inspection

Qualitative attention maps saved for ~40 correct and ~15 incorrect samples across multiple question types. Available in `logs/mm_diagnostic_test3/diagnostics/grounding/{correct,incorrect}/`. Each sample has a `.png` (overlay visualization) and `.pt` (raw attention weights). Not summarized here — these are for visual review when debugging specific failure modes.

---

## Infrastructure Notes

4 batch launches were needed to complete the 6 runs:

| Batch | Time | Outcome |
|---|---|---|
| `153613` | 15:36 | Dry run (validation only, no training) |
| `175847` | 17:58 | s42 pair completed; s53/s97 pairs failed (likely GPU resource / resume issues) |
| `205438` | 20:54 | s42 skipped; resume attempts for s53/s97 failed |
| `231655` | 23:17 | s42 skipped; s53 and s97 pairs completed fresh (~63 min each) |

Total wall time: ~12h from first launch to last completion. Effective GPU time: ~6.3h (6 runs × ~63 min). No memory leak issues observed at b96a2.

---

## Frozen Cement Config

This is the locked bridge configuration for all future VM/LM comparisons on VQAv2. When swapping a new VM or LM, keep everything here identical except the component being tested.

```
# --- Vision ---
--vision_model siglip_base
--vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224

# --- Bridge ---
--bridge_type perceiver_resampler
--bridge_query_depth 3
--bridge_num_heads 8
--num_visual_tokens 49
--bridge_token_reduce adaptive_pool
--bridge_add_2d_pos_emb
--bridge_question_conditioning
--bridge_query_bank_mode question_hidden_attn
--bridge_question_context_mode question_only       ← settled by Cement
--bridge_qquery_scale 1.0
--bridge_qcond_scale 0.5
--bridge_token_selector_type none
--bridge_token_select_k 0

# --- LM Adapters ---
--lm_visual_adapter_type cross_attn
--lm_visual_adapter_layers 3
--lm_visual_adapter_num_heads 8
--lm_visual_adapter_dropout 0.0
--lm_visual_adapter_gate_init 0.5

# --- Freeze ---
--freeze_strategy bridge_plus_top_lm
--train_top_lm_layers 2

# --- Training ---
--max_steps 9000
--lr 0.0002
--lr_schedule cosine
--lr_warmup_steps 600
--lr_min_ratio 0.15
--batch_size 96
--grad_accum_steps 2
--eval_batch_size 96
--precision bf16

# --- Eval ---
--eval_scorer official
--eval_every_steps 1000
--eval_max_batches 100
```

**Seed for reproducibility:** 42 (default). Expected accuracy: 0.616 ± 0.003.

---

## Reference Scorecard

| Metric | s42 | s53 | s97 | Mean | σ |
|---|---:|---:|---:|---:|---:|
| **Overall (peak)** | **0.6163** | **0.6203** | **0.6155** | **0.6174** | **0.0026** |
| Yes/No | 0.759 | 0.738 | 0.757 | 0.751 | 0.012 |
| Number | 0.457 | 0.454 | 0.453 | 0.455 | 0.002 |
| Other | 0.550 | 0.551 | 0.548 | 0.550 | 0.001 |
| Visual utilization | — | — | — | 0.303 | — |
| ECE | — | — | — | 0.049 | — |
| Language-only baseline | — | — | — | 0.315 | — |

### Known Weaknesses

1. **Reasoning/causal**: "why" questions at 15%. Bridge architecture has no explicit reasoning mechanism.
2. **OCR/reading**: "what does the say", "what is the name", "what number" all below 20%. SigLIP was not trained for text recognition.
3. **Temporal**: "what time" at 24%. No temporal features in a single-frame model.
4. **Calibration**: Overconfident by ~5–10% in the 0.3–0.5 and 0.7–0.9 confidence ranges.
5. **Yes/No language bias**: Zero-mode still gets 63.4% on yes/no questions — strong language prior. Real visual utilization on binary questions is only 12.5%.

### Comparison Protocol

To test a new VM or LM against this baseline:

1. Swap the component in the config above (e.g., replace `--vision_model siglip_base` with the new VM)
2. Keep all other flags identical
3. Train for 9k steps with seed=42, b96a2
4. Run `./tasks/mm_bridge/scripts/run_diagnostics.sh <new_run_id>`
5. Compare overall + per-type + visual sufficiency against this scorecard
6. A VM that beats 0.6174 overall AND improves visual utilization above 0.303 is a real gain
