# 41 Cement Sweep Plan (2026-03-16)

## Purpose

Cement is not a research sweep. It is a **lockdown sweep** that produces two deliverables:

1. A **frozen mid-tier bridge configuration** with seed-checked accuracy, to be used as the fixed test harness for evaluating future VMs and LMs.
2. A **diagnostic harness** (oracle experiments, per-type breakdowns, calibration analysis, grounding inspection) that can be re-run on any future model to characterize where information is being lost.

After Cement, the mm_bridge task pauses. The project shifts to custom VM pre-training (datasetting → ViT-S/16 training) and LM work. When those produce a candidate VM or LM, the bridge config and diagnostic harness from Cement are the stable reference to test against.

## Ancestry

Cement inherits from Hardhat (docs 37/37codex) and the future directions doc (40). It cherry-picks only the oracle/diagnostic work from Ironclad (doc 39, Tier 0 + Tier 4). The training methodology experiments (contrastive aux, TPCL, answer-type head) and architecture changes (deformable attention, more queries) are deferred to part 2, after the VM/LM work returns.

## Entering State

### Hardhat Frontier

| Run | Steps | Final | Y/N | Num | Other | Status |
|---|---:|---:|---:|---:|---:|---|
| SigLIP-B questiononly 18k | 10k/18k | 0.6173 (periodic 10k) | 0.7774 | 0.4384 | 0.5469 | crashed (mem leak) |
| SigLIP-B questiononly 18k (9k eval-only) | 9k | 0.6111 | 0.7603 | 0.4480 | 0.5407 | complete (eval-only on 9k ckpt, LR schedule not comparable — see caveat) |
| **SigLIP-B baseline (prompt_only) ← CEMENT CONFIG** | **9k** | **0.6095** | **0.7446** | **0.4532** | **0.5482** | **complete** |
| DINOv2-B baseline | 9k | 0.5953 | 0.7388 | 0.4208 | 0.5323 | complete |

### The questiononly Question Is Open

questiononly won cleanly on DINOv2-S: 0.5803 vs 0.5762 (+0.004), same seed, same schedule. On SigLIP, the only data point is the 18k run's 9k eval-only pass (0.6111 vs 0.6095), but that comparison is confounded by LR schedule differences (warmup=1200 vs 600 — see caveat below). We cannot call this settled.

Beyond accuracy, questiononly has a real engineering advantage: the query bank is derived from question text tokens only, with no dependency on having processed visual tokens through the LM first. This simplifies inference and removes an autoregressive dependency from the bridge's query generation path. If questiononly matches or beats prompt_only on accuracy, it's the strictly better config.

**Cement must produce a clean head-to-head on SigLIP-B: prompt_only vs question_only at matched seeds, schedule, and batch layout.** Then seed-check the winner.

### LR Schedule Caveat

The SigLIP questiononly 18k run used `--lr_warmup_steps 1200` (scaled for 18k total steps), while the SigLIP baseline used `--lr_warmup_steps 600` (standard for 9k). The eval-only pass at the 9k checkpoint from the questiononly 18k run (0.6111) was therefore trained under a different LR schedule than the baseline (0.6095) — a longer warmup means lower peak LR at step 9k and a different LR trajectory throughout. **The 0.6111 vs 0.6095 comparison is confounded by the schedule difference, not just questiononly vs prompt_only.** This reinforces the decision to not chase the questiononly delta on SigLIP — the observed +0.0016 may be partly or entirely a schedule artifact.

For Cement seed checks, all runs must use the same `--lr_warmup_steps 600` as the original baseline to ensure comparability.

### Batch Layout

All Cement runs use **b96a2** (batch_size=96, grad_accum_steps=2, eval_batch_size=96). No exceptions.

## What Cement Produces

### Deliverable 1: Frozen Bridge Config

A single bridge configuration, seed-checked, with known accuracy and per-type breakdown on VQAv2 val. This config is the fixed reference for all future VM/LM experiments. When a new VM or LM is ready to test, swap it in and compare against this baseline — any accuracy change is attributable to the new component, not bridge variance.

### Deliverable 2: Diagnostic Harness

A set of eval scripts and documented procedures that can be re-run on any model to answer:

- Where does information die? (oracle experiments)
- What question types fail? (per-type breakdown beyond yes/no/number/other)
- Does the model know when it's wrong? (calibration)
- Is the model looking at the right places? (grounding inspection)
- How much visual signal is the model actually using? (visual sufficiency)

These diagnostics should be runnable as a single script or a small set of scripts, not scattered one-off commands.

## Tier 1: Head-to-Head + Seed Check (6 training runs)

Tier 1 settles the question context mode on SigLIP-B, then seed-checks the winner.

### Shared Config (everything except question context mode)

```
VM: SigLIP-B/16
  --vision_model siglip_base
  --vision_checkpoint logs/hf_vision/google_siglip_base_patch16_224
  --batch_size 96 --grad_accum_steps 2 --eval_batch_size 96

Bridge: perceiver nodynbudget + attnqquery
  --bridge_type perceiver_resampler
  --bridge_query_depth 3
  --bridge_num_heads 8
  --num_visual_tokens 49
  --bridge_token_reduce adaptive_pool
  --bridge_add_2d_pos_emb
  --bridge_question_conditioning
  --bridge_query_bank_mode question_hidden_attn
  --bridge_qquery_scale 1.0
  --bridge_token_selector_type none
  --bridge_token_select_k 0

Adapters: cross_attn d3
  --lm_visual_adapter_type cross_attn
  --lm_visual_adapter_layers 3
  --lm_visual_adapter_num_heads 8
  --lm_visual_adapter_dropout 0.0
  --lm_visual_adapter_gate_init 0.5

Training: standard 9k
  --max_steps 9000 --lr 0.0002 --lr_schedule cosine
  --lr_warmup_steps 600 --lr_min_ratio 0.15
  + all standard COMMON_ARGS from Hardhat
```

### prompt_only arm (3 seeds)

| Run | Seed | Question context | Batch layout | Status |
|---|---:|---|---|---|
| `siglip_cement_promptonly_s42` | 42 | `--bridge_question_context_mode prompt_only` | b96a2 | pending |
| `siglip_cement_promptonly_s53` | 53 | `--bridge_question_context_mode prompt_only` | b96a2 | pending |
| `siglip_cement_promptonly_s97` | 97 | `--bridge_question_context_mode prompt_only` | b96a2 | pending |

Note: We have the Hardhat run `mmhardhat_v1_20260315_siglip_attnqquery_nodynbudget_adapter_d3` at seed=42 with prompt_only scoring 0.6095, but it ran at b192a1. The s42 run here re-runs at b96a2 to keep all 6 runs on the same batch layout. If the s42 b96a2 result matches 0.6095 closely, batch layout is not a significant confound and the Hardhat number can be treated as a 4th data point.

### question_only arm (3 seeds)

| Run | Seed | Question context | Batch layout | Status |
|---|---:|---|---|---|
| `siglip_cement_questiononly_s42` | 42 | `--bridge_question_context_mode question_only` | b96a2 | pending |
| `siglip_cement_questiononly_s53` | 53 | `--bridge_question_context_mode question_only` | b96a2 | pending |
| `siglip_cement_questiononly_s97` | 97 | `--bridge_question_context_mode question_only` | b96a2 | pending |

### Decision After Tier 1

Compare the two arms:

| Arm | s42 | s53 | s97 | Mean | σ |
|---|---:|---:|---:|---:|---:|
| prompt_only | | | | | |
| question_only | | | | | |

**Decision rule:** If the question_only mean ≥ prompt_only mean, question_only wins (accuracy tie goes to questiononly due to its engineering advantage — no autoregressive dependency in query generation). If prompt_only mean > question_only mean by more than 1σ, prompt_only wins.

The winning arm's median run becomes the Cement reference checkpoint. All 3 seeds of the winning arm populate the reference card.

### Tier 1 Subtotal: ~8.4h (6 runs × ~1.4h)

## Tier 2: Oracle Diagnostics (inference-only, no training)

All oracles use the Cement config checkpoint. No new training required.

### Oracle 2A: Visual Sufficiency Test

Feed blank images (zeros) and random noise images through the full pipeline. Eval on full val.

This produces three numbers:
- `real_acc`: accuracy with real images (the Cement frontier, ~0.61)
- `blank_acc`: accuracy with zero-valued images
- `random_acc`: accuracy with Gaussian noise images

The gap `real_acc - max(blank_acc, random_acc)` is the model's **effective visual utilization**. The `max(blank_acc, random_acc)` approximates the language-only baseline — what the model gets right by exploiting question priors alone.

At 0.61 overall and an expected language-only baseline of ~0.42, visual utilization is ~0.19. Breaking this down by answer type will show where visual signal matters most (expect: high for `other`, moderate for `number`, low for `yes/no`).

**Eng work:** Minimal. Modify the eval dataloader to replace `image` with `torch.zeros_like(image)` or `torch.randn_like(image)`. Two eval-only passes.

**Est. time:** ~40min (two full-val forward passes)

### Oracle 2B: Query Count Sweep (Perceiver Compression Test)

This replaces the "bridge bypass" oracle from Ironclad with a cleaner test that doesn't require architectural surgery.

Train three short probe runs (3k steps each) with different perceiver output query counts:
- 32 queries (more compression)
- 49 queries (current default)
- 96 queries (less compression)

If accuracy scales with query count: the perceiver is compressing too aggressively, and the bridge has headroom from more queries alone.
If accuracy is flat across query counts: 49 queries already capture what matters, and the bottleneck is downstream (LM reasoning or VM feature quality).

**Important:** These are 3k-step probe runs for diagnostics, not full 9k training runs. The point is the trend, not the absolute score. Use the Cement config for everything else.

**Config delta:** `--num_visual_tokens 32` / `49` / `96`

Note: 96 queries leaves 160 text positions in the LM's 256-token context — still enough for VQA questions (avg ~12 tokens + answer template).

**Est. time:** ~1.5h total (3 × 0.5h for 3k-step probes)

### Oracle 2C: Per-Type Deep Breakdown

Go beyond yes/no / number / other. Parse the VQA question text to categorize into fine-grained types:

| Category | Pattern | Example |
|---|---|---|
| Color | `what color`, `what colour` | "What color is the bus?" |
| Counting | `how many` | "How many people are there?" |
| Spatial | `where`, `which side`, `left`, `right`, `above`, `below` | "Where is the cat?" |
| Action | `what is .* doing`, `what are .* doing` | "What is the man doing?" |
| Object ID | `what is this`, `what is that`, `what kind` | "What kind of animal is this?" |
| Yes/No existence | `is there`, `are there` | "Is there a dog?" |
| Yes/No attribute | `is the .* (color/size/etc)` | "Is the sky blue?" |
| Reading | `what does .* say`, `what is written` | "What does the sign say?" |

This requires a categorization script that reads the eval answers JSONL + the VQA question annotations and computes per-category accuracy. No model changes, just post-processing.

**Eng work:** One Python script (~100 lines) that:
1. Loads `fixed_eval_val_answers.jsonl` (final eval entry)
2. Loads VQAv2 val questions + annotations
3. Categorizes each question by regex patterns
4. Computes accuracy per category
5. Outputs a summary table + JSON

This script becomes part of the diagnostic harness — re-runnable on any future model's eval output.

**Est. time:** ~2h eng, ~5min to run

### Oracle 2D: Confidence Calibration

Compute Expected Calibration Error (ECE) over the Cement config's predictions.

This requires softmax confidence scores from the model, which means running eval with logit output. The procedure:

1. Run eval saving per-sample top-1 confidence (softmax of the max logit) and correctness
2. Bin into 10 confidence bins
3. Compute ECE = Σ (|bin_samples| / total) × |accuracy(bin) - confidence(bin)|
4. Plot reliability diagram

At 0.61 accuracy, expect significant overconfidence. The calibration profile tells us whether temperature scaling or other post-processing would help, and gives a baseline for future models.

**Eng work:** Modify eval to output per-sample confidence. Add a calibration analysis script (~60 lines).

**Est. time:** ~2h eng, ~20min eval pass

### Oracle 2E: Grounding Inspection (Qualitative)

Extract perceiver cross-attention weights from the Cement config. For 100 correctly-answered and 100 incorrectly-answered samples:
1. Visualize which spatial positions each query token attends to
2. Overlay on the original image
3. Save as a grid of attention maps

This is qualitative, not quantitative. The point is to build intuition: is the model attending to question-relevant regions when it's right? Is it attending to irrelevant regions when it's wrong?

**Eng work:** Forward hook to extract cross-attention weights from the perceiver. Visualization script (~80 lines).

**Est. time:** ~3h eng, ~15min inference

### Tier 2 Subtotal: ~2.5h GPU + ~7h eng

## Tier 3: Harness Packaging

After Tier 1 (locked config) and Tier 2 (diagnostics), package everything into a reusable harness.

### 3A: Diagnostic Runner Script

A single entry point:

```bash
./tasks/mm_bridge/scripts/run_diagnostics.sh <run_id> [--skip-training-probes]
```

That runs:
1. Visual sufficiency (blank + random images)
2. Per-type breakdown
3. Calibration analysis
4. Grounding inspection (100+100 samples)
5. Optionally: query count probes (if `--skip-training-probes` is not set)

And outputs a structured report to `logs/<run_id>/diagnostics/`.

### 3B: Cement Reference Card

A concise document (part of this doc's final section, filled in after runs complete) recording:

- The frozen Cement config (exact flags)
- Accuracy: overall, per-type (coarse + fine), seed variance
- Oracle results: visual utilization, query count trend, calibration ECE
- Known weaknesses (which question types fail, which visual properties are lost)
- Comparison protocol: how to use this config as a reference when testing a new VM or LM

This reference card is the thing that gets consulted when the project resumes with a new VM.

## Execution Schedule

```
Phase A: Tier 1 head-to-head + seeds               ~8.4h GPU
  prompt_only:   s42, s53, s97                      (3 runs × ~1.4h)
  question_only: s42, s53, s97                      (3 runs × ~1.4h)
  → compare arms, pick winner, lock Cement config

Phase B: Eng work for diagnostics                  ~7h eng (can overlap with Phase A training)
  visual sufficiency eval mod
  per-type breakdown script
  calibration analysis script + eval logit output
  grounding visualization script
  diagnostic runner wrapper

Phase C: Tier 2 oracle runs                        ~2.5h GPU
  Oracle 2A: visual sufficiency (2 eval passes)
  Oracle 2B: query count probes (3 × 3k steps)
  Oracle 2C: per-type breakdown (post-processing only)
  Oracle 2D: calibration (1 eval pass)
  Oracle 2E: grounding inspection (1 partial eval pass)

Phase D: Write Cement reference card                ~1h
```

**Total: ~10.9h GPU + ~8h eng**

## What Cement Does NOT Do

- No training methodology experiments (contrastive aux, TPCL, answer-type head — deferred to part 2)
- No bridge architecture changes (deformable attention, MoE — deferred to part 2)
- No knowledge distillation
- No new VM integration
- No LM scaling
- No 18k runs (the 9k config is the stable reference; 18k is an optimization axis to revisit later)
- No multi-benchmark expansion yet (harness is built for VQAv2; other benchmarks come when the custom VM is ready)

## After Cement

The project pauses on mm_bridge. Work shifts to:

1. **datasetting task**: curate a training corpus for the custom ViT
2. **Custom ViT-S/16 pre-training**: CLIP-style contrastive training, ~22M params, 196 tokens, 384-dim
3. **LM work** (scope TBD)

When a candidate VM or LM is ready, the test procedure is:

1. Swap the new component into the Cement bridge config
2. Train for 9k steps with the same COMMON_ARGS
3. Run the diagnostic harness
4. Compare against the Cement reference card

If the new component beats the Cement baseline on overall accuracy AND the diagnostics show improvement in specific categories, it's a real gain. If it beats overall but the per-type breakdown is weird (e.g., yes/no jumps but other drops), the gain may be from language bias exploitation rather than real visual understanding.

That's the value of the diagnostic harness: it turns "the number went up" into "the number went up for the right reasons."

---

## Cement Reference Card

*(Partially filled. Seed variance and oracle results TBD after Tier 1 and Tier 2 complete.)*

### Frozen Config

```
VM: siglip_base @ logs/hf_vision/google_siglip_base_patch16_224
Bridge: perceiver_resampler, query_depth=3, 8 heads, 49 queries, nodynbudget
Token path: adaptive_pool, 2d_pos_emb, no token selection
Query: question_hidden_attn, prompt_only, qquery_scale=1.0
Adapters: cross_attn d3, 8 heads, gate_init=0.5, dropout=0.0
Training: 9k steps, b96a2, cosine LR 2e-4, warmup 600, min_ratio 0.15, bf16
Freeze: bridge_plus_top_lm, train_top_lm_layers=2
```

### Accuracy

| Metric | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | σ |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.6095 | | | | | |
| Yes/No | 0.7446 | | | | | |
| Number | 0.4532 | | | | | |
| Other | 0.5482 | | | | | |

**Run provenance:**

| Seed | Run ID | Seed value | Batch layout | LR warmup | Status |
|---|---|---:|---|---:|---|
| 1 | `mmhardhat_v1_20260315_siglip_attnqquery_nodynbudget_adapter_d3` | 42 (default) | b192a1 | 600 | complete |
| 2 | `siglip_cement_seed2` | 53 | b96a2 | 600 | pending |
| 3 | `siglip_cement_seed3` | 97 | b96a2 | 600 | pending |
| 4 | `siglip_cement_seed4` | 131 | b96a2 | 600 | pending |

**Caveat:** Seed 1 ran at b192a1 (Hardhat). Seeds 2-4 will run at b96a2 (Cement). Different batch layouts produce different gradient noise at the same effective batch size (192). If the b96a2 seeds cluster away from 0.6095, the batch layout difference is a confound. In that case, a 4th b96a2 run with seed=42 should be added to isolate batch layout effect from seed variance.

Note: No clean SigLIP questiononly run at 9k with matched LR schedule exists. The 0.6111 figure from Hardhat Phase 2 used warmup=1200 (18k schedule) and is not comparable — see LR Schedule Caveat above.

### Fine-Grained Per-Type

| Category | Accuracy | N samples |
|---|---:|---:|
| Color | | |
| Counting | | |
| Spatial | | |
| Action | | |
| Object ID | | |
| Yes/No existence | | |
| Yes/No attribute | | |
| Reading | | |

### Oracle Results

| Oracle | Result | Interpretation |
|---|---|---|
| Visual utilization (real - blank) | | |
| Query count trend (32 vs 49 vs 96) | | |
| Calibration ECE | | |
| Dead token fraction | | |

### Known Weaknesses

*(TBD from diagnostics)*

### Comparison Protocol

To test a new VM or LM against this baseline:
1. Swap the component in the config above
2. Keep all other flags identical
3. Train for 9k steps with b96a2
4. Run `./tasks/mm_bridge/scripts/run_diagnostics.sh <new_run_id>`
5. Compare overall + per-type + oracles against this card
