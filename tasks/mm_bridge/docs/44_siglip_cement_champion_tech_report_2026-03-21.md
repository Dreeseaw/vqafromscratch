# 44 SigLIP Cement Champion Tech Report (2026-03-21)

## Artifact

Current best single bridge checkpoint:

- run: `mmcement_v1_20260316_siglip_cement_questiononly_s53`
- peak checkpoint: [step_8000.tar](/home/wdree/percy/vqafromscratch/logs/mmcement_v1_20260316_siglip_cement_questiononly_s53/step_8000.tar)
- peak val accuracy: `0.6203`
- same-run final full eval at step 9000: `0.6082`

Primary source logs:

- [logfile.txt](/home/wdree/percy/vqafromscratch/logs/mmcement_v1_20260316_siglip_cement_questiononly_s53/logfile.txt)
- [43_cement_sweep_report_2026-03-17.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/43_cement_sweep_report_2026-03-17.md)

This report treats `step_8000.tar` as the current champion because Cement established that the `s53` run peaked at step `8000`, not `9000`.

## System Summary

This model is the Cement winner:

- frozen `SigLIP-B/16` vision tower
- `perceiver_resampler` bridge with depth `3`
- `question_hidden_attn` query path
- `question_only` bridge context
- no dynbudget or hard token selection
- LM residual visual adapters in top `3` LM blocks
- frozen VM, trainable bridge, and top-LM finetuning

Relevant runtime lines from the winning run:

- vision: `siglip_base`
- bridge: `perceiver_resampler`, `query_depth=3`, `num_visual_tokens=49`
- qquery mode: `question_hidden_attn`
- question context mode: `question_only`
- LM adapters: `cross_attn`, depth `3`
- batch layout: `96 x 2`
- optimizer horizon: `9000` steps, cosine decay, warmup `600`

## Parameter Breakdown

Counts below were taken from the actual champion checkpoint after applying its real freeze configuration.

| Component | Total Params | Trainable | Frozen |
|---|---:|---:|---:|
| VM | 92,884,224 | 0 | 92,884,224 |
| Bridge | 21,444,352 | 21,444,352 | 0 |
| LM | 46,174,208 | 19,903,488 | 26,270,720 |
| Total | 160,502,784 | 41,347,840 | 119,154,944 |

LM split:

- original LM params: `39,859,712`
- original LM trainable params: `13,588,992`
- LM-added adapter params: `6,314,496`

Interpretation:

- this is no longer a “tiny bridge on top of tiny frozen parts” system
- the champion is a medium-sized multimodal stack with a very large frozen VM and a substantial trainable bridge/LM interface
- most trainable capacity sits in the bridge plus LM-side multimodal adaptation, not in the vision tower

## Performance Snapshot

Peak checkpoint metrics at step `8000`:

| Metric | Value |
|---|---:|
| Overall | `0.6203` |
| Yes/No | `0.7693` |
| Number | `0.4586` |
| Other | `0.5534` |

Training state at the peak checkpoint line:

- step `8000`
- loss `0.5441`
- LR `3.58757e-05`
- train throughput `3.28 steps/s`

The same run drifted downward after the peak:

- periodic eval at step `9000`: `0.6113`
- final full eval at step `9000`: `0.6082`

So for this model family, “best checkpoint” and “last checkpoint” are materially different. Future frontier comparisons should preserve the peak checkpoint, not just the terminal one.

## Why This Model Won

Cement’s main result was not a new bridge family. It was the confirmation that, inside the stabilized SigLIP setup, `question_only` conditioning is better than `prompt_only`.

What this champion appears to get right:

- stronger language-aligned visual features from SigLIP
- attention-derived question querying rather than static or mean-only queries
- a fixed question-only bridge context, which avoids prompt clutter
- no hard visual token pruning before the perceiver
- enough LM-side visual reuse through residual adapters to exploit the bridge output

This is a fairly clean architecture. There is no exotic recurrence, no dynbudget routing, and no VM finetuning. The gain comes from a good VM plus a disciplined bridge/LM interface.

## Behavioral Read

From the Cement diagnostics and scorecard:

- the model is strongly image-dependent
- visual utilization is about `0.303` (`clean - zero`)
- it is especially vision-dependent on `number` and `other`
- it remains weak on OCR-like, temporal, and “why” reasoning questions
- calibration is decent but still overconfident overall (`ECE ≈ 0.049`)

The key practical implication is that this champion is a real multimodal model, not a language-prior artifact. But it still inherits the expected weaknesses of a frozen image-text vision tower and a relatively lightweight reasoning stack.

## Baseline Status

This checkpoint should be treated as the current reference baseline for future VM and LM swaps when the goal is “beat the best bridge system we have right now.”

Recommended reference artifact:

- [step_8000.tar](/home/wdree/percy/vqafromscratch/logs/mmcement_v1_20260316_siglip_cement_questiononly_s53/step_8000.tar)

Recommended comparison rule:

- compare against the peak `0.6203`, not the terminal `0.6082`
- keep the Cement bridge stack fixed unless the run is explicitly testing a bridge change

## Bottom Line

The current champion is a `160.5M`-parameter system with:

- a `92.9M` frozen SigLIP VM
- a `21.4M` fully trainable bridge stack
- a `46.2M` LM side, of which `19.9M` is trainable

Its best observed checkpoint reaches `0.6203` on VQAv2 val. The model wins by combining a strong frozen VM with a simple, stable, question-only attention-query bridge and modest LM-side multimodal adapters, not by aggressive token routing or deeper architectural novelty.
