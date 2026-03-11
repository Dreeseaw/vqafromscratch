# Oracle + TopK Hour Run (Non-QCond) - 2026-03-10

## Goal

Run one additional oracle-focused experiment for the next hour-ish without periodic eval overhead.

## Selected Run

- Run ID: `mmarch_cov_v1_20260310_perceiver_oracle196_topk64_h1`
- Bridge family: `perceiver_resampler`
- No qcond: `--no-bridge_question_conditioning`
- Oracle token count: `--num_visual_tokens 196`
- Adaptive token selector: `--bridge_token_selector_type topk --bridge_token_select_k 64`
- Eval cadence: `--eval_every 0` (final eval only at run end)
- Target: `--max_steps 5000`
- Memory-safe batching: `--batch_size 64 --grad_accum_steps 3`

## Why this run

1. Extends the oracle thread with a sparse-selection variant rather than repeating prior settings.
2. Keeps the run non-qcond to avoid the identified leakage failure mode.
3. Removes periodic eval interruptions that were harming runtime stability.

## Launcher

- Script: `tasks/mm_bridge/scripts/launch_oracle_topk64_single_run.sh`
- Resume behavior:
  - skips if `step_5000.tar` exists
  - resumes from latest `step_<N>.tar` if interrupted

## Launch command

```bash
./tasks/mm_bridge/scripts/launch_oracle_topk64_single_run.sh
```
