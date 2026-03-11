# Historical Gap Audit

## Goal
Confirm the observed gap: learned constant visual tokens outperform image-conditioned bridge tokens in this frozen VM+LM setup.

## Data Source
- Existing training logs under `logs/mm*` and `logs/mmbr*`.
- Primary full-val comparisons:
  - `logs/mmbr_basesweep_lt1/logfile.txt`
  - `logs/mmbr_basesweep_on_high/logfile.txt`
  - `logs/mmbr_basesweep_off_high/logfile.txt`

## Key Results (from logged final eval)

| run | bridge | K | val overall acc |
|---|---|---:|---:|
| `mmbr_basesweep_lt1` | `learned_tokens` | 49 | 0.3540 |
| `mmbr_basesweep_on_high` | `mlp` (+2D pos) | 49 | 0.3429 |
| `mmbr_basesweep_off_high` | `mlp` (no 2D pos) | 49 | 0.3368 |

## Takeaway
- The learned constant prefix is ahead of the best image-conditioned bridge by about `+0.0111` absolute accuracy on the logged full-val runs.
- The gap is persistent across nearby MLP bridge variants (with/without 2D positional embeddings).
