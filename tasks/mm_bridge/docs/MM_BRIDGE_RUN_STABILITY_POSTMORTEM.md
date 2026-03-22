# MM Bridge Run Stability Postmortem

Purpose:
- keep one simple record of the major run-stability failures during this bridge phase
- document the actual root causes, not the first guesses
- record what was fixed, what was only mitigated, and what still remains a caveat

This note covers three clusters:
- wall-clock timeout kills
- eval-path regressions
- host OOM kills during long eval or after periodic eval

## 1. Timeout Problem

### What happened

Live training and final eval runs were being killed by launcher logic, not by Docker and not by the trainer.

The key mistake was a wrapper like:

```bash
timeout --signal=INT --kill-after=30s "${TIMEOUT_SEC}s" "${CMD[@]}"
```

Important clarification:
- the `30s` was only the grace period after the real timeout fired
- it was not an inactivity timeout
- log frequency had nothing to do with whether the run died

So runs were dying because the overall wall-clock limit expired, even while they were actively making progress.

### What was not the cause

- not `docker run`
- not `runmm.sh`
- not missing logs for `30s`

`runmm.sh` uses plain `docker run --rm ...` and has no timeout flag.

### Fix

Time-based kill logic was removed from the bridge sweep/launcher scripts.

Current intended behavior:
- no live MM run should die because a launcher-side horizon expired
- restart behavior is checkpoint-driven, not time-driven

### Main lesson

For ML runs, wall-clock timeouts in launchers are the wrong default unless the user explicitly wants them.

## 2. Eval Regression Problem

There were two different eval regressions.

### 2A. `safeqcond` regression

Observed symptom:
- same checkpoint
- historical eval around `0.454`
- new eval collapsed to about `0.086`

Actual cause:
- eval-time semantics of `prompt_only` q-conditioning were changed in place
- historical behavior fed the growing decoded sequence back into the q-conditioned visual-prefix path
- the refactor changed `prompt_only` to mean literal prompt-only during eval

That semantic change broke compatibility with existing checkpoints and historical comparisons.

### Fix

Legacy `prompt_only` eval behavior was restored for q-conditioned checkpoints so historical results are reproduced again.

### Main lesson

Even if old behavior is conceptually messy, changing semantics in place is still a regression if it changes outputs for existing checkpoints.

### 2B. Non-qcond KV-cache regression

Observed symptom:
- non-qcond checkpoints like `structuredroles_frontier` collapsed under `--eval_use_kv_cache`
- example real A/B:
  - no KV: about `0.471`
  - KV: about `0.070`

Actual causes:
- first pass: mixed-length batched incremental decode was wrong
- deeper real-model cause: incremental LM block was missing the same `cap_vector_norm` operations used in normal forward

That missing logic made real checkpoints diverge badly even when toy checks looked okay.

### Fix

The LM incremental path was corrected to match the normal block behavior.

Then the cache path was made conservative:
- first generated token uses the original full-batch decode path
- continuation switches to per-sample cached decoding

That restored correctness on the real checked runs.

### Current status

Correctness:
- restored for the tested real checkpoints

Performance:
- disappointing
- the conservative exact path is often slower than the old batched eval path

Practical consequence:
- the sweep script no longer opts into `--eval_use_kv_cache`
- the flag still exists in the trainer, but it is not part of the standard sweep path

### Main lesson

Bit-for-bit eval correctness matters more than theoretical speedup. A slower correct eval is acceptable as a debug path, not as the default sweep path.

## 3. OOM Problem

This was the third major stability cluster.

### What happened

Runs died without Python tracebacks:
- some died during long full final eval
- later, one died in the middle of training after a periodic eval

The key signal came from host `dmesg`:
- the kernel OOM-killed the containerized Python process

So this was host RAM pressure, not a normal trainer exception and not a CUDA out-of-memory traceback.

### What was probably happening

There were multiple contributors.

#### A. Full final eval accumulates a large in-memory record list

Approximate footprint estimate:
- full final-eval prediction records: about `0.6 GB`

That alone is probably not enough to explain every OOM, but it is nontrivial.

#### B. Persistent DataLoader workers were stacking

This was the more important finding.

Before the worker-shutdown fixes:
- training used persistent `train_loader` workers
- periodic/full eval used persistent `val_loader` workers
- after eval, those `val_loader` workers could remain alive while training continued
- before final eval, the run could also still have resident `train_loader` workers

That means one process could carry:
- main Python process
- 4 train workers
- 4 val workers

That matches the observed OOM pattern much better than “visual-prefix GPU cache pile-up”.

#### C. CUDA cache clearing was not enough

We added periodic `torch.cuda.empty_cache()` every `400` batches during long full eval.

That did run.

But later failures still happened, which strongly suggests the main issue was host RAM / worker-process pressure, not unreleased CUDA allocator cache.

### Fixes made

#### Final eval worker cleanup

Before full final eval:
- shut down persistent `train_loader` workers

#### Periodic / eval-only / final eval worker cleanup

After eval phases:
- shut down persistent `val_loader` workers

This is the important structural fix:
- do not keep both train and val worker pools resident after an eval boundary

#### CUDA cache clear during long final eval

Still present as a small mitigation:
- clear CUDA cache every `400` eval batches during full eval

This is now treated as secondary, not the primary fix.

### Current best understanding

The OOM issue is mostly:
- host RAM pressure
- duplicated dataset / worker-process state
- long eval lifetime

It is not primarily:
- Docker timeout
- trainer exception
- pure GPU-memory fragmentation

### Main lesson

In this project, persistent DataLoader workers are the main hidden memory risk during long validation/eval phases.

## 4. Related Resume Caveat

Resume behavior is restart-safe for checkpoints, but not exact for dataset position.

Current MM checkpoint resume restores:
- model state
- optimizer state
- `global_step`
- `epoch`

It does not restore:
- sampler order
- DataLoader iterator state
- RNG state for exact within-epoch continuation

So a resumed run does not continue from the exact same train batch it would have seen without interruption.

This is a reproducibility caveat, not the cause of the timeout/eval/OOM failures above, but it is worth remembering.

## 5. Current Stable Read

What is actually fixed:
- launcher-side wall-clock timeout kill behavior
- `safeqcond` eval semantic regression
- catastrophic non-qcond KV-cache eval regression
- carrying val workers after periodic eval
- carrying train workers into full final eval

What is intentionally not standard right now:
- `--eval_use_kv_cache`

Why:
- correctness was recovered
- speed advantage was not good enough in the conservative exact path

What still remains a possible future improvement:
- stream full final-eval scoring/prediction output instead of retaining the entire eval record list in memory
- exact sampler/iterator-state checkpointing if exact resume semantics become important

## 6. Short Version

If a future run dies, the first questions should now be:

1. Was this an external host OOM kill?
2. Did both train and val worker pools stay resident?
3. Was a nonstandard eval path enabled?
4. Was the run resumed from checkpoint, implying non-exact dataset-position continuity?

That is a much better triage order than:
- blaming Docker
- blaming log cadence
- blaming CUDA cache first
