# Experiment DB

This task now has a simple DuckDB experiment index at `logs/experiments.duckdb`.

For this project, **experiment = sweep**. The DB is not the source of truth for training state. It is a lightweight index over the filesystem so agents can find experiment bundles, runs, and canonical logfile segments quickly.

The research tracker now reads experiment/run metadata from this DB directly. Raw artifacts still live under `logs/`.

## Current Build

- DB path: `logs/experiments.duckdb`
- Builder: `python3 scripts/build_experiment_db.py`
- Shared ingestion logic: `tracker/research/experimentindex.ts`
- Export bridge used by the builder: `tracker/research/experimentdbexport.ts`
- Bun app DB reader: `tracker/research/researchtrackerapp.ts`
- Inclusion rule: only experiments whose `max_last_step > 100`

Current materialized contents as of `2026-03-26`:

- `26` experiments
- `141` runs
- `157` canonical logfile segments
- task split:
  - `23` `mm_bridge`
  - `3` `datasetting`

## What Is Indexed

The DB only indexes tracker-style experiment bundles that already exist in `logs/` and have a `timeline.log`.

Grouping logic is intentionally the same logic used by the research tracker:

- experiment bundles are deduped across `_latest` symlinks and timestamped bundle dirs
- runs inside a bundle come from `START` lines in `timeline.log`
- repeated timestamp suffixes are normalized when merging the same run family
- run summaries come from the canonical logfile parser used by the tracker
- logfile segments are only `logfile.txt` and `logfile_from_<step>.txt`

## Tables

`metadata`

- one-row build metadata
- columns: `generated_at`, `repo_root`, `min_steps`

`experiments`

- one row per indexed experiment bundle
- columns:
  - `task_id`, `task_title`
  - `experiment_id`
  - `experiment_dir`
  - `timeline_path`
  - `status`
  - `started_at`, `ended_at`
  - `run_count`, `active_runs`
  - `best_accuracy`, `last_train_ce`
  - `max_last_step`, `min_last_step`

`runs`

- one row per run inside an indexed experiment
- columns:
  - `task_id`, `experiment_id`, `run_id`
  - `run_dir`, `run_stage`
  - `experiment_family`, `paired_run_id`
  - `final_accuracy`, `best_accuracy`, `last_train_ce`
  - `last_step`, `last_steps_per_sec`
  - `num_params`, `trainable_params`
  - `is_active`, `has_final_checkpoint`, `is_eval_only`
  - `logfile`, `logfile_path`, `logfile_mtime_ms`

`log_segments`

- one row per canonical logfile segment
- columns:
  - `task_id`, `experiment_id`, `run_id`
  - `segment_file`, `segment_path`
  - `kind`
  - `resume_step`
  - `mtime_ms`
  - `bytes`

## Views

`experiment_run_index`

- convenience join from experiments to runs

`run_log_index`

- convenience join from runs to logfile segments

## How To Rebuild

```bash
python3 scripts/build_experiment_db.py
```

Useful flags:

- `--db-path logs/experiments.duckdb`
- `--task mm_bridge`
- `--min-steps 100`

## Where New Data Should Go

Begin storing **derived experiment metadata** in this DB, not raw training logs.

Good fits:

- experiment-level notes keyed by `task_id + experiment_id`
- run-level annotations keyed by `task_id + experiment_id + run_id`
- artifact indexes keyed by the same ids plus a relative path
- analyst judgments, rankings, tags, or follow-up pointers

Bad fits:

- full logfile text blobs
- checkpoint payloads
- large eval outputs already stored as files under `logs/`

The filesystem remains canonical for raw artifacts:

- experiment bundle control data stays under `logs/<experiment_bundle>/timeline.log`
- run artifacts stay under `logs/<run_id>/`
- DuckDB should point to those files, not replace them

## Recommended Extension Pattern

If the research agent needs to start writing experiment data, start with small keyed tables like:

```sql
create table if not exists experiment_notes (
  task_id varchar,
  experiment_id varchar,
  note_type varchar,
  note_text varchar,
  created_at varchar
);

create table if not exists run_annotations (
  task_id varchar,
  experiment_id varchar,
  run_id varchar,
  label varchar,
  value varchar,
  created_at varchar
);
```

That keeps the current DB aligned with its real purpose: a searchable index over experiment bundles and their logfile-backed runs.

## Example Queries

Top-step experiments:

```sql
select experiment_id, run_count, max_last_step
from experiments
order by max_last_step desc, experiment_id;
```

Find the canonical logfiles for one run:

```sql
select segment_file, segment_path, resume_step
from log_segments
where run_id = 'mmcrane_v1_20260314_dinov2s_attnqquery_nodynbudget_adapter_d3'
order by resume_step;
```

Compare runs inside one experiment:

```sql
select run_id, final_accuracy, last_train_ce, last_step, logfile_path
from experiment_run_index
where experiment_id = 'mmcrane_v1'
order by final_accuracy desc nulls last;
```
