#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a simple DuckDB experiment index from tracker ingestion logic.")
    ap.add_argument("--db-path", default=str(ROOT / "logs" / "experiments.duckdb"))
    ap.add_argument("--task", default="", help="Optional task id to index. Defaults to all tasks.")
    ap.add_argument("--tasks-root", default="tasks")
    ap.add_argument("--min-steps", type=int, default=100, help="Include experiments whose max run step is greater than this threshold.")
    return ap.parse_args()


def load_export_payload(task: str, tasks_root: str) -> dict:
    cmd = ["bun", "run", "tracker/research/experimentdbexport.ts", "--tasks-root", tasks_root]
    if task:
        cmd.extend(["--task", task])
    proc = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        create or replace table metadata (
          generated_at varchar,
          repo_root varchar,
          min_steps integer
        )
        """
    )
    conn.execute(
        """
        create or replace table experiments (
          task_id varchar,
          task_title varchar,
          experiment_id varchar,
          experiment_dir varchar,
          timeline_path varchar,
          status varchar,
          started_at varchar,
          ended_at varchar,
          run_count integer,
          active_runs integer,
          best_accuracy double,
          last_train_ce double,
          max_last_step bigint,
          min_last_step bigint
        )
        """
    )
    conn.execute(
        """
        create or replace table runs (
          task_id varchar,
          experiment_id varchar,
          run_id varchar,
          run_dir varchar,
          run_stage varchar,
          experiment_family varchar,
          paired_run_id varchar,
          final_accuracy double,
          best_accuracy double,
          last_train_ce double,
          last_step bigint,
          last_steps_per_sec double,
          num_params bigint,
          trainable_params bigint,
          is_active boolean,
          has_final_checkpoint boolean,
          is_eval_only boolean,
          logfile varchar,
          logfile_path varchar,
          logfile_mtime_ms double
        )
        """
    )
    conn.execute(
        """
        create or replace table log_segments (
          task_id varchar,
          experiment_id varchar,
          run_id varchar,
          segment_file varchar,
          segment_path varchar,
          kind varchar,
          resume_step bigint,
          mtime_ms double,
          bytes bigint
        )
        """
    )
    conn.execute(
        """
        create or replace view experiment_run_index as
        select
          e.task_id,
          e.task_title,
          e.experiment_id,
          e.status as experiment_status,
          e.started_at,
          e.ended_at,
          e.max_last_step,
          r.run_id,
          r.run_stage,
          r.final_accuracy,
          r.best_accuracy,
          r.last_train_ce,
          r.last_step,
          r.logfile_path
        from experiments e
        join runs r using (task_id, experiment_id)
        """
    )
    conn.execute(
        """
        create or replace view run_log_index as
        select
          r.task_id,
          r.experiment_id,
          r.run_id,
          r.run_dir,
          r.last_step,
          r.logfile_path,
          s.segment_file,
          s.segment_path,
          s.kind,
          s.resume_step,
          s.bytes
        from runs r
        left join log_segments s using (task_id, experiment_id, run_id)
        """
    )


def main() -> None:
    args = parse_args()
    payload = load_export_payload(args.task.strip(), args.tasks_root)
    included_experiments = []
    for row in payload["experiments"]:
        max_last_step = row.get("maxLastStep")
        active_runs = row.get("activeRuns") or 0
        include_for_steps = isinstance(max_last_step, (int, float)) and max_last_step > args.min_steps
        include_for_activity = active_runs > 0
        if include_for_steps or include_for_activity:
            included_experiments.append(row)
    included_keys = {(row["taskId"], row["experimentId"]) for row in included_experiments}
    included_runs = [row for row in payload["runs"] if (row["taskId"], row["experimentId"]) in included_keys]
    included_segments = [row for row in payload["logSegments"] if (row["taskId"], row["experimentId"]) in included_keys]

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)

    conn.execute("delete from metadata")
    conn.execute("delete from experiments")
    conn.execute("delete from runs")
    conn.execute("delete from log_segments")

    conn.executemany(
        "insert into metadata values (?, ?, ?)",
        [(payload["generatedAt"], payload["repoRoot"], args.min_steps)],
    )
    if included_experiments:
        conn.executemany(
            "insert into experiments values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["taskId"],
                    row["taskTitle"],
                    row["experimentId"],
                    row["experimentDir"],
                    row["timelinePath"],
                    row["status"],
                    row["startedAt"],
                    row["endedAt"],
                    row["runCount"],
                    row["activeRuns"],
                    row["bestAccuracy"],
                    row["lastTrainCe"],
                    row["maxLastStep"],
                    row["minLastStep"],
                )
                for row in included_experiments
            ],
        )
    if included_runs:
        conn.executemany(
            "insert into runs values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["taskId"],
                    row["experimentId"],
                    row["runId"],
                    row["runDir"],
                    row["runStage"],
                    row["experimentFamily"],
                    row["pairedRunId"],
                    row["finalAccuracy"],
                    row["bestAccuracy"],
                    row["lastTrainCe"],
                    row["lastStep"],
                    row["lastStepsPerSec"],
                    row["numParams"],
                    row["trainableParams"],
                    row["isActive"],
                    row["hasFinalCheckpoint"],
                    row["isEvalOnly"],
                    row["logfile"],
                    row["logfilePath"],
                    row["logfileMtimeMs"],
                )
                for row in included_runs
            ],
        )
    if included_segments:
        conn.executemany(
            "insert into log_segments values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["taskId"],
                    row["experimentId"],
                    row["runId"],
                    row["segmentFile"],
                    row["segmentPath"],
                    row["kind"],
                    row["resumeStep"],
                    row["mtimeMs"],
                    row["bytes"],
                )
                for row in included_segments
            ],
        )

    summary = conn.execute(
        """
        select
          (select count(*) from experiments) as experiments,
          (select count(*) from runs) as runs,
          (select count(*) from log_segments) as log_segments
        """
    ).fetchone()
    conn.close()

    print(f"wrote: {db_path}")
    print(f"experiments: {summary[0]}")
    print(f"runs: {summary[1]}")
    print(f"log_segments: {summary[2]}")


if __name__ == "__main__":
    main()
