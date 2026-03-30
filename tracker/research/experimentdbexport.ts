import fs from "fs";
import path from "path";
import {
  getBootstrap,
  loadTasks,
  resolveTaskContext,
  type TaskContext,
} from "./experimentindex";
import { listRunLogSegments } from "./logstitch";

type Args = {
  taskId: string | null;
  tasksRootRel: string;
};

function usageAndExit(): never {
  console.error("Usage: bun run tracker/research/experimentdbexport.ts [--task <task_id>] [--tasks-root <dir>]");
  process.exit(1);
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const taskIdx = args.indexOf("--task");
  const tasksRootIdx = args.indexOf("--tasks-root");
  const taskId = taskIdx !== -1 ? String(args[taskIdx + 1] ?? "").trim() : "";
  const tasksRootRel = tasksRootIdx !== -1 ? String(args[tasksRootIdx + 1] ?? "").trim() : "tasks";
  if (!tasksRootRel) usageAndExit();
  return { taskId: taskId || null, tasksRootRel };
}

const { taskId, tasksRootRel } = parseArgs();
const repoRoot = path.resolve(import.meta.dir, "..", "..");
const tasksRoot = path.resolve(repoRoot, tasksRootRel);
const tasks = loadTasks(repoRoot, tasksRoot);
const tasksById = new Map(tasks.map((task) => [task.id, task] as const));

function statBytes(file: string): number {
  try {
    return fs.statSync(file).size;
  } catch {
    return 0;
  }
}

function collectTaskRows(task: TaskContext) {
  const bootstrap = getBootstrap(task, tasks, repoRoot);
  const experimentRows = bootstrap.experiments.map((experiment) => {
    const lastSteps = experiment.runs
      .map((run) => run.lastStep)
      .filter((value): value is number => Number.isFinite(value));
    return {
      taskId: task.id,
      taskTitle: task.title,
      experimentId: experiment.experimentId,
      experimentDir: path.relative(repoRoot, experiment.experimentDir),
      status: experiment.status,
      startedAt: experiment.startedAt,
      endedAt: experiment.endedAt,
      runCount: experiment.runs.length,
      activeRuns: experiment.activeRuns,
      bestAccuracy: experiment.bestAccuracy,
      lastTrainCe: experiment.lastTrainCe,
      maxLastStep: lastSteps.length > 0 ? Math.max(...lastSteps) : null,
      minLastStep: lastSteps.length > 0 ? Math.min(...lastSteps) : null,
      timelinePath: path.relative(repoRoot, path.join(experiment.experimentDir, "timeline.log")),
    };
  });

  const runRows = bootstrap.experiments.flatMap((experiment) =>
    experiment.runs.map((run) => ({
      taskId: task.id,
      experimentId: experiment.experimentId,
      runId: run.runId,
      runDir: path.relative(repoRoot, run.runDir),
      runStage: run.runStage,
      experimentFamily: run.experimentFamily,
      pairedRunId: run.pairedRunId,
      finalAccuracy: run.finalAccuracy,
      bestAccuracy: run.bestAccuracy,
      lastTrainCe: run.lastTrainCe,
      lastStep: run.lastStep,
      lastStepsPerSec: run.lastStepsPerSec,
      numParams: run.numParams,
      trainableParams: run.trainableParams,
      isActive: run.isActive,
      hasFinalCheckpoint: run.hasFinalCheckpoint,
      isEvalOnly: run.isEvalOnly,
      logfile: run.logfile,
      logfilePath: run.logfile ? path.relative(repoRoot, path.join(run.runDir, run.logfile)) : null,
      logfileMtimeMs: run.logfileMtimeMs,
    }))
  );

  const logSegmentRows = bootstrap.experiments.flatMap((experiment) =>
    experiment.runs.flatMap((run) =>
      listRunLogSegments(run.runDir).map((segment) => ({
        taskId: task.id,
        experimentId: experiment.experimentId,
        runId: run.runId,
        segmentFile: segment.file,
        segmentPath: path.relative(repoRoot, segment.fullPath),
        kind: segment.kind,
        resumeStep: segment.resumeStep,
        mtimeMs: segment.mtimeMs,
        bytes: statBytes(segment.fullPath),
      }))
    )
  );

  return { experimentRows, runRows, logSegmentRows };
}

let selectedTasks = tasks;
if (taskId) {
  const selectedTask = resolveTaskContext(tasksById, taskId, null);
  if (!selectedTask) {
    console.error(`Unknown task: ${taskId}`);
    process.exit(1);
  }
  selectedTasks = [selectedTask];
}
const experimentRows = [];
const runRows = [];
const logSegmentRows = [];
for (const task of selectedTasks) {
  const rows = collectTaskRows(task);
  experimentRows.push(...rows.experimentRows);
  runRows.push(...rows.runRows);
  logSegmentRows.push(...rows.logSegmentRows);
}

process.stdout.write(
  JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      repoRoot,
      tasks: selectedTasks.map((task) => ({ id: task.id, title: task.title })),
      experiments: experimentRows,
      runs: runRows,
      logSegments: logSegmentRows,
    },
    null,
    2
  )
);
