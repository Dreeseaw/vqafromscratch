import fs from "fs";
import path from "path";
import { parseRunLog } from "./logstitch";

export type RunStage = "vm" | "mm" | "other";

export type RunSummary = {
  runId: string;
  runDir: string;
  finalAccuracy: number | null;
  bestAccuracy: number | null;
  lastTrainCe: number | null;
  lastStep: number | null;
  lastStepsPerSec: number | null;
  numParams: number | null;
  trainableParams: number | null;
  isActive: boolean;
  hasFinalCheckpoint: boolean;
  isEvalOnly: boolean;
  logfile: string | null;
  logfileMtimeMs: number | null;
  runStage: RunStage;
  experimentFamily: string | null;
  pairedRunId: string | null;
};

export type ExperimentSummary = {
  experimentId: string;
  experimentDir: string;
  status: "running" | "completed";
  startedAt: string | null;
  endedAt: string | null;
  runs: RunSummary[];
  bestAccuracy: number | null;
  lastTrainCe: number | null;
  activeRuns: number;
};

type ParsedExperimentSummary = ExperimentSummary & {
  isSymlink: boolean;
  mtimeMs: number;
};

export type DocSummary = {
  file: string;
  title: string;
  updatedAt: string;
  runRefs: string[];
  mentionedAccuracies: number[];
};

type TaskConfigFile = {
  id?: string;
  title?: string;
  description?: string;
  docsDir?: string;
  scriptsDir?: string;
  logsDir?: string;
  logPrefixes?: string[];
  excludeLogPrefixes?: string[];
  default?: boolean;
  qaPromptHint?: string;
};

export type TaskContext = {
  id: string;
  title: string;
  description: string | null;
  docsDir: string;
  scriptsDir: string;
  logsDir: string;
  docsRoot: string;
  scriptsRoot: string;
  logsRoot: string;
  logPrefixes: string[];
  excludeLogPrefixes: string[];
  isDefault: boolean;
  qaPromptHint: string | null;
  taskFile: string;
};

export type ExperimentBootstrap = {
  generatedAt: string;
  repoRoot: string;
  tasks: Array<{ id: string; title: string; description: string | null }>;
  selectedTask: {
    id: string;
    title: string;
    description: string | null;
    docsRoot: string;
    scriptsRoot: string;
    logsRoot: string;
  };
  docsRoot: string;
  logsRoot: string;
  scriptsRoot: string;
  docs: DocSummary[];
  experiments: ExperimentSummary[];
  runs: RunSummary[];
  summary: {
    docsCount: number;
    experimentsCount: number;
    runsCount: number;
    runsWithAccuracy: number;
    bestRun: { runId: string; finalAccuracy: number | null; bestAccuracy: number | null } | null;
  };
};

const ACTIVE_WINDOW_MS = 45 * 60 * 1000;
const TIMELINE_RUN_MARKERS = new Set(["caption-align", "two-stage"]);

function readText(file: string): string {
  try {
    return fs.readFileSync(file, "utf-8");
  } catch {
    return "";
  }
}

function uniq<T>(xs: T[]): T[] {
  return [...new Set(xs)];
}

export function toIso(ms: number): string {
  return new Date(ms).toISOString();
}

function assertRelativeDir(value: string, field: string, taskFile: string): string {
  if (!value || path.isAbsolute(value)) {
    throw new Error(`${taskFile}: ${field} must be a non-empty repo-relative path`);
  }
  return value.replace(/\\/g, "/");
}

function parseTaskLogPrefixes(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return uniq(
    value
      .map((entry) => String(entry ?? "").trim())
      .filter((entry) => /^[A-Za-z0-9._-]+$/.test(entry))
  );
}

export function loadTasks(repoRoot: string, tasksRoot: string): TaskContext[] {
  if (!fs.existsSync(tasksRoot) || !fs.statSync(tasksRoot).isDirectory()) {
    throw new Error(`Missing tasks root: ${path.relative(repoRoot, tasksRoot)}`);
  }
  const taskDirs = fs
    .readdirSync(tasksRoot)
    .map((name) => path.join(tasksRoot, name))
    .filter((full) => fs.existsSync(full) && fs.statSync(full).isDirectory())
    .sort();
  const tasks: TaskContext[] = [];
  for (const taskDir of taskDirs) {
    const taskFile = path.join(taskDir, "task.json");
    if (!fs.existsSync(taskFile) || !fs.statSync(taskFile).isFile()) continue;
    const raw = JSON.parse(readText(taskFile)) as TaskConfigFile;
    const id = String(raw.id ?? "").trim();
    const title = String(raw.title ?? "").trim();
    const docsDir = assertRelativeDir(String(raw.docsDir ?? "").trim(), "docsDir", taskFile);
    const scriptsDir = assertRelativeDir(String(raw.scriptsDir ?? "").trim(), "scriptsDir", taskFile);
    const logsDir = assertRelativeDir(String(raw.logsDir ?? "").trim(), "logsDir", taskFile);
    const logPrefixes = parseTaskLogPrefixes(raw.logPrefixes);
    const excludeLogPrefixes = parseTaskLogPrefixes(raw.excludeLogPrefixes);
    if (!id || !title) throw new Error(`${taskFile}: missing id/title`);
    tasks.push({
      id,
      title,
      description: String(raw.description ?? "").trim() || null,
      docsDir,
      scriptsDir,
      logsDir,
      docsRoot: path.resolve(repoRoot, docsDir),
      scriptsRoot: path.resolve(repoRoot, scriptsDir),
      logsRoot: path.resolve(repoRoot, logsDir),
      logPrefixes,
      excludeLogPrefixes,
      isDefault: Boolean(raw.default),
      qaPromptHint: String(raw.qaPromptHint ?? "").trim() || null,
      taskFile,
    });
  }
  if (tasks.length === 0) {
    throw new Error(`No task.json files found under ${path.relative(repoRoot, tasksRoot)}`);
  }
  return tasks;
}

export function resolveTaskContext(
  tasksById: Map<string, TaskContext>,
  requestedTaskId: string | null | undefined,
  fallbackTaskId: string | null | undefined
): TaskContext | null {
  const requested = String(requestedTaskId ?? fallbackTaskId ?? "").trim();
  return tasksById.get(requested) ?? null;
}

export function taskIncludesLogName(task: TaskContext, name: string): boolean {
  if (task.excludeLogPrefixes.some((prefix) => name.startsWith(prefix))) return false;
  return task.logPrefixes.length === 0 || task.logPrefixes.some((prefix) => name.startsWith(prefix));
}

function detectRunStage(runId: string): RunStage {
  if (runId.startsWith("vm_")) return "vm";
  if (runId.startsWith("mm_")) return "mm";
  return "other";
}

function getExperimentFamily(runId: string): string | null {
  const match = runId.match(/^(?:vm|mm)_(.+)$/);
  return match ? match[1] : null;
}

function getPairedRunId(task: TaskContext, runId: string): string | null {
  const family = getExperimentFamily(runId);
  const stage = detectRunStage(runId);
  if (!family || stage === "other") return null;
  const counterpart = `${stage === "vm" ? "mm" : "vm"}_${family}`;
  const full = path.join(task.logsRoot, counterpart);
  return fs.existsSync(full) && fs.statSync(full).isDirectory() ? counterpart : null;
}

export function parseRunSummary(task: TaskContext, runId: string): RunSummary {
  const runStage = detectRunStage(runId);
  const experimentFamily = getExperimentFamily(runId);
  const pairedRunId = getPairedRunId(task, runId);
  const runDir = path.join(task.logsRoot, runId);
  const out: RunSummary = {
    runId,
    runDir,
    finalAccuracy: null,
    bestAccuracy: null,
    lastTrainCe: null,
    lastStep: null,
    lastStepsPerSec: null,
    numParams: null,
    trainableParams: null,
    isActive: false,
    hasFinalCheckpoint: false,
    isEvalOnly: false,
    logfile: null,
    logfileMtimeMs: null,
    runStage,
    experimentFamily,
    pairedRunId,
  };
  if (!taskIncludesLogName(task, runId)) return out;
  if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) return out;
  const parsed = parseRunLog(runDir);
  out.logfile = parsed.logfile;
  out.logfileMtimeMs = parsed.logfileMtimeMs;
  out.finalAccuracy = parsed.finalAccuracy;
  out.bestAccuracy = parsed.bestAccuracy;
  out.lastTrainCe = parsed.lastTrainCe;
  out.lastStep = parsed.lastStep;
  out.lastStepsPerSec = parsed.lastStepsPerSec;
  out.numParams = parsed.numParams;
  out.trainableParams = parsed.trainableParams;
  out.hasFinalCheckpoint = parsed.hasFinalCheckpoint;
  out.isEvalOnly = parsed.isEvalOnly;
  return out;
}

function parseTimelineLineDate(line: string): string | null {
  const match = line.match(/^\[([^\]]+)\]/);
  return match ? match[1] : null;
}

function parseTimelineRunId(task: TaskContext, line: string, event: "START" | "END" | "FAIL" | "SKIP" | "RESTART"): string | null {
  const match = line.match(new RegExp(`\\b${event}\\b\\s+(.+)$`));
  if (!match) return null;
  const tokens = (match[1].match(/[A-Za-z0-9._-]+/g) ?? []).filter((token) => !TIMELINE_RUN_MARKERS.has(token));
  if (tokens.length === 0) return null;
  const existingRunDirs = tokens.filter((token) => {
    const full = path.join(task.logsRoot, token);
    return fs.existsSync(full) && fs.statSync(full).isDirectory();
  });
  if (existingRunDirs.length > 0) return existingRunDirs.at(-1) ?? null;
  return tokens[0] ?? null;
}

function normalizeExperimentId(experimentId: string): string {
  return experimentId.replace(/_latest$/, "").replace(/_\d{8}_\d{6}$/, "");
}

export function normalizeRunId(runId: string): string {
  return runId.replace(/_\d{8}_\d{6}/g, "");
}

function pickPreferredRun(a: RunSummary, b: RunSummary): RunSummary {
  if ((b.lastStep ?? -1) !== (a.lastStep ?? -1)) return (b.lastStep ?? -1) > (a.lastStep ?? -1) ? b : a;
  if ((b.hasFinalCheckpoint ? 1 : 0) !== (a.hasFinalCheckpoint ? 1 : 0)) return b.hasFinalCheckpoint ? b : a;
  if ((b.finalAccuracy ?? -1) !== (a.finalAccuracy ?? -1)) return (b.finalAccuracy ?? -1) > (a.finalAccuracy ?? -1) ? b : a;
  if ((b.lastTrainCe ?? Number.POSITIVE_INFINITY) !== (a.lastTrainCe ?? Number.POSITIVE_INFINITY)) {
    return (b.lastTrainCe ?? Number.POSITIVE_INFINITY) < (a.lastTrainCe ?? Number.POSITIVE_INFINITY) ? b : a;
  }
  return b.runId > a.runId ? b : a;
}

function mergeRunState(base: RunSummary, other: RunSummary): RunSummary {
  const preferred = pickPreferredRun(base, other);
  return {
    ...preferred,
    isActive: base.isActive || other.isActive,
    isEvalOnly: preferred.isEvalOnly || base.isEvalOnly || other.isEvalOnly,
    trainableParams: preferred.trainableParams ?? base.trainableParams ?? other.trainableParams,
    numParams: preferred.numParams ?? base.numParams ?? other.numParams,
    runStage: preferred.runStage !== "other" ? preferred.runStage : base.runStage !== "other" ? base.runStage : other.runStage,
    experimentFamily: preferred.experimentFamily ?? base.experimentFamily ?? other.experimentFamily,
    pairedRunId: preferred.pairedRunId ?? base.pairedRunId ?? other.pairedRunId,
  };
}

export function shouldIncludeRun(run: RunSummary): boolean {
  return run.lastStep !== null || run.finalAccuracy !== null || run.bestAccuracy !== null || run.isEvalOnly;
}

function parseExperiment(task: TaskContext, experimentDirName: string, isSymlink: boolean, mtimeMs: number): ParsedExperimentSummary | null {
  const experimentDir = path.join(task.logsRoot, experimentDirName);
  const timeline = path.join(experimentDir, "timeline.log");
  if (!fs.existsSync(timeline) || !fs.statSync(experimentDir).isDirectory()) return null;
  const lines = readText(timeline).split("\n").filter((line) => line.trim().length > 0);
  const runIds = uniq(
    lines
      .map((line) => parseTimelineRunId(task, line, "START"))
      .filter((value): value is string => Boolean(value))
  );

  const activeRunIds = new Set<string>();
  for (const line of lines) {
    const startedRunId = parseTimelineRunId(task, line, "START");
    if (startedRunId) {
      activeRunIds.add(startedRunId);
      continue;
    }
    const terminalRunId =
      parseTimelineRunId(task, line, "END") ??
      parseTimelineRunId(task, line, "FAIL") ??
      parseTimelineRunId(task, line, "SKIP") ??
      (line.match(/\bSTOP\b.*\bbefore\s+([A-Za-z0-9._-]+)/)?.[1] ?? null);
    if (terminalRunId) activeRunIds.delete(terminalRunId);
  }
  const completionLine = [...lines].reverse().find((line) => /\b(?:SWEEP|PROBES)\s+COMPLETE\b/.test(line)) ?? null;
  const terminalLine =
    completionLine ?? [...lines].reverse().find((line) => /\b(?:END|FAIL|SKIP|STOP)\b/.test(line)) ?? null;
  const parsedRuns = runIds.map((runId) => {
    const run = parseRunSummary(task, runId);
    run.isActive =
      activeRunIds.has(runId) &&
      !completionLine &&
      Number.isFinite(run.logfileMtimeMs ?? NaN) &&
      Date.now() - (run.logfileMtimeMs as number) <= ACTIVE_WINDOW_MS;
    return run;
  });
  const activeRuns = parsedRuns.filter((run) => run.isActive).length;
  const runs = parsedRuns.filter(shouldIncludeRun);
  const allAcc = runs.map((run) => run.finalAccuracy).filter((value): value is number => Number.isFinite(value));
  const ceVals = runs.map((run) => run.lastTrainCe).filter((value): value is number => Number.isFinite(value));
  const startLine = lines.find((line) => /\bSTART\b/.test(line)) ?? lines[0] ?? "";

  return {
    experimentId: experimentDirName,
    experimentDir,
    status: activeRuns > 0 ? "running" : "completed",
    startedAt: parseTimelineLineDate(startLine),
    endedAt: activeRuns > 0 ? null : parseTimelineLineDate(terminalLine ?? ""),
    runs,
    bestAccuracy: allAcc.length > 0 ? Math.max(...allAcc) : null,
    lastTrainCe: ceVals.length > 0 ? Math.min(...ceVals) : null,
    activeRuns,
    isSymlink,
    mtimeMs,
  };
}

export function listExperiments(task: TaskContext): ExperimentSummary[] {
  if (!fs.existsSync(task.logsRoot) || !fs.statSync(task.logsRoot).isDirectory()) return [];
  const dirs = fs
    .readdirSync(task.logsRoot)
    .map((name) => {
      if (!taskIncludesLogName(task, name)) return null;
      const full = path.join(task.logsRoot, name);
      if (!fs.existsSync(full) || !fs.statSync(full).isDirectory() || !fs.existsSync(path.join(full, "timeline.log"))) {
        return null;
      }
      const lst = fs.lstatSync(full);
      return { name, isSymlink: lst.isSymbolicLink(), mtimeMs: fs.statSync(full).mtimeMs };
    })
    .filter((entry): entry is { name: string; isSymlink: boolean; mtimeMs: number } => entry !== null)
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
  const experiments = dirs
    .map((entry) => parseExperiment(task, entry.name, entry.isSymlink, entry.mtimeMs))
    .filter((entry): entry is ParsedExperimentSummary => entry !== null);
  const deduped = new Map<string, ParsedExperimentSummary>();
  for (const experiment of experiments) {
    const key = experiment.startedAt ? `started:${experiment.startedAt}` : `id:${experiment.experimentId}`;
    const prev = deduped.get(key);
    if (!prev) {
      deduped.set(key, experiment);
      continue;
    }
    const preferExperiment =
      Number(experiment.isSymlink) < Number(prev.isSymlink) ||
      (experiment.isSymlink === prev.isSymlink && experiment.runs.length > prev.runs.length) ||
      (experiment.isSymlink === prev.isSymlink &&
        experiment.runs.length === prev.runs.length &&
        experiment.mtimeMs > prev.mtimeMs);
    if (preferExperiment) deduped.set(key, experiment);
  }
  const grouped = new Map<string, ParsedExperimentSummary[]>();
  for (const experiment of deduped.values()) {
    const key = normalizeExperimentId(experiment.experimentId);
    const group = grouped.get(key);
    if (group) group.push(experiment);
    else grouped.set(key, [experiment]);
  }

  return [...grouped.entries()]
    .map(([experimentId, group]) => {
      const runMap = new Map<string, RunSummary>();
      for (const experiment of group) {
        for (const run of experiment.runs) {
          const runKey = normalizeRunId(run.runId);
          const prev = runMap.get(runKey);
          runMap.set(runKey, prev ? mergeRunState(prev, run) : run);
        }
      }
      const runs = [...runMap.values()];
      const allAcc = runs.map((run) => run.finalAccuracy).filter((value): value is number => Number.isFinite(value));
      const ceVals = runs.map((run) => run.lastTrainCe).filter((value): value is number => Number.isFinite(value));
      const representative = group.slice().sort((a, b) => b.runs.length - a.runs.length || b.mtimeMs - a.mtimeMs)[0];
      const startedAt = group
        .map((experiment) => experiment.startedAt)
        .filter((value): value is string => Boolean(value))
        .sort()[0] ?? null;
      const endedAt = group
        .map((experiment) => experiment.endedAt)
        .filter((value): value is string => Boolean(value))
        .sort()
        .at(-1) ?? null;
      return {
        experimentId,
        experimentDir: representative.experimentDir,
        status: group.some((experiment) => experiment.activeRuns > 0) ? "running" : "completed",
        startedAt,
        endedAt,
        runs,
        bestAccuracy: allAcc.length > 0 ? Math.max(...allAcc) : null,
        lastTrainCe: ceVals.length > 0 ? Math.min(...ceVals) : null,
        activeRuns: runs.filter((run) => run.isActive).length,
        mtimeMs: Math.max(...group.map((experiment) => experiment.mtimeMs)),
      };
    })
    .filter((experiment) => {
      if (experiment.runs.length > 1 || experiment.activeRuns > 0) return true;
      return experiment.runs.some(
        (run) =>
          (typeof run.lastStep === "number" && run.lastStep > 100) ||
          typeof run.finalAccuracy === "number" ||
          typeof run.bestAccuracy === "number"
      );
    })
    .sort((a, b) => b.mtimeMs - a.mtimeMs)
    .map(({ mtimeMs: _mtimeMs, ...experiment }) => experiment);
}

function parseDoc(task: TaskContext, fileName: string): DocSummary {
  const full = path.join(task.docsRoot, fileName);
  const txt = readText(full);
  const titleMatch = txt.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : fileName;
  const logsRefs = [...txt.matchAll(/logs\/([A-Za-z0-9._-]+)/g)].map((match) => match[1]);
  const inlineRefs = [...txt.matchAll(/`([A-Za-z0-9._-]{6,})`/g)]
    .map((match) => match[1])
    .filter((runId) => fs.existsSync(path.join(task.logsRoot, runId)));
  const runRefs = uniq([...logsRefs, ...inlineRefs]);
  const mentionedAccuracies = uniq(
    [...txt.matchAll(/\b0\.\d{3,4}\b/g)]
      .map((match) => Number(match[0]))
      .filter((value) => Number.isFinite(value) && value >= 0.1)
  ).sort((a, b) => b - a);
  return {
    file: fileName,
    title,
    updatedAt: toIso(fs.statSync(full).mtimeMs),
    runRefs,
    mentionedAccuracies,
  };
}

export function listDocs(task: TaskContext): DocSummary[] {
  if (!fs.existsSync(task.docsRoot) || !fs.statSync(task.docsRoot).isDirectory()) return [];
  return fs
    .readdirSync(task.docsRoot)
    .filter((file) => file.toLowerCase().endsWith(".md"))
    .sort((a, b) => fs.statSync(path.join(task.docsRoot, b)).mtimeMs - fs.statSync(path.join(task.docsRoot, a)).mtimeMs)
    .map((file) => parseDoc(task, file));
}

export function listStandaloneRuns(task: TaskContext): RunSummary[] {
  if (task.logPrefixes.length === 0) return [];
  if (!fs.existsSync(task.logsRoot) || !fs.statSync(task.logsRoot).isDirectory()) return [];
  return fs
    .readdirSync(task.logsRoot)
    .filter((name) => taskIncludesLogName(task, name))
    .filter((name) => {
      const full = path.join(task.logsRoot, name);
      return fs.existsSync(full) && fs.statSync(full).isDirectory();
    })
    .map((runId) => parseRunSummary(task, runId))
    .filter(shouldIncludeRun);
}

export function getBootstrap(task: TaskContext, allTasks: TaskContext[], repoRoot: string): ExperimentBootstrap {
  const experiments = listExperiments(task);
  const docs = listDocs(task);
  const runMap = new Map<string, RunSummary>();
  for (const experiment of experiments) {
    for (const run of experiment.runs) runMap.set(run.runId, run);
  }
  for (const run of listStandaloneRuns(task)) {
    if (!runMap.has(run.runId)) runMap.set(run.runId, run);
  }
  for (const doc of docs) {
    for (const runId of doc.runRefs) {
      if (!taskIncludesLogName(task, runId)) continue;
      if (runMap.has(runId)) continue;
      const run = parseRunSummary(task, runId);
      if (shouldIncludeRun(run)) runMap.set(runId, run);
    }
  }
  const allRuns = [...runMap.values()];
  const accRuns = allRuns.filter((run) => Number.isFinite(run.finalAccuracy ?? NaN));
  const bestRun = accRuns.slice().sort((a, b) => (b.finalAccuracy ?? -1) - (a.finalAccuracy ?? -1))[0] ?? null;

  return {
    generatedAt: new Date().toISOString(),
    repoRoot,
    tasks: allTasks.map((entry) => ({ id: entry.id, title: entry.title, description: entry.description })),
    selectedTask: {
      id: task.id,
      title: task.title,
      description: task.description,
      docsRoot: task.docsDir,
      scriptsRoot: task.scriptsDir,
      logsRoot: task.logsDir,
    },
    docsRoot: task.docsDir,
    logsRoot: task.logsDir,
    scriptsRoot: task.scriptsDir,
    docs,
    experiments,
    runs: allRuns,
    summary: {
      docsCount: docs.length,
      experimentsCount: experiments.length,
      runsCount: allRuns.length,
      runsWithAccuracy: accRuns.length,
      bestRun: bestRun ? { runId: bestRun.runId, finalAccuracy: bestRun.finalAccuracy, bestAccuracy: bestRun.bestAccuracy } : null,
    },
  };
}
