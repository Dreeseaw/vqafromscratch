import { serve } from "bun";
import fs from "fs";
import os from "os";
import path from "path";
import { parseRunLog, type SeriesPoint } from "./logstitch";

type RunSummary = {
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
  logfile: string | null;
  logfileMtimeMs: number | null;
};

type SweepSummary = {
  sweepId: string;
  sweepDir: string;
  status: "running" | "completed";
  startedAt: string | null;
  endedAt: string | null;
  runs: RunSummary[];
  bestAccuracy: number | null;
  lastTrainCe: number | null;
  activeRuns: number;
};

type ParsedSweepSummary = SweepSummary & {
  isSymlink: boolean;
  mtimeMs: number;
};

type DocSummary = {
  file: string;
  title: string;
  updatedAt: string;
  runRefs: string[];
  mentionedAccuracies: number[];
};

type RunDetail = {
  run: RunSummary;
  updatedAt: string | null;
  introLines: string[];
  tailLines: string[];
  trainCeSeries: SeriesPoint[];
  valAccuracySeries: SeriesPoint[];
  valStepsPerSecSeries: SeriesPoint[];
  latestEvalAccuracy: number | null;
  latestValStepsPerSec: number | null;
};

type TaskConfigFile = {
  id?: string;
  title?: string;
  description?: string;
  docsDir?: string;
  scriptsDir?: string;
  logsDir?: string;
  default?: boolean;
  qaPromptHint?: string;
};

type TaskContext = {
  id: string;
  title: string;
  description: string | null;
  docsDir: string;
  scriptsDir: string;
  logsDir: string;
  docsRoot: string;
  scriptsRoot: string;
  logsRoot: string;
  isDefault: boolean;
  qaPromptHint: string | null;
  taskFile: string;
};

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

type TaskQaResponse = {
  taskId: string;
  answer: string;
  evidence: string[];
  generatedAt: string;
};

const ACTIVE_WINDOW_MS = 45 * 60 * 1000;
const QA_TIMEOUT_MS = 2 * 60 * 1000;
const QA_MAX_TURNS = 8;
const userName = process.env.USER || process.env.LOGNAME || "";
const preferredUserHome =
  (userName && fs.existsSync(path.join("/home", userName)) && path.join("/home", userName)) ||
  process.env.HOME ||
  os.homedir();

function usageAndExit(): never {
  console.error("Usage: bun run tracker/research/researchtrackerapp.ts [-p <port>] [--task <task_id>] [--tasks-root <dir>]");
  process.exit(1);
}

function toIso(ms: number): string {
  return new Date(ms).toISOString();
}

function uniq<T>(xs: T[]): T[] {
  return [...new Set(xs)];
}

function trimTo(value: string, max: number): string {
  const text = String(value ?? "");
  return text.length <= max ? text : `${text.slice(0, max - 1)}...`;
}

function fmtNum(value: number | null): string {
  return Number.isFinite(value ?? NaN) ? String(value) : "-";
}

function fmtAcc(value: number | null): string {
  return Number.isFinite(value ?? NaN) ? (value as number).toFixed(4) : "-";
}

function parseArgs() {
  const args = process.argv.slice(2);
  const portIdx = args.indexOf("-p");
  const taskIdx = args.indexOf("--task");
  const tasksRootIdx = args.indexOf("--tasks-root");
  const port = portIdx !== -1 ? Number(args[portIdx + 1]) : 4090;
  if (!Number.isInteger(port) || port <= 0) usageAndExit();
  const taskId = taskIdx !== -1 ? String(args[taskIdx + 1] ?? "").trim() : "";
  const tasksRootRel = tasksRootIdx !== -1 ? String(args[tasksRootIdx + 1] ?? "").trim() : "tasks";
  if (!tasksRootRel) usageAndExit();
  return { port, taskId: taskId || null, tasksRootRel };
}

const { port, taskId: cliTaskId, tasksRootRel } = parseArgs();
const staticRoot = import.meta.dir;
const repoRoot = path.resolve(staticRoot, "..", "..");
const tasksRoot = path.resolve(repoRoot, tasksRootRel);
const htmlPath = path.join(staticRoot, "index.html");
const cssPath = path.join(staticRoot, "styles.css");
const appTsPath = path.join(staticRoot, "app.ts");
const docHtmlPath = path.join(staticRoot, "doc.html");
const docTsPath = path.join(staticRoot, "doc.ts");

if (
  !fs.existsSync(htmlPath) ||
  !fs.existsSync(cssPath) ||
  !fs.existsSync(appTsPath) ||
  !fs.existsSync(docHtmlPath) ||
  !fs.existsSync(docTsPath)
) {
  console.error("Missing frontend files in tracker/research.");
  process.exit(1);
}

function readText(file: string): string {
  try {
    return fs.readFileSync(file, "utf-8");
  } catch {
    return "";
  }
}

function findExecutable(name: string): string | null {
  const candidates = (process.env.PATH ?? "")
    .split(path.delimiter)
    .filter(Boolean);
  const homes = uniq(
    [
      process.env.HOME ?? "",
      os.homedir(),
      os.userInfo().homedir,
      preferredUserHome,
    ].filter(Boolean)
  );
  for (const home of homes) candidates.unshift(path.join(home, ".npm-global", "bin"));
  for (const dir of uniq(candidates)) {
    const full = path.join(dir, name);
    if (fs.existsSync(full) && fs.statSync(full).isFile()) return full;
  }
  return null;
}

function assertRelativeDir(value: string, field: string, taskFile: string): string {
  if (!value || path.isAbsolute(value)) {
    throw new Error(`${taskFile}: ${field} must be a non-empty repo-relative path`);
  }
  return value.replace(/\\/g, "/");
}

function loadTasks(): TaskContext[] {
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
    if (!id || !title) throw new Error(`${taskFile}: missing id/title`);
    const docsRoot = path.resolve(repoRoot, docsDir);
    const scriptsRoot = path.resolve(repoRoot, scriptsDir);
    const logsRoot = path.resolve(repoRoot, logsDir);
    tasks.push({
      id,
      title,
      description: String(raw.description ?? "").trim() || null,
      docsDir,
      scriptsDir,
      logsDir,
      docsRoot,
      scriptsRoot,
      logsRoot,
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

const allTasks = loadTasks();
const tasksById = new Map(allTasks.map((task) => [task.id, task] as const));
const defaultTask = (cliTaskId && tasksById.get(cliTaskId)) || allTasks.find((task) => task.isDefault) || allTasks[0];
const codexExecutable = findExecutable("codex");
const codexPathEntries = uniq(
  [
    path.join(preferredUserHome, ".npm-global", "bin"),
    ...(process.env.PATH ?? "").split(path.delimiter).filter(Boolean),
  ].filter(Boolean)
);

function resolveTaskContext(taskId: string | null | undefined): TaskContext | null {
  const requested = String(taskId ?? cliTaskId ?? defaultTask.id).trim();
  return tasksById.get(requested) ?? null;
}

function parseRunSummary(task: TaskContext, runId: string): RunSummary {
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
    logfile: null,
    logfileMtimeMs: null,
  };
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
  return out;
}

function parseRunDetail(task: TaskContext, runId: string): RunDetail | null {
  const parsed = parseRunLog(path.join(task.logsRoot, runId));
  const run = parseRunSummary(task, runId);
  if (run.lastStep === null) return null;

  return {
    run,
    updatedAt: run.logfileMtimeMs ? toIso(run.logfileMtimeMs) : null,
    introLines: parsed.introLines,
    tailLines: parsed.tailLines,
    trainCeSeries: parsed.trainCeSeries,
    valAccuracySeries: parsed.valAccuracySeries,
    valStepsPerSecSeries: parsed.valStepsPerSecSeries,
    latestEvalAccuracy:
      parsed.valAccuracySeries.length > 0 ? parsed.valAccuracySeries[parsed.valAccuracySeries.length - 1].value : null,
    latestValStepsPerSec:
      parsed.valStepsPerSecSeries.length > 0
        ? parsed.valStepsPerSecSeries[parsed.valStepsPerSecSeries.length - 1].value
        : null,
  };
}

function parseTimelineLineDate(line: string): string | null {
  const match = line.match(/^\[([^\]]+)\]/);
  return match ? match[1] : null;
}

function normalizeSweepId(sweepId: string): string {
  return sweepId.replace(/_latest$/, "").replace(/_\d{8}_\d{6}$/, "");
}

function normalizeRunId(runId: string): string {
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
    trainableParams: preferred.trainableParams ?? base.trainableParams ?? other.trainableParams,
    numParams: preferred.numParams ?? base.numParams ?? other.numParams,
  };
}

function parseSweep(task: TaskContext, sweepDirName: string, isSymlink: boolean, mtimeMs: number): ParsedSweepSummary | null {
  const sweepDir = path.join(task.logsRoot, sweepDirName);
  const timeline = path.join(sweepDir, "timeline.log");
  if (!fs.existsSync(timeline) || !fs.statSync(sweepDir).isDirectory()) return null;
  const lines = readText(timeline).split("\n").filter((line) => line.trim().length > 0);
  const runIds = uniq(
    lines
      .map((line) => {
        const match = line.match(/\bSTART\s+([A-Za-z0-9._-]+)/);
        return match ? match[1] : null;
      })
      .filter((value): value is string => Boolean(value))
  );

  const activeRunIds = new Set<string>();
  for (const line of lines) {
    const startMatch = line.match(/\bSTART\s+([A-Za-z0-9._-]+)/);
    if (startMatch) {
      activeRunIds.add(startMatch[1]);
      continue;
    }
    const terminalMatch =
      line.match(/\b(?:END|FAIL|SKIP)\s+([A-Za-z0-9._-]+)/) ??
      line.match(/\bSTOP\b.*\bbefore\s+([A-Za-z0-9._-]+)/);
    if (terminalMatch) activeRunIds.delete(terminalMatch[1]);
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
  const runs = parsedRuns.filter((run) => run.lastStep !== null);
  const allAcc = runs.map((run) => run.bestAccuracy).filter((value): value is number => Number.isFinite(value));
  const ceVals = runs.map((run) => run.lastTrainCe).filter((value): value is number => Number.isFinite(value));
  const startLine = lines.find((line) => /\bSTART\b/.test(line)) ?? lines[0] ?? "";

  return {
    sweepId: sweepDirName,
    sweepDir,
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

function listSweeps(task: TaskContext): SweepSummary[] {
  if (!fs.existsSync(task.logsRoot) || !fs.statSync(task.logsRoot).isDirectory()) return [];
  const dirs = fs
    .readdirSync(task.logsRoot)
    .map((name) => {
      const full = path.join(task.logsRoot, name);
      if (!fs.existsSync(full) || !fs.statSync(full).isDirectory() || !fs.existsSync(path.join(full, "timeline.log"))) {
        return null;
      }
      const lst = fs.lstatSync(full);
      return { name, isSymlink: lst.isSymbolicLink(), mtimeMs: fs.statSync(full).mtimeMs };
    })
    .filter((entry): entry is { name: string; isSymlink: boolean; mtimeMs: number } => entry !== null)
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
  const sweeps = dirs.map((entry) => parseSweep(task, entry.name, entry.isSymlink, entry.mtimeMs)).filter((entry): entry is ParsedSweepSummary => entry !== null);
  const deduped = new Map<string, ParsedSweepSummary>();
  for (const sweep of sweeps) {
    const key = sweep.startedAt ? `started:${sweep.startedAt}` : `id:${sweep.sweepId}`;
    const prev = deduped.get(key);
    if (!prev) {
      deduped.set(key, sweep);
      continue;
    }
    const preferSweep =
      Number(sweep.isSymlink) < Number(prev.isSymlink) ||
      (sweep.isSymlink === prev.isSymlink && sweep.runs.length > prev.runs.length) ||
      (sweep.isSymlink === prev.isSymlink && sweep.runs.length === prev.runs.length && sweep.mtimeMs > prev.mtimeMs);
    if (preferSweep) deduped.set(key, sweep);
  }
  const grouped = new Map<string, ParsedSweepSummary[]>();
  for (const sweep of deduped.values()) {
    const key = normalizeSweepId(sweep.sweepId);
    const group = grouped.get(key);
    if (group) group.push(sweep);
    else grouped.set(key, [sweep]);
  }

  return [...grouped.entries()]
    .map(([sweepId, group]) => {
      const runMap = new Map<string, RunSummary>();
      for (const sweep of group) {
        for (const run of sweep.runs) {
          const runKey = normalizeRunId(run.runId);
          const prev = runMap.get(runKey);
          runMap.set(runKey, prev ? mergeRunState(prev, run) : run);
        }
      }
      const runs = [...runMap.values()];
      const allAcc = runs.map((run) => run.bestAccuracy).filter((value): value is number => Number.isFinite(value));
      const ceVals = runs.map((run) => run.lastTrainCe).filter((value): value is number => Number.isFinite(value));
      const representative = group.slice().sort((a, b) => b.runs.length - a.runs.length || b.mtimeMs - a.mtimeMs)[0];
      const startedAt = group.map((sweep) => sweep.startedAt).filter((value): value is string => Boolean(value)).sort()[0] ?? null;
      const endedAt = group
        .map((sweep) => sweep.endedAt)
        .filter((value): value is string => Boolean(value))
        .sort()
        .at(-1) ?? null;
      return {
        sweepId,
        sweepDir: representative.sweepDir,
        status: group.some((sweep) => sweep.activeRuns > 0) ? "running" : "completed",
        startedAt,
        endedAt,
        runs,
        bestAccuracy: allAcc.length > 0 ? Math.max(...allAcc) : null,
        lastTrainCe: ceVals.length > 0 ? Math.min(...ceVals) : null,
        activeRuns: runs.filter((run) => run.isActive).length,
        mtimeMs: Math.max(...group.map((sweep) => sweep.mtimeMs)),
      };
    })
    .filter((sweep) => sweep.runs.length > 1 || sweep.activeRuns > 0)
    .sort((a, b) => b.mtimeMs - a.mtimeMs)
    .map(({ mtimeMs: _mtimeMs, ...sweep }) => sweep);
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

function listDocs(task: TaskContext): DocSummary[] {
  if (!fs.existsSync(task.docsRoot) || !fs.statSync(task.docsRoot).isDirectory()) return [];
  return fs
    .readdirSync(task.docsRoot)
    .filter((file) => file.toLowerCase().endsWith(".md"))
    .sort((a, b) => fs.statSync(path.join(task.docsRoot, b)).mtimeMs - fs.statSync(path.join(task.docsRoot, a)).mtimeMs)
    .map((file) => parseDoc(task, file));
}

function getBootstrap(task: TaskContext) {
  const sweeps = listSweeps(task);
  const docs = listDocs(task);
  const runMap = new Map<string, RunSummary>();
  for (const sweep of sweeps) {
    for (const run of sweep.runs) runMap.set(run.runId, run);
  }
  for (const doc of docs) {
    for (const runId of doc.runRefs) {
      if (runMap.has(runId)) continue;
      const run = parseRunSummary(task, runId);
      if (run.lastStep !== null) runMap.set(runId, run);
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
    sweeps,
    runs: allRuns,
    summary: {
      docsCount: docs.length,
      sweepsCount: sweeps.length,
      runsCount: allRuns.length,
      runsWithAccuracy: accRuns.length,
      bestRun: bestRun ? { runId: bestRun.runId, finalAccuracy: bestRun.finalAccuracy, bestAccuracy: bestRun.bestAccuracy } : null,
    },
  };
}

function transpileAppTs(): string {
  const src = readText(appTsPath);
  const transpiler = new Bun.Transpiler({ loader: "ts", target: "browser" });
  return transpiler.transformSync(src);
}

function transpileDocTs(): string {
  const src = readText(docTsPath);
  const transpiler = new Bun.Transpiler({ loader: "ts", target: "browser" });
  return transpiler.transformSync(src);
}

function resolveDocPath(task: TaskContext, fileName: string | null): string | null {
  const name = String(fileName ?? "").trim();
  if (!name) return null;
  const full = path.resolve(task.docsRoot, name);
  if (!full.startsWith(task.docsRoot)) return null;
  if (!fs.existsSync(full) || !fs.statSync(full).isFile()) return null;
  return full;
}

function resolveRunId(task: TaskContext, runId: string | null): string | null {
  const name = String(runId ?? "").trim();
  if (!name || !/^[A-Za-z0-9._-]+$/.test(name)) return null;
  const full = path.join(task.logsRoot, name);
  if (!fs.existsSync(full) || !fs.statSync(full).isDirectory()) return null;
  return name;
}

function formatRunForQa(run: RunSummary): string {
  return `${run.runId}: final_acc=${fmtAcc(run.finalAccuracy)} best_acc=${fmtAcc(run.bestAccuracy)} last_train_ce=${fmtAcc(run.lastTrainCe)} last_step=${fmtNum(run.lastStep)} active=${run.isActive ? "yes" : "no"} params=${fmtNum(run.numParams)}`;
}

function formatSweepForQa(sweep: SweepSummary): string {
  return `${sweep.sweepId}: status=${sweep.status} active_runs=${sweep.activeRuns} runs=${sweep.runs.length} best_acc=${fmtAcc(sweep.bestAccuracy)} last_train_ce=${fmtAcc(sweep.lastTrainCe)} started=${sweep.startedAt ?? "-"} ended=${sweep.endedAt ?? "-"}`;
}

function buildTaskQaPrompt(task: TaskContext, question: string, conversation: ChatTurn[]): string {
  const bootstrap = getBootstrap(task);
  const activeSweeps = bootstrap.sweeps.filter((sweep) => sweep.activeRuns > 0).slice(0, 6);
  const recentSweeps = bootstrap.sweeps.slice(0, 8);
  const activeRuns = bootstrap.runs.filter((run) => run.isActive).slice(0, 8);
  const bestRuns = bootstrap.runs
    .filter((run) => Number.isFinite(run.finalAccuracy ?? NaN))
    .slice()
    .sort((a, b) => (b.finalAccuracy ?? -1) - (a.finalAccuracy ?? -1))
    .slice(0, 6);
  const weakestRuns = bootstrap.runs
    .filter((run) => Number.isFinite(run.finalAccuracy ?? NaN))
    .slice()
    .sort((a, b) => (a.finalAccuracy ?? Number.POSITIVE_INFINITY) - (b.finalAccuracy ?? Number.POSITIVE_INFINITY))
    .slice(0, 6);
  const recentDocs = bootstrap.docs.slice(0, 8);
  const turns = conversation
    .slice(-QA_MAX_TURNS)
    .map((turn) => `${turn.role.toUpperCase()}: ${trimTo(turn.content, 1200)}`)
    .join("\n");

  return [
    "You are helping with a local autoresearch tracker.",
    `Task: ${task.title} (${task.id})`,
    task.description ? `Task description: ${task.description}` : null,
    task.qaPromptHint ? `Task-specific hint: ${task.qaPromptHint}` : null,
    "",
    "Answer the user's question about the task as a whole, not a single run unless the evidence points there.",
    "You may inspect local files if needed, but stay within this repo and prioritize the task's docs, scripts, and logs.",
    "Prefer current local evidence over generic ML advice. Compare against peer runs/sweeps when relevant.",
    "If the evidence is weak or the task is too early to judge, say that explicitly.",
    "",
    "Output format:",
    "Verdict: <one short paragraph>",
    "Likely causes:",
    "- <bullet>",
    "Evidence:",
    "- <bullet>",
    "Next action:",
    "- <bullet>",
    "",
    "Task paths:",
    `- docs: ${task.docsDir}`,
    `- scripts: ${task.scriptsDir}`,
    `- logs: ${task.logsDir}`,
    "",
    "Tracker snapshot:",
    `- generated_at: ${bootstrap.generatedAt}`,
    `- docs_count: ${bootstrap.summary.docsCount}`,
    `- sweeps_count: ${bootstrap.summary.sweepsCount}`,
    `- runs_count: ${bootstrap.summary.runsCount}`,
    `- runs_with_accuracy: ${bootstrap.summary.runsWithAccuracy}`,
    `- best_run: ${bootstrap.summary.bestRun ? `${bootstrap.summary.bestRun.runId} final_acc=${fmtAcc(bootstrap.summary.bestRun.finalAccuracy)} best_acc=${fmtAcc(bootstrap.summary.bestRun.bestAccuracy)}` : "-"}`,
    "",
    "Active sweeps:",
    ...(activeSweeps.length > 0 ? activeSweeps.map((sweep) => `- ${formatSweepForQa(sweep)}`) : ["- none"]),
    "",
    "Recent sweeps:",
    ...(recentSweeps.length > 0 ? recentSweeps.map((sweep) => `- ${formatSweepForQa(sweep)}`) : ["- none"]),
    "",
    "Active runs:",
    ...(activeRuns.length > 0 ? activeRuns.map((run) => `- ${formatRunForQa(run)}`) : ["- none"]),
    "",
    "Best completed runs:",
    ...(bestRuns.length > 0 ? bestRuns.map((run) => `- ${formatRunForQa(run)}`) : ["- none"]),
    "",
    "Weakest completed runs:",
    ...(weakestRuns.length > 0 ? weakestRuns.map((run) => `- ${formatRunForQa(run)}`) : ["- none"]),
    "",
    "Recent docs:",
    ...(recentDocs.length > 0 ? recentDocs.map((doc) => `- ${doc.file}: title=${doc.title} run_refs=${doc.runRefs.join(",") || "-"} updated=${doc.updatedAt}`) : ["- none"]),
    "",
    turns ? `Conversation so far:\n${turns}\n` : null,
    `User question: ${question}`,
  ]
    .filter((line): line is string => Boolean(line))
    .join("\n");
}

function parseEvidenceFromAnswer(answer: string): string[] {
  const lines = answer.split(/\r?\n/);
  const evidence: string[] = [];
  let inEvidence = false;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      if (inEvidence && evidence.length > 0) break;
      continue;
    }
    if (/^Evidence:/i.test(line)) {
      inEvidence = true;
      continue;
    }
    if (!inEvidence) continue;
    if (/^[A-Za-z][A-Za-z ]+:$/.test(line)) break;
    evidence.push(line.replace(/^[-*]\s+/, ""));
    if (evidence.length >= 8) break;
  }
  return evidence;
}

async function runCodexTaskQa(prompt: string): Promise<string> {
  if (!codexExecutable) throw new Error("codex executable not found on the server.");
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "research-task-qa-"));
  const outputFile = path.join(tempDir, "last-message.txt");
  let timedOut = false;
  const proc = Bun.spawn(
    [
      codexExecutable,
      "exec",
      "-C",
      repoRoot,
      "--dangerously-bypass-approvals-and-sandbox",
      "--output-last-message",
      outputFile,
      "-",
    ],
    {
      stdin: new Blob([prompt]),
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        HOME: preferredUserHome,
        CODEX_HOME: path.join(preferredUserHome, ".codex"),
        PATH: codexPathEntries.join(path.delimiter),
      },
    }
  );
  const timer = setTimeout(() => {
    timedOut = true;
    try {
      proc.kill();
    } catch {
      // no-op
    }
  }, QA_TIMEOUT_MS);

  try {
    const [exitCode, stdout, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    if (timedOut) throw new Error(`Task QA timed out after ${QA_TIMEOUT_MS / 1000}s.`);
    const output = (fs.existsSync(outputFile) ? readText(outputFile) : stdout).trim();
    if (output) return output;
    const detail = stderr.trim() || stdout.trim();
    throw new Error(detail || `codex exec exited with code ${exitCode}`);
  } finally {
    clearTimeout(timer);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

async function handleTaskQa(req: Request): Promise<Response> {
  let payload: { taskId?: string; question?: string; conversation?: ChatTurn[] };
  try {
    payload = (await req.json()) as { taskId?: string; question?: string; conversation?: ChatTurn[] };
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body." }), {
      status: 400,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
  const task = resolveTaskContext(payload.taskId ?? null);
  if (!task) {
    return new Response(JSON.stringify({ error: "Unknown task." }), {
      status: 404,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
  const question = String(payload.question ?? "").trim();
  if (!question) {
    return new Response(JSON.stringify({ error: "Missing question." }), {
      status: 400,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
  const conversation = Array.isArray(payload.conversation)
    ? payload.conversation
        .filter((turn): turn is ChatTurn => Boolean(turn) && (turn.role === "user" || turn.role === "assistant") && typeof turn.content === "string")
        .map((turn) => ({ role: turn.role, content: trimTo(turn.content, 4000) }))
    : [];

  try {
    const answer = await runCodexTaskQa(buildTaskQaPrompt(task, question, conversation));
    const response: TaskQaResponse = {
      taskId: task.id,
      answer,
      evidence: parseEvidenceFromAnswer(answer),
      generatedAt: new Date().toISOString(),
    };
    return new Response(JSON.stringify(response), {
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : String(error),
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      }
    );
  }
}

serve({
  port,
  async fetch(req) {
    const url = new URL(req.url);
    const task = resolveTaskContext(url.searchParams.get("task"));

    if (url.pathname === "/") {
      return new Response(readText(htmlPath), {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }
    if (url.pathname === "/styles.css") {
      return new Response(readText(cssPath), {
        headers: { "Content-Type": "text/css; charset=utf-8" },
      });
    }
    if (url.pathname === "/app.js") {
      return new Response(transpileAppTs(), {
        headers: { "Content-Type": "text/javascript; charset=utf-8" },
      });
    }
    if (url.pathname === "/doc") {
      return new Response(readText(docHtmlPath), {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }
    if (url.pathname === "/doc.js") {
      return new Response(transpileDocTs(), {
        headers: { "Content-Type": "text/javascript; charset=utf-8" },
      });
    }
    if (url.pathname === "/api/bootstrap") {
      if (!task) return new Response("Unknown task", { status: 404 });
      return new Response(JSON.stringify(getBootstrap(task)), {
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }
    if (url.pathname === "/api/doc") {
      if (!task) return new Response("Unknown task", { status: 404 });
      const full = resolveDocPath(task, url.searchParams.get("file"));
      if (!full) return new Response("Not found", { status: 404 });
      return new Response(readText(full), {
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    if (url.pathname === "/api/run") {
      if (!task) return new Response("Unknown task", { status: 404 });
      const runId = resolveRunId(task, url.searchParams.get("runId"));
      if (!runId) return new Response("Not found", { status: 404 });
      const detail = parseRunDetail(task, runId);
      if (!detail) return new Response("Not found", { status: 404 });
      return new Response(JSON.stringify(detail), {
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }
    if (url.pathname === "/api/qa/task") {
      if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });
      return handleTaskQa(req);
    }
    return new Response("Not found", { status: 404 });
  },
});

console.log(
  `[research-tracker] http://localhost:${port} task=${defaultTask.id} tasks_root=${path.relative(repoRoot, tasksRoot)}`
);
