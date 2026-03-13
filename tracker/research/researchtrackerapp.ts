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
  isEvalOnly: boolean;
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

type IdeaTreeNodeKind = "idea" | "property" | "variant";

type IdeaTreeNode = {
  id: string;
  label: string;
  kind: IdeaTreeNodeKind;
  summary: string;
  detail: string;
  aliases: string[];
  evidenceRefs: string[];
};

type IdeaTreeEdge = {
  id: string;
  fromId: string;
  toId: string;
  relationPhrase: string;
  reasoning: string;
  evidenceRefs: string[];
};

type IdeaTreeResponse = {
  taskId: string;
  generatedAt: string;
  anchors: string[];
  inputs: {
    snapshotGeneratedAt: string;
    docs: string[];
    snippetCount: number;
    runFamilyCount: number;
    pipelineVersion: string;
  };
  graph: {
    nodes: IdeaTreeNode[];
    edges: IdeaTreeEdge[];
  };
  debug: IdeaGraphDebug;
};

type IdeaTreeProgressStage = "harvest" | "attachment" | "synthesis";

type IdeaTreeProgressStatus = {
  taskId: string;
  state: "idle" | "running" | "completed" | "error";
  activeStage: IdeaTreeProgressStage | null;
  message: string;
  startedAt: string | null;
  updatedAt: string;
};

type IdeaTreeRequest = {
  taskId?: string;
};

type IdeaGraphEvidenceDoc = {
  file: string;
  title: string;
  updatedAt: string;
  runRefs: string[];
  snippetIds: string[];
};

type IdeaGraphEvidenceSnippet = {
  id: string;
  docFile: string;
  heading: string;
  updatedAt: string;
  excerpt: string;
  runRefs: string[];
  mentionedAccuracies: number[];
  score: number;
};

type IdeaGraphRunFamily = {
  id: string;
  familyKey: string;
  label: string;
  runCount: number;
  activeRuns: number;
  bestAccuracy: number | null;
  lastTrainCe: number | null;
  docRefs: string[];
  sampleRuns: string[];
};

type IdeaGraphHarvestCandidate = {
  id: string;
  label: string;
  kind: IdeaTreeNodeKind;
  aliases: string[];
  candidateRefs: string[];
  notes: string;
};

type IdeaGraphAttachedCandidate = {
  id: string;
  label: string;
  kind: IdeaTreeNodeKind;
  aliases: string[];
  summary: string;
  detail: string;
  coverageNote: string;
  evidenceRefs: string[];
  contradictionRefs: string[];
};

type IdeaGraphDroppedCandidate = {
  id: string;
  label: string;
  reason: string;
  candidateRefs: string[];
};

type IdeaGraphEdgeModel = {
  fromId: string;
  toId: string;
  relationPhrase: string;
  reasoning: string;
  evidenceRefs: string[];
};

type IdeaGraphDebug = {
  timings: {
    evidenceMs: number;
    harvestMs: number;
    attachmentMs: number;
    graphMs: number;
    totalMs: number;
  };
  evidence: {
    docs: IdeaGraphEvidenceDoc[];
    snippets: IdeaGraphEvidenceSnippet[];
    runFamilies: IdeaGraphRunFamily[];
  };
  harvest: {
    candidates: IdeaGraphHarvestCandidate[];
  };
  attachment: {
    kept: IdeaGraphAttachedCandidate[];
    dropped: IdeaGraphDroppedCandidate[];
  };
};

const ACTIVE_WINDOW_MS = 45 * 60 * 1000;
const QA_TIMEOUT_MS = 2 * 60 * 1000;
const QA_MAX_TURNS = 8;
const IDEA_TREE_TIMEOUT_MS = 90 * 1000;
const IDEA_TREE_MAX_ANCHORS = 5;
const IDEA_GRAPH_MAX_SNIPPETS = 96;
const IDEA_GRAPH_MAX_SNIPPETS_PER_DOC = 4;
const IDEA_GRAPH_SNIPPET_CHARS = 340;
const IDEA_GRAPH_HARVEST_MAX_CANDIDATES = 16;
const IDEA_GRAPH_ATTACH_MAX_REFS = 10;
const IDEA_GRAPH_HARVEST_TIMEOUT_MS = 90 * 1000;
const IDEA_GRAPH_ATTACHMENT_TIMEOUT_MS = 90 * 1000;
const IDEA_GRAPH_SYNTHESIS_TIMEOUT_MS = 90 * 1000;
const IDEA_TREE_REASONING_EFFORT = "low";
const userName = process.env.USER || process.env.LOGNAME || "";
const preferredUserHome =
  (userName && fs.existsSync(path.join("/home", userName)) && path.join("/home", userName)) ||
  process.env.HOME ||
  os.homedir();
const ideaTreeProgressByTask = new Map<string, IdeaTreeProgressStatus>();

const IDEA_GRAPH_HARVEST_OUTPUT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["candidates"],
  properties: {
    candidates: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["id", "label", "kind", "aliases", "candidateRefs", "notes"],
        properties: {
          id: { type: "string", minLength: 1, maxLength: 48 },
          label: { type: "string", minLength: 1, maxLength: 80 },
          kind: { type: "string", enum: ["idea", "property", "variant"] },
          aliases: {
            type: "array",
            items: { type: "string", minLength: 1, maxLength: 80 },
          },
          candidateRefs: {
            type: "array",
            minItems: 1,
            items: { type: "string", minLength: 1, maxLength: 80 },
          },
          notes: { type: "string", minLength: 1, maxLength: 220 },
        },
      },
    },
  },
} as const;

const IDEA_GRAPH_ATTACHMENT_OUTPUT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["kept", "dropped"],
  properties: {
    kept: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["id", "label", "kind", "aliases", "summary", "detail", "coverageNote", "evidenceRefs", "contradictionRefs"],
        properties: {
          id: { type: "string", minLength: 1, maxLength: 48 },
          label: { type: "string", minLength: 1, maxLength: 80 },
          kind: { type: "string", enum: ["idea", "property", "variant"] },
          aliases: {
            type: "array",
            items: { type: "string", minLength: 1, maxLength: 80 },
          },
          summary: { type: "string", minLength: 1, maxLength: 160 },
          detail: { type: "string", minLength: 1, maxLength: 420 },
          coverageNote: { type: "string", minLength: 1, maxLength: 200 },
          evidenceRefs: {
            type: "array",
            minItems: 1,
            items: { type: "string", minLength: 1, maxLength: 80 },
          },
          contradictionRefs: {
            type: "array",
            items: { type: "string", minLength: 1, maxLength: 80 },
          },
        },
      },
    },
    dropped: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["id", "label", "reason", "candidateRefs"],
        properties: {
          id: { type: "string", minLength: 1, maxLength: 48 },
          label: { type: "string", minLength: 1, maxLength: 80 },
          reason: { type: "string", minLength: 1, maxLength: 220 },
          candidateRefs: {
            type: "array",
            items: { type: "string", minLength: 1, maxLength: 80 },
          },
        },
      },
    },
  },
} as const;

const IDEA_GRAPH_SYNTHESIS_OUTPUT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["anchors", "edges"],
  properties: {
    anchors: {
      type: "array",
      minItems: 3,
      maxItems: IDEA_TREE_MAX_ANCHORS,
      items: { type: "string", minLength: 1, maxLength: 48 },
    },
    edges: {
      type: "array",
      minItems: 3,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["fromId", "toId", "relationPhrase", "reasoning", "evidenceRefs"],
        properties: {
          fromId: { type: "string", minLength: 1, maxLength: 48 },
          toId: { type: "string", minLength: 1, maxLength: 48 },
          relationPhrase: { type: "string", minLength: 1, maxLength: 48 },
          reasoning: { type: "string", minLength: 1, maxLength: 260 },
          evidenceRefs: {
            type: "array",
            minItems: 1,
            items: { type: "string", minLength: 1, maxLength: 80 },
          },
        },
      },
    },
  },
} as const;

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
    isEvalOnly: false,
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
  out.isEvalOnly = parsed.isEvalOnly;
  return out;
}

function parseRunDetail(task: TaskContext, runId: string): RunDetail | null {
  const parsed = parseRunLog(path.join(task.logsRoot, runId));
  const run = parseRunSummary(task, runId);
  if (!run.logfile) return null;

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
    isEvalOnly: preferred.isEvalOnly || base.isEvalOnly || other.isEvalOnly,
    trainableParams: preferred.trainableParams ?? base.trainableParams ?? other.trainableParams,
    numParams: preferred.numParams ?? base.numParams ?? other.numParams,
  };
}

function shouldIncludeRun(run: RunSummary): boolean {
  return run.lastStep !== null || run.finalAccuracy !== null || run.bestAccuracy !== null || run.isEvalOnly;
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
  const runs = parsedRuns.filter(shouldIncludeRun);
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

function sanitizeIdeaId(value: string, max = 48): string {
  const cleaned = String(value ?? "")
    .trim()
    .replace(/[^A-Za-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned.slice(0, max) || "item";
}

function setIdeaTreeProgress(
  taskId: string,
  patch: Partial<IdeaTreeProgressStatus> & Pick<IdeaTreeProgressStatus, "state" | "message">
): IdeaTreeProgressStatus {
  const prev = ideaTreeProgressByTask.get(taskId);
  const next: IdeaTreeProgressStatus = {
    taskId,
    state: patch.state,
    activeStage: patch.activeStage ?? prev?.activeStage ?? null,
    message: patch.message,
    startedAt: patch.startedAt ?? prev?.startedAt ?? null,
    updatedAt: new Date().toISOString(),
  };
  ideaTreeProgressByTask.set(taskId, next);
  return next;
}

function normalizeIdeaSearchText(value: string): string {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenizeIdeaSearchText(value: string): string[] {
  const stopwords = new Set([
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "task",
    "report",
    "plan",
    "note",
    "default",
    "main",
    "frontier",
    "family",
    "run",
    "runs",
    "bridge",
    "idea",
    "variant",
    "property",
  ]);
  return uniq(
    normalizeIdeaSearchText(value)
      .split(" ")
      .filter((token) => token.length >= 3 && !stopwords.has(token))
  );
}

function extractRunRefsFromText(task: TaskContext, text: string): string[] {
  const logsRefs = [...text.matchAll(/logs\/([A-Za-z0-9._-]+)/g)].map((match) => match[1]);
  const inlineRefs = [...text.matchAll(/`([A-Za-z0-9._-]{6,})`/g)]
    .map((match) => match[1])
    .filter((runId) => fs.existsSync(path.join(task.logsRoot, runId)));
  return uniq([...logsRefs, ...inlineRefs]);
}

function extractMentionedAccuracies(text: string): number[] {
  return uniq(
    [...text.matchAll(/\b0\.\d{3,4}\b/g)]
      .map((match) => Number(match[0]))
      .filter((value) => Number.isFinite(value) && value >= 0.1)
  ).sort((a, b) => b - a);
}

function compressIdeaSnippetText(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\|/g, " | ")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

function scoreIdeaGraphSnippet(doc: DocSummary, heading: string, excerpt: string, order: number): number {
  const headingScore = /(overview|scope|results|ranking|failure|conclusion|combination|patterns|purpose|baseline|novelty|answer|launch|decision|queue)/i.test(heading) ? 8 : 0;
  const docScore = /(analysis|report|overview|state|context|brainstorm|ideas|diagnosis|final|future|plan)/i.test(`${doc.file} ${doc.title}`) ? 6 : 0;
  const contentScore = /(best|delta|lift|regression|failure|throughput|constraint|anchor|baseline|stack|combine|question|qcond|earlylayer|geomcal|adapter|qquery|dynbudget|structuredroles|multiscale|perceiver)/i.test(excerpt)
    ? 7
    : 0;
  const runScore = Math.min(4, doc.runRefs.length);
  const accScore = Math.min(3, doc.mentionedAccuracies.length);
  return headingScore + docScore + contentScore + runScore + accScore - Math.min(order, 5);
}

function splitDocIntoIdeaSnippets(task: TaskContext, doc: DocSummary): IdeaGraphEvidenceSnippet[] {
  const raw = readText(path.join(task.docsRoot, doc.file));
  const lines = raw.split(/\r?\n/);
  const sections: Array<{ heading: string; lines: string[]; order: number }> = [];
  let current = { heading: doc.title, lines: [] as string[], order: 0 };
  for (const line of lines) {
    const headingMatch = line.match(/^#{1,3}\s+(.+)$/);
    if (headingMatch) {
      sections.push(current);
      current = {
        heading: headingMatch[1].trim(),
        lines: [],
        order: sections.length,
      };
      continue;
    }
    current.lines.push(line);
  }
  sections.push(current);
  const snippets: IdeaGraphEvidenceSnippet[] = [];
  for (const section of sections) {
    const rawSection = section.lines.join("\n").trim();
    const excerpt = trimTo(compressIdeaSnippetText(rawSection), IDEA_GRAPH_SNIPPET_CHARS);
    if (!excerpt) continue;
    const sectionRunRefs = uniq([...doc.runRefs, ...extractRunRefsFromText(task, rawSection)]).slice(0, 8);
    const sectionAccuracies = extractMentionedAccuracies(rawSection).slice(0, 6);
    snippets.push({
      id: sanitizeIdeaId(`snippet_${doc.file.replace(/\.md$/i, "")}_${section.order + 1}_${section.heading}`, 80),
      docFile: doc.file,
      heading: section.heading || doc.title,
      updatedAt: doc.updatedAt,
      excerpt,
      runRefs: sectionRunRefs,
      mentionedAccuracies: sectionAccuracies,
      score: scoreIdeaGraphSnippet(doc, section.heading, excerpt, section.order),
    });
  }
  return snippets;
}

function selectIdeaGraphSnippets(docs: DocSummary[], allSnippets: IdeaGraphEvidenceSnippet[]): IdeaGraphEvidenceSnippet[] {
  const byDoc = new Map<string, IdeaGraphEvidenceSnippet[]>();
  for (const snippet of allSnippets) {
    const group = byDoc.get(snippet.docFile);
    if (group) group.push(snippet);
    else byDoc.set(snippet.docFile, [snippet]);
  }
  const selected: IdeaGraphEvidenceSnippet[] = [];
  const selectedIds = new Set<string>();
  const perDocCount = new Map<string, number>();
  for (const doc of docs) {
    const lead = (byDoc.get(doc.file) ?? []).slice().sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt) || a.heading.localeCompare(b.heading))[0];
    if (!lead || selectedIds.has(lead.id)) continue;
    selected.push(lead);
    selectedIds.add(lead.id);
    perDocCount.set(doc.file, 1);
  }
  const extras = allSnippets
    .filter((snippet) => !selectedIds.has(snippet.id))
    .sort((a, b) => b.score - a.score || Date.parse(b.updatedAt) - Date.parse(a.updatedAt) || a.docFile.localeCompare(b.docFile));
  for (const snippet of extras) {
    if (selected.length >= IDEA_GRAPH_MAX_SNIPPETS) break;
    const count = perDocCount.get(snippet.docFile) ?? 0;
    if (count >= IDEA_GRAPH_MAX_SNIPPETS_PER_DOC) continue;
    selected.push(snippet);
    selectedIds.add(snippet.id);
    perDocCount.set(snippet.docFile, count + 1);
  }
  return selected.sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt) || b.score - a.score || a.docFile.localeCompare(b.docFile));
}

function buildIdeaGraphRunFamilies(runs: RunSummary[], docs: DocSummary[]): IdeaGraphRunFamily[] {
  const groups = new Map<string, RunSummary[]>();
  for (const run of runs) {
    const familyKey = normalizeRunId(run.runId);
    const group = groups.get(familyKey);
    if (group) group.push(run);
    else groups.set(familyKey, [run]);
  }
  const docFamilyRefs = new Map<string, Set<string>>();
  for (const doc of docs) {
    for (const runId of doc.runRefs) {
      const familyKey = normalizeRunId(runId);
      const refs = docFamilyRefs.get(familyKey);
      if (refs) refs.add(doc.file);
      else docFamilyRefs.set(familyKey, new Set([doc.file]));
    }
  }
  return [...groups.entries()]
    .map(([familyKey, group]) => {
      const accuracies = group.map((run) => run.finalAccuracy).filter((value): value is number => Number.isFinite(value));
      const ces = group.map((run) => run.lastTrainCe).filter((value): value is number => Number.isFinite(value));
      const sampleRuns = group
        .slice()
        .sort(
          (a, b) =>
            (b.finalAccuracy ?? -1) - (a.finalAccuracy ?? -1) ||
            (b.bestAccuracy ?? -1) - (a.bestAccuracy ?? -1) ||
            (b.lastStep ?? -1) - (a.lastStep ?? -1)
        )
        .slice(0, 4)
        .map((run) => run.runId);
      return {
        id: sanitizeIdeaId(`runfam_${familyKey}`, 80),
        familyKey,
        label: familyKey.replace(/_/g, " "),
        runCount: group.length,
        activeRuns: group.filter((run) => run.isActive).length,
        bestAccuracy: accuracies.length > 0 ? Math.max(...accuracies) : null,
        lastTrainCe: ces.length > 0 ? Math.min(...ces) : null,
        docRefs: [...(docFamilyRefs.get(familyKey) ?? new Set<string>())].sort().slice(0, 8),
        sampleRuns,
      };
    })
    .sort(
      (a, b) =>
        b.docRefs.length - a.docRefs.length ||
        b.activeRuns - a.activeRuns ||
        (b.bestAccuracy ?? -1) - (a.bestAccuracy ?? -1) ||
        b.runCount - a.runCount ||
        a.familyKey.localeCompare(b.familyKey)
    );
}

function buildIdeaGraphEvidencePack(task: TaskContext, bootstrap: ReturnType<typeof getBootstrap>): {
  docs: IdeaGraphEvidenceDoc[];
  snippets: IdeaGraphEvidenceSnippet[];
  runFamilies: IdeaGraphRunFamily[];
} {
  const allSnippets = bootstrap.docs.flatMap((doc) => splitDocIntoIdeaSnippets(task, doc));
  const snippets = selectIdeaGraphSnippets(bootstrap.docs, allSnippets);
  const snippetIdsByDoc = new Map<string, string[]>();
  for (const snippet of snippets) {
    const group = snippetIdsByDoc.get(snippet.docFile);
    if (group) group.push(snippet.id);
    else snippetIdsByDoc.set(snippet.docFile, [snippet.id]);
  }
  const docs = bootstrap.docs.map((doc) => ({
    file: doc.file,
    title: doc.title,
    updatedAt: doc.updatedAt,
    runRefs: doc.runRefs.slice(0, 8),
    snippetIds: snippetIdsByDoc.get(doc.file) ?? [],
  }));
  const runFamilies = buildIdeaGraphRunFamilies(bootstrap.runs, bootstrap.docs);
  return { docs, snippets, runFamilies };
}

type IdeaGraphEvidenceLookupEntry = {
  id: string;
  kind: "snippet" | "run_family";
  text: string;
  short: string;
};

function buildIdeaGraphEvidenceLookup(evidence: {
  snippets: IdeaGraphEvidenceSnippet[];
  runFamilies: IdeaGraphRunFamily[];
}): Map<string, IdeaGraphEvidenceLookupEntry> {
  const lookup = new Map<string, IdeaGraphEvidenceLookupEntry>();
  for (const snippet of evidence.snippets) {
    const short = `${snippet.docFile} • ${snippet.heading}`;
    lookup.set(snippet.id, {
      id: snippet.id,
      kind: "snippet",
      short,
      text: `snippet ${snippet.id} | doc=${snippet.docFile} | heading=${snippet.heading} | runs=${snippet.runRefs.join(",") || "-"} | excerpt=${snippet.excerpt}`,
    });
  }
  for (const family of evidence.runFamilies) {
    const short = `${family.familyKey} • runs=${family.runCount}`;
    lookup.set(family.id, {
      id: family.id,
      kind: "run_family",
      short,
      text: `run_family ${family.id} | family=${family.familyKey} | runs=${family.runCount} | active=${family.activeRuns} | best_acc=${fmtAcc(family.bestAccuracy)} | last_train_ce=${fmtAcc(family.lastTrainCe)} | docs=${family.docRefs.join(",") || "-"} | samples=${family.sampleRuns.join(",") || "-"}`,
    });
  }
  return lookup;
}

function formatIdeaGraphEvidencePackForHarvestPrompt(
  task: TaskContext,
  bootstrap: ReturnType<typeof getBootstrap>,
  evidence: { docs: IdeaGraphEvidenceDoc[]; snippets: IdeaGraphEvidenceSnippet[]; runFamilies: IdeaGraphRunFamily[] }
): string {
  return [
    "You are stage 1 of 3 for a semantic idea-graph build.",
    `Task: ${task.title} (${task.id})`,
    task.description ? `Task description: ${task.description}` : null,
    task.qaPromptHint ? `Task-specific hint: ${task.qaPromptHint}` : null,
    "",
    "Stage goal:",
    "- Harvest stable candidate concepts, properties, and concrete variants across the full task corpus.",
    "- Optimize for recall, not final pruning.",
    "- Do not build edges yet.",
    "",
    "Return JSON only with this shape:",
    "{",
    '  "candidates": [',
    "    {",
    '      "id": "string",',
    '      "label": "canonical concept label",',
    '      "kind": "idea|property|variant",',
    '      "aliases": ["optional alias"],',
    '      "candidateRefs": ["snippet_or_run_family_ref"],',
    '      "notes": "why this candidate matters"',
    "    }",
    "  ]",
    "}",
    "",
    "Rules:",
    `- Return between 10 and ${IDEA_GRAPH_HARVEST_MAX_CANDIDATES} candidates.`,
    "- candidateRefs must use only snippet ids or run_family ids from the evidence pack below.",
    "- Prefer stable project concepts over plan labels, queue labels, or filenames.",
    "- Collapse aliases if two labels describe the same underlying mechanism.",
    "- It is okay to keep partially supported candidates here; stage 2 will prune them.",
    "- Do not create edges, priorities, or next actions.",
    "",
    "Tracker snapshot:",
    `- generated_at=${bootstrap.generatedAt}`,
    `- docs=${bootstrap.summary.docsCount}`,
    `- runs=${bootstrap.summary.runsCount}`,
    `- runs_with_accuracy=${bootstrap.summary.runsWithAccuracy}`,
    "",
    "Docs in corpus:",
    ...(evidence.docs.length > 0
      ? evidence.docs.map(
          (doc) => `- ${doc.file} | title=${doc.title} | updated=${doc.updatedAt} | run_refs=${doc.runRefs.join(",") || "-"} | snippet_refs=${doc.snippetIds.join(",") || "-"}`
        )
      : ["- none"]),
    "",
    "Selected snippet evidence across the corpus:",
    ...(evidence.snippets.length > 0
      ? evidence.snippets.map(
          (snippet) =>
            `- ${snippet.id} | doc=${snippet.docFile} | heading=${snippet.heading} | runs=${snippet.runRefs.join(",") || "-"} | acc=${snippet.mentionedAccuracies.join(",") || "-"} | excerpt=${snippet.excerpt}`
        )
      : ["- none"]),
    "",
    "Run families:",
    ...(evidence.runFamilies.length > 0
      ? evidence.runFamilies.map(
          (family) =>
            `- ${family.id} | family=${family.familyKey} | runs=${family.runCount} | active=${family.activeRuns} | best_acc=${fmtAcc(family.bestAccuracy)} | last_train_ce=${fmtAcc(family.lastTrainCe)} | docs=${family.docRefs.join(",") || "-"} | samples=${family.sampleRuns.join(",") || "-"}`
        )
      : ["- none"]),
  ]
    .filter((line): line is string => Boolean(line))
    .join("\n");
}

function scoreIdeaEvidenceMatch(
  candidateTerms: string[],
  candidatePhrases: string[],
  entry: IdeaGraphEvidenceLookupEntry,
  candidateRefSet: Set<string>
): number {
  let score = candidateRefSet.has(entry.id) ? 8 : 0;
  const haystack = normalizeIdeaSearchText(entry.text);
  for (const phrase of candidatePhrases) {
    if (phrase.length >= 4 && haystack.includes(phrase)) score += 5;
  }
  const tokenSet = new Set(haystack.split(" ").filter(Boolean));
  for (const term of candidateTerms) {
    if (tokenSet.has(term)) score += 2;
  }
  return score;
}

function collectCandidateEvidenceRefs(
  candidate: IdeaGraphHarvestCandidate,
  evidence: { docs: IdeaGraphEvidenceDoc[]; snippets: IdeaGraphEvidenceSnippet[]; runFamilies: IdeaGraphRunFamily[] },
  lookup: Map<string, IdeaGraphEvidenceLookupEntry>
): string[] {
  const docSnippetMap = new Map(evidence.docs.map((doc) => [doc.file, doc.snippetIds] as const));
  const familyByKey = new Map(evidence.runFamilies.map((family) => [family.familyKey, family.id] as const));
  const familyByRun = new Map<string, string>();
  for (const family of evidence.runFamilies) {
    for (const runId of family.sampleRuns) familyByRun.set(runId, family.id);
  }
  const directRefs = new Set<string>();
  for (const rawRef of candidate.candidateRefs) {
    const ref = String(rawRef ?? "").trim();
    if (!ref) continue;
    if (lookup.has(ref)) {
      directRefs.add(ref);
      continue;
    }
    for (const snippetId of docSnippetMap.get(ref) ?? []) directRefs.add(snippetId);
    const familyRef = familyByKey.get(normalizeRunId(ref)) ?? familyByRun.get(ref);
    if (familyRef) directRefs.add(familyRef);
  }
  const candidatePhrases = uniq([candidate.label, ...candidate.aliases].map(normalizeIdeaSearchText).filter((value) => value.length >= 4));
  const candidateTerms = tokenizeIdeaSearchText([candidate.label, ...candidate.aliases].join(" "));
  const scored = [...lookup.values()]
    .map((entry) => ({
      id: entry.id,
      score: scoreIdeaEvidenceMatch(candidateTerms, candidatePhrases, entry, directRefs),
    }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
  const refs = [...directRefs];
  for (const entry of scored) {
    if (refs.length >= IDEA_GRAPH_ATTACH_MAX_REFS) break;
    if (refs.includes(entry.id)) continue;
    refs.push(entry.id);
  }
  return refs.slice(0, IDEA_GRAPH_ATTACH_MAX_REFS);
}

function buildIdeaGraphAttachmentPrompt(
  task: TaskContext,
  candidates: IdeaGraphHarvestCandidate[],
  evidenceRefsByCandidate: Map<string, string[]>,
  lookup: Map<string, IdeaGraphEvidenceLookupEntry>
): string {
  return [
    "You are stage 2 of 3 for a semantic idea-graph build.",
    `Task: ${task.title} (${task.id})`,
    "",
    "Stage goal:",
    "- Prune weak or duplicate candidates.",
    "- Attach explicit supporting evidence to the survivors.",
    "- Write compact summaries/details for the surviving candidates.",
    "",
    "Return JSON only with this shape:",
    "{",
    '  "kept": [',
    "    {",
    '      "id": "string",',
    '      "label": "canonical concept label",',
    '      "kind": "idea|property|variant",',
    '      "aliases": ["optional alias"],',
    '      "summary": "one compact sentence",',
    '      "detail": "2-3 short sentences",',
    '      "coverageNote": "what the evidence coverage looks like",',
    '      "evidenceRefs": ["supporting_ref"],',
    '      "contradictionRefs": ["optional_ref"]',
    "    }",
    "  ],",
    '  "dropped": [',
    "    {",
    '      "id": "string",',
    '      "label": "candidate label",',
    '      "reason": "why it was dropped",',
    '      "candidateRefs": ["original_ref"]',
    "    }",
    "  ]",
    "}",
    "",
    "Rules:",
    "- Keep only stable semantic concepts that are actually supported by the evidence.",
    "- Drop thin, duplicate, purely organizational, or speculative candidates.",
    "- Keep evidenceRefs focused and explicit. Prefer snippet ids and run_family ids that directly support the concept.",
    "- contradictionRefs should be empty unless the evidence genuinely cuts against the candidate.",
    "- coverageNote should state whether the support is broad, narrow, or mixed.",
    "",
    "Candidates and matched evidence:",
    ...candidates.flatMap((candidate) => {
      const refs = evidenceRefsByCandidate.get(candidate.id) ?? [];
      return [
        `- candidate ${candidate.id} | label=${candidate.label} | kind=${candidate.kind} | aliases=${candidate.aliases.join(",") || "-"} | seed_refs=${candidate.candidateRefs.join(",") || "-"} | notes=${candidate.notes}`,
        ...refs.map((ref) => `  - ${lookup.get(ref)?.text ?? ref}`),
      ];
    }),
  ]
    .filter((line): line is string => Boolean(line))
    .join("\n");
}

function buildIdeaGraphSynthesisPrompt(
  task: TaskContext,
  candidates: IdeaGraphAttachedCandidate[],
  lookup: Map<string, IdeaGraphEvidenceLookupEntry>
): string {
  return [
    "You are stage 3 of 3 for a semantic idea-graph build.",
    `Task: ${task.title} (${task.id})`,
    "",
    "Stage goal:",
    "- Build a sparse semantic DAG from the validated candidates below.",
    "- Choose the most central entry concepts as anchors.",
    "- Create only evidence-backed concept-to-concept edges.",
    "",
    "Return JSON only with this shape:",
    "{",
    '  "anchors": ["candidate_id"],',
    '  "edges": [',
    "    {",
    '      "fromId": "candidate_id",',
    '      "toId": "candidate_id",',
    '      "relationPhrase": "short semantic relation",',
    '      "reasoning": "why the evidence supports the edge",',
    '      "evidenceRefs": ["supporting_ref"]',
    "    }",
    "  ]",
    "}",
    "",
    "Rules:",
    `- Return between 3 and ${IDEA_TREE_MAX_ANCHORS} anchors.`,
    "- Nodes are fixed. Do not invent new node ids.",
    "- Prefer sparse, strong edges over dense speculative ones.",
    "- Edge direction matters and should read naturally from source to target.",
    "- Avoid organizational or process relations.",
    "- If two candidates do not have a real supported relation, leave them disconnected.",
    "",
    "Validated candidates:",
    ...candidates.flatMap((candidate) => [
      `- ${candidate.id} | label=${candidate.label} | kind=${candidate.kind} | aliases=${candidate.aliases.join(",") || "-"} | summary=${candidate.summary} | coverage=${candidate.coverageNote} | evidence=${candidate.evidenceRefs.join(",") || "-"} | contradictions=${candidate.contradictionRefs.join(",") || "-"}`,
      ...candidate.evidenceRefs.slice(0, 6).map((ref) => `  - ${lookup.get(ref)?.text ?? ref}`),
    ]),
  ]
    .filter((line): line is string => Boolean(line))
    .join("\n");
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
  return `${run.runId}: final_acc=${fmtAcc(run.finalAccuracy)} best_acc=${fmtAcc(run.bestAccuracy)} last_train_ce=${fmtAcc(run.lastTrainCe)} last_step=${fmtNum(run.lastStep)} active=${run.isActive ? "yes" : "no"} eval_only=${run.isEvalOnly ? "yes" : "no"} params=${fmtNum(run.numParams)}`;
}

function formatSweepForQa(sweep: SweepSummary): string {
  return `${sweep.sweepId}: status=${sweep.status} active_runs=${sweep.activeRuns} runs=${sweep.runs.length} best_acc=${fmtAcc(sweep.bestAccuracy)} last_train_ce=${fmtAcc(sweep.lastTrainCe)} started=${sweep.startedAt ?? "-"} ended=${sweep.endedAt ?? "-"}`;
}

function buildTaskQaPrompt(
  task: TaskContext,
  question: string,
  conversation: ChatTurn[]
): string {
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

async function runCodexPrompt(
  prompt: string,
  opts: {
    tempPrefix: string;
    timeoutMs: number;
    timeoutLabel: string;
    reasoningEffort?: "low" | "medium" | "high" | "xhigh";
    outputSchema?: Record<string, unknown>;
  }
): Promise<string> {
  if (!codexExecutable) throw new Error("codex executable not found on the server.");
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), opts.tempPrefix));
  const outputFile = path.join(tempDir, "last-message.txt");
  const args = [
    codexExecutable,
    "exec",
    "-C",
    repoRoot,
    "--ephemeral",
    "-s",
    "read-only",
  ];
  if (opts.reasoningEffort) args.push("-c", `model_reasoning_effort="${opts.reasoningEffort}"`);
  if (opts.outputSchema) {
    const schemaFile = path.join(tempDir, "output-schema.json");
    fs.writeFileSync(schemaFile, JSON.stringify(opts.outputSchema, null, 2));
    args.push("--output-schema", schemaFile);
  }
  args.push("--output-last-message", outputFile, "-");
  let timedOut = false;
  const proc = Bun.spawn(
    args,
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
  }, opts.timeoutMs);

  try {
    const [exitCode, stdout, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    if (timedOut) throw new Error(`${opts.timeoutLabel} timed out after ${opts.timeoutMs / 1000}s.`);
    const output = (fs.existsSync(outputFile) ? readText(outputFile) : stdout).trim();
    if (output) return output;
    const detail = stderr.trim() || stdout.trim();
    throw new Error(detail || `codex exec exited with code ${exitCode}`);
  } finally {
    clearTimeout(timer);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

async function runCodexTaskQa(prompt: string): Promise<string> {
  return runCodexPrompt(prompt, {
    tempPrefix: "research-task-qa-",
    timeoutMs: QA_TIMEOUT_MS,
    timeoutLabel: "Task QA",
    reasoningEffort: "high",
  });
}

async function runCodexIdeaGraphStage(
  prompt: string,
  opts: {
    stage: string;
    timeoutMs: number;
    outputSchema: Record<string, unknown>;
  }
): Promise<string> {
  return runCodexPrompt(prompt, {
    tempPrefix: `research-idea-graph-${opts.stage}-`,
    timeoutMs: opts.timeoutMs,
    timeoutLabel: `Idea graph ${opts.stage}`,
    reasoningEffort: IDEA_TREE_REASONING_EFFORT,
    outputSchema: opts.outputSchema,
  });
}

function extractJsonObject(raw: string): string {
  const trimmed = raw.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) return fenced[1].trim();
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) return trimmed.slice(start, end + 1).trim();
  return trimmed;
}

function normalizeIdeaNodeKind(value: unknown): IdeaTreeNodeKind {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "idea" || normalized === "property" || normalized === "variant") return normalized;
  return "idea";
}

function normalizeIdeaRefs(value: unknown, max = 6): string[] {
  if (!Array.isArray(value)) return [];
  return uniq(
    value
      .map((entry) => trimTo(String(entry ?? "").trim(), 80))
      .filter(Boolean)
      .slice(0, max)
  );
}

function normalizeIdeaGraphHarvestCandidates(value: unknown, evidenceIds: Set<string>): IdeaGraphHarvestCandidate[] {
  if (!Array.isArray(value)) throw new Error("Idea graph harvest response is missing candidates[].");
  const seen = new Set<string>();
  const candidates: IdeaGraphHarvestCandidate[] = [];
  for (const rawCandidate of value) {
    if (!rawCandidate || typeof rawCandidate !== "object") continue;
    const candidate = rawCandidate as Record<string, unknown>;
    const id = sanitizeIdeaId(String(candidate.id ?? ""), 48);
    if (!id || seen.has(id)) continue;
    seen.add(id);
    candidates.push({
      id,
      label: trimTo(String(candidate.label ?? "").trim(), 80) || id,
      kind: normalizeIdeaNodeKind(candidate.kind),
      aliases: normalizeIdeaRefs(candidate.aliases, 6).slice(0, 6),
      candidateRefs: normalizeIdeaRefs(candidate.candidateRefs, IDEA_GRAPH_ATTACH_MAX_REFS).filter((ref) => evidenceIds.has(ref)),
      notes: trimTo(String(candidate.notes ?? "").trim(), 240),
    });
  }
  return candidates;
}

function normalizeIdeaGraphAttachmentCandidates(
  value: unknown,
  allowedIds: Set<string>,
  evidenceIds: Set<string>
): IdeaGraphAttachedCandidate[] {
  if (!Array.isArray(value)) throw new Error("Idea graph attachment response is missing kept[].");
  const seen = new Set<string>();
  const kept: IdeaGraphAttachedCandidate[] = [];
  for (const rawCandidate of value) {
    if (!rawCandidate || typeof rawCandidate !== "object") continue;
    const candidate = rawCandidate as Record<string, unknown>;
    const id = sanitizeIdeaId(String(candidate.id ?? ""), 48);
    if (!id || seen.has(id) || !allowedIds.has(id)) continue;
    seen.add(id);
    kept.push({
      id,
      label: trimTo(String(candidate.label ?? "").trim(), 80) || id,
      kind: normalizeIdeaNodeKind(candidate.kind),
      aliases: normalizeIdeaRefs(candidate.aliases, 6).slice(0, 6),
      summary: trimTo(String(candidate.summary ?? "").trim(), 160),
      detail: trimTo(String(candidate.detail ?? "").trim(), 420),
      coverageNote: trimTo(String(candidate.coverageNote ?? "").trim(), 220),
      evidenceRefs: normalizeIdeaRefs(candidate.evidenceRefs, IDEA_GRAPH_ATTACH_MAX_REFS).filter((ref) => evidenceIds.has(ref)),
      contradictionRefs: normalizeIdeaRefs(candidate.contradictionRefs, 6).filter((ref) => evidenceIds.has(ref)),
    });
  }
  return kept;
}

function normalizeIdeaGraphDroppedCandidates(
  value: unknown,
  allowedIds: Set<string>,
  evidenceIds: Set<string>
): IdeaGraphDroppedCandidate[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  const dropped: IdeaGraphDroppedCandidate[] = [];
  for (const rawCandidate of value) {
    if (!rawCandidate || typeof rawCandidate !== "object") continue;
    const candidate = rawCandidate as Record<string, unknown>;
    const id = sanitizeIdeaId(String(candidate.id ?? ""), 48);
    if (!id || seen.has(id) || !allowedIds.has(id)) continue;
    seen.add(id);
    dropped.push({
      id,
      label: trimTo(String(candidate.label ?? "").trim(), 80) || id,
      reason: trimTo(String(candidate.reason ?? "").trim(), 240),
      candidateRefs: normalizeIdeaRefs(candidate.candidateRefs, IDEA_GRAPH_ATTACH_MAX_REFS).filter((ref) => evidenceIds.has(ref)),
    });
  }
  return dropped;
}

function normalizeIdeaGraphEdges(value: unknown): IdeaGraphEdgeModel[] {
  if (!Array.isArray(value)) return [];
  const edges: IdeaGraphEdgeModel[] = [];
  for (const rawEdge of value) {
    if (!rawEdge || typeof rawEdge !== "object") continue;
    const edge = rawEdge as Record<string, unknown>;
    const fromId = sanitizeIdeaId(String(edge.fromId ?? ""), 48);
    const toId = sanitizeIdeaId(String(edge.toId ?? ""), 48);
    if (!fromId || !toId || fromId === toId) continue;
    edges.push({
      fromId,
      toId,
      relationPhrase: trimTo(String(edge.relationPhrase ?? "").trim(), 48),
      reasoning: trimTo(String(edge.reasoning ?? "").trim(), 260),
      evidenceRefs: normalizeIdeaRefs(edge.evidenceRefs, IDEA_GRAPH_ATTACH_MAX_REFS),
    });
  }
  return edges;
}

function validateIdeaGraphHarvestResponse(
  raw: string,
  evidence: { docs: IdeaGraphEvidenceDoc[]; snippets: IdeaGraphEvidenceSnippet[]; runFamilies: IdeaGraphRunFamily[] }
): IdeaGraphHarvestCandidate[] {
  const evidenceIds = new Set<string>([
    ...evidence.snippets.map((snippet) => snippet.id),
    ...evidence.runFamilies.map((family) => family.id),
  ]);
  let parsed: { candidates?: unknown };
  try {
    parsed = JSON.parse(extractJsonObject(raw)) as { candidates?: unknown };
  } catch (error) {
    throw new Error(`Idea graph harvest JSON parse failed: ${error instanceof Error ? error.message : String(error)}`);
  }
  const candidates = normalizeIdeaGraphHarvestCandidates(parsed.candidates, evidenceIds)
    .filter((candidate) => candidate.label.length > 0 && candidate.candidateRefs.length > 0 && candidate.notes.length > 0)
    .map((candidate) => ({
      ...candidate,
      aliases: uniq(candidate.aliases.filter((alias) => alias.toLowerCase() !== candidate.label.toLowerCase())).slice(0, 6),
    }));
  if (candidates.length === 0) throw new Error("Idea graph harvest returned no valid candidates.");
  return candidates;
}

function validateIdeaGraphAttachmentResponse(
  raw: string,
  harvested: IdeaGraphHarvestCandidate[],
  evidence: { snippets: IdeaGraphEvidenceSnippet[]; runFamilies: IdeaGraphRunFamily[] }
): {
  kept: IdeaGraphAttachedCandidate[];
  dropped: IdeaGraphDroppedCandidate[];
} {
  const allowedIds = new Set(harvested.map((candidate) => candidate.id));
  const evidenceIds = new Set<string>([
    ...evidence.snippets.map((snippet) => snippet.id),
    ...evidence.runFamilies.map((family) => family.id),
  ]);
  let parsed: { kept?: unknown; dropped?: unknown };
  try {
    parsed = JSON.parse(extractJsonObject(raw)) as { kept?: unknown; dropped?: unknown };
  } catch (error) {
    throw new Error(`Idea graph attachment JSON parse failed: ${error instanceof Error ? error.message : String(error)}`);
  }
  const kept = normalizeIdeaGraphAttachmentCandidates(parsed.kept, allowedIds, evidenceIds)
    .filter((candidate) => candidate.label.length > 0 && candidate.summary.length > 0 && candidate.detail.length > 0 && candidate.coverageNote.length > 0 && candidate.evidenceRefs.length > 0)
    .map((candidate) => ({
      ...candidate,
      aliases: uniq(candidate.aliases.filter((alias) => alias.toLowerCase() !== candidate.label.toLowerCase())).slice(0, 6),
    }));
  if (kept.length < 4) throw new Error("Idea graph attachment kept too few evidence-backed candidates.");
  const dropped = normalizeIdeaGraphDroppedCandidates(parsed.dropped, allowedIds, evidenceIds).filter((candidate) => candidate.reason.length > 0);
  return { kept, dropped };
}

function validateIdeaGraphSynthesisResponse(
  raw: string,
  attached: IdeaGraphAttachedCandidate[],
  evidence: { snippets: IdeaGraphEvidenceSnippet[]; runFamilies: IdeaGraphRunFamily[] }
): { anchors: string[]; edges: IdeaTreeEdge[] } {
  const nodesById = new Map(attached.map((candidate) => [candidate.id, candidate] as const));
  const evidenceIds = new Set<string>([
    ...evidence.snippets.map((snippet) => snippet.id),
    ...evidence.runFamilies.map((family) => family.id),
  ]);
  let parsed: { anchors?: unknown; edges?: unknown };
  try {
    parsed = JSON.parse(extractJsonObject(raw)) as { anchors?: unknown; edges?: unknown };
  } catch (error) {
    throw new Error(`Idea graph synthesis JSON parse failed: ${error instanceof Error ? error.message : String(error)}`);
  }
  const anchors = Array.isArray(parsed.anchors)
    ? uniq(
        parsed.anchors
          .map((anchor) => sanitizeIdeaId(String(anchor ?? ""), 48))
          .filter((anchor) => Boolean(anchor) && nodesById.has(anchor))
      )
    : [];
  if (anchors.length === 0) throw new Error("Idea graph synthesis returned no valid anchors.");
  if (anchors.length > IDEA_TREE_MAX_ANCHORS) throw new Error(`Idea graph returned too many anchors (max ${IDEA_TREE_MAX_ANCHORS}).`);
  const edges = normalizeIdeaGraphEdges(parsed.edges);
  const dedupedEdges: IdeaTreeEdge[] = [];
  const seenEdges = new Set<string>();
  for (const edge of edges) {
    if (!nodesById.has(edge.fromId) || !nodesById.has(edge.toId)) continue;
    const evidenceRefs = edge.evidenceRefs.filter((ref) => evidenceIds.has(ref));
    if (!edge.relationPhrase || !edge.reasoning || evidenceRefs.length === 0) continue;
    const key = `${edge.fromId}|${edge.relationPhrase}|${edge.toId}`;
    if (seenEdges.has(key)) continue;
    seenEdges.add(key);
    dedupedEdges.push({
      id: `${edge.fromId}__${edge.toId}__${edge.relationPhrase.replace(/[^A-Za-z0-9]+/g, "_").slice(0, 24)}`,
      fromId: edge.fromId,
      toId: edge.toId,
      relationPhrase: edge.relationPhrase,
      reasoning: edge.reasoning,
      evidenceRefs,
    });
  }
  if (dedupedEdges.length === 0) throw new Error("Idea graph synthesis returned no valid semantic edges.");
  return { anchors, edges: dedupedEdges };
}

function buildIdeaTreeResponse(
  task: TaskContext,
  bootstrap: ReturnType<typeof getBootstrap>,
  evidence: { docs: IdeaGraphEvidenceDoc[]; snippets: IdeaGraphEvidenceSnippet[]; runFamilies: IdeaGraphRunFamily[] },
  harvested: IdeaGraphHarvestCandidate[],
  attached: { kept: IdeaGraphAttachedCandidate[]; dropped: IdeaGraphDroppedCandidate[] },
  graph: { anchors: string[]; edges: IdeaTreeEdge[] },
  timings: IdeaGraphDebug["timings"]
): IdeaTreeResponse {
  const nodes = attached.kept.map((candidate) => ({
    id: candidate.id,
    label: candidate.label,
    kind: candidate.kind,
    summary: candidate.summary,
    detail: candidate.detail,
    aliases: candidate.aliases,
    evidenceRefs: candidate.evidenceRefs,
  }));
  return {
    taskId: task.id,
    generatedAt: new Date().toISOString(),
    anchors: graph.anchors,
    inputs: {
      snapshotGeneratedAt: bootstrap.generatedAt,
      docs: evidence.docs.map((doc) => doc.file),
      snippetCount: evidence.snippets.length,
      runFamilyCount: evidence.runFamilies.length,
      pipelineVersion: "semantic_graph_v3",
    },
    graph: {
      nodes,
      edges: graph.edges,
    },
    debug: {
      timings,
      evidence,
      harvest: {
        candidates: harvested,
      },
      attachment: attached,
    },
  };
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
    const evidence = parseEvidenceFromAnswer(answer);
    const response: TaskQaResponse = {
      taskId: task.id,
      answer,
      evidence,
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

async function handleIdeaTree(req: Request, fallbackTask: TaskContext | null): Promise<Response> {
  let payload: IdeaTreeRequest;
  try {
    payload = (await req.json()) as IdeaTreeRequest;
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body." }), {
      status: 400,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
  const task = resolveTaskContext(payload.taskId ?? fallbackTask?.id ?? null);
  if (!task) {
    return new Response(JSON.stringify({ error: "Unknown task." }), {
      status: 404,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }

  try {
    setIdeaTreeProgress(task.id, {
      state: "running",
      activeStage: "harvest",
      message: "Harvesting candidates from the task corpus.",
      startedAt: new Date().toISOString(),
    });
    const totalStart = Date.now();
    const bootstrap = getBootstrap(task);

    const evidenceStart = Date.now();
    const evidence = buildIdeaGraphEvidencePack(task, bootstrap);
    const evidenceMs = Date.now() - evidenceStart;

    const harvestStart = Date.now();
    setIdeaTreeProgress(task.id, {
      state: "running",
      activeStage: "harvest",
      message: "Harvesting candidate ideas and variants.",
    });
    const harvestPrompt = formatIdeaGraphEvidencePackForHarvestPrompt(task, bootstrap, evidence);
    const harvestRaw = await runCodexIdeaGraphStage(harvestPrompt, {
      stage: "harvest",
      timeoutMs: IDEA_GRAPH_HARVEST_TIMEOUT_MS,
      outputSchema: IDEA_GRAPH_HARVEST_OUTPUT_SCHEMA as Record<string, unknown>,
    });
    const harvested = validateIdeaGraphHarvestResponse(harvestRaw, evidence);
    const harvestMs = Date.now() - harvestStart;

    const lookup = buildIdeaGraphEvidenceLookup(evidence);
    const evidenceRefsByCandidate = new Map(
      harvested.map((candidate) => [candidate.id, collectCandidateEvidenceRefs(candidate, evidence, lookup)] as const)
    );

    const attachmentStart = Date.now();
    setIdeaTreeProgress(task.id, {
      state: "running",
      activeStage: "attachment",
      message: "Attaching evidence to harvested candidates.",
    });
    const attachmentPrompt = buildIdeaGraphAttachmentPrompt(task, harvested, evidenceRefsByCandidate, lookup);
    const attachmentRaw = await runCodexIdeaGraphStage(attachmentPrompt, {
      stage: "attachment",
      timeoutMs: IDEA_GRAPH_ATTACHMENT_TIMEOUT_MS,
      outputSchema: IDEA_GRAPH_ATTACHMENT_OUTPUT_SCHEMA as Record<string, unknown>,
    });
    const attached = validateIdeaGraphAttachmentResponse(attachmentRaw, harvested, evidence);
    const attachmentMs = Date.now() - attachmentStart;

    const graphStart = Date.now();
    setIdeaTreeProgress(task.id, {
      state: "running",
      activeStage: "synthesis",
      message: "Synthesizing the final graph structure.",
    });
    const synthesisPrompt = buildIdeaGraphSynthesisPrompt(task, attached.kept, lookup);
    const graphRaw = await runCodexIdeaGraphStage(synthesisPrompt, {
      stage: "synthesis",
      timeoutMs: IDEA_GRAPH_SYNTHESIS_TIMEOUT_MS,
      outputSchema: IDEA_GRAPH_SYNTHESIS_OUTPUT_SCHEMA as Record<string, unknown>,
    });
    const graph = validateIdeaGraphSynthesisResponse(graphRaw, attached.kept, evidence);
    const graphMs = Date.now() - graphStart;

    const response = buildIdeaTreeResponse(task, bootstrap, evidence, harvested, attached, graph, {
      evidenceMs,
      harvestMs,
      attachmentMs,
      graphMs,
      totalMs: Date.now() - totalStart,
    });
    setIdeaTreeProgress(task.id, {
      state: "completed",
      activeStage: "synthesis",
      message: "Graph generation complete.",
    });
    return new Response(JSON.stringify(response), {
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  } catch (error) {
    setIdeaTreeProgress(task.id, {
      state: "error",
      message: error instanceof Error ? error.message : String(error),
    });
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
    if (url.pathname === "/api/ideas/tree") {
      if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });
      return handleIdeaTree(req, task);
    }
    if (url.pathname === "/api/ideas/tree/status") {
      if (!task) return new Response("Unknown task", { status: 404 });
      const status =
        ideaTreeProgressByTask.get(task.id) ??
        ({
          taskId: task.id,
          state: "idle",
          activeStage: null,
          message: "Not started.",
          startedAt: null,
          updatedAt: new Date().toISOString(),
        } satisfies IdeaTreeProgressStatus);
      return new Response(JSON.stringify(status), {
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }
    return new Response("Not found", { status: 404 });
  },
});

console.log(
  `[research-tracker] http://localhost:${port} task=${defaultTask.id} tasks_root=${path.relative(repoRoot, tasksRoot)}`
);
