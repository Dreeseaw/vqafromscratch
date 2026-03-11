import { serve } from "bun";
import fs from "fs";
import path from "path";

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

type SeriesPoint = {
  step: number;
  value: number;
};

type RunDetail = {
  run: RunSummary;
  updatedAt: string | null;
  introLines: string[];
  trainCeSeries: SeriesPoint[];
  valAccuracySeries: SeriesPoint[];
  valCeSeries: SeriesPoint[];
  latestEvalAccuracy: number | null;
  latestValCe: number | null;
};

function usageAndExit(): never {
  console.error(
    "Usage: bun run tracker/research/researchtrackerapp.ts [-p <port>] [--docs <docs_dir>] [--logs <logs_dir>]"
  );
  process.exit(1);
}

function toIso(ms: number): string {
  return new Date(ms).toISOString();
}

function uniq<T>(xs: T[]): T[] {
  return [...new Set(xs)];
}

function parseArgs() {
  const args = process.argv.slice(2);
  const portIdx = args.indexOf("-p");
  const docsIdx = args.indexOf("--docs");
  const logsIdx = args.indexOf("--logs");
  const port = portIdx !== -1 ? Number(args[portIdx + 1]) : 4090;
  if (!Number.isInteger(port) || port <= 0) usageAndExit();
  const docsRel = docsIdx !== -1 ? String(args[docsIdx + 1] ?? "").trim() : "docs/mm_bridge_diagnostics";
  const logsRel = logsIdx !== -1 ? String(args[logsIdx + 1] ?? "").trim() : "logs";
  if (!docsRel || !logsRel) usageAndExit();
  return { port, docsRel, logsRel };
}

const { port, docsRel, logsRel } = parseArgs();
const staticRoot = import.meta.dir;
const repoRoot = path.resolve(staticRoot, "..", "..");
const docsRoot = path.resolve(repoRoot, docsRel);
const logsRoot = path.resolve(repoRoot, logsRel);
const ACTIVE_WINDOW_MS = 45 * 60 * 1000;
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

function getLatestRunLogfilePath(runId: string): string | null {
  const runDir = path.join(logsRoot, runId);
  if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) return null;
  const cand = fs
    .readdirSync(runDir)
    .filter((f) => /^logfile.*\.txt$/i.test(f))
    .map((f) => path.join(runDir, f))
    .sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs);
  return cand.length > 0 ? cand[cand.length - 1] : null;
}

function parseRunSummary(runId: string): RunSummary {
  const runDir = path.join(logsRoot, runId);
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

  const logfile = getLatestRunLogfilePath(runId);
  if (!logfile) return out;
  out.logfile = path.basename(logfile);
  out.logfileMtimeMs = fs.statSync(logfile).mtimeMs;
  const text = readText(logfile);
  if (!text) return out;

  const accVals = [...text.matchAll(/overall_accuracy=([0-9]*\.[0-9]+)/g)].map((m) => Number(m[1]));
  if (accVals.length > 0) {
    out.finalAccuracy = accVals[accVals.length - 1];
    out.bestAccuracy = Math.max(...accVals);
  }

  for (const line of text.split(/\r?\n/)) {
    const mmStep = line.match(/\[mm\]\s+step=(\d+)/);
    const mmStepsPerSec = line.match(/\bsteps_per_s=([0-9]*\.?[0-9]+)/);
    if (mmStep && mmStepsPerSec) {
      const lossCe = line.match(/\bloss_ce=([0-9]*\.?[0-9]+)/);
      const loss = line.match(/\bloss=([0-9]*\.?[0-9]+)/);
      out.lastStep = Number(mmStep[1]);
      out.lastTrainCe = Number((lossCe ?? loss)?.[1] ?? NaN);
      out.lastStepsPerSec = Number(mmStepsPerSec[1]);
      continue;
    }

    const lmStep = line.match(/\bStep:\s*(\d+),/);
    const lmTrainCe = line.match(/\bTrain CE:\s*([0-9]*\.?[0-9]+)/);
    const lmTokensPerSec = line.match(/\bTokens\/s:\s*([0-9]*\.?[0-9]+)/);
    if (lmStep && lmTrainCe && lmTokensPerSec) {
      out.lastStep = Number(lmStep[1]);
      out.lastTrainCe = Number(lmTrainCe[1]);
      out.lastStepsPerSec = Number(lmTokensPerSec[1]);
    }
  }

  const trainableParamsMatch = text.match(/\btrainable_params=([\d,]+)/i);
  if (trainableParamsMatch) out.trainableParams = Number(trainableParamsMatch[1].replaceAll(",", ""));
  const totalParamsMatch =
    text.match(/\btotal_params=([\d,]+)/i) ?? text.match(/\bTotal params:\s*([\d,]+)/i);
  if (totalParamsMatch) out.numParams = Number(totalParamsMatch[1].replaceAll(",", ""));
  out.hasFinalCheckpoint = /\[mm\]\s+final checkpoint:\s+/m.test(text);
  return out;
}

function parseRunDetail(runId: string): RunDetail | null {
  const run = parseRunSummary(runId);
  if (run.lastStep === null) return null;
  const logfile = getLatestRunLogfilePath(runId);
  if (!logfile) return null;
  const text = readText(logfile);
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const trainCeSeries: SeriesPoint[] = [];
  const valAccuracySeries: SeriesPoint[] = [];
  const valCeSeries: SeriesPoint[] = [];
  let lastStepSeen: number | null = null;

  for (const line of lines) {
    const mmStep = line.match(/\[mm\]\s+step=(\d+)/);
    const mmLossCe = line.match(/\bloss_ce=([0-9]*\.?[0-9]+)/);
    const mmLoss = line.match(/\bloss=([0-9]*\.?[0-9]+)/);
    if (mmStep) {
      lastStepSeen = Number(mmStep[1]);
      const ce = Number((mmLossCe ?? mmLoss)?.[1] ?? NaN);
      if (Number.isFinite(ce)) trainCeSeries.push({ step: lastStepSeen, value: ce });
      continue;
    }

    const lmStep = line.match(/\bStep:\s*(\d+),/);
    const lmTrainCe = line.match(/\bTrain CE:\s*([0-9]*\.?[0-9]+)/);
    if (lmStep && lmTrainCe) {
      lastStepSeen = Number(lmStep[1]);
      trainCeSeries.push({ step: lastStepSeen, value: Number(lmTrainCe[1]) });
      const valCeMatch = line.match(/Validation Step=(\d+)\s+CE=([0-9]*\.?[0-9]+)/);
      if (valCeMatch) valCeSeries.push({ step: Number(valCeMatch[1]), value: Number(valCeMatch[2]) });
      continue;
    }

    const explicitValStep = line.match(/Validation Step=(\d+)\s+CE=([0-9]*\.?[0-9]+)/);
    if (explicitValStep) {
      valCeSeries.push({ step: Number(explicitValStep[1]), value: Number(explicitValStep[2]) });
      continue;
    }

    const valAccMatch = line.match(/\[eval:val\]\s+overall_accuracy=([0-9]*\.?[0-9]+)/);
    if (valAccMatch && lastStepSeen !== null) {
      const point = { step: lastStepSeen, value: Number(valAccMatch[1]) };
      const prev = valAccuracySeries[valAccuracySeries.length - 1];
      if (!prev || prev.step !== point.step || prev.value !== point.value) valAccuracySeries.push(point);
      continue;
    }

  }

  return {
    run,
    updatedAt: run.logfileMtimeMs ? toIso(run.logfileMtimeMs) : null,
    introLines: lines.slice(0, 20),
    trainCeSeries,
    valAccuracySeries,
    valCeSeries,
    latestEvalAccuracy: valAccuracySeries.length > 0 ? valAccuracySeries[valAccuracySeries.length - 1].value : null,
    latestValCe: valCeSeries.length > 0 ? valCeSeries[valCeSeries.length - 1].value : null,
  };
}

function parseTimelineLineDate(line: string): string | null {
  const m = line.match(/^\[([^\]]+)\]/);
  return m ? m[1] : null;
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

function parseSweep(sweepDirName: string, isSymlink: boolean, mtimeMs: number): ParsedSweepSummary | null {
  const sweepDir = path.join(logsRoot, sweepDirName);
  const timeline = path.join(sweepDir, "timeline.log");
  if (!fs.existsSync(timeline) || !fs.statSync(sweepDir).isDirectory()) return null;
  const lines = readText(timeline).split("\n").filter((x) => x.trim().length > 0);
  const runIds = uniq(
    lines
      .map((line) => {
        const m = line.match(/\bSTART\s+([A-Za-z0-9._-]+)/);
        return m ? m[1] : null;
      })
      .filter((x): x is string => Boolean(x))
  );

  const activeRunIds = new Set<string>();
  for (const line of lines) {
    const startMatch = line.match(/\bSTART\s+([A-Za-z0-9._-]+)/);
    if (startMatch) {
      activeRunIds.add(startMatch[1]);
      continue;
    }
    const endMatch = line.match(/\bEND\s+([A-Za-z0-9._-]+)/);
    if (endMatch) activeRunIds.delete(endMatch[1]);
  }
  const completionLine =
    [...lines].reverse().find((line) => /\b(?:SWEEP|PROBES)\s+COMPLETE\b/.test(line)) ?? null;
  const terminalLine =
    completionLine ??
    [...lines].reverse().find((line) => /\b(?:END|STOP)\b/.test(line)) ??
    null;
  const parsedRuns = runIds
    .map((rid) => {
      const run = parseRunSummary(rid);
      run.isActive =
        activeRunIds.has(rid) &&
        !completionLine &&
        Number.isFinite(run.logfileMtimeMs ?? NaN) &&
        Date.now() - (run.logfileMtimeMs as number) <= ACTIVE_WINDOW_MS;
      return run;
    });
  const activeRuns = parsedRuns.filter((run) => run.isActive).length;
  const runs = parsedRuns.filter((run) => run.lastStep !== null);
  const allAcc = runs.map((r) => r.bestAccuracy).filter((x): x is number => Number.isFinite(x));
  const bestAccuracy = allAcc.length > 0 ? Math.max(...allAcc) : null;
  const ceVals = runs.map((r) => r.lastTrainCe).filter((x): x is number => Number.isFinite(x));
  const lastTrainCe = ceVals.length > 0 ? Math.min(...ceVals) : null;
  const startLine = lines.find((line) => /\bSTART\b/.test(line)) ?? lines[0] ?? "";

  return {
    sweepId: sweepDirName,
    sweepDir,
    status: activeRuns > 0 ? "running" : "completed",
    startedAt: parseTimelineLineDate(startLine),
    endedAt: activeRuns > 0 ? null : parseTimelineLineDate(terminalLine ?? ""),
    runs,
    bestAccuracy,
    lastTrainCe,
    activeRuns,
    isSymlink,
    mtimeMs,
  };
}

function listSweeps(): SweepSummary[] {
  if (!fs.existsSync(logsRoot) || !fs.statSync(logsRoot).isDirectory()) return [];
  const dirs = fs
    .readdirSync(logsRoot)
    .map((name) => {
      const full = path.join(logsRoot, name);
      if (!fs.existsSync(full) || !fs.statSync(full).isDirectory() || !fs.existsSync(path.join(full, "timeline.log"))) {
        return null;
      }
      const lst = fs.lstatSync(full);
      return {
        name,
        isSymlink: lst.isSymbolicLink(),
        mtimeMs: fs.statSync(full).mtimeMs,
      };
    })
    .filter((entry): entry is { name: string; isSymlink: boolean; mtimeMs: number } => entry !== null)
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
  const sweeps = dirs
    .map((entry) => parseSweep(entry.name, entry.isSymlink, entry.mtimeMs))
    .filter((x): x is ParsedSweepSummary => x !== null);
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
      const allAcc = runs.map((r) => r.bestAccuracy).filter((x): x is number => Number.isFinite(x));
      const ceVals = runs.map((r) => r.lastTrainCe).filter((x): x is number => Number.isFinite(x));
      const representative = group.slice().sort((a, b) => b.runs.length - a.runs.length || b.mtimeMs - a.mtimeMs)[0];
      const startedAt = group
        .map((s) => s.startedAt)
        .filter((x): x is string => Boolean(x))
        .sort()[0] ?? null;
      const endedAt = group
        .map((s) => s.endedAt)
        .filter((x): x is string => Boolean(x))
        .sort()
        .at(-1) ?? null;
      return {
        sweepId,
        sweepDir: representative.sweepDir,
        status: group.some((s) => s.activeRuns > 0) ? "running" : "completed",
        startedAt,
        endedAt,
        runs,
        bestAccuracy: allAcc.length > 0 ? Math.max(...allAcc) : null,
        lastTrainCe: ceVals.length > 0 ? Math.min(...ceVals) : null,
        activeRuns: runs.filter((run) => run.isActive).length,
        mtimeMs: Math.max(...group.map((s) => s.mtimeMs)),
      };
    })
    .filter((sweep) => sweep.runs.length > 1 || sweep.activeRuns > 0)
    .sort((a, b) => b.mtimeMs - a.mtimeMs)
    .map(({ mtimeMs: _mtimeMs, ...sweep }) => sweep);
}

function parseDoc(fileName: string): DocSummary {
  const full = path.join(docsRoot, fileName);
  const txt = readText(full);
  const titleMatch = txt.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : fileName;
  const logsRefs = [...txt.matchAll(/logs\/([A-Za-z0-9._-]+)/g)].map((m) => m[1]);
  const inlineRefs = [...txt.matchAll(/`([A-Za-z0-9._-]{6,})`/g)]
    .map((m) => m[1])
    .filter((rid) => fs.existsSync(path.join(logsRoot, rid)));
  const runRefs = uniq([...logsRefs, ...inlineRefs]);
  const mentionedAccuracies = uniq(
    [...txt.matchAll(/\b0\.\d{3,4}\b/g)]
      .map((m) => Number(m[0]))
      .filter((x) => Number.isFinite(x) && x >= 0.1)
  ).sort((a, b) => b - a);
  return {
    file: fileName,
    title,
    updatedAt: toIso(fs.statSync(full).mtimeMs),
    runRefs,
    mentionedAccuracies,
  };
}

function listDocs(): DocSummary[] {
  if (!fs.existsSync(docsRoot) || !fs.statSync(docsRoot).isDirectory()) return [];
  return fs
    .readdirSync(docsRoot)
    .filter((f) => f.toLowerCase().endsWith(".md"))
    .sort((a, b) => fs.statSync(path.join(docsRoot, b)).mtimeMs - fs.statSync(path.join(docsRoot, a)).mtimeMs)
    .map((f) => parseDoc(f));
}

function getBootstrap() {
  const sweeps = listSweeps();
  const docs = listDocs();
  const runMap = new Map<string, RunSummary>();
  for (const s of sweeps) {
    for (const r of s.runs) runMap.set(r.runId, r);
  }
  for (const d of docs) {
    for (const rid of d.runRefs) {
      if (runMap.has(rid)) continue;
      const run = parseRunSummary(rid);
      if (run.lastStep !== null) runMap.set(rid, run);
    }
  }
  const allRuns = [...runMap.values()];
  const accRuns = allRuns.filter((r) => Number.isFinite(r.finalAccuracy ?? NaN));
  const bestRun = accRuns
    .slice()
    .sort((a, b) => (b.finalAccuracy ?? -1) - (a.finalAccuracy ?? -1))[0] ?? null;

  return {
    generatedAt: new Date().toISOString(),
    repoRoot,
    docsRoot,
    logsRoot,
    docs,
    sweeps,
    runs: allRuns,
    summary: {
      docsCount: docs.length,
      sweepsCount: sweeps.length,
      runsCount: allRuns.length,
      runsWithAccuracy: accRuns.length,
      bestRun: bestRun
        ? { runId: bestRun.runId, finalAccuracy: bestRun.finalAccuracy, bestAccuracy: bestRun.bestAccuracy }
        : null,
    },
  };
}

function transpileAppTs(): string {
  const src = readText(appTsPath);
  const t = new Bun.Transpiler({ loader: "ts", target: "browser" });
  return t.transformSync(src);
}

function transpileDocTs(): string {
  const src = readText(docTsPath);
  const t = new Bun.Transpiler({ loader: "ts", target: "browser" });
  return t.transformSync(src);
}

function resolveDocPath(fileName: string | null): string | null {
  const name = String(fileName ?? "").trim();
  if (!name) return null;
  const full = path.resolve(docsRoot, name);
  if (!full.startsWith(docsRoot)) return null;
  if (!fs.existsSync(full) || !fs.statSync(full).isFile()) return null;
  return full;
}

function resolveRunId(runId: string | null): string | null {
  const name = String(runId ?? "").trim();
  if (!name) return null;
  if (!/^[A-Za-z0-9._-]+$/.test(name)) return null;
  const full = path.join(logsRoot, name);
  if (!fs.existsSync(full) || !fs.statSync(full).isDirectory()) return null;
  return name;
}

serve({
  port,
  fetch(req) {
    const url = new URL(req.url);
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
      return new Response(JSON.stringify(getBootstrap()), {
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }
    if (url.pathname === "/api/doc") {
      const full = resolveDocPath(url.searchParams.get("file"));
      if (!full) return new Response("Not found", { status: 404 });
      return new Response(readText(full), {
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    if (url.pathname === "/api/run") {
      const runId = resolveRunId(url.searchParams.get("runId"));
      if (!runId) return new Response("Not found", { status: 404 });
      const detail = parseRunDetail(runId);
      if (!detail) return new Response("Not found", { status: 404 });
      return new Response(JSON.stringify(detail), {
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }
    return new Response("Not found", { status: 404 });
  },
});

console.log(
  `[research-tracker] http://localhost:${port} docs=${path.relative(repoRoot, docsRoot)} logs=${path.relative(repoRoot, logsRoot)}`
);
