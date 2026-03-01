import { serve } from "bun";
import fs from "fs";
import path from "path";

function usageAndExit() {
  console.error(
    "Usage: bun run tracker/lmtrackerapp.ts -f <run_dir> [-p <port>] [--log <logfile_name>] [--continued] [--compare-log <logfile_name_or_path>] [--tokenizer <tokenizer.pt>] [--attn-grid-cache <entries>] [--attn-prewarm <count|all>]"
  );
  process.exit(1);
}

const args = process.argv.slice(2);
const hasContinuedMode = args.includes("--continued");
const runDirIdx = args.indexOf("-f");
if (runDirIdx === -1 || !args[runDirIdx + 1]) usageAndExit();

const runDir = path.resolve(args[runDirIdx + 1]);
if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) {
  console.error(`Run directory does not exist: ${runDir}`);
  process.exit(1);
}

const tokenizerIdx = args.indexOf("--tokenizer");
const tokenizerArg = tokenizerIdx !== -1 ? args[tokenizerIdx + 1] : "";
if (tokenizerIdx !== -1 && !tokenizerArg) usageAndExit();
const explicitTokenizerPath = tokenizerArg ? path.resolve(tokenizerArg) : "";
if (explicitTokenizerPath) {
  if (!fs.existsSync(explicitTokenizerPath) || !fs.statSync(explicitTokenizerPath).isFile()) {
    console.error(`Tokenizer file does not exist: ${explicitTokenizerPath}`);
    process.exit(1);
  }
}

const explicitLogIdx = args.indexOf("--log");
if (hasContinuedMode && explicitLogIdx !== -1) {
  console.error("Use either --log or --continued, not both.");
  process.exit(1);
}

type RunLogInfo = {
  file: string;
  fullPath: string;
  kind: "base" | "continued" | "other";
  checkpoint: number;
  mtimeMs: number;
};

function listRunLogfiles(dir: string): RunLogInfo[] {
  const out: RunLogInfo[] = [];
  for (const file of fs.readdirSync(dir)) {
    if (!/^logfile.*\.txt$/i.test(file)) continue;
    const fullPath = path.join(dir, file);
    if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) continue;
    const st = fs.statSync(fullPath);
    if (/^logfile\.txt$/i.test(file)) {
      out.push({ file, fullPath, kind: "base", checkpoint: -1, mtimeMs: st.mtimeMs });
      continue;
    }
    const m = file.match(/^logfile_from_(\d+)\.txt$/i);
    if (m) {
      out.push({
        file,
        fullPath,
        kind: "continued",
        checkpoint: Number(m[1]),
        mtimeMs: st.mtimeMs,
      });
      continue;
    }
    out.push({ file, fullPath, kind: "other", checkpoint: Number.MAX_SAFE_INTEGER, mtimeMs: st.mtimeMs });
  }
  return out;
}

const allRunLogs = listRunLogfiles(runDir);
const baseLogPath = path.join(runDir, "logfile.txt");

let streamLogs: RunLogInfo[] = [];
if (hasContinuedMode) {
  streamLogs = allRunLogs
    .filter((x) => x.kind === "continued")
    .sort((a, b) => a.checkpoint - b.checkpoint || a.mtimeMs - b.mtimeMs || a.file.localeCompare(b.file));
  if (streamLogs.length === 0) {
    console.error(`No continuation logs found in ${runDir} (expected logfile_from_<step>.txt).`);
    process.exit(1);
  }
} else if (explicitLogIdx !== -1) {
  const name = args[explicitLogIdx + 1];
  if (!name) usageAndExit();
  const fullPath = path.isAbsolute(name) ? path.resolve(name) : path.resolve(runDir, name);
  if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
    console.error(`Missing logfile: ${fullPath}`);
    process.exit(1);
  }
  const st = fs.statSync(fullPath);
  const baseName = path.basename(fullPath);
  const contMatch = baseName.match(/^logfile_from_(\d+)\.txt$/i);
  streamLogs = [
    {
      file: baseName,
      fullPath,
      kind: /^logfile\.txt$/i.test(baseName) ? "base" : contMatch ? "continued" : "other",
      checkpoint: contMatch ? Number(contMatch[1]) : Number.MAX_SAFE_INTEGER,
      mtimeMs: st.mtimeMs,
    },
  ];
} else {
  const preferred = allRunLogs.find((x) => x.kind === "base");
  if (preferred) {
    streamLogs = [preferred];
  } else {
    const candidates = [...allRunLogs].sort((a, b) => a.file.localeCompare(b.file));
    if (candidates.length > 0) streamLogs = [candidates[candidates.length - 1]];
  }
}

if (streamLogs.length === 0) {
  console.error(`No logfile found in ${runDir}. Expected logfile.txt or logfile*.txt`);
  process.exit(1);
}
const logfile = streamLogs[streamLogs.length - 1].fullPath;
const streamLogPaths = streamLogs.map((x) => x.fullPath);
const streamLogNames = streamLogs.map((x) => x.file);

const compareLogIdx = args.indexOf("--compare-log");
let baselineLogfile = "";
if (compareLogIdx !== -1) {
  const raw = args[compareLogIdx + 1];
  if (!raw) usageAndExit();
  baselineLogfile = path.isAbsolute(raw) ? path.resolve(raw) : path.resolve(runDir, raw);
  if (!fs.existsSync(baselineLogfile) || !fs.statSync(baselineLogfile).isFile()) {
    console.error(`Compare logfile does not exist: ${baselineLogfile}`);
    process.exit(1);
  }
} else if (hasContinuedMode && fs.existsSync(baseLogPath) && fs.statSync(baseLogPath).isFile()) {
  baselineLogfile = baseLogPath;
}
const probeDebugDir = path.join(runDir, "probe_debug");
const latestAttentionFile = path.join(probeDebugDir, "latest_attention.json");
const attentionIndexFile = path.join(probeDebugDir, "attention_index.json");

const DEFAULT_ATTN_GRID_CACHE_LIMIT = 1024;
const DEFAULT_ATTN_PREWARM_LIMIT = 256;

const attnGridCacheIdx = args.indexOf("--attn-grid-cache");
let attentionGridCacheLimit = DEFAULT_ATTN_GRID_CACHE_LIMIT;
if (attnGridCacheIdx !== -1) {
  const raw = Number(args[attnGridCacheIdx + 1]);
  if (!Number.isInteger(raw) || raw < 0) usageAndExit();
  attentionGridCacheLimit = raw;
}

const attnPrewarmIdx = args.indexOf("--attn-prewarm");
let attentionPrewarmLimit = DEFAULT_ATTN_PREWARM_LIMIT;
if (attnPrewarmIdx !== -1) {
  const raw = String(args[attnPrewarmIdx + 1] ?? "").trim().toLowerCase();
  if (!raw) usageAndExit();
  if (raw === "all") {
    attentionPrewarmLimit = Number.MAX_SAFE_INTEGER;
  } else {
    const n = Number(raw);
    if (!Number.isInteger(n) || n < 0) usageAndExit();
    attentionPrewarmLimit = n;
  }
}

const portIdx = args.indexOf("-p");
const port = portIdx !== -1 ? Number(args[portIdx + 1]) : 3030;
if (!Number.isInteger(port) || port <= 0) {
  console.error(`Invalid port: ${args[portIdx + 1]}`);
  process.exit(1);
}

const staticRoot = import.meta.dir;
const repoRoot = path.resolve(staticRoot, "..");
const htmlPath = path.join(staticRoot, "lm_index.html");
if (!fs.existsSync(htmlPath)) {
  console.error(`Missing frontend file: ${htmlPath}`);
  process.exit(1);
}

function readSlice(file: string, from: number, to: number): string {
  if (to <= from) return "";
  const fd = fs.openSync(file, "r");
  try {
    const len = to - from;
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, from);
    return buf.toString("utf-8");
  } finally {
    fs.closeSync(fd);
  }
}

function _sanitizePathToken(raw: string): string {
  return String(raw || "").trim().replace(/^['"]+|['"]+$/g, "");
}

function _readMetaTokenizer(metaPath: string): string | null {
  if (!fs.existsSync(metaPath) || !fs.statSync(metaPath).isFile()) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
    const tok = typeof raw?.tokenizer === "string" ? raw.tokenizer : "";
    if (!tok) return null;
    const tokAbsA = path.isAbsolute(tok) ? tok : path.resolve(repoRoot, tok);
    if (fs.existsSync(tokAbsA) && fs.statSync(tokAbsA).isFile()) return tokAbsA;
    const tokAbsB = path.resolve(path.dirname(metaPath), tok);
    if (fs.existsSync(tokAbsB) && fs.statSync(tokAbsB).isFile()) return tokAbsB;
  } catch {
    return null;
  }
  return null;
}

function _collectTokenizerFromDataRef(refRaw: string, out: Set<string>) {
  const ref = _sanitizePathToken(refRaw);
  if (!ref) return;
  const refs = new Set<string>();
  refs.add(path.isAbsolute(ref) ? ref : path.resolve(repoRoot, ref));
  refs.add(path.resolve(runDir, ref));

  for (const abs of refs) {
    if (!fs.existsSync(abs)) continue;
    const st = fs.statSync(abs);
    const metaCandidates: string[] = [];
    if (st.isDirectory()) {
      metaCandidates.push(path.join(abs, "meta.json"));
      metaCandidates.push(path.join(path.dirname(abs), "meta.json"));
    } else if (st.isFile()) {
      if (path.basename(abs).toLowerCase() === "meta.json") {
        metaCandidates.push(abs);
      } else {
        metaCandidates.push(path.join(path.dirname(abs), "meta.json"));
      }
    }
    for (const mp of metaCandidates) {
      const tok = _readMetaTokenizer(mp);
      if (tok) out.add(tok);
    }
  }
}

function inferTokenizerPathFromRun(): string | null {
  const out = new Set<string>();
  const localTok = path.join(runDir, "tokenizer.pt");
  if (fs.existsSync(localTok) && fs.statSync(localTok).isFile()) out.add(localTok);

  try {
    const logStat = fs.statSync(logfile);
    const head = readSlice(logfile, 0, Math.min(logStat.size, 128 * 1024));
    const trainMatch = head.match(/train_data:\s*(.+?)(?:val_data:|test_data:|\n|$)/i);
    if (trainMatch?.[1]) _collectTokenizerFromDataRef(trainMatch[1], out);
    const probeMatch = head.match(/probe_file=(.+?)(?:sanity:|\n|$)/i);
    if (probeMatch?.[1]) {
      const probeFile = _sanitizePathToken(probeMatch[1]);
      if (probeFile) _collectTokenizerFromDataRef(path.dirname(probeFile), out);
    }
  } catch {
    // no-op
  }

  // Last fallback: find a tokenizer in logs with matching vocab size from config.
  if (out.size === 0) {
    try {
      const head = readSlice(logfile, 0, Math.min(fs.statSync(logfile).size, 128 * 1024));
      const vocabMatch = head.match(/['"]vocab_size['"]:\s*(\d+)/);
      const vocab = vocabMatch ? Number(vocabMatch[1]) : null;
      if (Number.isFinite(vocab)) {
        const logsRoot = path.resolve(repoRoot, "logs");
        if (fs.existsSync(logsRoot) && fs.statSync(logsRoot).isDirectory()) {
          for (const d of fs.readdirSync(logsRoot)) {
            const info = path.join(logsRoot, d, "tokenizer_info.json");
            const tok = path.join(logsRoot, d, "tokenizer.pt");
            if (!fs.existsSync(info) || !fs.existsSync(tok)) continue;
            try {
              const payload = JSON.parse(fs.readFileSync(info, "utf-8"));
              if (Number(payload?.vocab_size) === Number(vocab)) out.add(tok);
            } catch {
              // no-op
            }
          }
        }
      }
    } catch {
      // no-op
    }
  }

  return out.size > 0 ? Array.from(out)[0] : null;
}

function pickPythonWithTorch(): string | null {
  const preferred = ["python", "python3.10", "python3", "python3.11", "python3.12"];
  for (const bin of preferred) {
    try {
      const probe = Bun.spawnSync({
        cmd: [bin, "-c", "import torch; print('ok')"],
        cwd: process.cwd(),
        stdout: "pipe",
        stderr: "pipe",
      });
      if (probe.exitCode === 0) return bin;
    } catch {
      // no-op
    }
  }
  return null;
}

function loadTokenizerTokenTable(tokenizerPath: string, pythonBin: string): string[] | null {
  const code = [
    "import json, sys",
    "repo = sys.argv[1]",
    "if repo not in sys.path: sys.path.insert(0, repo)",
    "from models.bpe_tokenizer import ByteBPETokenizer",
    "tok = ByteBPETokenizer.load(sys.argv[2])",
    "out = [tok.decode([i], skip_special=False) for i in range(tok.vocab_size)]",
    "print(json.dumps(out, ensure_ascii=False))",
  ].join("\n");
  try {
    const proc = Bun.spawnSync({
      cmd: [pythonBin, "-c", code, repoRoot, tokenizerPath],
      cwd: repoRoot,
      stdout: "pipe",
      stderr: "pipe",
    });
    if (proc.exitCode !== 0) {
      const err = Buffer.from(proc.stderr).toString("utf-8").trim();
      if (err) console.warn(`tracker: tokenizer decode table load failed: ${err}`);
      return null;
    }
    const raw = Buffer.from(proc.stdout).toString("utf-8");
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    return parsed.map((x: unknown) => String(x ?? ""));
  } catch (e) {
    console.warn(`tracker: tokenizer decode table load error: ${String(e)}`);
    return null;
  }
}

function _toIntList(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x))
    .map((x) => Math.trunc(x));
}

function _decodeIds(ids: number[], table: string[] | null): string[] {
  if (!table) {
    const pyDecoded = decodeIdsWithPython(ids);
    if (pyDecoded) return pyDecoded;
    return ids.map((id) => `<id:${id}>`);
  }
  return ids.map((id) => {
    if (id >= 0 && id < table.length) return String(table[id] ?? "");
    return `<id:${id}>`;
  });
}

const inferredTokenizerPath = explicitTokenizerPath || inferTokenizerPathFromRun();
const tokenizerPythonBin = inferredTokenizerPath ? pickPythonWithTorch() : null;
const tokenizerTokenTable =
  inferredTokenizerPath && tokenizerPythonBin
    ? loadTokenizerTokenTable(inferredTokenizerPath, tokenizerPythonBin)
    : null;
const tokenizerDecodeCache = new Map<number, string>();
const promptTokenIdCache = new Map<string, number[]>();

function decodeIdsWithPython(ids: number[]): string[] | null {
  if (!inferredTokenizerPath || !tokenizerPythonBin) return null;
  const cleaned = ids
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x))
    .map((x) => Math.trunc(x));
  if (cleaned.length === 0) return [];

  const missing = Array.from(new Set(cleaned.filter((id) => !tokenizerDecodeCache.has(id))));
  if (missing.length > 0) {
    const code = [
      "import json, sys",
      "repo = sys.argv[1]",
      "if repo not in sys.path: sys.path.insert(0, repo)",
      "tok_path = sys.argv[2]",
      "ids = json.loads(sys.argv[3])",
      "from models.bpe_tokenizer import ByteBPETokenizer",
      "tok = ByteBPETokenizer.load(tok_path)",
      "out = [tok.decode([int(i)], skip_special=False) for i in ids]",
      "print(json.dumps(out, ensure_ascii=False))",
    ].join("\n");
    try {
      const proc = Bun.spawnSync({
        cmd: [tokenizerPythonBin, "-c", code, repoRoot, inferredTokenizerPath, JSON.stringify(missing)],
        cwd: repoRoot,
        stdout: "pipe",
        stderr: "pipe",
      });
      if (proc.exitCode !== 0) return null;
      const raw = Buffer.from(proc.stdout).toString("utf-8");
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed) || parsed.length !== missing.length) return null;
      for (let i = 0; i < missing.length; i++) {
        tokenizerDecodeCache.set(missing[i], String(parsed[i] ?? ""));
      }
    } catch {
      return null;
    }
  }
  return cleaned.map((id) => tokenizerDecodeCache.get(id) ?? `<id:${id}>`);
}

function _resolveStepProbe(payload: any, relFile?: string): { step: number | null; probeIdx: number | null } {
  let step = parseNum(payload?.step);
  let probeIdx = parseNum(payload?.probe_idx);
  if (step !== null && probeIdx !== null) {
    return { step, probeIdx };
  }
  if (!relFile) return { step, probeIdx };
  const m = String(relFile).match(/step_(\d+)[/\\]attn_probe(\d+)_/i);
  if (m) {
    if (step === null) step = Number(m[1]);
    if (probeIdx === null) probeIdx = Number(m[2]);
  }
  return { step, probeIdx };
}

function _loadPromptTokenIds(step: number, probeIdx: number): number[] {
  const key = `${step}:${probeIdx}`;
  const cached = promptTokenIdCache.get(key);
  if (cached) return cached;

  const summaryPath = path.join(probeDebugDir, `step_${step}`, "summary.json");
  if (!fs.existsSync(summaryPath) || !fs.statSync(summaryPath).isFile()) {
    promptTokenIdCache.set(key, []);
    return [];
  }
  try {
    const summary = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));
    const probes = Array.isArray(summary?.probes) ? summary.probes : [];
    for (const probe of probes) {
      if (parseNum(probe?.probe_idx) !== probeIdx) continue;
      const ids = _toIntList(probe?.generation?.prompt_token_ids);
      promptTokenIdCache.set(key, ids);
      return ids;
    }
  } catch {
    // no-op
  }
  promptTokenIdCache.set(key, []);
  return [];
}

function hydrateAttentionPayload(rawPayload: any, relFile?: string) {
  if (!rawPayload || typeof rawPayload !== "object") return rawPayload;
  const payload = { ...rawPayload };

  const grid = Array.isArray(payload.grid)
    ? payload.grid
    : Array.isArray(payload.attn_prob_grid)
      ? payload.attn_prob_grid
      : Array.isArray(payload.attn_score_grid)
        ? payload.attn_score_grid
        : null;
  const rows = Array.isArray(grid) ? grid.length : 0;
  const cols = rows > 0 && Array.isArray(grid?.[0]) ? grid[0].length : 0;

  let qIds = _toIntList(payload.q_token_ids);
  let kIds = _toIntList(payload.k_token_ids);

  if ((qIds.length === 0 || kIds.length === 0) && rows > 0 && cols > 0) {
    const idsHint = _resolveStepProbe(payload, relFile);
    if (idsHint.step !== null && idsHint.probeIdx !== null) {
      const promptIds = _loadPromptTokenIds(idsHint.step, idsHint.probeIdx);
      if (qIds.length === 0) qIds = promptIds.slice(0, rows);
      if (kIds.length === 0) kIds = promptIds.slice(0, cols);
    }
  }

  const qToksRaw = Array.isArray(payload.q_tokens) ? payload.q_tokens.map((x: unknown) => String(x ?? "")) : [];
  const kToksRaw = Array.isArray(payload.k_tokens) ? payload.k_tokens.map((x: unknown) => String(x ?? "")) : [];
  const qToks = qToksRaw.length === qIds.length ? qToksRaw : _decodeIds(qIds, tokenizerTokenTable);
  const kToks = kToksRaw.length === kIds.length ? kToksRaw : _decodeIds(kIds, tokenizerTokenTable);

  payload.q_token_ids = qIds;
  payload.k_token_ids = kIds;
  payload.q_tokens = qToks;
  payload.k_tokens = kToks;
  return payload;
}

function listCheckpoints() {
  const files = fs
    .readdirSync(runDir)
    .filter((f) => /^step_\d+.*\.tar$/i.test(f))
    .map((f) => {
      const m = f.match(/step_(\d+)/i);
      const step = m ? Number(m[1]) : -1;
      const fp = path.join(runDir, f);
      const st = fs.statSync(fp);
      return {
        file: f,
        step,
        mtimeMs: st.mtimeMs,
        bytes: st.size,
      };
    })
    .sort((a, b) => b.step - a.step || b.mtimeMs - a.mtimeMs)
    .slice(0, 30);

  return files;
}

function parseNum(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function detectPhase(segment: string): "train" | "val" | "test" | "other" {
  const s = segment.trim();
  if (/^validation\b/i.test(s)) return "val";
  if (/^test\b/i.test(s)) return "test";
  if (/\btrain\b/i.test(s) || /^epoch\b/i.test(s)) return "train";
  return "other";
}

function normalizeMetricKey(raw: string): string {
  let out = String(raw || "").toLowerCase();
  out = out.replace(/[()]/g, " ");
  out = out.replace(/[^a-z0-9]+/g, "_");
  out = out.replace(/^_+|_+$/g, "");
  out = out.replace(/^train_/, "");
  out = out.replace(/^validation_/, "");
  out = out.replace(/^val_/, "");
  out = out.replace(/^test_/, "");
  return out;
}

function splitByMarkers(text: string): string[] {
  if (!text) return [];
  const starts: number[] = [];
  const re = /Epoch:\s*\d+\/\d+|Validation\s+Step\s*[:=]\s*\d+|Test(?:\s+Step\s*[:=]\s*\d+|\s+CE\s*=)|\bStep\s*[:=]\s*\d+/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    starts.push(m.index);
    if (m.index === re.lastIndex) re.lastIndex += 1;
  }
  if (starts.length <= 1) return [text.trim()].filter(Boolean);
  const out: string[] = [];
  const prefix = text.slice(0, starts[0]).trim();
  if (prefix) out.push(prefix);
  for (let i = 0; i < starts.length; i++) {
    const from = starts[i];
    const to = i + 1 < starts.length ? starts[i + 1] : text.length;
    const seg = text.slice(from, to).trim();
    if (seg) out.push(seg);
  }
  return out;
}

type ParsedSeriesRow = { phase: string; metric: string; points: Array<{ step: number; value: number }> };

function parseLogSeries(text: string): ParsedSeriesRow[] {
  const phases = ["train", "val", "test", "other"] as const;
  const lastStepByPhase: Record<(typeof phases)[number], number> = { train: 0, val: 0, test: 0, other: 0 };
  let maxStep = 0;
  const seriesMap = new Map<string, Map<number, number>>();
  const pairRe = /([A-Za-z][A-Za-z0-9_\-\/\s]*?)\s*[:=]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/g;

  const parseSegment = (segment: string) => {
    if (!segment || segment.length < 4) return;
    const phase = detectPhase(segment);
    const stepMatch = segment.match(/\bStep\s*[:=]\s*(\d+)/i);
    let step = stepMatch ? Number(stepMatch[1]) : null;
    if (!Number.isFinite(step)) {
      if (lastStepByPhase[phase] > 0) step = lastStepByPhase[phase];
      else if (maxStep > 0) step = maxStep;
      else step = 0;
    }
    if (!stepMatch && phase === "other") return;
    pairRe.lastIndex = 0;
    let found = 0;
    let m: RegExpExecArray | null;
    while ((m = pairRe.exec(segment)) !== null) {
      const rawKey = String(m[1] || "").trim();
      const value = parseNum(m[2]);
      if (value === null) continue;
      const key = normalizeMetricKey(rawKey);
      if (!key || key === "step" || key === "epoch") continue;
      if (key.startsWith("nats_token")) continue;
      if (key === "tok" || key.endsWith("_tok")) continue;
      const seriesKey = `${phase}|${key}`;
      let stepMap = seriesMap.get(seriesKey);
      if (!stepMap) {
        stepMap = new Map<number, number>();
        seriesMap.set(seriesKey, stepMap);
      }
      stepMap.set(step as number, value);
      found += 1;
    }
    if (found > 0) {
      const stepValue = Number(step);
      lastStepByPhase[phase] = Math.max(lastStepByPhase[phase], stepValue);
      maxStep = Math.max(maxStep, stepValue);
    }
  };

  let partial = "";
  partial += String(text || "").replace(/\r/g, "\n");
  const lines = partial.split("\n");
  partial = lines.pop() || "";
  for (const line of lines) {
    const chunks = splitByMarkers(line);
    for (const chunk of chunks) parseSegment(chunk);
  }
  const inlineChunks = splitByMarkers(partial);
  for (const chunk of inlineChunks) parseSegment(chunk);

  const rows: ParsedSeriesRow[] = [];
  for (const [seriesKey, stepMap] of seriesMap.entries()) {
    const [phase, metric] = seriesKey.split("|");
    const points = Array.from(stepMap.entries())
      .map(([step, value]) => ({ step, value }))
      .sort((a, b) => a.step - b.step);
    rows.push({ phase, metric, points });
  }
  rows.sort((a, b) => a.phase.localeCompare(b.phase) || a.metric.localeCompare(b.metric));
  return rows;
}

let baselineCache: { mtimeMs: number; payload: any } | null = null;
type AttentionIndexItem = {
  id?: string;
  step: number;
  probe_idx: number;
  scope: string;
  layer: number;
  kind: string;
  file: string;
  prompt?: string;
};
type AttentionIndexSnapshot = {
  available: boolean;
  mtimeMs: number;
  bytes: number;
  count: number;
  fullItems: AttentionIndexItem[];
  compactItems: AttentionIndexItem[];
};
type AttentionGridCacheEntry = {
  mtimeMs: number;
  bytes: number;
  payload: any;
};

let attentionIndexCache: AttentionIndexSnapshot | null = null;
const attentionGridCache = new Map<string, AttentionGridCacheEntry>();

function loadBaselinePayload() {
  if (!baselineLogfile) return { available: false };
  if (!fs.existsSync(baselineLogfile) || !fs.statSync(baselineLogfile).isFile()) {
    return { available: false, error: "missing_file" };
  }
  const st = fs.statSync(baselineLogfile);
  if (baselineCache && baselineCache.mtimeMs === st.mtimeMs) {
    return baselineCache.payload;
  }
  try {
    const text = fs.readFileSync(baselineLogfile, "utf-8");
    const payload = {
      available: true,
      logfile: path.basename(baselineLogfile),
      bytes: st.size,
      series: parseLogSeries(text),
    };
    baselineCache = { mtimeMs: st.mtimeMs, payload };
    return payload;
  } catch {
    return { available: false, error: "read_failed" };
  }
}

function touchAttentionGridCache(key: string) {
  const entry = attentionGridCache.get(key);
  if (!entry) return;
  attentionGridCache.delete(key);
  attentionGridCache.set(key, entry);
}

function trimAttentionGridCache() {
  if (attentionGridCacheLimit <= 0) {
    attentionGridCache.clear();
    return;
  }
  while (attentionGridCache.size > attentionGridCacheLimit) {
    const oldest = attentionGridCache.keys().next().value;
    if (!oldest) break;
    attentionGridCache.delete(oldest);
  }
}

function loadAttentionIndexSnapshot(): AttentionIndexSnapshot {
  if (!fs.existsSync(attentionIndexFile) || !fs.statSync(attentionIndexFile).isFile()) {
    attentionIndexCache = {
      available: false,
      mtimeMs: 0,
      bytes: 0,
      count: 0,
      fullItems: [],
      compactItems: [],
    };
    return attentionIndexCache;
  }

  let st: fs.Stats;
  try {
    st = fs.statSync(attentionIndexFile);
  } catch {
    return {
      available: false,
      mtimeMs: 0,
      bytes: 0,
      count: 0,
      fullItems: [],
      compactItems: [],
    };
  }

  if (
    attentionIndexCache &&
    attentionIndexCache.mtimeMs === st.mtimeMs &&
    attentionIndexCache.bytes === st.size
  ) {
    return attentionIndexCache;
  }

  try {
    const parsed = JSON.parse(fs.readFileSync(attentionIndexFile, "utf-8"));
    const fullItems = Array.isArray(parsed) ? (parsed as AttentionIndexItem[]) : [];
    const compactItems = fullItems.map((item) => ({
      id: typeof item?.id === "string" ? item.id : undefined,
      step: Number(item?.step),
      probe_idx: Number(item?.probe_idx),
      scope: String(item?.scope ?? ""),
      layer: Number(item?.layer),
      kind: String(item?.kind ?? ""),
      file: String(item?.file ?? ""),
    }));
    attentionIndexCache = {
      available: compactItems.length > 0,
      mtimeMs: st.mtimeMs,
      bytes: st.size,
      count: compactItems.length,
      fullItems,
      compactItems,
    };
    return attentionIndexCache;
  } catch {
    attentionIndexCache = {
      available: false,
      mtimeMs: st.mtimeMs,
      bytes: st.size,
      count: 0,
      fullItems: [],
      compactItems: [],
    };
    return attentionIndexCache;
  }
}

function getAttentionGridFromFile(rel: string): { available: true; file: string; payload: any } | { available: false; error: string } {
  const normalized = path.normalize(rel).replace(/^(\.\.(\/|\\|$))+/, "");
  if (!normalized) return { available: false, error: "missing_file" };

  const fullPath = path.resolve(probeDebugDir, normalized);
  if (!fullPath.startsWith(path.resolve(probeDebugDir))) {
    return { available: false, error: "bad_path" };
  }
  if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
    return { available: false, error: "not_found" };
  }

  let st: fs.Stats;
  try {
    st = fs.statSync(fullPath);
  } catch {
    return { available: false, error: "not_found" };
  }

  const cached = attentionGridCache.get(normalized);
  if (cached && cached.mtimeMs === st.mtimeMs && cached.bytes === st.size) {
    touchAttentionGridCache(normalized);
    return { available: true, file: normalized, payload: cached.payload };
  }

  try {
    const payload = hydrateAttentionPayload(JSON.parse(fs.readFileSync(fullPath, "utf-8")), normalized);
    if (attentionGridCacheLimit > 0) {
      attentionGridCache.set(normalized, { mtimeMs: st.mtimeMs, bytes: st.size, payload });
      trimAttentionGridCache();
    }
    return { available: true, file: normalized, payload };
  } catch {
    return { available: false, error: "failed_to_parse" };
  }
}

function prewarmAttentionGridCache() {
  if (attentionPrewarmLimit <= 0 || attentionGridCacheLimit <= 0) return;
  const snapshot = loadAttentionIndexSnapshot();
  if (!snapshot.available || snapshot.compactItems.length === 0) return;

  const requested =
    attentionPrewarmLimit === Number.MAX_SAFE_INTEGER
      ? snapshot.compactItems.length
      : Math.min(snapshot.compactItems.length, attentionPrewarmLimit);
  if (requested > attentionGridCacheLimit) {
    console.log(
      `Attention prewarm request (${requested}) exceeds cache limit (${attentionGridCacheLimit}); prewarming cache capacity only.`
    );
  }

  const ordered = snapshot.compactItems
    .slice()
    .sort((a, b) => Number(b.step) - Number(a.step));
  const maxLoad = Math.min(
    ordered.length,
    attentionGridCacheLimit,
    attentionPrewarmLimit === Number.MAX_SAFE_INTEGER ? ordered.length : attentionPrewarmLimit
  );
  if (maxLoad <= 0) return;

  const t0 = Date.now();
  let loaded = 0;
  for (let i = 0; i < maxLoad; i++) {
    const rel = String(ordered[i]?.file ?? "");
    if (!rel) continue;
    const result = getAttentionGridFromFile(rel);
    if (result.available) loaded += 1;
  }
  const dt = Date.now() - t0;
  console.log(`Attention prewarm: loaded ${loaded}/${maxLoad} grids in ${(dt / 1000).toFixed(2)}s`);
}

function loadProbeRecords() {
  if (!fs.existsSync(probeDebugDir) || !fs.statSync(probeDebugDir).isDirectory()) {
    return { available: false, records: [], probes: [], layers: [], metrics: [] };
  }

  const stepDirs = fs
    .readdirSync(probeDebugDir)
    .filter((d) => /^step_\d+$/i.test(d))
    .sort((a, b) => {
      const sa = Number(a.replace(/[^0-9]/g, ""));
      const sb = Number(b.replace(/[^0-9]/g, ""));
      return sa - sb;
    });

  const records: Array<{ step: number; probe_idx: number; layer: string; metric: string; value: number }> = [];
  const probeSet = new Set<number>();
  const layerSet = new Set<string>();
  const metricSet = new Set<string>();

  for (const stepDir of stepDirs) {
    const summaryPath = path.join(probeDebugDir, stepDir, "summary.json");
    if (!fs.existsSync(summaryPath)) continue;
    let summary: any;
    try {
      summary = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));
    } catch {
      continue;
    }
    const step = parseNum(summary?.step);
    if (step === null) continue;
    const probes = Array.isArray(summary?.probes) ? summary.probes : [];
    for (const probe of probes) {
      const probeIdx = parseNum(probe?.probe_idx);
      if (probeIdx === null) continue;
      probeSet.add(probeIdx);

      const scopeEntries: Array<{ scope: string; entries: any[] }> = [
        { scope: "enc", entries: Array.isArray(probe?.encoder_layers) ? probe.encoder_layers : [] },
        { scope: "dec_self", entries: Array.isArray(probe?.decoder_self_layers) ? probe.decoder_self_layers : [] },
        { scope: "dec_cross", entries: Array.isArray(probe?.decoder_cross_layers) ? probe.decoder_cross_layers : [] },
      ];
      for (const { scope, entries } of scopeEntries) {
        for (const entry of entries) {
          const layerNum = parseNum(entry?.layer);
          if (layerNum === null) continue;
          const layer = `${scope}:l${layerNum}`;
          layerSet.add(layer);
          for (const [metric, rawVal] of Object.entries(entry || {})) {
            if (metric === "layer" || metric === "attn_prob_grid" || metric === "attn_score_grid") continue;
            const value = parseNum(rawVal);
            if (value === null) continue;
            metricSet.add(metric);
            records.push({ step, probe_idx: probeIdx, layer, metric, value });
          }
        }
      }

      const hidden = probe?.hidden_metrics && typeof probe.hidden_metrics === "object" ? probe.hidden_metrics : {};
      for (const [hiddenKey, metricMap] of Object.entries(hidden)) {
        if (!metricMap || typeof metricMap !== "object") continue;
        const layer = `hidden:${hiddenKey}`;
        layerSet.add(layer);
        for (const [metric, rawVal] of Object.entries(metricMap as Record<string, unknown>)) {
          const value = parseNum(rawVal);
          if (value === null) continue;
          metricSet.add(metric);
          records.push({ step, probe_idx: probeIdx, layer, metric, value });
        }
      }
    }
  }

  records.sort((a, b) => a.step - b.step || a.probe_idx - b.probe_idx || a.layer.localeCompare(b.layer));
  return {
    available: records.length > 0,
    records,
    probes: Array.from(probeSet).sort((a, b) => a - b),
    layers: Array.from(layerSet).sort(),
    metrics: Array.from(metricSet).sort(),
  };
}

function loadProbeGenerations() {
  if (!fs.existsSync(probeDebugDir) || !fs.statSync(probeDebugDir).isDirectory()) {
    return { available: false, records: [], probes: [] };
  }

  const stepDirs = fs
    .readdirSync(probeDebugDir)
    .filter((d) => /^step_\d+$/i.test(d))
    .sort((a, b) => {
      const sa = Number(a.replace(/[^0-9]/g, ""));
      const sb = Number(b.replace(/[^0-9]/g, ""));
      return sa - sb;
    });

  const records: Array<{
    step: number;
    probe_idx: number;
    prompt: string;
    generated_text: string;
    full_text: string;
    stop_reason: string;
    generated_token_ids: number[];
  }> = [];
  const probeSet = new Set<number>();

  for (const stepDir of stepDirs) {
    const summaryPath = path.join(probeDebugDir, stepDir, "summary.json");
    if (!fs.existsSync(summaryPath)) continue;
    let summary: any;
    try {
      summary = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));
    } catch {
      continue;
    }
    const gens = Array.isArray(summary?.generations) ? summary.generations : [];
    for (const g of gens) {
      const step = parseNum(g?.step);
      const probeIdx = parseNum(g?.probe_idx);
      if (step === null || probeIdx === null) continue;
      probeSet.add(probeIdx);
      records.push({
        step,
        probe_idx: probeIdx,
        prompt: String(g?.prompt ?? ""),
        generated_text: String(g?.generated_text ?? ""),
        full_text: String(g?.full_text ?? ""),
        stop_reason: String(g?.stop_reason ?? ""),
        generated_token_ids: Array.isArray(g?.generated_token_ids)
          ? g.generated_token_ids.map((x: unknown) => Number(x)).filter((x: number) => Number.isFinite(x))
          : [],
      });
    }
  }

  records.sort((a, b) => a.step - b.step || a.probe_idx - b.probe_idx);
  return {
    available: records.length > 0,
    records,
    probes: Array.from(probeSet).sort((a, b) => a - b),
  };
}

prewarmAttentionGridCache();

serve({
  port,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/") {
      return new Response(await Bun.file(htmlPath).text(), {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    if (url.pathname === "/meta") {
      const logfiles = streamLogs.map((entry) => {
        let bytes = 0;
        try {
          bytes = fs.statSync(entry.fullPath).size;
        } catch {
          bytes = 0;
        }
        return {
          file: entry.file,
          kind: entry.kind,
          checkpoint: entry.kind === "continued" ? entry.checkpoint : null,
          bytes,
        };
      });
      const totalBytes = logfiles.reduce((sum, x) => sum + Number(x.bytes || 0), 0);
      return Response.json({
        runDir,
        logfile: streamLogNames.join(", "),
        logfiles,
        logMode: hasContinuedMode ? "continued" : "single",
        logfileBytes: totalBytes,
        baselineLogfile: baselineLogfile ? path.basename(baselineLogfile) : null,
        checkpoints: listCheckpoints(),
      });
    }

    if (url.pathname === "/checkpoints") {
      return Response.json(listCheckpoints());
    }

    if (url.pathname === "/baseline") {
      return Response.json(loadBaselinePayload());
    }

    if (url.pathname === "/probe_debug/latest_attention") {
      if (!fs.existsSync(latestAttentionFile)) {
        return Response.json({ available: false });
      }
      try {
        const payload = hydrateAttentionPayload(JSON.parse(fs.readFileSync(latestAttentionFile, "utf-8")));
        return Response.json({ available: true, ...payload });
      } catch {
        return Response.json({ available: false, error: "failed_to_parse" });
      }
    }

    if (url.pathname === "/probe_debug/records") {
      return Response.json(loadProbeRecords());
    }

    if (url.pathname === "/probe_debug/generations") {
      return Response.json(loadProbeGenerations());
    }

    if (url.pathname === "/probe_debug/attention_index") {
      const snapshot = loadAttentionIndexSnapshot();
      const compact = url.searchParams.get("compact") === "1";
      const items = compact ? snapshot.compactItems : snapshot.fullItems;
      return Response.json({
        available: snapshot.available,
        count: snapshot.count,
        bytes: snapshot.bytes,
        mtimeMs: snapshot.mtimeMs,
        compact,
        items,
      });
    }

    if (url.pathname === "/probe_debug/attention_index_meta") {
      const snapshot = loadAttentionIndexSnapshot();
      return Response.json({
        available: snapshot.available,
        count: snapshot.count,
        bytes: snapshot.bytes,
        mtimeMs: snapshot.mtimeMs,
      });
    }

    if (url.pathname === "/probe_debug/attention_grid") {
      const rel = url.searchParams.get("file") || "";
      const loaded = getAttentionGridFromFile(rel);
      if (!loaded.available) return Response.json(loaded);
      return Response.json({ available: true, file: loaded.file, ...loaded.payload });
    }

    if (url.pathname === "/stream") {
      const headerOffset = Number(req.headers.get("last-event-id") ?? "0");
      let offset = Number.isFinite(headerOffset) ? Math.max(0, Math.floor(headerOffset)) : 0;
      let closed = false;
      const watchers: fs.FSWatcher[] = [];
      let ping: ReturnType<typeof setInterval> | null = null;

      const stream = new ReadableStream<string>({
        start(controller) {
          const closeWatchers = () => {
            while (watchers.length > 0) {
              const watcher = watchers.pop();
              if (!watcher) continue;
              try {
                watcher.close();
              } catch {
                // no-op
              }
            }
          };

          const safeSend = (nextOffset: number, text: string) => {
            if (closed || text.length === 0) return;
            try {
              controller.enqueue(`id: ${nextOffset}\n`);
              controller.enqueue(`data: ${JSON.stringify({ text })}\n\n`);
            } catch {
              closed = true;
              closeWatchers();
            }
          };

          const getTotalLogBytes = (): number => {
            let total = 0;
            for (const fp of streamLogPaths) {
              try {
                total += fs.statSync(fp).size;
              } catch {
                // file may disappear in edge cases; treat as empty
              }
            }
            return total;
          };

          const readVirtualSlice = (from: number, to: number): string => {
            if (to <= from) return "";
            let cursor = 0;
            let out = "";
            for (const fp of streamLogPaths) {
              let size = 0;
              try {
                size = fs.statSync(fp).size;
              } catch {
                size = 0;
              }
              const fileStart = cursor;
              const fileEnd = cursor + size;
              if (to > fileStart && from < fileEnd) {
                const readFrom = Math.max(0, from - fileStart);
                const readTo = Math.min(size, to - fileStart);
                if (readTo > readFrom) {
                  out += readSlice(fp, readFrom, readTo);
                }
              }
              cursor = fileEnd;
              if (cursor >= to) break;
            }
            return out;
          };

          const sendFromCurrentOffset = () => {
            if (closed) return;
            const totalBytes = getTotalLogBytes();
            if (totalBytes < offset) {
              offset = 0;
            }
            if (totalBytes === offset) return;
            const chunk = readVirtualSlice(offset, totalBytes);
            offset = totalBytes;
            safeSend(totalBytes, chunk);
          };

          // initial replay from last offset
          sendFromCurrentOffset();

          for (const fp of streamLogPaths) {
            try {
              const watcher = fs.watch(fp, () => {
                sendFromCurrentOffset();
              });
              watchers.push(watcher);
            } catch {
              // no-op
            }
          }

          ping = setInterval(() => {
            if (closed) return;
            try {
              controller.enqueue(": ping\n\n");
            } catch {
              // no-op
            }
          }, 15000);
        },

        cancel() {
          closed = true;
          while (watchers.length > 0) {
            const watcher = watchers.pop();
            if (!watcher) continue;
            try {
              watcher.close();
            } catch {
              // no-op
            }
          }
          if (ping) clearInterval(ping);
          ping = null;
        },
      });

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    return new Response("Not found", { status: 404 });
  },
});

console.log(`LM tracker listening on http://0.0.0.0:${port}`);
console.log(`Run dir: ${runDir}`);
console.log(`Log mode: ${hasContinuedMode ? "continued" : "single"}`);
console.log(`Log files: ${streamLogNames.join(", ")}`);
if (baselineLogfile) {
  console.log(`Baseline compare logfile: ${baselineLogfile}`);
}
console.log(`Attention grid cache limit: ${attentionGridCacheLimit} (default=${DEFAULT_ATTN_GRID_CACHE_LIMIT})`);
const prewarmLabel = attentionPrewarmLimit === Number.MAX_SAFE_INTEGER ? "all" : String(attentionPrewarmLimit);
console.log(`Attention prewarm target: ${prewarmLabel} (default=${DEFAULT_ATTN_PREWARM_LIMIT})`);
if (inferredTokenizerPath) {
  const tokSource = explicitTokenizerPath ? "explicit" : "inferred";
  const tokMode = tokenizerTokenTable ? "table" : tokenizerPythonBin ? "on-demand" : "none";
  console.log(`Tokenizer (${tokSource}): ${inferredTokenizerPath}`);
  console.log(`Tokenizer python: ${tokenizerPythonBin ?? "not found"}`);
  console.log(`Tokenizer table: ${tokenizerTokenTable ? "loaded" : "unavailable"}`);
  console.log(`Tokenizer decode mode: ${tokMode}`);
} else {
  console.log("Tokenizer: not inferred (token strings may be unavailable).");
}
