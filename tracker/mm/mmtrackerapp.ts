import { serve } from "bun";
import fs from "fs";
import path from "path";

type TrainEvent = {
  kind: "train";
  step: number;
  epoch: number;
  loss: number;
  lossTokens: number;
  stepsPerSec: number;
};

type EvalOverallEvent = {
  kind: "eval_overall";
  split: string;
  accuracy: number;
};

type EvalAnswerTypeEvent = {
  kind: "eval_answer_type";
  split: string;
  values: Record<string, number>;
};

type FixedEvalEvent = {
  kind: "fixed_eval";
  step: number;
  epoch: number;
  tag: string;
  split: string;
  answers: Array<{
    question_id: number;
    image_id: number;
    question: string;
    prediction: string | null;
  }>;
};

type ParsedEvent = TrainEvent | EvalOverallEvent | EvalAnswerTypeEvent | FixedEvalEvent;

function usageAndExit() {
  console.error(
    "Usage: bun run tracker/mm/mmtrackerapp.ts -f <run_dir> [-p <port>] [--log <logfile_name_or_path>] [--continued]"
  );
  process.exit(1);
}

const args = process.argv.slice(2);
const runDirIdx = args.indexOf("-f");
if (runDirIdx === -1 || !args[runDirIdx + 1]) usageAndExit();

const runDir = path.resolve(args[runDirIdx + 1]);
if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) {
  console.error(`Run directory does not exist: ${runDir}`);
  process.exit(1);
}

const continuedMode = args.includes("--continued");
const explicitLogIdx = args.indexOf("--log");
const portIdx = args.indexOf("-p");
const port = portIdx !== -1 ? Number(args[portIdx + 1]) : 3050;
if (!Number.isInteger(port) || port <= 0) {
  console.error(`Invalid port: ${args[portIdx + 1]}`);
  process.exit(1);
}

type LogInfo = {
  name: string;
  fullPath: string;
  checkpoint: number;
  mtimeMs: number;
};

function listLogfiles(dir: string): LogInfo[] {
  const out: LogInfo[] = [];
  for (const name of fs.readdirSync(dir)) {
    if (!/^logfile.*\.txt$/i.test(name)) continue;
    const fullPath = path.join(dir, name);
    if (!fs.statSync(fullPath).isFile()) continue;
    const st = fs.statSync(fullPath);
    if (/^logfile\.txt$/i.test(name)) {
      out.push({ name, fullPath, checkpoint: -1, mtimeMs: st.mtimeMs });
      continue;
    }
    const m = name.match(/^logfile_from_(\d+)\.txt$/i);
    if (m) {
      out.push({
        name,
        fullPath,
        checkpoint: Number(m[1]),
        mtimeMs: st.mtimeMs,
      });
      continue;
    }
    out.push({ name, fullPath, checkpoint: Number.MAX_SAFE_INTEGER, mtimeMs: st.mtimeMs });
  }
  return out.sort((a, b) => a.checkpoint - b.checkpoint || a.mtimeMs - b.mtimeMs || a.name.localeCompare(b.name));
}

const allLogs = listLogfiles(runDir);
if (allLogs.length === 0) {
  console.error(`No logfile*.txt found in ${runDir}`);
  process.exit(1);
}

let streamLogs: LogInfo[] = [];
if (explicitLogIdx !== -1) {
  const raw = args[explicitLogIdx + 1];
  if (!raw) usageAndExit();
  const fullPath = path.isAbsolute(raw) ? path.resolve(raw) : path.resolve(runDir, raw);
  if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
    console.error(`Missing logfile: ${fullPath}`);
    process.exit(1);
  }
  streamLogs = [
    {
      name: path.basename(fullPath),
      fullPath,
      checkpoint: Number.MAX_SAFE_INTEGER,
      mtimeMs: fs.statSync(fullPath).mtimeMs,
    },
  ];
} else if (continuedMode) {
  streamLogs = allLogs.filter((x) => /^logfile_from_\d+\.txt$/i.test(x.name));
  if (streamLogs.length === 0) {
    console.error("No continuation logs found. Expected logfile_from_<step>.txt");
    process.exit(1);
  }
} else {
  const base = allLogs.find((x) => x.name === "logfile.txt");
  streamLogs = [base ?? allLogs[allLogs.length - 1]];
}

const staticRoot = import.meta.dir;
const repoRoot = path.resolve(staticRoot, "..", "..");
const htmlPath = path.join(staticRoot, "index.html");
if (!fs.existsSync(htmlPath)) {
  console.error(`Missing frontend file: ${htmlPath}`);
  process.exit(1);
}

function parseAnswerTypeValues(raw: string): Record<string, number> {
  const out: Record<string, number> = {};
  const parts = raw.trim().split(/\s+/);
  for (const p of parts) {
    const i = p.indexOf("=");
    if (i <= 0) continue;
    const k = p.slice(0, i).trim();
    const v = Number(p.slice(i + 1).trim());
    if (!k || !Number.isFinite(v)) continue;
    out[k] = v;
  }
  return out;
}

function readNewestFixedEvalPromptsFile(): string {
  const files = fs
    .readdirSync(runDir)
    .filter((f) => /^fixed_eval_.*_prompts\.json$/i.test(f))
    .map((f) => path.join(runDir, f));
  if (files.length === 0) return "";
  files.sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs);
  return files[files.length - 1];
}

function readNewestFixedEvalAnswersFile(): string {
  const files = fs
    .readdirSync(runDir)
    .filter((f) => /^fixed_eval_.*_answers\.jsonl$/i.test(f))
    .map((f) => path.join(runDir, f));
  if (files.length === 0) return "";
  files.sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs);
  return files[files.length - 1];
}

function readFixedPrompts() {
  const f = readNewestFixedEvalPromptsFile();
  if (!f) return [];
  try {
    return JSON.parse(fs.readFileSync(f, "utf-8"));
  } catch {
    return [];
  }
}

function readFixedAnswers(maxRows = 500): FixedEvalEvent[] {
  const f = readNewestFixedEvalAnswersFile();
  if (!f) return [];
  const lines = fs.readFileSync(f, "utf-8").split("\n").filter((x) => x.trim().length > 0);
  const picked = lines.slice(Math.max(0, lines.length - maxRows));
  const out: FixedEvalEvent[] = [];
  for (const line of picked) {
    try {
      const obj = JSON.parse(line);
      out.push({
        kind: "fixed_eval",
        step: Number(obj.global_step ?? 0),
        epoch: Number(obj.epoch ?? 0),
        tag: String(obj.tag ?? ""),
        split: String(obj.split ?? "eval"),
        answers: Array.isArray(obj.answers) ? obj.answers : [],
      });
    } catch {
      // ignore bad rows
    }
  }
  return out;
}

function parseLine(line: string): ParsedEvent | null {
  const s = line.trim();
  if (!s) return null;

  let m = s.match(/^\[mm\]\s+step=(\d+)\s+epoch=(\d+)\s+loss=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+loss_tokens=(\d+)\s+steps_per_s=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$/);
  if (m) {
    return {
      kind: "train",
      step: Number(m[1]),
      epoch: Number(m[2]),
      loss: Number(m[3]),
      lossTokens: Number(m[4]),
      stepsPerSec: Number(m[5]),
    };
  }

  m = s.match(/^\[eval:([^\]]+)\]\s+overall_accuracy=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$/);
  if (m) {
    return {
      kind: "eval_overall",
      split: String(m[1]),
      accuracy: Number(m[2]),
    };
  }

  m = s.match(/^\[eval:([^\]]+)\]\s+answer_type:\s+(.+)$/);
  if (m) {
    return {
      kind: "eval_answer_type",
      split: String(m[1]),
      values: parseAnswerTypeValues(String(m[2])),
    };
  }

  m = s.match(/^\[mm\]\s+fixed-eval answers appended:\s+(.+)\s+step=(\d+)\s+tag=([^\s]+)$/);
  if (m) {
    const rows = readFixedAnswers(1);
    if (rows.length > 0) return rows[rows.length - 1];
  }

  return null;
}

function parseFileEvents(file: string): ParsedEvent[] {
  const out: ParsedEvent[] = [];
  if (!fs.existsSync(file)) return out;
  const lines = fs.readFileSync(file, "utf-8").split("\n");
  for (const line of lines) {
    const ev = parseLine(line);
    if (ev) out.push(ev);
  }
  return out;
}

function resolveSafeRepoFile(rawPath: string): string | null {
  const token = String(rawPath || "").trim();
  if (!token) return null;
  const abs = path.isAbsolute(token) ? path.resolve(token) : path.resolve(repoRoot, token);
  if (!abs.startsWith(repoRoot)) return null;
  if (!fs.existsSync(abs) || !fs.statSync(abs).isFile()) return null;
  return abs;
}

function bootstrapPayload() {
  const events = streamLogs.flatMap((x) => parseFileEvents(x.fullPath));
  return {
    runDir,
    logfiles: streamLogs.map((x) => x.name),
    fixedPrompts: readFixedPrompts(),
    fixedAnswers: readFixedAnswers(),
    events,
  };
}

serve({
  port,
  fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/") {
      return new Response(fs.readFileSync(htmlPath, "utf-8"), {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    if (url.pathname === "/api/bootstrap") {
      return new Response(JSON.stringify(bootstrapPayload()), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/api/fixed_prompts") {
      return new Response(JSON.stringify(readFixedPrompts()), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/api/fixed_answers") {
      return new Response(JSON.stringify(readFixedAnswers()), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/img") {
      const p = url.searchParams.get("path") ?? "";
      const full = resolveSafeRepoFile(p);
      if (!full) return new Response("Not found", { status: 404 });
      return new Response(Bun.file(full));
    }

    if (url.pathname === "/stream") {
      let closed = false;
      let seq = 1;
      const watchers: fs.FSWatcher[] = [];
      const offsets = new Map<string, number>();
      const carries = new Map<string, string>();
      let ping: ReturnType<typeof setInterval> | null = null;

      const stream = new ReadableStream<string>({
        start(controller) {
          const push = (ev: ParsedEvent) => {
            if (closed) return;
            try {
              controller.enqueue(`id: ${seq}\n`);
              controller.enqueue(`data: ${JSON.stringify(ev)}\n\n`);
              seq += 1;
            } catch {
              closed = true;
            }
          };

          const processText = (file: string, text: string) => {
            const carry = carries.get(file) ?? "";
            const merged = carry + text;
            const lines = merged.split("\n");
            const nextCarry = lines.pop() ?? "";
            carries.set(file, nextCarry);
            for (const line of lines) {
              const ev = parseLine(line);
              if (ev) push(ev);
            }
          };

          // backlog
          for (const log of streamLogs) {
            if (!fs.existsSync(log.fullPath)) continue;
            const txt = fs.readFileSync(log.fullPath, "utf-8");
            processText(log.fullPath, txt + "\n");
            offsets.set(log.fullPath, fs.statSync(log.fullPath).size);
            carries.set(log.fullPath, "");
          }

          const lastFixed = readFixedAnswers(1);
          if (lastFixed.length > 0) push(lastFixed[lastFixed.length - 1]);

          // tail
          for (const log of streamLogs) {
            const file = log.fullPath;
            if (!fs.existsSync(file)) continue;
            const watcher = fs.watch(file, () => {
              if (closed || !fs.existsSync(file)) return;
              const st = fs.statSync(file);
              const prev = offsets.get(file) ?? 0;
              if (st.size < prev) {
                offsets.set(file, st.size);
                carries.set(file, "");
                return;
              }
              if (st.size === prev) return;
              const fd = fs.openSync(file, "r");
              try {
                const len = st.size - prev;
                const buf = Buffer.alloc(len);
                fs.readSync(fd, buf, 0, len, prev);
                offsets.set(file, st.size);
                processText(file, buf.toString("utf-8"));
              } finally {
                fs.closeSync(fd);
              }
            });
            watchers.push(watcher);
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
          for (const w of watchers) {
            try {
              w.close();
            } catch {
              // no-op
            }
          }
          try {
            if (ping) clearInterval(ping);
          } catch {
            // no-op
          }
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

console.log(
  `[mm-tracker] serving run=${runDir} logs=${streamLogs.map((x) => x.name).join(",")} http://localhost:${port}`
);
