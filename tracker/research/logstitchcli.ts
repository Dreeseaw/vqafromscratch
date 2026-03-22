import fs from "fs";
import path from "path";
import { materializeRunLog, parseRunLog } from "./logstitch";

function usageAndExit(): never {
  console.error(
    "Usage: bun run tracker/research/logstitchcli.ts -f <run_dir> [-o <output_file>] [--summary-json]"
  );
  process.exit(1);
}

const args = process.argv.slice(2);
const runDirIdx = args.indexOf("-f");
if (runDirIdx === -1 || !args[runDirIdx + 1]) usageAndExit();

const outputIdx = args.findIndex((arg) => arg === "-o" || arg === "--output");
const runDir = path.resolve(args[runDirIdx + 1]);
const outputFile = outputIdx !== -1 ? String(args[outputIdx + 1] ?? "").trim() : "";
const summaryJson = args.includes("--summary-json");

if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) {
  console.error(`Run directory does not exist: ${runDir}`);
  process.exit(1);
}
if (outputIdx !== -1 && !outputFile) usageAndExit();

const materialized = materializeRunLog(runDir);
if (!materialized.text) {
  console.error(`No canonical logfile segments found in ${runDir}. Expected logfile.txt or logfile_from_<step>.txt`);
  process.exit(1);
}

if (outputFile) {
  const resolvedOutput = path.resolve(outputFile);
  fs.mkdirSync(path.dirname(resolvedOutput), { recursive: true });
  fs.writeFileSync(resolvedOutput, materialized.text, "utf-8");
  console.error(
    `Wrote stitched log to ${resolvedOutput} from ${materialized.sourceFiles.length} source file(s): ${materialized.sourceFiles.join(", ")}`
  );
} else {
  process.stdout.write(materialized.text);
}

if (summaryJson) {
  const parsed = parseRunLog(runDir);
  const summary = {
    runDir,
    logfile: parsed.logfile,
    updatedAt: parsed.logfileMtimeMs ? new Date(parsed.logfileMtimeMs).toISOString() : null,
    sourceFiles: materialized.sourceFiles,
    finalAccuracy: parsed.finalAccuracy,
    bestAccuracy: parsed.bestAccuracy,
    lastTrainCe: parsed.lastTrainCe,
    lastStep: parsed.lastStep,
    trainPoints: parsed.trainCeSeries.length,
    valPoints: parsed.valAccuracySeries.length,
  };
  process.stderr.write(`${JSON.stringify(summary, null, 2)}\n`);
}
