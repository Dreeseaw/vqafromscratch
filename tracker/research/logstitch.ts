import fs from "fs";
import path from "path";

export type SeriesPoint = {
  step: number;
  value: number;
};

export type RunLogSegment = {
  file: string;
  fullPath: string;
  kind: "base" | "continued";
  resumeStep: number;
  mtimeMs: number;
};

export type ParsedRunLogData = {
  logfile: string | null;
  logfileMtimeMs: number | null;
  introLines: string[];
  tailLines: string[];
  finalAccuracy: number | null;
  bestAccuracy: number | null;
  lastTrainCe: number | null;
  lastStep: number | null;
  lastStepsPerSec: number | null;
  numParams: number | null;
  trainableParams: number | null;
  hasFinalCheckpoint: boolean;
  isEvalOnly: boolean;
  trainCeSeries: SeriesPoint[];
  valAccuracySeries: SeriesPoint[];
  valStepsPerSecSeries: SeriesPoint[];
};

export type MaterializedRunLog = {
  logfile: string | null;
  logfileMtimeMs: number | null;
  text: string;
  sourceFiles: string[];
};

type EvalBlock = {
  split: string;
  step: number | null;
  accuracy: number | null;
  maxBatch: number | null;
  totalBatches: number | null;
  batchesPerSec: number | null;
  tag: string | null;
  order: number;
};

function readText(file: string): string {
  try {
    return fs.readFileSync(file, "utf-8");
  } catch {
    return "";
  }
}

export function listRunLogSegments(runDir: string): RunLogSegment[] {
  if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) return [];
  return fs
    .readdirSync(runDir)
    .map((file) => {
      const fullPath = path.join(runDir, file);
      if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) return null;
      if (/^logfile\.txt$/i.test(file)) {
        return {
          file,
          fullPath,
          kind: "base" as const,
          resumeStep: 0,
          mtimeMs: fs.statSync(fullPath).mtimeMs,
        };
      }
      const resumeMatch = file.match(/^logfile_from_(\d+)\.txt$/i);
      if (!resumeMatch) return null;
      return {
        file,
        fullPath,
        kind: "continued" as const,
        resumeStep: Number(resumeMatch[1]),
        mtimeMs: fs.statSync(fullPath).mtimeMs,
      };
    })
    .filter((segment): segment is RunLogSegment => Boolean(segment))
    .sort((a, b) => a.resumeStep - b.resumeStep || a.mtimeMs - b.mtimeMs || a.file.localeCompare(b.file));
}

function classifyLogLine(line: string, currentStep: number | null) {
  const dinoStep = line.match(/\[dino\]\s+step=(\d+)(?:\/\d+)?\s+epoch=\d+\/\d+/);
  if (dinoStep) {
    const step = Number(dinoStep[1]);
    return { nextStep: step, lineStep: step, progressLike: true };
  }

  const siglipAlignStep = line.match(/\[siglip_align\]\s+step=(\d+)\s+phase_step=\d+\/\d+\s+epoch=\d+/);
  if (siglipAlignStep) {
    const step = Number(siglipAlignStep[1]);
    return { nextStep: step, lineStep: step, progressLike: true };
  }

  const mmStep = line.match(/\[mm\]\s+step=(\d+)/);
  if (mmStep) {
    const step = Number(mmStep[1]);
    return { nextStep: step, lineStep: step, progressLike: true };
  }

  const lmStep = line.match(/\bStep:\s*(\d+),/);
  if (lmStep) {
    const step = Number(lmStep[1]);
    return { nextStep: step, lineStep: step, progressLike: true };
  }

  const fixedEvalMatch = line.match(/\[mm\]\s+fixed-eval answers appended:\s+.+\sstep=(\d+)\stag=/);
  if (fixedEvalMatch) {
    const step = Number(fixedEvalMatch[1]);
    return { nextStep: step, lineStep: step, progressLike: true };
  }

  if (/\[eval:[^\]]+\]\s+/.test(line)) {
    return { nextStep: currentStep, lineStep: currentStep, progressLike: true };
  }

  if (/\[mm\]\s+final checkpoint:\s+/.test(line)) {
    return { nextStep: currentStep, lineStep: currentStep, progressLike: true };
  }

  return { nextStep: currentStep, lineStep: null, progressLike: false };
}

export function materializeRunLog(runDir: string): MaterializedRunLog {
  const segments = listRunLogSegments(runDir);
  const latestSegment = segments.slice().sort((a, b) => b.mtimeMs - a.mtimeMs)[0] ?? null;
  if (segments.length === 0) {
    return {
      logfile: null,
      logfileMtimeMs: null,
      text: "",
      sourceFiles: [],
    };
  }

  const stitchedLines: string[] = [];
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index];
    const nextResumeStep = segments[index + 1]?.resumeStep ?? Number.POSITIVE_INFINITY;
    let currentStep: number | null = segment.resumeStep > 0 ? segment.resumeStep : null;
    let seenProgress = index === 0;
    for (const line of readText(segment.fullPath).split(/\r?\n/)) {
      const { nextStep, lineStep, progressLike } = classifyLogLine(line, currentStep);
      currentStep = nextStep;
      if (index > 0 && !seenProgress && !progressLike) continue;
      if (progressLike) seenProgress = true;
      if (Number.isFinite(lineStep ?? NaN) && lineStep !== null && lineStep >= nextResumeStep) continue;
      stitchedLines.push(line);
    }
  }

  const normalized = stitchedLines
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return {
    logfile: latestSegment?.file ?? null,
    logfileMtimeMs: latestSegment?.mtimeMs ?? null,
    text: normalized ? `${normalized}\n` : "",
    sourceFiles: segments.map((segment) => segment.file),
  };
}

function shouldKeepEvalBlock(block: EvalBlock): boolean {
  if (block.split !== "val") return false;
  if (!Number.isFinite(block.accuracy ?? NaN)) return false;
  if (!Number.isFinite(block.step ?? NaN)) return false;
  if (block.tag) return true;
  if (!Number.isFinite(block.maxBatch ?? NaN) || !Number.isFinite(block.totalBatches ?? NaN)) return false;
  if ((block.totalBatches ?? 0) < 50) return false;
  return (block.maxBatch as number) >= (block.totalBatches as number);
}

function finalizeEvalBlock(
  block: EvalBlock | null,
  valAccPoints: Map<number, number>,
  valRatePoints: Map<number, number>,
  canonicalEvals: EvalBlock[]
) {
  if (!block || !shouldKeepEvalBlock(block) || block.step === null || block.accuracy === null) return;
  canonicalEvals.push(block);
  valAccPoints.set(block.step, block.accuracy);
  if (Number.isFinite(block.batchesPerSec ?? NaN)) valRatePoints.set(block.step, block.batchesPerSec as number);
}

export function parseRunLog(runDir: string, options: { introLineCount?: number; tailLineCount?: number } = {}): ParsedRunLogData {
  const introLineCount = options.introLineCount ?? 20;
  const tailLineCount = options.tailLineCount ?? 40;
  const segments = listRunLogSegments(runDir);
  const latestSegment = segments.slice().sort((a, b) => b.mtimeMs - a.mtimeMs)[0] ?? null;
  const empty: ParsedRunLogData = {
    logfile: latestSegment?.file ?? null,
    logfileMtimeMs: latestSegment?.mtimeMs ?? null,
    introLines: [],
    tailLines: [],
    finalAccuracy: null,
    bestAccuracy: null,
    lastTrainCe: null,
    lastStep: null,
    lastStepsPerSec: null,
    numParams: null,
    trainableParams: null,
    hasFinalCheckpoint: false,
    isEvalOnly: false,
    trainCeSeries: [],
    valAccuracySeries: [],
    valStepsPerSecSeries: [],
  };
  if (segments.length === 0) return empty;

  const introLines = readText(segments[0].fullPath)
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .slice(0, introLineCount);
  const tailLines = readText((latestSegment ?? segments[segments.length - 1]).fullPath)
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .slice(-tailLineCount);

  const trainPoints = new Map<number, { ce: number; stepsPerSec: number | null }>();
  const valAccPoints = new Map<number, number>();
  const valRatePoints = new Map<number, number>();
  const canonicalEvals: EvalBlock[] = [];
  let numParams: number | null = null;
  let trainableParams: number | null = null;
  let hasFinalCheckpoint = false;
  let isEvalOnly = false;
  let lastEvalStepSeen: number | null = null;
  let lastEvalRateSeen: number | null = null;
  let evalOrder = 0;

  for (const segment of segments) {
    const text = readText(segment.fullPath);
    if (!text) continue;

    const dinoParamsMatch = text.match(/\[dino\]\s+student params trainable=([\d,]+)\s+total=([\d,]+)/i);
    if (dinoParamsMatch) {
      trainableParams = Number(dinoParamsMatch[1].replaceAll(",", ""));
      numParams = Number(dinoParamsMatch[2].replaceAll(",", ""));
    }
    const trainableParamsMatch = text.match(/\btrainable_params=([\d,]+)/i);
    if (trainableParamsMatch) trainableParams = Number(trainableParamsMatch[1].replaceAll(",", ""));
    const totalParamsMatch = text.match(/\btotal_params=([\d,]+)/i) ?? text.match(/\bTotal params:\s*([\d,]+)/i);
    if (totalParamsMatch) numParams = Number(totalParamsMatch[1].replaceAll(",", ""));
    if (/\[mm\]\s+final checkpoint:\s+/m.test(text)) hasFinalCheckpoint = true;
  if (/\[dino\]\s+done\s+global_step=\d+\s+checkpoint=.+/m.test(text)) hasFinalCheckpoint = true;
    if (/\[siglip_align\]\s+done\s+global_step=\d+\s+phase_completed=\d+\s+checkpoint=.+/m.test(text)) hasFinalCheckpoint = true;
    if (/\beval_only=1\b/i.test(text) || /\btag=eval_only(?:_|$)/i.test(text)) isEvalOnly = true;

    let lastStepSeen: number | null = segment.resumeStep > 0 ? segment.resumeStep : null;
    let pendingEval: EvalBlock | null = null;

    for (const line of text.split(/\r?\n/)) {
      const mmStep = line.match(/\[mm\]\s+step=(\d+)/);
      const mmStepsPerSec = line.match(/\bsteps_per_s=([0-9]*\.?[0-9]+)/);
      if (mmStep && mmStepsPerSec) {
        finalizeEvalBlock(pendingEval, valAccPoints, valRatePoints, canonicalEvals);
        pendingEval = null;
        lastStepSeen = Number(mmStep[1]);
        const lossCe = line.match(/\bloss_ce=([0-9]*\.?[0-9]+)/);
        const loss = line.match(/\bloss=([0-9]*\.?[0-9]+)/);
        const ce = Number((lossCe ?? loss)?.[1] ?? NaN);
        const stepsPerSec = Number(mmStepsPerSec[1]);
        if (Number.isFinite(ce)) {
          trainPoints.set(lastStepSeen, {
            ce,
            stepsPerSec: Number.isFinite(stepsPerSec) ? stepsPerSec : null,
          });
        }
        continue;
      }

      const dinoStep = line.match(/\[dino\]\s+step=(\d+)(?:\/\d+)?\s+epoch=\d+\/\d+\s+loss=([0-9]*\.?[0-9]+).*\bsteps_per_s=([0-9]*\.?[0-9]+)/);
      if (dinoStep) {
        finalizeEvalBlock(pendingEval, valAccPoints, valRatePoints, canonicalEvals);
        pendingEval = null;
        lastStepSeen = Number(dinoStep[1]);
        const ce = Number(dinoStep[2]);
        const stepsPerSec = Number(dinoStep[3]);
        if (Number.isFinite(ce)) {
          trainPoints.set(lastStepSeen, {
            ce,
            stepsPerSec: Number.isFinite(stepsPerSec) ? stepsPerSec : null,
          });
        }
        continue;
      }

      const siglipAlignStep = line.match(/\[siglip_align\]\s+step=(\d+)\s+phase_step=\d+\/\d+\s+epoch=\d+\s+loss=([0-9]*\.?[0-9]+).*\bsteps_per_s=([0-9]*\.?[0-9]+)/);
      if (siglipAlignStep) {
        finalizeEvalBlock(pendingEval, valAccPoints, valRatePoints, canonicalEvals);
        pendingEval = null;
        lastStepSeen = Number(siglipAlignStep[1]);
        const ce = Number(siglipAlignStep[2]);
        const stepsPerSec = Number(siglipAlignStep[3]);
        if (Number.isFinite(ce)) {
          trainPoints.set(lastStepSeen, {
            ce,
            stepsPerSec: Number.isFinite(stepsPerSec) ? stepsPerSec : null,
          });
        }
        continue;
      }

      const lmStep = line.match(/\bStep:\s*(\d+),/);
      const lmTrainCe = line.match(/\bTrain CE:\s*([0-9]*\.?[0-9]+)/);
      const lmTokensPerSec = line.match(/\bTokens\/s:\s*([0-9]*\.?[0-9]+)/);
      if (lmStep && lmTrainCe) {
        finalizeEvalBlock(pendingEval, valAccPoints, valRatePoints, canonicalEvals);
        pendingEval = null;
        lastStepSeen = Number(lmStep[1]);
        trainPoints.set(lastStepSeen, {
          ce: Number(lmTrainCe[1]),
          stepsPerSec: lmTokensPerSec ? Number(lmTokensPerSec[1]) : null,
        });
        continue;
      }

      const valRateMatch = line.match(/\[eval:([^\]]+)\]\s+batch=(\d+)(?:\/(\d+))?\s+samples=(\d+)\s+elapsed_s=([0-9]*\.?[0-9]+)/);
      if (valRateMatch) {
        const split = String(valRateMatch[1] ?? "").trim().toLowerCase();
        const currentBatch = Number(valRateMatch[2]);
        const totalBatches = Number(valRateMatch[3] ?? NaN);
        const elapsed = Number(valRateMatch[5]);
        if (!pendingEval || pendingEval.split !== split) {
          finalizeEvalBlock(pendingEval, valAccPoints, valRatePoints, canonicalEvals);
          pendingEval = {
            split,
            step: lastStepSeen,
            accuracy: null,
            maxBatch: null,
            totalBatches: null,
            batchesPerSec: null,
            tag: null,
            order: evalOrder++,
          };
        }
        pendingEval.step = pendingEval.step ?? lastStepSeen;
        pendingEval.maxBatch = Math.max(pendingEval.maxBatch ?? 0, currentBatch);
        if (Number.isFinite(totalBatches) && totalBatches > 0) pendingEval.totalBatches = totalBatches;
        if (Number.isFinite(elapsed) && elapsed > 0) pendingEval.batchesPerSec = currentBatch / elapsed;
        lastEvalStepSeen = pendingEval.step ?? lastEvalStepSeen;
        if (Number.isFinite(pendingEval.batchesPerSec ?? NaN)) lastEvalRateSeen = pendingEval.batchesPerSec;
        continue;
      }

      const valAccMatch = line.match(/\[eval:([^\]]+)\]\s+overall_accuracy=([0-9]*\.?[0-9]+)/);
      if (valAccMatch) {
        const split = String(valAccMatch[1] ?? "").trim().toLowerCase();
        const accuracy = Number(valAccMatch[2]);
        if (!pendingEval || pendingEval.split !== split) {
          finalizeEvalBlock(pendingEval, valAccPoints, valRatePoints, canonicalEvals);
          pendingEval = {
            split,
            step: lastStepSeen,
            accuracy: null,
            maxBatch: null,
            totalBatches: null,
            batchesPerSec: null,
            tag: null,
            order: evalOrder++,
          };
        }
        pendingEval.step = pendingEval.step ?? lastStepSeen;
        pendingEval.accuracy = accuracy;
        continue;
      }

      const fixedEvalMatch = line.match(/\[mm\]\s+fixed-eval answers appended:\s+.+\sstep=(\d+)\stag=([A-Za-z0-9._-]+)/);
      if (fixedEvalMatch && pendingEval) {
        const evalStep = Number(fixedEvalMatch[1]);
        if (!Number.isFinite(pendingEval.step ?? NaN)) pendingEval.step = evalStep;
        if (pendingEval.step === evalStep) {
          pendingEval.tag = fixedEvalMatch[2];
          finalizeEvalBlock(pendingEval, valAccPoints, valRatePoints, canonicalEvals);
          pendingEval = null;
        }
        continue;
      }
    }

    finalizeEvalBlock(pendingEval, valAccPoints, valRatePoints, canonicalEvals);
  }

  canonicalEvals.sort((a, b) => a.order - b.order);
  const finalEval = canonicalEvals.length > 0 ? canonicalEvals[canonicalEvals.length - 1] : null;
  const bestAccuracy =
    canonicalEvals.length > 0 ? Math.max(...canonicalEvals.map((block) => block.accuracy as number)) : null;
  const trainEntries = [...trainPoints.entries()].sort((a, b) => a[0] - b[0]);
  const trainCeSeries = trainEntries.map(([step, point]) => ({ step, value: point.ce }));
  const valAccuracySeries = [...valAccPoints.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, value]) => ({ step, value }));
  const valStepsPerSecSeries = [...valRatePoints.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, value]) => ({ step, value }));
  const lastTrainPoint = trainEntries.at(-1) ?? null;
  const lastEvalAccPoint = valAccuracySeries.at(-1) ?? null;
  const lastEvalRatePoint = valStepsPerSecSeries.at(-1) ?? null;

  return {
    logfile: latestSegment?.file ?? null,
    logfileMtimeMs: latestSegment?.mtimeMs ?? null,
    introLines,
    tailLines,
    finalAccuracy: finalEval?.accuracy ?? null,
    bestAccuracy,
    lastTrainCe: lastTrainPoint?.[1].ce ?? null,
    lastStep: lastTrainPoint?.[0] ?? lastEvalAccPoint?.step ?? lastEvalRatePoint?.step ?? lastEvalStepSeen ?? null,
    lastStepsPerSec: lastTrainPoint?.[1].stepsPerSec ?? lastEvalRateSeen,
    numParams,
    trainableParams,
    hasFinalCheckpoint,
    isEvalOnly,
    trainCeSeries,
    valAccuracySeries,
    valStepsPerSecSeries,
  };
}
