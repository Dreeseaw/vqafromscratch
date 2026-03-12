import { marked } from "https://esm.sh/marked@16.3.0";

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

type DocSummary = {
  file: string;
  title: string;
  updatedAt: string;
  runRefs: string[];
  mentionedAccuracies: number[];
};

type TaskSummary = {
  id: string;
  title: string;
  description: string | null;
};

type Bootstrap = {
  generatedAt: string;
  docsRoot: string;
  logsRoot: string;
  scriptsRoot: string;
  tasks: TaskSummary[];
  selectedTask: {
    id: string;
    title: string;
    description: string | null;
    docsRoot: string;
    scriptsRoot: string;
    logsRoot: string;
  };
  docs: DocSummary[];
  sweeps: SweepSummary[];
  runs: RunSummary[];
  summary: {
    docsCount: number;
    sweepsCount: number;
    runsCount: number;
    runsWithAccuracy: number;
    bestRun: { runId: string; finalAccuracy: number | null; bestAccuracy: number | null } | null;
  };
};

type SeriesPoint = {
  step: number;
  value: number;
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

type QaMessage = {
  role: "user" | "assistant";
  content: string;
  evidence?: string[];
  generatedAt?: string;
  pending?: boolean;
  error?: boolean;
};

type SortDirection = "asc" | "desc";
type SortState<K extends string> = {
  key: K | null;
  direction: SortDirection;
};
type SweepSortKey = "sweepId" | "status" | "activeRuns" | "runs" | "bestAccuracy" | "lastTrainCe" | "startedAt" | "endedAt";
type RunSortKey = "runId" | "finalAccuracy" | "lastTrainCe" | "lastStep" | "lastStepsPerSec" | "numParams" | "hasFinalCheckpoint";
type DocSortKey = "title" | "updatedAt" | "runRefs" | "topAccuracy";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

const fmtAcc = (v: number | null) => (Number.isFinite(v ?? NaN) ? (v as number).toFixed(4) : "-");
const fmtNum = (v: number | null) => (Number.isFinite(v ?? NaN) ? String(v) : "-");
const fmtRate = (v: number | null) => (Number.isFinite(v ?? NaN) ? (v as number).toFixed(4) : "-");
const fmtPct = (v: number | null) => (Number.isFinite(v ?? NaN) ? `${(v as number).toFixed(1)}%` : "-");
const fmtParams = (total: number | null, trainable: number | null) => {
  if (!Number.isFinite(total ?? NaN)) return "-";
  const totalStr = Intl.NumberFormat("en-US").format(total as number);
  if (!Number.isFinite(trainable ?? NaN) || !total || total <= 0) return totalStr;
  return `${totalStr} (${fmtPct(((trainable as number) / (total as number)) * 100)})`;
};

const SWEEPS_PAGE_SIZE = 5;
const RUNS_PAGE_SIZE = 10;
const DOCS_PAGE_SIZE = 5;
const BOOTSTRAP_REFRESH_MS = 10000;
const RUN_DETAIL_REFRESH_MS = 10000;
const COLLAPSE_STORAGE_KEY = "researchTrackerCollapsedSections";

let state: Bootstrap | null = null;
let selectedTaskId = getTaskFromUrl();
let selectedSweep = "__all__";
let activeOnly = false;
let sweepsPage = 0;
let runsPage = 0;
let docsPage = 0;
let selectedRunId: string | null = null;
let runDetail: RunDetail | null = null;
let selectedDocFile: string | null = null;
let selectedDocUpdatedAt: string | null = null;
let bootstrapLoadInFlight = false;
let runDetailLoadInFlight = false;
let taskQaLoadInFlight = false;
let sweepsSort: SortState<SweepSortKey> = { key: null, direction: "asc" };
let runsSort: SortState<RunSortKey> = { key: "finalAccuracy", direction: "desc" };
let docsSort: SortState<DocSortKey> = { key: null, direction: "asc" };
const qaThreads = new Map<string, QaMessage[]>();
let collapsedSections = new Set<string>();

function getTaskFromUrl(): string {
  const url = new URL(window.location.href);
  return String(url.searchParams.get("task") ?? "").trim();
}

function setTaskInUrl(taskId: string) {
  const url = new URL(window.location.href);
  if (taskId) url.searchParams.set("task", taskId);
  else url.searchParams.delete("task");
  history.replaceState(null, "", url.toString());
}

function getActiveTaskId(): string {
  return selectedTaskId || state?.selectedTask.id || "";
}

function apiUrl(path: string, extra: Record<string, string> = {}): string {
  const url = new URL(path, window.location.origin);
  const taskId = getActiveTaskId();
  if (taskId) url.searchParams.set("task", taskId);
  for (const [key, value] of Object.entries(extra)) url.searchParams.set(key, value);
  return `${url.pathname}${url.search}`;
}

function openDocInNewTab(file: string) {
  window.open(apiUrl("/doc", { file }), "_blank", "noopener,noreferrer");
}

function loadCollapsedSections(): Set<string> {
  try {
    const raw = window.localStorage.getItem(COLLAPSE_STORAGE_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw) as string[];
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

function saveCollapsedSections() {
  try {
    window.localStorage.setItem(COLLAPSE_STORAGE_KEY, JSON.stringify([...collapsedSections]));
  } catch {
    // ignore storage failures
  }
}

function setSectionCollapsed(sectionId: string, collapsed: boolean) {
  const button = document.querySelector<HTMLButtonElement>(`[data-section-id="${sectionId}"]`);
  const contentId = button?.dataset.target;
  const content = contentId ? document.getElementById(contentId) : null;
  if (!button || !content) return;
  content.hidden = collapsed;
  button.textContent = collapsed ? "Expand" : "Collapse";
  button.setAttribute("aria-expanded", String(!collapsed));
  if (collapsed) collapsedSections.add(sectionId);
  else collapsedSections.delete(sectionId);
}

function openSection(sectionId: string) {
  setSectionCollapsed(sectionId, false);
  saveCollapsedSections();
}

function setupCollapsibles() {
  collapsedSections = loadCollapsedSections();
  const buttons = document.querySelectorAll<HTMLButtonElement>(".collapse-button");
  for (const button of buttons) {
    const section = button.closest<HTMLElement>(".collapsible-section");
    if (!section?.id) continue;
    button.dataset.sectionId = section.id;
    setSectionCollapsed(section.id, collapsedSections.has(section.id));
    button.addEventListener("click", () => {
      const nextCollapsed = !collapsedSections.has(section.id);
      setSectionCollapsed(section.id, nextCollapsed);
      saveCollapsedSections();
    });
  }
}

function resetTaskScopedUi() {
  selectedSweep = "__all__";
  activeOnly = false;
  sweepsPage = 0;
  runsPage = 0;
  docsPage = 0;
  selectedRunId = null;
  runDetail = null;
  selectedDocFile = null;
  selectedDocUpdatedAt = null;
  $("runDetailPanel").hidden = true;
  $("docDetailPanel").hidden = true;
  $("docDetailOpen").hidden = true;
}

function toggleSort<K extends string>(sort: SortState<K>, key: K) {
  if (sort.key === key) {
    sort.direction = sort.direction === "asc" ? "desc" : "asc";
  } else {
    sort.key = key;
    sort.direction = "asc";
  }
}

function compareMaybeNumber(a: number | null, b: number | null, direction: SortDirection): number {
  const aMissing = !Number.isFinite(a ?? NaN);
  const bMissing = !Number.isFinite(b ?? NaN);
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;
  return direction === "asc" ? (a as number) - (b as number) : (b as number) - (a as number);
}

function compareMaybeString(a: string | null, b: string | null, direction: SortDirection): number {
  const av = String(a ?? "").trim();
  const bv = String(b ?? "").trim();
  if (!av && !bv) return 0;
  if (!av) return 1;
  if (!bv) return -1;
  return direction === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
}

function compareBoolean(a: boolean, b: boolean, direction: SortDirection): number {
  return direction === "asc" ? Number(a) - Number(b) : Number(b) - Number(a);
}

function compareMaybeDate(a: string | null, b: string | null, direction: SortDirection): number {
  const av = a ? Date.parse(a) : NaN;
  const bv = b ? Date.parse(b) : NaN;
  return compareMaybeNumber(Number.isFinite(av) ? av : null, Number.isFinite(bv) ? bv : null, direction);
}

function getSortedSweeps(sweeps: SweepSummary[]): SweepSummary[] {
  if (!sweepsSort.key) return sweeps.slice();
  const rows = sweeps.slice();
  rows.sort((a, b) => {
    switch (sweepsSort.key) {
      case "sweepId":
      case "status":
        return compareMaybeString(a[sweepsSort.key], b[sweepsSort.key], sweepsSort.direction);
      case "activeRuns":
        return compareMaybeNumber(a.activeRuns, b.activeRuns, sweepsSort.direction);
      case "runs":
        return compareMaybeNumber(a.runs.length, b.runs.length, sweepsSort.direction);
      case "bestAccuracy":
      case "lastTrainCe":
        return compareMaybeNumber(a[sweepsSort.key], b[sweepsSort.key], sweepsSort.direction);
      case "startedAt":
      case "endedAt":
        return compareMaybeDate(a[sweepsSort.key], b[sweepsSort.key], sweepsSort.direction);
    }
  });
  return rows;
}

function getSortedRuns(runs: RunSummary[]): RunSummary[] {
  const rows = runs.slice();
  if (!runsSort.key) return rows;
  rows.sort((a, b) => {
    switch (runsSort.key) {
      case "runId":
        return compareMaybeString(a.runId, b.runId, runsSort.direction);
      case "finalAccuracy":
      case "lastTrainCe":
      case "lastStep":
      case "lastStepsPerSec":
      case "numParams":
        return compareMaybeNumber(a[runsSort.key], b[runsSort.key], runsSort.direction);
      case "hasFinalCheckpoint":
        return compareBoolean(a.hasFinalCheckpoint, b.hasFinalCheckpoint, runsSort.direction);
    }
  });
  return rows;
}

function getSortedDocs(docs: DocSummary[]): DocSummary[] {
  if (!docsSort.key) return docs.slice();
  const rows = docs.slice();
  rows.sort((a, b) => {
    switch (docsSort.key) {
      case "title":
        return compareMaybeString(a.title, b.title, docsSort.direction);
      case "updatedAt":
        return compareMaybeDate(a.updatedAt, b.updatedAt, docsSort.direction);
      case "runRefs":
        return compareMaybeNumber(a.runRefs.length, b.runRefs.length, docsSort.direction);
      case "topAccuracy": {
        const aTop = a.mentionedAccuracies.length > 0 ? a.mentionedAccuracies[0] : null;
        const bTop = b.mentionedAccuracies.length > 0 ? b.mentionedAccuracies[0] : null;
        return compareMaybeNumber(aTop, bTop, docsSort.direction);
      }
    }
  });
  return rows;
}

function updateSortButton<K extends string>(id: string, sort: SortState<K>, key: K) {
  const button = $(id) as HTMLButtonElement;
  const label = button.dataset.label ?? button.textContent ?? "";
  button.textContent = sort.key === key ? `${label} (${sort.direction})` : label;
}

function renderSortButtons() {
  updateSortButton("sortSweepsSweep", sweepsSort, "sweepId");
  updateSortButton("sortSweepsStatus", sweepsSort, "status");
  updateSortButton("sortSweepsActive", sweepsSort, "activeRuns");
  updateSortButton("sortSweepsRuns", sweepsSort, "runs");
  updateSortButton("sortSweepsBestAcc", sweepsSort, "bestAccuracy");
  updateSortButton("sortSweepsLastTrainCe", sweepsSort, "lastTrainCe");
  updateSortButton("sortSweepsStarted", sweepsSort, "startedAt");
  updateSortButton("sortSweepsEnded", sweepsSort, "endedAt");

  updateSortButton("sortRunsRunId", runsSort, "runId");
  updateSortButton("sortRunsFinalAcc", runsSort, "finalAccuracy");
  updateSortButton("sortRunsLastTrainCe", runsSort, "lastTrainCe");
  updateSortButton("sortRunsLastStep", runsSort, "lastStep");
  updateSortButton("sortRunsStepsPerSec", runsSort, "lastStepsPerSec");
  updateSortButton("sortRunsParams", runsSort, "numParams");
  updateSortButton("sortRunsFinalCkpt", runsSort, "hasFinalCheckpoint");

  updateSortButton("sortDocsDoc", docsSort, "title");
  updateSortButton("sortDocsUpdated", docsSort, "updatedAt");
  updateSortButton("sortDocsRunRefs", docsSort, "runRefs");
  updateSortButton("sortDocsTopAcc", docsSort, "topAccuracy");
}

function syncSelectionToState(data: Bootstrap) {
  if (selectedSweep !== "__all__" && !data.sweeps.some((sweep) => sweep.sweepId === selectedSweep)) selectedSweep = "__all__";
  if (selectedRunId && !data.runs.some((run) => run.runId === selectedRunId)) {
    selectedRunId = null;
    runDetail = null;
    $("runDetailPanel").hidden = true;
  }
  if (selectedDocFile && !data.docs.some((doc) => doc.file === selectedDocFile)) {
    selectedDocFile = null;
    selectedDocUpdatedAt = null;
    $("docDetailPanel").hidden = true;
    $("docDetailOpen").hidden = true;
  }
}

function getDocSummary(data: Bootstrap, file: string): DocSummary | null {
  return data.docs.find((doc) => doc.file === file) ?? null;
}

function getQaThread(taskId: string): QaMessage[] {
  const thread = qaThreads.get(taskId);
  if (thread) return thread;
  const next: QaMessage[] = [];
  qaThreads.set(taskId, next);
  return next;
}

function renderHeader(data: Bootstrap) {
  const task = data.selectedTask;
  selectedTaskId = task.id;
  setTaskInUrl(task.id);
  document.title = `${task.title} Research Tracker`;
  $("taskTitle").textContent = task.title;
  $("taskMeta").textContent = task.description ? `${task.description} Docs: ${task.docsRoot}` : `Docs: ${task.docsRoot}`;
  $("updatedAt").textContent = `updated: ${data.generatedAt}`;
  $("docsHint").textContent = task.docsRoot;
  $("taskQaHint").textContent = `Answers use ${task.docsRoot}, ${task.scriptsRoot}, and ${task.logsRoot}.`;

  const select = $("taskSelect") as HTMLSelectElement;
  const prev = select.value || selectedTaskId;
  select.innerHTML = "";
  for (const item of data.tasks) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.title;
    select.appendChild(option);
  }
  select.value = data.tasks.some((item) => item.id === prev) ? prev : task.id;
}

function renderKpis(data: Bootstrap) {
  const kpis = $("kpis");
  const best = data.summary.bestRun;
  const items = [
    { label: "Docs", value: String(data.summary.docsCount) },
    { label: "Sweeps", value: String(data.summary.sweepsCount) },
    { label: "Runs", value: String(data.summary.runsCount) },
    { label: "Runs w/ Acc", value: String(data.summary.runsWithAccuracy) },
    {
      label: "Best Acc",
      value: best ? fmtAcc(best.finalAccuracy) : "-",
      subvalue: best?.runId ?? null,
      className: "kpi kpi-best",
    },
  ];
  kpis.innerHTML = "";
  for (const item of items) {
    const div = document.createElement("div");
    div.className = item.className ?? "kpi";
    div.innerHTML = `
      <div class="kpi-label">${item.label}</div>
      <div class="kpi-value">${item.value}</div>
      ${item.subvalue ? `<div class="kpi-subvalue mono muted">${item.subvalue}</div>` : ""}
    `;
    kpis.appendChild(div);
  }
}

function selectSweep(sweepId: string) {
  openSection("sweepsSection");
  openSection("runsSection");
  selectedSweep = sweepId;
  runsPage = 0;
  const select = $("sweepFilter") as HTMLSelectElement;
  select.value = sweepId;
  if (state) {
    renderSweeps(state);
    renderRuns(state);
  }
}

function selectRun(runId: string, scrollIntoView: boolean) {
  selectedRunId = runId;
  openSection("runsSection");
  openSection("runDetailPanel");
  if (state) {
    const ownerSweep = state.sweeps.find((sweep) => sweep.runs.some((run) => run.runId === runId));
    if (ownerSweep) {
      selectedSweep = ownerSweep.sweepId;
      const select = $("sweepFilter") as HTMLSelectElement;
      select.value = ownerSweep.sweepId;
    }
    renderRuns(state);
  }
  void loadRunDetail(runId, scrollIntoView);
}

function selectDoc(file: string, scrollIntoView: boolean) {
  openSection("docsSection");
  openSection("docDetailPanel");
  selectedDocFile = file;
  selectedDocUpdatedAt = null;
  if (state) renderDocs(state);
  void loadDocDetail(file, scrollIntoView);
}

function renderHighlights(data: Bootstrap) {
  const wrap = $("highlights");
  wrap.innerHTML = "";
  const bestRun = data.summary.bestRun ? data.runs.find((run) => run.runId === data.summary.bestRun?.runId) ?? null : null;
  const latestSweep = data.sweeps[0] ?? null;
  const latestDoc = data.docs[0] ?? null;
  const bestAcc = data.summary.bestRun?.finalAccuracy ?? null;
  const suspiciousRun =
    bestAcc === null
      ? null
      : data.runs
          .filter(
            (run) =>
              Number.isFinite(run.finalAccuracy ?? NaN) &&
              Number.isFinite(run.lastTrainCe ?? NaN) &&
              (run.finalAccuracy as number) < bestAcc * 0.92
          )
          .slice()
          .sort((a, b) => (a.lastTrainCe ?? Number.POSITIVE_INFINITY) - (b.lastTrainCe ?? Number.POSITIVE_INFINITY))[0] ?? null;

  const cards = [
    bestRun
      ? {
          eyebrow: "Best Run",
          title: bestRun.runId,
          body: `Final acc ${fmtAcc(bestRun.finalAccuracy)} with train CE ${fmtAcc(bestRun.lastTrainCe)}.`,
          action: "Open run",
          onClick: () => selectRun(bestRun.runId, true),
        }
      : null,
    latestSweep
      ? {
          eyebrow: "Freshest Sweep",
          title: latestSweep.sweepId,
          body: `${latestSweep.status} • ${latestSweep.runs.length} runs • best acc ${fmtAcc(latestSweep.bestAccuracy)}.`,
          action: "Filter runs",
          onClick: () => selectSweep(latestSweep.sweepId),
        }
      : null,
    suspiciousRun
      ? {
          eyebrow: "Oddball",
          title: suspiciousRun.runId,
          body: `Low train CE ${fmtAcc(suspiciousRun.lastTrainCe)} but final acc only ${fmtAcc(suspiciousRun.finalAccuracy)}.`,
          action: "Inspect run",
          onClick: () => selectRun(suspiciousRun.runId, true),
        }
      : null,
    latestDoc
      ? {
          eyebrow: "Latest Doc",
          title: latestDoc.title,
          body: `${latestDoc.file} • ${latestDoc.runRefs.length} run refs.`,
          action: "Open doc",
          onClick: () => selectDoc(latestDoc.file, true),
        }
      : null,
  ].filter((card): card is { eyebrow: string; title: string; body: string; action: string; onClick: () => void } => Boolean(card));

  for (const card of cards) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "highlight-card";
    button.innerHTML = `
      <div class="highlight-eyebrow">${card.eyebrow}</div>
      <div class="highlight-title">${card.title}</div>
      <div class="highlight-body">${card.body}</div>
      <div class="highlight-action mono">${card.action}</div>
    `;
    button.addEventListener("click", card.onClick);
    wrap.appendChild(button);
  }
}

function renderSweeps(data: Bootstrap) {
  const body = $("sweepsBody");
  body.innerHTML = "";
  const rows = getSortedSweeps(data.sweeps);
  const pages = Math.max(1, Math.ceil(rows.length / SWEEPS_PAGE_SIZE));
  sweepsPage = Math.min(sweepsPage, pages - 1);
  const pageSweeps = rows.slice(sweepsPage * SWEEPS_PAGE_SIZE, (sweepsPage + 1) * SWEEPS_PAGE_SIZE);
  for (const sweep of pageSweeps) {
    const tr = document.createElement("tr");
    tr.className = sweep.sweepId === selectedSweep ? "is-selected clickable-row" : "clickable-row";
    tr.innerHTML = `
      <td><code>${sweep.sweepId}</code></td>
      <td>${sweep.status}</td>
      <td>${String(sweep.activeRuns)}</td>
      <td>${sweep.runs.length}</td>
      <td>${fmtAcc(sweep.bestAccuracy)}</td>
      <td>${fmtAcc(sweep.lastTrainCe)}</td>
      <td>${sweep.startedAt ?? "-"}</td>
      <td>${sweep.endedAt ?? "-"}</td>
    `;
    tr.addEventListener("click", () => {
      selectSweep(sweep.sweepId);
    });
    body.appendChild(tr);
  }
  renderSweepsPager(rows.length);
}

function renderSweepsPager(totalSweeps: number) {
  const pages = Math.max(1, Math.ceil(totalSweeps / SWEEPS_PAGE_SIZE));
  const prevBtn = $("sweepsPrev") as HTMLButtonElement;
  const nextBtn = $("sweepsNext") as HTMLButtonElement;
  $("sweepsPageLabel").textContent = `Page ${sweepsPage + 1} / ${pages}`;
  prevBtn.disabled = sweepsPage === 0;
  nextBtn.disabled = sweepsPage >= pages - 1;
}

function renderSweepFilter(data: Bootstrap) {
  const select = $("sweepFilter") as HTMLSelectElement;
  const prev = selectedSweep || "__all__";
  select.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "__all__";
  allOpt.textContent = "All runs";
  select.appendChild(allOpt);
  for (const sweep of data.sweeps) {
    const option = document.createElement("option");
    option.value = sweep.sweepId;
    option.textContent = sweep.sweepId;
    select.appendChild(option);
  }
  select.value = data.sweeps.some((sweep) => sweep.sweepId === prev) ? prev : "__all__";
  selectedSweep = select.value;
}

function getFilteredRuns(data: Bootstrap): RunSummary[] {
  if (activeOnly) return data.runs.filter((run) => run.isActive);
  if (selectedSweep === "__all__") return data.runs;
  const sweep = data.sweeps.find((entry) => entry.sweepId === selectedSweep);
  return sweep ? sweep.runs : [];
}

function renderActiveToggle(data: Bootstrap) {
  const toggle = $("activeOnly") as HTMLInputElement;
  const count = data.runs.filter((run) => run.isActive).length;
  toggle.checked = activeOnly;
  $("activeCount").textContent = `${count} active`;
}

function renderRuns(data: Bootstrap) {
  const body = $("runsBody");
  body.innerHTML = "";
  const rows = getSortedRuns(getFilteredRuns(data));
  const pages = Math.max(1, Math.ceil(rows.length / RUNS_PAGE_SIZE));
  runsPage = Math.min(runsPage, pages - 1);
  const pageRows = rows.slice(runsPage * RUNS_PAGE_SIZE, (runsPage + 1) * RUNS_PAGE_SIZE);
  for (const run of pageRows) {
    const tr = document.createElement("tr");
    tr.className = run.runId === selectedRunId ? "is-selected clickable-row" : "clickable-row";
    tr.innerHTML = `
      <td><code>${run.runId}</code></td>
      <td>${fmtAcc(run.finalAccuracy)}</td>
      <td>${fmtAcc(run.lastTrainCe)}</td>
      <td>${fmtNum(run.lastStep)}</td>
      <td>${fmtNum(run.lastStepsPerSec)}</td>
      <td>${fmtParams(run.numParams, run.trainableParams)}</td>
      <td>${run.hasFinalCheckpoint ? "yes" : "no"}</td>
    `;
    tr.addEventListener("click", () => {
      selectRun(run.runId, true);
    });
    body.appendChild(tr);
  }
  renderRunsPager(rows.length);
}

function renderRunsPager(totalRuns: number) {
  const pages = Math.max(1, Math.ceil(totalRuns / RUNS_PAGE_SIZE));
  const prevBtn = $("runsPrev") as HTMLButtonElement;
  const nextBtn = $("runsNext") as HTMLButtonElement;
  $("runsPageLabel").textContent = `Page ${runsPage + 1} / ${pages}`;
  prevBtn.disabled = runsPage === 0;
  nextBtn.disabled = runsPage >= pages - 1;
}

function renderDocs(data: Bootstrap) {
  const body = $("docsBody");
  body.innerHTML = "";
  const rows = getSortedDocs(data.docs);
  const pages = Math.max(1, Math.ceil(rows.length / DOCS_PAGE_SIZE));
  docsPage = Math.min(docsPage, pages - 1);
  const pageDocs = rows.slice(docsPage * DOCS_PAGE_SIZE, (docsPage + 1) * DOCS_PAGE_SIZE);
  for (const doc of pageDocs) {
    const topAcc = doc.mentionedAccuracies.length > 0 ? doc.mentionedAccuracies[0].toFixed(4) : "-";
    const tr = document.createElement("tr");
    tr.className = doc.file === selectedDocFile ? "is-selected clickable-row" : "clickable-row";
    tr.innerHTML = `
      <td>
        <strong>${doc.title}</strong><br />
        <span class="mono muted">${doc.file}</span>
      </td>
      <td class="mono">${doc.updatedAt}</td>
      <td>${doc.runRefs.length}</td>
      <td>${topAcc}</td>
    `;
    tr.addEventListener("click", () => {
      selectDoc(doc.file, true);
    });
    body.appendChild(tr);
  }
  renderDocsPager(rows.length);
}

function renderDocsPager(totalDocs: number) {
  const pages = Math.max(1, Math.ceil(totalDocs / DOCS_PAGE_SIZE));
  const prevBtn = $("docsPrev") as HTMLButtonElement;
  const nextBtn = $("docsNext") as HTMLButtonElement;
  $("docsPageLabel").textContent = `Page ${docsPage + 1} / ${pages}`;
  prevBtn.disabled = docsPage === 0;
  nextBtn.disabled = docsPage >= pages - 1;
}

function renderTaskQa(data: Bootstrap) {
  const taskId = data.selectedTask.id;
  const starterWrap = $("taskQaStarters");
  const threadWrap = $("taskQaThread");
  const status = $("taskQaStatus");
  const thread = getQaThread(taskId);
  const starterPrompts = [
    "Why does this task look weak right now?",
    "What should I try next?",
  ];

  starterWrap.innerHTML = "";
  for (const prompt of starterPrompts) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost-button";
    button.textContent = prompt;
    button.disabled = taskQaLoadInFlight;
    button.addEventListener("click", () => {
      void submitTaskQa(prompt);
    });
    starterWrap.appendChild(button);
  }

  threadWrap.innerHTML = "";
  if (thread.length === 0) {
    threadWrap.innerHTML = `<div class="qa-empty muted">Ask about the whole task. The answer is grounded in current sweeps, runs, docs, and task scripts.</div>`;
  } else {
    for (const message of thread) {
      const bubble = document.createElement("div");
      bubble.className = `qa-message qa-${message.role}${message.pending ? " is-pending" : ""}${message.error ? " is-error" : ""}`;
      const meta = document.createElement("div");
      meta.className = "qa-meta mono muted";
      meta.textContent = message.role === "user" ? "You" : message.pending ? "Agent thinking..." : `Agent${message.generatedAt ? ` • ${message.generatedAt}` : ""}`;
      const body = document.createElement("div");
      body.className = "qa-content";
      body.textContent = message.content;
      bubble.appendChild(meta);
      bubble.appendChild(body);
      if (message.evidence && message.evidence.length > 0) {
        const evidence = document.createElement("div");
        evidence.className = "qa-evidence";
        evidence.innerHTML = `<div class="detail-section-title">Evidence</div><ul>${message.evidence.map((item) => `<li>${item}</li>`).join("")}</ul>`;
        bubble.appendChild(evidence);
      }
      threadWrap.appendChild(bubble);
    }
  }

  status.textContent = taskQaLoadInFlight ? "asking..." : "";
}

function renderAll(data: Bootstrap) {
  syncSelectionToState(data);
  renderHeader(data);
  renderSortButtons();
  renderKpis(data);
  renderHighlights(data);
  renderTaskQa(data);
  renderSweeps(data);
  renderSweepFilter(data);
  renderActiveToggle(data);
  renderRuns(data);
  renderDocs(data);
}

async function load() {
  if (bootstrapLoadInFlight) return;
  bootstrapLoadInFlight = true;
  const requestedTaskId = getActiveTaskId();
  try {
    const res = await fetch(apiUrl("/api/bootstrap"), { cache: "no-store" });
    if (!res.ok) throw new Error(`bootstrap failed: ${res.status}`);
    const data = (await res.json()) as Bootstrap;
    state = data;
    selectedTaskId = data.selectedTask.id;
    renderAll(data);
    if (selectedRunId) await loadRunDetail(selectedRunId, false);
    if (selectedDocFile) {
      const doc = getDocSummary(data, selectedDocFile);
      if (doc && doc.updatedAt !== selectedDocUpdatedAt) await loadDocDetail(selectedDocFile, false);
    }
  } finally {
    bootstrapLoadInFlight = false;
    if (requestedTaskId && requestedTaskId !== getActiveTaskId()) {
      // task switched while loading; leave current state alone
    }
  }
}

async function loadRunDetail(runId: string, scrollIntoView: boolean) {
  if (runDetailLoadInFlight) return;
  runDetailLoadInFlight = true;
  const taskId = getActiveTaskId();
  const panel = $("runDetailPanel");
  const title = $("runDetailTitle");
  const body = $("runDetailBody");
  panel.hidden = false;
  if (!runDetail || runDetail.run.runId !== runId) {
    title.textContent = runId;
    body.innerHTML = `<div class="muted mono">Loading run detail...</div>`;
  }
  try {
    const res = await fetch(apiUrl("/api/run", { runId }), { cache: "no-store" });
    if (!res.ok) throw new Error(`run detail failed: ${res.status}`);
    const detail = (await res.json()) as RunDetail;
    if (selectedRunId !== runId || taskId !== getActiveTaskId()) return;
    runDetail = detail;
    renderRunDetail(detail);
    if (scrollIntoView) panel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    if (selectedRunId !== runId || taskId !== getActiveTaskId()) return;
    body.innerHTML = `<div class="muted mono">Error: ${String(error)}</div>`;
  } finally {
    runDetailLoadInFlight = false;
  }
}

function renderRunDetail(detail: RunDetail) {
  const panel = $("runDetailPanel");
  const title = $("runDetailTitle");
  const body = $("runDetailBody");
  panel.hidden = false;
  openSection("runDetailPanel");
  title.textContent = detail.run.runId;
  body.innerHTML = "";

  const summary = document.createElement("div");
  summary.className = "detail-grid";
  const cards = [
    ["Status", detail.run.isActive ? "active" : detail.run.hasFinalCheckpoint ? "finished" : "idle"],
    ["Updated", detail.updatedAt ?? "-"],
    ["Final Acc", fmtAcc(detail.run.finalAccuracy)],
    ["Best Acc", fmtAcc(detail.run.bestAccuracy)],
    ["Last Train CE", fmtAcc(detail.run.lastTrainCe)],
    ["Last Step", fmtNum(detail.run.lastStep)],
    ["Steps/s", fmtNum(detail.run.lastStepsPerSec)],
    ["Params", fmtParams(detail.run.numParams, detail.run.trainableParams)],
    ["Val Acc", fmtAcc(detail.latestEvalAccuracy)],
    ["Val Steps/s", fmtRate(detail.latestValStepsPerSec)],
  ];
  for (const [label, value] of cards) {
    const item = document.createElement("div");
    item.className = "detail-card";
    item.innerHTML = `<div class="detail-label">${label}</div><div class="detail-value">${value}</div>`;
    summary.appendChild(item);
  }
  body.appendChild(summary);

  const charts = document.createElement("div");
  charts.className = "charts-grid";
  charts.appendChild(makeChartCard("Train CE", detail.trainCeSeries, "ce"));
  if (detail.valAccuracySeries.length > 0) charts.appendChild(makeChartCard("Val Acc", detail.valAccuracySeries, "acc"));
  body.appendChild(charts);

  if (detail.introLines.length > 0) {
    const introWrap = document.createElement("div");
    introWrap.className = "detail-tail";
    introWrap.innerHTML = `<div class="detail-section-title">Log Start</div>`;
    const pre = document.createElement("pre");
    pre.className = "detail-pre";
    pre.textContent = detail.introLines.join("\n");
    introWrap.appendChild(pre);
    body.appendChild(introWrap);
  }

  if (detail.tailLines.length > 0) {
    const tailWrap = document.createElement("div");
    tailWrap.className = "detail-tail";
    tailWrap.innerHTML = `<div class="detail-section-title">Log Tail</div>`;
    const pre = document.createElement("pre");
    pre.className = "detail-pre";
    pre.textContent = detail.tailLines.join("\n");
    tailWrap.appendChild(pre);
    body.appendChild(tailWrap);
  }
}

function makeChartCard(title: string, points: SeriesPoint[], kind: "acc" | "ce" | "rate") {
  const card = document.createElement("div");
  card.className = "chart-card";
  if (points.length === 0) {
    card.innerHTML = `<div class="detail-section-title">${title}</div><div class="muted mono">No data</div>`;
    return card;
  }
  const path = buildChartPath(points, 320, 140);
  const values = points.map((point) => point.value);
  const latest = values[values.length - 1];
  const best = kind === "acc" || kind === "rate" ? Math.max(...values) : Math.min(...values);
  card.innerHTML = `
    <div class="chart-head">
      <div class="detail-section-title">${title}</div>
      <div class="mono muted">latest ${latest.toFixed(4)} | ${kind === "ce" ? "min" : "best"} ${best.toFixed(4)}</div>
    </div>
    <svg class="chart-svg" viewBox="0 0 320 140" preserveAspectRatio="none" aria-label="${title}">
      <path class="chart-baseline" d="M 0 132 H 320"></path>
      <path class="chart-line ${kind === "acc" ? "chart-line-acc" : kind === "rate" ? "chart-line-rate" : "chart-line-ce"}" d="${path}"></path>
    </svg>
  `;
  return card;
}

function buildChartPath(points: SeriesPoint[], width: number, height: number) {
  const steps = points.map((point) => point.step);
  const values = points.map((point) => point.value);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const xSpan = Math.max(1, maxStep - minStep);
  const ySpan = Math.max(1e-9, maxVal - minVal);
  return points
    .map((point, index) => {
      const x = ((point.step - minStep) / xSpan) * width;
      const y = height - 8 - ((point.value - minVal) / ySpan) * (height - 16);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

async function loadDocDetail(file: string, scrollIntoView: boolean) {
  const taskId = getActiveTaskId();
  const panel = $("docDetailPanel");
  const title = $("docDetailTitle");
  const body = $("docDetailBody");
  panel.hidden = false;
  title.textContent = file;
  body.innerHTML = `<div class="muted mono">Loading doc...</div>`;
  try {
    const res = await fetch(apiUrl("/api/doc", { file }), { cache: "no-store" });
    if (!res.ok) throw new Error(`doc failed: ${res.status}`);
    const markdown = await res.text();
    if (selectedDocFile !== file || taskId !== getActiveTaskId()) return;
    selectedDocUpdatedAt = getDocSummary(state as Bootstrap, file)?.updatedAt ?? selectedDocUpdatedAt;
    renderDocDetail(file, markdown);
    if (scrollIntoView) panel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    if (selectedDocFile !== file || taskId !== getActiveTaskId()) return;
    body.innerHTML = `<div class="muted mono">Error: ${String(error)}</div>`;
  }
}

function renderDocDetail(file: string, markdown: string) {
  const panel = $("docDetailPanel");
  const title = $("docDetailTitle");
  const body = $("docDetailBody");
  const openBtn = $("docDetailOpen") as HTMLButtonElement;
  panel.hidden = false;
  openSection("docDetailPanel");
  title.textContent = file;
  openBtn.hidden = false;
  openBtn.onclick = () => openDocInNewTab(file);
  body.innerHTML = `<article class="markdown-body doc-render">${marked.parse(markdown, { async: false }) as string}</article>`;
}

function renderTaskQaError(taskId: string, message: string) {
  const thread = getQaThread(taskId);
  const pendingIndex = thread.findIndex((item) => item.pending);
  const replacement: QaMessage = {
    role: "assistant",
    content: message,
    error: true,
    generatedAt: new Date().toISOString(),
  };
  if (pendingIndex >= 0) thread.splice(pendingIndex, 1, replacement);
  else thread.push(replacement);
}

async function submitTaskQa(question: string) {
  const taskId = getActiveTaskId();
  if (!taskId || !state || taskQaLoadInFlight) return;
  const trimmed = question.trim();
  if (!trimmed) return;
  const input = $("taskQaInput") as HTMLTextAreaElement;
  input.value = "";
  const thread = getQaThread(taskId);
  const conversation: ChatTurn[] = thread
    .filter((item) => !item.pending)
    .map((item) => ({ role: item.role, content: item.content }));
  thread.push({ role: "user", content: trimmed, generatedAt: new Date().toISOString() });
  thread.push({ role: "assistant", content: "Thinking...", pending: true });
  taskQaLoadInFlight = true;
  renderTaskQa(state);

  try {
    const res = await fetch("/api/qa/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        taskId,
        question: trimmed,
        conversation,
      }),
    });
    const body = (await res.json()) as TaskQaResponse | { error?: string };
    if (!res.ok) throw new Error(body && "error" in body ? body.error ?? `qa failed: ${res.status}` : `qa failed: ${res.status}`);
    const pendingIndex = thread.findIndex((item) => item.pending);
    const answer = body as TaskQaResponse;
    const replacement: QaMessage = {
      role: "assistant",
      content: answer.answer,
      evidence: answer.evidence,
      generatedAt: answer.generatedAt,
    };
    if (pendingIndex >= 0) thread.splice(pendingIndex, 1, replacement);
    else thread.push(replacement);
  } catch (error) {
    renderTaskQaError(taskId, `Task QA failed: ${String(error)}`);
  } finally {
    taskQaLoadInFlight = false;
    if (state && state.selectedTask.id === taskId) renderTaskQa(state);
  }
}

function setup() {
  setupCollapsibles();
  const taskSelect = $("taskSelect") as HTMLSelectElement;
  taskSelect.addEventListener("change", () => {
    if (!taskSelect.value || taskSelect.value === getActiveTaskId()) return;
    selectedTaskId = taskSelect.value;
    resetTaskScopedUi();
    void load().catch((error) => {
      $("updatedAt").textContent = `error: ${String(error)}`;
    });
  });

  const taskQaForm = $("taskQaForm") as HTMLFormElement;
  taskQaForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("taskQaInput") as HTMLTextAreaElement;
    void submitTaskQa(input.value);
  });

  const sweepsPrevBtn = $("sweepsPrev") as HTMLButtonElement;
  const sweepsNextBtn = $("sweepsNext") as HTMLButtonElement;
  const sweepSortButtons: Array<[string, SweepSortKey]> = [
    ["sortSweepsSweep", "sweepId"],
    ["sortSweepsStatus", "status"],
    ["sortSweepsActive", "activeRuns"],
    ["sortSweepsRuns", "runs"],
    ["sortSweepsBestAcc", "bestAccuracy"],
    ["sortSweepsLastTrainCe", "lastTrainCe"],
    ["sortSweepsStarted", "startedAt"],
    ["sortSweepsEnded", "endedAt"],
  ];
  for (const [id, key] of sweepSortButtons) {
    ($(id) as HTMLButtonElement).addEventListener("click", () => {
      toggleSort(sweepsSort, key);
      sweepsPage = 0;
      if (state) {
        renderSortButtons();
        renderSweeps(state);
      }
    });
  }
  sweepsPrevBtn.addEventListener("click", () => {
    if (!state || sweepsPage === 0) return;
    sweepsPage -= 1;
    renderSweeps(state);
  });
  sweepsNextBtn.addEventListener("click", () => {
    if (!state) return;
    const pages = Math.max(1, Math.ceil(state.sweeps.length / SWEEPS_PAGE_SIZE));
    if (sweepsPage >= pages - 1) return;
    sweepsPage += 1;
    renderSweeps(state);
  });

  const sweepFilter = $("sweepFilter") as HTMLSelectElement;
  sweepFilter.addEventListener("change", () => {
    selectedSweep = sweepFilter.value;
    runsPage = 0;
    if (state) renderRuns(state);
  });

  const activeToggle = $("activeOnly") as HTMLInputElement;
  activeToggle.addEventListener("change", () => {
    activeOnly = activeToggle.checked;
    runsPage = 0;
    if (state) renderRuns(state);
  });

  const closeRunDetailBtn = $("runDetailClose") as HTMLButtonElement;
  closeRunDetailBtn.addEventListener("click", () => {
    selectedRunId = null;
    runDetail = null;
    $("runDetailPanel").hidden = true;
    if (state) renderRuns(state);
  });

  const closeDocBtn = $("docDetailClose") as HTMLButtonElement;
  closeDocBtn.addEventListener("click", () => {
    selectedDocFile = null;
    selectedDocUpdatedAt = null;
    $("docDetailPanel").hidden = true;
    $("docDetailOpen").hidden = true;
    if (state) renderDocs(state);
  });

  const runsPrevBtn = $("runsPrev") as HTMLButtonElement;
  const runsNextBtn = $("runsNext") as HTMLButtonElement;
  const runSortButtons: Array<[string, RunSortKey]> = [
    ["sortRunsRunId", "runId"],
    ["sortRunsFinalAcc", "finalAccuracy"],
    ["sortRunsLastTrainCe", "lastTrainCe"],
    ["sortRunsLastStep", "lastStep"],
    ["sortRunsStepsPerSec", "lastStepsPerSec"],
    ["sortRunsParams", "numParams"],
    ["sortRunsFinalCkpt", "hasFinalCheckpoint"],
  ];
  for (const [id, key] of runSortButtons) {
    ($(id) as HTMLButtonElement).addEventListener("click", () => {
      toggleSort(runsSort, key);
      runsPage = 0;
      if (state) {
        renderSortButtons();
        renderRuns(state);
      }
    });
  }
  runsPrevBtn.addEventListener("click", () => {
    if (!state || runsPage === 0) return;
    runsPage -= 1;
    renderRuns(state);
  });
  runsNextBtn.addEventListener("click", () => {
    if (!state) return;
    const pages = Math.max(1, Math.ceil(getFilteredRuns(state).length / RUNS_PAGE_SIZE));
    if (runsPage >= pages - 1) return;
    runsPage += 1;
    renderRuns(state);
  });

  const docsPrevBtn = $("docsPrev") as HTMLButtonElement;
  const docsNextBtn = $("docsNext") as HTMLButtonElement;
  const docSortButtons: Array<[string, DocSortKey]> = [
    ["sortDocsDoc", "title"],
    ["sortDocsUpdated", "updatedAt"],
    ["sortDocsRunRefs", "runRefs"],
    ["sortDocsTopAcc", "topAccuracy"],
  ];
  for (const [id, key] of docSortButtons) {
    ($(id) as HTMLButtonElement).addEventListener("click", () => {
      toggleSort(docsSort, key);
      docsPage = 0;
      if (state) {
        renderSortButtons();
        renderDocs(state);
      }
    });
  }
  docsPrevBtn.addEventListener("click", () => {
    if (!state || docsPage === 0) return;
    docsPage -= 1;
    renderDocs(state);
  });
  docsNextBtn.addEventListener("click", () => {
    if (!state) return;
    const pages = Math.max(1, Math.ceil(state.docs.length / DOCS_PAGE_SIZE));
    if (docsPage >= pages - 1) return;
    docsPage += 1;
    renderDocs(state);
  });
}

setup();
load().catch((error) => {
  $("updatedAt").textContent = `error: ${String(error)}`;
});

setInterval(() => {
  load().catch(() => {
    // keep old state on transient failures
  });
}, BOOTSTRAP_REFRESH_MS);

setInterval(() => {
  if (!selectedRunId) return;
  loadRunDetail(selectedRunId, false).catch(() => {
    // keep old detail on transient failures
  });
}, RUN_DETAIL_REFRESH_MS);
