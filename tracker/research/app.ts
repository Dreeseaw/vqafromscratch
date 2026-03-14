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
  isEvalOnly: boolean;
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
const SHORT_DATE_TIME = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});
const FULL_DATE_TIME = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  year: "numeric",
  hour: "numeric",
  minute: "2-digit",
});
const fmtParams = (total: number | null, trainable: number | null) => {
  if (!Number.isFinite(total ?? NaN)) return "-";
  const totalStr = Intl.NumberFormat("en-US").format(total as number);
  if (!Number.isFinite(trainable ?? NaN) || !total || total <= 0) return totalStr;
  return `${totalStr} (${fmtPct(((trainable as number) / (total as number)) * 100)})`;
};

function fmtDateTime(value: string | null, style: "short" | "full" = "short"): string {
  if (!value) return "-";
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) return value;
  const date = new Date(ms);
  const now = new Date();
  const formatter = style === "full" || date.getFullYear() !== now.getFullYear() ? FULL_DATE_TIME : SHORT_DATE_TIME;
  return formatter.format(date);
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

const SWEEPS_PAGE_SIZE = 5;
const RUNS_PAGE_SIZE = 10;
const DOCS_PAGE_SIZE = 5;
const BOOTSTRAP_REFRESH_MS = 10000;
const RUN_DETAIL_REFRESH_MS = 10000;
const IDEA_TREE_STATUS_POLL_MS = 1000;
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
let ideaTreeLoadInFlight = false;
let sweepsSort: SortState<SweepSortKey> = { key: null, direction: "asc" };
let runsSort: SortState<RunSortKey> = { key: "finalAccuracy", direction: "desc" };
let docsSort: SortState<DocSortKey> = { key: null, direction: "asc" };
let ideaTree: IdeaTreeResponse | null = null;
let ideaTreeError: string | null = null;
let ideaTreeProgress: IdeaTreeProgressStatus | null = null;
let ideaTreeEdgeRenderFrame: number | null = null;
let ideaTreeStatusPollTimer: number | null = null;
let showRunLogHead = true;
let showRunLogTail = true;
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

function stopIdeaTreeStatusPolling() {
  if (ideaTreeStatusPollTimer !== null) {
    window.clearInterval(ideaTreeStatusPollTimer);
    ideaTreeStatusPollTimer = null;
  }
}

async function pollIdeaTreeStatus(taskId: string) {
  try {
    const res = await fetch(apiUrl("/api/ideas/tree/status", { task: taskId }), { cache: "no-store" });
    if (!res.ok) throw new Error(`idea tree status failed: ${res.status}`);
    const status = (await res.json()) as IdeaTreeProgressStatus;
    if (taskId !== getActiveTaskId()) return;
    ideaTreeProgress = status;
    if (state) renderIdeaTree(state);
  } catch {
    // keep existing loading UI if status polling fails
  }
}

function startIdeaTreeStatusPolling(taskId: string) {
  stopIdeaTreeStatusPolling();
  void pollIdeaTreeStatus(taskId);
  ideaTreeStatusPollTimer = window.setInterval(() => {
    void pollIdeaTreeStatus(taskId);
  }, IDEA_TREE_STATUS_POLL_MS);
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
  if (!collapsed && sectionId === "ideaTreeSection") scheduleIdeaTreeEdgeRender();
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
  stopIdeaTreeStatusPolling();
  selectedSweep = "__all__";
  activeOnly = false;
  sweepsPage = 0;
  runsPage = 0;
  docsPage = 0;
  selectedRunId = null;
  runDetail = null;
  selectedDocFile = null;
  selectedDocUpdatedAt = null;
  ideaTree = null;
  ideaTreeError = null;
  ideaTreeProgress = null;
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
  $("updatedAt").textContent = `Updated ${fmtDateTime(data.generatedAt, "full")}`;
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

function sortIdeaNodes(nodes: IdeaTreeNode[]): IdeaTreeNode[] {
  return nodes.slice().sort((a, b) => a.label.localeCompare(b.label));
}

function formatIdeaKind(kind: IdeaTreeNodeKind): string {
  return kind[0].toUpperCase() + kind.slice(1);
}

function makeIdeaChipListHtml(values: string[], className: string): string {
  return values.map((value) => `<span class="${className} mono">${escapeHtml(value)}</span>`).join("");
}

function makeIdeaSourceListHtml(sourceRefs: string[]): string {
  if (sourceRefs.length === 0) return `<span class="muted">No explicit evidence refs</span>`;
  return makeIdeaChipListHtml(sourceRefs, "idea-source-chip");
}

function positionIdeaTreePopover(popover: HTMLElement, shell: HTMLElement, anchor: { x: number; y: number }) {
  const shellRect = shell.getBoundingClientRect();
  const popoverWidth = popover.offsetWidth;
  const popoverHeight = popover.offsetHeight;
  const maxLeft = Math.max(12, shellRect.width - popoverWidth - 12);
  const maxTop = Math.max(12, shellRect.height - popoverHeight - 12);
  const left = Math.min(Math.max(12, anchor.x + 14), maxLeft);
  const top = Math.min(Math.max(12, anchor.y + 14), maxTop);
  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;
}

function showIdeaTreePopover(
  payload: {
    eyebrow: string;
    title: string;
    summary: string;
    detail?: string;
    aliases?: string[];
    evidenceRefs: string[];
  },
  opts: {
    clientX?: number;
    clientY?: number;
    anchorEl?: Element | null;
  } = {}
) {
  const shell = document.getElementById("ideaTreeShell") as HTMLElement | null;
  const popover = document.getElementById("ideaTreePopover") as HTMLElement | null;
  if (!shell || !popover) return;
  popover.innerHTML = `
    <div class="idea-popover-eyebrow">${escapeHtml(payload.eyebrow)}</div>
    <div class="idea-popover-title">${escapeHtml(payload.title)}</div>
    <div class="idea-popover-summary">${escapeHtml(payload.summary)}</div>
    ${payload.detail ? `<div class="idea-popover-detail">${escapeHtml(payload.detail)}</div>` : ""}
    ${payload.aliases && payload.aliases.length > 0 ? `<div class="idea-popover-section"><div class="idea-popover-label">Aliases</div><div class="idea-chip-row">${makeIdeaChipListHtml(payload.aliases, "idea-alias-chip")}</div></div>` : ""}
    <div class="idea-popover-section"><div class="idea-popover-label">Evidence</div><div class="idea-chip-row">${makeIdeaSourceListHtml(payload.evidenceRefs)}</div></div>
  `;
  popover.hidden = false;
  const shellRect = shell.getBoundingClientRect();
  const anchorRect = opts.anchorEl?.getBoundingClientRect();
  const anchor =
    Number.isFinite(opts.clientX ?? NaN) && Number.isFinite(opts.clientY ?? NaN)
      ? {
          x: (opts.clientX as number) - shellRect.left,
          y: (opts.clientY as number) - shellRect.top,
        }
      : anchorRect
        ? {
            x: anchorRect.left - shellRect.left + anchorRect.width / 2,
            y: anchorRect.top - shellRect.top + anchorRect.height / 2,
          }
        : {
            x: 24,
            y: 24,
          };
  positionIdeaTreePopover(popover, shell, anchor);
}

function hideIdeaTreePopover() {
  const popover = document.getElementById("ideaTreePopover") as HTMLElement | null;
  if (!popover) return;
  popover.hidden = true;
}

function createIdeaNodeCard(node: IdeaTreeNode, opts: { isAnchor?: boolean } = {}): HTMLElement {
  const card = document.createElement("article");
  card.className = `idea-node idea-node-kind-${node.kind}${opts.isAnchor ? " is-anchor" : ""}`;
  card.dataset.nodeId = node.id;
  card.tabIndex = 0;
  card.innerHTML = `
    <div class="idea-node-kicker mono">${opts.isAnchor ? `Anchor • ${formatIdeaKind(node.kind)}` : formatIdeaKind(node.kind)}</div>
    <div class="idea-node-title">${escapeHtml(node.label)}</div>
    <div class="idea-node-summary">${escapeHtml(node.summary)}</div>
    <div class="idea-node-footer mono">${node.evidenceRefs.length} refs${node.aliases.length > 0 ? ` • ${node.aliases.length} aliases` : ""}</div>
  `;
  const show = (event?: MouseEvent) =>
    showIdeaTreePopover(
      {
        eyebrow: opts.isAnchor ? `Anchor concept • ${formatIdeaKind(node.kind)}` : formatIdeaKind(node.kind),
        title: node.label,
        summary: node.summary,
        detail: node.detail,
        aliases: node.aliases,
        evidenceRefs: node.evidenceRefs,
      },
      {
        clientX: event?.clientX,
        clientY: event?.clientY,
        anchorEl: card,
      }
    );
  card.addEventListener("mouseenter", (event) => show(event));
  card.addEventListener("mousemove", (event) => show(event));
  card.addEventListener("mouseleave", () => hideIdeaTreePopover());
  card.addEventListener("focus", () => show());
  card.addEventListener("blur", () => hideIdeaTreePopover());
  return card;
}

function buildIdeaTreeLayout(payload: IdeaTreeResponse) {
  const nodesById = new Map(payload.graph.nodes.map((node) => [node.id, node] as const));
  const outgoing = new Map<string, string[]>();
  for (const edge of payload.graph.edges) {
    const group = outgoing.get(edge.fromId);
    if (group) group.push(edge.toId);
    else outgoing.set(edge.fromId, [edge.toId]);
  }
  const anchorIds = payload.anchors.filter((anchorId) => nodesById.has(anchorId));
  const anchorOrder = new Map(anchorIds.map((anchorId, index) => [anchorId, index] as const));
  const assignments = new Map<string, { anchorId: string; depth: number; anchorOrder: number }>();

  for (const anchorId of anchorIds) {
    const queue: Array<{ id: string; depth: number }> = [{ id: anchorId, depth: 0 }];
    const seen = new Set<string>();
    while (queue.length > 0) {
      const current = queue.shift() as { id: string; depth: number };
      if (seen.has(current.id)) continue;
      seen.add(current.id);
      const nextAssignment = {
        anchorId,
        depth: current.depth,
        anchorOrder: anchorOrder.get(anchorId) ?? 0,
      };
      const existing = assignments.get(current.id);
      if (
        !existing ||
        nextAssignment.depth < existing.depth ||
        (nextAssignment.depth === existing.depth && nextAssignment.anchorOrder < existing.anchorOrder)
      ) {
        assignments.set(current.id, nextAssignment);
      }
      for (const nextId of outgoing.get(current.id) ?? []) queue.push({ id: nextId, depth: current.depth + 1 });
    }
  }

  const columns = anchorIds.map((anchorId) => ({
    anchorId,
    anchor: nodesById.get(anchorId) as IdeaTreeNode,
    depthGroups: new Map<number, IdeaTreeNode[]>(),
  }));
  const columnsByAnchor = new Map(columns.map((column) => [column.anchorId, column] as const));
  const overflow: IdeaTreeNode[] = [];
  for (const node of nodesById.values()) {
    const assignment = assignments.get(node.id);
    if (!assignment) {
      overflow.push(node);
      continue;
    }
    if (assignment.depth === 0) continue;
    const column = columnsByAnchor.get(assignment.anchorId);
    if (!column) {
      overflow.push(node);
      continue;
    }
    const bucket = column.depthGroups.get(assignment.depth);
    if (bucket) bucket.push(node);
    else column.depthGroups.set(assignment.depth, [node]);
  }

  return {
    columns,
    overflow: sortIdeaNodes(overflow),
    assignments,
  };
}

function renderIdeaTreeGraph(container: HTMLElement, payload: IdeaTreeResponse) {
  const layout = buildIdeaTreeLayout(payload);
  const shell = document.createElement("div");
  shell.id = "ideaTreeShell";
  shell.className = "idea-tree-shell";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("id", "ideaTreeEdgeLayer");
  svg.setAttribute("class", "idea-tree-edge-layer");
  svg.setAttribute("aria-hidden", "true");

  const columns = document.createElement("div");
  columns.className = "idea-tree-columns";

  const popover = document.createElement("div");
  popover.id = "ideaTreePopover";
  popover.className = "idea-tree-popover";
  popover.hidden = true;
  shell.addEventListener("mouseleave", () => hideIdeaTreePopover());

  for (const column of layout.columns) {
    const columnEl = document.createElement("div");
    columnEl.className = "idea-anchor-column";
    columnEl.dataset.anchorId = column.anchorId;

    const header = document.createElement("div");
    header.className = "idea-column-label mono";
    header.textContent = "Anchor";
    columnEl.appendChild(header);
    columnEl.appendChild(createIdeaNodeCard(column.anchor, { isAnchor: true }));

    for (const depth of [...column.depthGroups.keys()].sort((a, b) => a - b)) {
      const depthGroup = document.createElement("div");
      depthGroup.className = "idea-depth-group";
      const depthLabel = document.createElement("div");
      depthLabel.className = "idea-depth-label mono";
      depthLabel.textContent = `Layer ${depth}`;
      const depthGrid = document.createElement("div");
      depthGrid.className = "idea-depth-grid";
      for (const node of sortIdeaNodes(column.depthGroups.get(depth) ?? [])) {
        depthGrid.appendChild(createIdeaNodeCard(node));
      }
      depthGroup.appendChild(depthLabel);
      depthGroup.appendChild(depthGrid);
      columnEl.appendChild(depthGroup);
    }
    columns.appendChild(columnEl);
  }

  if (layout.overflow.length > 0) {
    const overflowEl = document.createElement("div");
    overflowEl.className = "idea-anchor-column idea-overflow-column";
    const header = document.createElement("div");
    header.className = "idea-column-label mono";
    header.textContent = "Related";
    const depthGrid = document.createElement("div");
    depthGrid.className = "idea-depth-grid";
    for (const node of layout.overflow) depthGrid.appendChild(createIdeaNodeCard(node));
    overflowEl.appendChild(header);
    overflowEl.appendChild(depthGrid);
    columns.appendChild(overflowEl);
  }

  shell.appendChild(svg);
  shell.appendChild(columns);
  shell.appendChild(popover);
  container.replaceChildren(shell, buildIdeaTreeDebug(payload));
  scheduleIdeaTreeEdgeRender();
}

function formatIdeaGraphMs(ms: number): string {
  if (!Number.isFinite(ms)) return "-";
  return `${Math.round(ms)}ms`;
}

function makeIdeaDebugCardHtml(
  title: string,
  body: string,
  refs: string[] = [],
  eyebrow?: string
): string {
  return `
    <article class="idea-debug-card">
      ${eyebrow ? `<div class="idea-debug-eyebrow mono">${escapeHtml(eyebrow)}</div>` : ""}
      <div class="idea-debug-title">${escapeHtml(title)}</div>
      <div class="idea-debug-body">${escapeHtml(body)}</div>
      ${refs.length > 0 ? `<div class="idea-chip-row">${makeIdeaChipListHtml(refs, "idea-source-chip")}</div>` : ""}
    </article>
  `;
}

function buildIdeaTreeDebug(payload: IdeaTreeResponse): HTMLElement {
  const debug = document.createElement("div");
  debug.className = "idea-debug";
  const timings = payload.debug.timings;
  debug.innerHTML = `
    <div class="idea-debug-summary">
      <article class="idea-debug-stat"><div class="idea-debug-stat-label mono">Docs</div><div class="idea-debug-stat-value">${payload.debug.evidence.docs.length}</div></article>
      <article class="idea-debug-stat"><div class="idea-debug-stat-label mono">Snippets</div><div class="idea-debug-stat-value">${payload.debug.evidence.snippets.length}</div></article>
      <article class="idea-debug-stat"><div class="idea-debug-stat-label mono">Run Families</div><div class="idea-debug-stat-value">${payload.debug.evidence.runFamilies.length}</div></article>
      <article class="idea-debug-stat"><div class="idea-debug-stat-label mono">Harvested</div><div class="idea-debug-stat-value">${payload.debug.harvest.candidates.length}</div></article>
      <article class="idea-debug-stat"><div class="idea-debug-stat-label mono">Kept</div><div class="idea-debug-stat-value">${payload.debug.attachment.kept.length}</div></article>
      <article class="idea-debug-stat"><div class="idea-debug-stat-label mono">Dropped</div><div class="idea-debug-stat-value">${payload.debug.attachment.dropped.length}</div></article>
      <article class="idea-debug-stat"><div class="idea-debug-stat-label mono">Total</div><div class="idea-debug-stat-value">${formatIdeaGraphMs(timings.totalMs)}</div></article>
    </div>
  `;

  const evidenceDetails = document.createElement("details");
  evidenceDetails.className = "idea-debug-section";
  evidenceDetails.innerHTML = `
    <summary>Evidence Pack</summary>
    <div class="idea-debug-meta muted mono">evidence ${formatIdeaGraphMs(timings.evidenceMs)} • docs ${payload.debug.evidence.docs.length} • snippets ${payload.debug.evidence.snippets.length} • run families ${payload.debug.evidence.runFamilies.length}</div>
    <div class="idea-debug-grid">
      ${payload.debug.evidence.docs.map((doc) => makeIdeaDebugCardHtml(doc.title, `${doc.file} • ${doc.snippetIds.length} snippets • ${doc.runRefs.length} run refs`, doc.snippetIds, "Doc")).join("")}
    </div>
    <div class="idea-debug-grid">
      ${payload.debug.evidence.runFamilies.map((family) => makeIdeaDebugCardHtml(family.label, `runs ${family.runCount} • active ${family.activeRuns} • best ${fmtAcc(family.bestAccuracy)} • ce ${fmtAcc(family.lastTrainCe)}`, [...family.docRefs, ...family.sampleRuns].slice(0, 8), "Run Family")).join("")}
    </div>
    <div class="idea-debug-grid">
      ${payload.debug.evidence.snippets.map((snippet) => makeIdeaDebugCardHtml(`${snippet.docFile} • ${snippet.heading}`, snippet.excerpt, [snippet.id, ...snippet.runRefs].slice(0, 6), "Snippet")).join("")}
    </div>
  `;

  const harvestDetails = document.createElement("details");
  harvestDetails.className = "idea-debug-section";
  harvestDetails.open = true;
  harvestDetails.innerHTML = `
    <summary>Harvested Candidates</summary>
    <div class="idea-debug-meta muted mono">harvest ${formatIdeaGraphMs(timings.harvestMs)} • ${payload.debug.harvest.candidates.length} candidates</div>
    <div class="idea-debug-grid">
      ${payload.debug.harvest.candidates.map((candidate) => makeIdeaDebugCardHtml(candidate.label, candidate.notes, candidate.candidateRefs, formatIdeaKind(candidate.kind))).join("")}
    </div>
  `;

  const attachmentDetails = document.createElement("details");
  attachmentDetails.className = "idea-debug-section";
  attachmentDetails.open = true;
  attachmentDetails.innerHTML = `
    <summary>Evidence Attachment</summary>
    <div class="idea-debug-meta muted mono">attachment ${formatIdeaGraphMs(timings.attachmentMs)} • kept ${payload.debug.attachment.kept.length} • dropped ${payload.debug.attachment.dropped.length}</div>
    <div class="idea-debug-grid">
      ${payload.debug.attachment.kept.map((candidate) => makeIdeaDebugCardHtml(candidate.label, `${candidate.coverageNote} ${candidate.summary}`, [...candidate.evidenceRefs, ...candidate.contradictionRefs].slice(0, 8), `Kept • ${formatIdeaKind(candidate.kind)}`)).join("")}
    </div>
    ${payload.debug.attachment.dropped.length > 0 ? `<div class="idea-debug-grid">${payload.debug.attachment.dropped.map((candidate) => makeIdeaDebugCardHtml(candidate.label, candidate.reason, candidate.candidateRefs, "Dropped")).join("")}</div>` : ""}
  `;

  const graphDetails = document.createElement("details");
  graphDetails.className = "idea-debug-section";
  graphDetails.innerHTML = `
    <summary>Graph Build</summary>
    <div class="idea-debug-meta muted mono">graph ${formatIdeaGraphMs(timings.graphMs)} • anchors ${payload.anchors.length} • edges ${payload.graph.edges.length} • version ${escapeHtml(payload.inputs.pipelineVersion)}</div>
    <div class="idea-debug-grid">
      ${payload.graph.edges.map((edge) => makeIdeaDebugCardHtml(`${edge.fromId} -> ${edge.toId}`, edge.reasoning, edge.evidenceRefs, edge.relationPhrase)).join("")}
    </div>
  `;

  debug.appendChild(evidenceDetails);
  debug.appendChild(harvestDetails);
  debug.appendChild(attachmentDetails);
  debug.appendChild(graphDetails);
  return debug;
}

function makeIdeaTreeLoadingHtml(status: IdeaTreeProgressStatus | null): string {
  const steps: Array<{ key: IdeaTreeProgressStage; label: string }> = [
    { key: "harvest", label: "Harvest" },
    { key: "attachment", label: "Attach" },
    { key: "synthesis", label: "Synthesize" },
  ];
  const activeStage = status?.activeStage ?? "harvest";
  const activeIndex = steps.findIndex((step) => step.key === activeStage);
  const title =
    status?.state === "error"
      ? "Graph generation failed"
      : status?.state === "completed"
        ? "Graph generation complete"
        : "Generating graph";
  const copy =
    status?.message ??
    "Harvesting candidates, attaching evidence, and assembling the final structure.";

  return `
    <div class="idea-tree-empty idea-tree-loading" aria-live="polite">
      <div class="idea-tree-loading-head">
        <div class="idea-tree-loading-spinner" aria-hidden="true"></div>
        <div>
          <div class="idea-tree-loading-title">${escapeHtml(title)}</div>
          <div class="idea-tree-loading-copy muted">${escapeHtml(copy)}</div>
        </div>
      </div>
      <div class="idea-tree-loading-steps mono">
        ${steps
          .map((step, index) => {
            const classNames = ["idea-tree-loading-step"];
            if (status?.state === "completed" || index < activeIndex) classNames.push("is-complete");
            if (status?.state === "running" && index === activeIndex) classNames.push("is-active");
            if (status?.state === "error" && index === activeIndex) classNames.push("is-error");
            return `<span class="${classNames.join(" ")}">${step.label}</span>`;
          })
          .join("")}
      </div>
    </div>
  `;
}

function renderIdeaTree(data: Bootstrap) {
  if (ideaTree && ideaTree.taskId !== data.selectedTask.id) {
    ideaTree = null;
    ideaTreeError = null;
  }
  if (ideaTreeProgress && ideaTreeProgress.taskId !== data.selectedTask.id) ideaTreeProgress = null;
  const meta = $("ideaTreeMeta");
  const body = $("ideaTreeBody");
  const generateBtn = $("ideaTreeGenerate") as HTMLButtonElement;
  generateBtn.disabled = ideaTreeLoadInFlight;
  generateBtn.textContent = ideaTreeLoadInFlight ? "Generating..." : "Generate Graph";

  if (ideaTreeLoadInFlight) {
    meta.textContent = ideaTreeProgress ? ideaTreeProgress.message : "generating graph...";
    meta.title = "";
  } else if (ideaTreeError) {
    meta.textContent = `error: ${ideaTreeError}`;
    meta.title = ideaTreeError;
  } else if (ideaTree) {
    meta.textContent = `generated: ${ideaTree.generatedAt} • anchors: ${ideaTree.anchors.length} • docs: ${ideaTree.inputs.docs.length} • snippets: ${ideaTree.inputs.snippetCount} • run families: ${ideaTree.inputs.runFamilyCount} • kept: ${ideaTree.debug.attachment.kept.length} • total: ${formatIdeaGraphMs(ideaTree.debug.timings.totalMs)}`;
    meta.title = "";
  } else {
    meta.textContent = "Not generated yet.";
    meta.title = "";
  }

  if (ideaTreeLoadInFlight && !ideaTree) {
    body.innerHTML = makeIdeaTreeLoadingHtml(ideaTreeProgress);
    return;
  }
  if (ideaTreeError && !ideaTree) {
    body.innerHTML = `<div class="idea-tree-empty muted">Graph generation failed: ${escapeHtml(ideaTreeError)}</div>`;
    return;
  }
  if (!ideaTree) {
    body.innerHTML = `<div class="idea-tree-empty muted">Click <strong>Generate Graph</strong> to harvest candidates from the corpus, attach evidence, and then synthesize the final graph.</div>`;
    return;
  }
  renderIdeaTreeGraph(body, ideaTree);
}

function scheduleIdeaTreeEdgeRender() {
  if (ideaTreeEdgeRenderFrame !== null) cancelAnimationFrame(ideaTreeEdgeRenderFrame);
  ideaTreeEdgeRenderFrame = requestAnimationFrame(() => {
    ideaTreeEdgeRenderFrame = null;
    renderIdeaTreeEdges();
  });
}

function renderIdeaTreeEdges() {
  if (!ideaTree) return;
  const shell = document.getElementById("ideaTreeShell") as HTMLElement | null;
  const svg = document.getElementById("ideaTreeEdgeLayer") as SVGSVGElement | null;
  if (!shell || !svg) return;
  const layout = buildIdeaTreeLayout(ideaTree);
  const shellRect = shell.getBoundingClientRect();
  if (shellRect.width <= 0 || shellRect.height <= 0) return;

  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${shellRect.width} ${shellRect.height}`);
  const nodeEls = new Map(
    [...shell.querySelectorAll<HTMLElement>(".idea-node")]
      .map((nodeEl) => [nodeEl.dataset.nodeId ?? "", nodeEl] as const)
      .filter(([nodeId]) => Boolean(nodeId))
  );
  const nodesById = new Map(ideaTree.graph.nodes.map((node) => [node.id, node] as const));

  for (const edge of ideaTree.graph.edges) {
    const fromEl = nodeEls.get(edge.fromId);
    const toEl = nodeEls.get(edge.toId);
    const fromNode = nodesById.get(edge.fromId);
    const toNode = nodesById.get(edge.toId);
    if (!fromEl || !toEl || !fromNode || !toNode) continue;
    const fromAssignment = layout.assignments.get(edge.fromId);
    const toAssignment = layout.assignments.get(edge.toId);
    const fromRect = fromEl.getBoundingClientRect();
    const toRect = toEl.getBoundingClientRect();
    const shouldFlip =
      (fromAssignment?.depth ?? Number.POSITIVE_INFINITY) > (toAssignment?.depth ?? Number.POSITIVE_INFINITY) ||
      ((fromAssignment?.depth ?? 0) === (toAssignment?.depth ?? 0) && fromRect.top > toRect.top);
    const sourceRect = shouldFlip ? toRect : fromRect;
    const targetRect = shouldFlip ? fromRect : toRect;
    const startX = sourceRect.left - shellRect.left + sourceRect.width / 2;
    const startY = sourceRect.bottom - shellRect.top;
    const endX = targetRect.left - shellRect.left + targetRect.width / 2;
    const endY = targetRect.top - shellRect.top;
    const curveY = Math.max(28, Math.abs(endY - startY) / 2);
    const startControlY = startY + curveY;
    const endControlY = endY - curveY;
    const d = `M ${startX} ${startY} C ${startX} ${startControlY} ${endX} ${endControlY} ${endX} ${endY}`;

    const visiblePath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    visiblePath.setAttribute("class", "idea-edge-visible");
    visiblePath.setAttribute("d", d);
    const isCrossLink =
      (fromAssignment?.anchorId ?? edge.fromId) !== (toAssignment?.anchorId ?? edge.toId) ||
      (toAssignment?.depth ?? 0) <= (fromAssignment?.depth ?? -1);
    if (isCrossLink) visiblePath.classList.add("is-cross-link");

    const hitPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    hitPath.setAttribute("class", "idea-edge-hit");
    hitPath.setAttribute("d", d);
    const show = (event?: MouseEvent) => {
      visiblePath.classList.add("is-active");
      showIdeaTreePopover(
        {
          eyebrow: edge.relationPhrase,
          title: `${fromNode.label} -> ${toNode.label}`,
          summary: edge.reasoning,
          evidenceRefs: edge.evidenceRefs,
        },
        {
          clientX: event?.clientX,
          clientY: event?.clientY,
          anchorEl: hitPath,
        }
      );
    };
    const hide = () => {
      visiblePath.classList.remove("is-active");
      hideIdeaTreePopover();
    };
    hitPath.addEventListener("mouseenter", (event) => show(event));
    hitPath.addEventListener("mousemove", (event) => show(event));
    hitPath.addEventListener("mouseleave", hide);

    svg.appendChild(visiblePath);
    svg.appendChild(hitPath);
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
      <td>${fmtDateTime(sweep.startedAt)}</td>
      <td>${fmtDateTime(sweep.endedAt)}</td>
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
      <td><code>${run.runId}</code>${run.isEvalOnly ? '<div><span class="run-badge mono">Eval-only</span></div>' : ""}</td>
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
      <td class="mono">${fmtDateTime(doc.updatedAt)}</td>
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
      meta.textContent =
        message.role === "user"
          ? `You${message.generatedAt ? ` • ${fmtDateTime(message.generatedAt)}` : ""}`
          : message.pending
            ? "Agent thinking..."
            : `Agent${message.generatedAt ? ` • ${fmtDateTime(message.generatedAt)}` : ""}`;
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
  renderIdeaTree(data);
  renderTaskQa(data);
  renderSweeps(data);
  renderSweepFilter(data);
  renderActiveToggle(data);
  renderRuns(data);
  renderDocs(data);
}

async function loadIdeaTree(scrollIntoView: boolean) {
  const taskId = getActiveTaskId();
  if (!taskId || ideaTreeLoadInFlight) return;
  ideaTreeLoadInFlight = true;
  ideaTreeError = null;
  ideaTreeProgress = {
    taskId,
    state: "running",
    activeStage: "harvest",
    message: "Starting graph generation...",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  openSection("ideaTreeSection");
  startIdeaTreeStatusPolling(taskId);
  if (state) renderIdeaTree(state);
  try {
    const res = await fetch("/api/ideas/tree", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        taskId,
      }),
    });
    const body = (await res.json()) as IdeaTreeResponse | { error?: string };
    if (!res.ok) throw new Error(body && "error" in body ? body.error ?? `idea graph failed: ${res.status}` : `idea graph failed: ${res.status}`);
    if (taskId !== getActiveTaskId()) return;
    ideaTree = body as IdeaTreeResponse;
    ideaTreeError = null;
    ideaTreeProgress = null;
    if (state) renderIdeaTree(state);
    if (scrollIntoView) $("ideaTreeSection").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    if (taskId !== getActiveTaskId()) return;
    ideaTree = null;
    ideaTreeError = String(error);
    ideaTreeProgress = {
      taskId,
      state: "error",
      activeStage: ideaTreeProgress?.activeStage ?? null,
      message: String(error),
      startedAt: ideaTreeProgress?.startedAt ?? new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    if (state) renderIdeaTree(state);
  } finally {
    stopIdeaTreeStatusPolling();
    ideaTreeLoadInFlight = false;
    if (state && state.selectedTask.id === taskId) renderIdeaTree(state);
  }
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
    ["Status", detail.run.isActive ? "active" : detail.run.isEvalOnly ? "eval-only" : detail.run.hasFinalCheckpoint ? "finished" : "idle"],
    ["Updated", fmtDateTime(detail.updatedAt, "full")],
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

  if (detail.introLines.length > 0 || detail.tailLines.length > 0) {
    const toggles = document.createElement("div");
    toggles.className = "detail-toggle-row";
    if (detail.introLines.length > 0) toggles.appendChild(makeRunLogToggle("Show log head", showRunLogHead, (checked) => {
      showRunLogHead = checked;
      if (runDetail) renderRunDetail(runDetail);
    }));
    if (detail.tailLines.length > 0) toggles.appendChild(makeRunLogToggle("Show log tail", showRunLogTail, (checked) => {
      showRunLogTail = checked;
      if (runDetail) renderRunDetail(runDetail);
    }));
    body.appendChild(toggles);
  }

  if (detail.introLines.length > 0 && showRunLogHead) {
    const introWrap = document.createElement("div");
    introWrap.className = "detail-tail";
    introWrap.innerHTML = `<div class="detail-section-title">Log Start</div>`;
    const pre = document.createElement("pre");
    pre.className = "detail-pre";
    pre.textContent = detail.introLines.join("\n");
    introWrap.appendChild(pre);
    body.appendChild(introWrap);
  }

  if (detail.tailLines.length > 0 && showRunLogTail) {
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

function makeRunLogToggle(label: string, checked: boolean, onChange: (checked: boolean) => void): HTMLElement {
  const wrap = document.createElement("label");
  wrap.className = "detail-toggle";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;
  input.addEventListener("change", () => onChange(input.checked));
  const text = document.createElement("span");
  text.textContent = label;
  wrap.appendChild(input);
  wrap.appendChild(text);
  return wrap;
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

  const ideaTreeGenerate = $("ideaTreeGenerate") as HTMLButtonElement;
  ideaTreeGenerate.addEventListener("click", () => {
    void loadIdeaTree(true);
  });

  window.addEventListener("resize", () => {
    scheduleIdeaTreeEdgeRender();
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
