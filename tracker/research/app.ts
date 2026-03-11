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

type Bootstrap = {
  generatedAt: string;
  docsRoot: string;
  logsRoot: string;
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
  trainCeSeries: SeriesPoint[];
  valAccuracySeries: SeriesPoint[];
  valCeSeries: SeriesPoint[];
  latestEvalAccuracy: number | null;
  latestValCe: number | null;
};

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

const fmtAcc = (v: number | null) => (Number.isFinite(v ?? NaN) ? (v as number).toFixed(4) : "-");
const fmtNum = (v: number | null) => (Number.isFinite(v ?? NaN) ? String(v) : "-");
const fmtPct = (v: number | null) => (Number.isFinite(v ?? NaN) ? `${(v as number).toFixed(1)}%` : "-");
const fmtParams = (total: number | null, trainable: number | null) => {
  if (!Number.isFinite(total ?? NaN)) return "-";
  const totalStr = Intl.NumberFormat("en-US").format(total as number);
  if (!Number.isFinite(trainable ?? NaN) || !total || total <= 0) return totalStr;
  return `${totalStr} (${fmtPct(((trainable as number) / (total as number)) * 100)})`;
};

let state: Bootstrap | null = null;
let selectedSweep = "__all__";
let activeOnly = false;
let sweepsPage = 0;
let runsPage = 0;
let docsPage = 0;
let selectedRunId: string | null = null;
let runDetail: RunDetail | null = null;
let selectedDocFile: string | null = null;
let bootstrapLoadInFlight = false;
let runDetailLoadInFlight = false;
const SWEEPS_PAGE_SIZE = 5;
const RUNS_PAGE_SIZE = 10;
const DOCS_PAGE_SIZE = 5;
const BOOTSTRAP_REFRESH_MS = 10000;
const RUN_DETAIL_REFRESH_MS = 10000;

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

function renderSweeps(data: Bootstrap) {
  const body = $("sweepsBody");
  body.innerHTML = "";
  const pages = Math.max(1, Math.ceil(data.sweeps.length / SWEEPS_PAGE_SIZE));
  sweepsPage = Math.min(sweepsPage, pages - 1);
  const pageSweeps = data.sweeps.slice(sweepsPage * SWEEPS_PAGE_SIZE, (sweepsPage + 1) * SWEEPS_PAGE_SIZE);
  for (const s of pageSweeps) {
    const tr = document.createElement("tr");
    tr.className = s.sweepId === selectedSweep ? "is-selected clickable-row" : "clickable-row";
    tr.innerHTML = `
      <td><code>${s.sweepId}</code></td>
      <td>${s.status}</td>
      <td>${String(s.activeRuns)}</td>
      <td>${s.runs.length}</td>
      <td>${fmtAcc(s.bestAccuracy)}</td>
      <td>${fmtAcc(s.lastTrainCe)}</td>
      <td>${s.startedAt ?? "-"}</td>
      <td>${s.endedAt ?? "-"}</td>
    `;
    tr.addEventListener("click", () => {
      selectedSweep = s.sweepId;
      runsPage = 0;
      const sel = $("sweepFilter") as HTMLSelectElement;
      sel.value = s.sweepId;
      renderSweeps(data);
      renderRuns(data);
    });
    body.appendChild(tr);
  }
  renderSweepsPager(data.sweeps.length);
}

function renderSweepsPager(totalSweeps: number) {
  const pages = Math.max(1, Math.ceil(totalSweeps / SWEEPS_PAGE_SIZE));
  const prevBtn = $("sweepsPrev") as HTMLButtonElement;
  const nextBtn = $("sweepsNext") as HTMLButtonElement;
  const label = $("sweepsPageLabel");
  label.textContent = `Page ${sweepsPage + 1} / ${pages}`;
  prevBtn.disabled = sweepsPage === 0;
  nextBtn.disabled = sweepsPage >= pages - 1;
}

function getFilteredRuns(data: Bootstrap): RunSummary[] {
  if (activeOnly) return data.runs.filter((r) => r.isActive);
  if (selectedSweep === "__all__") return data.runs;
  const sweep = data.sweeps.find((s) => s.sweepId === selectedSweep);
  return sweep ? sweep.runs : [];
}

function renderRuns(data: Bootstrap) {
  const body = $("runsBody");
  body.innerHTML = "";
  const rows = getFilteredRuns(data)
    .slice()
    .sort((a, b) => (b.finalAccuracy ?? -1) - (a.finalAccuracy ?? -1));
  const pages = Math.max(1, Math.ceil(rows.length / RUNS_PAGE_SIZE));
  runsPage = Math.min(runsPage, pages - 1);
  const pageRows = rows.slice(runsPage * RUNS_PAGE_SIZE, (runsPage + 1) * RUNS_PAGE_SIZE);
  for (const r of pageRows) {
    const tr = document.createElement("tr");
    tr.className = r.runId === selectedRunId ? "is-selected clickable-row" : "clickable-row";
    tr.innerHTML = `
      <td><code>${r.runId}</code></td>
      <td>${fmtAcc(r.finalAccuracy)}</td>
      <td>${fmtAcc(r.lastTrainCe)}</td>
      <td>${fmtNum(r.lastStep)}</td>
      <td>${fmtNum(r.lastStepsPerSec)}</td>
      <td>${fmtParams(r.numParams, r.trainableParams)}</td>
      <td>${r.hasFinalCheckpoint ? "yes" : "no"}</td>
    `;
    tr.addEventListener("click", () => {
      selectedRunId = r.runId;
      renderRuns(data);
      void loadRunDetail(r.runId, true);
    });
    body.appendChild(tr);
  }
  renderRunsPager(rows.length);
}

function renderRunsPager(totalRuns: number) {
  const pages = Math.max(1, Math.ceil(totalRuns / RUNS_PAGE_SIZE));
  const prevBtn = $("runsPrev") as HTMLButtonElement;
  const nextBtn = $("runsNext") as HTMLButtonElement;
  const label = $("runsPageLabel");
  label.textContent = `Page ${runsPage + 1} / ${pages}`;
  prevBtn.disabled = runsPage === 0;
  nextBtn.disabled = runsPage >= pages - 1;
}

function shouldOpenDocsInNewTab(): boolean {
  return window.matchMedia("(max-width: 700px)").matches;
}

function openDocInNewTab(file: string) {
  window.open(`/doc?file=${encodeURIComponent(file)}`, "_blank", "noopener,noreferrer");
}

function renderDocs(data: Bootstrap) {
  const body = $("docsBody");
  body.innerHTML = "";
  const pages = Math.max(1, Math.ceil(data.docs.length / DOCS_PAGE_SIZE));
  docsPage = Math.min(docsPage, pages - 1);
  const start = docsPage * DOCS_PAGE_SIZE;
  const pageDocs = data.docs.slice(start, start + DOCS_PAGE_SIZE);
  for (const d of pageDocs) {
    const topAcc = d.mentionedAccuracies.length > 0 ? d.mentionedAccuracies[0].toFixed(4) : "-";
    const tr = document.createElement("tr");
    tr.className = d.file === selectedDocFile ? "is-selected clickable-row" : "clickable-row";
    tr.innerHTML = `
      <td>
        <strong>${d.title}</strong><br />
        <span class="mono muted">${d.file}</span>
      </td>
      <td class="mono">${d.updatedAt}</td>
      <td>${d.runRefs.length}</td>
      <td>${topAcc}</td>
    `;
    tr.addEventListener("click", () => {
      if (shouldOpenDocsInNewTab()) {
        openDocInNewTab(d.file);
        return;
      }
      selectedDocFile = d.file;
      renderDocs(data);
      void loadDocDetail(d.file, true);
    });
    body.appendChild(tr);
  }
  renderDocsPager(data.docs.length);
}

function renderDocsPager(totalDocs: number) {
  const pages = Math.max(1, Math.ceil(totalDocs / DOCS_PAGE_SIZE));
  const prevBtn = $("docsPrev") as HTMLButtonElement;
  const nextBtn = $("docsNext") as HTMLButtonElement;
  const label = $("docsPageLabel");
  label.textContent = `Page ${docsPage + 1} / ${pages}`;
  prevBtn.disabled = docsPage === 0;
  nextBtn.disabled = docsPage >= pages - 1;
}

function renderSweepFilter(data: Bootstrap) {
  const sel = $("sweepFilter") as HTMLSelectElement;
  const prev = sel.value || "__all__";
  sel.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "__all__";
  allOpt.textContent = "All runs";
  sel.appendChild(allOpt);
  for (const s of data.sweeps) {
    const opt = document.createElement("option");
    opt.value = s.sweepId;
    opt.textContent = s.sweepId;
    sel.appendChild(opt);
  }
  sel.value = data.sweeps.some((s) => s.sweepId === prev) ? prev : "__all__";
  selectedSweep = sel.value;
}

function renderAll(data: Bootstrap) {
  $("updatedAt").textContent = `updated: ${data.generatedAt}`;
  renderKpis(data);
  renderSweeps(data);
  renderSweepFilter(data);
  renderActiveToggle(data);
  renderRuns(data);
  renderDocs(data);
}

function renderActiveToggle(data: Bootstrap) {
  const toggle = $("activeOnly") as HTMLInputElement;
  const label = $("activeCount");
  const count = data.runs.filter((r) => r.isActive).length;
  toggle.checked = activeOnly;
  label.textContent = `${count} active`;
}

async function load() {
  if (bootstrapLoadInFlight) return;
  bootstrapLoadInFlight = true;
  try {
  const res = await fetch("/api/bootstrap", { cache: "no-store" });
  if (!res.ok) throw new Error(`bootstrap failed: ${res.status}`);
  const data = (await res.json()) as Bootstrap;
  state = data;
  renderAll(data);
  if (selectedRunId) await loadRunDetail(selectedRunId, false);
  if (selectedDocFile) await loadDocDetail(selectedDocFile, false);
  } finally {
    bootstrapLoadInFlight = false;
  }
}

async function loadRunDetail(runId: string, scrollIntoView: boolean) {
  if (runDetailLoadInFlight) return;
  runDetailLoadInFlight = true;
  const panel = $("runDetailPanel");
  const title = $("runDetailTitle");
  const body = $("runDetailBody");
  panel.hidden = false;
  if (!runDetail || runDetail.run.runId !== runId) {
    title.textContent = runId;
    body.innerHTML = `<div class="muted mono">Loading run detail...</div>`;
  }
  try {
    const res = await fetch(`/api/run?runId=${encodeURIComponent(runId)}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`run detail failed: ${res.status}`);
    runDetail = (await res.json()) as RunDetail;
    if (selectedRunId !== runId) return;
    renderRunDetail(runDetail);
    if (scrollIntoView) panel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    if (selectedRunId !== runId) return;
    body.innerHTML = `<div class="muted mono">Error: ${String(e)}</div>`;
  } finally {
    runDetailLoadInFlight = false;
  }
}

function renderRunDetail(detail: RunDetail) {
  const panel = $("runDetailPanel");
  const title = $("runDetailTitle");
  const body = $("runDetailBody");
  panel.hidden = false;
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
    ["Val CE", fmtAcc(detail.latestValCe)],
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
  if (detail.valCeSeries.length > 0) charts.appendChild(makeChartCard("Val CE", detail.valCeSeries, "ce"));
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
}

function makeChartCard(title: string, points: SeriesPoint[], kind: "acc" | "ce") {
  const card = document.createElement("div");
  card.className = "chart-card";
  if (points.length === 0) {
    card.innerHTML = `<div class="detail-section-title">${title}</div><div class="muted mono">No data</div>`;
    return card;
  }
  const path = buildChartPath(points, 320, 140);
  const values = points.map((p) => p.value);
  const latest = values[values.length - 1];
  const best = kind === "acc" ? Math.max(...values) : Math.min(...values);
  card.innerHTML = `
    <div class="chart-head">
      <div class="detail-section-title">${title}</div>
      <div class="mono muted">latest ${latest.toFixed(4)} | ${kind === "acc" ? "best" : "min"} ${best.toFixed(4)}</div>
    </div>
    <svg class="chart-svg" viewBox="0 0 320 140" preserveAspectRatio="none" aria-label="${title}">
      <path class="chart-baseline" d="M 0 132 H 320"></path>
      <path class="chart-line ${kind === "acc" ? "chart-line-acc" : "chart-line-ce"}" d="${path}"></path>
    </svg>
  `;
  return card;
}

function buildChartPath(points: SeriesPoint[], width: number, height: number) {
  const steps = points.map((p) => p.step);
  const values = points.map((p) => p.value);
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
  const panel = $("docDetailPanel");
  const title = $("docDetailTitle");
  const body = $("docDetailBody");
  panel.hidden = false;
  title.textContent = file;
  body.innerHTML = `<div class="muted mono">Loading doc...</div>`;
  try {
    const res = await fetch(`/api/doc?file=${encodeURIComponent(file)}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`doc failed: ${res.status}`);
    const markdown = await res.text();
    if (selectedDocFile !== file) return;
    renderDocDetail(file, markdown);
    if (scrollIntoView) panel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    if (selectedDocFile !== file) return;
    body.innerHTML = `<div class="muted mono">Error: ${String(e)}</div>`;
  }
}

function renderDocDetail(file: string, markdown: string) {
  const panel = $("docDetailPanel");
  const title = $("docDetailTitle");
  const body = $("docDetailBody");
  const openBtn = $("docDetailOpen") as HTMLButtonElement;
  panel.hidden = false;
  title.textContent = file;
  openBtn.hidden = false;
  openBtn.onclick = () => openDocInNewTab(file);
  body.innerHTML = `<article class="markdown-body doc-render">${marked.parse(markdown, { async: false }) as string}</article>`;
}

function setup() {
  const sweepsPrevBtn = $("sweepsPrev") as HTMLButtonElement;
  const sweepsNextBtn = $("sweepsNext") as HTMLButtonElement;
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
  const sel = $("sweepFilter") as HTMLSelectElement;
  sel.addEventListener("change", () => {
    selectedSweep = sel.value;
    runsPage = 0;
    if (state) renderRuns(state);
  });
  const activeToggle = $("activeOnly") as HTMLInputElement;
  activeToggle.addEventListener("change", () => {
    activeOnly = activeToggle.checked;
    runsPage = 0;
    if (state) renderRuns(state);
  });
  const closeDetailBtn = $("runDetailClose") as HTMLButtonElement;
  closeDetailBtn.addEventListener("click", () => {
    selectedRunId = null;
    runDetail = null;
    $("runDetailPanel").hidden = true;
    if (state) renderRuns(state);
  });
  const closeDocBtn = $("docDetailClose") as HTMLButtonElement;
  closeDocBtn.addEventListener("click", () => {
    selectedDocFile = null;
    $("docDetailPanel").hidden = true;
    $("docDetailOpen").hidden = true;
    if (state) renderDocs(state);
  });
  const runsPrevBtn = $("runsPrev") as HTMLButtonElement;
  const runsNextBtn = $("runsNext") as HTMLButtonElement;
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
  const prevBtn = $("docsPrev") as HTMLButtonElement;
  const nextBtn = $("docsNext") as HTMLButtonElement;
  prevBtn.addEventListener("click", () => {
    if (!state || docsPage === 0) return;
    docsPage -= 1;
    renderDocs(state);
  });
  nextBtn.addEventListener("click", () => {
    if (!state) return;
    const pages = Math.max(1, Math.ceil(state.docs.length / DOCS_PAGE_SIZE));
    if (docsPage >= pages - 1) return;
    docsPage += 1;
    renderDocs(state);
  });
}

setup();
load().catch((e) => {
  $("updatedAt").textContent = `error: ${String(e)}`;
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
