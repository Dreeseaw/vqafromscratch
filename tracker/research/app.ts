type RunSummary = {
  runId: string;
  runDir: string;
  finalAccuracy: number | null;
  bestAccuracy: number | null;
  lastTrainCe: number | null;
  lastStep: number | null;
  lastStepsPerSec: number | null;
  numParams: number | null;
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

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

const fmtAcc = (v: number | null) => (Number.isFinite(v ?? NaN) ? (v as number).toFixed(4) : "-");
const fmtNum = (v: number | null) => (Number.isFinite(v ?? NaN) ? String(v) : "-");
const fmtParams = (v: number | null) => (Number.isFinite(v ?? NaN) ? Intl.NumberFormat("en-US").format(v as number) : "-");

let state: Bootstrap | null = null;
let selectedSweep = "__all__";
let docsPage = 0;
const DOCS_PAGE_SIZE = 5;

function renderKpis(data: Bootstrap) {
  const kpis = $("kpis");
  const best = data.summary.bestRun;
  const items = [
    ["Docs", String(data.summary.docsCount)],
    ["Sweeps", String(data.summary.sweepsCount)],
    ["Runs", String(data.summary.runsCount)],
    ["Runs w/ Acc", String(data.summary.runsWithAccuracy)],
    ["Best", best ? `${best.runId} (${fmtAcc(best.finalAccuracy)})` : "-"],
  ];
  kpis.innerHTML = "";
  for (const [label, value] of items) {
    const div = document.createElement("div");
    div.className = "kpi";
    div.innerHTML = `<div class="kpi-label">${label}</div><div class="kpi-value">${value}</div>`;
    kpis.appendChild(div);
  }
}

function renderSweeps(data: Bootstrap) {
  const body = $("sweepsBody");
  body.innerHTML = "";
  for (const s of data.sweeps) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${s.sweepId}</code></td>
      <td>${s.status}</td>
      <td>${s.runs.length}</td>
      <td>${fmtAcc(s.lastTrainCe)}</td>
      <td>${s.startedAt ?? "-"}</td>
      <td>${s.endedAt ?? "-"}</td>
    `;
    body.appendChild(tr);
  }
}

function getFilteredRuns(data: Bootstrap): RunSummary[] {
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
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${r.runId}</code></td>
      <td>${fmtAcc(r.finalAccuracy)}</td>
      <td>${fmtAcc(r.lastTrainCe)}</td>
      <td>${fmtNum(r.lastStep)}</td>
      <td>${fmtNum(r.lastStepsPerSec)}</td>
      <td>${fmtParams(r.numParams)}</td>
      <td>${r.hasFinalCheckpoint ? "yes" : "no"}</td>
    `;
    body.appendChild(tr);
  }
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
    tr.innerHTML = `
      <td>
        <strong>${d.title}</strong><br />
        <span class="mono muted">${d.file}</span><br />
        <a href="/doc?file=${encodeURIComponent(d.file)}" target="_blank" rel="noopener">Open Markdown</a>
      </td>
      <td class="mono">${d.updatedAt}</td>
      <td>${d.runRefs.length}</td>
      <td>${topAcc}</td>
    `;
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
  renderRuns(data);
  renderDocs(data);
}

async function load() {
  const res = await fetch("/api/bootstrap", { cache: "no-store" });
  if (!res.ok) throw new Error(`bootstrap failed: ${res.status}`);
  const data = (await res.json()) as Bootstrap;
  state = data;
  renderAll(data);
}

function setup() {
  const sel = $("sweepFilter") as HTMLSelectElement;
  sel.addEventListener("change", () => {
    selectedSweep = sel.value;
    if (state) renderRuns(state);
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
}, 15000);
