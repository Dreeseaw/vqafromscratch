import { marked } from "https://esm.sh/marked@16.3.0";

const titleEl = document.getElementById("docTitle") as HTMLElement;
const metaEl = document.getElementById("docMeta") as HTMLElement;
const bodyEl = document.getElementById("docBody") as HTMLElement;
const downloadLinkEl = document.getElementById("docDownload") as HTMLAnchorElement;
const homeLinkEl = document.getElementById("docHome") as HTMLAnchorElement;

function getParams() {
  const url = new URL(window.location.href);
  return {
    task: String(url.searchParams.get("task") ?? "").trim(),
    file: String(url.searchParams.get("file") ?? "").trim(),
  };
}

async function loadDoc() {
  const { task, file } = getParams();
  if (!file) {
    titleEl.textContent = "Missing file";
    bodyEl.textContent = "No ?file=<markdown_name> provided.";
    return;
  }

  titleEl.textContent = file;
  metaEl.textContent = task ? `task: ${task}` : "default task";
  homeLinkEl.href = task ? `/?task=${encodeURIComponent(task)}` : "/";

  const url = new URL("/api/doc", window.location.origin);
  if (task) url.searchParams.set("task", task);
  url.searchParams.set("file", file);
  downloadLinkEl.href = url.toString();
  downloadLinkEl.download = file.split("/").at(-1) || file;
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    bodyEl.textContent = `Failed to load markdown (${res.status}).`;
    return;
  }
  const markdown = await res.text();
  bodyEl.innerHTML = `<article class="markdown-body doc-render">${marked.parse(markdown, { async: false }) as string}</article>`;
}

loadDoc().catch((error) => {
  bodyEl.textContent = `Error: ${String(error)}`;
});
