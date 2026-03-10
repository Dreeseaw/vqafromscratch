const titleEl = document.getElementById("docTitle") as HTMLElement;
const bodyEl = document.getElementById("docBody") as HTMLElement;

function getFileParam(): string {
  const u = new URL(window.location.href);
  return String(u.searchParams.get("file") ?? "").trim();
}

async function loadDoc() {
  const file = getFileParam();
  if (!file) {
    titleEl.textContent = "Missing file";
    bodyEl.textContent = "No ?file=<markdown_name> provided.";
    return;
  }
  titleEl.textContent = file;
  const res = await fetch(`/api/doc?file=${encodeURIComponent(file)}`, { cache: "no-store" });
  if (!res.ok) {
    bodyEl.textContent = `Failed to load markdown (${res.status}).`;
    return;
  }
  bodyEl.textContent = await res.text();
}

loadDoc().catch((e) => {
  bodyEl.textContent = `Error: ${String(e)}`;
});

