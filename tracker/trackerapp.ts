// trackerapp.ts
import { serve } from "bun";
import fs from "fs";
import path from "path";

// -------------------------
// CLI ARG PARSING
// -------------------------
const args = process.argv;
const dirIndex = args.indexOf("-f");
if (dirIndex === -1) {
  console.error("Usage: bun run trackerapp.ts -f <run_dir> [-p <port>]");
  process.exit(1);
}

const runDir = path.resolve(args[dirIndex + 1]);
const logfile = path.join(runDir, "logfile.txt");

if (!fs.existsSync(runDir)) {
  console.error(`Run directory does not exist: ${runDir}`);
  process.exit(1);
}
if (!fs.existsSync(logfile)) {
  console.error(`Missing logfile.txt in ${runDir}`);
  process.exit(1);
}

const portIndex = args.indexOf("-p");
const port =
  portIndex !== -1
    ? Number(args[portIndex + 1])
    : 3000;

if (!Number.isInteger(port) || port <= 0) {
  console.error(`Invalid port: ${args[portIndex + 1]}`);
  process.exit(1);
}

// -------------------------
// LOG PARSER
// -------------------------
function parseLine(line: string) {
  // Step: 1, Loss: 3.31 (RL: 1.72, KL: 39.80, KLw: 0.04)
  const m = line.match(
    /Step:\s*(\d+),\s*Loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?).*?\(\s*RL:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*KL:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*KLw:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*MMD:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*MMDw:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)/
  );
  if (!m) return null;

  return {
    step: Number(m[1]),
    loss: Number(m[2]),
    rl: Number(m[3]),
    kl: Number(m[4]),
    klw: Number(m[5]),
    mmd: Number(m[6]),
    mmdw: Number(m[7]),
  };
}

// -------------------------
// SERVER
// -------------------------
serve({
  port: port,

  async fetch(req) {
    const url = new URL(req.url);

    // -------------------------
    // MAIN PAGE
    // -------------------------
    if (url.pathname === "/") {
      return new Response(await Bun.file("index.html").text(), {
        headers: { "Content-Type": "text/html" },
      });
    }

    // -------------------------
    // SSE STREAM (LOG METRICS)
    // -------------------------
    if (url.pathname === "/stream") {
      const lastEventId = Number(req.headers.get("last-event-id") ?? "-1");
      const lastStepSeen = Number.isFinite(lastEventId) ? lastEventId : -1;
      let closed = false;
      let watcher: fs.FSWatcher | null = null;

      const stream = new ReadableStream<string>({
        start(controller) {
	  const safeEnqueue = (obj: any) => {
            if (closed) return;
  	    try {
              // Make reconnects resumable
    	      controller.enqueue(`id: ${obj.step}\n`);
              controller.enqueue(`data: ${JSON.stringify(obj)}\n\n`);
            } catch {
              closed = true;
	      // ew
              try { watcher?.close(); } catch {}
            }
          };

          // 1) send existing logfile contents
	  let maxStepSent = lastStepSeen;
          const lines = fs.readFileSync(logfile, "utf-8").split("\n");
          for (const line of lines) {
            const parsed = parseLine(line);
            if (parsed && parsed.step > maxStepSent) {
	      maxStepSent = parsed.step;
	      safeEnqueue(parsed);
	    }
          }

          // 2) tail logfile
          let size = fs.statSync(logfile).size;

          watcher = fs.watch(logfile, () => {
            if (closed) return;

            const stats = fs.statSync(logfile);
            if (stats.size <= size) return;

            const fd = fs.openSync(logfile, "r");
            const buffer = Buffer.alloc(stats.size - size);
            fs.readSync(fd, buffer, 0, buffer.length, size);
            fs.closeSync(fd);
            size = stats.size;

            buffer
              .toString()
              .split("\n")
              .forEach((line) => {
                const parsed = parseLine(line);
                if (parsed) safeEnqueue(parsed);
              });
          });
	  const ping = setInterval(() => {
	    if (closed) return;
            try { controller.enqueue(`: ping\n\n`); } catch {}
          }, 15000);
        },

        cancel() {
          closed = true;
          try { watcher?.close(); } catch {}
	  try { clearInterval(ping); } catch {}
          watcher = null;
        },
      });

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        },
      });
    }

    // -------------------------
    // LIST RECON IMAGES
    // -------------------------
    if (url.pathname === "/images") {
      const files = fs.readdirSync(runDir)
        .filter(f => f.endsWith(".png"))
        .sort((a, b) => {
          const na = parseInt(a.match(/\d+/)?.[0] ?? "0");
          const nb = parseInt(b.match(/\d+/)?.[0] ?? "0");
          return na - nb;
        });

      return new Response(JSON.stringify(files), {
        headers: { "Content-Type": "application/json" },
      });
    }

    // -------------------------
    // SERVE IMAGE FILES
    // -------------------------
    if (url.pathname.startsWith("/img/")) {
      const fname = url.pathname.replace("/img/", "");
      const fpath = path.join(runDir, fname);

      if (!fpath.startsWith(runDir) || !fs.existsSync(fpath)) {
        return new Response("Not found", { status: 404 });
      }

      return new Response(Bun.file(fpath));
    }

    return new Response("Not found", { status: 404 });
  },
});
