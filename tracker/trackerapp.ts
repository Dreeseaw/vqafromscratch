// trackerapp.ts
import { serve } from "bun";
import fs from "fs";

const args = process.argv;
const fileIndex = args.indexOf("-f");
if (fileIndex === -1) {
  console.error("Usage: bun run trackerapp.ts -f logfile.txt");
  process.exit(1);
}
const logfile = args[fileIndex + 1];

function parseLine(line: string) {
  // Example:
  // Step: 1, Loss: 3.31 (RL: 1.72, KL: 39.80, KLw: 0.04)
  const m = line.match(
    /Step:\s*(\d+),\s*Loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?).*?\(.*?RL:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*KL:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*KLw:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)/
  );
  if (!m) return null;

  return {
    step: Number(m[1]),
    loss: Number(m[2]),
    rl: Number(m[3]),
    kl: Number(m[4]),
    klw: Number(m[5]),
  };
}

serve({
  port: 3000,

  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/") {
      return new Response(await Bun.file("index.html").text(), {
        headers: { "Content-Type": "text/html" },
      });
    }

    // SSE endpoint
    if (url.pathname === "/stream") {
      let closed = false;
      let watcher: fs.FSWatcher | null = null;
    
      const stream = new ReadableStream<string>({
        start(controller) {
          const safeEnqueue = (obj: any) => {
            if (closed) return;
            try {
              controller.enqueue(`data: ${JSON.stringify(obj)}\n\n`);
            } catch {
              // client disconnected / stream closed mid-enqueue
              closed = true;
              try { watcher?.close(); } catch {}
            }
          };
    
          // 1) send existing data
          const lines = fs.readFileSync(logfile, "utf-8").split("\n");
          for (const line of lines) {
            const parsed = parseLine(line);
            if (parsed) safeEnqueue(parsed);
          }
    
          // 2) tail file
          let size = fs.statSync(logfile).size;
    
          watcher = fs.watch(logfile, { persistent: true }, () => {
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
        },
    
        cancel() {
          // called when browser disconnects / refreshes
          closed = true;
          try { watcher?.close(); } catch {}
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
    return new Response("Not found", { status: 404 });
  },
});
