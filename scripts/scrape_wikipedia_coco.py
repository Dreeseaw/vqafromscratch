#!/usr/bin/env python3
import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "mscoco-wiki-scraper/1.0 (research; contact=local)"


def load_coco_topics(annotations_dir: Path) -> Tuple[List[str], Dict[str, str]]:
    topics: Set[str] = set()
    super_of: Dict[str, str] = {}
    for fname in ("instances_train2014.json", "instances_val2014.json"):
        path = annotations_dir / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            name = cat.get("name")
            supercat = cat.get("supercategory")
            if name:
                topics.add(name)
            if supercat:
                topics.add(supercat)
                if name:
                    super_of[name] = supercat
    base_topics = [
        "COCO dataset",
        "Common Objects in Context",
        "object detection",
        "computer vision",
        "visual question answering",
    ]
    topics.update(base_topics)
    return sorted(topics), super_of


def word_count(text: str) -> int:
    return len(text.split())


class RateLimiter:
    def __init__(self, max_rps: float):
        self.max_rps = max_rps
        self.min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self.last_time = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        now = time.time()
        elapsed = now - self.last_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_time = time.time()


class WikiClient:
    def __init__(self, max_rps: float, timeout: float):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.limiter = RateLimiter(max_rps)
        self.timeout = timeout

    def get(self, params: Dict) -> Dict:
        for attempt in range(6):
            self.limiter.wait()
            r = self.session.get(API_URL, params=params, timeout=self.timeout)
            if r.status_code in (429, 503, 502):
                backoff = 1.5 ** attempt + random.random()
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    def search(self, query: str, limit: int, offset: int) -> Tuple[List[int], int]:
        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query,
            "srlimit": limit,
            "sroffset": offset,
            "srnamespace": 0,
        }
        data = self.get(params)
        results = data.get("query", {}).get("search", [])
        pageids = [r.get("pageid") for r in results if r.get("pageid")]
        next_offset = offset + len(results)
        return pageids, next_offset

    def resolve_titles(self, titles: List[str]) -> List[int]:
        if not titles:
            return []
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(titles),
            "redirects": 1,
            "prop": "pageprops",
        }
        data = self.get(params)
        pages = data.get("query", {}).get("pages", {})
        pageids = []
        for pid, p in pages.items():
            try:
                pid_int = int(pid)
            except ValueError:
                continue
            if pid_int <= 0:
                continue
            if "missing" in p:
                continue
            if "pageprops" in p and "disambiguation" in p["pageprops"]:
                continue
            pageids.append(pid_int)
        return pageids

    def fetch_page(self, pageid: int) -> Optional[Dict]:
        params = {
            "action": "query",
            "format": "json",
            "pageids": pageid,
            "redirects": 1,
            "prop": "extracts|pageprops|categories",
            "explaintext": 1,
            "exsectionformat": "plain",
            "cllimit": 50,
        }
        data = self.get(params)
        pages = data.get("query", {}).get("pages", {})
        p = pages.get(str(pageid))
        if not p or "missing" in p:
            return None
        if "pageprops" in p and "disambiguation" in p["pageprops"]:
            return None
        title = p.get("title", "")
        if title.lower().startswith("list of "):
            return None
        return {
            "pageid": pageid,
            "title": title,
            "extract": p.get("extract", "").strip(),
            "categories": [c.get("title") for c in p.get("categories", []) if c.get("title")],
        }

    def page_links(self, pageid: int, limit: int) -> List[str]:
        params = {
            "action": "query",
            "format": "json",
            "pageids": pageid,
            "prop": "links",
            "pllimit": limit,
            "plnamespace": 0,
        }
        data = self.get(params)
        pages = data.get("query", {}).get("pages", {})
        p = pages.get(str(pageid), {})
        links = p.get("links", [])
        return [l.get("title") for l in links if l.get("title")]


def load_seen(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(int(line))
            except ValueError:
                continue
    return seen


def append_seen(path: Path, pageid: int):
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{pageid}\n")


def save_state(path: Path, state: Dict):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f)
    tmp.replace(path)


def load_state(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-dir", default="./annotations/annotations")
    ap.add_argument("--output-dir", default="./data/wiki_coco")
    ap.add_argument("--target-words", type=int, default=100_000_000)
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--max-rps", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--seed-limit", type=int, default=20)
    ap.add_argument("--seed-pages-per-topic", type=int, default=3)
    ap.add_argument("--expand-links", action="store_true")
    ap.add_argument("--links-per-page", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    articles_path = output_dir / "articles.jsonl"
    seen_path = output_dir / "seen_pageids.txt"
    state_path = output_dir / "state.json"
    manifest_path = output_dir / "manifest.json"

    topics, super_of = load_coco_topics(annotations_dir)
    if not topics:
        print(f"No COCO categories found under {annotations_dir}", file=sys.stderr)
        sys.exit(1)

    client = WikiClient(max_rps=args.max_rps, timeout=args.timeout)

    seen_pageids = load_seen(seen_path)
    total_words = 0
    page_queue: Deque[int] = collections.deque()
    title_queue: Deque[str] = collections.deque()
    search_offsets: Dict[str, int] = {t: 0 for t in topics}
    fetched = 0

    if args.resume:
        state = load_state(state_path)
        if state:
            total_words = state.get("total_words", 0)
            fetched = state.get("fetched", 0)
            search_offsets.update(state.get("search_offsets", {}))
            page_queue.extend(state.get("page_queue", []))
            title_queue.extend(state.get("title_queue", []))

    def refill_from_search():
        for t in topics:
            offset = search_offsets.get(t, 0)
            for _ in range(args.seed_pages_per_topic):
                pageids, next_offset = client.search(t, args.seed_limit, offset)
                offset = next_offset
                for pid in pageids:
                    if pid and pid not in seen_pageids:
                        page_queue.append(pid)
            search_offsets[t] = offset

    if not page_queue:
        refill_from_search()

    def maybe_resolve_titles():
        if not title_queue:
            return
        batch = []
        while title_queue and len(batch) < 50:
            batch.append(title_queue.popleft())
        pageids = client.resolve_titles(batch)
        for pid in pageids:
            if pid not in seen_pageids:
                page_queue.append(pid)

    try:
        with articles_path.open("a", encoding="utf-8") as out:
            while total_words < args.target_words:
                if not page_queue:
                    maybe_resolve_titles()
                if not page_queue:
                    refill_from_search()
                if not page_queue:
                    break

                pageid = page_queue.popleft()
                if pageid in seen_pageids:
                    continue

                page = client.fetch_page(pageid)
                if not page:
                    seen_pageids.add(pageid)
                    append_seen(seen_path, pageid)
                    continue

                text = page.get("extract", "")
                if not text:
                    seen_pageids.add(pageid)
                    append_seen(seen_path, pageid)
                    continue

                wc = word_count(text)
                if wc < args.min_words:
                    seen_pageids.add(pageid)
                    append_seen(seen_path, pageid)
                    continue

                print(f"Got {wc} words from {page['title']}")

                title = page.get("title", "")
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                doc = {
                    "id": f"wikipedia:{pageid}",
                    "pageid": pageid,
                    "title": title,
                    "url": url,
                    "text": text,
                    "word_count": wc,
                    "source": "wikipedia",
                }
                out.write(json.dumps(doc, ensure_ascii=True) + "\n")
                out.flush()

                seen_pageids.add(pageid)
                append_seen(seen_path, pageid)
                total_words += wc
                fetched += 1

                if args.expand_links and total_words < args.target_words:
                    links = client.page_links(pageid, args.links_per_page)
                    for l in links:
                        title_queue.append(l)

                if fetched % 200 == 0:
                    save_state(
                        state_path,
                        {
                            "total_words": total_words,
                            "fetched": fetched,
                            "search_offsets": search_offsets,
                            "page_queue": list(page_queue)[:10000],
                            "title_queue": list(title_queue)[:10000],
                        },
                    )

    except KeyboardInterrupt:
        save_state(
            state_path,
            {
                "total_words": total_words,
                "fetched": fetched,
                "search_offsets": search_offsets,
                "page_queue": list(page_queue)[:10000],
                "title_queue": list(title_queue)[:10000],
            },
        )
        print("\nInterrupted. State saved.", file=sys.stderr)
        sys.exit(1)

    save_state(
        state_path,
        {
            "total_words": total_words,
            "fetched": fetched,
            "search_offsets": search_offsets,
            "page_queue": list(page_queue)[:10000],
            "title_queue": list(title_queue)[:10000],
        },
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_words": total_words,
                "fetched": fetched,
                "target_words": args.target_words,
                "min_words": args.min_words,
                "annotations_dir": str(annotations_dir),
                "output_dir": str(output_dir),
            },
            f,
        )

    print(f"Done. fetched={fetched} total_words={total_words}")


if __name__ == "__main__":
    main()
