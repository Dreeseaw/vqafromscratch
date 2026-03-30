#!/usr/bin/env python3
"""
Prune old checkpoint tarballs from large log directories.

Rule:
- look at top-level run directories under logs/
- if a run directory exceeds the size threshold, inspect all *.tar files beneath it
- group checkpoints by (parent directory, numeric series prefix)
- keep the highest numeric checkpoint in each series
- delete older checkpoints only

This deliberately parses trailing digits numerically so names like
step_5000.tar and step_45000.tar are ordered correctly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CheckpointFile:
    path: Path
    parent: Path
    prefix: str
    number: int
    size_bytes: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prune old checkpoints from oversized log directories.")
    ap.add_argument("--root", default="logs", help="Logs root directory.")
    ap.add_argument("--threshold-gib", type=float, default=1.0, help="Only prune run dirs above this recursive size.")
    ap.add_argument("--apply", action="store_true", help="Actually delete files. Default is dry-run.")
    ap.add_argument("--report-path", default="", help="Optional JSON report path. Defaults under logs/background/.")
    return ap.parse_args()


def parse_checkpoint_name(name: str) -> tuple[str, int] | None:
    if not name.endswith(".tar"):
        return None
    stem = name[:-4]
    idx = len(stem) - 1
    while idx >= 0 and stem[idx].isdigit():
        idx -= 1
    if idx == len(stem) - 1:
        return None
    prefix = stem[: idx + 1]
    number_text = stem[idx + 1 :]
    try:
        number = int(number_text)
    except ValueError:
        return None
    return prefix, number


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def recursive_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                continue
    return total


def collect_checkpoint_groups(run_dir: Path) -> dict[tuple[str, str], list[CheckpointFile]]:
    groups: dict[tuple[str, str], list[CheckpointFile]] = {}
    for p in run_dir.rglob("*.tar"):
        parsed = parse_checkpoint_name(p.name)
        if parsed is None:
            continue
        prefix, number = parsed
        try:
            size_bytes = p.stat().st_size
        except FileNotFoundError:
            continue
        rel_parent = str(p.parent.relative_to(run_dir))
        item = CheckpointFile(
            path=p,
            parent=p.parent,
            prefix=prefix,
            number=number,
            size_bytes=size_bytes,
        )
        groups.setdefault((rel_parent, prefix), []).append(item)
    return groups


def default_report_path(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "background" / f"checkpoint_prune_report_{stamp}.json"


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    threshold_bytes = int(float(args.threshold_gib) * (1 << 30))
    report_path = Path(args.report_path) if args.report_path else default_report_path(root)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict] = []
    delete_paths: list[Path] = []
    delete_bytes = 0

    for run_dir in iter_run_dirs(root):
        total_size = recursive_size_bytes(run_dir)
        if total_size <= threshold_bytes:
            continue
        groups = collect_checkpoint_groups(run_dir)
        if not groups:
            continue

        run_summary = {
            "run_dir": str(run_dir),
            "total_size_bytes": total_size,
            "threshold_bytes": threshold_bytes,
            "series": [],
        }

        for (rel_parent, prefix), items in sorted(groups.items()):
            items_sorted = sorted(items, key=lambda x: (x.number, x.path.name))
            keep = items_sorted[-1]
            prune = items_sorted[:-1]
            series_summary = {
                "relative_parent": rel_parent,
                "prefix": prefix,
                "keep": {
                    "path": str(keep.path),
                    "number": keep.number,
                    "size_bytes": keep.size_bytes,
                },
                "delete": [
                    {
                        "path": str(item.path),
                        "number": item.number,
                        "size_bytes": item.size_bytes,
                    }
                    for item in prune
                ],
            }
            run_summary["series"].append(series_summary)
            for item in prune:
                delete_paths.append(item.path)
                delete_bytes += item.size_bytes
        run_summaries.append(run_summary)

    if args.apply:
        for path in delete_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue

    report = {
        "root": str(root),
        "threshold_gib": float(args.threshold_gib),
        "apply": bool(args.apply),
        "run_count": len(run_summaries),
        "delete_count": len(delete_paths),
        "delete_bytes": delete_bytes,
        "runs": run_summaries,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    gib = delete_bytes / float(1 << 30)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: runs={len(run_summaries)} delete_files={len(delete_paths)} reclaim_gib={gib:.2f}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
