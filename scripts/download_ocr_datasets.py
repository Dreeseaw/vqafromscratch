#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import threading
import time
from pathlib import Path
from typing import Iterable

import requests
from huggingface_hub import snapshot_download


OPEN_IMAGES_TRAIN_INFO_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
TEXT_OCR_TRAIN_JSON = "data/vm_ssl/raw/textocr_full/TextOCR_0.1_train.json"
TEXT_OCR_VAL_JSON = "data/vm_ssl/raw/textocr_full/TextOCR_0.1_val.json"
TEXT_OCR_DEST = Path("data/vm_ssl/raw/textocr_trainval")
COCO_TEXT_DEST = Path("data/vm_ssl/raw/coco_text_hf")
USER_AGENT = "vqafromscratch-ocr-downloader/0.1"


def load_textocr_ids() -> tuple[dict[str, dict], set[str]]:
    meta: dict[str, dict] = {}
    ids: set[str] = set()
    for source_path in [TEXT_OCR_TRAIN_JSON, TEXT_OCR_VAL_JSON]:
        payload = json.loads(Path(source_path).read_text())
        for image_id, record in payload["imgs"].items():
            ids.add(image_id)
            meta[image_id] = record
    return meta, ids


def resolve_textocr_image_urls(target_ids: set[str]) -> dict[str, dict]:
    response = requests.get(OPEN_IMAGES_TRAIN_INFO_URL, stream=True, timeout=120, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    response.raw.decode_content = True
    reader = csv.DictReader((line.decode("utf-8") for line in response.iter_lines() if line))
    matched: dict[str, dict] = {}
    for row in reader:
        image_id = row["ImageID"]
        if image_id not in target_ids:
            continue
        matched[image_id] = {
            "image_id": image_id,
            "subset": row.get("Subset"),
            "thumbnail_url": row.get("Thumbnail300KURL") or "",
            "original_url": row.get("OriginalURL") or "",
            "rotation": row.get("Rotation"),
        }
        if len(matched) == len(target_ids):
            break
    return matched


def choose_textocr_url(info: dict) -> str:
    return info["thumbnail_url"] or info["original_url"]


def download_one_textocr(
    session: requests.Session,
    image_id: str,
    url: str,
    dest_dir: Path,
) -> tuple[str, bool, str | None]:
    dest = dest_dir / f"{image_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return image_id, True, None
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return image_id, True, None
    except Exception as exc:  # noqa: BLE001
        return image_id, False, str(exc)


def download_textocr_trainval(max_workers: int) -> None:
    meta, ids = load_textocr_ids()
    print(f"textocr_trainval: target ids = {len(ids)}")
    resolved = resolve_textocr_image_urls(ids)
    print(f"textocr_trainval: resolved URLs = {len(resolved)}")
    TEXT_OCR_DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = TEXT_OCR_DEST / "resolved_openimages_urls.json"
    manifest_path.write_text(json.dumps(resolved, indent=2, sort_keys=True))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    lock = threading.Lock()
    completed = 0
    failures: list[tuple[str, str]] = []
    start = time.time()

    def task(image_id: str, info: dict) -> tuple[str, bool, str | None]:
        return download_one_textocr(session, image_id, choose_textocr_url(info), TEXT_OCR_DEST)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, image_id, info) for image_id, info in resolved.items()]
        for future in concurrent.futures.as_completed(futures):
            image_id, ok, err = future.result()
            with lock:
                completed += 1
                if not ok and err is not None:
                    failures.append((image_id, err))
                if completed % 500 == 0 or completed == len(futures):
                    elapsed = max(time.time() - start, 1e-6)
                    rate = completed / elapsed
                    print(
                        f"textocr_trainval: completed {completed}/{len(futures)} "
                        f"({rate:.1f}/s), failures={len(failures)}",
                        flush=True,
                    )

    if failures:
        (TEXT_OCR_DEST / "download_failures.json").write_text(json.dumps(failures, indent=2))
        print(f"textocr_trainval: failures written to {TEXT_OCR_DEST / 'download_failures.json'}")

    # Keep the subset metadata next to the images for later indexing.
    filtered_meta = {image_id: meta[image_id] for image_id in resolved}
    (TEXT_OCR_DEST / "textocr_trainval_meta.json").write_text(json.dumps(filtered_meta, indent=2, sort_keys=True))


def download_coco_text_snapshot() -> None:
    COCO_TEXT_DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="howard-hou/COCO-Text",
        repo_type="dataset",
        local_dir=str(COCO_TEXT_DEST),
    )
    print(f"coco_text_hf: snapshot downloaded to {COCO_TEXT_DEST}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OCR-heavy datasets for VM SSL.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["textocr_trainval", "coco_text_full"],
        default=["textocr_trainval", "coco_text_full"],
    )
    parser.add_argument("--max-workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "coco_text_full" in args.sources:
        download_coco_text_snapshot()
    if "textocr_trainval" in args.sources:
        download_textocr_trainval(max_workers=args.max_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
