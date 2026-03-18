#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import duckdb
import requests
from PIL import Image, UnidentifiedImageError


DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "vqafromscratch-vm-ssl-bootstrap/0.1"


@dataclass(frozen=True)
class SourceConfig:
    name: str
    dataset: str
    config: str
    split: str
    default_count: int
    extract: Callable[[dict[str, Any]], dict[str, Any]]
    fetch_rows_fn: Callable[[requests.Session, "SourceConfig", int, int], list[dict[str, Any]]] | None = None


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "item"


def extract_openimages(row_record: dict[str, Any]) -> dict[str, Any]:
    row = row_record["row"]
    native_id = str(row.get("image_id") or row_record["row_idx"])
    return {
        "source_native_id": native_id,
        "image_url": row["url"],
        "source_record": row,
        "ocr_token_count": None,
        "ocr_area_fraction": None,
    }


def extract_mapillary(row_record: dict[str, Any]) -> dict[str, Any]:
    row = row_record["row"]
    native_id = str(row.get("id") or row_record["row_idx"])
    image = row["image"]
    return {
        "source_native_id": native_id,
        "image_url": image["src"],
        "source_record": row,
        "ocr_token_count": None,
        "ocr_area_fraction": None,
    }


def extract_cocotext(row_record: dict[str, Any]) -> dict[str, Any]:
    row = row_record["row"]
    native_id = str(row.get("image_id") or row_record["row_idx"])
    ocr_tokens = row.get("ocr_tokens") or []
    ocr_info = row.get("ocr_info") or []
    image_width = row.get("image_width") or 0
    image_height = row.get("image_height") or 0
    image_area = max(int(image_width) * int(image_height), 1)
    text_area = 0.0
    for item in ocr_info:
        bbox = item.get("bounding_box") or {}
        text_area += float(bbox.get("width", 0.0)) * float(bbox.get("height", 0.0))
    return {
        "source_native_id": native_id,
        "image_url": row["image"]["src"],
        "source_record": row,
        "ocr_token_count": len(ocr_tokens),
        "ocr_area_fraction": min(text_area / image_area, 1.0) if image_area else None,
    }


def extract_textocr_test(row_record: dict[str, Any]) -> dict[str, Any]:
    row = row_record["row"]
    native_id = str(row["id"])
    return {
        "source_native_id": native_id,
        "image_url": row["download_url"],
        "source_record": row,
        "ocr_token_count": None,
        "ocr_area_fraction": None,
    }


SOURCES: dict[str, SourceConfig] = {
    "openimages_v7": SourceConfig(
        name="openimages_v7",
        dataset="abcd10987/open-images-v7",
        config="default",
        split="train",
        default_count=60,
        extract=extract_openimages,
    ),
    "mapillary_vistas": SourceConfig(
        name="mapillary_vistas",
        dataset="moritzef/mapillary_vistas_semantic_edges_and_segmentation",
        config="default",
        split="train",
        default_count=40,
        extract=extract_mapillary,
    ),
    "coco_text": SourceConfig(
        name="coco_text",
        dataset="howard-hou/COCO-Text",
        config="default",
        split="train",
        default_count=80,
        extract=extract_cocotext,
    ),
    "textocr_test": SourceConfig(
        name="textocr_test",
        dataset="yunusserhat/TextOCR-Dataset",
        config="default",
        split="test",
        default_count=30,
        extract=extract_textocr_test,
        fetch_rows_fn=None,
    ),
}


def fetch_rows(
    session: requests.Session,
    source: SourceConfig,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    if source.fetch_rows_fn is not None:
        return source.fetch_rows_fn(session, source, offset, length)
    response = session.get(
        DATASETS_SERVER,
        params={
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
            "offset": offset,
            "length": length,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("rows", [])


_TEXT_OCR_TREE_CACHE: list[dict[str, Any]] | None = None
_TEXT_OCR_META_CACHE: dict[str, dict[str, Any]] | None = None


def fetch_textocr_test_rows(
    session: requests.Session,
    source: SourceConfig,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    global _TEXT_OCR_TREE_CACHE, _TEXT_OCR_META_CACHE
    if _TEXT_OCR_TREE_CACHE is None:
        tree_url = f"https://huggingface.co/api/datasets/{source.dataset}/tree/main/test_images?recursive=false"
        response = session.get(tree_url, timeout=60)
        response.raise_for_status()
        _TEXT_OCR_TREE_CACHE = response.json()
    if _TEXT_OCR_META_CACHE is None:
        meta_url = f"https://huggingface.co/datasets/{source.dataset}/raw/main/TextOCR_0.1_test.json"
        response = session.get(meta_url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        _TEXT_OCR_META_CACHE = payload.get("imgs", {})

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(_TEXT_OCR_TREE_CACHE[offset : offset + length], start=offset):
        image_id = Path(item["path"]).stem
        rows.append(
            {
                "row_idx": idx,
                "row": {
                    "id": image_id,
                    "path": item["path"],
                    "download_url": f"https://huggingface.co/datasets/{source.dataset}/resolve/main/{item['path']}",
                    "meta": _TEXT_OCR_META_CACHE.get(image_id, {}),
                },
            }
        )
    return rows


SOURCES["textocr_test"] = SourceConfig(
    name="textocr_test",
    dataset="yunusserhat/TextOCR-Dataset",
    config="default",
    split="test",
    default_count=30,
    extract=extract_textocr_test,
    fetch_rows_fn=fetch_textocr_test_rows,
)


def choose_extension(content_type: str | None, image_url: str, fmt: str | None) -> str:
    if fmt:
        fmt = fmt.lower()
        if fmt == "jpeg":
            return ".jpg"
        if fmt == "png":
            return ".png"
        if fmt == "webp":
            return ".webp"
    if content_type:
        content_type = content_type.lower()
        if "jpeg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"
    suffix = Path(image_url.split("?", 1)[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".img"


def download_image(
    session: requests.Session,
    image_url: str,
    dest_dir: Path,
    file_stem: str,
) -> dict[str, Any]:
    response = session.get(image_url, timeout=60)
    response.raise_for_status()
    raw = response.content
    digest = hashlib.sha256(raw).hexdigest()
    file_size_bytes = len(raw)
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"decode failed for {image_url}") from exc

    width, height = image.size
    image_format = image.format
    extension = choose_extension(response.headers.get("Content-Type"), image_url, image_format)
    local_path = dest_dir / f"{file_stem}{extension}"
    local_path.write_bytes(raw)
    return {
        "sha256": digest,
        "file_size_bytes": file_size_bytes,
        "width": width,
        "height": height,
        "local_path": local_path,
        "content_type": response.headers.get("Content-Type"),
    }


def ensure_db(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        create table if not exists images (
          image_id text primary key,
          source_name text not null,
          source_native_id text not null,
          source_split text not null,
          source_path_or_url text not null,
          local_path text not null,
          sha256 text not null,
          width integer not null,
          height integer not null,
          aspect_ratio double not null,
          file_size_bytes bigint not null,
          decode_ok boolean not null,
          drop_reason text,
          imported_at timestamp not null default current_timestamp
        );
        """
    )
    conn.execute(
        """
        create table if not exists image_source_rows (
          image_id text primary key,
          row_idx bigint,
          source_record_json text not null
        );
        """
    )
    conn.execute(
        """
        create table if not exists image_ocr (
          image_id text primary key,
          ocr_token_count integer,
          ocr_area_fraction double
        );
        """
    )
    conn.execute(
        """
        create table if not exists source_downloads (
          source_name text primary key,
          dataset text not null,
          config text not null,
          split text not null,
          downloaded_count integer not null,
          updated_at timestamp not null default current_timestamp
        );
        """
    )
    conn.execute(
        """
        create or replace view valid_images as
        select *
        from images
        where decode_ok = true and drop_reason is null;
        """
    )


def upsert_image(
    conn: duckdb.DuckDBPyConnection,
    image_row: dict[str, Any],
    source_record_json: str,
    row_idx: int,
    ocr_token_count: int | None,
    ocr_area_fraction: float | None,
) -> None:
    conn.execute(
        """
        insert or replace into images (
          image_id,
          source_name,
          source_native_id,
          source_split,
          source_path_or_url,
          local_path,
          sha256,
          width,
          height,
          aspect_ratio,
          file_size_bytes,
          decode_ok,
          drop_reason
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            image_row["image_id"],
            image_row["source_name"],
            image_row["source_native_id"],
            image_row["source_split"],
            image_row["source_path_or_url"],
            image_row["local_path"],
            image_row["sha256"],
            image_row["width"],
            image_row["height"],
            image_row["aspect_ratio"],
            image_row["file_size_bytes"],
            True,
            None,
        ],
    )
    conn.execute(
        """
        insert or replace into image_source_rows (
          image_id,
          row_idx,
          source_record_json
        ) values (?, ?, ?);
        """,
        [image_row["image_id"], row_idx, source_record_json],
    )
    conn.execute(
        """
        insert or replace into image_ocr (
          image_id,
          ocr_token_count,
          ocr_area_fraction
        ) values (?, ?, ?);
        """,
        [image_row["image_id"], ocr_token_count, ocr_area_fraction],
    )


def update_source_download(
    conn: duckdb.DuckDBPyConnection,
    source: SourceConfig,
    downloaded_count: int,
) -> None:
    conn.execute(
        """
        insert or replace into source_downloads (
          source_name,
          dataset,
          config,
          split,
          downloaded_count,
          updated_at
        ) values (?, ?, ?, ?, ?, current_timestamp);
        """,
        [
            source.name,
            source.dataset,
            source.config,
            source.split,
            downloaded_count,
        ],
    )


def bootstrap_source(
    session: requests.Session,
    conn: duckdb.DuckDBPyConnection,
    raw_root: Path,
    source: SourceConfig,
    target_count: int,
    batch_size: int,
    sleep_s: float,
) -> int:
    dest_dir = raw_root / source.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    offset = 0
    while downloaded < target_count:
        rows = fetch_rows(session, source, offset=offset, length=batch_size)
        if not rows:
            break
        for row_record in rows:
            if downloaded >= target_count:
                break
            extracted = source.extract(row_record)
            source_native_id = sanitize_filename(extracted["source_native_id"])
            image_id = f"{source.name}:{source_native_id}"
            image_url = extracted["image_url"]
            try:
                downloaded_image = download_image(session, image_url, dest_dir, source_native_id)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {source.name} {source_native_id}: {exc}", file=sys.stderr)
                continue

            aspect_ratio = (
                downloaded_image["width"] / downloaded_image["height"]
                if downloaded_image["height"]
                else 0.0
            )
            image_row = {
                "image_id": image_id,
                "source_name": source.name,
                "source_native_id": source_native_id,
                "source_split": source.split,
                "source_path_or_url": image_url,
                "local_path": str(downloaded_image["local_path"]),
                "sha256": downloaded_image["sha256"],
                "width": downloaded_image["width"],
                "height": downloaded_image["height"],
                "aspect_ratio": aspect_ratio,
                "file_size_bytes": downloaded_image["file_size_bytes"],
            }
            upsert_image(
                conn,
                image_row=image_row,
                source_record_json=json.dumps(extracted["source_record"], sort_keys=True),
                row_idx=row_record["row_idx"],
                ocr_token_count=extracted["ocr_token_count"],
                ocr_area_fraction=extracted["ocr_area_fraction"],
            )
            downloaded += 1
        offset += len(rows)
        if sleep_s > 0:
            time.sleep(sleep_s)
    update_source_download(conn, source, downloaded)
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a pilot VM SSL image corpus and register it in DuckDB."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["openimages_v7", "mapillary_vistas", "coco_text", "textocr_test"],
        choices=sorted(SOURCES),
        help="Sources to pull pilot slices from.",
    )
    parser.add_argument(
        "--count-per-source",
        type=int,
        default=None,
        help="Override the default image count for every selected source.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Dataset-server row fetch batch size.",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.2,
        help="Sleep between row batches to stay polite.",
    )
    parser.add_argument(
        "--data-root",
        default="data/vm_ssl",
        help="VM SSL data root directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    raw_root = data_root / "raw"
    db_dir = data_root / "db"
    raw_root.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "vm_ssl.duckdb"

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    conn = duckdb.connect(str(db_path))
    ensure_db(conn)

    totals: dict[str, int] = {}
    for source_name in args.sources:
        source = SOURCES[source_name]
        target_count = args.count_per_source or source.default_count
        count = bootstrap_source(
            session=session,
            conn=conn,
            raw_root=raw_root,
            source=source,
            target_count=target_count,
            batch_size=args.batch_size,
            sleep_s=args.sleep_s,
        )
        totals[source_name] = count
        print(f"{source_name}: downloaded {count} images")

    print(f"duckdb: {db_path}")
    for source_name, count in totals.items():
        print(f"  {source_name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
