#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import duckdb
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass(frozen=True)
class RowPayload:
    image_id: str
    source_name: str
    source_native_id: str
    source_split: str
    source_path_or_url: str
    local_path: str
    sha256: str
    width: int
    height: int
    aspect_ratio: float
    file_size_bytes: int
    decode_ok: bool
    drop_reason: str | None
    artifact_name: str
    member_path: str
    source_record_json: str | None
    ocr_token_count: int | None = None
    ocr_area_fraction: float | None = None


def ensure_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        create table if not exists image_artifact_members (
          image_id text primary key,
          artifact_name text not null,
          member_path text not null,
          indexed_at timestamp not null default current_timestamp
        )
        """
    )
    conn.execute(
        """
        create table if not exists artifact_index_runs (
          artifact_name text primary key,
          source_name text not null,
          indexed_image_count bigint not null,
          status text,
          updated_at timestamp not null default current_timestamp
        )
        """
    )
    try:
        conn.execute("alter table artifact_index_runs add column status text")
    except duckdb.Error:
        pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image_info(data: bytes) -> tuple[int, int]:
    image = Image.open(BytesIO(data))
    return image.size


def image_rows_from_tar(
    archive_path: Path,
    artifact_name: str,
    source_name: str,
    source_split: str,
    indexed_member_paths: set[str] | None = None,
) -> Iterable[RowPayload]:
    with tarfile.open(archive_path, "r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            if indexed_member_paths is not None and member.name in indexed_member_paths:
                continue
            file_obj = tf.extractfile(member)
            if file_obj is None:
                continue
            raw = file_obj.read()
            digest = sha256_bytes(raw)
            source_native_id = member.name
            image_id = f"{source_name}:{source_split}:{member.name}"
            pseudo_path = f"{archive_path}::{member.name}"
            width = 0
            height = 0
            decode_ok = True
            drop_reason = None
            try:
                width, height = decode_image_info(raw)
            except (UnidentifiedImageError, OSError) as exc:
                decode_ok = False
                drop_reason = f"decode_failed:{type(exc).__name__}"
            yield RowPayload(
                image_id=image_id,
                source_name=source_name,
                source_native_id=source_native_id,
                source_split=source_split,
                source_path_or_url=pseudo_path,
                local_path=pseudo_path,
                sha256=digest,
                width=width,
                height=height,
                aspect_ratio=(width / height) if height else 0.0,
                file_size_bytes=len(raw),
                decode_ok=decode_ok,
                drop_reason=drop_reason,
                artifact_name=artifact_name,
                member_path=member.name,
                source_record_json=None,
            )


def load_textocr_test_meta(textocr_root: Path) -> dict[str, dict]:
    meta_path = textocr_root / "TextOCR_0.1_test.json"
    if not meta_path.exists():
        return {}
    payload = json.loads(meta_path.read_text())
    return payload.get("imgs", {})


def load_textocr_trainval_meta(textocr_root: Path) -> dict[str, dict]:
    meta_path = textocr_root / "textocr_trainval_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def image_rows_from_directory(
    root_dir: Path,
    artifact_name: str,
    source_name: str,
    source_split: str,
    relative_glob: str,
    meta_by_id: dict[str, dict] | None = None,
    indexed_member_paths: set[str] | None = None,
) -> Iterable[RowPayload]:
    for path in sorted(root_dir.glob(relative_glob)):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        rel_path = str(path.relative_to(root_dir))
        if indexed_member_paths is not None and rel_path in indexed_member_paths:
            continue
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        image_stem = path.stem
        width = 0
        height = 0
        decode_ok = True
        drop_reason = None
        try:
            width, height = decode_image_info(raw)
        except (UnidentifiedImageError, OSError) as exc:
            decode_ok = False
            drop_reason = f"decode_failed:{type(exc).__name__}"
        source_record = meta_by_id.get(image_stem) if meta_by_id else None
        image_id = f"{source_name}:{source_split}:{image_stem}"
        yield RowPayload(
            image_id=image_id,
            source_name=source_name,
            source_native_id=image_stem,
            source_split=source_split,
            source_path_or_url=str(path),
            local_path=str(path),
            sha256=digest,
            width=width,
            height=height,
            aspect_ratio=(width / height) if height else 0.0,
            file_size_bytes=len(raw),
            decode_ok=decode_ok,
            drop_reason=drop_reason,
            artifact_name=artifact_name,
            member_path=rel_path,
            source_record_json=json.dumps(source_record, sort_keys=True) if source_record else None,
        )


def image_rows_from_coco_local(
    root_dir: Path,
    artifact_name: str,
    source_name: str,
    indexed_member_paths: set[str] | None = None,
) -> Iterable[RowPayload]:
    for split in ("train2014", "val2014", "test2015"):
        for path in sorted((root_dir / split).glob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            rel_path = str(path.relative_to(root_dir))
            if indexed_member_paths is not None and rel_path in indexed_member_paths:
                continue
            raw = path.read_bytes()
            digest = sha256_bytes(raw)
            width = 0
            height = 0
            decode_ok = True
            drop_reason = None
            try:
                width, height = decode_image_info(raw)
            except (UnidentifiedImageError, OSError) as exc:
                decode_ok = False
                drop_reason = f"decode_failed:{type(exc).__name__}"
            file_name = path.name
            yield RowPayload(
                image_id=f"{source_name}:{split}:{file_name}",
                source_name=source_name,
                source_native_id=file_name,
                source_split=split,
                source_path_or_url=str(path),
                local_path=str(path),
                sha256=digest,
                width=width,
                height=height,
                aspect_ratio=(width / height) if height else 0.0,
                file_size_bytes=len(raw),
                decode_ok=decode_ok,
                drop_reason=drop_reason,
                artifact_name=artifact_name,
                member_path=rel_path,
                source_record_json=None,
            )


def image_rows_from_textocr_trainval(
    root_dir: Path,
    artifact_name: str,
    source_name: str,
    indexed_member_paths: set[str] | None = None,
) -> Iterable[RowPayload]:
    meta_by_id = load_textocr_trainval_meta(root_dir)
    for path in sorted(root_dir.glob("*.jpg")):
        if not path.is_file():
            continue
        rel_path = path.name
        if indexed_member_paths is not None and rel_path in indexed_member_paths:
            continue
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        image_stem = path.stem
        meta = meta_by_id.get(image_stem, {})
        source_split = str(meta.get("set", "trainval"))
        width = 0
        height = 0
        decode_ok = True
        drop_reason = None
        try:
            width, height = decode_image_info(raw)
        except (UnidentifiedImageError, OSError) as exc:
            decode_ok = False
            drop_reason = f"decode_failed:{type(exc).__name__}"
        yield RowPayload(
            image_id=f"{source_name}:{source_split}:{image_stem}",
            source_name=source_name,
            source_native_id=image_stem,
            source_split=source_split,
            source_path_or_url=str(path),
            local_path=str(path),
            sha256=digest,
            width=width,
            height=height,
            aspect_ratio=(width / height) if height else 0.0,
            file_size_bytes=len(raw),
            decode_ok=decode_ok,
            drop_reason=drop_reason,
            artifact_name=artifact_name,
            member_path=rel_path,
            source_record_json=json.dumps(meta, sort_keys=True) if meta else None,
        )


def image_rows_from_cocotext_parquet(
    root_dir: Path,
    artifact_name: str,
    source_name: str,
    indexed_member_paths: set[str] | None = None,
) -> Iterable[RowPayload]:
    parquet_glob = str(root_dir / "data" / "*.parquet")
    scan = duckdb.connect()
    cur = scan.execute(
        f"""
        select image, coco_file_name, image_id, caption, ocr_tokens, ocr_info, image_width, image_height
        from read_parquet('{parquet_glob}')
        """
    )
    while True:
        batch = cur.fetchmany(256)
        if not batch:
            break
        for image, coco_file_name, image_id, caption, ocr_tokens, ocr_info, image_width, image_height in batch:
            source_split = "train" if "train" in (coco_file_name or "").lower() else "validation"
            member_path = f"{source_split}/{image_id}"
            if indexed_member_paths is not None and member_path in indexed_member_paths:
                continue
            raw = image["bytes"]
            digest = sha256_bytes(raw)
            text_area = 0.0
            for item in ocr_info or []:
                bbox = item.get("bounding_box") or {}
                text_area += float(bbox.get("width", 0.0)) * float(bbox.get("height", 0.0))
            image_area = max(int(image_width or 0) * int(image_height or 0), 1)
            yield RowPayload(
                image_id=f"{source_name}:{source_split}:{image_id}",
                source_name=source_name,
                source_native_id=str(image_id),
                source_split=source_split,
                source_path_or_url=str(root_dir / "data"),
                local_path=f"{root_dir}::{coco_file_name}",
                sha256=digest,
                width=int(image_width or 0),
                height=int(image_height or 0),
                aspect_ratio=(float(image_width) / float(image_height)) if image_height else 0.0,
                file_size_bytes=len(raw),
                decode_ok=True,
                drop_reason=None,
                artifact_name=artifact_name,
                member_path=member_path,
                source_record_json=json.dumps(
                    {
                        "coco_file_name": coco_file_name,
                        "caption": caption,
                    },
                    sort_keys=True,
                ),
                ocr_token_count=len(ocr_tokens or []),
                ocr_area_fraction=min(text_area / image_area, 1.0) if image_area else None,
            )
    scan.close()


def existing_artifact_member_paths(
    conn: duckdb.DuckDBPyConnection,
    artifact_name: str,
) -> set[str]:
    rows = conn.execute(
        """
        select member_path
        from image_artifact_members
        where artifact_name = ?
        """,
        [artifact_name],
    ).fetchall()
    return {row[0] for row in rows}


def image_rows_from_cocotext_parquet_materialized(
    root_dir: Path,
    output_dir: Path,
    artifact_name: str,
    source_name: str,
    target_split: str,
    indexed_member_paths: set[str] | None = None,
) -> Iterable[RowPayload]:
    parquet_glob = str(root_dir / "data" / "*.parquet")
    scan = duckdb.connect()
    cur = scan.execute(
        f"""
        select image, coco_file_name, image_id, caption, ocr_tokens, ocr_info, image_width, image_height
        from read_parquet('{parquet_glob}')
        """
    )
    split_dir = output_dir / target_split
    split_dir.mkdir(parents=True, exist_ok=True)
    while True:
        batch = cur.fetchmany(256)
        if not batch:
            break
        for image, coco_file_name, image_id, caption, ocr_tokens, ocr_info, image_width, image_height in batch:
            source_split = "train" if "train" in (coco_file_name or "").lower() else "validation"
            if source_split != target_split:
                continue
            member_path = f"{target_split}/{coco_file_name}"
            if indexed_member_paths is not None and member_path in indexed_member_paths:
                continue
            raw = image["bytes"]
            out_path = split_dir / str(coco_file_name)
            if not out_path.exists():
                out_path.write_bytes(raw)
            digest = sha256_bytes(raw)
            text_area = 0.0
            for item in ocr_info or []:
                bbox = item.get("bounding_box") or {}
                text_area += float(bbox.get("width", 0.0)) * float(bbox.get("height", 0.0))
            image_area = max(int(image_width or 0) * int(image_height or 0), 1)
            yield RowPayload(
                image_id=f"{source_name}:{source_split}:{image_id}",
                source_name=source_name,
                source_native_id=str(image_id),
                source_split=source_split,
                source_path_or_url=str(out_path),
                local_path=str(out_path),
                sha256=digest,
                width=int(image_width or 0),
                height=int(image_height or 0),
                aspect_ratio=(float(image_width) / float(image_height)) if image_height else 0.0,
                file_size_bytes=len(raw),
                decode_ok=True,
                drop_reason=None,
                artifact_name=artifact_name,
                member_path=member_path,
                source_record_json=json.dumps(
                    {
                        "coco_file_name": coco_file_name,
                        "caption": caption,
                    },
                    sort_keys=True,
                ),
                ocr_token_count=len(ocr_tokens or []),
                ocr_area_fraction=min(text_area / image_area, 1.0) if image_area else None,
            )
    scan.close()


def flush_rows(conn: duckdb.DuckDBPyConnection, rows: list[RowPayload]) -> None:
    if not rows:
        return
    conn.executemany(
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
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            [
                row.image_id,
                row.source_name,
                row.source_native_id,
                row.source_split,
                row.source_path_or_url,
                row.local_path,
                row.sha256,
                row.width,
                row.height,
                row.aspect_ratio,
                row.file_size_bytes,
                row.decode_ok,
                row.drop_reason,
            ]
            for row in rows
        ],
    )
    conn.executemany(
        """
        insert or replace into image_artifact_members (
          image_id,
          artifact_name,
          member_path,
          indexed_at
        ) values (?, ?, ?, current_timestamp)
        """,
        [[row.image_id, row.artifact_name, row.member_path] for row in rows],
    )
    source_rows = [row for row in rows if row.source_record_json is not None]
    if source_rows:
        conn.executemany(
            """
            insert or replace into image_source_rows (
              image_id,
              row_idx,
              source_record_json
            ) values (?, ?, ?)
            """,
            [[row.image_id, None, row.source_record_json] for row in source_rows],
        )
    ocr_rows = [row for row in rows if row.ocr_token_count is not None or row.ocr_area_fraction is not None]
    if ocr_rows:
        conn.executemany(
            """
            insert or replace into image_ocr (
              image_id,
              ocr_token_count,
              ocr_area_fraction
            ) values (?, ?, ?)
            """,
            [[row.image_id, row.ocr_token_count, row.ocr_area_fraction] for row in ocr_rows],
        )


def existing_member_paths(
    conn: duckdb.DuckDBPyConnection,
    artifact_name: str,
    source_name: str,
    source_split: str,
) -> set[str]:
    rows = conn.execute(
        """
        select member_path
        from image_artifact_members
        where artifact_name = ?
        union
        select source_native_id
        from images
        where source_name = ? and source_split = ?
        union
        select source_native_id || '.jpg'
        from images
        where source_name = ? and source_split = ?
        union
        select source_split || '/' || source_native_id
        from images
        where source_name = ? and source_split = ?
        union
        select source_split || '/' || source_native_id || '.jpg'
        from images
        where source_name = ? and source_split = ?
        """,
        [
            artifact_name,
            source_name,
            source_split,
            source_name,
            source_split,
            source_name,
            source_split,
            source_name,
            source_split,
        ],
    ).fetchall()
    return {row[0] for row in rows}


def record_progress(
    conn: duckdb.DuckDBPyConnection,
    artifact_name: str,
    source_name: str,
    indexed_image_count: int,
    status: str,
) -> None:
    conn.execute(
        """
        insert or replace into artifact_index_runs (
          artifact_name,
          source_name,
          indexed_image_count,
          status,
          updated_at
        ) values (?, ?, ?, ?, current_timestamp)
        """,
        [artifact_name, source_name, indexed_image_count, status],
    )


def index_stream(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[RowPayload],
    artifact_name: str,
    source_name: str,
    batch_size: int,
    progress_every: int,
    starting_count: int = 0,
) -> int:
    buffer: list[RowPayload] = []
    count = 0
    total_count = starting_count
    start = time.time()
    record_progress(conn, artifact_name, source_name, total_count, "running")
    for row in rows:
        buffer.append(row)
        count += 1
        total_count += 1
        if len(buffer) >= batch_size:
            flush_rows(conn, buffer)
            buffer.clear()
            record_progress(conn, artifact_name, source_name, total_count, "running")
        if total_count % progress_every == 0:
            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0.0
            print(f"{artifact_name}: indexed {total_count} images ({rate:.1f}/s in this run)", flush=True)
    flush_rows(conn, buffer)
    record_progress(conn, artifact_name, source_name, total_count, "complete")
    return total_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index downloaded VM SSL assets into DuckDB image rows.")
    parser.add_argument(
        "--db-path",
        default="data/vm_ssl/db/vm_ssl.duckdb",
        help="DuckDB path.",
    )
    parser.add_argument(
        "--artifacts",
        nargs="+",
        choices=[
            "inat2021_train_mini",
            "textocr_full",
            "textocr_trainval",
            "coco_text_full",
            "coco_text_train_materialized",
            "coco_local",
            "gqa_images",
        ],
        default=["inat2021_train_mini", "textocr_full"],
        help="Artifacts to index.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conn = duckdb.connect(args.db_path)
    ensure_tables(conn)

    for artifact in args.artifacts:
        if artifact == "inat2021_train_mini":
            indexed = existing_member_paths(conn, "inat2021_train_mini", "inat2021", "train_mini")
            if indexed:
                print(f"inat2021_train_mini: resuming with {len(indexed)} members already indexed", flush=True)
            count = index_stream(
                conn,
                image_rows_from_tar(
                    archive_path=Path("data/vm_ssl/archives/inat2021_train_mini.tar.gz"),
                    artifact_name="inat2021_train_mini",
                    source_name="inat2021",
                    source_split="train_mini",
                    indexed_member_paths=indexed,
                ),
                artifact_name="inat2021_train_mini",
                source_name="inat2021",
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                starting_count=len(indexed),
            )
            print(f"inat2021_train_mini: finished indexing {count} images", flush=True)
        elif artifact == "textocr_full":
            textocr_root = Path("data/vm_ssl/raw/textocr_full")
            indexed = existing_member_paths(conn, "textocr_full_repo", "textocr", "test")
            if indexed:
                print(f"textocr_full_repo: resuming with {len(indexed)} members already indexed", flush=True)
            count = index_stream(
                conn,
                image_rows_from_directory(
                    root_dir=textocr_root,
                    artifact_name="textocr_full_repo",
                    source_name="textocr",
                    source_split="test",
                    relative_glob="test_images/*",
                    meta_by_id=load_textocr_test_meta(textocr_root),
                    indexed_member_paths=indexed,
                ),
                artifact_name="textocr_full_repo",
                source_name="textocr",
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                starting_count=len(indexed),
            )
            print(f"textocr_full: finished indexing {count} images", flush=True)
        elif artifact == "textocr_trainval":
            root = Path("data/vm_ssl/raw/textocr_trainval")
            indexed = existing_member_paths(conn, "textocr_trainval_openimages", "textocr", "train")
            indexed |= existing_member_paths(conn, "textocr_trainval_openimages", "textocr", "val")
            if indexed:
                print(f"textocr_trainval_openimages: resuming with {len(indexed)} members already indexed", flush=True)
            count = index_stream(
                conn,
                image_rows_from_textocr_trainval(
                    root_dir=root,
                    artifact_name="textocr_trainval_openimages",
                    source_name="textocr",
                    indexed_member_paths=indexed,
                ),
                artifact_name="textocr_trainval_openimages",
                source_name="textocr",
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                starting_count=len(indexed),
            )
            print(f"textocr_trainval: finished indexing {count} images", flush=True)
        elif artifact == "coco_text_full":
            root = Path("data/vm_ssl/raw/coco_text_hf")
            indexed = existing_member_paths(conn, "coco_text_hf_snapshot", "coco_text", "train")
            indexed |= existing_member_paths(conn, "coco_text_hf_snapshot", "coco_text", "validation")
            if indexed:
                print(f"coco_text_hf_snapshot: resuming with {len(indexed)} members already indexed", flush=True)
            count = index_stream(
                conn,
                image_rows_from_cocotext_parquet(
                    root_dir=root,
                    artifact_name="coco_text_hf_snapshot",
                    source_name="coco_text",
                    indexed_member_paths=indexed,
                ),
                artifact_name="coco_text_hf_snapshot",
                source_name="coco_text",
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                starting_count=len(indexed),
            )
            print(f"coco_text_full: finished indexing {count} images", flush=True)
        elif artifact == "coco_text_train_materialized":
            root = Path("data/vm_ssl/raw/coco_text_hf")
            output_dir = Path("data/vm_ssl/raw/coco_text_materialized")
            indexed = existing_artifact_member_paths(conn, "coco_text_train_materialized")
            if indexed:
                print(
                    f"coco_text_train_materialized: resuming with {len(indexed)} members already materialized",
                    flush=True,
                )
            count = index_stream(
                conn,
                image_rows_from_cocotext_parquet_materialized(
                    root_dir=root,
                    output_dir=output_dir,
                    artifact_name="coco_text_train_materialized",
                    source_name="coco_text",
                    target_split="train",
                    indexed_member_paths=indexed,
                ),
                artifact_name="coco_text_train_materialized",
                source_name="coco_text",
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                starting_count=len(indexed),
            )
            print(f"coco_text_train_materialized: finished indexing {count} images", flush=True)
        elif artifact == "coco_local":
            root = Path("images")
            indexed = existing_member_paths(conn, "coco_local_images", "coco_local", "train2014")
            indexed |= existing_member_paths(conn, "coco_local_images", "coco_local", "val2014")
            indexed |= existing_member_paths(conn, "coco_local_images", "coco_local", "test2015")
            if indexed:
                print(f"coco_local_images: resuming with {len(indexed)} members already indexed", flush=True)
            count = index_stream(
                conn,
                image_rows_from_coco_local(
                    root_dir=root,
                    artifact_name="coco_local_images",
                    source_name="coco_local",
                    indexed_member_paths=indexed,
                ),
                artifact_name="coco_local_images",
                source_name="coco_local",
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                starting_count=len(indexed),
            )
            print(f"coco_local: finished indexing {count} images", flush=True)
        elif artifact == "gqa_images":
            root = Path("data/gqa/raw_images")
            indexed = existing_member_paths(conn, "gqa_images_zip", "gqa", "all")
            if indexed:
                print(f"gqa_images_zip: resuming with {len(indexed)} members already indexed", flush=True)
            count = index_stream(
                conn,
                image_rows_from_directory(
                    root_dir=root,
                    artifact_name="gqa_images_zip",
                    source_name="gqa",
                    source_split="all",
                    relative_glob="images/*",
                    indexed_member_paths=indexed,
                ),
                artifact_name="gqa_images_zip",
                source_name="gqa",
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                starting_count=len(indexed),
            )
            print(f"gqa_images: finished indexing {count} images", flush=True)
        else:
            raise ValueError(f"unsupported artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
