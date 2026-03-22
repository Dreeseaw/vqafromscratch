#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

import duckdb
from PIL import Image, UnidentifiedImageError


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
        create table if not exists image_text_datasets (
          dataset_name text primary key,
          license text,
          tier text,
          local_path text,
          notes text,
          updated_at timestamp not null default current_timestamp
        )
        """
    )
    conn.execute(
        """
        create table if not exists image_text_pairs (
          pair_id text primary key,
          dataset_name text not null,
          image_id text not null,
          pair_split text not null,
          text_kind text not null,
          text text not null,
          char_length integer not null,
          source_record_json text,
          imported_at timestamp not null default current_timestamp
        )
        """
    )
    conn.execute(
        """
        create or replace view valid_image_text_pairs as
        select
          p.pair_id,
          p.dataset_name,
          p.image_id,
          p.pair_split,
          p.text_kind,
          p.text,
          p.char_length,
          p.source_record_json,
          i.source_name,
          i.source_split,
          i.local_path,
          i.width,
          i.height
        from image_text_pairs p
        join images i using (image_id)
        where i.decode_ok and (i.drop_reason is null or i.drop_reason = '')
        """
    )


def register_dataset(
    conn: duckdb.DuckDBPyConnection,
    dataset_name: str,
    *,
    license_name: str,
    tier: str,
    local_path: str,
    notes: str,
) -> None:
    conn.execute(
        """
        insert or replace into image_text_datasets (
          dataset_name,
          license,
          tier,
          local_path,
          notes,
          updated_at
        ) values (?, ?, ?, ?, ?, current_timestamp)
        """,
        [dataset_name, license_name, tier, local_path, notes],
    )


def existing_image_ids(conn: duckdb.DuckDBPyConnection, source_name: str) -> set[str]:
    rows = conn.execute(
        "select image_id from images where source_name = ?",
        [source_name],
    ).fetchall()
    return {row[0] for row in rows}


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if not text:
        return None
    if text.lower() == "none":
        return None
    return text


def truncate_first_sentence(value: str | None) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    period_idx = text.find(".")
    if period_idx < 0:
        return text
    clipped = clean_text(text[: period_idx + 1])
    return clipped or text


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def decode_image_info(raw: bytes) -> tuple[int, int]:
    image = Image.open(BytesIO(raw))
    image.load()
    return image.size


def insert_image_rows(conn: duckdb.DuckDBPyConnection, rows: list[list[object]]) -> None:
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
        rows,
    )


def insert_member_rows(conn: duckdb.DuckDBPyConnection, rows: list[list[object]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        insert or replace into image_artifact_members (
          image_id,
          artifact_name,
          member_path,
          indexed_at
        ) values (?, ?, ?, current_timestamp)
        """,
        rows,
    )


def insert_pair_rows(conn: duckdb.DuckDBPyConnection, rows: list[list[object]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        insert or replace into image_text_pairs (
          pair_id,
          dataset_name,
          image_id,
          pair_split,
          text_kind,
          text,
          char_length,
          source_record_json
        ) values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def import_coco_captions(conn: duckdb.DuckDBPyConnection, captions_path: Path) -> tuple[int, int]:
    register_dataset(
        conn,
        "coco_captions_2014",
        license_name="CC-BY-4.0",
        tier="core",
        local_path=str(captions_path),
        notes="COCO 2014 train captions linked against local COCO images.",
    )
    payload = json.loads(captions_path.read_text())
    valid_images = existing_image_ids(conn, "coco_local")
    pair_rows: list[list[object]] = []
    inserted = 0
    skipped = 0
    for ann in payload.get("annotations", []):
        image_native = f"COCO_train2014_{int(ann['image_id']):012d}.jpg"
        image_id = f"coco_local:train2014:{image_native}"
        if image_id not in valid_images:
            skipped += 1
            continue
        text = clean_text(ann.get("caption"))
        if text is None:
            continue
        pair_id = f"coco_captions_2014:train2014:{ann['id']}"
        pair_rows.append(
            [
                pair_id,
                "coco_captions_2014",
                image_id,
                "train2014",
                "caption",
                text,
                len(text),
                json.dumps({"annotation_id": ann["id"]}, sort_keys=True),
            ]
        )
        if len(pair_rows) >= 5000:
            insert_pair_rows(conn, pair_rows)
            inserted += len(pair_rows)
            pair_rows.clear()
            if inserted % 50000 == 0:
                print(f"coco_captions: inserted {inserted} pairs", flush=True)
    insert_pair_rows(conn, pair_rows)
    inserted += len(pair_rows)
    return inserted, skipped


def import_coco_text_captions(conn: duckdb.DuckDBPyConnection) -> int:
    register_dataset(
        conn,
        "coco_text_captions",
        license_name="dataset-specific / benchmark",
        tier="research",
        local_path="data/vm_ssl/raw/coco_text_hf",
        notes="COCO-Text captions extracted from the HF parquet snapshot and linked to materialized COCO-Text images.",
    )
    rows = conn.execute(
        """
        select i.image_id, i.source_split, s.source_record_json
        from images i
        join image_source_rows s using (image_id)
        where i.source_name = 'coco_text'
        """
    ).fetchall()
    pair_rows: list[list[object]] = []
    inserted = 0
    for image_id, source_split, source_record_json in rows:
        payload = json.loads(source_record_json)
        captions = payload.get("caption") or []
        for idx, raw in enumerate(captions):
            text = clean_text(raw)
            if text is None:
                continue
            pair_rows.append(
                [
                    f"coco_text_captions:{image_id}:{idx}",
                    "coco_text_captions",
                    image_id,
                    str(source_split),
                    "caption",
                    text,
                    len(text),
                    json.dumps({"caption_idx": idx}, sort_keys=True),
                ]
            )
            if len(pair_rows) >= 5000:
                insert_pair_rows(conn, pair_rows)
                inserted += len(pair_rows)
                pair_rows.clear()
                if inserted % 20000 == 0:
                    print(f"coco_text_captions: inserted {inserted} pairs", flush=True)
    insert_pair_rows(conn, pair_rows)
    inserted += len(pair_rows)
    return inserted


def import_textcaps(conn: duckdb.DuckDBPyConnection, root_dir: Path) -> tuple[int, int]:
    register_dataset(
        conn,
        "textcaps",
        license_name="research / benchmark",
        tier="research",
        local_path=str(root_dir),
        notes="TextCaps parquet snapshot with materialized local images and caption pairs.",
    )
    parquet_glob = str(root_dir / "data" / "*.parquet")
    out_root = Path("data/vm_ssl/raw/textcaps_materialized")
    out_root.mkdir(parents=True, exist_ok=True)

    scan = duckdb.connect()
    cur = scan.execute(
        f"""
        select
          image,
          image_id,
          set_name,
          image_name,
          image_path,
          image_width,
          image_height,
          caption_str,
          reference_strs
        from read_parquet('{parquet_glob}')
        """
    )

    seen_images: set[str] = existing_image_ids(conn, "textcaps")
    image_rows: list[list[object]] = []
    member_rows: list[list[object]] = []
    captions_by_image: dict[tuple[str, str], set[str]] = {}
    materialized = 0
    next_materialized_log = 5000

    while True:
        batch = cur.fetchmany(128)
        if not batch:
            break
        for image, image_id, set_name, image_name, image_path, image_width, image_height, caption_str, reference_strs in batch:
            split = str(set_name or "train")
            native_name = str(Path(str(image_path)).name or image_name or image_id)
            full_image_id = f"textcaps:{split}:{native_name}"
            image_key = (split, native_name)
            captions_by_image.setdefault(image_key, set())

            for seq in (caption_str or [], reference_strs or []):
                for item in seq or []:
                    text = clean_text(item)
                    if text is not None:
                        captions_by_image[image_key].add(text)

            if full_image_id in seen_images:
                continue

            raw = image["bytes"]
            out_dir = out_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / native_name
            if not out_path.exists():
                out_path.write_bytes(raw)
            try:
                width, height = decode_image_info(raw)
                decode_ok = True
                drop_reason = None
            except (UnidentifiedImageError, OSError) as exc:
                width = int(image_width or 0)
                height = int(image_height or 0)
                decode_ok = False
                drop_reason = f"decode_failed:{type(exc).__name__}"

            image_rows.append(
                [
                    full_image_id,
                    "textcaps",
                    native_name,
                    split,
                    str(out_path),
                    str(out_path),
                    sha256_bytes(raw),
                    width,
                    height,
                    (float(width) / float(height)) if height else 0.0,
                    len(raw),
                    decode_ok,
                    drop_reason,
                ]
            )
            member_rows.append([full_image_id, "textcaps_hf_snapshot", f"{split}/{native_name}"])
            seen_images.add(full_image_id)
            materialized += 1

            if len(image_rows) >= 256:
                insert_image_rows(conn, image_rows)
                insert_member_rows(conn, member_rows)
                image_rows.clear()
                member_rows.clear()
                while materialized >= next_materialized_log:
                    print(f"textcaps: materialized {next_materialized_log} images", flush=True)
                    next_materialized_log += 5000

    insert_image_rows(conn, image_rows)
    insert_member_rows(conn, member_rows)
    scan.close()

    pair_rows: list[list[object]] = []
    pair_count = 0
    next_pair_log = 10000
    for (split, native_name), texts in captions_by_image.items():
        full_image_id = f"textcaps:{split}:{native_name}"
        for idx, text in enumerate(sorted(texts)):
            pair_rows.append(
                [
                    f"textcaps:{split}:{native_name}:{idx}",
                    "textcaps",
                    full_image_id,
                    split,
                    "caption",
                    text,
                    len(text),
                    json.dumps({"caption_idx": idx}, sort_keys=True),
                ]
            )
            pair_count += 1
            if len(pair_rows) >= 1024:
                insert_pair_rows(conn, pair_rows)
                pair_rows.clear()
                while pair_count >= next_pair_log:
                    print(f"textcaps: inserted {next_pair_log} pairs", flush=True)
                    next_pair_log += 10000
    insert_pair_rows(conn, pair_rows)

    image_count = conn.execute("select count(*) from images where source_name = 'textcaps'").fetchone()[0]
    return int(image_count), pair_count


def import_flickr30k(conn: duckdb.DuckDBPyConnection, root_dir: Path) -> tuple[int, int]:
    register_dataset(
        conn,
        "flickr30k",
        license_name="research / benchmark",
        tier="research",
        local_path=str(root_dir),
        notes="Flickr30k zip snapshot with local images and paired captions.",
    )
    zip_path = root_dir / "flickr30k-images.zip"
    csv_path = root_dir / "flickr_annotations_30k.csv"
    if not zip_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"Missing Flickr30k assets under {root_dir}")

    image_dir = root_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    captions_by_file: dict[str, list[str]] = {}
    split_by_file: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            native_name = row.get("filename") or row.get("image_name") or row.get("image")
            split = row.get("split") or "train"
            raw_list = row.get("raw") or "[]"
            if not native_name:
                continue
            try:
                texts = json.loads(raw_list)
            except json.JSONDecodeError:
                texts = [raw_list]
            clean_texts = [text for text in (clean_text(item) for item in texts) if text is not None]
            captions_by_file[native_name] = clean_texts
            split_by_file[native_name] = split

    seen_images = existing_image_ids(conn, "flickr30k")
    image_rows: list[list[object]] = []
    member_rows: list[list[object]] = []
    materialized = 0
    next_materialized_log = 5000

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member_name in zf.namelist():
            if member_name.endswith("/"):
                continue
            member_path = Path(member_name)
            if "__MACOSX" in member_path.parts:
                continue
            native_name = Path(member_name).name
            if native_name.startswith("._"):
                continue
            if not native_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            split = split_by_file.get(native_name, "train")
            full_image_id = f"flickr30k:{split}:{native_name}"
            if full_image_id in seen_images:
                continue
            raw = zf.read(member_name)
            out_path = image_dir / native_name
            if not out_path.exists():
                out_path.write_bytes(raw)
            try:
                width, height = decode_image_info(raw)
                decode_ok = True
                drop_reason = None
            except (UnidentifiedImageError, OSError) as exc:
                width = 0
                height = 0
                decode_ok = False
                drop_reason = f"decode_failed:{type(exc).__name__}"
            image_rows.append(
                [
                    full_image_id,
                    "flickr30k",
                    native_name,
                    split,
                    str(out_path),
                    str(out_path),
                    sha256_bytes(raw),
                    width,
                    height,
                    (float(width) / float(height)) if height else 0.0,
                    len(raw),
                    decode_ok,
                    drop_reason,
                ]
            )
            member_rows.append([full_image_id, "flickr30k_hf_snapshot", native_name])
            seen_images.add(full_image_id)
            materialized += 1

            if len(image_rows) >= 256:
                insert_image_rows(conn, image_rows)
                insert_member_rows(conn, member_rows)
                image_rows.clear()
                member_rows.clear()
                while materialized >= next_materialized_log:
                    print(f"flickr30k: materialized {next_materialized_log} images", flush=True)
                    next_materialized_log += 5000

    insert_image_rows(conn, image_rows)
    insert_member_rows(conn, member_rows)

    pair_rows: list[list[object]] = []
    pair_count = 0
    next_pair_log = 10000
    for native_name, texts in captions_by_file.items():
        split = split_by_file.get(native_name, "train")
        full_image_id = f"flickr30k:{split}:{native_name}"
        for idx, text in enumerate(texts):
            pair_rows.append(
                [
                    f"flickr30k:{split}:{native_name}:{idx}",
                    "flickr30k",
                    full_image_id,
                    split,
                    "caption",
                    text,
                    len(text),
                    json.dumps({"caption_idx": idx}, sort_keys=True),
                ]
            )
            pair_count += 1
            if len(pair_rows) >= 1024:
                insert_pair_rows(conn, pair_rows)
                pair_rows.clear()
                while pair_count >= next_pair_log:
                    print(f"flickr30k: inserted {next_pair_log} pairs", flush=True)
                    next_pair_log += 10000
    insert_pair_rows(conn, pair_rows)

    image_count = conn.execute("select count(*) from images where source_name = 'flickr30k'").fetchone()[0]
    pair_count = conn.execute("select count(*) from image_text_pairs where dataset_name = 'flickr30k'").fetchone()[0]
    return int(image_count), int(pair_count)


def import_cc3m_subset(conn: duckdb.DuckDBPyConnection, root_dir: Path) -> tuple[int, int]:
    register_dataset(
        conn,
        "cc3m_subset_50k",
        license_name="dataset-specific / benchmark",
        tier="research",
        local_path=str(root_dir),
        notes="Random materialized subset from pixparse/cc3m-wds with local JPEG/text pairs.",
    )
    manifest_path = root_dir / "subset_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing subset manifest: {manifest_path}")

    seen_images = existing_image_ids(conn, "cc3m_subset_50k")
    image_rows: list[list[object]] = []
    member_rows: list[list[object]] = []
    pair_rows: list[list[object]] = []
    inserted_images = 0
    inserted_pairs = 0

    with manifest_path.open("r", encoding="utf-8") as f:
      for line in f:
        payload = json.loads(line)
        sample_id = str(payload["sample_id"])
        split = str(payload.get("pair_split") or "train")
        image_path = Path(str(payload["image_path"]))
        if not image_path.exists():
            continue
        native_name = image_path.name
        image_id = f"cc3m_subset_50k:{split}:{native_name}"
        if image_id not in seen_images:
            raw = image_path.read_bytes()
            try:
                width, height = decode_image_info(raw)
                decode_ok = True
                drop_reason = None
            except (UnidentifiedImageError, OSError) as exc:
                width = 0
                height = 0
                decode_ok = False
                drop_reason = f"decode_failed:{type(exc).__name__}"
            image_rows.append(
                [
                    image_id,
                    "cc3m_subset_50k",
                    native_name,
                    split,
                    str(image_path),
                    str(image_path),
                    sha256_bytes(raw),
                    width,
                    height,
                    (float(width) / float(height)) if height else 0.0,
                    len(raw),
                    decode_ok,
                    drop_reason,
                ]
            )
            member_rows.append([image_id, "cc3m_subset_50k_materialized", sample_id])
            seen_images.add(image_id)
            if len(image_rows) >= 256:
                insert_image_rows(conn, image_rows)
                insert_member_rows(conn, member_rows)
                inserted_images += len(image_rows)
                image_rows.clear()
                member_rows.clear()

        text = clean_text(payload.get("text"))
        if text is None:
            continue
        pair_rows.append(
            [
                f"cc3m_subset_50k:{sample_id}",
                "cc3m_subset_50k",
                image_id,
                split,
                "caption",
                text,
                len(text),
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "shard": payload.get("shard"),
                        "key": payload.get("key"),
                    },
                    sort_keys=True,
                ),
            ]
        )
        if len(pair_rows) >= 1024:
            insert_pair_rows(conn, pair_rows)
            inserted_pairs += len(pair_rows)
            pair_rows.clear()

    insert_image_rows(conn, image_rows)
    insert_member_rows(conn, member_rows)
    inserted_images += len(image_rows)
    insert_pair_rows(conn, pair_rows)
    inserted_pairs += len(pair_rows)

    image_count = conn.execute("select count(*) from images where source_name = 'cc3m_subset_50k'").fetchone()[0]
    pair_count = conn.execute("select count(*) from image_text_pairs where dataset_name = 'cc3m_subset_50k'").fetchone()[0]
    return int(image_count), int(pair_count)


def import_midjourney_v6_recap_subset(conn: duckdb.DuckDBPyConnection, root_dir: Path) -> tuple[int, int, int, int]:
    register_dataset(
        conn,
        "midjourney_v6_recap_llava",
        license_name="research / benchmark",
        tier="research",
        local_path=str(root_dir),
        notes="Random 30k subset from Photoroom/midjourney-v6-recap using LLaVA distilled captions, truncated to the first sentence.",
    )
    register_dataset(
        conn,
        "midjourney_v6_recap_gemini",
        license_name="research / benchmark",
        tier="research",
        local_path=str(root_dir),
        notes="Random 30k subset from Photoroom/midjourney-v6-recap using Gemini distilled captions, truncated to the first sentence.",
    )
    register_dataset(
        conn,
        "midjourney_v6_recap_qwen3",
        license_name="research / benchmark",
        tier="research",
        local_path=str(root_dir),
        notes="Random 30k subset from Photoroom/midjourney-v6-recap using Qwen distilled captions, truncated to the first sentence.",
    )

    manifest_path = root_dir / "subset_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing subset manifest: {manifest_path}")

    seen_images = existing_image_ids(conn, "midjourney_v6_recap_30k")
    image_rows: list[list[object]] = []
    member_rows: list[list[object]] = []
    pair_rows: list[list[object]] = []

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        sample_id = str(payload["sample_id"])
        split = str(payload.get("pair_split") or "train")
        image_path = Path(str(payload["image_path"]))
        meta_path = Path(str(payload["meta_path"]))
        if not image_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        native_name = image_path.name
        image_id = f"midjourney_v6_recap_30k:{split}:{native_name}"
        if image_id not in seen_images:
            raw = image_path.read_bytes()
            try:
                width, height = decode_image_info(raw)
                decode_ok = True
                drop_reason = None
            except (UnidentifiedImageError, OSError) as exc:
                width = 0
                height = 0
                decode_ok = False
                drop_reason = f"decode_failed:{type(exc).__name__}"
            image_rows.append(
                [
                    image_id,
                    "midjourney_v6_recap_30k",
                    native_name,
                    split,
                    str(image_path),
                    str(image_path),
                    sha256_bytes(raw),
                    width,
                    height,
                    (float(width) / float(height)) if height else 0.0,
                    len(raw),
                    decode_ok,
                    drop_reason,
                ]
            )
            member_rows.append([image_id, "midjourney_v6_recap_30k_materialized", sample_id])
            seen_images.add(image_id)
            if len(image_rows) >= 256:
                insert_image_rows(conn, image_rows)
                insert_member_rows(conn, member_rows)
                image_rows.clear()
                member_rows.clear()

        caption_specs = [
            ("midjourney_v6_recap_llava", "llava", bool(meta.get("llava_status", True))),
            ("midjourney_v6_recap_gemini", "gemini", bool(meta.get("gemini_status", True))),
            ("midjourney_v6_recap_qwen3", "qwen3", bool(meta.get("qwen3_status", True))),
        ]
        for dataset_name, key, is_valid in caption_specs:
            if not is_valid:
                continue
            text = truncate_first_sentence(meta.get(key))
            if text is None:
                continue
            pair_rows.append(
                [
                    f"{dataset_name}:{sample_id}",
                    dataset_name,
                    image_id,
                    split,
                    "caption",
                    text,
                    len(text),
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "source_key": key,
                            "shard": meta.get("shard"),
                            "prompt": meta.get("prompt"),
                        },
                        sort_keys=True,
                    ),
                ]
            )
            if len(pair_rows) >= 1024:
                insert_pair_rows(conn, pair_rows)
                pair_rows.clear()

    insert_image_rows(conn, image_rows)
    insert_member_rows(conn, member_rows)
    insert_pair_rows(conn, pair_rows)

    image_count = conn.execute("select count(*) from images where source_name = 'midjourney_v6_recap_30k'").fetchone()[0]
    llava_count = conn.execute("select count(*) from image_text_pairs where dataset_name = 'midjourney_v6_recap_llava'").fetchone()[0]
    gemini_count = conn.execute("select count(*) from image_text_pairs where dataset_name = 'midjourney_v6_recap_gemini'").fetchone()[0]
    qwen_count = conn.execute("select count(*) from image_text_pairs where dataset_name = 'midjourney_v6_recap_qwen3'").fetchone()[0]
    return int(image_count), int(llava_count), int(gemini_count), int(qwen_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register image-text datasets for VM recipe experiments.")
    parser.add_argument(
        "--db-path",
        default="data/vm_ssl/db/vm_ssl.duckdb",
        help="DuckDB path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["coco_captions", "coco_text_captions", "textcaps", "flickr30k", "cc3m_subset", "midjourney_v6_recap_subset"],
        default=["coco_captions", "coco_text_captions", "textcaps"],
        help="Pair datasets to register.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conn = duckdb.connect(args.db_path)
    ensure_tables(conn)

    for dataset_name in args.datasets:
        if dataset_name == "coco_captions":
            inserted, skipped = import_coco_captions(conn, Path("data/vqav2/captions_train2014.json"))
            print(f"coco_captions: inserted {inserted} pairs (skipped {skipped} missing-image rows)")
        elif dataset_name == "coco_text_captions":
            inserted = import_coco_text_captions(conn)
            print(f"coco_text_captions: inserted {inserted} pairs")
        elif dataset_name == "textcaps":
            image_count, pair_count = import_textcaps(conn, Path("data/vm_ssl/raw/textcaps_hf"))
            print(f"textcaps: {image_count} images registered, {pair_count} pairs registered")
        elif dataset_name == "flickr30k":
            image_count, pair_count = import_flickr30k(conn, Path("data/vm_ssl/raw/flickr30k_hf"))
            print(f"flickr30k: {image_count} images registered, {pair_count} pairs registered")
        elif dataset_name == "cc3m_subset":
            image_count, pair_count = import_cc3m_subset(conn, Path("data/vm_ssl/raw/cc3m_subset_50k"))
            print(f"cc3m_subset_50k: {image_count} images registered, {pair_count} pairs registered")
        elif dataset_name == "midjourney_v6_recap_subset":
            image_count, llava_count, gemini_count, qwen_count = import_midjourney_v6_recap_subset(
                conn, Path("data/vm_ssl/raw/midjourney_v6_recap_30k")
            )
            print(
                "midjourney_v6_recap_30k: "
                f"{image_count} images registered, llava={llava_count}, gemini={gemini_count}, qwen3={qwen_count} pairs registered"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
