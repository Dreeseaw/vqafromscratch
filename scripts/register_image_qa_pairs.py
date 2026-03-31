#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

import duckdb
from PIL import Image
try:
    import ijson
except ImportError:
    ijson = None


def ensure_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        create table if not exists image_qa_datasets (
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
        create table if not exists image_qa_pairs (
          pair_id text primary key,
          dataset_name text not null,
          image_id text not null,
          pair_split text not null,
          question text not null,
          answer text,
          has_answer boolean not null,
          question_char_length integer not null,
          answer_char_length integer,
          source_record_json text,
          imported_at timestamp not null default current_timestamp
        )
        """
    )
    conn.execute(
        """
        create or replace view valid_image_qa_pairs as
        select
          p.pair_id,
          p.dataset_name,
          p.image_id,
          p.pair_split,
          p.question,
          p.answer,
          p.has_answer,
          p.question_char_length,
          p.answer_char_length,
          p.source_record_json,
          i.source_name,
          i.source_split,
          i.local_path,
          i.width,
          i.height
        from image_qa_pairs p
        join images i using (image_id)
        where i.decode_ok and (i.drop_reason is null or i.drop_reason = '')
        """
    )
    conn.execute(
        """
        create or replace view valid_labeled_image_qa_pairs as
        select *
        from valid_image_qa_pairs
        where has_answer
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
        insert or replace into image_qa_datasets (
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


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if not text:
        return None
    if text.lower() == "none":
        return None
    return text


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_chart_image(image: Image.Image, out_path: Path) -> tuple[str, int, int, int]:
    normalized = image.convert("RGB")
    buffer = BytesIO()
    normalized.save(buffer, format="PNG", optimize=True)
    raw = buffer.getvalue()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        out_path.write_bytes(raw)
    width, height = normalized.size
    return sha256_bytes(raw), width, height, len(raw)


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


def insert_pair_rows(conn: duckdb.DuckDBPyConnection, rows: list[list[object]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        insert or replace into image_qa_pairs (
          pair_id,
          dataset_name,
          image_id,
          pair_split,
          question,
          answer,
          has_answer,
          question_char_length,
          answer_char_length,
          source_record_json
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def iter_gqa_question_items(root_dir: Path, allowed_splits: set[str] | None = None) -> Iterable[tuple[str, str, dict]]:
    if ijson is None:
        raise RuntimeError("ijson is required for GQA import; install it before running this script.")

    extracted_dir = root_dir / "questions"
    if extracted_dir.exists():
        for path in sorted(extracted_dir.glob("*.json")):
            split = split_name_from_filename(path.name)
            if allowed_splits is not None and split not in allowed_splits:
                continue
            with path.open("rb") as f:
                for qid, record in ijson.kvitems(f, ""):
                    yield path.name, str(qid), record
        return

    zip_path = root_dir / "questions1.2.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing GQA question archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member_name in sorted(zf.namelist()):
            if member_name.endswith("/") or not member_name.lower().endswith(".json"):
                continue
            split = split_name_from_filename(Path(member_name).name)
            if allowed_splits is not None and split not in allowed_splits:
                continue
            with zf.open(member_name, "r") as f:
                for qid, record in ijson.kvitems(f, ""):
                    yield Path(member_name).name, str(qid), record


def split_name_from_filename(filename: str) -> str:
    name = Path(filename).stem
    if name.endswith("_questions"):
        name = name[: -len("_questions")]
    return name


def import_gqa(conn: duckdb.DuckDBPyConnection, root_dir: Path, allowed_splits: set[str] | None = None) -> dict[str, int]:
    local_path = str((root_dir / "questions").resolve()) if (root_dir / "questions").exists() else str((root_dir / "questions1.2.zip").resolve())
    split_note = "all official splits" if not allowed_splits else "selected splits: " + ", ".join(sorted(allowed_splits))
    register_dataset(
        conn,
        "gqa_questions_1_2",
        license_name="research / benchmark",
        tier="research",
        local_path=local_path,
        notes=f"Official GQA question files linked against local GQA images; {split_note}. Use valid_labeled_image_qa_pairs for supervised subsets.",
    )

    valid_images = {
        row[0]
        for row in conn.execute(
            "select image_id from images where source_name = 'gqa' and decode_ok and (drop_reason is null or drop_reason = '')"
        ).fetchall()
    }

    pair_rows: list[list[object]] = []
    inserted = 0
    counts_by_split: dict[str, int] = {}

    for filename, qid, record in iter_gqa_question_items(root_dir, allowed_splits=allowed_splits):
        split = split_name_from_filename(filename)
        image_native_id = str(record.get("imageId") or "").strip()
        question = clean_text(record.get("question"))
        if not image_native_id or question is None:
            continue
        image_id = f"gqa:all:{image_native_id}"
        if image_id not in valid_images:
            continue
        answer = clean_text(record.get("answer"))
        pair_rows.append(
            [
                f"gqa_questions_1_2:{split}:{qid}",
                "gqa_questions_1_2",
                image_id,
                split,
                question,
                answer,
                answer is not None,
                len(question),
                len(answer) if answer is not None else None,
                json.dumps(record, sort_keys=True),
            ]
        )
        inserted += 1
        counts_by_split[split] = counts_by_split.get(split, 0) + 1

        if len(pair_rows) >= 2048:
            insert_pair_rows(conn, pair_rows)
            pair_rows.clear()
            if inserted % 50000 == 0:
                print(f"gqa_questions_1_2: inserted {inserted} qa rows", flush=True)

    insert_pair_rows(conn, pair_rows)
    return counts_by_split


def import_chartqa(
    conn: duckdb.DuckDBPyConnection,
    root_dir: Path,
    allowed_splits: set[str] | None = None,
) -> tuple[dict[str, int], int]:
    from datasets import load_dataset

    requested_splits = sorted(allowed_splits or {"train", "val"})
    images_root = root_dir / "images"
    register_dataset(
        conn,
        "chartqa",
        license_name="dataset-specific / benchmark",
        tier="research",
        local_path=str(root_dir.resolve()),
        notes=(
            "HuggingFaceM4/ChartQA train+val import. Chart images are materialized once by content hash "
            "and shared across multiple question-answer rows via image_qa_pairs."
        ),
    )

    existing_ids = {
        row[0]
        for row in conn.execute(
            "select image_id from images where source_name = 'chartqa'"
        ).fetchall()
    }

    image_rows: list[list[object]] = []
    pair_rows: list[list[object]] = []
    counts_by_split: dict[str, int] = {}
    new_images = 0

    for split in requested_splits:
        ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
        human_or_machine_feature = ds.features["human_or_machine"]
        for row_idx, row in enumerate(ds):
            image = row["image"]
            question = clean_text(row.get("query"))
            labels = row.get("label") or []
            answer = clean_text(labels[0] if labels else None)
            if image is None or question is None:
                continue

            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG", optimize=True)
            encoded = buffer.getvalue()
            image_sha = sha256_bytes(encoded)
            image_id = f"chartqa:shared:{image_sha}"
            rel_path = Path("data/vm_ssl/raw/chartqa/images") / image_sha[:2] / f"{image_sha}.png"
            abs_path = Path.cwd() / rel_path

            if image_id not in existing_ids:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                if not abs_path.exists():
                    abs_path.write_bytes(encoded)
                width, height = image.size
                image_rows.append(
                    [
                        image_id,
                        "chartqa",
                        image_sha,
                        "shared",
                        f"hf://HuggingFaceM4/ChartQA/{split}/{row_idx}",
                        str(rel_path),
                        image_sha,
                        width,
                        height,
                        float(width) / float(height) if height else None,
                        len(encoded),
                        True,
                        None,
                    ]
                )
                existing_ids.add(image_id)
                new_images += 1

            record_json = {
                "row_idx": row_idx,
                "human_or_machine": human_or_machine_feature.int2str(int(row["human_or_machine"])),
                "answers": labels,
            }
            pair_rows.append(
                [
                    f"chartqa:{split}:{row_idx}",
                    "chartqa",
                    image_id,
                    split,
                    question,
                    answer,
                    answer is not None,
                    len(question),
                    len(answer) if answer is not None else None,
                    json.dumps(record_json, sort_keys=True),
                ]
            )
            counts_by_split[split] = counts_by_split.get(split, 0) + 1

            if len(image_rows) >= 512:
                insert_image_rows(conn, image_rows)
                image_rows.clear()
            if len(pair_rows) >= 2048:
                insert_pair_rows(conn, pair_rows)
                pair_rows.clear()

        if image_rows:
            insert_image_rows(conn, image_rows)
            image_rows.clear()
        if pair_rows:
            insert_pair_rows(conn, pair_rows)
            pair_rows.clear()

    return counts_by_split, new_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register image-question-answer datasets into DuckDB.")
    parser.add_argument(
        "--db-path",
        default="data/vm_ssl/db/vm_ssl.duckdb",
        help="DuckDB path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["gqa", "chartqa"],
        default=["gqa"],
        help="QA datasets to register.",
    )
    parser.add_argument(
        "--gqa-splits",
        nargs="+",
        choices=[
            "challenge_all",
            "challenge_balanced",
            "submission_all",
            "test_all",
            "test_balanced",
            "testdev_all",
            "testdev_balanced",
            "train_balanced",
            "val_all",
            "val_balanced",
        ],
        default=None,
        help="Optional subset of GQA split files to import.",
    )
    parser.add_argument(
        "--chartqa-splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
        help="ChartQA splits to import. Defaults to train+val.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conn = duckdb.connect(args.db_path)
    ensure_tables(conn)

    for dataset_name in args.datasets:
        if dataset_name == "gqa":
            counts = import_gqa(conn, Path("data/gqa"), allowed_splits=set(args.gqa_splits) if args.gqa_splits else None)
            total = sum(counts.values())
            print(f"gqa_questions_1_2: {total} qa rows registered", flush=True)
            for split, count in sorted(counts.items()):
                print(f"  {split}: {count}", flush=True)
        elif dataset_name == "chartqa":
            counts, new_images = import_chartqa(
                conn,
                Path("data/vm_ssl/raw/chartqa"),
                allowed_splits=set(args.chartqa_splits) if args.chartqa_splits else None,
            )
            total = sum(counts.values())
            print(f"chartqa: {total} qa rows registered; {new_images} new images materialized", flush=True)
            for split, count in sorted(counts.items()):
                print(f"  {split}: {count}", flush=True)
        else:
            raise ValueError(f"unsupported dataset: {dataset_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
