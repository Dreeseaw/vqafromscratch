#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import duckdb
import requests
from PIL import Image

try:
    import ijson
except ImportError:
    ijson = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
POINTING_ROOT = ROOT / "data" / "pointing"
IMAGES_ROOT = POINTING_ROOT / "images"
INDEX_ROOT = POINTING_ROOT / "index"
GQA_ROOT = ROOT / "data" / "gqa"

PIXMO_REPO = "allenai/pixmo-points"
VIDEOPOINT_REPO = "allenai/Molmo2-VideoPoint"
MULTIIMAGE_REPO = "allenai/Molmo2-MultiImagePoint"
VIDEO_YOUTUBE_MAP = "youtube_id_to_urls_mapping.json"
VIDEO_GENERATED_ARCHIVES = [
    "generated_videos/videos.tar.gz",
    "generated_videos/sora2.tar.gz",
    "generated_videos/sora2-1128.tar.gz",
]
GQA_SCENEGRAPH_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"


def require_dep(name: str, value: Any) -> Any:
    if value is None:
        raise RuntimeError(f"Missing required dependency: {name}. Install it before running this script.")
    return value


def progress(iterable: Iterable, *, total: int | None = None, desc: str = "") -> Iterable:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ensure_dirs() -> None:
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


def load_existing_sample_ids(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    seen: set[str] = set()
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = payload.get("sample_id")
            if sample_id:
                seen.add(str(sample_id))
    return seen


def sanitize_split(split: str | None) -> str:
    raw = str(split or "train").strip().lower()
    if raw in {"validation", "valid"}:
        return "val"
    if raw.startswith("val"):
        return "val"
    if raw.startswith("test"):
        return "test"
    return "train"


def directory_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def cap_bytes_from_args(args: argparse.Namespace) -> int | None:
    gib = float(getattr(args, "per_dataset_gib_cap", 0.0) or 0.0)
    if gib <= 0.0:
        return None
    return int(gib * (1024 ** 3))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def detect_point_scale(point_group: list[dict[str, Any]]) -> float | None:
    max_coord = 0.0
    for pt in point_group:
        max_coord = max(max_coord, float(pt.get("x", 0.0)), float(pt.get("y", 0.0)))
    if max_coord <= 1.0:
        return 1.0
    if max_coord <= 100.0:
        return 100.0
    if max_coord <= 1000.0:
        return 1000.0
    return None


def normalize_xy(x: float, y: float, *, width: int, height: int, explicit_scale: float | None) -> tuple[float, float] | None:
    x = float(x)
    y = float(y)
    if explicit_scale is not None:
        if explicit_scale <= 1.0:
            return clamp01(x), clamp01(y)
        return clamp01(x / explicit_scale), clamp01(y / explicit_scale)
    if width <= 0 or height <= 0:
        return None
    return clamp01(x / float(width)), clamp01(y / float(height))


def write_jpeg(image: Image.Image, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    rgb.save(out_path, format="JPEG", quality=95)
    return rgb.size


def fetch_image(session: requests.Session, url: str, *, expected_sha256: str | None = None) -> tuple[Image.Image, bytes]:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    raw = resp.content
    if expected_sha256 is not None and sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"sha256 mismatch for {url}")
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.load()
    return image, raw


def first_nonempty_point_group(points: list[list[dict[str, Any]]], timestamps: list[float]) -> tuple[int, list[dict[str, Any]], float] | None:
    for idx, group in enumerate(points):
        if not group:
            continue
        if idx >= len(timestamps):
            continue
        return idx, group, float(timestamps[idx])
    return None


def path_for_video_id(video_root: Path, video_id: str) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".avi"):
        matches = list(video_root.rglob(f"{video_id}{ext}"))
        if matches:
            return matches[0]
    return None


def cache_download(session: requests.Session, url: str, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return out_path


def derive_2fps_gcp_url(gcp_url: str, video_id: str) -> str | None:
    parsed = urlparse(gcp_url)
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) != 2:
        return None
    bucket, blob = parts
    blob_parts = blob.split("/", 1)
    top_dir = blob_parts[0]
    rest = blob_parts[1] if len(blob_parts) == 2 else ""
    twofps_blob = f"{top_dir}-2fps/{video_id}_2fps.mp4"
    return f"{parsed.scheme}://{parsed.netloc}/{bucket}/{twofps_blob}"


def load_youtube_mapping(cache_dir: Path) -> dict[str, dict[str, Any]]:
    require_dep("datasets", load_dataset)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=VIDEOPOINT_REPO, repo_type="dataset", filename=VIDEO_YOUTUBE_MAP, local_dir=cache_dir)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_generated_video(video_id: str, cache_dir: Path) -> Path | None:
    from huggingface_hub import hf_hub_download

    extracted_dir = cache_dir / "generated_videos"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    maybe_existing = path_for_video_id(extracted_dir, video_id)
    if maybe_existing is not None:
        return maybe_existing

    for archive_name in VIDEO_GENERATED_ARCHIVES:
        archive_path = Path(
            hf_hub_download(repo_id=VIDEOPOINT_REPO, repo_type="dataset", filename=archive_name, local_dir=cache_dir)
        )
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                member_name = Path(member.name).name
                if Path(member_name).stem != video_id:
                    continue
                target = extracted_dir / member_name
                if not target.exists():
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    target.write_bytes(extracted.read())
                return target
    return None


def resolve_video_asset(
    *,
    session: requests.Session,
    cache_dir: Path,
    video_id: str,
    video_source: str,
    mammalnet_root: Path | None,
    youtube_mapping: dict[str, dict[str, Any]] | None,
) -> Path | None:
    video_source = str(video_source or "").strip().lower()
    if video_source == "youtube":
        mapping = youtube_mapping or {}
        if video_id not in mapping:
            return None
        gcp_url = mapping[video_id].get("gcp_url")
        if not gcp_url:
            return None
        cache_path = cache_dir / "youtube" / f"{video_id}_2fps.mp4"
        derived = derive_2fps_gcp_url(str(gcp_url), video_id)
        urls = [u for u in (derived, gcp_url) if u]
        for url in urls:
            try:
                return cache_download(session, str(url), cache_path)
            except Exception:
                continue
        return None
    if video_source == "generated":
        return find_generated_video(video_id, cache_dir)
    if video_source == "mammalnet":
        if mammalnet_root is None:
            return None
        return path_for_video_id(mammalnet_root, video_id)
    return None


def maybe_delete_temp_video(video_path: Path, cache_dir: Path) -> None:
    try:
        resolved_video = video_path.resolve()
        resolved_cache = cache_dir.resolve()
    except FileNotFoundError:
        return
    if resolved_cache in resolved_video.parents:
        video_path.unlink(missing_ok=True)


def extract_video_frame(video_path: Path, timestamp_s: float) -> tuple[Image.Image, int]:
    require_dep("decord", VideoReader)
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(vr.get_avg_fps() or 0.0)
    if fps <= 0.0:
        raise RuntimeError(f"Could not determine fps for {video_path}")
    frame_idx = int(round(max(0.0, float(timestamp_s)) * fps))
    frame_idx = max(0, min(frame_idx, len(vr) - 1))
    frame = vr[frame_idx].asnumpy()
    image = Image.fromarray(frame).convert("RGB")
    return image, frame_idx


def validate_index_file(index_path: Path, invalid_path: Path) -> tuple[int, int]:
    valid = 0
    invalid = 0
    with index_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            payload = json.loads(line)
            image_path = ROOT / payload["image_path"] if not Path(payload["image_path"]).is_absolute() else Path(payload["image_path"])
            reason = None
            if not image_path.exists():
                reason = "missing_image"
            else:
                for point in payload.get("points", []):
                    x = float(point.get("x", -1.0))
                    y = float(point.get("y", -1.0))
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        reason = "point_out_of_range"
                        break
            if reason is None:
                valid += 1
            else:
                invalid += 1
                with invalid_path.open("a", encoding="utf-8") as bad:
                    bad.write(json.dumps({"source_file": str(index_path), "line_no": line_no, "reason": reason, "record": payload}) + "\n")
    return valid, invalid


def safe_question(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def ingest_pixmo_points(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    require_dep("datasets", load_dataset)
    session = requests.Session()
    ds = load_dataset(PIXMO_REPO, split="train")
    image_dir = IMAGES_ROOT / "pixmo_points" / "train"
    index_path = INDEX_ROOT / "pixmo_points.jsonl"
    existing_ids = load_existing_sample_ids(index_path)
    cap_bytes = cap_bytes_from_args(args)
    bytes_written = directory_bytes(image_dir)
    skip_reasons = Counter()
    written = 0
    with index_path.open("a" if existing_ids else "w", encoding="utf-8") as out:
        for row_idx, row in enumerate(progress(ds, total=len(ds), desc="pixmo_points")):
            if cap_bytes is not None and bytes_written >= cap_bytes:
                skip_reasons["cap_reached"] += 1
                break
            question = safe_question(row.get("label"))
            point_list = list(row.get("points") or [])
            sample_id = f"pixmo_{row_idx:08d}"
            if sample_id in existing_ids:
                continue
            if question is None:
                skip_reasons["missing_question"] += 1
                continue
            if not point_list:
                skip_reasons["no_points"] += 1
                continue
            out_path = image_dir / f"{sample_id}.jpg"
            try:
                image, _ = fetch_image(session, row["image_url"], expected_sha256=row.get("image_sha256"))
                width, height = write_jpeg(image, out_path)
                file_bytes = out_path.stat().st_size if out_path.exists() else 0
                scale = detect_point_scale(point_list)
                norm_points = []
                for pt in point_list:
                    xy = normalize_xy(pt["x"], pt["y"], width=width, height=height, explicit_scale=scale)
                    if xy is None:
                        continue
                    norm_points.append({"x": xy[0], "y": xy[1], "label": row.get("label")})
                if not norm_points:
                    skip_reasons["no_valid_points"] += 1
                    out_path.unlink(missing_ok=True)
                    continue
                answer = None
                if row.get("count") is not None:
                    answer = str(int(row["count"]))
                record = {
                    "sample_id": sample_id,
                    "source": "pixmo_points",
                    "image_path": str(out_path.relative_to(ROOT)),
                    "question": question,
                    "points": norm_points,
                    "answer": answer,
                    "metadata": {
                        "original_id": str(row.get("image_sha256") or row_idx),
                        "split": "train",
                        "source_format": "image",
                        "frame_index": None,
                        "bbox": None,
                        "collection_method": row.get("collection_method"),
                    },
                }
                out.write(json.dumps(record) + "\n")
                written += 1
                bytes_written += file_bytes
            except Exception as exc:
                skip_reasons[f"error:{type(exc).__name__}"] += 1
    summary["pixmo_points"] = {"records": written, "skips": dict(skip_reasons), "bytes_on_disk": bytes_written}


def ingest_molmo2_videopoint(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    require_dep("datasets", load_dataset)
    session = requests.Session()
    ds = load_dataset(VIDEOPOINT_REPO, split="train")
    image_dir = IMAGES_ROOT / "molmo2_videopoint" / "train"
    index_path = INDEX_ROOT / "molmo2_videopoint.jsonl"
    existing_ids = load_existing_sample_ids(index_path)
    cap_bytes = cap_bytes_from_args(args)
    bytes_written = directory_bytes(image_dir)
    cache_dir = POINTING_ROOT / ".cache" / "molmo2_videopoint"
    cache_dir.mkdir(parents=True, exist_ok=True)
    youtube_mapping = load_youtube_mapping(cache_dir)
    mammalnet_root = Path(args.mammalnet_root).resolve() if args.mammalnet_root else None
    skip_reasons = Counter()
    written = 0
    with index_path.open("a" if existing_ids else "w", encoding="utf-8") as out:
        for row_idx, row in enumerate(progress(ds, total=len(ds), desc="molmo2_videopoint")):
            if cap_bytes is not None and bytes_written >= cap_bytes:
                skip_reasons["cap_reached"] += 1
                break
            question = safe_question(row.get("question"))
            sample_id = f"videopoint_{row_idx:08d}"
            if sample_id in existing_ids:
                continue
            if question is None:
                skip_reasons["missing_question"] += 1
                continue
            twofps = list(row.get("two_fps_timestamps") or [])
            points = list(row.get("points") or [])
            selection = first_nonempty_point_group(points, twofps)
            if selection is None:
                skip_reasons["no_annotated_timestamp"] += 1
                continue
            pt_idx, point_group, timestamp = selection
            out_path = image_dir / f"{sample_id}.jpg"
            try:
                video_path = resolve_video_asset(
                    session=session,
                    cache_dir=cache_dir,
                    video_id=str(row["video_id"]),
                    video_source=str(row.get("video_source") or ""),
                    mammalnet_root=mammalnet_root,
                    youtube_mapping=youtube_mapping,
                )
                if video_path is None:
                    skip_reasons[f"missing_video:{row.get('video_source', 'unknown')}"] += 1
                    continue
                image, frame_idx = extract_video_frame(video_path, timestamp)
                width, height = write_jpeg(image, out_path)
                file_bytes = out_path.stat().st_size if out_path.exists() else 0
                scale = detect_point_scale(point_group)
                norm_points = []
                for pt in point_group:
                    xy = normalize_xy(pt["x"], pt["y"], width=width, height=height, explicit_scale=scale)
                    if xy is None:
                        continue
                    norm_points.append({"x": xy[0], "y": xy[1], "label": row.get("label")})
                if not norm_points:
                    skip_reasons["no_valid_points"] += 1
                    out_path.unlink(missing_ok=True)
                    continue
                answer = None
                if row.get("count") is not None:
                    answer = str(int(row["count"]))
                record = {
                    "sample_id": sample_id,
                    "source": "molmo2_videopoint",
                    "image_path": str(out_path.relative_to(ROOT)),
                    "question": question,
                    "points": norm_points,
                    "answer": answer,
                    "metadata": {
                        "original_id": str(row.get("video_id") or row_idx),
                        "split": "train",
                        "source_format": "video_frame",
                        "frame_index": int(frame_idx),
                        "bbox": None,
                        "video_source": row.get("video_source"),
                        "timestamp": timestamp,
                        "point_time_index": pt_idx,
                        "category": row.get("category"),
                    },
                }
                out.write(json.dumps(record) + "\n")
                written += 1
                bytes_written += file_bytes
                maybe_delete_temp_video(video_path, cache_dir)
            except Exception as exc:
                skip_reasons[f"error:{type(exc).__name__}"] += 1
    summary["molmo2_videopoint"] = {"records": written, "skips": dict(skip_reasons), "bytes_on_disk": bytes_written}


def ingest_molmo2_multiimagepoint(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    require_dep("datasets", load_dataset)
    session = requests.Session()
    ds = load_dataset(MULTIIMAGE_REPO, split="train")
    image_dir = IMAGES_ROOT / "molmo2_multiimagepoint" / "train"
    index_path = INDEX_ROOT / "molmo2_multiimagepoint.jsonl"
    existing_ids = load_existing_sample_ids(index_path)
    cap_bytes = cap_bytes_from_args(args)
    bytes_written = directory_bytes(image_dir)
    skip_reasons = Counter()
    written = 0
    with index_path.open("a" if existing_ids else "w", encoding="utf-8") as out:
        for row_idx, row in enumerate(progress(ds, total=len(ds), desc="molmo2_multiimagepoint")):
            if cap_bytes is not None and bytes_written >= cap_bytes:
                skip_reasons["cap_reached"] += 1
                break
            image_urls = list(row.get("image_urls") or [])
            image_sha256s = list(row.get("image_sha256s") or [])
            labels = list(row.get("labels") or [])
            points_per_image = list(row.get("points") or [])
            counts = list(row.get("counts") or [])
            collection_methods = list(row.get("collection_method") or [])
            any_written = False
            for img_idx, image_url in enumerate(image_urls):
                point_group = points_per_image[img_idx] if img_idx < len(points_per_image) else []
                if not point_group:
                    continue
                question = safe_question(labels[img_idx] if img_idx < len(labels) else None)
                if question is None:
                    skip_reasons["missing_question"] += 1
                    continue
                sample_id = f"multiimage_{row_idx:08d}_{img_idx:02d}"
                if sample_id in existing_ids:
                    any_written = True
                    continue
                out_path = image_dir / f"{sample_id}.jpg"
                try:
                    image, _ = fetch_image(session, image_url, expected_sha256=None)
                    width, height = write_jpeg(image, out_path)
                    file_bytes = out_path.stat().st_size if out_path.exists() else 0
                    scale = detect_point_scale(point_group)
                    norm_points = []
                    for pt in point_group:
                        xy = normalize_xy(pt["x"], pt["y"], width=width, height=height, explicit_scale=scale)
                        if xy is None:
                            continue
                        norm_points.append({"x": xy[0], "y": xy[1], "label": labels[img_idx] if img_idx < len(labels) else None})
                    if not norm_points:
                        skip_reasons["no_valid_points"] += 1
                        out_path.unlink(missing_ok=True)
                        continue
                    answer = None
                    if img_idx < len(counts) and counts[img_idx] is not None:
                        answer = str(int(counts[img_idx]))
                    record = {
                        "sample_id": sample_id,
                        "source": "molmo2_multiimagepoint",
                        "image_path": str(out_path.relative_to(ROOT)),
                        "question": question,
                        "points": norm_points,
                        "answer": answer,
                        "metadata": {
                            "original_id": image_sha256s[img_idx] if img_idx < len(image_sha256s) else f"{row_idx}:{img_idx}",
                            "split": "train",
                            "source_format": "multi_image",
                            "frame_index": None,
                            "bbox": None,
                            "collection_method": collection_methods[img_idx] if img_idx < len(collection_methods) else None,
                        },
                    }
                    out.write(json.dumps(record) + "\n")
                    written += 1
                    bytes_written += file_bytes
                    any_written = True
                except Exception as exc:
                    skip_reasons[f"error:{type(exc).__name__}"] += 1
            if not any_written:
                skip_reasons["no_pointed_images"] += 1
    summary["molmo2_multiimagepoint"] = {"records": written, "skips": dict(skip_reasons), "bytes_on_disk": bytes_written}


def download_if_missing(session: requests.Session, url: str, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return out_path


def iter_scene_graphs(scene_graph_zip: Path, needed_image_ids: set[str]) -> dict[str, dict[str, Any]]:
    if ijson is None:
        raise RuntimeError("ijson is required for streaming scene graphs.")
    found: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(scene_graph_zip, "r") as zf:
        for member_name in sorted(n for n in zf.namelist() if n.endswith(".json")):
            with zf.open(member_name, "r") as f:
                for image_id, payload in ijson.kvitems(f, ""):
                    image_id = str(image_id)
                    if image_id in needed_image_ids:
                        found[image_id] = payload
            if len(found) >= len(needed_image_ids):
                break
    return found


def extract_object_ids(record: dict[str, Any]) -> list[str]:
    annotations = record.get("annotations") or {}
    found: list[str] = []
    for key in ("question", "answer", "fullAnswer"):
        values = annotations.get(key) or {}
        if isinstance(values, dict):
            for obj_id in values.values():
                text = str(obj_id).strip()
                if text:
                    found.append(text)
        elif isinstance(values, list):
            for obj_id in values:
                text = str(obj_id).strip()
                if text:
                    found.append(text)
    deduped = []
    seen = set()
    for obj_id in found:
        if obj_id not in seen:
            seen.add(obj_id)
            deduped.append(obj_id)
    return deduped


def ingest_gqa_sidecar(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    session = requests.Session()
    scene_graph_zip = download_if_missing(session, GQA_SCENEGRAPH_URL, GQA_ROOT / "sceneGraphs.zip")
    conn = duckdb.connect(args.db_path, read_only=True)
    rows = conn.execute(
        """
        select pair_id, pair_split, question, answer, source_record_json, image_id, local_path, width, height
        from valid_labeled_image_qa_pairs
        where dataset_name = 'gqa_questions_1_2'
        """
    ).fetchall()
    needed_image_ids = {str(row[5]).split(":")[-1] for row in rows}
    scene_graphs = iter_scene_graphs(scene_graph_zip, needed_image_ids)
    index_path = INDEX_ROOT / "gqa_point_sidecar.jsonl"
    existing_ids = load_existing_sample_ids(index_path)
    skip_reasons = Counter()
    written = 0
    with index_path.open("a" if existing_ids else "w", encoding="utf-8") as out:
        for pair_id, pair_split, question, answer, source_record_json, image_id, local_path, width, height in progress(rows, total=len(rows), desc="gqa_point_sidecar"):
            sample_id = f"gqa_point_{pair_id}"
            if sample_id in existing_ids:
                continue
            record = json.loads(source_record_json)
            object_ids = extract_object_ids(record)
            if not object_ids:
                skip_reasons["no_object_refs"] += 1
                continue
            image_native_id = str(image_id).split(":")[-1]
            scene_graph = scene_graphs.get(image_native_id)
            if scene_graph is None:
                skip_reasons["missing_scene_graph"] += 1
                continue
            objects = scene_graph.get("objects") or {}
            points = []
            bboxes = []
            for obj_id in object_ids:
                obj = objects.get(str(obj_id))
                if not obj:
                    continue
                x1 = float(obj.get("x", 0.0))
                y1 = float(obj.get("y", 0.0))
                w = float(obj.get("w", 0.0))
                h = float(obj.get("h", 0.0))
                if w <= 0 or h <= 0 or width <= 0 or height <= 0:
                    continue
                x2 = x1 + w
                y2 = y1 + h
                cx = clamp01((x1 + x2) / 2.0 / float(width))
                cy = clamp01((y1 + y2) / 2.0 / float(height))
                label = obj.get("name")
                points.append({"x": cx, "y": cy, "label": label})
                bboxes.append([x1, y1, x2, y2])
            if not points:
                skip_reasons["no_localizable_objects"] += 1
                continue
            meta = {
                "original_id": str(pair_id),
                "split": sanitize_split(str(pair_split)),
                "source_format": "bbox_centroid",
                "frame_index": None,
                "bbox": bboxes[0] if len(bboxes) == 1 else None,
            }
            if len(bboxes) > 1:
                meta["bboxes"] = bboxes
            out.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "source": "gqa",
                        "image_path": str(local_path),
                        "question": question,
                        "points": points,
                        "answer": answer,
                        "metadata": meta,
                    }
                )
                + "\n"
            )
            written += 1
    summary["gqa"] = {"records": written, "skips": dict(skip_reasons)}


def validate_all() -> dict[str, tuple[int, int]]:
    invalid_path = INDEX_ROOT / "invalid_records.jsonl"
    invalid_path.unlink(missing_ok=True)
    results = {}
    for index_path in sorted(INDEX_ROOT.glob("*.jsonl")):
        if index_path.name == "invalid_records.jsonl":
            continue
        valid, invalid = validate_index_file(index_path, invalid_path)
        results[index_path.name] = (valid, invalid)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest pointing datasets and write normalized JSONL indexes.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["pixmo_points", "molmo2_videopoint", "molmo2_multiimagepoint", "gqa"],
        default=["pixmo_points", "molmo2_videopoint", "molmo2_multiimagepoint", "gqa"],
    )
    parser.add_argument("--db-path", default=str(ROOT / "data" / "vm_ssl" / "db" / "vm_ssl.duckdb"))
    parser.add_argument("--mammalnet-root", default="", help="Optional local MammalNet video root for Molmo2-VideoPoint.")
    parser.add_argument("--per-dataset-gib-cap", type=float, default=20.0, help="Approximate per-image-dataset disk cap in GiB. Set <=0 to disable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    summary: dict[str, Any] = {}

    if "pixmo_points" in args.datasets:
        ingest_pixmo_points(args, summary)
    if "molmo2_videopoint" in args.datasets:
        ingest_molmo2_videopoint(args, summary)
    if "molmo2_multiimagepoint" in args.datasets:
        ingest_molmo2_multiimagepoint(args, summary)
    if "gqa" in args.datasets:
        ingest_gqa_sidecar(args, summary)

    validation = validate_all()
    print("\nIngestion Summary")
    for dataset_name, payload in summary.items():
        extra = f" bytes_on_disk={payload.get('bytes_on_disk')}" if "bytes_on_disk" in payload else ""
        print(f"- {dataset_name}: records={payload['records']} skips={payload['skips']}{extra}")
    print("\nValidation Summary")
    for file_name, (valid, invalid) in validation.items():
        print(f"- {file_name}: valid={valid} invalid={invalid}")
    print(f"\nIndex root: {INDEX_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
