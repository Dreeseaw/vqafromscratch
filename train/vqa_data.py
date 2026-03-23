"""
VQAv2 dataset helpers.

Role:
- Download/prepare official VQAv2 + COCO image assets.
- Provide a thin dataset wrapper with explicit multimodal boundaries:
  image tensor + question text + canonical answer target + eval metadata.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import deque
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request as urlrequest

from PIL import Image

import torch
from torch.utils.data import BatchSampler, ConcatDataset, Dataset
from torchvision import transforms

try:
    import ijson
except Exception:
    ijson = None


COLOR_MEAN = (0.485, 0.456, 0.406)
COLOR_STD = (0.229, 0.224, 0.225)

_NUM_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s']")
_SP_RE = re.compile(r"\s+")


def normalize_vqa_answer(text: str) -> str:
    t = str(text or "").strip().lower()
    t = t.replace("\n", " ").replace("\t", " ")
    t = _PUNCT_RE.sub(" ", t)
    words = [_NUM_WORDS.get(w, w) for w in _SP_RE.split(t) if w]
    words = [w for w in words if w not in _ARTICLES]
    return " ".join(words).strip()


def is_number_text(text: str) -> bool:
    t = normalize_vqa_answer(text)
    if not t:
        return False
    if t.isdigit():
        return True
    try:
        float(t)
        return True
    except Exception:
        return False


def heuristic_answer_type(answer: str) -> str:
    t = normalize_vqa_answer(answer)
    if t in ("yes", "no"):
        return "yes/no"
    if is_number_text(t):
        return "number"
    return "other"


def heuristic_question_category(question: str) -> str:
    q = normalize_vqa_answer(question)
    if not q:
        return "other"
    if q.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "was ", "were ", "will ")):
        return "yes/no"
    if "color" in q or "colour" in q:
        return "color"
    if q.startswith("how many") or " number of " in f" {q} " or q.startswith("count "):
        return "count"
    if any(k in q for k in ("left", "right", "behind", "front", "under", "above", "next to", "between")):
        return "spatial relation"
    if any(k in q for k in ("doing", "holding", "playing", "riding", "running", "eating", "drinking")):
        return "action"
    if q.startswith(("what is", "what are", "who is", "who are", "which ")):
        return "object identity"
    if any(k in q for k in ("what kind", "what type", "what size", "what shape", "what material")):
        return "attribute"
    return "other"


def coarse_gqa_question_group(question: str, types: Optional[Dict[str, Any]] = None) -> str:
    q = normalize_vqa_answer(question)
    type_info = dict(types or {})
    semantic = str(type_info.get("semantic", "")).strip().lower()
    detailed = str(type_info.get("detailed", "")).strip().lower()
    if q.startswith("how many") or " number of " in f" {q} " or q.startswith("count "):
        return "count"
    if detailed.startswith("exist"):
        return "exist"
    if semantic == "attr" or "attr" in detailed or detailed in ("material", "color", "place"):
        return "attribute"
    if semantic == "rel" or detailed.startswith(("position", "rel", "direct")):
        return "spatial"
    return "other"


def majority_answer(answers: Sequence[str]) -> str:
    if not answers:
        return ""
    norm = [normalize_vqa_answer(x) for x in answers if str(x).strip()]
    norm = [x for x in norm if x]
    if not norm:
        return ""
    return Counter(norm).most_common(1)[0][0]


def vqa_soft_accuracy(prediction: str, gt_answers: Sequence[str]) -> float:
    if not gt_answers:
        return 0.0
    pred = normalize_vqa_answer(prediction)
    gts = [normalize_vqa_answer(a) for a in gt_answers if str(a).strip()]
    if not gts:
        return 0.0
    match_count = sum(1 for a in gts if a == pred)
    return min(1.0, float(match_count) / 3.0)


@dataclass(frozen=True)
class VQAv2Paths:
    images_root: str
    annotations_root: str


_IMAGE_URLS = {
    "train": "https://images.cocodataset.org/zips/train2014.zip",
    "val": "https://images.cocodataset.org/zips/val2014.zip",
    "test": "https://images.cocodataset.org/zips/test2015.zip",
}
_QUESTION_URLS = {
    "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
}
_ANNOTATION_URLS = {
    "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}
_QUESTION_FILE = {
    "train": "v2_OpenEnded_mscoco_train2014_questions.json",
    "val": "v2_OpenEnded_mscoco_val2014_questions.json",
    "test": "v2_OpenEnded_mscoco_test2015_questions.json",
}
_ANNOTATION_FILE = {
    "train": "v2_mscoco_train2014_annotations.json",
    "val": "v2_mscoco_val2014_annotations.json",
}
_IMAGE_SPLIT_NAME = {
    "train": "train2014",
    "val": "val2014",
    "test": "test2015",
}


def _download_file(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with urlrequest.urlopen(url) as r, open(out_path, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _extract_zip(zip_path: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def _find_file(root: str, filename: str) -> Optional[str]:
    if not root or not os.path.isdir(root):
        return None
    direct = os.path.join(root, filename)
    if os.path.isfile(direct):
        return direct
    for sub in ("annotations", "questions"):
        cand = os.path.join(root, sub, filename)
        if os.path.isfile(cand):
            return cand
    for cur, _, files in os.walk(root):
        if filename in files:
            return os.path.join(cur, filename)
    return None


def resolve_question_file(annotations_root: str, split: str) -> str:
    fn = _QUESTION_FILE[split]
    path = _find_file(annotations_root, fn)
    if path is None:
        raise FileNotFoundError(f"Could not find {fn} under {annotations_root}")
    return path


def resolve_annotation_file(annotations_root: str, split: str) -> Optional[str]:
    fn = _ANNOTATION_FILE.get(split)
    if fn is None:
        return None
    return _find_file(annotations_root, fn)


def resolve_image_dir(images_root: str, split: str) -> str:
    split_name = _IMAGE_SPLIT_NAME[split]
    candidates = [
        os.path.join(images_root, split_name),
        os.path.join(images_root, split_name, split_name),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Could not find image split '{split_name}' under {images_root}")


def resolve_image_path(images_root: str, split: str, image_id: int) -> str:
    split_name = _IMAGE_SPLIT_NAME[split]
    img_dir = resolve_image_dir(images_root, split)
    fn = f"COCO_{split_name}_{int(image_id):012d}.jpg"
    return os.path.join(img_dir, fn)


def prepare_vqav2(
    paths: VQAv2Paths,
    splits: Sequence[str] = ("train", "val"),
    *,
    download_images: bool = True,
    download_test: bool = False,
) -> None:
    needed = set(str(s) for s in splits)
    if download_test:
        needed.add("test")

    os.makedirs(paths.images_root, exist_ok=True)
    os.makedirs(paths.annotations_root, exist_ok=True)
    cache_dir = os.path.join(paths.annotations_root, "_downloads")
    os.makedirs(cache_dir, exist_ok=True)

    for split in sorted(needed):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}")

        if download_images:
            try:
                _ = resolve_image_dir(paths.images_root, split)
            except FileNotFoundError:
                img_zip = os.path.join(cache_dir, f"{_IMAGE_SPLIT_NAME[split]}.zip")
                _download_file(_IMAGE_URLS[split], img_zip)
                _extract_zip(img_zip, paths.images_root)

        try:
            _ = resolve_question_file(paths.annotations_root, split)
        except FileNotFoundError:
            q_zip = os.path.join(cache_dir, f"{split}_questions.zip")
            _download_file(_QUESTION_URLS[split], q_zip)
            _extract_zip(q_zip, paths.annotations_root)

        if split in _ANNOTATION_URLS:
            ann_path = resolve_annotation_file(paths.annotations_root, split)
            if ann_path is None:
                a_zip = os.path.join(cache_dir, f"{split}_annotations.zip")
                _download_file(_ANNOTATION_URLS[split], a_zip)
                _extract_zip(a_zip, paths.annotations_root)


def build_image_transform(train_mode: bool) -> transforms.Compose:
    if train_mode:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(3 / 4, 4 / 3)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(COLOR_MEAN), std=list(COLOR_STD)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(COLOR_MEAN), std=list(COLOR_STD)),
        ]
    )


class VQAv2Dataset(Dataset):
    def __init__(
        self,
        images_root: str,
        annotations_root: str,
        split: str,
        *,
        transform: Optional[transforms.Compose] = None,
        limit: int = 0,
        skip_missing_images: bool = True,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: train, val, test")
        self.images_root = images_root
        self.annotations_root = annotations_root
        self.split = split
        self.transform = transform or build_image_transform(train_mode=(split == "train"))
        self.image_corruption_mode = "none"
        self._corruption_indices: Optional[List[int]] = None

        q_path = resolve_question_file(annotations_root, split)
        with open(q_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
        questions = q_data.get("questions", [])

        ann_map: Dict[int, dict] = {}
        a_path = resolve_annotation_file(annotations_root, split)
        if split in ("train", "val") and (a_path is None or not os.path.isfile(a_path)):
            raise FileNotFoundError(
                f"Missing VQAv2 annotations for split='{split}'. "
                f"Expected file like '{_ANNOTATION_FILE[split]}' under {annotations_root}"
            )
        if a_path is not None and os.path.isfile(a_path):
            with open(a_path, "r", encoding="utf-8") as f:
                a_data = json.load(f)
            for ann in a_data.get("annotations", []):
                qid = int(ann.get("question_id", -1))
                if qid >= 0:
                    ann_map[qid] = ann

        items: List[dict] = []
        for q in questions:
            qid = int(q["question_id"])
            image_id = int(q["image_id"])
            question = str(q.get("question", "")).strip()
            ann = ann_map.get(qid)

            raw_answers: List[str] = []
            if ann is not None:
                raw_answers = [str(a.get("answer", "")).strip() for a in ann.get("answers", []) if str(a.get("answer", "")).strip()]
            canonical = ""
            if ann is not None:
                canonical = normalize_vqa_answer(str(ann.get("multiple_choice_answer", "")).strip())
                if not canonical:
                    canonical = majority_answer(raw_answers)
            question_type_official = ann.get("question_type") if ann is not None else None
            answer_type_official = ann.get("answer_type") if ann is not None else None
            question_type = str(question_type_official) if question_type_official else heuristic_question_category(question)
            answer_type = str(answer_type_official) if answer_type_official else heuristic_answer_type(canonical)

            item = {
                "question_id": qid,
                "image_id": image_id,
                "question": question,
                "answer": canonical,
                "all_answers_raw": raw_answers,
                "all_answers": [normalize_vqa_answer(a) for a in raw_answers if normalize_vqa_answer(a)],
                "metadata": {
                    "split": split,
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "official_question_type": question_type_official,
                    "official_answer_type": answer_type_official,
                    "heuristic_question_type": heuristic_question_category(question),
                    "heuristic_answer_type": heuristic_answer_type(canonical),
                },
            }

            try:
                img_path = resolve_image_path(images_root, split, image_id)
            except FileNotFoundError:
                if skip_missing_images:
                    continue
                raise
            if skip_missing_images and not os.path.isfile(img_path):
                continue
            item["image_path"] = img_path
            items.append(item)

        if limit > 0:
            items = items[: int(limit)]
        self.items = items

    def set_image_corruption(self, mode: str = "none", *, seed: int = 0) -> None:
        mode = str(mode or "none")
        if mode not in ("none", "zero", "shuffle", "random_swap"):
            raise ValueError(
                f"Unsupported image_corruption_mode={mode}. Supported: none, zero, shuffle, random_swap"
            )
        self.image_corruption_mode = mode
        self._corruption_indices = None
        n = len(self.items)
        if n <= 0 or mode in ("none", "zero"):
            return
        if mode == "shuffle":
            if n == 1:
                self._corruption_indices = [0]
            else:
                self._corruption_indices = [((i + 1) % n) for i in range(n)]
            return
        idxs = list(range(n))
        if n == 1:
            self._corruption_indices = idxs
            return
        rng = random.Random(int(seed))
        rng.shuffle(idxs)
        for i in range(n):
            if idxs[i] == i:
                j = (i + 1) % n
                idxs[i], idxs[j] = idxs[j], idxs[i]
        self._corruption_indices = idxs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        source_item = item
        if self._corruption_indices is not None:
            source_item = self.items[self._corruption_indices[idx]]
        img = Image.open(source_item["image_path"]).convert("RGB")
        img_t = self.transform(img)
        if self.image_corruption_mode == "zero":
            img_t = torch.zeros_like(img_t)
        return {
            "image": img_t,
            "question": item["question"],
            "answer": item["answer"],
            "all_answers_raw": item.get("all_answers_raw", []),
            "all_answers": item["all_answers"],
            "question_id": item["question_id"],
            "image_id": item["image_id"],
            "metadata": item["metadata"],
        }


class GQADataset(Dataset):
    def __init__(
        self,
        gqa_root: str,
        split: str,
        *,
        transform: Optional[transforms.Compose] = None,
        limit: int = 0,
        skip_missing_images: bool = True,
        question_group: str = "",
    ) -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError("GQADataset currently only supports split='train' or split='val'")
        self.gqa_root = gqa_root
        self.split = split
        self.transform = transform or build_image_transform(train_mode=(split == "train"))
        self.image_corruption_mode = "none"
        self._corruption_indices: Optional[List[int]] = None
        self.question_group = str(question_group or "").strip().lower()

        image_dir = os.path.join(gqa_root, "raw_images", "images")
        zip_path = os.path.join(gqa_root, "questions1.2.zip")
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Missing GQA question archive: {zip_path}")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Missing GQA image dir: {image_dir}")

        items: List[dict] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            if split == "train":
                member_names = sorted(
                    n
                    for n in zf.namelist()
                    if n.startswith("train_all_questions/") and n.lower().endswith(".json")
                )
            else:
                member_names = ["val_all_questions.json"]
            for member_name in member_names:
                with zf.open(member_name, "r") as f:
                    if ijson is not None:
                        iterator = ijson.kvitems(f, "")
                    else:
                        iterator = json.load(f).items()
                    # Stream key/value pairs so smoke tests can stop early on --limit_train.
                    for qid, record in iterator:
                        image_id_raw = str(record.get("imageId") or "").strip()
                        question = str(record.get("question") or "").strip()
                        answer = normalize_vqa_answer(str(record.get("answer") or "").strip())
                        type_info = dict(record.get("types") or {})
                        coarse_group = coarse_gqa_question_group(question, type_info)
                        if not image_id_raw or not question:
                            continue
                        if self.question_group and coarse_group != self.question_group:
                            continue
                        img_path = os.path.join(image_dir, f"{image_id_raw}.jpg")
                        if skip_missing_images and not os.path.isfile(img_path):
                            continue
                        image_id = int(image_id_raw) if image_id_raw.isdigit() else image_id_raw
                        question_id = int(str(qid)) if str(qid).isdigit() else len(items)
                        items.append(
                            {
                                "question_id": question_id,
                                "image_id": image_id,
                                "image_path": img_path,
                                "question": question,
                                "answer": answer,
                                "all_answers_raw": [answer] if answer else [],
                                "all_answers": [answer] if answer else [],
                                "metadata": {
                                    "split": f"gqa_{split}",
                                    "question_type": str(type_info.get("detailed") or heuristic_question_category(question)),
                                    "answer_type": heuristic_answer_type(answer),
                                    "source_dataset": "gqa",
                                    "heuristic_question_type": heuristic_question_category(question),
                                    "heuristic_answer_type": heuristic_answer_type(answer),
                                    "gqa_structural_type": str(type_info.get("structural", "")),
                                    "gqa_semantic_type": str(type_info.get("semantic", "")),
                                    "gqa_detailed_type": str(type_info.get("detailed", "")),
                                    "gqa_question_group": coarse_group,
                                },
                            }
                        )
                        if limit > 0 and len(items) >= int(limit):
                            break
                if limit > 0 and len(items) >= int(limit):
                    break

        if not items:
            raise FileNotFoundError("No usable GQA train items found.")
        self.items = items

    def set_image_corruption(self, mode: str = "none", *, seed: int = 0) -> None:
        mode = str(mode or "none")
        if mode not in ("none", "zero", "shuffle", "random_swap"):
            raise ValueError(
                f"Unsupported image_corruption_mode={mode}. Supported: none, zero, shuffle, random_swap"
            )
        self.image_corruption_mode = mode
        self._corruption_indices = None
        n = len(self.items)
        if n <= 0 or mode in ("none", "zero"):
            return
        if mode == "shuffle":
            if n == 1:
                self._corruption_indices = [0]
            else:
                self._corruption_indices = [((i + 1) % n) for i in range(n)]
            return
        idxs = list(range(n))
        if n == 1:
            self._corruption_indices = idxs
            return
        rng = random.Random(int(seed))
        rng.shuffle(idxs)
        for i in range(n):
            if idxs[i] == i:
                j = (i + 1) % n
                idxs[i], idxs[j] = idxs[j], idxs[i]
        self._corruption_indices = idxs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        source_item = item
        if self._corruption_indices is not None:
            source_item = self.items[self._corruption_indices[idx]]
        img = Image.open(source_item["image_path"]).convert("RGB")
        img_t = self.transform(img)
        if self.image_corruption_mode == "zero":
            img_t = torch.zeros_like(img_t)
        return {
            "image": img_t,
            "question": item["question"],
            "answer": item["answer"],
            "all_answers_raw": item.get("all_answers_raw", []),
            "all_answers": item["all_answers"],
            "question_id": item["question_id"],
            "image_id": item["image_id"],
            "metadata": item["metadata"],
        }


class MixedVQAv2Dataset(Dataset):
    """
    Deterministic percentage-based supervised QA source mixer.

    Supported source keys:
    - train / vqav2_train
    - val / vqav2_val
    - gqa_train
    """

    def __init__(
        self,
        images_root: str,
        annotations_root: str,
        gqa_root: str,
        mix: Dict[str, float],
        *,
        transform: Optional[transforms.Compose] = None,
        seed: int = 0,
        limit: int = 0,
        skip_missing_images: bool = True,
    ) -> None:
        super().__init__()
        self.images_root = images_root
        self.annotations_root = annotations_root
        self.transform = transform or build_image_transform(train_mode=True)
        self.image_corruption_mode = "none"
        self._corruption_indices: Optional[List[int]] = None
        self.source_counts: Dict[str, Tuple[int, int, int]] = {}

        items: List[dict] = []
        active_mix = {str(k): float(v) for k, v in (mix or {}).items() if float(v) > 0.0}
        total_mix_weight = sum(active_mix.values())
        split_aliases = {
            "train": "train",
            "vqav2_train": "train",
            "val": "val",
            "vqav2_val": "val",
        }
        for mix_key, pct in sorted(active_mix.items()):
            if pct <= 0.0:
                continue
            if pct > 100.0:
                raise ValueError(f"Percentage for '{mix_key}' is {pct}, must be <= 100")
            source_limit = 0
            if limit > 0 and total_mix_weight > 0.0:
                # Bound source dataset construction for large corpora like GQA.
                source_limit = max(1, int(math.ceil(float(limit) * (pct / total_mix_weight))))
            split = split_aliases.get(str(mix_key))
            if split is not None:
                ds = VQAv2Dataset(
                    images_root=images_root,
                    annotations_root=annotations_root,
                    split=split,
                    transform=self.transform,
                    limit=source_limit,
                    skip_missing_images=skip_missing_images,
                )
            elif str(mix_key) == "gqa_train":
                ds = GQADataset(
                    gqa_root=gqa_root,
                    split="train",
                    transform=self.transform,
                    limit=source_limit,
                    skip_missing_images=skip_missing_images,
                    question_group="",
                )
            else:
                raise ValueError(
                    f"Unsupported dataset_mix key={mix_key!r}. Supported: train, vqav2_train, val, vqav2_val, gqa_train"
                )
            total = len(ds.items)
            rng = random.Random(f"{seed}_{mix_key}")
            indices = list(range(total))
            rng.shuffle(indices)
            keep = max(1, int(total * pct / 100.0))
            picked = [ds.items[i] for i in indices[:keep]]
            self.source_counts[str(mix_key)] = (total, len(picked), 0)
            items.extend(picked)

        if not items:
            raise FileNotFoundError("No usable VQAv2 items for dataset_mix.")
        if limit > 0:
            items = items[: int(limit)]
        self.items = items

    def set_image_corruption(self, mode: str = "none", *, seed: int = 0) -> None:
        mode = str(mode or "none")
        if mode not in ("none", "zero", "shuffle", "random_swap"):
            raise ValueError(
                f"Unsupported image_corruption_mode={mode}. Supported: none, zero, shuffle, random_swap"
            )
        self.image_corruption_mode = mode
        self._corruption_indices = None
        n = len(self.items)
        if n <= 0 or mode in ("none", "zero"):
            return
        if mode == "shuffle":
            if n == 1:
                self._corruption_indices = [0]
            else:
                self._corruption_indices = [((i + 1) % n) for i in range(n)]
            return
        idxs = list(range(n))
        if n == 1:
            self._corruption_indices = idxs
            return
        rng = random.Random(int(seed))
        rng.shuffle(idxs)
        for i in range(n):
            if idxs[i] == i:
                j = (i + 1) % n
                idxs[i], idxs[j] = idxs[j], idxs[i]
        self._corruption_indices = idxs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        source_item = item
        if self._corruption_indices is not None:
            source_item = self.items[self._corruption_indices[idx]]
        img = Image.open(source_item["image_path"]).convert("RGB")
        img_t = self.transform(img)
        if self.image_corruption_mode == "zero":
            img_t = torch.zeros_like(img_t)
        return {
            "image": img_t,
            "question": item["question"],
            "answer": item["answer"],
            "all_answers_raw": item.get("all_answers_raw", []),
            "all_answers": item["all_answers"],
            "question_id": item["question_id"],
            "image_id": item["image_id"],
            "metadata": item["metadata"],
        }


def _extract_pointing_source_name(record: Dict[str, Any]) -> str:
    for key in ("source_dataset", "dataset", "dataset_name", "source", "subset", "mixture_name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = record.get("metadata")
    if isinstance(meta, dict):
        for key in ("source_dataset", "dataset", "dataset_name", "source", "subset"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "pointing"


def _extract_pointing_question(record: Dict[str, Any]) -> str:
    for key in ("question", "prompt", "text", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    qa = record.get("qa")
    if isinstance(qa, dict):
        for key in ("question", "prompt", "text"):
            value = qa.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_pointing_answer(record: Dict[str, Any]) -> Tuple[str, List[str], bool]:
    has_vqa_target = bool(record.get("has_vqa_target", False))
    answer = ""
    raw_answers: List[str] = []
    if has_vqa_target:
        for key in ("answer", "canonical_answer"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                answer = normalize_vqa_answer(value)
                raw_answers = [str(value).strip()]
                break
        if not answer:
            qa = record.get("qa")
            if isinstance(qa, dict):
                value = qa.get("answer")
                if isinstance(value, str) and value.strip():
                    answer = normalize_vqa_answer(value)
                    raw_answers = [str(value).strip()]
        if not answer:
            answers = record.get("all_answers_raw") or record.get("answers") or []
            if isinstance(answers, list):
                raw_answers = [str(x).strip() for x in answers if str(x).strip()]
                answer = majority_answer(raw_answers)
    return answer, raw_answers, has_vqa_target


def _extract_pointing_soft_target(record: Dict[str, Any], target_len: int = 196) -> List[float]:
    value = None
    grid_targets = record.get("grid_targets")
    if isinstance(grid_targets, dict):
        value = grid_targets.get("soft_target")
    if value is None:
        value = record.get("soft_target")
    if value is None:
        return [0.0] * int(target_len)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("Expected pointing soft_target to be a JSON list.")
    out = [float(x) for x in value]
    if len(out) != int(target_len):
        raise ValueError(f"Expected pointing soft_target length {target_len}, got {len(out)}")
    return out


def _extract_pointing_bbox(record: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("bbox_xyxy", "bbox", "box_xyxy"):
        value = record.get(key)
        if isinstance(value, list) and len(value) == 4:
            return [float(x) for x in value]
    grid_targets = record.get("grid_targets")
    if isinstance(grid_targets, dict):
        for key in ("bbox_xyxy", "bbox"):
            value = grid_targets.get(key)
            if isinstance(value, list) and len(value) == 4:
                return [float(x) for x in value]
    return None


def _resolve_pointing_image_path(record: Dict[str, Any], *, images_root: str, index_dir: str) -> str:
    candidates: List[str] = []
    for key in ("image_path", "image_file", "relative_image_path", "path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    image = record.get("image")
    if isinstance(image, dict):
        for key in ("path", "image_path", "relative_path"):
            value = image.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    for cand in candidates:
        if os.path.isabs(cand) and os.path.isfile(cand):
            return cand
        rel_candidates = [
            os.path.join(index_dir, cand),
            os.path.join(images_root, cand),
        ]
        for rel in rel_candidates:
            if os.path.isfile(rel):
                return rel
    raise FileNotFoundError(f"Could not resolve pointing image path from record keys under {index_dir} / {images_root}")


class PointingIndexDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        *,
        images_root: str,
        transform: Optional[transforms.Compose] = None,
        limit: int = 0,
        skip_missing_images: bool = True,
        target_len: int = 196,
    ) -> None:
        super().__init__()
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing pointing index: {index_path}")
        self.index_path = index_path
        self.images_root = images_root
        self.transform = transform or build_image_transform(train_mode=True)
        self.image_corruption_mode = "none"
        self._corruption_indices: Optional[List[int]] = None
        self.target_len = int(target_len)
        self.items: List[Dict[str, Any]] = []
        self.source_to_indices: Dict[str, List[int]] = {}
        self.recommended_sampling_weights: Dict[str, float] = {}

        mix_cfg_path = os.path.join(os.path.dirname(os.path.abspath(index_path)), "mix_config.json")
        if os.path.isfile(mix_cfg_path):
            with open(mix_cfg_path, "r", encoding="utf-8") as f:
                mix_cfg = json.load(f)
            weights = mix_cfg.get("recommended_sampling_weights", {})
            if isinstance(weights, dict):
                self.recommended_sampling_weights = {
                    str(k): float(v) for k, v in weights.items() if float(v) > 0.0
                }

        index_dir = os.path.dirname(os.path.abspath(index_path))
        with open(index_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if limit > 0 and len(self.items) >= int(limit):
                    break
                row = line.strip()
                if not row:
                    continue
                record = json.loads(row)
                question = _extract_pointing_question(record)
                if not question:
                    continue
                try:
                    image_path = _resolve_pointing_image_path(record, images_root=images_root, index_dir=index_dir)
                except FileNotFoundError:
                    if skip_missing_images:
                        continue
                    raise
                answer, raw_answers, has_vqa_target = _extract_pointing_answer(record)
                source_name = _extract_pointing_source_name(record)
                soft_target = _extract_pointing_soft_target(record, target_len=self.target_len)
                bbox = _extract_pointing_bbox(record)
                image_info = record.get("image")
                width = None
                height = None
                if isinstance(image_info, dict):
                    width = image_info.get("width")
                    height = image_info.get("height")
                qid_raw = record.get("question_id", record.get("id", line_idx))
                qid = int(qid_raw) if str(qid_raw).isdigit() else int(line_idx)
                image_id_raw = record.get("image_id", record.get("image", {}).get("id", line_idx))
                image_id = int(image_id_raw) if str(image_id_raw).isdigit() else int(line_idx)
                item = {
                    "question_id": qid,
                    "image_id": image_id,
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "all_answers_raw": raw_answers,
                    "all_answers": [normalize_vqa_answer(x) for x in raw_answers] if raw_answers else ([] if not answer else [answer]),
                    "grounding_soft_target": soft_target,
                    "has_grounding_target": True,
                    "bbox_xyxy": bbox,
                    "image_width": None if width is None else float(width),
                    "image_height": None if height is None else float(height),
                    "metadata": {
                        "split": str(record.get("split", "pointing_train")),
                        "question_type": heuristic_question_category(question),
                        "answer_type": heuristic_answer_type(answer),
                        "source_dataset": source_name,
                        "has_vqa_target": bool(has_vqa_target),
                        "heuristic_question_type": heuristic_question_category(question),
                        "heuristic_answer_type": heuristic_answer_type(answer),
                    },
                }
                self.source_to_indices.setdefault(source_name, []).append(len(self.items))
                self.items.append(item)

        if not self.items:
            raise FileNotFoundError("No usable pointing items found.")

    def set_image_corruption(self, mode: str = "none", *, seed: int = 0) -> None:
        mode = str(mode or "none")
        if mode not in ("none", "zero", "shuffle", "random_swap"):
            raise ValueError(
                f"Unsupported image_corruption_mode={mode}. Supported: none, zero, shuffle, random_swap"
            )
        self.image_corruption_mode = mode
        self._corruption_indices = None
        n = len(self.items)
        if n <= 0 or mode in ("none", "zero"):
            return
        if mode == "shuffle":
            self._corruption_indices = [((i + 1) % n) if n > 1 else 0 for i in range(n)]
            return
        idxs = list(range(n))
        if n == 1:
            self._corruption_indices = idxs
            return
        rng = random.Random(int(seed))
        rng.shuffle(idxs)
        for i in range(n):
            if idxs[i] == i:
                j = (i + 1) % n
                idxs[i], idxs[j] = idxs[j], idxs[i]
        self._corruption_indices = idxs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        source_item = item if self._corruption_indices is None else self.items[self._corruption_indices[idx]]
        img = Image.open(source_item["image_path"]).convert("RGB")
        img_t = self.transform(img)
        if self.image_corruption_mode == "zero":
            img_t = torch.zeros_like(img_t)
        return {
            "image": img_t,
            "question": item["question"],
            "answer": item["answer"],
            "all_answers_raw": item.get("all_answers_raw", []),
            "all_answers": item["all_answers"],
            "question_id": item["question_id"],
            "image_id": item["image_id"],
            "grounding_soft_target": item["grounding_soft_target"],
            "has_grounding_target": bool(item.get("has_grounding_target", False)),
            "bbox_xyxy": item.get("bbox_xyxy"),
            "image_width": item.get("image_width"),
            "image_height": item.get("image_height"),
            "metadata": item["metadata"],
        }


class GroundingMixBatchSampler(BatchSampler):
    def __init__(
        self,
        *,
        base_dataset: Dataset,
        pointing_dataset: PointingIndexDataset,
        batch_size: int,
        pointing_mix_ratio: float,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.pointing_dataset = pointing_dataset
        self.batch_size = max(1, int(batch_size))
        self.pointing_mix_ratio = max(0.0, min(1.0, float(pointing_mix_ratio)))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 1
        self.pointing_batch = max(1, int(round(float(self.batch_size) * self.pointing_mix_ratio)))
        self.base_batch = max(0, self.batch_size - self.pointing_batch)
        if self.base_batch <= 0:
            raise ValueError("pointing_mix_ratio leaves no room for VQA samples in the batch.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))

    def __len__(self) -> int:
        n_base = len(self.base_dataset)
        if self.drop_last:
            return max(1, n_base // self.base_batch)
        return max(1, int(math.ceil(float(n_base) / float(self.base_batch))))

    def _make_source_cycles(self) -> Tuple[List[str], List[float], Dict[str, deque[int]]]:
        rng = random.Random(f"{self.seed}_{self.epoch}_pointing")
        source_cycles: Dict[str, deque[int]] = {}
        for source, idxs in self.pointing_dataset.source_to_indices.items():
            shuffled = list(idxs)
            rng.shuffle(shuffled)
            source_cycles[source] = deque(shuffled)
        weights = []
        sources = []
        for source in sorted(source_cycles.keys()):
            w = float(self.pointing_dataset.recommended_sampling_weights.get(source, 1.0))
            if w <= 0.0:
                continue
            sources.append(source)
            weights.append(w)
        if not sources:
            sources = sorted(source_cycles.keys())
            weights = [1.0 for _ in sources]
        return sources, weights, source_cycles

    def __iter__(self) -> Iterable[List[int]]:
        base_rng = random.Random(f"{self.seed}_{self.epoch}_base")
        base_indices = list(range(len(self.base_dataset)))
        base_rng.shuffle(base_indices)
        base_pos = 0

        point_sources, point_weights, source_cycles = self._make_source_cycles()
        point_rng = random.Random(f"{self.seed}_{self.epoch}_pointpick")

        def next_point_index() -> int:
            source = point_rng.choices(point_sources, weights=point_weights, k=1)[0]
            bucket = source_cycles[source]
            if not bucket:
                refill = list(self.pointing_dataset.source_to_indices[source])
                point_rng.shuffle(refill)
                bucket.extend(refill)
            return int(bucket.popleft())

        offset = len(self.pointing_dataset)
        num_batches = len(self)
        for _ in range(num_batches):
            if base_pos + self.base_batch > len(base_indices):
                if self.drop_last:
                    break
                base_rng.shuffle(base_indices)
                base_pos = 0
            batch: List[int] = []
            for _ in range(self.pointing_batch):
                batch.append(next_point_index())
            batch.extend(offset + int(idx) for idx in base_indices[base_pos : base_pos + self.base_batch])
            base_pos += self.base_batch
            point_rng.shuffle(batch)
            yield batch
