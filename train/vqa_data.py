"""
VQAv2 dataset helpers.

Role:
- Download/prepare official VQAv2 + COCO image assets.
- Provide a thin dataset wrapper with explicit multimodal boundaries:
  image tensor + question text + canonical answer target + eval metadata.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from urllib import request as urlrequest

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_t = self.transform(img)
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
