"""
COCO 2014 caption dataset for bridge caption-align pre-training.

Pairs each COCO training image with one randomly sampled caption per epoch.
Returns (image_tensor, caption_text) for cosine alignment with the LM.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


_COLOR_MEAN = (0.485, 0.456, 0.406)
_COLOR_STD = (0.229, 0.224, 0.225)

_CAPTION_ANN_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
_CAPTION_FILENAME = "captions_train2014.json"


def _build_caption_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(3 / 4, 4 / 3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(_COLOR_MEAN), std=list(_COLOR_STD)),
        ]
    )


def ensure_caption_annotations(annotations_root: str) -> str:
    """Download COCO caption annotations if not present. Returns path to JSON."""
    caption_path = os.path.join(annotations_root, _CAPTION_FILENAME)
    if os.path.isfile(caption_path):
        return caption_path

    import zipfile
    from urllib import request as urlrequest

    dl_dir = os.path.join(annotations_root, "_downloads")
    os.makedirs(dl_dir, exist_ok=True)
    zip_path = os.path.join(dl_dir, "annotations_trainval2014.zip")

    if not os.path.isfile(zip_path):
        print(f"[caption_data] Downloading COCO caption annotations to {zip_path} ...")
        urlrequest.urlretrieve(_CAPTION_ANN_URL, zip_path)

    target_member = f"annotations/{_CAPTION_FILENAME}"
    with zipfile.ZipFile(zip_path, "r") as zf:
        if target_member in zf.namelist():
            with zf.open(target_member) as src, open(caption_path, "wb") as dst:
                dst.write(src.read())
        else:
            raise FileNotFoundError(
                f"{target_member} not found in {zip_path}. "
                f"Available: {zf.namelist()[:10]}"
            )

    print(f"[caption_data] Extracted to {caption_path}")
    return caption_path


class COCOCaptionDataset(Dataset):
    """COCO 2014 caption dataset.  One random caption per image per __getitem__."""

    def __init__(
        self,
        images_root: str,
        annotations_root: str,
        *,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.images_root = images_root
        self.transform = transform or _build_caption_transform()

        caption_path = ensure_caption_annotations(annotations_root)
        with open(caption_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build image_id -> [captions] map.
        img_caps: Dict[int, List[str]] = defaultdict(list)
        for ann in data.get("annotations", []):
            img_id = int(ann["image_id"])
            cap = str(ann.get("caption", "")).strip()
            if cap:
                img_caps[img_id].append(cap)

        # Build items — one entry per image that exists on disk.
        self.items: List[Tuple[str, List[str]]] = []
        train_dir = os.path.join(images_root, "train2014")
        for img_id, caps in sorted(img_caps.items()):
            fname = f"COCO_train2014_{img_id:012d}.jpg"
            fpath = os.path.join(train_dir, fname)
            if os.path.isfile(fpath):
                self.items.append((fpath, caps))

        print(
            f"[caption_data] Loaded {len(self.items)} images "
            f"with {sum(len(c) for _, c in self.items)} total captions"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        fpath, caps = self.items[idx]
        img = Image.open(fpath).convert("RGB")
        img_t = self.transform(img)
        caption = random.choice(caps)
        return {"image": img_t, "caption": caption}
