from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from models.bpe_tokenizer import ByteBPETokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ImageTextExample:
    pair_id: str
    dataset_name: str
    pair_split: str
    image_id: str
    local_path: str
    text: str
    token_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


def _select_mixed_pairs(
    db_path: str,
    mix: Dict[str, float],
    *,
    seed: int = 0,
    max_pairs: int = 0,
) -> tuple[List[ImageTextExample], Dict[str, tuple[int, int, int]]]:
    import duckdb

    source_counts: Dict[str, tuple[int, int, int]] = {}
    items: List[ImageTextExample] = []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for mix_key, pct in sorted(mix.items()):
            pct = float(pct)
            if pct <= 0:
                continue
            if pct > 100:
                raise ValueError(f"Percentage for '{mix_key}' is {pct}, must be <= 100")

            if ":" in mix_key:
                dataset_name, pair_split = mix_key.split(":", 1)
                rows = con.execute(
                    """
                    select pair_id, dataset_name, pair_split, image_id, local_path, text
                    from valid_image_text_pairs
                    where dataset_name = ? and pair_split = ?
                    order by pair_id
                    """,
                    [dataset_name, pair_split],
                ).fetchall()
            else:
                dataset_name = mix_key
                rows = con.execute(
                    """
                    select pair_id, dataset_name, pair_split, image_id, local_path, text
                    from valid_image_text_pairs
                    where dataset_name = ?
                    order by pair_id
                    """,
                    [dataset_name],
                ).fetchall()

            if not rows:
                available = [
                    f"{row[0]}:{row[1]}"
                    for row in con.execute(
                        "select distinct dataset_name, pair_split from image_text_pairs order by dataset_name, pair_split"
                    ).fetchall()
                ]
                raise ValueError(
                    f"No valid image-text pairs for '{mix_key}' in {db_path}. "
                    f"Available: {available}"
                )

            total = len(rows)
            rng = random.Random(f"{seed}_{mix_key}")
            indices = list(range(total))
            rng.shuffle(indices)
            keep = max(1, int(total * pct / 100.0))

            kept = 0
            missing = 0
            for idx in indices[:keep]:
                pair_id, ds_name, split_name, image_id, local_path, text = rows[idx]
                if not local_path or not os.path.isfile(local_path):
                    missing += 1
                    continue
                items.append(
                    ImageTextExample(
                        pair_id=str(pair_id),
                        dataset_name=str(ds_name),
                        pair_split=str(split_name),
                        image_id=str(image_id),
                        local_path=str(local_path),
                        text=str(text),
                    )
                )
                kept += 1
            source_counts[mix_key] = (total, kept, missing)
    finally:
        con.close()

    if int(max_pairs) > 0:
        items = items[: int(max_pairs)]
    if not items:
        raise FileNotFoundError(f"No usable image-text pairs for dataset mix from {db_path}")
    return items, source_counts


def build_siglip_pair_transform(
    image_size: int = 224,
    *,
    resize_scale: tuple[float, float] = (0.7, 1.0),
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                int(image_size),
                scale=(float(resize_scale[0]), float(resize_scale[1])),
                interpolation=Image.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.1,
                        hue=0.05,
                    )
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _pretokenize_items(
    items: List[ImageTextExample],
    *,
    tokenizer_path: str,
    max_text_len: int,
    batch_size: int = 512,
) -> None:
    if not items:
        return
    tokenizer = ByteBPETokenizer.load(str(tokenizer_path))
    chunk = max(1, int(batch_size))
    for start in range(0, len(items), chunk):
        batch = items[start : start + chunk]
        texts = [item.text for item in batch]
        token_ids, attention_mask = tokenizer(
            texts,
            max_len=int(max_text_len),
            return_attention_mask=True,
        )
        token_ids = token_ids.to(dtype=torch.int32)
        attention_mask = attention_mask.to(dtype=torch.bool)
        for idx, item in enumerate(batch):
            item.token_ids = token_ids[idx].clone()
            item.attention_mask = attention_mask[idx].clone()


class MixedImageTextDataset(Dataset):
    """DuckDB-backed image-text dataset mixed by per-dataset percentages.

    Mix keys use `dataset_name` values from `image_text_pairs`, optionally
    qualified by split as `dataset_name:pair_split`.
    """

    def __init__(
        self,
        db_path: str,
        mix: Dict[str, float],
        transform: transforms.Compose,
        *,
        seed: int = 0,
        max_pairs: int = 0,
        pretokenize_text: bool = False,
        tokenizer_path: str = "",
        max_text_len: int = 64,
    ) -> None:
        self.transform = transform
        self.items, self.source_counts = _select_mixed_pairs(
            str(db_path),
            mix,
            seed=int(seed),
            max_pairs=int(max_pairs),
        )
        if bool(pretokenize_text):
            if not str(tokenizer_path).strip():
                raise ValueError("tokenizer_path is required when pretokenize_text is enabled")
            _pretokenize_items(
                self.items,
                tokenizer_path=str(tokenizer_path),
                max_text_len=int(max_text_len),
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.items[int(idx)]
        image = Image.open(item.local_path).convert("RGB")
        return {
            "pair_id": item.pair_id,
            "dataset_name": item.dataset_name,
            "pair_split": item.pair_split,
            "image_id": item.image_id,
            "image": self.transform(image),
            "text": item.text,
            "token_ids": item.token_ids,
            "attention_mask": item.attention_mask,
            "local_path": item.local_path,
        }


class MixedImageTextCrossDataset(Dataset):
    def __init__(
        self,
        db_path: str,
        mix: Dict[str, float],
        siglip_transform,
        dino_transform,
        *,
        seed: int = 0,
        max_pairs: int = 0,
        pretokenize_text: bool = False,
        tokenizer_path: str = "",
        max_text_len: int = 64,
    ) -> None:
        self.siglip_transform = siglip_transform
        self.dino_transform = dino_transform
        self.items, self.source_counts = _select_mixed_pairs(
            str(db_path),
            mix,
            seed=int(seed),
            max_pairs=int(max_pairs),
        )
        if bool(pretokenize_text):
            if not str(tokenizer_path).strip():
                raise ValueError("tokenizer_path is required when pretokenize_text is enabled")
            _pretokenize_items(
                self.items,
                tokenizer_path=str(tokenizer_path),
                max_text_len=int(max_text_len),
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.items[int(idx)]
        image = Image.open(item.local_path).convert("RGB")
        dino_views = self.dino_transform(image)
        if len(dino_views) < 2:
            raise ValueError("DINO cross-stage transform must return at least 2 global views")
        return {
            "pair_id": item.pair_id,
            "dataset_name": item.dataset_name,
            "pair_split": item.pair_split,
            "image_id": item.image_id,
            "siglip_image": self.siglip_transform(image),
            "dino_global_1": dino_views[0],
            "dino_global_2": dino_views[1],
            "text": item.text,
            "token_ids": item.token_ids,
            "attention_mask": item.attention_mask,
            "local_path": item.local_path,
        }


def image_text_collate(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not samples:
        return {
            "pair_id": [],
            "dataset_name": [],
            "pair_split": [],
            "image_id": [],
            "image": torch.empty(0),
            "text": [],
            "token_ids": None,
            "attention_mask": None,
            "local_path": [],
        }
    token_ids = None
    attention_mask = None
    if samples[0].get("token_ids") is not None and samples[0].get("attention_mask") is not None:
        token_ids = torch.stack([sample["token_ids"] for sample in samples], dim=0)
        attention_mask = torch.stack([sample["attention_mask"] for sample in samples], dim=0)
    return {
        "pair_id": [str(sample["pair_id"]) for sample in samples],
        "dataset_name": [str(sample["dataset_name"]) for sample in samples],
        "pair_split": [str(sample["pair_split"]) for sample in samples],
        "image_id": [str(sample["image_id"]) for sample in samples],
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "text": [str(sample["text"]) for sample in samples],
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "local_path": [str(sample["local_path"]) for sample in samples],
    }


def image_text_cross_collate(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not samples:
        return {
            "pair_id": [],
            "dataset_name": [],
            "pair_split": [],
            "image_id": [],
            "siglip_image": torch.empty(0),
            "dino_global_1": torch.empty(0),
            "dino_global_2": torch.empty(0),
            "text": [],
            "token_ids": None,
            "attention_mask": None,
            "local_path": [],
        }
    token_ids = None
    attention_mask = None
    if samples[0].get("token_ids") is not None and samples[0].get("attention_mask") is not None:
        token_ids = torch.stack([sample["token_ids"] for sample in samples], dim=0)
        attention_mask = torch.stack([sample["attention_mask"] for sample in samples], dim=0)
    return {
        "pair_id": [str(sample["pair_id"]) for sample in samples],
        "dataset_name": [str(sample["dataset_name"]) for sample in samples],
        "pair_split": [str(sample["pair_split"]) for sample in samples],
        "image_id": [str(sample["image_id"]) for sample in samples],
        "siglip_image": torch.stack([sample["siglip_image"] for sample in samples], dim=0),
        "dino_global_1": torch.stack([sample["dino_global_1"] for sample in samples], dim=0),
        "dino_global_2": torch.stack([sample["dino_global_2"] for sample in samples], dim=0),
        "text": [str(sample["text"]) for sample in samples],
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "local_path": [str(sample["local_path"]) for sample in samples],
    }
