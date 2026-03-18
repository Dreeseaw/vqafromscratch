from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.vit_ssl import DINOStudentTeacher, ViTSSLConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return str(requested)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model_from_checkpoint(path: str, device: str) -> DINOStudentTeacher:
    payload = load_checkpoint(path)
    train_args = payload.get("train_args", {})
    cfg = ViTSSLConfig(
        image_size=int(train_args.get("image_size", 224)),
        patch_size=int(train_args.get("patch_size", 16)),
        dim=int(train_args.get("dim", 192)),
        depth=int(train_args.get("depth", 12)),
        heads=int(train_args.get("heads", 3)),
        mlp_ratio=float(train_args.get("mlp_ratio", 4.0)),
        dropout=float(train_args.get("dropout", 0.0)),
        attn_dropout=float(train_args.get("attn_dropout", 0.0)),
        drop_path=float(train_args.get("drop_path", 0.05)),
    )
    model = DINOStudentTeacher(
        cfg,
        out_dim=int(train_args.get("out_dim", 4096)),
        head_hidden_dim=int(train_args.get("head_hidden_dim", 1024)),
        head_bottleneck_dim=int(train_args.get("head_bottleneck_dim", 256)),
        head_layers=int(train_args.get("head_layers", 3)),
    ).to(device)
    state = payload.get("teacher") or payload.get("student")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size) + 32),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataset(name: str, root: str, train: bool, image_size: int):
    tfm = make_transforms(int(image_size))
    name = str(name).lower()
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=bool(train), download=True, transform=tfm)
    if name == "stl10":
        split = "train" if bool(train) else "test"
        return datasets.STL10(root=root, split=split, download=True, transform=tfm)
    if name == "imagefolder":
        split_dir = "train" if bool(train) else "val"
        return datasets.ImageFolder(root=os.path.join(root, split_dir), transform=tfm)
    raise ValueError(f"Unsupported dataset={name}. Use cifar10, stl10, or imagefolder.")


@torch.no_grad()
def extract_features(
    model: DINOStudentTeacher,
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        pooled = model.forward_pooled(images)
        pooled = F.normalize(pooled.float(), dim=-1)
        feats.append(pooled.cpu())
        labels.append(target.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def knn_predict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    *,
    k: int,
    temperature: float,
    num_classes: int,
) -> torch.Tensor:
    sims = test_features @ train_features.t()
    topk_sims, topk_idx = sims.topk(k=min(int(k), int(train_features.shape[0])), dim=1, largest=True, sorted=True)
    topk_labels = train_labels[topk_idx]
    weights = (topk_sims / float(temperature)).exp()
    votes = torch.zeros(int(test_features.shape[0]), int(num_classes), dtype=torch.float32)
    votes.scatter_add_(1, topk_labels, weights)
    return votes.argmax(dim=1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Frozen-feature kNN eval for SSL ViT checkpoints.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "stl10", "imagefolder"])
    ap.add_argument("--data_root", type=str, default="data/ssl_eval")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_test", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(str(args.device))
    model = build_model_from_checkpoint(str(args.checkpoint), device)
    image_size = int(model.backbone.cfg.image_size)

    train_ds = build_dataset(str(args.dataset), str(args.data_root), True, image_size)
    test_ds = build_dataset(str(args.dataset), str(args.data_root), False, image_size)
    if int(args.limit_train) > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(len(train_ds), int(args.limit_train)))))
    if int(args.limit_test) > 0:
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(len(test_ds), int(args.limit_test)))))

    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "drop_last": False,
    }
    train_loader = DataLoader(train_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    num_classes = int(train_labels.max().item()) + 1
    preds = knn_predict(
        train_features,
        train_labels.long(),
        test_features,
        k=int(args.k),
        temperature=float(args.temperature),
        num_classes=num_classes,
    )
    acc = (preds == test_labels.long()).float().mean().item()
    print(
        f"[ssl_knn] checkpoint={os.path.abspath(args.checkpoint)} "
        f"dataset={args.dataset} k={args.k} temp={args.temperature} "
        f"train={len(train_ds)} test={len(test_ds)} acc={acc:.4f}"
    )


if __name__ == "__main__":
    main()
