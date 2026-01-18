"""
probe.py

Linear probe on VAE latent z (flattened [B,16,7,7] -> [B,784]) using COCO-style annotations.

Key changes vs prior version:
- Uses your Logger from train.py (same folder import).
- Adds CLI arg to choose label mode: --label_mode {multilabel,singlelabel}
  - multilabel: multi-hot object presence (BCEWithLogits + F1 metric)
  - singlelabel: pick ONE category per image (most frequent category in that image) (CrossEntropy + top1 acc)

Example:
python3 probe.py --ckpt logs/my_run/step_4000.tar --label_mode multilabel
python3 probe.py --ckpt logs/my_run/step_4000.tar --label_mode singlelabel
"""

import os
import re
import json
import time
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import VariationalAutoEncoder as VAE, VAEConfig
from train import Logger

# match train.py defaults
DATA_DIR_DEFAULT = "/Users/williamdreese/percy/vqa/VQA/Images/mscoco/"
ANNO_DIR_DEFAULT = "/Users/williamdreese/percy/vqa/VQA/Annotations/"
COLOR_MEAN = (0.485, 0.456, 0.406)
COLOR_STD  = (0.229, 0.224, 0.225)


# -------------------------
# ckpt parsing
# -------------------------
def infer_run_and_step_from_ckpt(ckpt_path: str) -> Tuple[str, int]:
    m = re.search(r"logs[/\\]([^/\\]+)[/\\]step_(\d+)\.tar$", ckpt_path)
    if m:
        return m.group(1), int(m.group(2))
    run_id = os.path.basename(os.path.dirname(ckpt_path)) or "unknown_run"
    return run_id, -1


# -------------------------
# filename -> COCO image_id
# -------------------------
_COCO_ID_RE = re.compile(r"_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def coco_image_id_from_filename(fn: str) -> Optional[int]:
    m = _COCO_ID_RE.search(fn)
    if not m:
        return None
    return int(m.group(1))


# -------------------------
# COCO object task parsing
# -------------------------
@dataclass
class CocoObjTargets:
    num_classes: int
    cat_id_to_idx: Dict[int, int]
    image_id_to_multihot: Dict[int, torch.Tensor]
    image_id_to_single: Dict[int, int]  # class idx


def load_json(path: str) -> dict:
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        return json.load(f)

def build_coco_object_targets(ann: dict) -> CocoObjTargets:
    if "categories" not in ann or "annotations" not in ann:
        raise ValueError("Expected COCO-style JSON with 'categories' and 'annotations'.")

    cats = sorted(ann["categories"], key=lambda c: int(c["id"]))
    cat_id_to_idx = {int(c["id"]): i for i, c in enumerate(cats)}
    C = len(cats)

    image_to_cat_counts: Dict[int, Counter] = defaultdict(Counter)
    for a in ann["annotations"]:
        if "image_id" not in a or "category_id" not in a:
            continue
        image_to_cat_counts[int(a["image_id"])][int(a["category_id"])] += 1

    image_id_to_multihot: Dict[int, torch.Tensor] = {}
    image_id_to_single: Dict[int, int] = {}

    for image_id, cnt in image_to_cat_counts.items():
        y = torch.zeros(C, dtype=torch.float32)
        for cid in cnt.keys():
            if cid in cat_id_to_idx:
                y[cat_id_to_idx[cid]] = 1.0
        image_id_to_multihot[image_id] = y

        # single-label = most frequent category for that image
        cid_major, _ = cnt.most_common(1)[0]
        if cid_major in cat_id_to_idx:
            image_id_to_single[image_id] = cat_id_to_idx[cid_major]

    return CocoObjTargets(
        num_classes=C,
        cat_id_to_idx=cat_id_to_idx,
        image_id_to_multihot=image_id_to_multihot,
        image_id_to_single=image_id_to_single,
    )


# -------------------------
# Dataset
# -------------------------
class CocoProbeDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        targets: CocoObjTargets,
        label_mode: str,  # "multilabel" or "singlelabel"
        rrc: bool = True,
        flip: bool = True,
        limit: Optional[int] = None,
    ):
        self.image_dir = os.path.expanduser(image_dir)
        files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        # keep only those with labels
        kept = []
        for f in files:
            image_id = coco_image_id_from_filename(f)
            if image_id is None:
                continue
            if label_mode == "multilabel":
                if image_id in targets.image_id_to_multihot:
                    kept.append(f)
            else:
                if image_id in targets.image_id_to_single:
                    kept.append(f)

        if limit is not None and limit > 0:
            kept = kept[: int(limit)]

        self.files = kept
        self.targets = targets
        self.label_mode = label_mode

        # transforms (same “feel” as train.py)
        if rrc:
            trans = [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(3 / 4, 4 / 3)),
            ]
        else:
            trans = [transforms.Resize((224, 224))]

        if flip:
            trans.append(transforms.RandomHorizontalFlip(p=0.5))

        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=list(COLOR_MEAN), std=list(COLOR_STD)),
        ])
        self.transform = transforms.Compose(trans)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        image_id = coco_image_id_from_filename(fn)
        assert image_id is not None

        path = os.path.join(self.image_dir, fn)
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        if self.label_mode == "multilabel":
            y = self.targets.image_id_to_multihot[image_id]
            return x, y
        else:
            y = self.targets.image_id_to_single[image_id]
            return x, torch.tensor(y, dtype=torch.long)


# -------------------------
# Probe + z extraction
# -------------------------
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        return self.fc(z_flat)

@torch.no_grad()
def vae_encode_z(vae: VAE, x: torch.Tensor, use_mu: bool) -> torch.Tensor:
    features = vae._encoder(x)
    mu, lv = vae._post_head(features)
    lv = torch.clamp(lv, min=-10.0, max=1.0)
    if use_mu:
        return mu
    std = torch.exp(lv / 2.0)
    return mu + torch.randn_like(std) * std


# -------------------------
# Metrics
# -------------------------
def multilabel_micro_f1(logits: torch.Tensor, y: torch.Tensor, thresh: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs >= thresh).to(y.dtype)
    tp = (pred * y).sum().item()
    fp = (pred * (1 - y)).sum().item()
    fn = ((1 - pred) * y).sum().item()
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0

def top1_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="logs/<run_id>/step_<N>.tar")
    ap.add_argument("--train_image_dir", type=str, default=os.path.join(DATA_DIR_DEFAULT, "train2014/train2014/"))
    ap.add_argument("--val_image_dir", type=str, default=os.path.join(DATA_DIR_DEFAULT, "val2014/val2014/"))
    ap.add_argument("--train_ann_json", type=str, default=os.path.join(ANNO_DIR_DEFAULT, "annotations/instances_train2014.json"))
    ap.add_argument("--val_ann_json", type=str, default=os.path.join(ANNO_DIR_DEFAULT, "annotations/instances_val2014.json"))

    ap.add_argument("--label_mode", type=str, choices=["multilabel", "singlelabel"], default="multilabel",
                    help="multilabel => BCE + micro-F1, singlelabel => CE + top1 acc")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)

    ap.add_argument("--use_mu", action="store_true", help="Probe on mu instead of sampled z.")
    ap.add_argument("--no_aug", action="store_true", help="Disable train-time RRC/flip.")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--interop_threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)

    ckpt_path = os.path.expanduser(args.ckpt)
    run_id, ckpt_step = infer_run_and_step_from_ckpt(ckpt_path)
    checkpoint_id_for_logger = ckpt_step if ckpt_step >= 0 else 0

    logger = Logger(run_id, checkpoint_id_for_logger, probe=True)
    logger.log(f"\n[probe] ckpt: {ckpt_path}\n")
    logger.log(f"[probe] label_mode: {args.label_mode} | use_mu: {args.use_mu}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"[probe] device: {device}\n")

    # load targets
    train_ann = load_json(args.train_ann_json)
    val_ann = load_json(args.val_ann_json)
    train_targets = build_coco_object_targets(train_ann)
    val_targets = build_coco_object_targets(val_ann)

    C = train_targets.num_classes
    logger.log(f"[probe] num_classes: {C}\n")

    use_aug = not args.no_aug
    train_ds = CocoProbeDataset(
        image_dir=args.train_image_dir,
        targets=train_targets,
        label_mode=args.label_mode,
        rrc=use_aug,
        flip=use_aug,
        limit=(args.limit_train if args.limit_train > 0 else None),
    )
    val_ds = CocoProbeDataset(
        image_dir=args.val_image_dir,
        targets=val_targets,
        label_mode=args.label_mode,
        rrc=False,
        flip=False,
        limit=(args.limit_val if args.limit_val > 0 else None),
    )
    logger.log(f"[probe] train samples: {len(train_ds)} | val samples: {len(val_ds)}\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=1 if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=1 if args.num_workers > 0 else None,
        drop_last=False,
    )

    # load frozen VAE
    vae = VAE(VAEConfig()).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # infer z_dim
    x0, _ = next(iter(train_loader))
    x0 = x0.to(device)
    with torch.no_grad():
        z0 = vae_encode_z(vae, x0, use_mu=args.use_mu).flatten(1)
    z_dim = int(z0.shape[1])
    logger.log(f"[probe] z_dim(flat): {z_dim}\n")

    probe = LinearProbe(z_dim, C).to(device)

    if args.label_mode == "multilabel":
        loss_fn = nn.BCEWithLogitsLoss()
        metric_name = "microF1"
    else:
        loss_fn = nn.CrossEntropyLoss()
        metric_name = "top1"

    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    @torch.no_grad()
    def eval_epoch() -> Tuple[float, float]:
        probe.eval()
        total_loss, total_metric, n = 0.0, 0.0, 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            z = vae_encode_z(vae, x, use_mu=args.use_mu).flatten(1)
            logits = probe(z)

            loss = loss_fn(logits, y) if args.label_mode == "multilabel" else loss_fn(logits, y)
            metric = multilabel_micro_f1(logits, y) if args.label_mode == "multilabel" else top1_acc(logits, y)

            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_metric += metric * bs
            n += bs
        return total_loss / max(1, n), total_metric / max(1, n)

    best_val = -1.0
    start = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        probe.train()
        total_loss, total_metric, n = 0.0, 0.0, 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                z = vae_encode_z(vae, x, use_mu=args.use_mu).flatten(1)

            logits = probe(z)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            metric = multilabel_micro_f1(logits.detach(), y) if args.label_mode == "multilabel" else top1_acc(logits.detach(), y)

            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_metric += metric * bs
            n += bs
            global_step += 1

            if global_step % 50 == 1:
                logger.log(f"Step: {global_step}, Loss: {loss.detach():.4f}")

        tr_loss = total_loss / max(1, n)
        tr_metric = total_metric / max(1, n)
        va_loss, va_metric = eval_epoch()
        elapsed = time.time() - start

        logger.log(
            f"[probe] epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} {metric_name} {tr_metric:.4f} | "
            f"val loss {va_loss:.4f} {metric_name} {va_metric:.4f} | "
            f"elapsed {elapsed:.1f}s\n"
        )

        if va_metric > best_val:
            best_val = va_metric
            out_path = f"logs/{run_id}/probe_step_{ckpt_step if ckpt_step>=0 else 'unknown'}.tar"
            torch.save(
                {
                    "label_mode": args.label_mode,
                    "metric": metric_name,
                    "z_dim": z_dim,
                    "num_classes": C,
                    "use_mu": args.use_mu,
                    "probe_state_dict": probe.state_dict(),
                    "probe_opt_state_dict": opt.state_dict(),
                    "best_val_metric": best_val,
                    "vae_ckpt_path": ckpt_path,
                },
                out_path,
            )
            logger.log(f"[probe] saved best probe -> {out_path} (best val {metric_name}={best_val:.4f})\n")

    logger.log("[probe] done\n")


if __name__ == "__main__":
    main()
