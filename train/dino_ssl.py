from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import tarfile
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageFilter, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.vit_ssl import DINOStudentTeacher, ViTSSLConfig


LOGDIR = "logs"


class Logger:
    def __init__(self, run_id: str, checkpoint_id: Optional[int]) -> None:
        self.run_id = run_id
        self.ckpt = checkpoint_id
        self.base = os.path.join(LOGDIR, run_id)
        os.makedirs(self.base, exist_ok=True)
        if checkpoint_id is None:
            name = "logfile.txt"
        else:
            name = f"logfile_from_{checkpoint_id}.txt"
        self.path = os.path.join(self.base, name)

    def log(self, text: str) -> None:
        print(text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class EpochShuffleSampler(torch.utils.data.Sampler[int]):
    def __init__(self, data_source: Sequence[Any], *, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 1
        self._skip_first_n = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))

    def skip_first_n(self, n: int) -> None:
        """Skip the first n indices on the next iteration (for mid-epoch resume)."""
        self._skip_first_n = max(0, int(n))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(int(self.seed) + int(self.epoch))
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        skip = self._skip_first_n
        self._skip_first_n = 0
        if skip > 0:
            indices = indices[skip:]
        yield from indices

    def __len__(self) -> int:
        return len(self.data_source)


def _seed_loader_worker(worker_id: int) -> None:
    del worker_id
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    torch.manual_seed(seed)


class RecursiveImageDataset(Dataset):
    def __init__(self, root: str, transform: transforms.Compose, *, max_images: int = 0) -> None:
        self.root = os.path.abspath(str(root))
        self.transform = transform
        items: List[str] = []
        for cur, _dirs, files in os.walk(self.root):
            for name in sorted(files):
                if name.lower().endswith(IMAGE_EXTS):
                    items.append(os.path.join(cur, name))
        if int(max_images) > 0:
            items = items[: int(max_images)]
        if not items:
            raise FileNotFoundError(f"No images found under {self.root}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        path = self.items[int(idx)]
        image = Image.open(path).convert("RGB")
        return self.transform(image)


def _extract_tar_subset(
    archive_path: str,
    members_needed: set[str],
    staging_dir: str,
) -> Dict[str, str]:
    """Extract a subset of members from a tar(.gz) archive to staging_dir.

    Returns mapping from archive member name to extracted filesystem path.
    Skips members already extracted (cached).
    """
    os.makedirs(staging_dir, exist_ok=True)
    result: Dict[str, str] = {}
    still_needed: set[str] = set()
    for m in members_needed:
        dest = os.path.join(staging_dir, m)
        if os.path.isfile(dest):
            result[m] = dest
        else:
            still_needed.add(m)
    if not still_needed:
        return result

    mode = "r:gz" if archive_path.endswith(".gz") else "r:"
    with tarfile.open(archive_path, mode) as tf:
        for member in tf:
            if member.name in still_needed and member.isfile():
                dest = os.path.join(staging_dir, member.name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with tf.extractfile(member) as src:
                    with open(dest, "wb") as dst:
                        dst.write(src.read())
                result[member.name] = dest
                still_needed.discard(member.name)
                if not still_needed:
                    break
    return result


class MixedImageDataset(Dataset):
    """DuckDB-backed dataset that mixes sources by percentage.

    Keys in `mix` are source_name values from the DuckDB images table,
    optionally qualified with a split as "source_name:split" (e.g. "coco_local:train2014").
    Values are the percentage (0-100) of each source to use.
    Supports filesystem paths and tar.gz archive paths (auto-extracted to staging).
    """

    def __init__(
        self,
        db_path: str,
        mix: Dict[str, float],
        transform: transforms.Compose,
        *,
        seed: int = 0,
        max_images: int = 0,
        staging_dir: str = "data/vm_ssl/staged/extracted",
    ) -> None:
        import duckdb

        self.transform = transform
        items: List[str] = []
        self.source_counts: Dict[str, tuple[int, int, int]] = {}  # (total, selected, skipped_archive)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            for mix_key, pct in sorted(mix.items()):
                pct = float(pct)
                if pct <= 0:
                    continue
                if pct > 100:
                    raise ValueError(f"Percentage for '{mix_key}' is {pct}, must be <= 100")

                # Parse optional "source_name:split" syntax
                if ":" in mix_key:
                    source_name, source_split = mix_key.split(":", 1)
                    rows = con.execute(
                        "SELECT local_path FROM valid_images "
                        "WHERE source_name = ? AND source_split = ? ORDER BY image_id",
                        [source_name, source_split],
                    ).fetchall()
                else:
                    source_name = mix_key
                    rows = con.execute(
                        "SELECT local_path FROM valid_images WHERE source_name = ? ORDER BY image_id",
                        [source_name],
                    ).fetchall()

                if not rows:
                    available = [
                        f"{r[0]}:{r[1]}" for r in con.execute(
                            "SELECT DISTINCT source_name, source_split FROM images "
                            "ORDER BY source_name, source_split"
                        ).fetchall()
                    ]
                    raise ValueError(
                        f"No valid images for '{mix_key}' in {db_path}. "
                        f"Available: {available}"
                    )

                all_paths = [r[0] for r in rows]
                total = len(all_paths)

                # Deterministic subsample (per-key seed so adding sources is stable)
                rng = random.Random(f"{seed}_{mix_key}")
                indices = list(range(total))
                rng.shuffle(indices)
                keep = max(1, int(total * pct / 100.0))
                selected_paths = [all_paths[i] for i in indices[:keep]]

                # Resolve paths: filesystem vs archive
                fs_paths: List[str] = []
                archive_members: Dict[str, List[str]] = {}  # archive_path -> [member_name, ...]
                for p in selected_paths:
                    if "::" in p:
                        archive, member = p.split("::", 1)
                        archive_members.setdefault(archive, []).append(member)
                    else:
                        fs_paths.append(p)

                # Extract archive members to staging
                skipped = 0
                for archive, members in archive_members.items():
                    src_staging = os.path.join(staging_dir, source_name)
                    extracted = _extract_tar_subset(archive, set(members), src_staging)
                    for m in members:
                        if m in extracted:
                            fs_paths.append(extracted[m])
                        else:
                            skipped += 1

                self.source_counts[mix_key] = (total, len(fs_paths), skipped)
                items.extend(fs_paths)
        finally:
            con.close()

        if int(max_images) > 0:
            items = items[: int(max_images)]
        if not items:
            raise FileNotFoundError(f"No usable images for dataset mix from {db_path}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        path = self.items[int(idx)]
        image = Image.open(path).convert("RGB")
        return self.transform(image)


class GaussianBlur:
    def __init__(self, p: float, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.p = float(p)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        radius = random.uniform(self.radius_min, self.radius_max)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class Solarization:
    def __init__(self, p: float) -> None:
        self.p = float(p)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        return ImageOps.solarize(image)


class DINOMultiCropTransform:
    def __init__(
        self,
        image_size: int = 224,
        local_crop_size: int = 96,
        global_crops_scale: tuple[float, float] = (0.4, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 6,
    ) -> None:
        flip_color = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.global_crop1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(int(image_size), scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_color,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        self.global_crop2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(int(image_size), scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_color,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        self.local_crop = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    int(local_crop_size),
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_color,
                GaussianBlur(0.5),
                normalize,
            ]
        )
        self.local_crops_number = int(local_crops_number)

    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        crops = [self.global_crop1(image), self.global_crop2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_crop(image))
        return crops


def multiview_collate(samples: Sequence[List[torch.Tensor]]) -> List[torch.Tensor]:
    if not samples:
        return []
    num_views = len(samples[0])
    out: List[torch.Tensor] = []
    for view_idx in range(num_views):
        out.append(torch.stack([sample[view_idx] for sample in samples], dim=0))
    return out


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_epoch: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> List[float]:
    warmup_iters = int(warmup_epochs) * int(niter_per_epoch)
    total_iters = int(epochs) * int(niter_per_epoch)
    if total_iters <= 0:
        return []
    schedule: List[float] = []
    if warmup_iters > 0:
        for i in range(warmup_iters):
            alpha = float(i) / float(max(1, warmup_iters))
            schedule.append(float(start_warmup_value) + alpha * (float(base_value) - float(start_warmup_value)))
    remain = total_iters - len(schedule)
    for i in range(max(0, remain)):
        cosine = 0.5 * (1.0 + math.cos(math.pi * i / max(1, remain)))
        schedule.append(float(final_value) + cosine * (float(base_value) - float(final_value)))
    return schedule[:total_iters]


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float,
        center_momentum: float,
    ) -> None:
        super().__init__()
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, int(out_dim)))

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center * (1.0 - self.center_momentum))

    def forward(
        self,
        student_outputs: Sequence[torch.Tensor],
        teacher_outputs: Sequence[torch.Tensor],
        teacher_temp: float,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        student_log_probs = [F.log_softmax(x / self.student_temp, dim=-1) for x in student_outputs]
        teacher_probs = [F.softmax((x - self.center) / float(teacher_temp), dim=-1).detach() for x in teacher_outputs]

        total_loss = 0.0
        num_terms = 0
        for iq, tq in enumerate(teacher_probs):
            for iv, sv in enumerate(student_log_probs):
                if iv == iq:
                    continue
                total_loss = total_loss + torch.sum(-tq * sv, dim=-1).mean()
                num_terms += 1
        if num_terms <= 0:
            raise RuntimeError("DINO loss received no valid student/teacher view pairs.")
        loss = total_loss / float(num_terms)

        with torch.no_grad():
            teacher_cat = torch.cat(list(teacher_outputs), dim=0)
            teacher_prob_cat = torch.cat(list(teacher_probs), dim=0)
            entropy = -(teacher_prob_cat * teacher_prob_cat.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
            center_norm = self.center.norm().item()
            self.update_center(teacher_cat)
        stats = {
            "teacher_entropy": float(entropy),
            "center_norm": float(center_norm),
            "teacher_temp": float(teacher_temp),
        }
        return loss, stats


def dino_param_groups(model: nn.Module) -> List[Dict[str, Any]]:
    regularized: List[nn.Parameter] = []
    no_wd: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_wd.append(param)
        else:
            regularized.append(param)
    return [
        {"params": regularized, "weight_decay": 0.0},
        {"params": no_wd, "weight_decay": 0.0},
    ]


def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(float(momentum)).add_(s_param.data * (1.0 - float(momentum)))


def freeze_teacher(teacher: nn.Module) -> None:
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)


def feature_std(x: torch.Tensor) -> float:
    if x.ndim != 2:
        x = x.reshape(int(x.shape[0]), -1)
    return float(x.float().std(dim=0, unbiased=False).mean().item())


def save_checkpoint(
    path: str,
    *,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    dino_loss: DINOLoss,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (None if scaler is None else scaler.state_dict()),
        "dino_loss": dino_loss.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "train_args": vars(args),
        "rng_state": capture_rng_state(),
        "model_config": {
            "image_size": int(args.image_size),
            "patch_size": int(args.patch_size),
            "dim": int(args.dim),
            "depth": int(args.depth),
            "heads": int(args.heads),
            "mlp_ratio": float(args.mlp_ratio),
            "dropout": float(args.dropout),
            "attn_dropout": float(args.attn_dropout),
            "drop_path": float(args.drop_path),
        },
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def latest_ckpt_path(run_id: str, step: Optional[int]) -> Optional[str]:
    run_dir = os.path.join(LOGDIR, str(run_id))
    if step is not None:
        path = os.path.join(run_dir, f"step_{int(step)}.tar")
        return path if os.path.isfile(path) else None
    if not os.path.isdir(run_dir):
        return None
    best_step = -1
    best_path: Optional[str] = None
    for name in os.listdir(run_dir):
        if not (name.startswith("step_") and name.endswith(".tar")):
            continue
        try:
            cur_step = int(name[len("step_") : -len(".tar")])
        except Exception:
            continue
        if cur_step > best_step:
            best_step = cur_step
            best_path = os.path.join(run_dir, name)
    return best_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a small DINO-style SSL ViT.")
    ap.add_argument("run_id", type=str)
    ap.add_argument("checkpoint", nargs="?", type=int)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--data_dir", type=str, default="images/train2014")
    ap.add_argument(
        "--dataset_mix",
        type=str,
        default="",
        help='JSON dict mapping source_name (from DuckDB) to %% of each to use, '
             'e.g. \'{"coco_local": 80, "textocr": 100, "inat2021": 10}\'',
    )
    ap.add_argument("--db_path", type=str, default="data/vm_ssl/db/vm_ssl.duckdb")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=35)

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--attn_dropout", type=float, default=0.0)
    ap.add_argument("--drop_path", type=float, default=0.05)

    ap.add_argument("--out_dim", type=int, default=4096)
    ap.add_argument("--head_hidden_dim", type=int, default=1024)
    ap.add_argument("--head_bottleneck_dim", type=int, default=256)
    ap.add_argument("--head_layers", type=int, default=3)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--local_crops_number", type=int, default=6)
    ap.add_argument("--local_crop_size", type=int, default=96)
    ap.add_argument("--global_crops_scale", type=float, nargs=2, default=(0.4, 1.0))
    ap.add_argument("--local_crops_scale", type=float, nargs=2, default=(0.05, 0.4))

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--weight_decay", type=float, default=0.04)
    ap.add_argument("--weight_decay_end", type=float, default=0.4)
    ap.add_argument("--teacher_momentum", type=float, default=0.996)
    ap.add_argument("--teacher_temp", type=float, default=0.04)
    ap.add_argument("--teacher_temp_warmup", type=float, default=0.04)
    ap.add_argument("--teacher_temp_warmup_epochs", type=int, default=0)
    ap.add_argument("--student_temp", type=float, default=0.1)
    ap.add_argument("--center_momentum", type=float, default=0.9)
    ap.add_argument("--grad_clip", type=float, default=3.0)
    ap.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--save_every_epoch", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def build_models(args: argparse.Namespace, device: str) -> tuple[DINOStudentTeacher, DINOStudentTeacher]:
    cfg = ViTSSLConfig(
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        dim=int(args.dim),
        depth=int(args.depth),
        heads=int(args.heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attn_dropout=float(args.attn_dropout),
        drop_path=float(args.drop_path),
    )
    student = DINOStudentTeacher(
        cfg,
        out_dim=int(args.out_dim),
        head_hidden_dim=int(args.head_hidden_dim),
        head_bottleneck_dim=int(args.head_bottleneck_dim),
        head_layers=int(args.head_layers),
    ).to(device)
    teacher = DINOStudentTeacher(
        cfg,
        out_dim=int(args.out_dim),
        head_hidden_dim=int(args.head_hidden_dim),
        head_bottleneck_dim=int(args.head_bottleneck_dim),
        head_layers=int(args.head_layers),
    ).to(device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    freeze_teacher(teacher)
    return student, teacher


def parse_json_object_arg(name: str, raw: object) -> dict[str, object]:
    text = str(raw or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a single valid JSON object string; got {text!r}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object, got: {text!r}")
    return value


def run_training(args: argparse.Namespace) -> None:
    device = resolve_device(str(args.device))
    set_seed(int(args.seed))

    run_dir = os.path.join(LOGDIR, str(args.run_id))
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(str(args.run_id), checkpoint_id=args.checkpoint)

    transform = DINOMultiCropTransform(
        image_size=int(args.image_size),
        local_crop_size=int(args.local_crop_size),
        global_crops_scale=tuple(float(x) for x in args.global_crops_scale),
        local_crops_scale=tuple(float(x) for x in args.local_crops_scale),
        local_crops_number=int(args.local_crops_number),
    )
    if args.dataset_mix:
        mix = parse_json_object_arg("--dataset_mix", args.dataset_mix)
        dataset = MixedImageDataset(
            str(args.db_path), mix, transform, seed=int(args.seed), max_images=int(args.max_images),
        )
        logger.log(f"[dino] === effective dataset: {len(dataset)} images ===")
        for src, (total, kept, skipped) in sorted(dataset.source_counts.items()):
            pct_of_source = 100 * kept / total if total else 0
            pct_of_dataset = 100 * kept / len(dataset) if len(dataset) else 0
            line = (
                f"[dino]   {src:30s}  {kept:>7d} / {total:>7d} "
                f"({pct_of_source:5.1f}% of source, {pct_of_dataset:5.1f}% of dataset)"
            )
            if skipped:
                line += f"  skipped_archive={skipped}"
            logger.log(line)
        logger.log(f"[dino] {'':30s}  {len(dataset):>7d} total")
    else:
        dataset = RecursiveImageDataset(str(args.data_dir), transform, max_images=int(args.max_images))
        logger.log(f"[dino] === effective dataset: {len(dataset)} images from {args.data_dir} ===")
    sampler = EpochShuffleSampler(dataset, seed=int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        collate_fn=multiview_collate,
        pin_memory=bool(args.pin_memory),
        drop_last=True,
        persistent_workers=bool(int(args.num_workers) > 0),
        prefetch_factor=(None if int(args.num_workers) <= 0 else max(1, int(args.prefetch_factor))),
        worker_init_fn=(_seed_loader_worker if int(args.num_workers) > 0 else None),
    )

    student, teacher = build_models(args, device)
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    logger.log(
        f"[dino] device={device} data_dir={os.path.abspath(args.data_dir)} "
        f"images={len(dataset)} batch_size={args.batch_size} epochs={args.epochs}"
    )
    logger.log(f"[dino] student params trainable={trainable_params:,} total={total_params:,}")
    logger.log(f"[dino] backbone_cfg={asdict(student.backbone.cfg)} out_dim={args.out_dim}")

    optimizer = torch.optim.AdamW(dino_param_groups(student), lr=float(args.lr), betas=(0.9, 0.999))
    use_amp = str(args.precision) in ("bf16", "fp16") and device == "cuda"
    amp_dtype = torch.bfloat16 if str(args.precision) == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and str(args.precision) == "fp16"))
    dino_loss = DINOLoss(
        int(args.out_dim),
        student_temp=float(args.student_temp),
        center_momentum=float(args.center_momentum),
    ).to(device)

    start_epoch = 1
    global_step = 0
    batch_in_epoch = 0
    ckpt_path = latest_ckpt_path(str(args.run_id), args.checkpoint)
    if ckpt_path is not None:
        payload = load_checkpoint(ckpt_path, map_location="cpu")
        student.load_state_dict(payload["student"], strict=True)
        teacher.load_state_dict(payload["teacher"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if payload.get("dino_loss") is not None:
            dino_loss.load_state_dict(payload["dino_loss"])
        restore_rng_state(payload.get("rng_state"))
        global_step = int(payload.get("global_step", 0))
        start_epoch = int(payload.get("epoch", 1))
        batch_in_epoch = int(payload.get("batch_in_epoch", 0))
        logger.log(
            f"[dino] resume checkpoint={ckpt_path} global_step={global_step} "
            f"epoch={start_epoch} batch_in_epoch={batch_in_epoch}"
        )

    iters_per_epoch = len(loader)
    total_steps = int(args.epochs) * int(iters_per_epoch)
    lr_schedule = cosine_scheduler(
        float(args.lr),
        float(args.lr_min),
        int(args.epochs),
        int(iters_per_epoch),
        warmup_epochs=int(args.warmup_epochs),
        start_warmup_value=0.0,
    )
    wd_schedule = cosine_scheduler(
        float(args.weight_decay),
        float(args.weight_decay_end),
        int(args.epochs),
        int(iters_per_epoch),
    )
    momentum_schedule = cosine_scheduler(
        float(args.teacher_momentum),
        1.0,
        int(args.epochs),
        int(iters_per_epoch),
    )
    temp_schedule = cosine_scheduler(
        float(args.teacher_temp_warmup),
        float(args.teacher_temp),
        max(1, int(args.teacher_temp_warmup_epochs)),
        int(iters_per_epoch),
    )
    if int(args.teacher_temp_warmup_epochs) <= 0:
        temp_schedule = []

    wall_start = time.time()
    student.train()
    for epoch in range(int(start_epoch), int(args.epochs) + 1):
        sampler.set_epoch(epoch)
        bidx_offset = 0
        if epoch == int(start_epoch) and int(batch_in_epoch) > 0:
            skip = int(batch_in_epoch) * int(args.batch_size)
            sampler.skip_first_n(skip)
            bidx_offset = int(batch_in_epoch)
            logger.log(f"[dino] resuming epoch={epoch} skipping {batch_in_epoch} batches via sampler")
        if epoch != int(start_epoch):
            batch_in_epoch = 0

        for bidx, crops in enumerate(loader):
            if global_step >= total_steps:
                break

            step_idx = int(global_step)
            lr = lr_schedule[min(step_idx, len(lr_schedule) - 1)]
            wd = wd_schedule[min(step_idx, len(wd_schedule) - 1)]
            momentum = momentum_schedule[min(step_idx, len(momentum_schedule) - 1)]
            teacher_temp = (
                temp_schedule[step_idx]
                if step_idx < len(temp_schedule)
                else float(args.teacher_temp)
            )
            for group_idx, group in enumerate(optimizer.param_groups):
                group["lr"] = float(lr)
                group["weight_decay"] = float(wd) if group_idx == 0 else 0.0

            crops = [crop.to(device, non_blocking=True) for crop in crops]
            global_crops = crops[:2]

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    teacher_outputs = [teacher(crop) for crop in global_crops]

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                student_outputs = [student(crop) for crop in crops]
                loss, loss_stats = dino_loss(
                    [out["logits"] for out in student_outputs],
                    [out["logits"] for out in teacher_outputs],
                    teacher_temp=float(teacher_temp),
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            update_teacher(student, teacher, momentum=float(momentum))

            global_step += 1
            batch_in_epoch = bidx + bidx_offset + 1

            with torch.no_grad():
                student_std = feature_std(torch.cat([x["pooled"] for x in student_outputs], dim=0))
                teacher_std = feature_std(torch.cat([x["pooled"] for x in teacher_outputs], dim=0))
                steps_per_s = float(global_step) / max(1e-6, (time.time() - wall_start))
            if global_step == 1 or (global_step % int(args.log_every) == 0):
                logger.log(
                    f"[dino] step={global_step}/{total_steps} epoch={epoch}/{args.epochs} "
                    f"loss={float(loss.item()):.4f} lr={float(lr):.6g} wd={float(wd):.6g} "
                    f"teacher_m={float(momentum):.6f} teacher_t={float(teacher_temp):.4f} "
                    f"student_std={student_std:.4f} teacher_std={teacher_std:.4f} "
                    f"teacher_entropy={float(loss_stats['teacher_entropy']):.4f} "
                    f"center_norm={float(loss_stats['center_norm']):.4f} "
                    f"steps_per_s={steps_per_s:.2f}"
                )

            if int(args.ckpt_every) > 0 and global_step % int(args.ckpt_every) == 0:
                path = os.path.join(run_dir, f"step_{global_step}.tar")
                save_checkpoint(
                    path,
                    student=student,
                    teacher=teacher,
                    optimizer=optimizer,
                    scaler=scaler,
                    dino_loss=dino_loss,
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    args=args,
                )
                logger.log(f"[dino] checkpoint saved: {path}")

        batch_in_epoch = 0
        if bool(args.save_every_epoch):
            path = os.path.join(run_dir, f"epoch_{epoch}.tar")
            save_checkpoint(
                path,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scaler=scaler,
                dino_loss=dino_loss,
                global_step=global_step,
                epoch=epoch + 1,
                batch_in_epoch=0,
                args=args,
            )
            logger.log(f"[dino] epoch checkpoint saved: {path}")
        if global_step >= total_steps:
            break

    final_path = os.path.join(run_dir, f"step_{global_step}.tar")
    save_checkpoint(
        final_path,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scaler=scaler,
        dino_loss=dino_loss,
        global_step=global_step,
        epoch=int(args.epochs) + 1,
        batch_in_epoch=0,
        args=args,
    )
    logger.log(f"[dino] done global_step={global_step} checkpoint={final_path}")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
