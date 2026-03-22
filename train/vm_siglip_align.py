from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.bpe_tokenizer import ByteBPETokenizer
from models.vm_text_encoder import VMTextEncoder, VMTextEncoderConfig
from models.vit_ssl import DINOStudentTeacher, ViTSSLConfig
from train.dino_ssl import (
    DINOLoss,
    DINOMultiCropTransform,
    Logger,
    EpochShuffleSampler,
    _seed_loader_worker,
    capture_rng_state,
    latest_ckpt_path,
    resolve_device,
    restore_rng_state,
    set_seed,
    update_teacher,
)
from train.vm_recipe_data import (
    MixedImageTextCrossDataset,
    MixedImageTextDataset,
    build_siglip_pair_transform,
    image_text_collate,
    image_text_cross_collate,
)


LOGDIR = "logs"


class SiglipAlignHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 0,
        logit_scale_init: float = math.log(10.0),
        logit_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(in_dim))
        hidden_dim = int(hidden_dim)
        if hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(int(in_dim), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, int(out_dim)),
            )
        else:
            self.proj = nn.Linear(int(in_dim), int(out_dim))
        self.logit_scale = nn.Parameter(torch.tensor([float(logit_scale_init)], dtype=torch.float32))
        self.logit_bias = nn.Parameter(torch.tensor([float(logit_bias_init)], dtype=torch.float32))

    def forward(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(pooled_tokens)
        x = self.proj(x)
        return F.normalize(x.float(), dim=-1)


class TrainableTextTower(nn.Module):
    def __init__(
        self,
        tokenizer_path: str,
        *,
        device: str,
        max_text_len: int,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
        use_rope: bool,
    ) -> None:
        super().__init__()
        self.tokenizer_path = os.path.abspath(str(tokenizer_path))
        self.device_name = str(device)
        self.max_text_len = int(max_text_len)
        self.tokenizer = ByteBPETokenizer.load(self.tokenizer_path)
        self.encoder = VMTextEncoder(
            VMTextEncoderConfig(
                vocab_size=int(self.tokenizer.vocab_size),
                max_seq_len=int(max_text_len),
                dim=int(dim),
                layers=int(layers),
                heads=int(heads),
                mlp_ratio=int(mlp_ratio),
                dropout=float(dropout),
                rope=bool(use_rope),
            )
        ).to(device)

    @property
    def embedding_dim(self) -> int:
        return int(self.encoder.cfg.dim)

    @property
    def config_dict(self) -> Dict[str, object]:
        return self.encoder.get_config()

    def encode(self, texts: list[str]) -> torch.Tensor:
        input_ids, attention_mask = self.tokenizer(texts, max_len=int(self.max_text_len), return_attention_mask=True)
        return self.encode_pretokenized(input_ids, attention_mask)

    def encode_pretokenized(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device_name, dtype=torch.long, non_blocking=True)
        attention_mask = attention_mask.to(self.device_name, dtype=torch.bool, non_blocking=True)
        return self.encoder(input_ids, attention_mask=attention_mask)["pooled"]


def build_models_from_checkpoint(
    payload: Dict[str, Any],
    *,
    device: str,
    align_hidden_dim: int,
    text_dim: int,
    logit_scale_init: float,
    logit_bias_init: float,
) -> tuple[DINOStudentTeacher, DINOStudentTeacher, SiglipAlignHead]:
    cfg_data = dict(payload.get("model_config") or {})
    if not cfg_data:
        train_args = dict(payload.get("train_args") or {})
        cfg_data = {
            "image_size": int(train_args.get("image_size", 224)),
            "patch_size": int(train_args.get("patch_size", 16)),
            "dim": int(train_args.get("dim", 192)),
            "depth": int(train_args.get("depth", 12)),
            "heads": int(train_args.get("heads", 3)),
            "mlp_ratio": float(train_args.get("mlp_ratio", 4.0)),
            "dropout": float(train_args.get("dropout", 0.0)),
            "attn_dropout": float(train_args.get("attn_dropout", 0.0)),
            "drop_path": float(train_args.get("drop_path", 0.05)),
        }
    cfg = ViTSSLConfig(**cfg_data)

    base_args = dict(payload.get("train_args") or {})
    student = DINOStudentTeacher(
        cfg,
        out_dim=int(base_args.get("out_dim", 4096)),
        head_hidden_dim=int(base_args.get("head_hidden_dim", 1024)),
        head_bottleneck_dim=int(base_args.get("head_bottleneck_dim", 256)),
        head_layers=int(base_args.get("head_layers", 3)),
    ).to(device)
    teacher = DINOStudentTeacher(
        cfg,
        out_dim=int(base_args.get("out_dim", 4096)),
        head_hidden_dim=int(base_args.get("head_hidden_dim", 1024)),
        head_bottleneck_dim=int(base_args.get("head_bottleneck_dim", 256)),
        head_layers=int(base_args.get("head_layers", 3)),
    ).to(device)
    student.load_state_dict(payload["student"], strict=True)
    teacher.load_state_dict(payload["teacher"], strict=True)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()

    align_head = SiglipAlignHead(
        int(cfg.dim),
        int(text_dim),
        hidden_dim=int(align_hidden_dim),
        logit_scale_init=float(logit_scale_init),
        logit_bias_init=float(logit_bias_init),
    ).to(device)
    align_state = payload.get("siglip_align") if isinstance(payload.get("siglip_align"), dict) else {}
    if isinstance(align_state.get("image_proj"), dict):
        align_head.load_state_dict(align_state["image_proj"], strict=True)
    return student, teacher, align_head


def build_optimizer(
    args: argparse.Namespace,
    student: DINOStudentTeacher,
    align_head: SiglipAlignHead,
    text_tower: TrainableTextTower,
    *,
    device: str,
) -> tuple[torch.optim.Optimizer, str]:
    regularized = []
    no_wd = []
    named_params = list(student.backbone.named_parameters())
    named_params += [(f"align_head.{name}", param) for name, param in align_head.named_parameters()]
    named_params += [(f"text_encoder.{name}", param) for name, param in text_tower.encoder.named_parameters()]
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_wd.append(param)
        else:
            regularized.append(param)
    groups: list[dict[str, Any]] = []
    if regularized:
        groups.append({"params": regularized, "weight_decay": float(args.weight_decay)})
    if no_wd:
        groups.append({"params": no_wd, "weight_decay": 0.0})
    optim_kwargs: dict[str, Any] = {
        "lr": float(args.lr),
        "betas": (0.9, 0.999),
    }
    mode = "standard"
    if str(device) == "cuda":
        if bool(args.optimizer_fused):
            optim_kwargs["fused"] = True
            mode = "fused"
        elif bool(args.optimizer_foreach):
            optim_kwargs["foreach"] = True
            mode = "foreach"
    try:
        optimizer = torch.optim.AdamW(groups, **optim_kwargs)
    except TypeError:
        optim_kwargs.pop("fused", None)
        optim_kwargs.pop("foreach", None)
        mode = "standard"
        optimizer = torch.optim.AdamW(groups, **optim_kwargs)
    return optimizer, mode


def cosine_schedule(base_value: float, final_value: float, total_steps: int, warmup_steps: int = 0) -> list[float]:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps))
    values: list[float] = []
    if warmup_steps > 0:
        for idx in range(warmup_steps):
            alpha = float(idx + 1) / float(max(1, warmup_steps))
            values.append(alpha * float(base_value))
    remain = total_steps - len(values)
    for idx in range(remain):
        cosine = 0.5 * (1.0 + math.cos(math.pi * idx / max(1, remain)))
        values.append(float(final_value) + cosine * (float(base_value) - float(final_value)))
    return values[:total_steps]


def pairwise_sigmoid_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    logits = image_features @ text_features.t()
    logits = logits * logit_scale.exp() + logit_bias
    labels = torch.eye(int(logits.shape[0]), device=logits.device, dtype=logits.dtype)
    labels = labels * 2.0 - 1.0
    loss = -F.logsigmoid(labels * logits).mean()
    with torch.no_grad():
        pos_mask = torch.eye(int(logits.shape[0]), device=logits.device, dtype=torch.bool)
        pos_logits = logits[pos_mask]
        neg_logits = logits[~pos_mask]
        stats = {
            "logit_pos": float(pos_logits.mean().item()) if pos_logits.numel() > 0 else 0.0,
            "logit_neg": float(neg_logits.mean().item()) if neg_logits.numel() > 0 else 0.0,
            "image_std": float(image_features.std(dim=0, unbiased=False).mean().item()),
            "text_std": float(text_features.std(dim=0, unbiased=False).mean().item()),
        }
    return loss, stats


def parse_dino_weight_schedule(spec: str) -> list[tuple[float, float]]:
    text = str(spec or "").strip()
    if not text:
        return []
    points: list[tuple[float, float]] = []
    for chunk in text.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "@" in piece:
            weight_text, frac_text = piece.split("@", 1)
            weight = float(weight_text)
            frac = float(frac_text)
        else:
            weight = float(piece)
            frac = 0.0 if not points else 1.0
        frac = max(0.0, min(1.0, frac))
        points.append((frac, weight))
    if not points:
        return []
    points.sort(key=lambda item: item[0])
    return points


def parse_json_object_arg(name: str, raw: object) -> dict[str, object]:
    text = str(raw or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a single valid JSON object string; got {text!r}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object, got: {text!r}")
    return value


def current_dino_weight(schedule: list[tuple[float, float]], phase_progress: float) -> float:
    if not schedule:
        return 0.0
    progress = max(0.0, min(1.0, float(phase_progress)))
    weight = float(schedule[0][1])
    for frac, candidate in schedule:
        if progress >= float(frac):
            weight = float(candidate)
        else:
            break
    return weight


def save_checkpoint(
    path: str,
    *,
    student: DINOStudentTeacher,
    teacher: DINOStudentTeacher,
    align_head: SiglipAlignHead,
    text_tower: TrainableTextTower,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    phase_start_step: int,
    args: argparse.Namespace,
    base_payload: Dict[str, Any],
    dino_loss_state: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "train_args": vars(args),
        "rng_state": capture_rng_state(),
        "model_config": dict(base_payload.get("model_config") or {}),
        "siglip_align": {
            "image_proj": align_head.state_dict(),
            "text_encoder": text_tower.encoder.state_dict(),
            "text_encoder_config": text_tower.config_dict,
            "phase_start_step": int(phase_start_step),
            "phase_steps": int(args.phase_steps),
            "base_checkpoint": str(args.base_checkpoint or ""),
            "pair_mix": str(args.pair_mix),
            "text_tokenizer_path": str(args.text_tokenizer_path),
            "phase_name": str(args.phase_name),
            "dino_weight_schedule": str(args.dino_weight_schedule),
            "dino_loss": dino_loss_state,
        },
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Image-text recipe stage for DINO-pretrained VM runs.")
    ap.add_argument("run_id", type=str)
    ap.add_argument("checkpoint", nargs="?", type=int)
    ap.add_argument("--phase_name", type=str, default="align", choices=["cross", "align"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--db_path", type=str, default="data/vm_ssl/db/vm_ssl.duckdb")
    ap.add_argument(
        "--pair_mix",
        type=str,
        required=True,
        help='JSON dict mapping pair dataset names to percentages, e.g. \'{"coco_captions_2014:train2014":100,"coco_text_captions:train":100}\'',
    )
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])
    ap.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pretokenize_text", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--optimizer_foreach", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--optimizer_fused", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log_cuda_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--phase_steps", type=int, default=12000)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_min", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=3.0)
    ap.add_argument("--teacher_momentum", type=float, default=0.999)
    ap.add_argument("--teacher_temp", type=float, default=0.04)
    ap.add_argument("--student_temp", type=float, default=0.1)
    ap.add_argument("--center_momentum", type=float, default=0.9)
    ap.add_argument("--dino_weight_schedule", type=str, default="")
    ap.add_argument("--align_hidden_dim", type=int, default=0)
    ap.add_argument("--text_tokenizer_path", type=str, default="logs/mix_bpe_16k/tokenizer.pt")
    ap.add_argument("--max_text_len", type=int, default=64)
    ap.add_argument("--text_dim", type=int, default=384)
    ap.add_argument("--text_layers", type=int, default=8)
    ap.add_argument("--text_heads", type=int, default=6)
    ap.add_argument("--text_mlp_ratio", type=int, default=4)
    ap.add_argument("--text_dropout", type=float, default=0.1)
    ap.add_argument("--text_rope", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--logit_scale_init", type=float, default=math.log(10.0))
    ap.add_argument("--logit_bias_init", type=float, default=0.0)
    ap.add_argument("--base_checkpoint", type=str, default="")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--save_every_epoch", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def run_training(args: argparse.Namespace) -> None:
    device = resolve_device(str(args.device))
    set_seed(int(args.seed))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        tf32_mode = "tf32" if bool(args.allow_tf32) else "ieee"
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = tf32_mode
        elif hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = tf32_mode
        elif hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(args.matmul_precision))

    run_dir = os.path.join(LOGDIR, str(args.run_id))
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(str(args.run_id), checkpoint_id=args.checkpoint)

    pair_mix = parse_json_object_arg("--pair_mix", args.pair_mix)

    phase_name = str(args.phase_name)
    siglip_transform = build_siglip_pair_transform(int(args.image_size))
    if phase_name == "cross":
        dino_transform = DINOMultiCropTransform(
            image_size=int(args.image_size),
            local_crop_size=96,
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.05, 0.4),
            local_crops_number=0,
        )
        dataset = MixedImageTextCrossDataset(
            str(args.db_path),
            pair_mix,
            siglip_transform,
            dino_transform,
            seed=int(args.seed),
            max_pairs=int(args.max_pairs),
            pretokenize_text=bool(args.pretokenize_text),
            tokenizer_path=str(args.text_tokenizer_path),
            max_text_len=int(args.max_text_len),
        )
        collate_fn = image_text_cross_collate
    else:
        dataset = MixedImageTextDataset(
            str(args.db_path),
            pair_mix,
            siglip_transform,
            seed=int(args.seed),
            max_pairs=int(args.max_pairs),
            pretokenize_text=bool(args.pretokenize_text),
            tokenizer_path=str(args.text_tokenizer_path),
            max_text_len=int(args.max_text_len),
        )
        collate_fn = image_text_collate
    logger.log(f"[siglip_align] === effective pair dataset: {len(dataset)} pairs ===")
    for src, (total, kept, missing) in sorted(dataset.source_counts.items()):
        pct_of_source = 100 * kept / total if total else 0
        pct_of_dataset = 100 * kept / len(dataset) if len(dataset) else 0
        line = (
            f"[siglip_align]   {src:30s}  {kept:>7d} / {total:>7d} "
            f"({pct_of_source:5.1f}% of source, {pct_of_dataset:5.1f}% of dataset)"
        )
        if missing:
            line += f"  missing={missing}"
        logger.log(line)
    logger.log(f"[siglip_align] {'':30s}  {len(dataset):>7d} total")

    sampler = EpochShuffleSampler(dataset, seed=int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        pin_memory=bool(args.pin_memory),
        drop_last=True,
        persistent_workers=bool(int(args.num_workers) > 0),
        prefetch_factor=(None if int(args.num_workers) <= 0 else max(1, int(args.prefetch_factor))),
        worker_init_fn=(_seed_loader_worker if int(args.num_workers) > 0 else None),
    )

    ckpt_path = latest_ckpt_path(str(args.run_id), args.checkpoint)
    if ckpt_path is None and args.base_checkpoint:
        ckpt_path = os.path.abspath(str(args.base_checkpoint))
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found for run_id={args.run_id}. "
            f"Pass --base_checkpoint to start the align phase from a DINO checkpoint."
        )

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    text_tower = TrainableTextTower(
        str(args.text_tokenizer_path),
        device=device,
        max_text_len=int(args.max_text_len),
        dim=int(args.text_dim),
        layers=int(args.text_layers),
        heads=int(args.text_heads),
        mlp_ratio=int(args.text_mlp_ratio),
        dropout=float(args.text_dropout),
        use_rope=bool(args.text_rope),
    )
    student, teacher, align_head = build_models_from_checkpoint(
        payload,
        device=device,
        align_hidden_dim=int(args.align_hidden_dim),
        text_dim=int(text_tower.embedding_dim),
        logit_scale_init=float(args.logit_scale_init),
        logit_bias_init=float(args.logit_bias_init),
    )

    phase_state = payload.get("siglip_align") if isinstance(payload.get("siglip_align"), dict) else {}
    if isinstance(phase_state.get("text_encoder"), dict):
        text_tower.encoder.load_state_dict(phase_state["text_encoder"], strict=True)
    optimizer, optimizer_mode = build_optimizer(args, student, align_head, text_tower, device=device)
    dino_weight_schedule = parse_dino_weight_schedule(str(args.dino_weight_schedule))
    dino_loss_module: Optional[DINOLoss] = None
    if phase_name == "cross":
        dino_loss_module = DINOLoss(
            int(student.head.last_layer.out_features),
            student_temp=float(args.student_temp),
            center_momentum=float(args.center_momentum),
        ).to(device)
        if isinstance(phase_state.get("dino_loss"), dict):
            dino_loss_module.load_state_dict(phase_state["dino_loss"], strict=True)

    use_amp = str(args.precision) in ("bf16", "fp16") and device == "cuda"
    amp_dtype = torch.bfloat16 if str(args.precision) == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and str(args.precision) == "fp16"))

    saved_phase_name = str(phase_state.get("phase_name", "") or "")
    if saved_phase_name and saved_phase_name != phase_name:
        phase_start_step = int(payload.get("global_step", 0))
    else:
        phase_start_step = int(phase_state.get("phase_start_step", payload.get("global_step", 0)))
    if phase_state:
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload.get("rng_state"))

    global_step = int(payload.get("global_step", 0))
    start_epoch = int(payload.get("epoch", 1))
    batch_in_epoch = int(payload.get("batch_in_epoch", 0))
    logger.log(
        f"[siglip_align] resume checkpoint={ckpt_path} global_step={global_step} "
        f"epoch={start_epoch} batch_in_epoch={batch_in_epoch} phase_start_step={phase_start_step} phase_name={phase_name}"
    )

    student.train()
    teacher.eval()
    text_tower.train()
    if device == "cuda" and bool(args.channels_last):
        student.backbone = student.backbone.to(memory_format=torch.channels_last)
        teacher.backbone = teacher.backbone.to(memory_format=torch.channels_last)
    trainable_params = (
        sum(p.numel() for p in student.backbone.parameters() if p.requires_grad)
        + sum(p.numel() for p in align_head.parameters() if p.requires_grad)
        + sum(p.numel() for p in text_tower.encoder.parameters() if p.requires_grad)
    )
    total_params = (
        sum(p.numel() for p in student.parameters())
        + sum(p.numel() for p in align_head.parameters())
        + sum(p.numel() for p in text_tower.encoder.parameters())
    )
    logger.log(f"[siglip_align] trainable_params={trainable_params:,} total_params={total_params:,}")
    logger.log(
        f"[siglip_align] pair_mix={json.dumps(pair_mix, sort_keys=True)} phase_steps={args.phase_steps} "
        f"lr={args.lr} text_tokenizer_path={args.text_tokenizer_path} dino_weight_schedule={args.dino_weight_schedule} "
        f"pretokenize_text={args.pretokenize_text} optimizer_mode={optimizer_mode} channels_last={args.channels_last} "
        f"allow_tf32={args.allow_tf32} matmul_precision={args.matmul_precision}"
    )
    logger.log(
        f"[siglip_align] backbone_cfg={asdict(student.backbone.cfg)} align_hidden_dim={args.align_hidden_dim} "
        f"text_cfg={json.dumps(text_tower.config_dict, sort_keys=True)}"
    )

    completed_phase_steps = max(0, global_step - phase_start_step)
    if completed_phase_steps >= int(args.phase_steps):
        logger.log(
            f"[siglip_align] phase already complete: completed_phase_steps={completed_phase_steps} "
            f">= target={args.phase_steps}"
        )
        return

    phase_lr_schedule = cosine_schedule(float(args.lr), float(args.lr_min), int(args.phase_steps), int(args.warmup_steps))
    wall_start = time.time()
    last_log_wall = wall_start
    last_log_phase_step = completed_phase_steps
    grad_clip_params = list(student.backbone.parameters()) + list(align_head.parameters()) + list(text_tower.encoder.parameters())
    if device == "cuda" and bool(args.log_cuda_memory):
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(int(start_epoch), int(start_epoch) + 100000):
        sampler.set_epoch(epoch)
        bidx_offset = 0
        if epoch == int(start_epoch) and int(batch_in_epoch) > 0:
            skip = int(batch_in_epoch) * int(args.batch_size)
            sampler.skip_first_n(skip)
            bidx_offset = int(batch_in_epoch)
            logger.log(f"[siglip_align] resuming epoch={epoch} skipping {batch_in_epoch} batches via sampler")
        if epoch != int(start_epoch):
            batch_in_epoch = 0

        for bidx, batch in enumerate(loader):
            phase_step = max(0, global_step - phase_start_step)
            if phase_step >= int(args.phase_steps):
                break
            lr = phase_lr_schedule[min(phase_step, len(phase_lr_schedule) - 1)]
            for group in optimizer.param_groups:
                group["lr"] = float(lr)

            if phase_name == "cross":
                images = batch["siglip_image"].to(device, non_blocking=True)
                dino_global_1 = batch["dino_global_1"].to(device, non_blocking=True)
                dino_global_2 = batch["dino_global_2"].to(device, non_blocking=True)
                if device == "cuda" and bool(args.channels_last):
                    images = images.contiguous(memory_format=torch.channels_last)
                    dino_global_1 = dino_global_1.contiguous(memory_format=torch.channels_last)
                    dino_global_2 = dino_global_2.contiguous(memory_format=torch.channels_last)
            else:
                images = batch["image"].to(device, non_blocking=True)
                if device == "cuda" and bool(args.channels_last):
                    images = images.contiguous(memory_format=torch.channels_last)
            texts = list(batch["text"])
            token_ids = batch.get("token_ids")
            attention_mask = batch.get("attention_mask")
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()

            with amp_ctx:
                tokens = student.backbone.forward_tokens(images)
                pooled = tokens.mean(dim=1)
                image_features = align_head(pooled)
                if token_ids is not None and attention_mask is not None:
                    text_features = text_tower.encode_pretokenized(token_ids, attention_mask)
                else:
                    text_features = text_tower.encode(texts)
                siglip_loss, loss_stats = pairwise_sigmoid_loss(
                    image_features,
                    text_features,
                    logit_scale=align_head.logit_scale,
                    logit_bias=align_head.logit_bias,
                )
                dino_loss_value = torch.zeros((), device=device, dtype=siglip_loss.dtype)
                dino_weight = 0.0
                dino_stats = {"teacher_entropy": 0.0, "center_norm": 0.0}
                if phase_name == "cross":
                    assert dino_loss_module is not None
                    with torch.no_grad():
                        teacher_outputs = [teacher(dino_global_1), teacher(dino_global_2)]
                    student_outputs = [student(dino_global_1), student(dino_global_2)]
                    dino_loss_value, dino_stats = dino_loss_module(
                        [out["logits"] for out in student_outputs],
                        [out["logits"] for out in teacher_outputs],
                        teacher_temp=float(args.teacher_temp),
                    )
                    phase_progress = float(phase_step) / float(max(1, int(args.phase_steps)))
                    dino_weight = current_dino_weight(dino_weight_schedule, phase_progress)
                loss = siglip_loss + float(dino_weight) * dino_loss_value

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(grad_clip_params, float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            update_teacher(student, teacher, momentum=float(args.teacher_momentum))

            global_step += 1
            batch_in_epoch = bidx + bidx_offset + 1
            phase_step = max(0, global_step - phase_start_step)
            steps_per_s = phase_step / max(1e-6, (time.time() - wall_start))

            if global_step == phase_start_step + 1 or (phase_step % int(args.log_every) == 0):
                now = time.time()
                step_delta = max(1, phase_step - int(last_log_phase_step))
                window_steps_per_s = step_delta / max(1e-6, (now - last_log_wall))
                log_line = (
                    f"[siglip_align] step={global_step} phase_step={phase_step}/{args.phase_steps} epoch={epoch} "
                    f"phase={phase_name} loss={float(loss.item()):.4f} siglip_loss={float(siglip_loss.item()):.4f} "
                    f"lr={float(lr):.6g} logit_pos={loss_stats['logit_pos']:.4f} logit_neg={loss_stats['logit_neg']:.4f} "
                    f"image_std={loss_stats['image_std']:.4f} text_std={loss_stats['text_std']:.4f}"
                )
                if phase_name == "cross":
                    log_line += (
                        f" dino_loss={float(dino_loss_value.item()):.4f} dino_weight={float(dino_weight):.4f} "
                        f"teacher_entropy={float(dino_stats['teacher_entropy']):.4f} center_norm={float(dino_stats['center_norm']):.4f}"
                    )
                if device == "cuda" and bool(args.log_cuda_memory):
                    mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    log_line += f" max_mem_gb={mem_gb:.2f}"
                log_line += f" steps_per_s={steps_per_s:.2f} window_steps_per_s={window_steps_per_s:.2f}"
                logger.log(log_line)
                last_log_wall = now
                last_log_phase_step = phase_step

            if int(args.ckpt_every) > 0 and phase_step % int(args.ckpt_every) == 0:
                path = os.path.join(run_dir, f"step_{global_step}.tar")
                save_checkpoint(
                    path,
                    student=student,
                    teacher=teacher,
                    align_head=align_head,
                    text_tower=text_tower,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    phase_start_step=phase_start_step,
                    args=args,
                    base_payload=payload,
                    dino_loss_state=(None if dino_loss_module is None else dino_loss_module.state_dict()),
                )
                logger.log(f"[siglip_align] checkpoint saved: {path}")

        batch_in_epoch = 0
        if bool(args.save_every_epoch):
            path = os.path.join(run_dir, f"epoch_siglip_{epoch}.tar")
            save_checkpoint(
                path,
                student=student,
                teacher=teacher,
                align_head=align_head,
                text_tower=text_tower,
                optimizer=optimizer,
                scaler=scaler,
                global_step=global_step,
                epoch=epoch + 1,
                batch_in_epoch=0,
                phase_start_step=phase_start_step,
                args=args,
                base_payload=payload,
                dino_loss_state=(None if dino_loss_module is None else dino_loss_module.state_dict()),
            )
            logger.log(f"[siglip_align] epoch checkpoint saved: {path}")
        if max(0, global_step - phase_start_step) >= int(args.phase_steps):
            break

    final_path = os.path.join(run_dir, f"step_{global_step}.tar")
    save_checkpoint(
        final_path,
        student=student,
        teacher=teacher,
        align_head=align_head,
        text_tower=text_tower,
        optimizer=optimizer,
        scaler=scaler,
        global_step=global_step,
        epoch=epoch + 1,
        batch_in_epoch=0,
        phase_start_step=phase_start_step,
        args=args,
        base_payload=payload,
        dino_loss_state=(None if dino_loss_module is None else dino_loss_module.state_dict()),
    )
    logger.log(
        f"[siglip_align] done global_step={global_step} phase_completed={global_step - phase_start_step} "
        f"checkpoint={final_path}"
    )


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
