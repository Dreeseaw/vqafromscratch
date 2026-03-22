"""
Caption-align pre-training for the bridge module.

Stage 1 of two-stage training:
  1. Align bridge output with LM caption encoding via cosine similarity.
  2. Then hand off to standard VQA fine-tuning from the pre-trained bridge.

Only bridge + prefix_calibrator params are optimized.  VM and LM are frozen.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.bpe_tokenizer import ByteBPETokenizer
from train.caption_data import COCOCaptionDataset
from train.mm import (
    LOGDIR,
    Logger,
    MultimodalPrefixLM,
    build_runtime_from_args,
    capture_rng_state,
    resolve_amp,
    resolve_device,
    save_mm_checkpoint,
    set_seed,
    _apply_runtime_defaults,
    _seed_loader_worker,
    EpochShuffleSampler,
    BridgeConfig,
)


def _collect_trainable_params(model: MultimodalPrefixLM):
    """Return bridge + prefix_calibrator params (the only trainable parts)."""
    trainable = []
    for name, p in model.named_parameters():
        if name.startswith("bridge.") or name.startswith("prefix_calibrator."):
            p.requires_grad_(True)
            trainable.append(p)
        else:
            p.requires_grad_(False)
    return trainable


@torch.no_grad()
def _encode_captions(
    lm: nn.Module,
    tokenizer: ByteBPETokenizer,
    captions: list[str],
    device: str,
    max_len: int = 64,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Tokenize captions, forward through frozen LM, mean-pool hidden states."""
    ids_list = []
    for cap in captions:
        toks = tokenizer.encode(cap, add_bos=True, add_eos=True).tolist()
        if len(toks) > max_len:
            toks = toks[:max_len]
        ids_list.append(toks)
    max_seq = max(len(t) for t in ids_list)
    pad_id = 0
    input_ids = torch.full((len(ids_list), max_seq), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros(len(ids_list), max_seq, dtype=torch.bool, device=device)
    for i, toks in enumerate(ids_list):
        input_ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
        mask[i, : len(toks)] = True
    pad_mask = ~mask  # True = padding

    text_emb = lm._embed_dropout(lm._embed(input_ids))
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else torch.inference_mode(mode=False)
    with ctx:
        hidden = lm._decode_only(text_emb, pad_mask=pad_mask, is_causal=True)

    valid = mask.unsqueeze(-1).float()
    pooled = (hidden.float() * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled, dim=-1)


def run_caption_pretrain(args: argparse.Namespace) -> None:
    device = resolve_device(getattr(args, "device", "auto"))
    set_seed(getattr(args, "seed", 42))
    use_amp, amp_dtype, use_scaler = resolve_amp(device, getattr(args, "precision", "bf16"))

    # Build model (same as VQA training).
    model, tokenizer, bridge_cfg = build_runtime_from_args(args, device=device)

    # Materialize lazy params with a dummy forward through VM → bridge.
    with torch.no_grad():
        dummy_img = torch.randn(2, 3, 224, 224, device=device)
        dummy_feat = model.vision_adapter(dummy_img)
        _ = model._bridge_forward(dummy_feat, question_context=None)
    del dummy_img, dummy_feat

    # Freeze everything except bridge + calibrator.
    trainable = _collect_trainable_params(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())

    run_id = args.run_id
    logger = Logger(run_id, checkpoint_id=None)
    logger.log(f"[caption_pretrain] trainable={n_trainable:,} / total={n_total:,}")
    logger.log(f"[caption_pretrain] bridge_cfg={asdict(bridge_cfg)}")

    # Dataset.
    images_root = getattr(args, "images_root", "images")
    annotations_root = getattr(args, "annotations_root", "annotations")
    dataset = COCOCaptionDataset(images_root, annotations_root)
    sampler = EpochShuffleSampler(dataset, seed=getattr(args, "seed", 42))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(getattr(args, "num_workers", 4)),
        prefetch_factor=int(getattr(args, "prefetch_factor", 2)),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_loader_worker,
    )

    # Optimizer (bridge params only).
    lr = float(getattr(args, "lr", 2e-4))
    max_steps = int(getattr(args, "max_steps", 3000))
    warmup_steps = int(getattr(args, "lr_warmup_steps", 200))
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    log_every = int(getattr(args, "log_every", 10))
    ckpt_every = int(getattr(args, "ckpt_every", 1000))

    # Collapse monitoring state.
    collapse_window = 100
    low_std_count = 0
    collapse_threshold = 0.01

    model.train()
    global_step = 0
    epoch = 0
    t_start = time.time()

    logger.log(f"[caption_pretrain] max_steps={max_steps} lr={lr} batch_size={args.batch_size}")
    logger.log(f"[caption_pretrain] dataset_size={len(dataset)} images")

    while global_step < max_steps:
        epoch += 1
        sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= max_steps:
                break
            global_step += 1
            images = batch["image"].to(device, non_blocking=True)
            captions = batch["caption"]

            # LR schedule: linear warmup + cosine decay.
            if global_step <= warmup_steps:
                cur_lr = lr * (global_step / max(1, warmup_steps))
            else:
                progress = (global_step - warmup_steps) / max(1, max_steps - warmup_steps)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # Forward: image → VM → bridge → mean-pool → L2-normalize.
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else torch.inference_mode(mode=False)
            with amp_ctx:
                visual_features = model.vision_adapter(images)
                bridge_out = model._bridge_forward(visual_features, question_context=None)
                bridge_out = model.prefix_calibrator(bridge_out)
                v = bridge_out.float().mean(dim=1)
                v = F.normalize(v, dim=-1)

                # Target: caption → frozen LM → mean-pool → L2-normalize.
                t = _encode_captions(
                    model.lm, tokenizer, captions, device,
                    amp_dtype=amp_dtype if use_amp else None,
                )

                # Loss: 1 - cosine_similarity.
                cos_sim = (v * t).sum(dim=-1)
                loss = (1.0 - cos_sim).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Collapse monitoring.
            with torch.no_grad():
                bridge_std = v.std().item()
            if bridge_std < collapse_threshold:
                low_std_count += 1
            else:
                low_std_count = 0
            if low_std_count >= collapse_window:
                logger.log(
                    f"[caption_pretrain] COLLAPSE DETECTED at step={global_step} "
                    f"bridge_std={bridge_std:.6f} < {collapse_threshold} for {collapse_window} steps"
                )
                sys.exit(1)

            # Logging.
            if global_step % log_every == 0 or global_step == 1:
                elapsed = time.time() - t_start
                sps = global_step / max(elapsed, 1e-6)
                logger.log(
                    f"[caption_pretrain] step={global_step}/{max_steps} "
                    f"loss={loss.item():.4f} cos_sim={cos_sim.mean().item():.4f} "
                    f"bridge_std={bridge_std:.4f} lr={cur_lr:.6f} "
                    f"steps_per_s={sps:.2f} elapsed={elapsed:.1f}s"
                )

            # Checkpoint.
            if ckpt_every > 0 and global_step % ckpt_every == 0:
                ckpt_path = os.path.join(LOGDIR, run_id, f"step_{global_step:06d}.tar")
                save_mm_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=0,
                    args=args,
                    bridge_cfg=bridge_cfg,
                )
                logger.log(f"[caption_pretrain] saved checkpoint {ckpt_path}")

    # Final checkpoint.
    ckpt_path = os.path.join(LOGDIR, run_id, f"step_{global_step:06d}.tar")
    save_mm_checkpoint(
        ckpt_path,
        model,
        optimizer,
        global_step=global_step,
        epoch=epoch,
        batch_in_epoch=0,
        args=args,
        bridge_cfg=bridge_cfg,
    )
    elapsed = time.time() - t_start
    logger.log(
        f"[caption_pretrain] DONE step={global_step} "
        f"final_loss={loss.item():.4f} elapsed={elapsed:.1f}s "
        f"checkpoint={ckpt_path}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Caption-align bridge pre-training")
    ap.add_argument("run_id", type=str)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--precision", type=str, default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_warmup_steps", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--images_root", type=str, default="images")
    ap.add_argument("--annotations_root", type=str, default="annotations")

    # Model/bridge args (forwarded to build_runtime_from_args).
    ap.add_argument("--tokenizer_path", type=str, default=None)
    ap.add_argument("--lm_checkpoint", type=str, default=None)
    ap.add_argument("--lm_config", type=str, default=None)
    ap.add_argument("--vision_model", type=str, default="mobilevit_hf")
    ap.add_argument("--vision_checkpoint", type=str, default=None)
    ap.add_argument("--vision_config", type=str, default=None)
    ap.add_argument("--vision_feature_source", type=str, default="encoder")
    ap.add_argument("--vision_feature_mode", type=str, default="auto")
    ap.add_argument("--bridge_type", type=str, default="perceiver_resampler")
    ap.add_argument("--num_visual_tokens", type=int, default=49)
    ap.add_argument("--bridge_hidden_dim", type=int, default=1024)
    ap.add_argument("--bridge_token_reduce", type=str, default="adaptive_pool")
    ap.add_argument("--bridge_add_2d_pos_emb", action="store_true")
    ap.add_argument("--bridge_num_heads", type=int, default=8)
    ap.add_argument("--bridge_query_depth", type=int, default=3)
    ap.add_argument("--bridge_pre_mixer_type", type=str, default="none")
    ap.add_argument("--bridge_question_conditioning", action="store_true")
    ap.add_argument("--bridge_question_context_mode", type=str, default="prompt_only")
    ap.add_argument("--bridge_query_bank_mode", type=str, default="learned")
    ap.add_argument("--bridge_token_selector_type", type=str, default="none")
    ap.add_argument("--bridge_token_select_k", type=int, default=0)
    ap.add_argument("--bridge_token_select_k_min", type=int, default=0)
    ap.add_argument("--prefix_calibration", action="store_true")
    ap.add_argument("--prefix_calib_layernorm", action="store_true")
    ap.add_argument("--prefix_calib_bias", action="store_true")
    ap.add_argument("--prefix_calib_gate_init", type=float, default=1.0)
    ap.add_argument("--prefix_geom_mlp_ratio", type=float, default=0.0)
    ap.add_argument("--prefix_geom_token_mixer_layers", type=int, default=0)
    ap.add_argument("--freeze_mode", type=str, default="bridge_only")

    args = ap.parse_args()
    args = _apply_runtime_defaults(args)
    run_caption_pretrain(args)


if __name__ == "__main__":
    main()
