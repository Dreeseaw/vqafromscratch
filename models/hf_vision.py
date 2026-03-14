from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class HFMobileViTSmallBackbone(nn.Module):
    """
    Hugging Face MobileViT-small wrapper that exposes token features through the same
    `_encoder` convention used by the existing MM runtime.

    This path is intended as a frozen-vision PoC bridge target, not a full trainable VM path.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        from transformers import MobileViTModel

        model_dir = os.path.abspath(str(model_dir))
        self.model_dir = model_dir
        self.model = MobileViTModel.from_pretrained(model_dir, local_files_only=True)
        if device is not None:
            self.model.to(device)

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_input_mean", mean, persistent=False)
        self.register_buffer("_input_std", std, persistent=False)
        self._target_hw = (224, 224)

    def _prepare_inputs(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")
        device = next(self.model.parameters()).device
        imgs = images.detach().float()
        mean = self._input_mean.to(device=imgs.device, dtype=imgs.dtype)
        std = self._input_std.to(device=imgs.device, dtype=imgs.dtype)
        imgs = (imgs * std) + mean
        imgs = imgs.clamp(0.0, 1.0)
        if tuple(imgs.shape[-2:]) != self._target_hw:
            imgs = F.interpolate(imgs, size=self._target_hw, mode="bilinear", align_corners=False)
        # MobileViT image processor flips RGB to BGR and rescales to [0, 1].
        imgs = imgs[:, [2, 1, 0], :, :]
        return imgs.to(device=device, non_blocking=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_inputs(images)
        use_amp = pixel_values.device.type == "cuda" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = self.model(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
        hidden = outputs.last_hidden_state
        return hidden.float()

    def _encoder(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


def download_mobilevit_small(repo_id: str, out_dir: str) -> str:
    from transformers import AutoImageProcessor, MobileViTModel

    os.makedirs(out_dir, exist_ok=True)
    MobileViTModel.from_pretrained(repo_id).save_pretrained(out_dir)
    AutoImageProcessor.from_pretrained(repo_id).save_pretrained(out_dir)
    return os.path.abspath(out_dir)
