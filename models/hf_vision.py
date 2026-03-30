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


class HFDINOv2SmallBackbone(nn.Module):
    """
    DINOv2 ViT-S/14 wrapper.  22 M params, self-supervised on LVD-142M.
    Returns 256 patch tokens (CLS stripped) at 384-dim for 224×224 input.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        model_dir = os.path.abspath(str(model_dir))
        self.model_dir = model_dir
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
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
        return imgs.to(device=device, non_blocking=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_inputs(images)
        use_amp = pixel_values.device.type == "cuda" and bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = self.model(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
        # Strip the CLS token (index 0), keep only patch tokens.
        hidden = outputs.last_hidden_state[:, 1:, :]
        return hidden.float()

    def _encoder(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


class HFDINOv2BaseBackbone(nn.Module):
    """
    DINOv2 ViT-B/14 wrapper.  ~86 M params, self-supervised on LVD-142M.
    Returns 256 patch tokens (CLS stripped) at 768-dim for 224×224 input.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        model_dir = os.path.abspath(str(model_dir))
        self.model_dir = model_dir
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
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
        return imgs.to(device=device, non_blocking=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_inputs(images)
        use_amp = pixel_values.device.type == "cuda" and bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = self.model(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
        # Strip the CLS token (index 0), keep only patch tokens.
        hidden = outputs.last_hidden_state[:, 1:, :]
        return hidden.float()

    def _encoder(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


class HFSigLIPBasePatch16Backbone(nn.Module):
    """
    SigLIP ViT-B/16 wrapper.  ~86M vision params, sigmoid CLIP on WebLI.
    Returns 196 patch tokens (no CLS token) at 768-dim for 224x224 input.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        model_dir = os.path.abspath(str(model_dir))
        self.model_dir = model_dir
        full_model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.model = full_model.vision_model
        del full_model
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
        return imgs.to(device=device, non_blocking=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_inputs(images)
        use_amp = pixel_values.device.type == "cuda" and bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = self.model(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
        # SigLIP has no CLS token — all 196 outputs are patch tokens.
        hidden = outputs.last_hidden_state
        return hidden.float()

    def _encoder(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


class ViTSTRTinyBackbone(nn.Module):
    """
    ViTSTR-Tiny OCR encoder wrapper.

    Uses the ViT backbone only: grayscale 224x224 input, patch16, embed_dim=192,
    depth=12, heads=3. Returns patch tokens only (CLS stripped), [B, 196, 192].
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer

        checkpoint_path = os.path.abspath(str(checkpoint_path))
        self.checkpoint_path = checkpoint_path
        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=1,
            num_classes=0,
            global_pool="",
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            class_token=True,
        )
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        cleaned = {}
        for key, value in dict(state_dict).items():
            if key.startswith("module.vitstr."):
                key = key[len("module.vitstr.") :]
            elif key.startswith("vitstr."):
                key = key[len("vitstr.") :]
            if key.startswith("head."):
                continue
            cleaned[key] = value
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        unexpected = [k for k in unexpected if not str(k).startswith("head.")]
        missing = [k for k in missing if not str(k).startswith("head.")]
        if unexpected:
            raise RuntimeError(f"Unexpected ViTSTR checkpoint keys: {unexpected[:8]}")
        self.load_report = {
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
        if device is not None:
            self.model.to(device)

        self._target_hw = (224, 224)
        self.input_resolution = (224, 224)
        self.patch_tokens = 196
        self.embed_dim = 192
        self.arch_summary = {
            "name": "vitstr_tiny_patch16_224_aug",
            "input_resolution": self.input_resolution,
            "patch_size": 16,
            "embed_dim": self.embed_dim,
            "depth": 12,
            "num_heads": 3,
            "patch_tokens": self.patch_tokens,
        }

    def _prepare_inputs(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")
        device = next(self.model.parameters()).device
        imgs = images.detach().float()
        mean = torch.tensor(_IMAGENET_MEAN, dtype=imgs.dtype, device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=imgs.dtype, device=imgs.device).view(1, 3, 1, 1)
        imgs = (imgs * std) + mean
        imgs = imgs.clamp(0.0, 1.0)
        if tuple(imgs.shape[-2:]) != self._target_hw:
            imgs = F.interpolate(imgs, size=self._target_hw, mode="bilinear", align_corners=False)
        gray = (
            0.299 * imgs[:, 0:1, :, :]
            + 0.587 * imgs[:, 1:2, :, :]
            + 0.114 * imgs[:, 2:3, :, :]
        )
        gray = (gray - 0.5) / 0.5
        return gray.to(device=device, non_blocking=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_inputs(images)
        use_amp = pixel_values.device.type == "cuda" and bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                hidden = self.model.forward_features(pixel_values)
        else:
            hidden = self.model.forward_features(pixel_values)
        if hidden.ndim != 3:
            raise RuntimeError(f"ViTSTR backbone expected token features [B,N,D], got {tuple(hidden.shape)}")
        return hidden[:, 1:, :].float()

    def _encoder(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


class DualSigLIPViTSTRTinyBackbone(nn.Module):
    """
    Frozen dual-VM backbone for OCR-aware bridge experiments.

    Returns a tuple:
    - SigLIP patch tokens [B, 196, 768]
    - ViTSTR patch tokens [B, 196, 192]

    The bridge is responsible for projecting the second stream and concatenating.
    """

    def __init__(
        self,
        siglip_model_dir: str,
        vitstr_checkpoint_path: str,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.siglip = HFSigLIPBasePatch16Backbone(siglip_model_dir, device=device)
        self.vitstr = ViTSTRTinyBackbone(vitstr_checkpoint_path, device=device)
        self.is_dual_visual = True
        self.last_feature_splits = (
            ("siglip", 196, 768),
            ("vitstr", 196, 192),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        siglip_tokens = self.siglip(images)
        vitstr_tokens = self.vitstr(images)
        self.last_feature_splits = (
            ("siglip", int(siglip_tokens.shape[1]), int(siglip_tokens.shape[2])),
            ("vitstr", int(vitstr_tokens.shape[1]), int(vitstr_tokens.shape[2])),
        )
        return siglip_tokens, vitstr_tokens

    def _encoder(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(images)


class OpenCLIPBackbone(nn.Module):
    """
    Generic OpenCLIP vision-encoder wrapper.  Loads a saved open_clip checkpoint
    and returns the *token-level* feature map (pre-pool) as [B, Nv, Dv].

    Works with MobileCLIP2-S0, MobileCLIP-S1, and other open_clip vision models.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        *,
        device: Optional[str] = None,
        strip_cls_token: bool = False,
    ) -> None:
        super().__init__()
        import open_clip

        self.model_name = model_name
        self.strip_cls_token = bool(strip_cls_token)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=checkpoint_path,
        )
        self.visual = model.visual
        # Expose the actual transformer trunk so the MM runtime can selectively
        # unfreeze the last N blocks for controlled VM finetuning.
        trunk = getattr(self.visual, "trunk", self.visual)
        self._encoder = trunk
        self._encoder_module = trunk
        core_blocks = getattr(trunk, "blocks", None)
        if isinstance(core_blocks, (nn.Sequential, nn.ModuleList)):
            setattr(trunk, "_core_blocks", core_blocks)
            setattr(self, "_core_blocks", core_blocks)
        # Drop text encoder to save memory.
        del model
        if device is not None:
            self.visual.to(device)

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_input_mean", mean, persistent=False)
        self.register_buffer("_input_std", std, persistent=False)
        self._target_hw = (224, 224)

    def _maybe_strip_cls(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.strip_cls_token):
            return x
        if x.ndim == 3 and int(x.shape[1]) > 1:
            return x[:, 1:, :]
        return x

    def _prepare_inputs(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")
        device = next(self.visual.parameters()).device
        imgs = images.detach().float()
        mean = self._input_mean.to(device=imgs.device, dtype=imgs.dtype)
        std = self._input_std.to(device=imgs.device, dtype=imgs.dtype)
        imgs = (imgs * std) + mean
        imgs = imgs.clamp(0.0, 1.0)
        if tuple(imgs.shape[-2:]) != self._target_hw:
            imgs = F.interpolate(imgs, size=self._target_hw, mode="bilinear", align_corners=False)
        return imgs.to(device=device, non_blocking=True)

    def _extract_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the visual encoder and return token features before pooling."""
        # open_clip ViT models: visual.trunk.forward_features or visual.forward
        # open_clip CNN/hybrid models: visual.trunk or visual.forward
        v = self.visual

        # Strategy 1: ViT-style open_clip models expose forward_features on trunk.
        trunk = getattr(v, "trunk", None)
        if trunk is not None and hasattr(trunk, "forward_features"):
            out = trunk.forward_features(pixel_values)
            if out.ndim == 3:
                return self._maybe_strip_cls(out)
            # CNN trunk may give [B, C, H, W] — flatten to [B, HW, C].
            if out.ndim == 4:
                b, c, h, w = out.shape
                return out.reshape(b, c, h * w).permute(0, 2, 1)

        # Strategy 2: The model itself may have forward_features.
        if hasattr(v, "forward_features"):
            out = v.forward_features(pixel_values)
            if out.ndim == 3:
                return self._maybe_strip_cls(out)
            if out.ndim == 4:
                b, c, h, w = out.shape
                return out.reshape(b, c, h * w).permute(0, 2, 1)

        # Strategy 3: Fallback — call forward and try to detect pre-pool tensors.
        # This uses hooks to intercept the last feature map before the pooler.
        captured = {}

        def _hook(module: nn.Module, inp: object, out: object) -> None:
            if isinstance(out, torch.Tensor):
                captured["feat"] = out

        # Find the last normalization or feature layer before the head.
        target = None
        for name, mod in v.named_modules():
            if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                target = mod
        if target is None:
            # Use second-to-last module as best guess.
            mods = list(v.named_modules())
            if len(mods) > 2:
                target = mods[-2][1]
        if target is not None:
            handle = target.register_forward_hook(_hook)
            try:
                v(pixel_values)
            finally:
                handle.remove()
            feat = captured.get("feat")
            if feat is not None:
                if feat.ndim == 3:
                    return self._maybe_strip_cls(feat)
                if feat.ndim == 4:
                    b, c, h, w = feat.shape
                    return feat.reshape(b, c, h * w).permute(0, 2, 1)

        raise RuntimeError(
            f"Could not extract token-level features from OpenCLIP model {self.model_name}. "
            "Please inspect the visual encoder architecture."
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_inputs(images)
        use_amp = pixel_values.device.type == "cuda" and bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                tokens = self._extract_tokens(pixel_values)
        else:
            tokens = self._extract_tokens(pixel_values)
        return tokens.float()

    def _encoder(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


def download_mobilevit_small(repo_id: str, out_dir: str) -> str:
    from transformers import AutoImageProcessor, MobileViTModel

    os.makedirs(out_dir, exist_ok=True)
    MobileViTModel.from_pretrained(repo_id).save_pretrained(out_dir)
    AutoImageProcessor.from_pretrained(repo_id).save_pretrained(out_dir)
    return os.path.abspath(out_dir)


def download_dinov2_small(repo_id: str, out_dir: str) -> str:
    from transformers import AutoModel, AutoImageProcessor

    os.makedirs(out_dir, exist_ok=True)
    AutoModel.from_pretrained(repo_id).save_pretrained(out_dir)
    AutoImageProcessor.from_pretrained(repo_id).save_pretrained(out_dir)
    return os.path.abspath(out_dir)


def download_mobileclip_s0(out_dir: str) -> str:
    """Download MobileCLIP2-S0 checkpoint via open_clip.

    MobileCLIP-S0 (v1) is not in open_clip>=3.3.  MobileCLIP2-S0 (v2) is the
    same parameter budget (~11.4 M image-encoder params) with updated training
    on DFN-filtered data.  At 224×224 input the feature map is 7×7 = 49 tokens
    at 1024-dim, exactly matching MobileViT's token count.
    """
    import open_clip

    os.makedirs(out_dir, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained="dfndr2b",
    )
    ckpt_path = os.path.join(out_dir, "open_clip_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(out_dir, "model_name.txt"), "w") as f:
        f.write("MobileCLIP2-S0\n")
    return os.path.abspath(out_dir)


def download_openclip_backbone(model_name: str, pretrained_tag: str, out_dir: str) -> str:
    import open_clip

    os.makedirs(out_dir, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_tag,
    )
    ckpt_path = os.path.join(out_dir, "open_clip_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(out_dir, "model_name.txt"), "w", encoding="utf-8") as f:
        f.write(f"{model_name}\n")
    with open(os.path.join(out_dir, "pretrained_tag.txt"), "w", encoding="utf-8") as f:
        f.write(f"{pretrained_tag}\n")
    return os.path.abspath(out_dir)


def download_siglip2_b16(out_dir: str) -> str:
    return download_openclip_backbone("ViT-B-16-SigLIP2", "webli", out_dir)


def download_pe_core_b16(out_dir: str) -> str:
    return download_openclip_backbone("PE-Core-B-16", "meta", out_dir)


def download_dinov2_base(repo_id: str, out_dir: str) -> str:
    from transformers import AutoModel, AutoImageProcessor

    os.makedirs(out_dir, exist_ok=True)
    AutoModel.from_pretrained(repo_id).save_pretrained(out_dir)
    AutoImageProcessor.from_pretrained(repo_id).save_pretrained(out_dir)
    return os.path.abspath(out_dir)


def download_siglip_base_patch16(repo_id: str, out_dir: str) -> str:
    from transformers import AutoModel, AutoImageProcessor

    os.makedirs(out_dir, exist_ok=True)
    AutoModel.from_pretrained(repo_id).save_pretrained(out_dir)
    AutoImageProcessor.from_pretrained(repo_id).save_pretrained(out_dir)
    return os.path.abspath(out_dir)


def download_vitstr_tiny_checkpoint(out_dir: str, *, augmented: bool = True) -> str:
    import urllib.request

    os.makedirs(out_dir, exist_ok=True)
    filename = "vitstr_tiny_patch16_224_aug.pth" if bool(augmented) else "vitstr_tiny_patch16_224.pth"
    url = f"https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/{filename}"
    out_path = os.path.join(out_dir, filename)
    if not os.path.isfile(out_path):
        urllib.request.urlretrieve(url, out_path)
    return os.path.abspath(out_path)
