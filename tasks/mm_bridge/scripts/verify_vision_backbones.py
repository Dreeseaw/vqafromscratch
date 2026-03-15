"""Quick verification of new HF vision backbone output shapes and grad isolation."""
from __future__ import annotations

import argparse
import sys
import torch


def verify_dinov2(model_dir: str, device: str) -> None:
    from models.hf_vision import HFDINOv2SmallBackbone

    print(f"\n{'='*60}")
    print("DINOv2-small verification")
    print(f"{'='*60}")
    model = HFDINOv2SmallBackbone(model_dir=model_dir, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dummy = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Input shape:  {tuple(dummy.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print(f"  Output dtype: {out.dtype}")
    print(f"  Params:       {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Grad check:   {any(p.requires_grad for p in model.parameters())}")
    expected_tokens = (224 // 14) ** 2  # 256
    assert out.shape == (2, expected_tokens, 384), f"Expected (2, {expected_tokens}, 384), got {tuple(out.shape)}"
    print("  PASS")


def verify_mobileclip(model_dir: str, device: str) -> None:
    from models.hf_vision import OpenCLIPBackbone
    import os

    print(f"\n{'='*60}")
    print("MobileCLIP-S0 verification")
    print(f"{'='*60}")
    ckpt_path = os.path.join(model_dir, "open_clip_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"  SKIP — checkpoint not found at {ckpt_path}")
        return

    model = OpenCLIPBackbone("MobileCLIP2-S0", checkpoint_path=ckpt_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dummy = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Input shape:  {tuple(dummy.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print(f"  Output dtype: {out.dtype}")
    print(f"  Params:       {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Grad check:   {any(p.requires_grad for p in model.parameters())}")
    print(f"  Tokens:       {out.shape[1]}")
    print(f"  Feature dim:  {out.shape[2]}")
    print("  PASS")


def verify_mobileclip_architecture(model_dir: str) -> None:
    """Print the visual encoder architecture for debugging token extraction."""
    import os

    print(f"\n{'='*60}")
    print("MobileCLIP-S0 architecture inspection")
    print(f"{'='*60}")
    ckpt_path = os.path.join(model_dir, "open_clip_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"  SKIP — checkpoint not found at {ckpt_path}")
        return

    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "MobileCLIP2-S0", pretrained=ckpt_path,
        )
        visual = model.visual
        print(f"  Visual encoder type: {type(visual).__name__}")
        print(f"  Has .trunk: {hasattr(visual, 'trunk')}")
        if hasattr(visual, "trunk"):
            print(f"  Trunk type: {type(visual.trunk).__name__}")
            print(f"  Trunk has forward_features: {hasattr(visual.trunk, 'forward_features')}")

        print(f"\n  Top-level children:")
        for name, child in visual.named_children():
            print(f"    {name}: {type(child).__name__}")

        # Try forward_features if available
        trunk = getattr(visual, "trunk", None)
        if trunk is not None and hasattr(trunk, "forward_features"):
            dummy = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                ff_out = trunk.forward_features(dummy)
            print(f"\n  trunk.forward_features output shape: {tuple(ff_out.shape)}")
            print(f"  trunk.forward_features output dtype: {ff_out.dtype}")
        elif hasattr(visual, "forward_features"):
            dummy = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                ff_out = visual.forward_features(dummy)
            print(f"\n  visual.forward_features output shape: {tuple(ff_out.shape)}")

        # Also try standard forward to see pooled shape
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            pooled = visual(dummy)
        print(f"  visual() pooled output shape: {tuple(pooled.shape)}")

    except Exception as e:
        print(f"  ERROR: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov2_dir", type=str, default="logs/hf_vision/facebook_dinov2_small")
    ap.add_argument("--mobileclip_dir", type=str, default="logs/hf_vision/apple_mobileclip_s0")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_dinov2", action="store_true")
    ap.add_argument("--skip_mobileclip", action="store_true")
    args = ap.parse_args()

    if not args.skip_dinov2:
        try:
            verify_dinov2(args.dinov2_dir, args.device)
        except Exception as e:
            print(f"  FAIL: {e}", file=sys.stderr)

    if not args.skip_mobileclip:
        verify_mobileclip_architecture(args.mobileclip_dir)
        try:
            verify_mobileclip(args.mobileclip_dir, args.device)
        except Exception as e:
            print(f"  FAIL: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
