from __future__ import annotations

import argparse
import json
import os

import torch

from models.hf_vision import OpenCLIPBackbone, download_pe_core_b16, download_siglip2_b16


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download and verify Tier 0 VM backbones.")
    ap.add_argument("--siglip2_dir", type=str, default="logs/hf_vision/openclip_siglip2_b16_webli")
    ap.add_argument("--pe_core_dir", type=str, default="logs/hf_vision/openclip_pe_core_b16_meta")
    ap.add_argument("--skip_siglip2", action="store_true")
    ap.add_argument("--skip_pe_core", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output_json", type=str, default="")
    return ap.parse_args()


@torch.no_grad()
def _verify(backbone: OpenCLIPBackbone, *, device: str) -> dict[str, object]:
    x = torch.randn(1, 3, 224, 224, device=device)
    y = backbone(x)
    trunk = getattr(backbone, "_encoder", None)
    blocks = getattr(trunk, "_core_blocks", None) if trunk is not None else None
    return {
        "token_shape": [int(v) for v in y.shape],
        "strip_cls_token": bool(getattr(backbone, "strip_cls_token", False)),
        "encoder_type": type(trunk).__name__ if trunk is not None else None,
        "core_block_count": int(len(blocks)) if isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)) else 0,
    }


def main() -> None:
    args = parse_args()
    device = str(args.device)
    results: dict[str, object] = {}

    if not bool(args.skip_siglip2):
        out_dir = download_siglip2_b16(args.siglip2_dir)
        ckpt_path = os.path.join(out_dir, "open_clip_model.pt")
        model = OpenCLIPBackbone("ViT-B-16-SigLIP2", checkpoint_path=ckpt_path, device=device, strip_cls_token=False)
        results["siglip2_b16"] = {
            "dir": os.path.abspath(out_dir),
            "checkpoint": os.path.abspath(ckpt_path),
            **_verify(model, device=device),
        }
        print(f"[tier0-vm] siglip2_b16 dir={out_dir} shape={results['siglip2_b16']['token_shape']}")

    if not bool(args.skip_pe_core):
        out_dir = download_pe_core_b16(args.pe_core_dir)
        ckpt_path = os.path.join(out_dir, "open_clip_model.pt")
        model = OpenCLIPBackbone("PE-Core-B-16", checkpoint_path=ckpt_path, device=device, strip_cls_token=True)
        results["pe_core_b16"] = {
            "dir": os.path.abspath(out_dir),
            "checkpoint": os.path.abspath(ckpt_path),
            **_verify(model, device=device),
        }
        print(f"[tier0-vm] pe_core_b16 dir={out_dir} shape={results['pe_core_b16']['token_shape']}")

    if str(args.output_json).strip():
        out_path = os.path.abspath(str(args.output_json))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=True)
        print(f"[tier0-vm] wrote: {out_path}")


if __name__ == "__main__":
    main()
