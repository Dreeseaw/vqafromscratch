"""
Multimodal VQA training harness.

Role:
- Build an explicit image-feature -> bridge -> LM-prefix pipeline.
- Reuse existing VAE/LM/tokenizer/checkpoint components with minimal glue.
- Keep extension points clear for future visual feature types and bridge modules.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import sys
import time
import gc
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from models.bpe_tokenizer import ByteBPETokenizer
from models.bridge import BridgeConfig, build_bridge
from models.hf_vision import HFMobileViTSmallBackbone, HFDINOv2SmallBackbone, HFDINOv2BaseBackbone, OpenCLIPBackbone, HFSigLIPBasePatch16Backbone
from models.vit_ssl import DINOCheckpointBackbone
from models.lm import LMConfig, TransformerDecoderOnlyV1
from models.vae import VAEConfig, VariationalAutoEncoder, VariationalAutoEncoderRes, ViTVAE, ViTVAE2
from train.vqa_data import GQADataset, GroundingMixBatchSampler, MixedVQAv2Dataset, PointingIndexDataset, VQAv2Dataset, VQAv2Paths, build_image_transform, prepare_vqav2


LOGDIR = "logs"
LOGFILE = "logfile.txt"
DEFAULT_FINAL_SANITY_COUNT = 4
DEFAULT_TRAIN_SAMPLE_BUDGET = 1_152_000
DEFAULT_EVAL_FRACTION = 0.5
FINAL_SANITY_MIN_PROMPTS = 3
FINAL_SANITY_MAX_PROMPTS = 5
LOW_TRAIN_SPS_EXIT_CODE = 86


class Logger:
    def __init__(self, run_id: str, checkpoint_id: Optional[int]) -> None:
        self.run_id = run_id
        self.ckpt = checkpoint_id
        self.base = os.path.join(LOGDIR, run_id)
        os.makedirs(self.base, exist_ok=True)
        if checkpoint_id is None:
            fn = LOGFILE
        else:
            fn = f"logfile_from_{checkpoint_id}.txt"
        self.path = os.path.join(self.base, fn)

    def log(self, text: str) -> None:
        print(text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not isinstance(state, dict):
        return
    py_state = state.get("python")
    torch_cpu = state.get("torch_cpu")
    torch_cuda = state.get("torch_cuda")
    if py_state is not None:
        random.setstate(py_state)
    if torch_cpu is not None:
        torch.set_rng_state(torch_cpu)
    if torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda)


class EpochShuffleSampler(Sampler[int]):
    def __init__(self, data_source: Sequence[Any], *, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        g = torch.Generator()
        g.manual_seed(int(self.seed) + int(self.epoch))
        yield from torch.randperm(n, generator=g).tolist()


def _seed_loader_worker(worker_id: int) -> None:
    del worker_id
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable.")
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise SystemExit("MPS requested but unavailable.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_amp(device: str, precision: str) -> Tuple[bool, Optional[torch.dtype], bool]:
    p = str(precision).lower()
    if p not in ("fp32", "bf16", "fp16"):
        raise SystemExit("--precision must be fp32, bf16, or fp16")
    if device != "cuda":
        return False, None, False
    if p == "fp32":
        return False, None, False
    if p == "bf16":
        ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not ok:
            raise SystemExit("bf16 requested but CUDA bf16 support not available.")
        return True, torch.bfloat16, False
    return True, torch.float16, True


def load_json_or_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    txt_strip = txt.lstrip()
    if txt_strip.startswith("{") or txt_strip.startswith("["):
        return json.loads(txt)
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("YAML config requested but PyYAML not available.") from e
    data = yaml.safe_load(txt)
    return dict(data or {})


def parse_json_object_arg(name: str, raw: object) -> dict[str, object]:
    text = str(raw or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a single valid JSON object string; got {text!r}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object, got: {text!r}")
    return value


def _extract_state_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    raise ValueError("Could not infer state_dict from checkpoint payload.")


def _load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def _infer_tokenizer_from_lm_checkpoint(lm_checkpoint: Optional[str]) -> Optional[str]:
    if not lm_checkpoint:
        return None
    m = re.search(r"(.*[/\\]logs[/\\][^/\\]+)[/\\]step_\d+\.tar$", lm_checkpoint)
    if not m:
        m = re.search(r"(.*[/\\][^/\\]+)[/\\]step_\d+\.tar$", lm_checkpoint)
    if not m:
        return None
    cand = os.path.join(m.group(1), "tokenizer.pt")
    return cand if os.path.isfile(cand) else None


def resolve_tokenizer_path(args: argparse.Namespace, checkpoint_payload: Optional[Dict[str, Any]] = None) -> str:
    if args.tokenizer_path:
        return args.tokenizer_path
    if checkpoint_payload is not None:
        p = checkpoint_payload.get("tokenizer_path")
        if isinstance(p, str) and os.path.isfile(p):
            return p
    inf = _infer_tokenizer_from_lm_checkpoint(getattr(args, "lm_checkpoint", None))
    if inf:
        return inf
    raise SystemExit("Could not resolve tokenizer path. Set --tokenizer_path explicitly.")


def _apply_runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "tokenizer_path": None,
        "vision_model": "vitvae2",
        "vision_checkpoint": None,
        "vision_config": None,
        "vision_latent_dim": 768,
        "vision_cbld": 1536,
        "vision_device": "auto",
        "vision_feature_mode": "auto",
        "vision_feature_source": "posterior_mu",
        "lm_checkpoint": None,
        "lm_config": None,
        "lm_d_model": 768,
        "lm_num_heads": 8,
        "lm_layers": 5,
        "lm_mlp_ratio": 4,
        "lm_dropout": 0.1,
        "lm_max_seq_len": 512,
        "mm_sdp_backend": "math",
        "mm_lm_autocast": True,
        "bridge_type": "mlp",
        "bridge_hidden_dim": 1024,
        "num_visual_tokens": 8,
        "bridge_token_reduce": "adaptive_pool",
        "bridge_learned_init_std": 0.02,
        "bridge_add_2d_pos_emb": False,
        "bridge_num_heads": 8,
        "bridge_attn_dropout": 0.0,
        "bridge_query_depth": 2,
        "bridge_refine_layers": 0,
        "bridge_pre_mixer_type": "none",
        "bridge_pre_mixer_layers": 1,
        "bridge_pre_mixer_kernel_size": 3,
        "bridge_hybrid_alpha_mode": "scalar",
        "bridge_hybrid_alpha_init": 0.5,
        "bridge_hybrid_image_bridge_type": "learned_query",
        "bridge_question_conditioning": False,
        "bridge_qcond_scale": 0.5,
        "bridge_query_bank_mode": "learned",
        "bridge_qquery_basis_count": 4,
        "bridge_qquery_scale": 1.0,
        "bridge_qquery_multi_count": 1,
        "bridge_qquery_hybrid_gate_init": 0.5,
        "bridge_query_role_specialization": False,
        "bridge_question_context_mode": "all_text",
        "bridge_iterative_qquery_steps": 1,
        "bridge_iterative_qquery_residual_scale": 1.0,
        "eval_use_kv_cache": False,
        "eval_kv_cache_mode": "batched",
        "bridge_token_selector_type": "none",
        "bridge_token_select_k": 0,
        "bridge_token_select_k_min": 0,
        "bridge_num_roles": 4,
        "bridge_evidence_topk": 0,
        "semantic_bottleneck": False,
        "semantic_tokens": 16,
        "semantic_latent_dim": 256,
        "semantic_recon_loss_weight": 0.1,
        "semantic_consistency_loss_weight": 0.1,
        "semantic_token_schedule": "",
        "semantic_teacher_checkpoint": "",
        "init_from_mm_checkpoint": "",
        "use_compression": False,
        "compression_k": 16,
        "compression_distill_weight": 0.1,
        "use_grounding_loss": False,
        "grounding_loss_weight": 0.05,
        "grounding_sigma": 1.5,
        "pointing_index_path": "",
        "pointing_mix_ratio": 0.25,
        "eval_bypass_compression": False,
        "eval_tiny_head": False,
        "eval_grounding": False,
        "eval_grounding_index_path": "",
        "tiny_head_epochs": 1,
        "tiny_head_answer_top_k": 3000,
        "tiny_head_feature_pool": "flatten",
        "visual_feature_adapter_type": "none",
        "visual_feature_adapter_hidden_dim": 0,
        "visual_feature_adapter_dropout": 0.0,
        "train_vision_last_n_blocks": 0,
        "vision_lr_scale": 1.0,
        "lm_visual_adapter_type": "none",
        "lm_visual_adapter_layers": 0,
        "lm_visual_adapter_num_heads": 8,
        "lm_visual_adapter_dropout": 0.0,
        "lm_visual_adapter_gate_init": 0.5,
        "prefix_calibration": False,
        "prefix_calib_layernorm": True,
        "prefix_calib_bias": True,
        "prefix_calib_gate_init": 1.0,
        "prefix_geom_mlp_ratio": 0.0,
        "prefix_geom_token_mixer_layers": 0,
        "prefix_norm_target_ratio": 0.0,
        "prefix_norm_reg_weight": 0.0,
        "prefix_batchvar_reg_weight": 0.0,
        "prefix_dropout": 0.0,
        "lr_schedule": "constant",
        "lr_warmup_steps": 0,
        "lr_min_ratio": 0.1,
        "min_train_steps_per_s": 1.0,
        "min_train_steps_window": 100,
        "cuda_empty_cache_after_eval": True,
        "images_root": "images",
        "annotations_root": "annotations",
        "gqa_root": "data/gqa",
        "dataset_mix": "",
        "max_question_length": 64,
        "max_answer_length": 16,
        "max_text_tokens": 256,
        "batch_size": 16,
        "eval_batch_size": 0,
        "eval_image_corruption": "none",
        "eval_image_corruption_seed": 123,
        "num_workers": 2,
        "prefetch_factor": 2,
        "pin_memory": True,
        "grad_accum_steps": 1,
        "fixed_eval_count": 5,
        "freeze_mode": "bridge_only",
        "train_top_lm_layers": 1,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args = _normalize_semantic_aliases(args)
    return args


def _normalize_semantic_aliases(args: argparse.Namespace) -> argparse.Namespace:
    if bool(getattr(args, "use_compression", False)):
        args.semantic_bottleneck = True
    if hasattr(args, "compression_k") and int(getattr(args, "compression_k", 0)) > 0:
        args.semantic_tokens = int(getattr(args, "compression_k"))
    if hasattr(args, "compression_distill_weight"):
        args.semantic_recon_loss_weight = float(getattr(args, "compression_distill_weight"))
    return args


def build_vision_model_from_args(args: argparse.Namespace, device: str, ckpt_payload: Optional[Dict[str, Any]] = None) -> nn.Module:
    meta = {}
    if ckpt_payload is not None:
        meta = dict(ckpt_payload.get("vision_meta", {}) or {})

    vision_model_name = str(meta.get("vision_model", args.vision_model))
    latent_dim = int(meta.get("vision_latent_dim", args.vision_latent_dim))
    cbld = meta.get("vision_cbld", args.vision_cbld)
    cbld = None if cbld in (None, "None") else int(cbld)

    file_cfg = load_json_or_yaml(args.vision_config)
    if "latent_dim" in file_cfg:
        latent_dim = int(file_cfg["latent_dim"])
    if "cbld" in file_cfg:
        cbld = None if file_cfg["cbld"] is None else int(file_cfg["cbld"])

    cfg = VAEConfig(latent_dim=latent_dim, cbld=cbld)
    if vision_model_name == "vae":
        model = VariationalAutoEncoder(cfg)
    elif vision_model_name == "vaer":
        model = VariationalAutoEncoderRes(cfg)
    elif vision_model_name == "vitvae":
        model = ViTVAE(cfg)
    elif vision_model_name == "vitvae2":
        model = ViTVAE2(cfg)
    elif vision_model_name == "mobilevit_hf":
        if int(getattr(args, "train_vision_last_n_blocks", 0)) > 0 or str(getattr(args, "freeze_mode", "")) == "full_finetune":
            raise SystemExit("mobilevit_hf support is currently frozen-vision only; VM finetuning is not wired for this path.")
        model_dir = str(args.vision_checkpoint or meta.get("vision_checkpoint", ""))
        if not model_dir:
            raise SystemExit("--vision_checkpoint must point to a local MobileViT directory for --vision_model mobilevit_hf")
        model = HFMobileViTSmallBackbone(model_dir=model_dir, device=device)
        return model.to(device)
    elif vision_model_name == "dinov2_small":
        if int(getattr(args, "train_vision_last_n_blocks", 0)) > 0 or str(getattr(args, "freeze_mode", "")) == "full_finetune":
            raise SystemExit("dinov2_small is currently frozen-vision only; VM finetuning is not wired for this path.")
        model_dir = str(args.vision_checkpoint or meta.get("vision_checkpoint", ""))
        if not model_dir:
            raise SystemExit("--vision_checkpoint must point to a local DINOv2-small directory for --vision_model dinov2_small")
        model = HFDINOv2SmallBackbone(model_dir=model_dir, device=device)
        return model.to(device)
    elif vision_model_name == "dinov2_base":
        if int(getattr(args, "train_vision_last_n_blocks", 0)) > 0 or str(getattr(args, "freeze_mode", "")) == "full_finetune":
            raise SystemExit("dinov2_base is currently frozen-vision only; VM finetuning is not wired for this path.")
        model_dir = str(args.vision_checkpoint or meta.get("vision_checkpoint", ""))
        if not model_dir:
            raise SystemExit("--vision_checkpoint must point to a local DINOv2-base directory for --vision_model dinov2_base")
        model = HFDINOv2BaseBackbone(model_dir=model_dir, device=device)
        return model.to(device)
    elif vision_model_name == "mobileclip_s0":
        if int(getattr(args, "train_vision_last_n_blocks", 0)) > 0 or str(getattr(args, "freeze_mode", "")) == "full_finetune":
            raise SystemExit("mobileclip_s0 is currently frozen-vision only; VM finetuning is not wired for this path.")
        model_dir = str(args.vision_checkpoint or meta.get("vision_checkpoint", ""))
        if not model_dir:
            raise SystemExit("--vision_checkpoint must point to a local MobileCLIP-S0 directory for --vision_model mobileclip_s0")
        ckpt_path = os.path.join(model_dir, "open_clip_model.pt")
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"Expected checkpoint at {ckpt_path}. Run the download script first.")
        model = OpenCLIPBackbone("MobileCLIP2-S0", checkpoint_path=ckpt_path, device=device)
        return model.to(device)
    elif vision_model_name == "siglip_base":
        if int(getattr(args, "train_vision_last_n_blocks", 0)) > 0 or str(getattr(args, "freeze_mode", "")) == "full_finetune":
            raise SystemExit("siglip_base is currently frozen-vision only; VM finetuning is not wired for this path.")
        model_dir = str(args.vision_checkpoint or meta.get("vision_checkpoint", ""))
        if not model_dir:
            raise SystemExit("--vision_checkpoint must point to a local SigLIP directory for --vision_model siglip_base")
        model = HFSigLIPBasePatch16Backbone(model_dir=model_dir, device=device)
        return model.to(device)
    elif vision_model_name == "dinovit_ssl":
        if int(getattr(args, "train_vision_last_n_blocks", 0)) > 0 or str(getattr(args, "freeze_mode", "")) == "full_finetune":
            raise SystemExit("dinovit_ssl is currently frozen-vision only; VM finetuning is not wired for this path.")
        ckpt_path = str(args.vision_checkpoint or meta.get("vision_checkpoint", ""))
        if not ckpt_path:
            raise SystemExit("--vision_checkpoint must point to a local DINO SSL checkpoint for --vision_model dinovit_ssl")
        model = DINOCheckpointBackbone(checkpoint_path=ckpt_path, device=device)
        return model.to(device)
    else:
        raise ValueError(f"Unsupported --vision_model: {vision_model_name}")

    if args.vision_checkpoint:
        payload = _load_checkpoint(args.vision_checkpoint, map_location="cpu")
        model.load_state_dict(_extract_state_dict(payload), strict=True)
    return model.to(device)


def _lmcfg_from_dict(cfg_data: Dict[str, Any], tokenizer: ByteBPETokenizer, args: argparse.Namespace) -> LMConfig:
    sig = inspect.signature(LMConfig.__init__)
    allowed = {k for k in sig.parameters.keys() if k != "self"}
    merged = {
        "vocab_size": int(tokenizer.vocab_size),
        "embed_size": int(args.lm_d_model),
        "num_heads": int(args.lm_num_heads),
        "mlp_ratio": int(args.lm_mlp_ratio),
        "layers": int(args.lm_layers),
        "max_seq_len": int(args.lm_max_seq_len),
        "dropout": float(args.lm_dropout),
        "causal_lm": True,
    }
    for k, v in cfg_data.items():
        if k in allowed:
            merged[k] = v
    merged["vocab_size"] = int(tokenizer.vocab_size)
    filtered = {k: v for k, v in merged.items() if k in allowed}
    return LMConfig(**filtered)


def build_lm_from_args(
    args: argparse.Namespace,
    tokenizer: ByteBPETokenizer,
    device: str,
    ckpt_payload: Optional[Dict[str, Any]] = None,
) -> TransformerDecoderOnlyV1:
    cfg_data: Dict[str, Any] = {}
    if args.lm_config:
        cfg_data.update(load_json_or_yaml(args.lm_config))
    if ckpt_payload is not None and isinstance(ckpt_payload.get("lm_config"), dict):
        cfg_data.update(dict(ckpt_payload["lm_config"]))
    lm_payload = None
    if args.lm_checkpoint:
        lm_payload = _load_checkpoint(args.lm_checkpoint, map_location="cpu")
        if isinstance(lm_payload.get("config"), dict):
            cfg_data.update(dict(lm_payload["config"]))

    cfg = _lmcfg_from_dict(cfg_data, tokenizer, args)
    model = TransformerDecoderOnlyV1(cfg).to(device)
    if str(getattr(model._config, "attn_impl", "sdpa")) == "sdpa":
        backend = str(args.mm_sdp_backend)
        model._config.sdp_backend = backend
        for blk in getattr(model, "_dec_blocks", []):
            if hasattr(blk, "_sdp_backend"):
                blk._sdp_backend = backend
    if lm_payload is not None:
        model.load_state_dict(_extract_state_dict(lm_payload), strict=True)
    return model


class VisionFeatureAdapter(nn.Module):
    """
    Explicit feature-extraction boundary.

    Accepts model outputs in any of:
    - [B, Dv]
    - [B, Nv, Dv]
    - [B, C, H, W] (converted to [B, Nv, Dv])
    """

    def __init__(
        self,
        vision_model: nn.Module,
        feature_mode: str = "auto",
        feature_source: str = "posterior_mu",
        *,
        vision_device: Optional[str] = None,
        output_device: Optional[str] = None,
        force_no_grad: bool = True,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.feature_mode = feature_mode
        self.feature_source = feature_source
        self.vision_device = None if vision_device in (None, "auto") else str(vision_device)
        self.output_device = None if output_device in (None, "auto") else str(output_device)
        self.force_no_grad = bool(force_no_grad)

    def _extract_raw(self, images: torch.Tensor) -> torch.Tensor:
        src = str(self.feature_source)
        m = self.vision_model
        if src == "encoder":
            if not hasattr(m, "_encoder"):
                raise ValueError("feature_source=encoder requires vision model._encoder")
            return m._encoder(images)
        if src == "encoder_plus_posterior_mu":
            if not hasattr(m, "_encoder") or not hasattr(m, "_post_head"):
                raise ValueError("feature_source=encoder_plus_posterior_mu requires _encoder and _post_head")
            h = m._encoder(images)
            mu, _ = m._post_head(h)
            return h, mu
        if src == "posterior_mu":
            if not hasattr(m, "_encoder") or not hasattr(m, "_post_head"):
                raise ValueError("feature_source=posterior_mu requires _encoder and _post_head")
            h = m._encoder(images)
            mu, _ = m._post_head(h)
            return mu
        out = m(images)
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            for x in out:
                if torch.is_tensor(x):
                    return x
        raise ValueError("Could not extract tensor features from vision model output.")

    def _move_output(self, out: torch.Tensor) -> torch.Tensor:
        if self.output_device is not None and str(out.device) != self.output_device:
            out = out.to(self.output_device, non_blocking=True)
        return out

    def _format_single(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)  # [B,C,H,W] -> [B,N,C]
        if self.feature_mode == "global":
            if x.ndim == 2:
                return self._move_output(x)
            if x.ndim == 3:
                return self._move_output(x.mean(dim=1))
            raise ValueError(f"Unsupported feature shape for global mode: {tuple(x.shape)}")
        if self.feature_mode == "token":
            if x.ndim == 3:
                return self._move_output(x)
            if x.ndim == 2:
                return self._move_output(x.unsqueeze(1))
            raise ValueError(f"Unsupported feature shape for token mode: {tuple(x.shape)}")
        if x.ndim in (2, 3):
            return self._move_output(x)
        raise ValueError(f"Unsupported feature shape for auto mode: {tuple(x.shape)}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x_in = images
        if self.vision_device is not None and str(x_in.device) != self.vision_device:
            x_in = x_in.to(self.vision_device, non_blocking=True)
        ctx = torch.no_grad() if self.force_no_grad else nullcontext()
        with ctx:
            x = self._extract_raw(x_in)
        if isinstance(x, (tuple, list)):
            if self.feature_mode == "global":
                raise ValueError("feature_source=encoder_plus_posterior_mu does not support feature_mode=global")
            return tuple(self._format_single(t) for t in x)
        return self._format_single(x)


class VisualFeatureResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = max(0, int(hidden_dim))
        self.dropout = max(0.0, float(dropout))
        self._token_ff = nn.ModuleDict()

    def _ensure_token_ff(self, dim: int, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        key = str(int(dim))
        if key in self._token_ff:
            return self._token_ff[key]
        h = self.hidden_dim if self.hidden_dim > 0 else max(4, 2 * int(dim))
        mod = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), h),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(h, int(dim)),
        ).to(device=device, dtype=dtype)
        self._token_ff[key] = mod
        return mod

    def _adapt_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeezed = True
        else:
            squeezed = False
        if x.ndim != 3:
            raise ValueError(f"VisualFeatureResidualAdapter expected [B,D] or [B,N,D], got {tuple(x.shape)}")
        ff = self._ensure_token_ff(int(x.shape[-1]), device=x.device, dtype=x.dtype)
        y = x + ff(x)
        return y.squeeze(1) if squeezed else y

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]):
        if isinstance(x, (tuple, list)):
            return type(x)(self._adapt_tensor(t) for t in x)
        return self._adapt_tensor(x)


class _PrefixGeomTokenMixerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(int(dim))
        self.dw = nn.Conv1d(int(dim), int(dim), kernel_size=3, padding=1, groups=int(dim), bias=False)
        self.pw = nn.Conv1d(int(dim), int(dim), kernel_size=1, bias=True)
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x).transpose(1, 2)
        y = self.dw(y)
        y = F.gelu(y)
        y = self.pw(y).transpose(1, 2)
        return x + y


class PrefixCalibrator(nn.Module):
    """
    Optional bridge-output calibrator to better match frozen LM embedding statistics.

    y = gate * LN(x) + bias
    """

    def __init__(
        self,
        dim: int,
        *,
        enabled: bool,
        use_layernorm: bool,
        use_bias: bool,
        gate_init: float,
        geom_mlp_ratio: float,
        geom_token_mixer_layers: int,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        if not self.enabled:
            self.ln = nn.Identity()
            self.log_gate = None
            self.bias = None
            self.geom_blocks = nn.ModuleList()
            self.geom_ln = nn.Identity()
            self.geom_mlp = None
            return

        self.ln = nn.LayerNorm(int(dim), elementwise_affine=False) if bool(use_layernorm) else nn.Identity()
        init = max(1e-5, float(gate_init))
        self.log_gate = nn.Parameter(torch.full((1, 1, int(dim)), math.log(init)))
        self.bias = nn.Parameter(torch.zeros(1, 1, int(dim))) if bool(use_bias) else None
        self.geom_blocks = nn.ModuleList(
            [_PrefixGeomTokenMixerBlock(int(dim)) for _ in range(max(0, int(geom_token_mixer_layers)))]
        )
        ratio = float(geom_mlp_ratio)
        if ratio > 0.0:
            h = max(4, int(round(int(dim) * ratio)))
            self.geom_ln = nn.LayerNorm(int(dim))
            self.geom_mlp = nn.Sequential(
                nn.Linear(int(dim), h),
                nn.GELU(),
                nn.Linear(h, int(dim)),
            )
            last = self.geom_mlp[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            self.geom_ln = nn.Identity()
            self.geom_mlp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        y = self.ln(x)
        assert self.log_gate is not None
        gate = torch.exp(self.log_gate).clamp(min=1e-5, max=100.0)
        y = y * gate
        if self.bias is not None:
            y = y + self.bias
        for blk in self.geom_blocks:
            y = blk(y)
        if self.geom_mlp is not None:
            y = y + self.geom_mlp(self.geom_ln(y))
        return y


def _resolve_mha_heads(dim: int, requested: int) -> int:
    d = max(1, int(dim))
    h = max(1, min(int(requested), d))
    while h > 1 and (d % h) != 0:
        h -= 1
    return h


class ResidualVisualAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        d = int(dim)
        p = max(0.0, float(dropout))
        gate = min(1.0 - 1e-4, max(1e-4, float(gate_init)))
        logit = math.log(gate / (1.0 - gate))
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(
            d,
            num_heads=_resolve_mha_heads(d, int(num_heads)),
            dropout=p,
            batch_first=True,
        )
        self.cross_drop = nn.Dropout(p)
        self.cross_gate_logit = nn.Parameter(torch.full((1, 1, d), logit))
        h = max(4, 2 * d)
        self.ff_ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h, d),
        )
        self.ff_drop = nn.Dropout(p)
        self.ff_gate_logit = nn.Parameter(torch.full((1, 1, d), logit))

    def forward(
        self,
        x: torch.Tensor,
        *,
        visual_tokens: torch.Tensor,
        query_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.ln_q(x)
        kv = self.ln_kv(visual_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        if query_keep_mask is not None:
            keep = query_keep_mask.unsqueeze(-1).to(dtype=attn_out.dtype)
            attn_out = attn_out * keep
        x = x + torch.sigmoid(self.cross_gate_logit) * self.cross_drop(attn_out)

        ff_out = self.ffn(self.ff_ln(x))
        if query_keep_mask is not None:
            keep = query_keep_mask.unsqueeze(-1).to(dtype=ff_out.dtype)
            ff_out = ff_out * keep
        x = x + torch.sigmoid(self.ff_gate_logit) * self.ff_drop(ff_out)
        return x


class MultimodalPrefixLM(nn.Module):
    def __init__(
        self,
        vision_adapter: VisionFeatureAdapter,
        bridge: nn.Module,
        lm: TransformerDecoderOnlyV1,
        *,
        lm_autocast: bool = True,
        prefix_calibration: bool = False,
        prefix_calib_layernorm: bool = True,
        prefix_calib_bias: bool = True,
        prefix_calib_gate_init: float = 1.0,
        prefix_geom_mlp_ratio: float = 0.0,
        prefix_geom_token_mixer_layers: int = 0,
        prefix_norm_target_ratio: float = 0.0,
        prefix_norm_reg_weight: float = 0.0,
        prefix_batchvar_reg_weight: float = 0.0,
        prefix_dropout: float = 0.0,
        question_context_mode: str = "all_text",
        question_prompt_prefix_len: int = 0,
        answer_prompt_prefix_len: int = 0,
        eval_use_kv_cache: bool = False,
        eval_kv_cache_mode: str = "batched",
        visual_feature_adapter_type: str = "none",
        visual_feature_adapter_hidden_dim: int = 0,
        visual_feature_adapter_dropout: float = 0.0,
        lm_visual_adapter_type: str = "none",
        lm_visual_adapter_layers: int = 0,
        lm_visual_adapter_num_heads: int = 8,
        lm_visual_adapter_dropout: float = 0.0,
        lm_visual_adapter_gate_init: float = 0.5,
    ):
        super().__init__()
        self.vision_adapter = vision_adapter
        self.bridge = bridge
        self.lm = lm
        self.lm_autocast = bool(lm_autocast)
        d_model = int(getattr(lm._config, "embed_size"))
        self.prefix_calibrator = PrefixCalibrator(
            d_model,
            enabled=bool(prefix_calibration),
            use_layernorm=bool(prefix_calib_layernorm),
            use_bias=bool(prefix_calib_bias),
            gate_init=float(prefix_calib_gate_init),
            geom_mlp_ratio=float(prefix_geom_mlp_ratio),
            geom_token_mixer_layers=int(prefix_geom_token_mixer_layers),
        )
        self.prefix_calibration = bool(prefix_calibration)
        self.prefix_norm_target_ratio = float(prefix_norm_target_ratio)
        self.prefix_norm_reg_weight = float(prefix_norm_reg_weight)
        self.prefix_batchvar_reg_weight = float(prefix_batchvar_reg_weight)
        self.prefix_dropout = max(0.0, float(prefix_dropout))
        self.question_context_mode = str(question_context_mode)
        self.question_prompt_prefix_len = max(0, int(question_prompt_prefix_len))
        self.answer_prompt_prefix_len = max(0, int(answer_prompt_prefix_len))
        self.eval_use_kv_cache = bool(eval_use_kv_cache)
        self.eval_kv_cache_mode = str(eval_kv_cache_mode)
        if self.eval_kv_cache_mode not in ("serial", "batched"):
            raise ValueError("eval_kv_cache_mode must be 'serial' or 'batched'.")
        self.visual_feature_adapter_type = str(visual_feature_adapter_type)
        if self.visual_feature_adapter_type not in ("none", "res_mlp"):
            raise ValueError(
                f"Unsupported visual_feature_adapter_type={self.visual_feature_adapter_type}. Supported: none, res_mlp"
            )
        self.visual_feature_adapter = (
            VisualFeatureResidualAdapter(
                hidden_dim=int(visual_feature_adapter_hidden_dim),
                dropout=float(visual_feature_adapter_dropout),
            )
            if self.visual_feature_adapter_type != "none"
            else nn.Identity()
        )
        self.lm_visual_adapter_type = str(lm_visual_adapter_type)
        self.lm_visual_adapter_layers = max(0, int(lm_visual_adapter_layers))
        self.visual_adapters = nn.ModuleDict()
        self.visual_adapter_layer_ids: List[int] = []
        if self.lm_visual_adapter_type not in ("none", "cross_attn"):
            raise ValueError(
                f"Unsupported lm_visual_adapter_type={self.lm_visual_adapter_type}. Supported: none, cross_attn"
            )
        if self.lm_visual_adapter_type != "none" and self.lm_visual_adapter_layers > 0:
            total_layers = len(self.lm._dec_blocks)
            top_n = max(0, min(self.lm_visual_adapter_layers, total_layers))
            self.visual_adapter_layer_ids = list(range(total_layers - top_n, total_layers))
            for layer_idx in self.visual_adapter_layer_ids:
                self.visual_adapters[str(layer_idx)] = ResidualVisualAdapter(
                    d_model,
                    num_heads=int(lm_visual_adapter_num_heads),
                    dropout=float(lm_visual_adapter_dropout),
                    gate_init=float(lm_visual_adapter_gate_init),
                )
        self.uses_visual_adapters = len(self.visual_adapter_layer_ids) > 0

    def _lm_autocast_dtype(self) -> Optional[torch.dtype]:
        if not self.lm_autocast:
            return None
        if not torch.cuda.is_available():
            return None
        if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _question_context_from_text(
        text_emb: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = context_mask.unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (text_emb * valid).sum(dim=1) / denom

    def _bridge_forward(
        self,
        bridge_input: torch.Tensor,
        question_context: Optional[torch.Tensor],
        question_tokens: Optional[torch.Tensor] = None,
        question_token_mask: Optional[torch.Tensor] = None,
        semantic_target_latents: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        if bool(getattr(self.bridge, "supports_question_context", False)) or bool(
            getattr(self.bridge, "supports_question_tokens", False)
        ):
            return self.bridge(
                bridge_input,
                question_context=question_context,
                question_tokens=question_tokens,
                question_token_mask=question_token_mask,
                semantic_target_latents=semantic_target_latents,
                return_attn=return_attn,
            )
        return self.bridge(bridge_input, return_attn=return_attn)

    @torch.no_grad()
    def compute_bridge_evidence(
        self,
        *,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        text_pad_mask: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_emb = self.lm._embed_dropout(self.lm._embed(input_ids))
        question_context, question_tokens, question_token_mask = self._compute_question_views(
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            prompt_mask=prompt_mask,
            question_mask=question_mask,
        )
        visual_features = self.vision_adapter(images)
        if self.visual_feature_adapter_type != "none":
            visual_features = self.visual_feature_adapter(visual_features)
        if hasattr(self.bridge, "forward_evidence"):
            return self.bridge.forward_evidence(
                visual_features,
                question_context=question_context,
                question_tokens=question_tokens,
                question_token_mask=question_token_mask,
            )
        return self._bridge_forward(
            visual_features,
            question_context,
            question_tokens=question_tokens,
            question_token_mask=question_token_mask,
        )

    def _compute_question_views(
        self,
        text_emb: torch.Tensor,
        text_pad_mask: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        supports_qctx = bool(getattr(self.bridge, "supports_question_context", False))
        supports_qtokens = bool(getattr(self.bridge, "supports_question_tokens", False))
        if not supports_qctx and not supports_qtokens:
            return None, None, None
        if self.question_context_mode == "question_only" and question_mask is not None:
            context_mask = question_mask & (~text_pad_mask)
        elif self.question_context_mode == "prompt_only" and prompt_mask is not None:
            context_mask = prompt_mask & (~text_pad_mask)
        else:
            context_mask = ~text_pad_mask
        question_context = None
        if supports_qctx:
            question_context = self._question_context_from_text(text_emb, context_mask)
        question_tokens = text_emb if supports_qtokens else None
        question_token_mask = context_mask if supports_qtokens else None
        return question_context, question_tokens, question_token_mask

    def _adapter_query_keep_mask(
        self,
        *,
        text_pad_mask: torch.Tensor,
        prefix_k: int,
    ) -> torch.Tensor:
        b = int(text_pad_mask.shape[0])
        prefix_keep = torch.zeros((b, int(prefix_k)), dtype=torch.bool, device=text_pad_mask.device)
        return torch.cat([prefix_keep, ~text_pad_mask], dim=1)

    def _compute_visual_prefix(
        self,
        *,
        images: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        text_pad_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        semantic_target_latents: Optional[torch.Tensor] = None,
        return_bridge_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        question_context: Optional[torch.Tensor] = None
        question_tokens: Optional[torch.Tensor] = None
        question_token_mask: Optional[torch.Tensor] = None
        if text_emb is not None and text_pad_mask is not None:
            question_context, question_tokens, question_token_mask = self._compute_question_views(
                text_emb=text_emb,
                text_pad_mask=text_pad_mask,
                prompt_mask=prompt_mask,
                question_mask=question_mask,
            )

        needs_visual = bool(getattr(self.bridge, "requires_visual_features", True))
        used_visual_features = visual_features
        if needs_visual:
            if used_visual_features is None:
                used_visual_features = self.vision_adapter(images)
                if self.visual_feature_adapter_type != "none":
                    used_visual_features = self.visual_feature_adapter(used_visual_features)
            visual_prefix = self._bridge_forward(
                used_visual_features,
                question_context,
                question_tokens=question_tokens,
                question_token_mask=question_token_mask,
                semantic_target_latents=semantic_target_latents,
                return_attn=return_bridge_attn,
            )
        else:
            # Image-agnostic bridge baselines only need batch size/device from images.
            visual_prefix = self._bridge_forward(
                images,
                question_context,
                question_tokens=question_tokens,
                question_token_mask=question_token_mask,
                semantic_target_latents=semantic_target_latents,
                return_attn=return_bridge_attn,
            )
        if bool(return_bridge_attn) and isinstance(visual_prefix, tuple):
            visual_prefix = visual_prefix[0]

        visual_prefix = self.prefix_calibrator(visual_prefix)
        if self.prefix_dropout > 0.0 and self.training:
            visual_prefix = F.dropout(visual_prefix, p=self.prefix_dropout, training=True)
        return visual_prefix, used_visual_features

    def _decode_hidden_with_visual_prefix(
        self,
        *,
        text_emb: torch.Tensor,
        text_pad_mask: torch.Tensor,
        visual_prefix: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        x = torch.cat([visual_prefix, text_emb], dim=1)
        b = int(text_emb.shape[0])
        k = int(visual_prefix.shape[1])
        prefix_pad = torch.zeros((b, k), dtype=torch.bool, device=text_emb.device)
        full_pad = torch.cat([prefix_pad, text_pad_mask], dim=1)
        if not self.uses_visual_adapters:
            hidden = self.lm._decode_only(
                x,
                pad_mask=full_pad,
                is_causal=bool(getattr(self.lm._config, "causal_lm", True)),
            )
            return hidden, k

        query_keep_mask = self._adapter_query_keep_mask(text_pad_mask=text_pad_mask, prefix_k=k)
        hidden = x
        is_causal = bool(getattr(self.lm._config, "causal_lm", True))
        for layer_idx, block in enumerate(self.lm._dec_blocks):
            hidden = block(
                hidden,
                pad_mask=full_pad,
                rope=self.lm._rope,
                is_causal=is_causal,
            )
            adapter_key = str(layer_idx)
            if adapter_key in self.visual_adapters:
                hidden = self.visual_adapters[adapter_key](
                    hidden,
                    visual_tokens=visual_prefix,
                    query_keep_mask=query_keep_mask,
                )
        return hidden, k

    def _decode_with_visual_prefix(
        self,
        *,
        input_ids: torch.Tensor,
        text_emb: torch.Tensor,
        text_pad_mask: torch.Tensor,
        visual_prefix: torch.Tensor,
        debug_shapes: bool = False,
        visual_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        use_lm_amp = (
            text_emb.device.type == "cuda"
            and str(getattr(self.lm._config, "attn_impl", "sdpa")) == "sdpa"
            and self._lm_autocast_dtype() is not None
        )
        if use_lm_amp:
            with torch.autocast(device_type="cuda", dtype=self._lm_autocast_dtype(), enabled=True):
                hidden, k = self._decode_hidden_with_visual_prefix(
                    text_emb=text_emb,
                    text_pad_mask=text_pad_mask,
                    visual_prefix=visual_prefix,
                )
                logits = self.lm._lm_head(hidden)
            logits = logits.float()
        else:
            hidden, k = self._decode_hidden_with_visual_prefix(
                text_emb=text_emb,
                text_pad_mask=text_pad_mask,
                visual_prefix=visual_prefix,
            )
            logits = self.lm._lm_head(hidden)
        if debug_shapes:
            if images is not None:
                print(f"[mm:shape] images={tuple(images.shape)}")
            if visual_features is None:
                print("[mm:shape] visual_features=<skipped by bridge>")
            else:
                print(f"[mm:shape] visual_features={tuple(visual_features.shape)}")
            print(f"[mm:shape] visual_prefix={tuple(visual_prefix.shape)}")
            print(f"[mm:shape] text_ids={tuple(input_ids.shape)} text_emb={tuple(text_emb.shape)}")
            print(f"[mm:shape] combined={(int(hidden.shape[0]), int(hidden.shape[1]), int(hidden.shape[2]))} logits={tuple(logits.shape)}")
        return logits, k

    def _lm_head_with_amp(self, hidden: torch.Tensor) -> torch.Tensor:
        use_lm_amp = (
            hidden.device.type == "cuda"
            and str(getattr(self.lm._config, "attn_impl", "sdpa")) == "sdpa"
            and self._lm_autocast_dtype() is not None
        )
        if use_lm_amp:
            with torch.autocast(device_type="cuda", dtype=self._lm_autocast_dtype(), enabled=True):
                logits = self.lm._lm_head(hidden)
            return logits.float()
        return self.lm._lm_head(hidden)

    def _prefill_decode_with_visual_prefix(
        self,
        *,
        text_emb: torch.Tensor,
        text_pad_mask: torch.Tensor,
        visual_prefix: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, Any]:
        if self.uses_visual_adapters:
            raise RuntimeError("KV-cache prefill is unsupported when lm_visual_adapter_type is enabled.")
        x = torch.cat([visual_prefix, text_emb], dim=1)
        b = int(text_emb.shape[0])
        k = int(visual_prefix.shape[1])
        prefix_pad = torch.zeros((b, k), dtype=torch.bool, device=text_emb.device)
        full_pad = torch.cat([prefix_pad, text_pad_mask], dim=1)
        hidden, kv_cache = self.lm._prefill_decode_only(
            x,
            pad_mask=full_pad,
            is_causal=bool(getattr(self.lm._config, "causal_lm", True)),
        )
        logits = self._lm_head_with_amp(hidden)
        return logits, k, kv_cache

    def _pack_generation_text_batch(
        self,
        ids: Sequence[Sequence[int]],
        *,
        prompt_lengths: Sequence[int],
        pad_id: int,
        device: torch.device,
        legacy_prompt_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(ids) != len(prompt_lengths):
            raise ValueError(f"Expected prompt_lengths len={len(ids)}, got {len(prompt_lengths)}")
        if not ids:
            empty_ids = torch.empty((0, 0), dtype=torch.long, device=device)
            empty_mask = torch.empty((0, 0), dtype=torch.bool, device=device)
            return empty_ids, empty_mask, empty_mask, empty_mask
        max_len = max(len(s) for s in ids)
        input_ids = torch.full((len(ids), max_len), int(pad_id), dtype=torch.long, device=device)
        text_pad_mask = torch.ones((len(ids), max_len), dtype=torch.bool, device=device)
        prompt_mask = torch.zeros((len(ids), max_len), dtype=torch.bool, device=device)
        question_mask = torch.zeros((len(ids), max_len), dtype=torch.bool, device=device)
        for i, s in enumerate(ids):
            if not s:
                continue
            t = torch.tensor(s, dtype=torch.long, device=device)
            seq_len = int(len(s))
            input_ids[i, :seq_len] = t
            text_pad_mask[i, :seq_len] = False
            prompt_len = seq_len if legacy_prompt_mask else min(int(prompt_lengths[i]), seq_len)
            if prompt_len > 0:
                prompt_mask[i, :prompt_len] = True
                q_start = min(self.question_prompt_prefix_len, prompt_len)
                q_end = max(q_start, prompt_len - self.answer_prompt_prefix_len)
                if q_end > q_start:
                    question_mask[i, q_start:q_end] = True
        return input_ids, text_pad_mask, prompt_mask, question_mask

    @staticmethod
    def _select_kv_cache_rows(
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        row_indices: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        row_idx = row_indices.to(dtype=torch.long)
        return [
            (
                wk.index_select(0, row_idx),
                wv.index_select(0, row_idx),
            )
            for wk, wv in kv_cache
        ]

    def _generate_answers_with_kv_cache_serial(
        self,
        *,
        seqs: List[List[int]],
        generated_ids: List[List[int]],
        done: List[bool],
        cached_visual_prefix: torch.Tensor,
        eos_id: int,
        max_new_tokens: int,
    ) -> List[List[int]]:
        device = cached_visual_prefix.device
        for i in range(len(seqs)):
            if done[i]:
                continue
            cur_ids = list(seqs[i])
            cur_len = int(len(cur_ids))
            single_input_ids = torch.tensor(cur_ids, dtype=torch.long, device=device).unsqueeze(0)
            single_text_pad_mask = torch.zeros((1, cur_len), dtype=torch.bool, device=device)
            single_text_emb = self.lm._embed_dropout(self.lm._embed(single_input_ids))
            single_visual_prefix = cached_visual_prefix[i : i + 1]
            logits_i, prefix_k_i, kv_cache = self._prefill_decode_with_visual_prefix(
                text_emb=single_text_emb,
                text_pad_mask=single_text_pad_mask,
                visual_prefix=single_visual_prefix,
            )
            next_id = int(torch.argmax(logits_i[0, prefix_k_i + cur_len - 1]).item())
            if next_id == int(eos_id):
                done[i] = True
                continue
            seqs[i].append(next_id)
            generated_ids[i].append(next_id)
            step_pos = int(prefix_k_i + cur_len)
            tokens_generated = 2

            while tokens_generated < int(max_new_tokens):
                step_input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
                step_text_emb = self.lm._embed_dropout(self.lm._embed(step_input_ids))
                hidden, kv_cache = self.lm._decode_only_incremental(
                    step_text_emb,
                    kv_cache=kv_cache,
                    pad_mask=None,
                    q_pad_mask=None,
                    start_pos=step_pos,
                )
                step_pos += 1
                next_id = int(torch.argmax(self._lm_head_with_amp(hidden)[0, 0]).item())
                if next_id == int(eos_id):
                    done[i] = True
                    break
                seqs[i].append(next_id)
                generated_ids[i].append(next_id)
                tokens_generated += 1
        return generated_ids

    def _generate_answers_with_kv_cache_batched(
        self,
        *,
        seqs: List[List[int]],
        prompt_lengths: Sequence[int],
        generated_ids: List[List[int]],
        done: List[bool],
        cached_visual_prefix: torch.Tensor,
        pad_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> List[List[int]]:
        device = cached_visual_prefix.device
        active_indices = [i for i, finished in enumerate(done) if not finished]
        if not active_indices:
            return generated_ids

        select_idx = torch.tensor(active_indices, dtype=torch.long, device=device)
        active_visual_prefix = cached_visual_prefix.index_select(0, select_idx)
        active_seqs = [list(seqs[i]) for i in active_indices]
        active_prompt_lengths = [int(prompt_lengths[i]) for i in active_indices]
        input_ids, text_pad_mask, _, _ = self._pack_generation_text_batch(
            active_seqs,
            prompt_lengths=active_prompt_lengths,
            pad_id=pad_id,
            device=device,
            legacy_prompt_mask=False,
        )
        text_emb = self.lm._embed_dropout(self.lm._embed(input_ids))
        logits, prefix_k, kv_cache = self._prefill_decode_with_visual_prefix(
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            visual_prefix=active_visual_prefix,
        )
        prefix_pad = torch.zeros((len(active_indices), prefix_k), dtype=torch.bool, device=device)
        cache_pad_mask = torch.cat([prefix_pad, text_pad_mask], dim=1)
        cur_lens = [int(len(s)) for s in active_seqs]

        continue_rows: List[int] = []
        continue_indices: List[int] = []
        continue_token_ids: List[int] = []
        continue_step_pos: List[int] = []
        for row_idx, global_idx in enumerate(active_indices):
            next_id = int(torch.argmax(logits[row_idx, prefix_k + cur_lens[row_idx] - 1]).item())
            if next_id == int(eos_id):
                done[global_idx] = True
                continue
            seqs[global_idx].append(next_id)
            generated_ids[global_idx].append(next_id)
            continue_rows.append(row_idx)
            continue_indices.append(global_idx)
            continue_token_ids.append(next_id)
            continue_step_pos.append(prefix_k + cur_lens[row_idx])

        if int(max_new_tokens) <= 2 or not continue_rows:
            return generated_ids

        keep_rows = torch.tensor(continue_rows, dtype=torch.long, device=device)
        kv_cache = self._select_kv_cache_rows(kv_cache, keep_rows)
        cache_pad_mask = cache_pad_mask.index_select(0, keep_rows)
        step_pos = torch.tensor(continue_step_pos, dtype=torch.long, device=device)
        active_indices = continue_indices
        step_input_ids = torch.tensor(continue_token_ids, dtype=torch.long, device=device).unsqueeze(1)
        tokens_generated = 2

        while tokens_generated < int(max_new_tokens) and active_indices:
            step_text_emb = self.lm._embed_dropout(self.lm._embed(step_input_ids))
            # Batched prefill keeps padded prompt slots in-cache; keep them masked
            # while letting newly appended generated tokens participate normally.
            step_pad_mask = torch.cat(
                [cache_pad_mask, torch.zeros((len(active_indices), 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            hidden, kv_cache = self.lm._decode_only_incremental(
                step_text_emb,
                kv_cache=kv_cache,
                pad_mask=step_pad_mask,
                q_pad_mask=None,
                start_pos=step_pos,
            )
            step_pos = step_pos + 1
            next_ids = torch.argmax(self._lm_head_with_amp(hidden)[:, 0], dim=-1)

            keep_rows_list: List[int] = []
            keep_indices: List[int] = []
            keep_token_ids: List[int] = []
            for row_idx, global_idx in enumerate(active_indices):
                next_id = int(next_ids[row_idx].item())
                if next_id == int(eos_id):
                    done[global_idx] = True
                    continue
                seqs[global_idx].append(next_id)
                generated_ids[global_idx].append(next_id)
                keep_rows_list.append(row_idx)
                keep_indices.append(global_idx)
                keep_token_ids.append(next_id)

            tokens_generated += 1
            if tokens_generated >= int(max_new_tokens) or not keep_rows_list:
                break

            keep_rows = torch.tensor(keep_rows_list, dtype=torch.long, device=device)
            kv_cache = self._select_kv_cache_rows(kv_cache, keep_rows)
            cache_pad_mask = step_pad_mask.index_select(0, keep_rows)
            step_pos = step_pos.index_select(0, keep_rows)
            active_indices = keep_indices
            step_input_ids = torch.tensor(keep_token_ids, dtype=torch.long, device=device).unsqueeze(1)
        return generated_ids

    def _generate_answers_with_kv_cache(
        self,
        *,
        seqs: List[List[int]],
        prompt_lengths: Sequence[int],
        cached_visual_prefix: torch.Tensor,
        pad_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> List[List[int]]:
        generated_ids: List[List[int]] = [[] for _ in seqs]
        input_ids, text_pad_mask, _, _ = self._pack_generation_text_batch(
            seqs,
            prompt_lengths=prompt_lengths,
            pad_id=pad_id,
            device=cached_visual_prefix.device,
            legacy_prompt_mask=False,
        )
        text_emb = self.lm._embed_dropout(self.lm._embed(input_ids))
        logits, prefix_k = self._decode_with_visual_prefix(
            input_ids=input_ids,
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            visual_prefix=cached_visual_prefix,
        )
        text_logits = logits[:, prefix_k:, :]
        done = [False for _ in seqs]
        for i in range(len(seqs)):
            next_id = int(torch.argmax(text_logits[i, len(seqs[i]) - 1]).item())
            if next_id == int(eos_id):
                done[i] = True
                continue
            seqs[i].append(next_id)
            generated_ids[i].append(next_id)

        if int(max_new_tokens) <= 1 or all(done):
            return generated_ids
        if str(getattr(self, "eval_kv_cache_mode", "batched")) == "serial":
            return self._generate_answers_with_kv_cache_serial(
                seqs=seqs,
                generated_ids=generated_ids,
                done=done,
                cached_visual_prefix=cached_visual_prefix,
                eos_id=eos_id,
                max_new_tokens=max_new_tokens,
            )
        return self._generate_answers_with_kv_cache_batched(
            seqs=seqs,
            prompt_lengths=prompt_lengths,
            generated_ids=generated_ids,
            done=done,
            cached_visual_prefix=cached_visual_prefix,
            pad_id=pad_id,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
        )

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        text_pad_mask: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        *,
        debug_shapes: bool = False,
        return_aux: bool = False,
        semantic_target_latents: Optional[torch.Tensor] = None,
        return_bridge_attn: bool = False,
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        text_emb = self.lm._embed_dropout(self.lm._embed(input_ids))
        visual_prefix, visual_features = self._compute_visual_prefix(
            images=images,
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            prompt_mask=prompt_mask,
            question_mask=question_mask,
            semantic_target_latents=semantic_target_latents,
            return_bridge_attn=return_bridge_attn,
        )
        logits, k = self._decode_with_visual_prefix(
            input_ids=input_ids,
            text_emb=text_emb,
            text_pad_mask=text_pad_mask,
            visual_prefix=visual_prefix,
            debug_shapes=debug_shapes,
            visual_features=visual_features,
            images=images,
        )
        if not return_aux:
            return logits, k
        aux = {
            "prefix_norm_mean": visual_prefix.norm(dim=-1).mean(),
            "text_norm_mean": text_emb.norm(dim=-1).mean(),
            "prefix_batch_variance_mean": visual_prefix.var(dim=0, unbiased=False).mean(),
        }
        bridge_aux = getattr(self.bridge, "last_aux_info", None)
        if isinstance(bridge_aux, dict):
            for key, value in bridge_aux.items():
                if isinstance(value, torch.Tensor):
                    aux[key] = value
        return logits, k, aux

    def compute_loss(
        self,
        batch: Dict[str, Any],
        *,
        answer_only: bool = True,
        debug_shapes: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        semantic_target_latents = None
        teacher_model = self.__dict__.get("semantic_teacher_model", None)
        if teacher_model is not None and bool(getattr(self.bridge.cfg, "semantic_bottleneck", False)):
            semantic_target_latents = teacher_model.compute_bridge_evidence(
                input_ids=batch["input_ids"],
                images=batch["images"],
                text_pad_mask=batch["text_pad_mask"],
                prompt_mask=batch.get("prompt_mask"),
                question_mask=batch.get("question_mask"),
            )
        logits, prefix_k, aux = self.forward_logits(
            input_ids=batch["input_ids"],
            images=batch["images"],
            text_pad_mask=batch["text_pad_mask"],
            prompt_mask=batch.get("prompt_mask"),
            question_mask=batch.get("question_mask"),
            debug_shapes=debug_shapes,
            return_aux=True,
            semantic_target_latents=semantic_target_latents,
            return_bridge_attn=bool(getattr(self, "use_grounding_loss", False)),
        )
        text_logits = logits[:, prefix_k:, :]
        if text_logits.shape[1] < 2:
            return text_logits.sum() * 0.0, {"loss_tokens": 0.0}

        next_logits = text_logits[:, :-1, :]
        targets = batch["input_ids"][:, 1:]
        ce = F.cross_entropy(next_logits.reshape(-1, next_logits.size(-1)), targets.reshape(-1), reduction="none")
        ce = ce.view_as(targets)

        valid = batch["target_mask"]
        if answer_only:
            valid = valid & batch["answer_loss_mask"]
        denom = valid.sum().clamp_min(1)
        ce_loss = (ce * valid.float()).sum() / denom
        loss = ce_loss

        info: Dict[str, float] = {
            "loss_tokens": float(denom.item()),
            "loss_ce": float(ce_loss.item()),
            "loss_vqa": float(ce_loss.item()),
            "prefix_norm_mean": float(aux["prefix_norm_mean"].item()),
            "text_norm_mean": float(aux["text_norm_mean"].item()),
            "prefix_batch_variance_mean": float(aux["prefix_batch_variance_mean"].item()),
            "semantic_bottleneck_enabled": float(aux.get("semantic_bottleneck_enabled", ce_loss.new_tensor(0.0)).item()),
            "semantic_token_count": float(aux.get("semantic_token_count", ce_loss.new_tensor(0.0)).item()),
            "semantic_latent_dim": float(aux.get("semantic_latent_dim", ce_loss.new_tensor(0.0)).item()),
            "semantic_target_token_count": float(aux.get("semantic_target_token_count", ce_loss.new_tensor(0.0)).item()),
            "semantic_teacher_enabled": 1.0 if teacher_model is not None else 0.0,
            "loss_grounding": 0.0,
            "loss_distill": 0.0,
            "grounding_mean_kl": 0.0,
            "compression_mean_attn_entropy": float(aux.get("compression_mean_attn_entropy", ce_loss.new_tensor(0.0)).item()),
        }

        if self.prefix_norm_reg_weight > 0.0 and self.prefix_norm_target_ratio > 0.0:
            ratio = aux["prefix_norm_mean"] / aux["text_norm_mean"].clamp_min(1e-8)
            reg = (ratio - float(self.prefix_norm_target_ratio)) ** 2
            loss = loss + float(self.prefix_norm_reg_weight) * reg
            info["loss_prefix_norm_reg"] = float(reg.item())
            info["prefix_text_norm_ratio"] = float(ratio.item())
        if self.prefix_batchvar_reg_weight > 0.0:
            reg_var = aux["prefix_batch_variance_mean"]
            loss = loss + float(self.prefix_batchvar_reg_weight) * reg_var
            info["loss_prefix_batchvar_reg"] = float(reg_var.item())
        semantic_recon = aux.get("semantic_recon_loss")
        semantic_consistency = aux.get("semantic_consistency_loss")
        semantic_recon_weight = float(getattr(self.bridge.cfg, "semantic_recon_loss_weight", 0.0))
        semantic_consistency_weight = float(getattr(self.bridge.cfg, "semantic_consistency_loss_weight", 0.0))
        if semantic_recon is not None and semantic_recon_weight > 0.0:
            loss = loss + semantic_recon_weight * semantic_recon
            info["loss_semantic_recon"] = float(semantic_recon.item())
            info["loss_distill"] = float(semantic_recon.item())
        if semantic_consistency is not None and semantic_consistency_weight > 0.0:
            loss = loss + semantic_consistency_weight * semantic_consistency
            info["loss_semantic_consistency"] = float(semantic_consistency.item())
        if bool(getattr(self, "use_grounding_loss", False)):
            perceiver_final_attn = aux.get("perceiver_final_attn")
            has_grounding_target = batch.get("has_grounding_target")
            grounding_soft_target = batch.get("grounding_soft_target")
            if (
                isinstance(perceiver_final_attn, torch.Tensor)
                and isinstance(has_grounding_target, torch.Tensor)
                and isinstance(grounding_soft_target, torch.Tensor)
            ):
                ground_mask = has_grounding_target.bool()
                if bool(ground_mask.any().item()):
                    attn = perceiver_final_attn[ground_mask].float()
                    attn_dist = attn.mean(dim=1).mean(dim=1)
                    attn_dist = attn_dist / attn_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    target = grounding_soft_target[ground_mask].float()
                    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    mean_kl = F.kl_div(attn_dist.clamp_min(1e-8).log(), target, reduction="batchmean")
                    ground_weight = float(getattr(self, "grounding_loss_weight", 0.0))
                    if ground_weight > 0.0:
                        loss = loss + ground_weight * mean_kl
                        info["loss_grounding"] = float(mean_kl.item())
                        info["grounding_mean_kl"] = float(mean_kl.item())

        info["loss_total"] = float(loss.item())
        return loss, info

    @torch.no_grad()
    def generate_answers(
        self,
        images: torch.Tensor,
        prompt_ids: Sequence[Sequence[int]],
        *,
        pad_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> List[List[int]]:
        seqs = [list(x) for x in prompt_ids]
        done = [False for _ in seqs]
        if not seqs:
            return []

        device = images.device
        prompt_lengths = [int(len(x)) for x in prompt_ids]
        legacy_prompt_conditioned = bool(
            getattr(self.bridge, "supports_question_context", False)
            and self.question_context_mode == "prompt_only"
        )

        needs_visual = bool(getattr(self.bridge, "requires_visual_features", True))
        supports_qcond = bool(getattr(self.bridge, "supports_question_context", False))
        supports_qtokens = bool(getattr(self.bridge, "supports_question_tokens", False))
        prompt_conditioned = (supports_qcond or supports_qtokens) and self.question_context_mode in (
            "prompt_only",
            "question_only",
        )

        cached_visual_features: Optional[torch.Tensor] = None
        if needs_visual:
            cached_visual_features = self.vision_adapter(images)
            if self.visual_feature_adapter_type != "none":
                cached_visual_features = self.visual_feature_adapter(cached_visual_features)

        cached_visual_prefix: Optional[torch.Tensor] = None
        if (not supports_qcond) or prompt_conditioned:
            prompt_input_ids, prompt_text_pad_mask, prompt_mask, question_mask = self._pack_generation_text_batch(
                prompt_ids,
                prompt_lengths=prompt_lengths,
                pad_id=pad_id,
                device=device,
                legacy_prompt_mask=False,
            )
            prompt_text_emb = self.lm._embed_dropout(self.lm._embed(prompt_input_ids))
            cached_visual_prefix, cached_visual_features = self._compute_visual_prefix(
                images=images,
                text_emb=prompt_text_emb,
                text_pad_mask=prompt_text_pad_mask,
                prompt_mask=prompt_mask,
                question_mask=question_mask,
                visual_features=cached_visual_features,
            )

        can_use_kv_cache = (
            bool(self.eval_use_kv_cache)
            and cached_visual_prefix is not None
            and not self.uses_visual_adapters
            and hasattr(self.lm, "_prefill_decode_only")
            and hasattr(self.lm, "_decode_only_incremental")
            and int(max_new_tokens) > 0
        )
        if can_use_kv_cache:
            assert cached_visual_prefix is not None
            # Preserve exact historical first-token behavior by using the original
            # mixed-length full-batch decode once before switching to cached continuation.
            return self._generate_answers_with_kv_cache(
                seqs=seqs,
                prompt_lengths=prompt_lengths,
                cached_visual_prefix=cached_visual_prefix,
                pad_id=pad_id,
                eos_id=eos_id,
                max_new_tokens=max_new_tokens,
            )

        for _ in range(int(max_new_tokens)):
            input_ids, text_pad_mask, prompt_mask, question_mask = self._pack_generation_text_batch(
                seqs,
                prompt_lengths=prompt_lengths,
                pad_id=pad_id,
                device=device,
                legacy_prompt_mask=legacy_prompt_conditioned,
            )
            text_emb = self.lm._embed_dropout(self.lm._embed(input_ids))

            visual_prefix = cached_visual_prefix
            if visual_prefix is None:
                visual_prefix, cached_visual_features = self._compute_visual_prefix(
                    images=images,
                    text_emb=text_emb,
                    text_pad_mask=text_pad_mask,
                    prompt_mask=prompt_mask,
                    question_mask=question_mask,
                    visual_features=cached_visual_features,
                )

            logits, prefix_k = self._decode_with_visual_prefix(
                input_ids=input_ids,
                text_emb=text_emb,
                text_pad_mask=text_pad_mask,
                visual_prefix=visual_prefix,
            )
            text_logits = logits[:, prefix_k:, :]

            for i in range(len(seqs)):
                if done[i]:
                    continue
                last_pos = len(seqs[i]) - 1
                next_id = int(torch.argmax(text_logits[i, last_pos]).item())
                if next_id == int(eos_id):
                    done[i] = True
                    continue
                seqs[i].append(next_id)
            if all(done):
                break

        out: List[List[int]] = []
        for i, s in enumerate(seqs):
            start = len(prompt_ids[i])
            out.append(s[start:])
        return out


class QACollator:
    def __init__(self, tokenizer: ByteBPETokenizer, max_q: int, max_a: int, max_text_tokens: int):
        self.tok = tokenizer
        self.max_q = int(max_q)
        self.max_a = int(max_a)
        self.max_text_tokens = int(max_text_tokens)
        self.q_prefix = tokenizer.encode("Question: ", add_bos=True, add_eos=False).tolist()
        self.a_prefix = tokenizer.encode("\nAnswer: ", add_bos=False, add_eos=False).tolist()

    def _encode_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        q_ids = self.tok.encode(sample["question"], add_bos=False, add_eos=False).tolist()[: self.max_q]
        a_text = sample.get("answer", "")
        a_ids = self.tok.encode(a_text, add_bos=False, add_eos=False).tolist()[: self.max_a] if a_text else []
        prompt_ids = self.q_prefix + q_ids + self.a_prefix
        full_ids = prompt_ids + a_ids + [self.tok.eos_id]

        if len(full_ids) > self.max_text_tokens:
            overflow = len(full_ids) - self.max_text_tokens
            if overflow > 0 and len(q_ids) > 0:
                keep_q = max(0, len(q_ids) - overflow)
                q_ids = q_ids[:keep_q]
            prompt_ids = self.q_prefix + q_ids + self.a_prefix
            full_ids = prompt_ids + a_ids + [self.tok.eos_id]
            if len(full_ids) > self.max_text_tokens:
                full_ids = full_ids[: self.max_text_tokens]

        answer_start = len(prompt_ids)
        has_answer = len(a_ids) > 0
        return {
            "input_ids": full_ids,
            "prompt_ids": prompt_ids,
            "answer_start": answer_start,
            "question_start": len(self.q_prefix),
            "question_end": len(self.q_prefix) + len(q_ids),
            "has_answer": has_answer,
        }

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        enc = [self._encode_sample(s) for s in samples]
        b = len(samples)
        max_len = max(len(x["input_ids"]) for x in enc) if enc else 1
        grounding_dim = 196
        for sample in samples:
            soft_target = sample.get("grounding_soft_target")
            if isinstance(soft_target, (list, tuple)) and soft_target:
                grounding_dim = int(len(soft_target))
                break
        input_ids = torch.full((b, max_len), int(self.tok.pad_id), dtype=torch.long)
        text_pad_mask = torch.ones((b, max_len), dtype=torch.bool)
        prompt_mask = torch.zeros((b, max_len), dtype=torch.bool)
        question_mask = torch.zeros((b, max_len), dtype=torch.bool)
        target_mask = torch.zeros((b, max_len - 1), dtype=torch.bool)
        answer_loss_mask = torch.zeros((b, max_len - 1), dtype=torch.bool)
        grounding_soft_target = torch.zeros((b, grounding_dim), dtype=torch.float32)
        has_grounding_target = torch.zeros((b,), dtype=torch.bool)

        for i, e in enumerate(enc):
            ids = e["input_ids"]
            L = len(ids)
            P = len(e["prompt_ids"])
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            text_pad_mask[i, :L] = False
            prompt_mask[i, :P] = True
            q_start = min(int(e["question_start"]), L)
            q_end = min(int(e["question_end"]), L)
            if q_end > q_start:
                question_mask[i, q_start:q_end] = True
            if L > 1:
                target_mask[i, : L - 1] = True
                if e["has_answer"]:
                    start = max(0, int(e["answer_start"]) - 1)
                    answer_loss_mask[i, start : L - 1] = True
            soft_target = samples[i].get("grounding_soft_target")
            if isinstance(soft_target, (list, tuple)) and len(soft_target) == grounding_dim:
                grounding_soft_target[i] = torch.tensor(soft_target, dtype=torch.float32)
                has_grounding_target[i] = bool(samples[i].get("has_grounding_target", False))

        return {
            "images": torch.stack([s["image"] for s in samples], dim=0),
            "input_ids": input_ids,
            "text_pad_mask": text_pad_mask,
            "prompt_mask": prompt_mask,
            "question_mask": question_mask,
            "target_mask": target_mask,
            "answer_loss_mask": answer_loss_mask,
            "grounding_soft_target": grounding_soft_target,
            "has_grounding_target": has_grounding_target,
            "prompt_ids": [e["prompt_ids"] for e in enc],
            "question_ids": [int(s["question_id"]) for s in samples],
            "image_ids": [int(s["image_id"]) for s in samples],
            "questions": [s["question"] for s in samples],
            "answers": [s["answer"] for s in samples],
            "all_answers_raw": [s.get("all_answers_raw", []) for s in samples],
            "all_answers": [s["all_answers"] for s in samples],
            "bbox_xyxy": [s.get("bbox_xyxy") for s in samples],
            "image_widths": [s.get("image_width") for s in samples],
            "image_heights": [s.get("image_height") for s in samples],
            "metadata": [s["metadata"] for s in samples],
        }


def build_loader(
    args: argparse.Namespace,
    tokenizer: ByteBPETokenizer,
    split: str,
    *,
    train_mode: bool,
    limit: int = 0,
) -> DataLoader:
    transform = build_image_transform(train_mode=train_mode)
    dataset_mix_raw = str(getattr(args, "dataset_mix", "") or "").strip()
    base_train_dataset: Any = None
    if bool(train_mode) and split == "train" and dataset_mix_raw:
        mix = parse_json_object_arg("--dataset_mix", dataset_mix_raw)
        base_train_dataset = MixedVQAv2Dataset(
            images_root=args.images_root,
            annotations_root=args.annotations_root,
            gqa_root=args.gqa_root,
            mix=mix,
            transform=transform,
            seed=int(args.seed),
            limit=limit,
            skip_missing_images=True,
        )
    elif bool(train_mode) and split == "train" and str(getattr(args, "gqa_root", "") or "").strip() and str(getattr(args, "images_root", "") or "").strip() == "__gqa_only__":
        base_train_dataset = GQADataset(
            gqa_root=args.gqa_root,
            split="train",
            transform=transform,
            limit=limit,
            skip_missing_images=True,
            question_group="",
        )
    elif split in ("gqa_train", "gqa_val"):
        gqa_split = "train" if split == "gqa_train" else "val"
        base_train_dataset = GQADataset(
            gqa_root=args.gqa_root,
            split=gqa_split,
            transform=transform,
            limit=limit,
            skip_missing_images=True,
            question_group=str(getattr(args, "gqa_eval_group", "") or "").strip().lower(),
        )
    else:
        base_train_dataset = VQAv2Dataset(
            images_root=args.images_root,
            annotations_root=args.annotations_root,
            split=split,
            transform=transform,
            limit=limit,
            skip_missing_images=True,
        )
    ds = base_train_dataset
    batch_sampler = None
    if bool(train_mode) and split == "train" and bool(getattr(args, "use_grounding_loss", False)):
        index_path = str(getattr(args, "pointing_index_path", "") or "").strip()
        if not index_path:
            raise SystemExit("--use_grounding_loss requires --pointing_index_path")
        pointing_ds = PointingIndexDataset(
            index_path=index_path,
            images_root=str(getattr(args, "images_root", "")),
            transform=transform,
            limit=0,
            skip_missing_images=True,
            target_len=196,
        )
        ds = ConcatDataset([pointing_ds, base_train_dataset])
        batch_sampler = GroundingMixBatchSampler(
            base_dataset=base_train_dataset,
            pointing_dataset=pointing_ds,
            batch_size=int(args.batch_size),
            pointing_mix_ratio=float(getattr(args, "pointing_mix_ratio", 0.25)),
            seed=int(args.seed),
            drop_last=True,
        )
    if (not train_mode) and int(limit) <= 0:
        eval_fraction = float(getattr(args, "eval_fraction", 1.0))
        if 0.0 < eval_fraction < 1.0:
            keep = max(1, int(math.ceil(float(len(ds)) * eval_fraction)))
            if hasattr(ds, "items"):
                ds.items = ds.items[:keep]
    if not train_mode:
        if hasattr(ds, "set_image_corruption"):
            ds.set_image_corruption(
                str(getattr(args, "eval_image_corruption", "none")),
                seed=int(getattr(args, "eval_image_corruption_seed", 123)),
            )
    collator = QACollator(
        tokenizer=tokenizer,
        max_q=args.max_question_length,
        max_a=args.max_answer_length,
        max_text_tokens=args.max_text_tokens,
    )
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(args.seed) + (11 if bool(train_mode) else 29))
    kwargs: Dict[str, Any] = {
        "num_workers": int(args.num_workers),
        "collate_fn": collator,
        "generator": loader_gen,
    }
    if train_mode and batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = (
            int(args.batch_size)
            if bool(train_mode) or int(getattr(args, "eval_batch_size", 0)) <= 0
            else int(args.eval_batch_size)
        )
        kwargs["drop_last"] = bool(train_mode)
        if train_mode:
            kwargs["sampler"] = EpochShuffleSampler(ds, seed=int(args.seed))
        else:
            kwargs["shuffle"] = False
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        kwargs["worker_init_fn"] = _seed_loader_worker
    if args.pin_memory:
        kwargs["pin_memory"] = True
    return DataLoader(ds, **kwargs)


def _trainable_param_count(model: nn.Module) -> Tuple[int, int]:
    n_all = 0
    n_tr = 0
    for p in model.parameters():
        n = int(p.numel())
        n_all += n
        if p.requires_grad:
            n_tr += n
    return n_tr, n_all


def _module_param_summary(module: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _vision_tail_modules(vision_model: nn.Module) -> List[nn.Module]:
    enc = getattr(vision_model, "_encoder", None)
    if enc is None:
        return []
    core_blocks = getattr(enc, "_core_blocks", None)
    if isinstance(core_blocks, (nn.Sequential, nn.ModuleList)):
        return list(core_blocks)
    enc_seq = getattr(enc, "_encoder", None)
    if isinstance(enc_seq, (nn.Sequential, nn.ModuleList)):
        return list(enc_seq)
    children = list(enc.children())
    return children if children else [enc]


def configure_freezing(model: MultimodalPrefixLM, args: argparse.Namespace) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    mode = str(args.freeze_mode)

    if mode == "semantic_adapter_only":
        semantic_mod = getattr(model.bridge, "semantic_bottleneck", None)
        if semantic_mod is not None:
            for p in semantic_mod.parameters():
                p.requires_grad_(True)
        for p in model.visual_adapters.parameters():
            p.requires_grad_(True)
    else:
        for p in model.bridge.parameters():
            p.requires_grad_(True)
        for p in model.prefix_calibrator.parameters():
            p.requires_grad_(True)
        for p in model.visual_feature_adapter.parameters():
            p.requires_grad_(True)
        for p in model.visual_adapters.parameters():
            p.requires_grad_(True)
    vision_tail_n = max(0, int(getattr(args, "train_vision_last_n_blocks", 0)))
    if vision_tail_n > 0:
        tail_modules = _vision_tail_modules(model.vision_adapter.vision_model)
        for mod in tail_modules[-vision_tail_n:]:
            for p in mod.parameters():
                p.requires_grad_(True)

    if mode == "bridge_only":
        return
    if mode == "bridge_plus_top_lm":
        n_layers = len(model.lm._dec_blocks)
        top_n = max(0, min(int(args.train_top_lm_layers), n_layers))
        if top_n > 0:
            for blk in model.lm._dec_blocks[-top_n:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        for p in model.lm._unembed.parameters():
            p.requires_grad_(True)
        return
    if mode == "semantic_adapter_only":
        return
    if mode == "full_finetune":
        for p in model.parameters():
            p.requires_grad_(True)
        return
    raise ValueError(f"Unsupported freeze mode: {mode}")


def _set_module_modes(model: MultimodalPrefixLM, freeze_mode: str) -> None:
    model.train()
    if freeze_mode in ("bridge_only", "bridge_plus_top_lm", "semantic_adapter_only"):
        model.vision_adapter.vision_model.eval()
        if int(getattr(model, "train_vision_last_n_blocks", 0)) > 0:
            for mod in _vision_tail_modules(model.vision_adapter.vision_model)[-int(model.train_vision_last_n_blocks):]:
                mod.train()
    if freeze_mode in ("bridge_only", "semantic_adapter_only"):
        model.lm.eval()


def _to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = dict(batch)
    for k in (
        "images",
        "input_ids",
        "text_pad_mask",
        "prompt_mask",
        "question_mask",
        "target_mask",
        "answer_loss_mask",
        "grounding_soft_target",
        "has_grounding_target",
    ):
        if k in batch:
            out[k] = batch[k].to(device)
    return out


def _materialize_lazy_params(model: MultimodalPrefixLM, batch: Dict[str, Any], device: str) -> None:
    with torch.no_grad():
        b = _to_device(batch, device)
        _ = model.forward_logits(
            input_ids=b["input_ids"],
            images=b["images"],
            text_pad_mask=b["text_pad_mask"],
            prompt_mask=b.get("prompt_mask"),
            question_mask=b.get("question_mask"),
            debug_shapes=False,
        )


def _checkpoint_path(run_id: str, step: int) -> str:
    return os.path.join(LOGDIR, run_id, f"step_{step}.tar")


def _lm_config_to_dict(cfg: LMConfig) -> Dict[str, Any]:
    return {k: v for k, v in cfg.__dict__.items() if not str(k).startswith("_")}


def _maybe_initialize_bridge_from_state_dict(model: MultimodalPrefixLM, state_dict: Dict[str, Any]) -> None:
    bridge = model.bridge
    if not hasattr(bridge, "_ensure_built"):
        return
    if getattr(bridge, "_input_dim", None) is not None:
        return

    w_tok = state_dict.get("bridge._token_proj.0.weight")
    w_glb = state_dict.get("bridge._global_proj.0.weight")
    has_tok = torch.is_tensor(w_tok) and w_tok.ndim == 2
    has_glb = torch.is_tensor(w_glb) and w_glb.ndim == 2
    if not has_tok and not has_glb:
        return

    ref = model.lm._embed.weight
    if has_tok and hasattr(bridge, "_ensure_token_built"):
        bridge._ensure_token_built(int(w_tok.shape[1]), device=ref.device, dtype=ref.dtype)
    if has_glb and hasattr(bridge, "_ensure_global_built"):
        bridge._ensure_global_built(int(w_glb.shape[1]), device=ref.device, dtype=ref.dtype)
    if (has_tok and has_glb) and hasattr(bridge, "_ensure_built"):
        # Older checkpoints may contain both branches; ensure both are present.
        bridge._ensure_built(int(w_tok.shape[1]), device=ref.device, dtype=ref.dtype)


def load_model_weights_from_mm_checkpoint(
    model: MultimodalPrefixLM,
    checkpoint_path: str,
    *,
    logger: Optional[Logger] = None,
) -> Dict[str, Any]:
    payload = _load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = payload["model_state_dict"]
    _maybe_initialize_bridge_from_state_dict(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if logger is not None:
        logger.log(f"[mm] initialized model weights from {checkpoint_path}")
        if unexpected:
            logger.log(f"[mm] WARNING: unexpected init keys: {unexpected}")
        if missing:
            logger.log(
                f"[mm] note: {len(missing)} init keys missing (newly initialized): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
    return payload


def attach_semantic_teacher(
    model: MultimodalPrefixLM,
    *,
    checkpoint_path: str,
    device: str,
    logger: Optional[Logger] = None,
) -> None:
    teacher_model, _teacher_tok, _teacher_bridge_cfg, _teacher_payload, _teacher_args = load_runtime_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    # Avoid registering the teacher as a trainable submodule in the student state_dict.
    model.__dict__["semantic_teacher_model"] = teacher_model
    if logger is not None:
        logger.log(f"[mm] semantic teacher attached from {checkpoint_path}")


def save_mm_checkpoint(
    path: str,
    model: MultimodalPrefixLM,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    args: argparse.Namespace,
    bridge_cfg: BridgeConfig,
) -> None:
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "train_args": vars(args),
        "tokenizer_path": str(args.tokenizer_path),
        "lm_config": _lm_config_to_dict(model.lm._config),
        "bridge_config": asdict(bridge_cfg),
        "vision_meta": {
            "vision_model": args.vision_model,
            "vision_latent_dim": args.vision_latent_dim,
            "vision_cbld": args.vision_cbld,
            "feature_mode": args.vision_feature_mode,
            "feature_source": args.vision_feature_source,
        },
    }
    torch.save(payload, path)


def build_runtime_from_args(
    args: argparse.Namespace,
    device: str,
    *,
    checkpoint_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[MultimodalPrefixLM, ByteBPETokenizer, BridgeConfig]:
    args = _apply_runtime_defaults(args)
    tokenizer_path = resolve_tokenizer_path(args, checkpoint_payload=checkpoint_payload)
    args.tokenizer_path = tokenizer_path
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    question_prompt_prefix_len = len(tokenizer.encode("Question: ", add_bos=True, add_eos=False).tolist())
    answer_prompt_prefix_len = len(tokenizer.encode("\nAnswer: ", add_bos=False, add_eos=False).tolist())

    vision_device = device if str(args.vision_device) == "auto" else resolve_device(str(args.vision_device))
    vision = build_vision_model_from_args(args, device=vision_device, ckpt_payload=checkpoint_payload)
    lm = build_lm_from_args(args, tokenizer=tokenizer, device=device, ckpt_payload=checkpoint_payload)
    feature_mode = args.vision_feature_mode
    feature_source = args.vision_feature_source
    if checkpoint_payload is not None:
        meta = checkpoint_payload.get("vision_meta", {}) or {}
        feature_mode = str(meta.get("feature_mode", feature_mode))
        feature_source = str(meta.get("feature_source", feature_source))
    force_no_grad = not (
        str(getattr(args, "freeze_mode", "bridge_only")) == "full_finetune"
        or int(getattr(args, "train_vision_last_n_blocks", 0)) > 0
    )
    vision_adapter = VisionFeatureAdapter(
        vision_model=vision,
        feature_mode=feature_mode,
        feature_source=feature_source,
        vision_device=vision_device,
        output_device=device,
        force_no_grad=force_no_grad,
    )

    bcfg_data = {
        "bridge_type": args.bridge_type,
        "num_visual_tokens": args.num_visual_tokens,
        "lm_hidden_size": lm._config.embed_size,
        "bridge_hidden_dim": args.bridge_hidden_dim,
        "input_feature_mode": args.vision_feature_mode,
        "token_reduce": args.bridge_token_reduce,
        "learned_init_std": args.bridge_learned_init_std,
        "add_2d_pos_emb": bool(args.bridge_add_2d_pos_emb),
        "bridge_num_heads": int(args.bridge_num_heads),
        "bridge_attn_dropout": float(args.bridge_attn_dropout),
        "bridge_query_depth": int(args.bridge_query_depth),
        "bridge_refine_layers": int(args.bridge_refine_layers),
        "bridge_pre_mixer_type": str(args.bridge_pre_mixer_type),
        "bridge_pre_mixer_layers": int(args.bridge_pre_mixer_layers),
        "bridge_pre_mixer_kernel_size": int(args.bridge_pre_mixer_kernel_size),
        "bridge_hybrid_alpha_mode": str(args.bridge_hybrid_alpha_mode),
        "bridge_hybrid_alpha_init": float(args.bridge_hybrid_alpha_init),
        "bridge_hybrid_image_bridge_type": str(args.bridge_hybrid_image_bridge_type),
        "bridge_question_conditioning": bool(args.bridge_question_conditioning),
        "bridge_qcond_scale": float(args.bridge_qcond_scale),
        "bridge_query_bank_mode": str(args.bridge_query_bank_mode),
        "bridge_qquery_basis_count": int(args.bridge_qquery_basis_count),
        "bridge_qquery_scale": float(args.bridge_qquery_scale),
        "bridge_qquery_multi_count": int(args.bridge_qquery_multi_count),
        "bridge_qquery_hybrid_gate_init": float(args.bridge_qquery_hybrid_gate_init),
        "bridge_query_role_specialization": bool(args.bridge_query_role_specialization),
        "bridge_question_context_mode": str(args.bridge_question_context_mode),
        "bridge_iterative_qquery_steps": int(args.bridge_iterative_qquery_steps),
        "bridge_iterative_qquery_residual_scale": float(args.bridge_iterative_qquery_residual_scale),
        "bridge_token_selector_type": str(args.bridge_token_selector_type),
        "bridge_token_select_k": int(args.bridge_token_select_k),
        "bridge_token_select_k_min": int(args.bridge_token_select_k_min),
        "bridge_num_roles": int(args.bridge_num_roles),
        "bridge_evidence_topk": int(args.bridge_evidence_topk),
        "semantic_bottleneck": bool(getattr(args, "use_compression", False) or args.semantic_bottleneck),
        "semantic_tokens": int(getattr(args, "compression_k", args.semantic_tokens)),
        "semantic_latent_dim": int(args.semantic_latent_dim),
        "semantic_recon_loss_weight": float(getattr(args, "compression_distill_weight", args.semantic_recon_loss_weight)),
        "semantic_consistency_loss_weight": float(args.semantic_consistency_loss_weight),
        "semantic_token_schedule": str(args.semantic_token_schedule),
    }
    if checkpoint_payload is not None and isinstance(checkpoint_payload.get("bridge_config"), dict):
        bcfg_data.update(dict(checkpoint_payload["bridge_config"]))
        bcfg_data["lm_hidden_size"] = int(lm._config.embed_size)
    else:
        # Only apply legacy compression aliases when we are not reconstructing
        # the bridge directly from a checkpoint payload. Otherwise the parser
        # defaults (for example compression_k=16) can silently clobber the true
        # checkpoint bottleneck width during eval/diagnostic reload.
        bcfg_data["semantic_bottleneck"] = bool(
            getattr(args, "use_compression", False) or bcfg_data.get("semantic_bottleneck", False)
        )
        bcfg_data["semantic_tokens"] = int(getattr(args, "compression_k", bcfg_data.get("semantic_tokens", 16)))
        bcfg_data["semantic_recon_loss_weight"] = float(
            getattr(args, "compression_distill_weight", bcfg_data.get("semantic_recon_loss_weight", 0.0))
        )
    bridge_cfg = BridgeConfig(**bcfg_data)
    bridge = build_bridge(bridge_cfg).to(device)
    model = MultimodalPrefixLM(
        vision_adapter=vision_adapter,
        bridge=bridge,
        lm=lm,
        lm_autocast=bool(args.mm_lm_autocast),
        prefix_calibration=bool(args.prefix_calibration),
        prefix_calib_layernorm=bool(args.prefix_calib_layernorm),
        prefix_calib_bias=bool(args.prefix_calib_bias),
        prefix_calib_gate_init=float(args.prefix_calib_gate_init),
        prefix_geom_mlp_ratio=float(args.prefix_geom_mlp_ratio),
        prefix_geom_token_mixer_layers=int(args.prefix_geom_token_mixer_layers),
        prefix_norm_target_ratio=float(args.prefix_norm_target_ratio),
        prefix_norm_reg_weight=float(args.prefix_norm_reg_weight),
        prefix_batchvar_reg_weight=float(args.prefix_batchvar_reg_weight),
        prefix_dropout=float(args.prefix_dropout),
        question_context_mode=str(args.bridge_question_context_mode),
        question_prompt_prefix_len=question_prompt_prefix_len,
        answer_prompt_prefix_len=answer_prompt_prefix_len,
        eval_use_kv_cache=bool(getattr(args, "eval_use_kv_cache", False)),
        eval_kv_cache_mode=str(getattr(args, "eval_kv_cache_mode", "batched")),
        visual_feature_adapter_type=str(getattr(args, "visual_feature_adapter_type", "none")),
        visual_feature_adapter_hidden_dim=int(getattr(args, "visual_feature_adapter_hidden_dim", 0)),
        visual_feature_adapter_dropout=float(getattr(args, "visual_feature_adapter_dropout", 0.0)),
        lm_visual_adapter_type=str(getattr(args, "lm_visual_adapter_type", "none")),
        lm_visual_adapter_layers=int(getattr(args, "lm_visual_adapter_layers", 0)),
        lm_visual_adapter_num_heads=int(getattr(args, "lm_visual_adapter_num_heads", 8)),
        lm_visual_adapter_dropout=float(getattr(args, "lm_visual_adapter_dropout", 0.0)),
        lm_visual_adapter_gate_init=float(getattr(args, "lm_visual_adapter_gate_init", 0.5)),
    ).to(device)
    model.train_vision_last_n_blocks = max(0, int(getattr(args, "train_vision_last_n_blocks", 0)))
    model.use_grounding_loss = bool(getattr(args, "use_grounding_loss", False))
    model.grounding_loss_weight = float(getattr(args, "grounding_loss_weight", 0.0))
    if hasattr(model.bridge, "eval_bypass_compression"):
        model.bridge.eval_bypass_compression = bool(getattr(args, "eval_bypass_compression", False))
    if str(vision_device) != str(device):
        model.vision_adapter.vision_model.to(vision_device)
    return model, tokenizer, bridge_cfg


def load_runtime_from_checkpoint(
    checkpoint_path: str,
    device: str,
    args_override: Optional[Dict[str, Any]] = None,
) -> Tuple[MultimodalPrefixLM, ByteBPETokenizer, BridgeConfig, Dict[str, Any], argparse.Namespace]:
    payload = _load_checkpoint(checkpoint_path, map_location="cpu")
    train_args = dict(payload.get("train_args", {}) or {})
    if args_override:
        for k, v in args_override.items():
            if v is not None:
                train_args[k] = v
    args = SimpleNamespace(**train_args)
    args = _apply_runtime_defaults(args)
    model, tokenizer, bridge_cfg = build_runtime_from_args(args, device=device, checkpoint_payload=payload)
    _maybe_initialize_bridge_from_state_dict(model, payload["model_state_dict"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, tokenizer, bridge_cfg, payload, argparse.Namespace(**vars(args))


@torch.no_grad()
def run_generation_predictions(
    model: MultimodalPrefixLM,
    loader: DataLoader,
    tokenizer: ByteBPETokenizer,
    device: str,
    *,
    max_answer_length: int,
    max_batches: int = 0,
    debug_shapes: bool = False,
    logger: Optional[Logger] = None,
    split_name: str = "eval",
    log_every: int = 10,
    cuda_empty_cache_every: int = 0,
) -> List[Dict[str, Any]]:
    model.eval()
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for bidx, batch in enumerate(loader):
        batch = _to_device(batch, device)
        gens = model.generate_answers(
            images=batch["images"],
            prompt_ids=batch["prompt_ids"],
            pad_id=tokenizer.pad_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_answer_length,
        )
        if debug_shapes and bidx == 0:
            _ = model.forward_logits(
                input_ids=batch["input_ids"],
                images=batch["images"],
                text_pad_mask=batch["text_pad_mask"],
                debug_shapes=True,
            )
        for i in range(len(gens)):
            pred = tokenizer.decode(gens[i], skip_special=True).strip()
            records.append(
                {
                    "question_id": int(batch["question_ids"][i]),
                    "image_id": int(batch["image_ids"][i]),
                    "question": batch["questions"][i],
                    "prediction": pred,
                    "canonical_answer": batch["answers"][i],
                    "all_answers_raw": batch["all_answers_raw"][i],
                    "all_answers": batch["all_answers"][i],
                    "metadata": batch["metadata"][i],
                }
            )
        if int(log_every) > 0 and ((bidx + 1) % int(log_every) == 0 or (bidx == 0)):
            elapsed = max(1e-6, float(time.time() - t0))
            msg = (
                f"[eval:{split_name}] batch={bidx + 1}"
                + (f"/{max_batches}" if int(max_batches) > 0 else "")
                + f" samples={len(records)} elapsed_s={elapsed:.1f} steps_per_s={(float(bidx + 1) / elapsed):.2f}"
            )
            if logger is not None:
                logger.log(msg)
            else:
                print(msg)
        if int(cuda_empty_cache_every) > 0 and ((bidx + 1) % int(cuda_empty_cache_every) == 0):
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                if logger is not None:
                    logger.log(f"[mm] cuda empty_cache during eval batch={bidx + 1}")
        if max_batches > 0 and (bidx + 1) >= int(max_batches):
            break
    return records


@torch.no_grad()
def run_final_greedy_sanity_pass(
    model: MultimodalPrefixLM,
    loader: DataLoader,
    tokenizer: ByteBPETokenizer,
    device: str,
    *,
    max_answer_length: int,
    sanity_count: int,
    logger: Logger,
    tag: str,
) -> None:
    dataset = getattr(loader, "dataset", None)
    collate_fn = getattr(loader, "collate_fn", None)
    if dataset is None or collate_fn is None:
        logger.log(f"[mm:sanity:{tag}] skipped (missing dataset/collate_fn)")
        return
    if len(dataset) <= 0:
        logger.log(f"[mm:sanity:{tag}] skipped (empty eval dataset)")
        return

    requested = int(sanity_count)
    if requested <= 0:
        logger.log(f"[mm:sanity:{tag}] disabled (--final_sanity_count={requested})")
        return

    n_prompts = max(FINAL_SANITY_MIN_PROMPTS, min(FINAL_SANITY_MAX_PROMPTS, requested))
    if requested != n_prompts:
        logger.log(f"[mm:sanity:{tag}] clamped prompt count from {requested} to {n_prompts}")

    samples: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(dataset) and len(samples) < n_prompts:
        try:
            samples.append(dataset[idx])
        except Exception as e:
            logger.log(f"[mm:sanity:{tag}] skipping sample idx={idx} due to load error: {e}")
        idx += 1

    if not samples:
        logger.log(f"[mm:sanity:{tag}] skipped (no loadable samples)")
        return

    batch = collate_fn(samples)
    batch = _to_device(batch, device)
    model.eval()
    gens = model.generate_answers(
        images=batch["images"],
        prompt_ids=batch["prompt_ids"],
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        max_new_tokens=int(max_answer_length),
    )
    logger.log(f"[mm:sanity:{tag}] greedy decode pass count={len(gens)}")
    for i in range(len(gens)):
        qid = int(batch["question_ids"][i])
        question = str(batch["questions"][i]).replace("\n", " ").strip()
        pred = tokenizer.decode(gens[i], skip_special=True).strip()
        logger.log(
            f"[mm:sanity:{tag}] #{i + 1} qid={qid} question={question!r} prediction={pred!r}"
        )


def run_predictions_from_checkpoint(
    checkpoint_path: str,
    *,
    split: str = "val",
    device: str = "auto",
    batch_size: int = 16,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    images_root: Optional[str] = None,
    annotations_root: Optional[str] = None,
    limit: int = 0,
    max_batches: int = 0,
    max_answer_length: Optional[int] = None,
    debug_shapes: bool = False,
    progress_every: int = 10,
) -> List[Dict[str, Any]]:
    dev = resolve_device(device)
    overrides = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "images_root": images_root,
        "annotations_root": annotations_root,
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=dev,
        args_override=overrides,
    )
    split_limit = int(limit) if int(limit) > 0 else 0
    loader_args = argparse.Namespace(**vars(run_args))
    if images_root:
        loader_args.images_root = images_root
    if annotations_root:
        loader_args.annotations_root = annotations_root
    loader_args.batch_size = int(batch_size)
    loader_args.num_workers = int(num_workers)
    loader_args.prefetch_factor = int(prefetch_factor)
    loader_args.pin_memory = bool(pin_memory)
    loader = build_loader(loader_args, tokenizer=tokenizer, split=split, train_mode=False, limit=split_limit)
    max_answer_len = int(max_answer_length) if max_answer_length is not None else int(loader_args.max_answer_length)
    return run_generation_predictions(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=dev,
        max_answer_length=max_answer_len,
        max_batches=max_batches,
        debug_shapes=debug_shapes,
        logger=None,
        split_name=split,
        log_every=progress_every,
    )


def _probe_answer_vocab(dataset: Any, top_k: int) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in getattr(dataset, "items", []):
        ans = str(item.get("answer", "")).strip()
        if ans:
            counts[ans] += 1
    return {ans: idx for idx, (ans, _n) in enumerate(counts.most_common(max(1, int(top_k))))}


@torch.no_grad()
def _extract_probe_features(
    model: MultimodalPrefixLM,
    loader: DataLoader,
    *,
    device: str,
    answer_to_idx: Dict[str, int],
    feature_pool: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    model.eval()
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    answer_types: List[str] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        text_emb = model.lm._embed_dropout(model.lm._embed(batch["input_ids"]))
        prefix, _ = model._compute_visual_prefix(
            images=batch["images"],
            text_emb=text_emb,
            text_pad_mask=batch["text_pad_mask"],
            prompt_mask=batch.get("prompt_mask"),
            question_mask=batch.get("question_mask"),
        )
        if str(feature_pool) == "mean":
            x = prefix.mean(dim=1)
        else:
            x = prefix.reshape(int(prefix.shape[0]), -1)
        for i in range(int(x.shape[0])):
            answer = str(batch["answers"][i]).strip()
            idx = answer_to_idx.get(answer)
            if idx is None:
                continue
            feats.append(x[i].detach().cpu().to(dtype=torch.float16))
            labels.append(torch.tensor(idx, dtype=torch.long))
            answer_types.append(str(batch["metadata"][i].get("answer_type", "other")))
    if not feats:
        raise RuntimeError("No usable tiny-head samples after answer-vocab filtering.")
    return torch.stack(feats, dim=0), torch.stack(labels, dim=0), answer_types


def _eval_probe_classifier(
    probe: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    answer_types: List[str],
) -> Dict[str, Any]:
    probe.eval()
    total = 0
    correct = 0
    by_type_total: Dict[str, int] = defaultdict(int)
    by_type_correct: Dict[str, int] = defaultdict(int)
    offset = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)
            pred = torch.argmax(probe(xb), dim=-1)
            match = (pred == yb).detach().cpu()
            for i in range(int(match.shape[0])):
                atype = str(answer_types[offset + i])
                by_type_total[atype] += 1
                if bool(match[i].item()):
                    by_type_correct[atype] += 1
            correct += int(match.sum().item())
            total += int(match.numel())
            offset += int(match.shape[0])
    out = {
        "accuracy": (float(correct) / float(total)) if total else 0.0,
        "count": int(total),
        "by_answer_type": {},
    }
    for key in sorted(by_type_total):
        denom = max(1, int(by_type_total[key]))
        out["by_answer_type"][key] = float(by_type_correct[key]) / float(denom)
    return out


def run_tiny_head_probe_eval(
    *,
    model: MultimodalPrefixLM,
    tokenizer: ByteBPETokenizer,
    args: argparse.Namespace,
    device: str,
    logger: Logger,
) -> None:
    probe_args = argparse.Namespace(**vars(args))
    probe_args.eval_batch_size = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)
    train_loader = build_loader(
        probe_args,
        tokenizer=tokenizer,
        split="train",
        train_mode=False,
        limit=max(0, int(args.limit_train)),
    )
    val_loader = build_loader(
        probe_args,
        tokenizer=tokenizer,
        split="val",
        train_mode=False,
        limit=max(0, int(args.limit_eval) if int(args.limit_eval) > 0 else int(args.limit_val)),
    )
    counts: Counter[str] = Counter()
    for ds in (train_loader.dataset, val_loader.dataset):
        for item in getattr(ds, "items", []):
            ans = str(item.get("answer", "")).strip()
            if ans:
                counts[ans] += 1
    answer_to_idx = {ans: idx for idx, (ans, _n) in enumerate(counts.most_common(max(1, int(args.tiny_head_answer_top_k))))}
    if not answer_to_idx:
        raise RuntimeError("Tiny-head probe answer vocabulary is empty.")
    train_x, train_y, _ = _extract_probe_features(
        model,
        train_loader,
        device=device,
        answer_to_idx=answer_to_idx,
        feature_pool=str(args.tiny_head_feature_pool),
    )
    val_x, val_y, val_types = _extract_probe_features(
        model,
        val_loader,
        device=device,
        answer_to_idx=answer_to_idx,
        feature_pool=str(args.tiny_head_feature_pool),
    )
    probe = nn.Linear(int(train_x.shape[1]), len(answer_to_idx)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_probe_loader = DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=256, shuffle=True)
    val_probe_loader = DataLoader(torch.utils.data.TensorDataset(val_x, val_y), batch_size=256, shuffle=False)
    best_summary: Optional[Dict[str, Any]] = None
    for epoch in range(1, max(1, int(args.tiny_head_epochs)) + 1):
        probe.train()
        for xb, yb in train_probe_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)
            loss = loss_fn(probe(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        summary = _eval_probe_classifier(probe, val_probe_loader, device=device, answer_types=val_types)
        summary["epoch"] = int(epoch)
        best_summary = summary
        logger.log(f"[tiny-head] epoch={epoch} accuracy={float(summary['accuracy']):.4f} count={int(summary['count'])}")
    if best_summary is not None:
        logger.log(f"[tiny-head] best_accuracy={float(best_summary['accuracy']):.4f} by_answer_type={json.dumps(best_summary['by_answer_type'], sort_keys=True)}")


def _bbox_to_grid_mask(
    bbox_xyxy: Sequence[float],
    *,
    image_width: float,
    image_height: float,
    grid_side: int = 14,
) -> torch.Tensor:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return torch.zeros((grid_side * grid_side,), dtype=torch.float32)
    cell_w = float(image_width) / float(grid_side)
    cell_h = float(image_height) / float(grid_side)
    mask = torch.zeros((grid_side, grid_side), dtype=torch.float32)
    for gy in range(grid_side):
        cy = (float(gy) + 0.5) * cell_h
        if cy < y1 or cy > y2:
            continue
        for gx in range(grid_side):
            cx = (float(gx) + 0.5) * cell_w
            if x1 <= cx <= x2:
                mask[gy, gx] = 1.0
    return mask.reshape(-1)


@torch.no_grad()
def run_grounding_eval(
    *,
    model: MultimodalPrefixLM,
    tokenizer: ByteBPETokenizer,
    args: argparse.Namespace,
    device: str,
    logger: Logger,
) -> None:
    index_path = str(getattr(args, "eval_grounding_index_path", "") or getattr(args, "pointing_index_path", "") or "").strip()
    if not index_path:
        raise SystemExit("--eval_grounding requires --eval_grounding_index_path or --pointing_index_path")
    dataset = PointingIndexDataset(
        index_path=index_path,
        images_root=str(args.images_root),
        transform=build_image_transform(train_mode=False),
        limit=max(0, int(args.limit_eval)),
        skip_missing_images=True,
        target_len=196,
    )
    collator = QACollator(
        tokenizer=tokenizer,
        max_q=args.max_question_length,
        max_a=args.max_answer_length,
        max_text_tokens=args.max_text_tokens,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collator,
        pin_memory=bool(args.pin_memory),
    )
    model.eval()
    total = 0
    mass_sum = 0.0
    entropy_sum = 0.0
    entropy_count = 0
    for bidx, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        _ = model.forward_logits(
            input_ids=batch["input_ids"],
            images=batch["images"],
            text_pad_mask=batch["text_pad_mask"],
            prompt_mask=batch.get("prompt_mask"),
            question_mask=batch.get("question_mask"),
            return_aux=True,
            return_bridge_attn=True,
        )
        aux = getattr(model.bridge, "last_aux_info", {})
        perceiver_attn = aux.get("perceiver_final_attn")
        if not isinstance(perceiver_attn, torch.Tensor):
            raise RuntimeError("Grounding eval requires perceiver_final_attn in bridge aux.")
        attn_dist = perceiver_attn.float().mean(dim=1).mean(dim=1)
        attn_dist = attn_dist / attn_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        for i in range(int(attn_dist.shape[0])):
            bbox = raw_batch["bbox_xyxy"][i]
            width = raw_batch["image_widths"][i]
            height = raw_batch["image_heights"][i]
            if bbox is None or width in (None, 0) or height in (None, 0):
                continue
            mask = _bbox_to_grid_mask(bbox, image_width=float(width), image_height=float(height), grid_side=14).to(attn_dist.device)
            if float(mask.sum().item()) <= 0.0:
                continue
            mass_sum += float((attn_dist[i] * mask).sum().item())
            total += 1
        bottleneck_entropy = aux.get("compression_mean_attn_entropy")
        if isinstance(bottleneck_entropy, torch.Tensor):
            entropy_sum += float(bottleneck_entropy.item())
            entropy_count += 1
        if int(args.eval_batches) > 0 and (bidx + 1) >= int(args.eval_batches):
            break
    logger.log(
        f"[grounding-eval] samples={total} mean_mass_in_box={(mass_sum / max(1, total)):.4f} "
        f"compression_attn_entropy={(entropy_sum / max(1, entropy_count)):.4f}"
    )


def evaluate_records(
    records: Sequence[Dict[str, Any]], *, qualitative_samples: int, confusion_top_k: int, scorer: str = "official"
) -> Dict[str, Any]:
    from evals.vqa import (
        build_confusion_summary,
        format_qualitative_samples,
        summarize_vqa_predictions,
    )

    summary = summarize_vqa_predictions(records, scorer=scorer)
    summary["qualitative"] = format_qualitative_samples(records, n=qualitative_samples, scorer=scorer)
    summary["confusions"] = build_confusion_summary(records, top_k=confusion_top_k, scorer=scorer)
    return summary


def init_fixed_eval_tracker(
    run_dir: str,
    split: str,
    dataset: Any,
    *,
    count: int,
    logger: Logger,
) -> Dict[str, Any]:
    prompts_path = os.path.join(run_dir, f"fixed_eval_{split}_prompts.json")
    answers_path = os.path.join(run_dir, f"fixed_eval_{split}_answers.jsonl")
    prompts: List[Dict[str, Any]]

    if os.path.isfile(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        logger.log(f"[mm] fixed-eval prompts loaded: {prompts_path} (count={len(prompts)})")
    else:
        prompts = []
        src_items = getattr(dataset, "items", None)
        if isinstance(src_items, list):
            for it in src_items[: max(0, int(count))]:
                prompts.append(
                    {
                        "question_id": int(it.get("question_id")),
                        "image_id": int(it.get("image_id")),
                        "question": str(it.get("question", "")),
                        "image_path": str(it.get("image_path", "")),
                        "canonical_answer": str(it.get("answer", "")),
                    }
                )
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=True)
        logger.log(f"[mm] fixed-eval prompts saved: {prompts_path} (count={len(prompts)})")

    return {
        "split": split,
        "prompts_path": prompts_path,
        "answers_path": answers_path,
        "prompts": prompts,
    }


def append_fixed_eval_answers(
    tracker: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    *,
    global_step: int,
    epoch: int,
    tag: str,
    logger: Optional[Logger] = None,
) -> None:
    prompts = tracker.get("prompts", [])
    if not prompts:
        return
    rec_by_qid = {int(r.get("question_id")): r for r in records}
    answers: List[Dict[str, Any]] = []
    for p in prompts:
        qid = int(p.get("question_id"))
        r = rec_by_qid.get(qid)
        answers.append(
            {
                "question_id": qid,
                "image_id": int(p.get("image_id", -1)),
                "question": str(p.get("question", "")),
                "prediction": (None if r is None else str(r.get("prediction", ""))),
            }
        )
    row = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "tag": str(tag),
        "split": str(tracker.get("split", "eval")),
        "answers": answers,
    }
    out_path = str(tracker["answers_path"])
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    if logger is not None:
        logger.log(f"[mm] fixed-eval answers appended: {out_path} step={global_step} tag={tag}")


def print_eval_summary(logger: Logger, split: str, summary: Dict[str, Any]) -> None:
    overall = summary.get("overall_accuracy")
    if overall is None:
        logger.log(f"[eval:{split}] no labels available; generated predictions only.")
        return
    scorer = str(summary.get("scorer", "unknown"))
    logger.log(f"[eval:{split}] overall_accuracy={overall:.4f} scorer={scorer}")
    by_answer = summary.get("answer_type_accuracy", {})
    if by_answer:
        order = ("yes/no", "number", "other")
        rendered = [f"{k}={float(by_answer.get(k, 0.0)):.4f}" for k in order]
        for k in sorted(by_answer.keys()):
            if k not in order:
                rendered.append(f"{k}={float(by_answer.get(k, 0.0)):.4f}")
        logger.log(
            "[eval:{split}] answer_type: ".format(split=split)
            + " ".join(rendered)
        )
    qtype = summary.get("question_type_accuracy", {})
    if qtype:
        top = sorted(qtype.items(), key=lambda kv: kv[1], reverse=True)[:10]
        logger.log(
            "[eval:{split}] top question-type accuracy: ".format(split=split)
            + " | ".join(f"{k}:{v:.3f}" for (k, v) in top)
        )


def maybe_cuda_empty_cache(logger: Logger, *, enabled: bool, tag: str) -> None:
    if not bool(enabled):
        return
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    logger.log(f"[mm] cuda empty_cache after {tag}")


def maybe_shutdown_loader_workers(logger: Logger, loader: Optional[DataLoader], *, tag: str) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None) if iterator is not None else None
    if not callable(shutdown):
        return
    try:
        shutdown()
        try:
            loader._iterator = None
        except Exception:
            pass
        logger.log(f"[mm] dataloader workers shut down for {tag}")
    except Exception as e:
        logger.log(f"[mm] dataloader worker shutdown skipped for {tag}: {e}")


def log_startup_config(
    logger: Logger,
    args: argparse.Namespace,
    *,
    device: str,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    tokenizer: ByteBPETokenizer,
    bridge_cfg: BridgeConfig,
) -> None:
    logger.log(
        f"[mm] run_id={args.run_id} seed={args.seed} resume_step={args.checkpoint} "
        f"eval_only={int(bool(args.eval_only))} eval_split={args.eval_split}"
    )
    logger.log(
        f"[mm] device={device} precision={args.precision} amp={int(amp_enabled)} "
        f"amp_dtype={(str(amp_dtype).replace('torch.', '') if amp_dtype is not None else 'none')}"
    )
    logger.log(
        f"[mm] vision model={args.vision_model} ckpt={args.vision_checkpoint} "
        f"feature_source={args.vision_feature_source} feature_mode={args.vision_feature_mode} "
        f"vision_device={args.vision_device}"
    )
    logger.log(
        f"[mm] lm ckpt={args.lm_checkpoint} tokenizer={args.tokenizer_path} vocab={tokenizer.vocab_size}"
    )
    logger.log(
        f"[mm] bridge type={bridge_cfg.bridge_type} visual_tokens={bridge_cfg.num_visual_tokens} "
        f"hidden={bridge_cfg.bridge_hidden_dim} token_reduce={bridge_cfg.token_reduce} "
        f"learned_init_std={bridge_cfg.learned_init_std:.6g} add_2d_pos_emb={int(bool(bridge_cfg.add_2d_pos_emb))} "
        f"heads={bridge_cfg.bridge_num_heads} depth={bridge_cfg.bridge_query_depth} "
        f"refine={bridge_cfg.bridge_refine_layers} pre_mixer={bridge_cfg.bridge_pre_mixer_type} "
        f"hybrid_alpha_mode={bridge_cfg.bridge_hybrid_alpha_mode} "
        f"qcond={int(bool(bridge_cfg.bridge_question_conditioning))} qcond_scale={bridge_cfg.bridge_qcond_scale:.6g} "
        f"query_bank_mode={bridge_cfg.bridge_query_bank_mode} "
        f"qquery_basis={bridge_cfg.bridge_qquery_basis_count} qquery_scale={bridge_cfg.bridge_qquery_scale:.6g} "
        f"qquery_multi={int(getattr(bridge_cfg, 'bridge_qquery_multi_count', 1))} "
        f"qquery_hybrid_gate_init={float(getattr(bridge_cfg, 'bridge_qquery_hybrid_gate_init', 0.5)):.6g} "
        f"qquery_role_specialization={int(bool(getattr(bridge_cfg, 'bridge_query_role_specialization', False)))} "
        f"qctx_mode={bridge_cfg.bridge_question_context_mode} "
        f"iter_qquery_steps={int(getattr(bridge_cfg, 'bridge_iterative_qquery_steps', 1))} "
        f"iter_qquery_resid={float(getattr(bridge_cfg, 'bridge_iterative_qquery_residual_scale', 1.0)):.6g} "
        f"token_selector={bridge_cfg.bridge_token_selector_type} token_select_k={bridge_cfg.bridge_token_select_k} "
        f"token_select_k_min={bridge_cfg.bridge_token_select_k_min} "
        f"num_roles={bridge_cfg.bridge_num_roles} evidence_topk={bridge_cfg.bridge_evidence_topk} "
        f"semantic_bottleneck={int(bool(getattr(bridge_cfg, 'semantic_bottleneck', False)))} "
        f"semantic_tokens={int(getattr(bridge_cfg, 'semantic_tokens', 16))} "
        f"semantic_latent_dim={int(getattr(bridge_cfg, 'semantic_latent_dim', 256))} "
        f"semantic_recon_w={float(getattr(bridge_cfg, 'semantic_recon_loss_weight', 0.0)):.6g} "
        f"semantic_consistency_w={float(getattr(bridge_cfg, 'semantic_consistency_loss_weight', 0.0)):.6g} "
        f"semantic_token_schedule={str(getattr(bridge_cfg, 'semantic_token_schedule', ''))}"
    )
    logger.log(
        f"[mm] visual_feature_adapter_type={str(getattr(args, 'visual_feature_adapter_type', 'none'))} "
        f"visual_feature_adapter_hidden_dim={int(getattr(args, 'visual_feature_adapter_hidden_dim', 0))} "
        f"visual_feature_adapter_dropout={float(getattr(args, 'visual_feature_adapter_dropout', 0.0)):.6g} "
        f"train_vision_last_n_blocks={int(getattr(args, 'train_vision_last_n_blocks', 0))} "
        f"vision_lr_scale={float(getattr(args, 'vision_lr_scale', 1.0)):.6g}"
    )
    logger.log(
        f"[mm] lm_visual_adapter_type={getattr(args, 'lm_visual_adapter_type', 'none')} "
        f"lm_visual_adapter_layers={int(getattr(args, 'lm_visual_adapter_layers', 0))} "
        f"lm_visual_adapter_heads={int(getattr(args, 'lm_visual_adapter_num_heads', 8))} "
        f"lm_visual_adapter_dropout={float(getattr(args, 'lm_visual_adapter_dropout', 0.0)):.6g} "
        f"lm_visual_adapter_gate_init={float(getattr(args, 'lm_visual_adapter_gate_init', 0.5)):.6g}"
    )
    logger.log(
        f"[mm] eval_image_corruption={str(getattr(args, 'eval_image_corruption', 'none'))} "
        f"eval_image_corruption_seed={int(getattr(args, 'eval_image_corruption_seed', 123))}"
    )
    logger.log(
        f"[mm] prefix_calibration={int(bool(args.prefix_calibration))} "
        f"prefix_calib_layernorm={int(bool(args.prefix_calib_layernorm))} "
        f"prefix_calib_bias={int(bool(args.prefix_calib_bias))} "
        f"prefix_calib_gate_init={float(args.prefix_calib_gate_init):.6g} "
        f"prefix_geom_mlp_ratio={float(args.prefix_geom_mlp_ratio):.6g} "
        f"prefix_geom_token_mixer_layers={int(args.prefix_geom_token_mixer_layers)} "
        f"prefix_norm_target_ratio={float(args.prefix_norm_target_ratio):.6g} "
        f"prefix_norm_reg_weight={float(args.prefix_norm_reg_weight):.6g} "
        f"prefix_batchvar_reg_weight={float(args.prefix_batchvar_reg_weight):.6g} "
        f"prefix_dropout={float(args.prefix_dropout):.6g}"
    )
    logger.log(
        f"[mm] freeze_mode={args.freeze_mode} train_top_lm_layers={args.train_top_lm_layers} "
        f"loss_on_answer_only={int(bool(args.loss_on_answer_only))}"
    )
    logger.log(
        f"[mm] init_from_mm_checkpoint={str(getattr(args, 'init_from_mm_checkpoint', '') or '')} "
        f"semantic_teacher_checkpoint={str(getattr(args, 'semantic_teacher_checkpoint', '') or '')} "
        f"use_compression={int(bool(getattr(args, 'use_compression', False)))} "
        f"compression_k={int(getattr(args, 'compression_k', getattr(args, 'semantic_tokens', 16)))} "
        f"compression_distill_weight={float(getattr(args, 'compression_distill_weight', getattr(args, 'semantic_recon_loss_weight', 0.0))):.6g}"
    )
    logger.log(
        f"[mm] seq_lens q={args.max_question_length} a={args.max_answer_length} text={args.max_text_tokens}"
    )
    logger.log(
        f"[mm] dataloader batch_size={args.batch_size} "
        f"eval_batch_size={(int(args.eval_batch_size) if int(getattr(args, 'eval_batch_size', 0)) > 0 else int(args.batch_size))} "
        f"num_workers={args.num_workers} "
        f"prefetch_factor={args.prefetch_factor} pin_memory={int(bool(args.pin_memory))} "
        f"grad_accum_steps={args.grad_accum_steps} effective_batch_size={int(args.batch_size) * max(1, int(args.grad_accum_steps))}"
    )
    logger.log(
        f"[mm] data images_root={args.images_root} annotations_root={args.annotations_root} gqa_root={getattr(args, 'gqa_root', '')} "
        f"dataset_mix={str(getattr(args, 'dataset_mix', '') or '')} "
        f"pointing_index_path={str(getattr(args, 'pointing_index_path', '') or '')} "
        f"pointing_mix_ratio={float(getattr(args, 'pointing_mix_ratio', 0.0)):.6g} "
        f"use_grounding_loss={int(bool(getattr(args, 'use_grounding_loss', False)))} "
        f"grounding_loss_weight={float(getattr(args, 'grounding_loss_weight', 0.0)):.6g} "
        f"limit_train={args.limit_train} limit_val={args.limit_val} limit_eval={args.limit_eval} "
        f"eval_fraction={float(args.eval_fraction):.4g}"
    )
    logger.log(
        f"[mm] loop epochs={args.epochs} max_steps={args.max_steps} overfit_small_batch={int(bool(args.overfit_small_batch))} "
        f"lr_schedule={args.lr_schedule} lr_warmup_steps={args.lr_warmup_steps} lr_min_ratio={args.lr_min_ratio} "
        f"min_train_steps_per_s={float(getattr(args, 'min_train_steps_per_s', 0.0)):.6g} "
        f"min_train_steps_window={int(getattr(args, 'min_train_steps_window', 0))} "
        f"train_sample_budget={int(args.train_sample_budget)} manual_max_steps={int(bool(args.manual_max_steps))} "
        f"eval_use_kv_cache={int(bool(getattr(args, 'eval_use_kv_cache', False)))} "
        f"eval_kv_cache_mode={str(getattr(args, 'eval_kv_cache_mode', 'batched'))} "
        f"cuda_empty_cache_after_eval={int(bool(args.cuda_empty_cache_after_eval))} "
        f"log_every={args.log_every} eval_every={args.eval_every} eval_batches={args.eval_batches} "
        f"final_eval_batches={args.final_eval_batches} "
        f"eval_log_every={args.eval_log_every} eval_scorer={args.eval_scorer} "
        f"fixed_eval_count={args.fixed_eval_count} ckpt_every={args.ckpt_every} "
        f"final_sanity_count={args.final_sanity_count}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", type=str)
    ap.add_argument("--checkpoint", type=int, default=None, help="Resume from logs/<run_id>/step_<N>.tar")
    ap.add_argument("--reset_schedule", action="store_true", help="Load model weights from checkpoint but reset step/epoch/LR to 0 (for two-stage training)")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"])

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=35)

    ap.add_argument(
        "--vision_model",
        type=str,
        default="vitvae2",
        choices=["vae", "vaer", "vitvae", "vitvae2", "mobilevit_hf", "dinov2_small", "dinov2_base", "mobileclip_s0", "siglip_base", "dinovit_ssl"],
    )
    ap.add_argument("--vision_checkpoint", type=str, default=None)
    ap.add_argument("--vision_config", type=str, default=None)
    ap.add_argument("--vision_latent_dim", type=int, default=768)
    ap.add_argument("--vision_cbld", type=int, default=1536)
    ap.add_argument(
        "--vision_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device used for frozen vision encoder forward. 'auto' follows main --device.",
    )
    ap.add_argument(
        "--vision_feature_source",
        type=str,
        default="posterior_mu",
        choices=["posterior_mu", "encoder", "encoder_plus_posterior_mu", "model_output"],
    )
    ap.add_argument("--vision_feature_mode", type=str, default="auto", choices=["auto", "global", "token"])

    ap.add_argument("--lm_checkpoint", type=str, default=None)
    ap.add_argument("--lm_config", type=str, default=None)
    ap.add_argument("--lm_d_model", type=int, default=768)
    ap.add_argument("--lm_num_heads", type=int, default=8)
    ap.add_argument("--lm_layers", type=int, default=5)
    ap.add_argument("--lm_mlp_ratio", type=int, default=4)
    ap.add_argument("--lm_dropout", type=float, default=0.1)
    ap.add_argument("--lm_max_seq_len", type=int, default=512)
    ap.add_argument(
        "--mm_sdp_backend",
        type=str,
        default="math",
        choices=["auto", "flash", "mem_efficient", "math"],
        help="Override LM SDPA backend for multimodal runs; default 'math' avoids flash/mask incompatibilities.",
    )
    ap.add_argument(
        "--mm_lm_autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Autocast LM decode on CUDA (bf16/fp16) to satisfy SDPA kernel dtype requirements.",
    )
    ap.add_argument("--tokenizer_path", type=str, default=None)

    ap.add_argument(
        "--bridge_type",
        type=str,
        default="mlp",
        choices=[
            "mlp",
            "learned_tokens",
            "learned_query",
            "query_cross_attn",
            "perceiver_resampler",
            "multiscale_perceiver",
            "qformer_lite",
            "structured_roles",
            "evidence_sparse",
            "hybrid_const_image",
        ],
    )
    ap.add_argument("--bridge_hidden_dim", type=int, default=1024)
    ap.add_argument("--num_visual_tokens", type=int, default=8)
    ap.add_argument(
        "--bridge_add_2d_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add 2D positional embeddings to visual token features before bridge projection.",
    )
    ap.add_argument(
        "--bridge_learned_init_std",
        type=float,
        default=0.02,
        help="Init std for --bridge_type learned_tokens.",
    )
    ap.add_argument(
        "--bridge_token_reduce",
        type=str,
        default="adaptive_pool",
        choices=["adaptive_pool", "mean_expand", "all"],
    )
    ap.add_argument("--bridge_num_heads", type=int, default=8)
    ap.add_argument("--bridge_attn_dropout", type=float, default=0.0)
    ap.add_argument("--bridge_query_depth", type=int, default=2)
    ap.add_argument("--bridge_refine_layers", type=int, default=0)
    ap.add_argument(
        "--bridge_pre_mixer_type",
        type=str,
        default="none",
        choices=["none", "self_attn", "conv1d"],
    )
    ap.add_argument("--bridge_pre_mixer_layers", type=int, default=1)
    ap.add_argument("--bridge_pre_mixer_kernel_size", type=int, default=3)
    ap.add_argument(
        "--bridge_hybrid_alpha_mode",
        type=str,
        default="scalar",
        choices=["scalar", "token"],
    )
    ap.add_argument("--bridge_hybrid_alpha_init", type=float, default=0.5)
    ap.add_argument(
        "--bridge_hybrid_image_bridge_type",
        type=str,
        default="learned_query",
        choices=[
            "mlp",
            "learned_query",
            "query_cross_attn",
            "perceiver_resampler",
            "multiscale_perceiver",
            "qformer_lite",
            "structured_roles",
            "evidence_sparse",
        ],
    )
    ap.add_argument(
        "--bridge_question_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition perceiver/hybrid image extraction on pooled question embeddings.",
    )
    ap.add_argument(
        "--bridge_qcond_scale",
        type=float,
        default=0.5,
        help="Scale factor for question-conditioned FiLM modulation in perceiver latents.",
    )
    ap.add_argument(
        "--bridge_query_bank_mode",
        type=str,
        default="learned",
        choices=[
            "learned",
            "question_mix",
            "question_hidden_mean",
            "question_hidden_attn",
            "question_hidden_mean_multi",
            "question_hidden_hybrid",
        ],
        help=(
            "Use the standard learned perceiver latent bank, question-mixed query latents, "
            "mean-projected question-token latents, attention-derived question-token latents, "
            "multi-query mean-pooled latents, or hybrid mean-plus-attention latents."
        ),
    )
    ap.add_argument(
        "--bridge_qquery_basis_count",
        type=int,
        default=4,
        help="Number of learned question-query basis tensors mixed by the pooled question context.",
    )
    ap.add_argument(
        "--bridge_qquery_scale",
        type=float,
        default=1.0,
        help="Residual scale applied to question-mixed perceiver queries.",
    )
    ap.add_argument(
        "--bridge_qquery_multi_count",
        type=int,
        default=1,
        help="Number of LM-conditioned query groups for question_hidden_mean_multi.",
    )
    ap.add_argument(
        "--bridge_qquery_hybrid_gate_init",
        type=float,
        default=0.5,
        help="Initial gate for blending mean-projected and attention-derived qquery deltas.",
    )
    ap.add_argument(
        "--bridge_query_role_specialization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add learned role-specialization embeddings to perceiver query slots.",
    )
    ap.add_argument(
        "--bridge_question_context_mode",
        type=str,
        default="all_text",
        choices=["all_text", "prompt_only", "question_only"],
        help="Pool q-conditioning context from all text tokens, prompt tokens, or question tokens only.",
    )
    ap.add_argument(
        "--bridge_iterative_qquery_steps",
        type=int,
        default=1,
        help="Number of iterative query/retrieve passes inside the perceiver core.",
    )
    ap.add_argument(
        "--bridge_iterative_qquery_residual_scale",
        type=float,
        default=1.0,
        help="Residual scale used when feeding retrieved visual summaries into later query passes.",
    )
    ap.add_argument(
        "--bridge_token_selector_type",
        type=str,
        default="none",
        choices=["none", "topk", "qtopk", "qadaptive"],
        help="Optional token selector before query/cross-attention bridges.",
    )
    ap.add_argument(
        "--bridge_token_select_k",
        type=int,
        default=0,
        help="Selected token count (or max token budget for qadaptive) before the bridge extractor.",
    )
    ap.add_argument(
        "--bridge_token_select_k_min",
        type=int,
        default=0,
        help="Minimum kept token budget for --bridge_token_selector_type=qadaptive (<=0 uses max_k/2).",
    )
    ap.add_argument("--bridge_num_roles", type=int, default=4)
    ap.add_argument("--bridge_evidence_topk", type=int, default=0)
    ap.add_argument(
        "--semantic_bottleneck",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a late semantic bottleneck after dense evidence retrieval and before LM prefix export.",
    )
    ap.add_argument(
        "--semantic_tokens",
        type=int,
        default=16,
        help="Exported semantic token count M for the post-perceiver bottleneck.",
    )
    ap.add_argument(
        "--semantic_latent_dim",
        type=int,
        default=256,
        help="Internal semantic latent width Z before decoding back to LM width.",
    )
    ap.add_argument(
        "--semantic_recon_loss_weight",
        type=float,
        default=0.1,
        help="Weight for MSE between decoded semantic tokens and an adaptive-pooled target from pre-bottleneck evidence latents.",
    )
    ap.add_argument(
        "--semantic_consistency_loss_weight",
        type=float,
        default=0.1,
        help="Weight for cosine consistency between decoded semantic tokens and the same pooled evidence target.",
    )
    ap.add_argument(
        "--semantic_token_schedule",
        type=str,
        default="",
        help="Reserved schedule string for future semantic token-count curricula; currently logged only.",
    )
    ap.add_argument(
        "--semantic_teacher_checkpoint",
        type=str,
        default="",
        help="Optional frozen MM checkpoint used as a teacher for semantic-token distillation targets.",
    )
    ap.add_argument(
        "--init_from_mm_checkpoint",
        type=str,
        default="",
        help="Optional MM checkpoint to load model weights from before starting a new run.",
    )
    ap.add_argument(
        "--use_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alias for --semantic_bottleneck for compression-tuning experiments.",
    )
    ap.add_argument(
        "--compression_k",
        type=int,
        default=16,
        help="Alias for --semantic_tokens for compression-tuning experiments.",
    )
    ap.add_argument(
        "--compression_distill_weight",
        type=float,
        default=0.1,
        help="Alias for --semantic_recon_loss_weight for compression-tuning experiments.",
    )
    ap.add_argument(
        "--use_grounding_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable grounding KL supervision over perceiver cross-attention using pointing annotations.",
    )
    ap.add_argument("--grounding_loss_weight", type=float, default=0.05)
    ap.add_argument("--grounding_sigma", type=float, default=1.5)
    ap.add_argument("--pointing_index_path", type=str, default="")
    ap.add_argument("--pointing_mix_ratio", type=float, default=0.25)
    ap.add_argument(
        "--prefix_calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable bridge output calibration: y = gate * LN(prefix) + bias.",
    )
    ap.add_argument(
        "--prefix_calib_layernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply LayerNorm inside prefix calibration.",
    )
    ap.add_argument(
        "--prefix_calib_bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use trainable additive bias in prefix calibration.",
    )
    ap.add_argument(
        "--prefix_calib_gate_init",
        type=float,
        default=1.0,
        help="Initial multiplicative gate for calibrated prefix tokens.",
    )
    ap.add_argument(
        "--prefix_geom_mlp_ratio",
        type=float,
        default=0.0,
        help="Optional residual geometry MLP ratio inside prefix calibration. <=0 disables.",
    )
    ap.add_argument(
        "--prefix_geom_token_mixer_layers",
        type=int,
        default=0,
        help="Optional token-mixer layers inside prefix calibration.",
    )
    ap.add_argument(
        "--prefix_norm_target_ratio",
        type=float,
        default=0.0,
        help="Target ratio (prefix_norm / text_norm). <=0 disables ratio target.",
    )
    ap.add_argument(
        "--prefix_norm_reg_weight",
        type=float,
        default=0.0,
        help="Weight for (prefix/text norm ratio - target)^2 regularization.",
    )
    ap.add_argument(
        "--prefix_batchvar_reg_weight",
        type=float,
        default=0.0,
        help="Weight for prefix batch-variance penalty (stability regularization).",
    )
    ap.add_argument(
        "--prefix_dropout",
        type=float,
        default=0.0,
        help="Dropout probability applied to calibrated visual prefix during training.",
    )

    ap.add_argument(
        "--freeze_mode",
        type=str,
        default="bridge_only",
        choices=["bridge_only", "bridge_plus_top_lm", "semantic_adapter_only", "full_finetune"],
    )
    ap.add_argument("--train_top_lm_layers", type=int, default=1)
    ap.add_argument(
        "--lm_visual_adapter_type",
        type=str,
        default="none",
        choices=["none", "cross_attn"],
        help="Optional residual visual adapters inserted into the top LM blocks.",
    )
    ap.add_argument(
        "--lm_visual_adapter_layers",
        type=int,
        default=0,
        help="Number of top LM layers that receive residual visual adapters.",
    )
    ap.add_argument("--lm_visual_adapter_num_heads", type=int, default=8)
    ap.add_argument("--lm_visual_adapter_dropout", type=float, default=0.0)
    ap.add_argument("--lm_visual_adapter_gate_init", type=float, default=0.5)
    ap.add_argument(
        "--visual_feature_adapter_type",
        type=str,
        default="none",
        choices=["none", "res_mlp"],
        help="Optional trainable visual-side adapter applied to frozen VM features before the bridge.",
    )
    ap.add_argument("--visual_feature_adapter_hidden_dim", type=int, default=0)
    ap.add_argument("--visual_feature_adapter_dropout", type=float, default=0.0)
    ap.add_argument("--train_vision_last_n_blocks", type=int, default=0)
    ap.add_argument("--vision_lr_scale", type=float, default=1.0)

    ap.add_argument("--images_root", type=str, default="images")
    ap.add_argument("--annotations_root", type=str, default="annotations")
    ap.add_argument("--gqa_root", type=str, default="data/gqa")
    ap.add_argument("--gqa_eval_group", type=str, default="", choices=["", "spatial", "attribute", "count", "exist"])
    ap.add_argument(
        "--dataset_mix",
        type=str,
        default="",
        help='JSON dict mapping supervised source keys to percentages, e.g. \'{"vqav2_train": 100, "vqav2_val": 10}\' or \'{"gqa_train": 100}\'.',
    )
    ap.add_argument("--auto_download", action="store_true")
    ap.add_argument("--download_images", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--download_test", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--max_question_length", type=int, default=64)
    ap.add_argument("--max_answer_length", type=int, default=16)
    ap.add_argument("--max_text_tokens", type=int, default=256)
    ap.add_argument("--loss_on_answer_only", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=0,
        help="Eval dataloader batch size. <=0 reuses --batch_size.",
    )
    ap.add_argument(
        "--eval_image_corruption",
        type=str,
        default="none",
        choices=["none", "zero", "shuffle", "random_swap"],
        help="Optional eval-only image corruption mode for grounding diagnostics.",
    )
    ap.add_argument(
        "--eval_image_corruption_seed",
        type=int,
        default=123,
        help="Seed for eval image corruption modes that require a randomized image remap.",
    )
    ap.add_argument(
        "--eval_bypass_compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="At eval time, bypass the semantic bottleneck and feed raw perceiver evidence tokens to the LM.",
    )
    ap.add_argument(
        "--eval_tiny_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a 1-epoch linear probe over exported visual tokens instead of standard generation eval.",
    )
    ap.add_argument(
        "--eval_grounding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run grounding evaluation over an annotated pointing/GQA-style index instead of standard generation eval.",
    )
    ap.add_argument(
        "--eval_grounding_index_path",
        type=str,
        default="",
        help="Annotated JSONL index used by --eval_grounding. Falls back to --pointing_index_path when omitted.",
    )
    ap.add_argument("--tiny_head_epochs", type=int, default=1)
    ap.add_argument("--tiny_head_answer_top_k", type=int, default=3000)
    ap.add_argument("--tiny_head_feature_pool", type=str, default="flatten", choices=["flatten", "mean"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument(
        "--manual_max_steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Respect the provided --max_steps exactly instead of deriving it from effective batch size.",
    )
    ap.add_argument(
        "--train_sample_budget",
        type=int,
        default=DEFAULT_TRAIN_SAMPLE_BUDGET,
        help="Total training samples budget used to derive max_steps for normal runs.",
    )
    ap.add_argument("--overfit_small_batch", action="store_true")
    ap.add_argument("--overfit_steps", type=int, default=200)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"])
    ap.add_argument("--lr_warmup_steps", type=int, default=0)
    ap.add_argument("--lr_min_ratio", type=float, default=0.1)
    ap.add_argument(
        "--min_train_steps_per_s",
        type=float,
        default=1.0,
        help="Checkpoint and exit with a special code if sustained train throughput falls below this value. <=0 disables.",
    )
    ap.add_argument(
        "--min_train_steps_window",
        type=int,
        default=100,
        help="Number of consecutive train steps below --min_train_steps_per_s required before triggering the watchdog.",
    )
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--grad_accum_steps", type=int, default=1)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument(
        "--final_eval_batches",
        type=int,
        default=0,
        help="Maximum eval batches for the end-of-training eval; <=0 means full eval split.",
    )
    ap.add_argument("--eval_log_every", type=int, default=10)
    ap.add_argument(
        "--eval_use_kv_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use incremental LM KV-cache during eval generation when the visual prefix is decode-invariant.",
    )
    ap.add_argument(
        "--eval_kv_cache_mode",
        type=str,
        default="batched",
        choices=["serial", "batched"],
        help="KV-cache eval continuation mode after the first generated token.",
    )
    ap.add_argument(
        "--cuda_empty_cache_after_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call gc.collect()+torch.cuda.empty_cache() before/after eval loops to reduce eval-induced slowdown.",
    )
    ap.add_argument("--fixed_eval_count", type=int, default=5)
    ap.add_argument(
        "--final_sanity_count",
        type=int,
        default=DEFAULT_FINAL_SANITY_COUNT,
        help="Final greedy sanity pass prompt count (default 4, clamped to 3-5; <=0 disables).",
    )
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument(
        "--eval_fraction",
        type=float,
        default=DEFAULT_EVAL_FRACTION,
        help="Fraction of the eval split to use when --limit_eval is not set.",
    )
    ap.add_argument("--qualitative_samples", type=int, default=8)
    ap.add_argument("--confusion_top_k", type=int, default=20)
    ap.add_argument(
        "--eval_scorer",
        type=str,
        default="official",
        choices=["official", "proxy"],
        help="VQA metric scorer for eval summaries.",
    )
    ap.add_argument("--save_predictions_jsonl", type=str, default=None)
    ap.add_argument("--debug_shapes", action="store_true")
    return ap.parse_args()


def _resolve_training_max_steps(args: argparse.Namespace) -> int:
    explicit_max_steps = int(args.max_steps)
    if bool(args.overfit_small_batch):
        return explicit_max_steps
    if bool(args.manual_max_steps):
        return explicit_max_steps
    eff_batch = max(1, int(args.batch_size) * max(1, int(args.grad_accum_steps)))
    budget = max(1, int(args.train_sample_budget))
    return int(math.ceil(float(budget) / float(eff_batch)))


def main() -> None:
    args = parse_args()
    args = _apply_runtime_defaults(args)
    resolved_max_steps = _resolve_training_max_steps(args)
    max_steps_override_msg = None
    if (not bool(args.overfit_small_batch)) and (not bool(args.manual_max_steps)):
        explicit_max_steps = int(args.max_steps)
        args.max_steps = int(resolved_max_steps)
        eff_batch = max(1, int(args.batch_size) * max(1, int(args.grad_accum_steps)))
        if explicit_max_steps > 0 and explicit_max_steps != int(resolved_max_steps):
            max_steps_override_msg = (
                f"[mm] overriding explicit --max_steps {explicit_max_steps} with derived max_steps={resolved_max_steps} "
                f"from train_sample_budget={int(args.train_sample_budget)} effective_batch_size={eff_batch}"
            )
    else:
        args.max_steps = int(resolved_max_steps)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    amp_enabled, amp_dtype, use_scaler = resolve_amp(device, args.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    run_dir = os.path.join(LOGDIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(run_id=args.run_id, checkpoint_id=args.checkpoint)
    if max_steps_override_msg:
        logger.log(max_steps_override_msg)
    if args.auto_download:
        prepare_vqav2(
            VQAv2Paths(images_root=args.images_root, annotations_root=args.annotations_root),
            splits=("train", "val"),
            download_images=bool(args.download_images),
            download_test=bool(args.download_test),
        )
        logger.log("[mm] dataset prepare complete")

    resume_payload = None
    global_step = 0
    start_epoch = 0
    resume_batch_in_epoch = 0
    legacy_resume_no_batch = False
    if args.checkpoint is not None:
        ckpt_path = _checkpoint_path(args.run_id, int(args.checkpoint))
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
        resume_payload = _load_checkpoint(ckpt_path, map_location="cpu")
        global_step = int(resume_payload.get("global_step", int(args.checkpoint)))
        start_epoch = int(resume_payload.get("epoch", 0))
        legacy_resume_no_batch = "batch_in_epoch" not in resume_payload
        resume_batch_in_epoch = int(resume_payload.get("batch_in_epoch", 0))
        if getattr(args, "reset_schedule", False):
            logger.log(
                f"[mm] loading weights from {ckpt_path} (checkpoint step={global_step}) "
                f"but resetting schedule: global_step=0, epoch=0"
            )
            global_step = 0
            start_epoch = 0
            resume_batch_in_epoch = 0
        else:
            logger.log(
                f"[mm] resuming from {ckpt_path} "
                f"(step={global_step}, epoch={start_epoch}, batch_in_epoch={resume_batch_in_epoch})"
            )

    model, tokenizer, bridge_cfg = build_runtime_from_args(args, device=device, checkpoint_payload=resume_payload)
    if resume_payload is not None:
        _maybe_initialize_bridge_from_state_dict(model, resume_payload["model_state_dict"])
        missing, unexpected = model.load_state_dict(resume_payload["model_state_dict"], strict=False)
        if unexpected:
            logger.log(f"[mm] WARNING: unexpected keys in checkpoint: {unexpected}")
        if missing:
            logger.log(f"[mm] note: {len(missing)} keys not in checkpoint (newly initialized): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    block_backend = getattr(model.lm._dec_blocks[0], "_sdp_backend", "n/a") if len(model.lm._dec_blocks) > 0 else "n/a"
    logger.log(
        f"[mm] lm attn_impl={model.lm._config.attn_impl} "
        f"sdp_backend={model.lm._config.sdp_backend} block0_backend={block_backend} "
        f"lm_autocast={'on' if model.lm_autocast else 'off'}"
    )

    max_supported = int(model.lm._config.max_seq_len) - int(bridge_cfg.num_visual_tokens)
    if args.max_text_tokens > max_supported:
        logger.log(f"[mm] reducing --max_text_tokens from {args.max_text_tokens} to {max_supported} to fit LM context")
        args.max_text_tokens = max_supported

    log_startup_config(
        logger,
        args,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        tokenizer=tokenizer,
        bridge_cfg=bridge_cfg,
    )

    val_limit = int(args.limit_val) if int(args.limit_val) > 0 else 0
    train_loader = None
    if not args.eval_only:
        train_loader = build_loader(
            args,
            tokenizer=tokenizer,
            split="train",
            train_mode=True,
            limit=(int(args.limit_train) if int(args.limit_train) > 0 else 0),
        )
        logger.log(f"[mm] train samples={len(train_loader.dataset)}")
        if hasattr(train_loader.dataset, "source_counts"):
            logger.log(f"[mm] === effective train dataset mix: {len(train_loader.dataset)} samples ===")
            for src, (total, kept, skipped) in sorted(train_loader.dataset.source_counts.items()):
                pct_of_source = 100.0 * float(kept) / float(total) if total else 0.0
                pct_of_dataset = 100.0 * float(kept) / float(len(train_loader.dataset)) if len(train_loader.dataset) else 0.0
                line = (
                    f"[mm]   {src:20s} {kept:>7d} / {total:>7d} "
                    f"({pct_of_source:5.1f}% of source, {pct_of_dataset:5.1f}% of dataset)"
                )
                if skipped:
                    line += f" skipped={skipped}"
                logger.log(line)
    val_loader = build_loader(
        args,
        tokenizer=tokenizer,
        split=args.eval_split,
        train_mode=False,
        limit=(int(args.limit_eval) if int(args.limit_eval) > 0 else val_limit),
    )
    logger.log(f"[mm] eval split={args.eval_split} samples={len(val_loader.dataset)}")
    fixed_eval_tracker = init_fixed_eval_tracker(
        run_dir=run_dir,
        split=args.eval_split,
        dataset=val_loader.dataset,
        count=int(args.fixed_eval_count),
        logger=logger,
    )

    if resume_payload is None:
        warmup_batch = None
        if train_loader is not None:
            try:
                warmup_batch = next(iter(train_loader))
            except StopIteration:
                warmup_batch = None
        if warmup_batch is None:
            try:
                warmup_batch = next(iter(val_loader))
            except StopIteration:
                warmup_batch = None
        if warmup_batch is None:
            raise SystemExit("No data available to initialize multimodal modules.")
        _materialize_lazy_params(model, warmup_batch, device)
        init_from_mm_checkpoint = str(getattr(args, "init_from_mm_checkpoint", "") or "").strip()
        if init_from_mm_checkpoint:
            load_model_weights_from_mm_checkpoint(
                model,
                init_from_mm_checkpoint,
                logger=logger,
            )
    elif str(getattr(args, "init_from_mm_checkpoint", "") or "").strip():
        logger.log("[mm] note: --init_from_mm_checkpoint ignored because --checkpoint resume is active")

    semantic_teacher_checkpoint = str(getattr(args, "semantic_teacher_checkpoint", "") or "").strip()
    if semantic_teacher_checkpoint:
        attach_semantic_teacher(
            model,
            checkpoint_path=semantic_teacher_checkpoint,
            device=device,
            logger=logger,
        )

    configure_freezing(model, args)
    _set_module_modes(model, args.freeze_mode)
    tr_params, all_params = _trainable_param_count(model)
    logger.log(f"[mm] trainable_params={tr_params:,} / total_params={all_params:,}")
    module_rows = [
        ("vision", model.vision_adapter.vision_model),
        ("bridge", model.bridge),
        ("semantic_bottleneck", getattr(model.bridge, "semantic_bottleneck", None)),
        ("prefix_calibrator", model.prefix_calibrator),
        ("lm", model.lm),
        ("lm_visual_adapters", model.visual_adapters),
    ]
    for name, module in module_rows:
        if module is None:
            continue
        total_n, train_n = _module_param_summary(module)
        logger.log(
            f"[mm] module={name} total_params={total_n:,} trainable_params={train_n:,} "
            f"frozen_params={max(0, total_n - train_n):,}"
        )
    if hasattr(model.bridge, "core"):
        core_total, core_train = _module_param_summary(model.bridge.core)
        logger.log(
            f"[mm] perceiver_core_frozen={int(core_train == 0)} total_params={core_total:,} trainable_params={core_train:,}"
        )

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if not optim_params:
        raise SystemExit("No trainable parameters. Check freeze mode configuration.")
    vision_lr_scale = float(getattr(args, "vision_lr_scale", 1.0))
    vision_param_ids = {id(p) for p in model.vision_adapter.vision_model.parameters() if p.requires_grad}
    main_params = [p for p in optim_params if id(p) not in vision_param_ids]
    param_groups: List[Dict[str, Any]] = []
    if main_params:
        param_groups.append({"params": main_params, "lr": float(args.lr), "base_lr": float(args.lr)})
    if vision_param_ids:
        vision_params = [p for p in optim_params if id(p) in vision_param_ids]
        param_groups.append(
            {
                "params": vision_params,
                "lr": float(args.lr) * vision_lr_scale,
                "base_lr": float(args.lr) * vision_lr_scale,
            }
        )
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))
    logger.log(
        f"[mm] optimizer=AdamW trainable_param_tensors={len(optim_params)} "
        f"lr={float(args.lr):.6g} vision_lr_scale={vision_lr_scale:.6g} "
        f"weight_decay={float(args.weight_decay):.6g} grad_clip={float(args.grad_clip):.4g}"
    )
    if resume_payload is not None and "optimizer_state_dict" in resume_payload and not args.eval_only:
        try:
            opt.load_state_dict(resume_payload["optimizer_state_dict"])
        except (ValueError, RuntimeError) as e:
            logger.log(f"[mm] optimizer state incompatible (two-stage handoff?), reinitializing optimizer: {e}")
    if resume_payload is not None:
        restore_rng_state(resume_payload.get("rng_state"))

    if args.eval_only:
        if bool(getattr(args, "eval_tiny_head", False)):
            run_tiny_head_probe_eval(
                model=model,
                tokenizer=tokenizer,
                args=args,
                device=device,
                logger=logger,
            )
            return
        if bool(getattr(args, "eval_grounding", False)):
            run_grounding_eval(
                model=model,
                tokenizer=tokenizer,
                args=args,
                device=device,
                logger=logger,
            )
            return
        maybe_cuda_empty_cache(
            logger,
            enabled=bool(args.cuda_empty_cache_after_eval),
            tag="eval_only_pre_eval",
        )
        records = run_generation_predictions(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_answer_length=int(args.max_answer_length),
            max_batches=int(args.eval_batches),
            debug_shapes=bool(args.debug_shapes),
            logger=logger,
            split_name=args.eval_split,
            log_every=int(args.eval_log_every),
            cuda_empty_cache_every=(400 if int(args.eval_batches) == 0 else 0),
        )
        summary = evaluate_records(
            records,
            qualitative_samples=int(args.qualitative_samples),
            confusion_top_k=int(args.confusion_top_k),
            scorer=str(args.eval_scorer),
        )
        print_eval_summary(logger, args.eval_split, summary)
        append_fixed_eval_answers(
            fixed_eval_tracker,
            records,
            global_step=global_step,
            epoch=start_epoch,
            tag="eval_only",
            logger=logger,
        )
        maybe_cuda_empty_cache(
            logger,
            enabled=bool(args.cuda_empty_cache_after_eval),
            tag="eval_only_post_eval",
        )
        maybe_shutdown_loader_workers(
            logger,
            val_loader,
            tag="val_loader_post_eval_only",
        )
        if args.save_predictions_jsonl:
            with open(args.save_predictions_jsonl, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=True) + "\n")
            logger.log(f"[mm] wrote predictions: {args.save_predictions_jsonl}")
        run_final_greedy_sanity_pass(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_answer_length=int(args.max_answer_length),
            sanity_count=int(args.final_sanity_count),
            logger=logger,
            tag="eval_only_final",
        )
        return

    debug_shape_once = bool(args.debug_shapes)
    accum_steps = max(1, int(args.grad_accum_steps))
    train_iter: Iterable[Dict[str, Any]]
    fixed_batch = None
    if args.overfit_small_batch:
        if train_loader is None:
            raise RuntimeError("train_loader missing")
        try:
            fixed_batch = next(iter(train_loader))
        except StopIteration:
            raise RuntimeError("No data available for overfit_small_batch.")
        train_steps_target = int(args.max_steps) if int(args.max_steps) > 0 else int(args.overfit_steps)
        train_iter = [fixed_batch for _ in range(train_steps_target)]
        logger.log(f"[mm] overfit_small_batch enabled steps={train_steps_target}")
    else:
        train_iter = ()
    train_batches_per_epoch = int(len(train_iter) if args.overfit_small_batch else len(train_loader))
    if train_batches_per_epoch <= 0:
        raise SystemExit("No train batches available.")
    if legacy_resume_no_batch and resume_batch_in_epoch <= 0:
        resume_batch_in_epoch = int(train_batches_per_epoch)
    if resume_batch_in_epoch >= train_batches_per_epoch:
        start_epoch += int(resume_batch_in_epoch // train_batches_per_epoch)
        resume_batch_in_epoch = int(resume_batch_in_epoch % train_batches_per_epoch)

    steps_budget = int(args.max_steps) if int(args.max_steps) > 0 else None
    for pg in opt.param_groups:
        pg["base_lr"] = float(pg.get("base_lr", pg["lr"]))

    def _lr_scale(step_id: int) -> float:
        sched = str(args.lr_schedule)
        if sched == "constant":
            return 1.0
        if sched != "cosine":
            raise ValueError(f"Unsupported lr schedule: {sched}")
        warm = max(0, int(args.lr_warmup_steps))
        min_ratio = float(args.lr_min_ratio)
        min_ratio = max(0.0, min(1.0, min_ratio))
        if warm > 0 and step_id <= warm:
            return max(1e-8, float(step_id) / float(warm))
        if steps_budget is None or int(steps_budget) <= 0:
            return 1.0
        if step_id >= int(steps_budget):
            return min_ratio
        denom = max(1, int(steps_budget) - warm)
        progress = (float(step_id) - float(warm)) / float(denom)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    def _set_optimizer_lr(scale: float) -> float:
        for pg in opt.param_groups:
            pg["lr"] = float(pg["base_lr"]) * float(scale)
        return float(opt.param_groups[0]["lr"])

    def _optimizer_step() -> None:
        if use_scaler:
            if float(args.grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(optim_params, float(args.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, float(args.grad_clip))
            opt.step()
        opt.zero_grad(set_to_none=True)

    train_log_time_sec = 0.0
    train_log_steps = 0
    low_train_sps_steps = 0

    def _post_optimizer_step(
        *,
        loss_value: float,
        info_dict: Dict[str, float],
        current_lr: float,
        step_train_elapsed: float,
    ) -> None:
        nonlocal train_log_time_sec, train_log_steps, low_train_sps_steps
        train_log_time_sec += max(0.0, float(step_train_elapsed))
        train_log_steps += 1
        if global_step % int(args.log_every) == 0:
            window_steps = int(train_log_steps)
            tok_count = int(info_dict.get("loss_tokens", 0.0))
            loss_ce = float(info_dict.get("loss_ce", loss_value))
            ratio = info_dict.get("prefix_text_norm_ratio", None)
            reg_norm = info_dict.get("loss_prefix_norm_reg", None)
            reg_var = info_dict.get("loss_prefix_batchvar_reg", None)
            sem_recon = info_dict.get("loss_semantic_recon", None)
            sem_consistency = info_dict.get("loss_semantic_consistency", None)
            loss_ground = info_dict.get("loss_grounding", None)
            mean_kl = info_dict.get("grounding_mean_kl", None)
            attn_entropy = info_dict.get("compression_mean_attn_entropy", None)
            ratio_txt = "" if ratio is None else f" pfx_txt_norm_ratio={float(ratio):.4f}"
            reg_txt = ""
            if reg_norm is not None:
                reg_txt += f" reg_norm={float(reg_norm):.6f}"
            if reg_var is not None:
                reg_txt += f" reg_var={float(reg_var):.6f}"
            if sem_recon is not None:
                reg_txt += f" distill={float(sem_recon):.6f}"
            if sem_consistency is not None:
                reg_txt += f" sem_consistency={float(sem_consistency):.6f}"
            if loss_ground is not None and float(loss_ground) > 0.0:
                reg_txt += f" grounding={float(loss_ground):.6f}"
            if mean_kl is not None and float(mean_kl) > 0.0:
                reg_txt += f" ground_kl={float(mean_kl):.6f}"
            if attn_entropy is not None and float(attn_entropy) > 0.0:
                reg_txt += f" comp_attn_ent={float(attn_entropy):.4f}"
            steps_per_s = float(train_log_steps) / max(1e-6, float(train_log_time_sec))
            logger.log(
                f"[mm] step={global_step} epoch={epoch} loss={float(loss_value):.4f} "
                f"loss_vqa={loss_ce:.4f} loss_tokens={tok_count}{ratio_txt}{reg_txt} "
                f"lr={current_lr:.6g} steps_per_s={steps_per_s:.2f}"
            )
            min_train_sps = float(getattr(args, "min_train_steps_per_s", 0.0))
            min_train_window = max(0, int(getattr(args, "min_train_steps_window", 0)))
            if min_train_sps > 0.0 and min_train_window > 0:
                if steps_per_s < min_train_sps:
                    low_train_sps_steps += max(0, window_steps)
                else:
                    low_train_sps_steps = 0
                if low_train_sps_steps >= min_train_window:
                    ckpt_path = _checkpoint_path(args.run_id, global_step)
                    save_mm_checkpoint(
                        ckpt_path,
                        model,
                        opt,
                        global_step=global_step,
                        epoch=epoch,
                        batch_in_epoch=current_batch_in_epoch,
                        args=args,
                        bridge_cfg=bridge_cfg,
                    )
                    logger.log(f"[mm] checkpoint saved: {ckpt_path}")
                    logger.log(
                        f"[mm] low_train_sps_watchdog triggered threshold={min_train_sps:.4f} "
                        f"window={min_train_window} bad_steps={low_train_sps_steps} "
                        f"measured_steps_per_s={steps_per_s:.4f} exit_code={LOW_TRAIN_SPS_EXIT_CODE}"
                    )
                    raise SystemExit(LOW_TRAIN_SPS_EXIT_CODE)
            train_log_time_sec = 0.0
            train_log_steps = 0

        if int(args.eval_every) > 0 and global_step % int(args.eval_every) == 0:
            maybe_cuda_empty_cache(
                logger,
                enabled=bool(args.cuda_empty_cache_after_eval),
                tag=f"periodic_eval_pre_step_{global_step}",
            )
            records = run_generation_predictions(
                model=model,
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                max_answer_length=int(args.max_answer_length),
                max_batches=int(args.eval_batches),
                debug_shapes=False,
                logger=logger,
                split_name=args.eval_split,
                log_every=int(args.eval_log_every),
            )
            summary = evaluate_records(
                records,
                qualitative_samples=int(args.qualitative_samples),
                confusion_top_k=int(args.confusion_top_k),
                scorer=str(args.eval_scorer),
            )
            print_eval_summary(logger, args.eval_split, summary)
            append_fixed_eval_answers(
                fixed_eval_tracker,
                records,
                global_step=global_step,
                epoch=epoch,
                tag="periodic_eval",
                logger=logger,
            )
            maybe_cuda_empty_cache(
                logger,
                enabled=bool(args.cuda_empty_cache_after_eval),
                tag=f"periodic_eval_post_step_{global_step}",
            )
            maybe_shutdown_loader_workers(
                logger,
                val_loader,
                tag=f"val_loader_post_periodic_eval_step_{global_step}",
            )

        if int(args.ckpt_every) > 0 and global_step % int(args.ckpt_every) == 0:
            ckpt_path = _checkpoint_path(args.run_id, global_step)
            save_mm_checkpoint(
                ckpt_path,
                model,
                opt,
                global_step=global_step,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                args=args,
                bridge_cfg=bridge_cfg,
            )
            logger.log(f"[mm] checkpoint saved: {ckpt_path}")

    epoch = max(1, start_epoch) if start_epoch > 0 else 1
    opt.zero_grad(set_to_none=True)
    accum_count = 0
    accum_loss_sum = 0.0
    accum_info_sums: Dict[str, float] = {}
    optimizer_step_started_at: Optional[float] = None
    current_batch_in_epoch = 0

    while True:
        if args.overfit_small_batch:
            iter_obj = train_iter
        else:
            if train_loader is None:
                raise RuntimeError("train_loader missing")
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, EpochShuffleSampler):
                sampler.set_epoch(epoch)
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if isinstance(batch_sampler, GroundingMixBatchSampler):
                batch_sampler.set_epoch(epoch)
            iter_obj = train_loader

        for batch_in_epoch, batch in enumerate(iter_obj, start=1):
            current_batch_in_epoch = int(batch_in_epoch)
            if epoch == start_epoch and batch_in_epoch <= resume_batch_in_epoch:
                continue
            if accum_count == 0:
                optimizer_step_started_at = time.perf_counter()
            current_lr = _set_optimizer_lr(_lr_scale(global_step + 1))
            _set_module_modes(model, args.freeze_mode)
            batch = _to_device(batch, device)
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with ctx:
                loss, info = model.compute_loss(
                    batch,
                    answer_only=bool(args.loss_on_answer_only),
                    debug_shapes=debug_shape_once,
                )
            debug_shape_once = False

            loss_back = loss / float(accum_steps)
            if use_scaler:
                scaler.scale(loss_back).backward()
            else:
                loss_back.backward()
            accum_count += 1
            accum_loss_sum += float(loss.item())
            for k, v in info.items():
                accum_info_sums[k] = float(accum_info_sums.get(k, 0.0)) + float(v)

            should_step = accum_count >= accum_steps
            if steps_budget is not None and (global_step + 1) >= steps_budget:
                should_step = True
            if not should_step:
                continue

            _optimizer_step()
            global_step += 1
            step_train_elapsed = 0.0 if optimizer_step_started_at is None else (time.perf_counter() - optimizer_step_started_at)
            optimizer_step_started_at = None
            info_step: Dict[str, float] = {}
            for k, v in accum_info_sums.items():
                if k == "loss_tokens":
                    info_step[k] = float(v)
                else:
                    info_step[k] = float(v) / max(1.0, float(accum_count))
            loss_step = float(accum_loss_sum) / max(1.0, float(accum_count))
            accum_count = 0
            accum_loss_sum = 0.0
            accum_info_sums = {}

            _post_optimizer_step(
                loss_value=loss_step,
                info_dict=info_step,
                current_lr=current_lr,
                step_train_elapsed=step_train_elapsed,
            )

            if steps_budget is not None and global_step >= steps_budget:
                break

        if accum_count > 0 and (steps_budget is None or global_step < steps_budget):
            current_lr = _set_optimizer_lr(_lr_scale(global_step + 1))
            _optimizer_step()
            global_step += 1
            step_train_elapsed = 0.0 if optimizer_step_started_at is None else (time.perf_counter() - optimizer_step_started_at)
            optimizer_step_started_at = None
            info_step = {}
            for k, v in accum_info_sums.items():
                if k == "loss_tokens":
                    info_step[k] = float(v)
                else:
                    info_step[k] = float(v) / max(1.0, float(accum_count))
            loss_step = float(accum_loss_sum) / max(1.0, float(accum_count))
            accum_count = 0
            accum_loss_sum = 0.0
            accum_info_sums = {}
            _post_optimizer_step(
                loss_value=loss_step,
                info_dict=info_step,
                current_lr=current_lr,
                step_train_elapsed=step_train_elapsed,
            )

        if steps_budget is not None and global_step >= steps_budget:
            break
        if not args.overfit_small_batch and epoch >= int(args.epochs):
            break
        if args.overfit_small_batch:
            break
        start_epoch = 0
        resume_batch_in_epoch = 0
        epoch += 1

    final_ckpt = _checkpoint_path(args.run_id, global_step)
    save_mm_checkpoint(
        final_ckpt,
        model,
        opt,
        global_step=global_step,
        epoch=epoch,
        batch_in_epoch=int(current_batch_in_epoch),
        args=args,
        bridge_cfg=bridge_cfg,
    )
    logger.log(f"[mm] final checkpoint: {final_ckpt}")

    maybe_shutdown_loader_workers(
        logger,
        train_loader,
        tag="train_loader_pre_final_eval",
    )
    train_loader = None
    gc.collect()

    maybe_cuda_empty_cache(
        logger,
        enabled=bool(args.cuda_empty_cache_after_eval),
        tag="final_eval_pre",
    )
    records = run_generation_predictions(
        model=model,
        loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_answer_length=int(args.max_answer_length),
        max_batches=int(args.final_eval_batches),
        debug_shapes=False,
        logger=logger,
        split_name=args.eval_split,
        log_every=int(args.eval_log_every),
        cuda_empty_cache_every=(400 if int(args.final_eval_batches) == 0 else 0),
    )
    summary = evaluate_records(
        records,
        qualitative_samples=int(args.qualitative_samples),
        confusion_top_k=int(args.confusion_top_k),
        scorer=str(args.eval_scorer),
    )
    print_eval_summary(logger, args.eval_split, summary)
    append_fixed_eval_answers(
        fixed_eval_tracker,
        records,
        global_step=global_step,
        epoch=epoch,
        tag="final_eval",
        logger=logger,
    )
    maybe_cuda_empty_cache(
        logger,
        enabled=bool(args.cuda_empty_cache_after_eval),
        tag="final_eval_post",
    )
    maybe_shutdown_loader_workers(
        logger,
        val_loader,
        tag="val_loader_post_final_eval",
    )
    if args.save_predictions_jsonl:
        with open(args.save_predictions_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        logger.log(f"[mm] wrote predictions: {args.save_predictions_jsonl}")
    run_final_greedy_sanity_pass(
        model=model,
        loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_answer_length=int(args.max_answer_length),
        sanity_count=int(args.final_sanity_count),
        logger=logger,
        tag="train_final",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
