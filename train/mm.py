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
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.bpe_tokenizer import ByteBPETokenizer
from models.bridge import BridgeConfig, build_bridge
from models.lm import LMConfig, TransformerDecoderOnlyV1
from models.vae import VAEConfig, VariationalAutoEncoder, VariationalAutoEncoderRes, ViTVAE, ViTVAE2
from train.vqa_data import VQAv2Dataset, VQAv2Paths, build_image_transform, prepare_vqav2


LOGDIR = "logs"
LOGFILE = "logfile.txt"


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
        "images_root": "images",
        "annotations_root": "annotations",
        "max_question_length": 64,
        "max_answer_length": 16,
        "max_text_tokens": 256,
        "batch_size": 16,
        "num_workers": 2,
        "prefetch_factor": 2,
        "pin_memory": True,
        "fixed_eval_count": 5,
        "freeze_mode": "bridge_only",
        "train_top_lm_layers": 1,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
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

    def __init__(self, vision_model: nn.Module, feature_mode: str = "auto", feature_source: str = "posterior_mu"):
        super().__init__()
        self.vision_model = vision_model
        self.feature_mode = feature_mode
        self.feature_source = feature_source

    def _extract_raw(self, images: torch.Tensor) -> torch.Tensor:
        src = str(self.feature_source)
        m = self.vision_model
        if src == "encoder":
            if not hasattr(m, "_encoder"):
                raise ValueError("feature_source=encoder requires vision model._encoder")
            return m._encoder(images)
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._extract_raw(images)
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)  # [B,C,H,W] -> [B,N,C]
        if self.feature_mode == "global":
            if x.ndim == 2:
                return x
            if x.ndim == 3:
                return x.mean(dim=1)
            raise ValueError(f"Unsupported feature shape for global mode: {tuple(x.shape)}")
        if self.feature_mode == "token":
            if x.ndim == 3:
                return x
            if x.ndim == 2:
                return x.unsqueeze(1)
            raise ValueError(f"Unsupported feature shape for token mode: {tuple(x.shape)}")
        # auto
        if x.ndim in (2, 3):
            return x
        raise ValueError(f"Unsupported feature shape for auto mode: {tuple(x.shape)}")


class MultimodalPrefixLM(nn.Module):
    def __init__(
        self,
        vision_adapter: VisionFeatureAdapter,
        bridge: nn.Module,
        lm: TransformerDecoderOnlyV1,
        *,
        lm_autocast: bool = True,
    ):
        super().__init__()
        self.vision_adapter = vision_adapter
        self.bridge = bridge
        self.lm = lm
        self.lm_autocast = bool(lm_autocast)

    def _lm_autocast_dtype(self) -> Optional[torch.dtype]:
        if not self.lm_autocast:
            return None
        if not torch.cuda.is_available():
            return None
        if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            return torch.bfloat16
        return torch.float16

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        text_pad_mask: torch.Tensor,
        *,
        debug_shapes: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        visual_features = self.vision_adapter(images)
        visual_prefix = self.bridge(visual_features)
        text_emb = self.lm._embed_dropout(self.lm._embed(input_ids))
        x = torch.cat([visual_prefix, text_emb], dim=1)

        b = int(input_ids.shape[0])
        k = int(visual_prefix.shape[1])
        prefix_pad = torch.zeros((b, k), dtype=torch.bool, device=input_ids.device)
        full_pad = torch.cat([prefix_pad, text_pad_mask], dim=1)
        use_lm_amp = (
            x.device.type == "cuda"
            and str(getattr(self.lm._config, "attn_impl", "sdpa")) == "sdpa"
            and self._lm_autocast_dtype() is not None
        )
        if use_lm_amp:
            with torch.autocast(device_type="cuda", dtype=self._lm_autocast_dtype(), enabled=True):
                hidden = self.lm._decode_only(
                    x,
                    pad_mask=full_pad,
                    is_causal=bool(getattr(self.lm._config, "causal_lm", True)),
                )
                logits = self.lm._lm_head(hidden)
            logits = logits.float()
        else:
            hidden = self.lm._decode_only(
                x,
                pad_mask=full_pad,
                is_causal=bool(getattr(self.lm._config, "causal_lm", True)),
            )
            logits = self.lm._lm_head(hidden)
        if debug_shapes:
            print(f"[mm:shape] images={tuple(images.shape)}")
            print(f"[mm:shape] visual_features={tuple(visual_features.shape)}")
            print(f"[mm:shape] visual_prefix={tuple(visual_prefix.shape)}")
            print(f"[mm:shape] text_ids={tuple(input_ids.shape)} text_emb={tuple(text_emb.shape)}")
            print(f"[mm:shape] combined={tuple(x.shape)} logits={tuple(logits.shape)}")
        return logits, k

    def compute_loss(
        self,
        batch: Dict[str, Any],
        *,
        answer_only: bool = True,
        debug_shapes: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits, prefix_k = self.forward_logits(
            input_ids=batch["input_ids"],
            images=batch["images"],
            text_pad_mask=batch["text_pad_mask"],
            debug_shapes=debug_shapes,
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
        loss = (ce * valid.float()).sum() / denom
        return loss, {"loss_tokens": float(denom.item())}

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

        for _ in range(int(max_new_tokens)):
            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), max_len), int(pad_id), dtype=torch.long, device=images.device)
            text_pad_mask = torch.ones((len(seqs), max_len), dtype=torch.bool, device=images.device)
            for i, s in enumerate(seqs):
                if not s:
                    continue
                t = torch.tensor(s, dtype=torch.long, device=images.device)
                input_ids[i, : len(s)] = t
                text_pad_mask[i, : len(s)] = False

            logits, prefix_k = self.forward_logits(input_ids=input_ids, images=images, text_pad_mask=text_pad_mask)
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
            "has_answer": has_answer,
        }

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        enc = [self._encode_sample(s) for s in samples]
        b = len(samples)
        max_len = max(len(x["input_ids"]) for x in enc) if enc else 1
        input_ids = torch.full((b, max_len), int(self.tok.pad_id), dtype=torch.long)
        text_pad_mask = torch.ones((b, max_len), dtype=torch.bool)
        target_mask = torch.zeros((b, max_len - 1), dtype=torch.bool)
        answer_loss_mask = torch.zeros((b, max_len - 1), dtype=torch.bool)

        for i, e in enumerate(enc):
            ids = e["input_ids"]
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            text_pad_mask[i, :L] = False
            if L > 1:
                target_mask[i, : L - 1] = True
                if e["has_answer"]:
                    start = max(0, int(e["answer_start"]) - 1)
                    answer_loss_mask[i, start : L - 1] = True

        return {
            "images": torch.stack([s["image"] for s in samples], dim=0),
            "input_ids": input_ids,
            "text_pad_mask": text_pad_mask,
            "target_mask": target_mask,
            "answer_loss_mask": answer_loss_mask,
            "prompt_ids": [e["prompt_ids"] for e in enc],
            "question_ids": [int(s["question_id"]) for s in samples],
            "image_ids": [int(s["image_id"]) for s in samples],
            "questions": [s["question"] for s in samples],
            "answers": [s["answer"] for s in samples],
            "all_answers": [s["all_answers"] for s in samples],
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
    ds = VQAv2Dataset(
        images_root=args.images_root,
        annotations_root=args.annotations_root,
        split=split,
        transform=transform,
        limit=limit,
        skip_missing_images=True,
    )
    collator = QACollator(
        tokenizer=tokenizer,
        max_q=args.max_question_length,
        max_a=args.max_answer_length,
        max_text_tokens=args.max_text_tokens,
    )
    kwargs: Dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": bool(train_mode),
        "drop_last": bool(train_mode),
        "num_workers": int(args.num_workers),
        "collate_fn": collator,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
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


def configure_freezing(model: MultimodalPrefixLM, args: argparse.Namespace) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.bridge.parameters():
        p.requires_grad_(True)

    mode = str(args.freeze_mode)
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
    if mode == "full_finetune":
        for p in model.parameters():
            p.requires_grad_(True)
        return
    raise ValueError(f"Unsupported freeze mode: {mode}")


def _set_module_modes(model: MultimodalPrefixLM, freeze_mode: str) -> None:
    model.train()
    if freeze_mode in ("bridge_only", "bridge_plus_top_lm"):
        model.vision_adapter.vision_model.eval()
    if freeze_mode == "bridge_only":
        model.lm.eval()


def _to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = dict(batch)
    for k in ("images", "input_ids", "text_pad_mask", "target_mask", "answer_loss_mask"):
        out[k] = batch[k].to(device)
    return out


def _materialize_lazy_params(model: MultimodalPrefixLM, batch: Dict[str, Any], device: str) -> None:
    with torch.no_grad():
        b = _to_device(batch, device)
        _ = model.forward_logits(
            input_ids=b["input_ids"],
            images=b["images"],
            text_pad_mask=b["text_pad_mask"],
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
    in_dim = None
    for key in ("bridge._token_proj.0.weight", "bridge._global_proj.0.weight"):
        w = state_dict.get(key)
        if torch.is_tensor(w) and w.ndim == 2:
            in_dim = int(w.shape[1])
            break
    if in_dim is None:
        return
    ref = model.lm._embed.weight
    bridge._ensure_built(in_dim, device=ref.device, dtype=ref.dtype)


def save_mm_checkpoint(
    path: str,
    model: MultimodalPrefixLM,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    epoch: int,
    args: argparse.Namespace,
    bridge_cfg: BridgeConfig,
) -> None:
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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

    vision = build_vision_model_from_args(args, device=device, ckpt_payload=checkpoint_payload)
    lm = build_lm_from_args(args, tokenizer=tokenizer, device=device, ckpt_payload=checkpoint_payload)
    feature_mode = args.vision_feature_mode
    feature_source = args.vision_feature_source
    if checkpoint_payload is not None:
        meta = checkpoint_payload.get("vision_meta", {}) or {}
        feature_mode = str(meta.get("feature_mode", feature_mode))
        feature_source = str(meta.get("feature_source", feature_source))
    vision_adapter = VisionFeatureAdapter(
        vision_model=vision,
        feature_mode=feature_mode,
        feature_source=feature_source,
    )

    bcfg_data = {
        "bridge_type": args.bridge_type,
        "num_visual_tokens": args.num_visual_tokens,
        "lm_hidden_size": lm._config.embed_size,
        "bridge_hidden_dim": args.bridge_hidden_dim,
        "input_feature_mode": args.vision_feature_mode,
        "token_reduce": args.bridge_token_reduce,
    }
    if checkpoint_payload is not None and isinstance(checkpoint_payload.get("bridge_config"), dict):
        bcfg_data.update(dict(checkpoint_payload["bridge_config"]))
        bcfg_data["lm_hidden_size"] = int(lm._config.embed_size)
    bridge_cfg = BridgeConfig(**bcfg_data)
    bridge = build_bridge(bridge_cfg).to(device)
    model = MultimodalPrefixLM(
        vision_adapter=vision_adapter,
        bridge=bridge,
        lm=lm,
        lm_autocast=bool(args.mm_lm_autocast),
    ).to(device)
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
                    "all_answers": batch["all_answers"][i],
                    "metadata": batch["metadata"][i],
                }
            )
        if int(log_every) > 0 and ((bidx + 1) % int(log_every) == 0 or (bidx == 0)):
            msg = (
                f"[eval:{split_name}] batch={bidx + 1}"
                + (f"/{max_batches}" if int(max_batches) > 0 else "")
                + f" samples={len(records)} elapsed_s={time.time() - t0:.1f}"
            )
            if logger is not None:
                logger.log(msg)
            else:
                print(msg)
        if max_batches > 0 and (bidx + 1) >= int(max_batches):
            break
    return records


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


def evaluate_records(records: Sequence[Dict[str, Any]], *, qualitative_samples: int, confusion_top_k: int) -> Dict[str, Any]:
    from evals.vqa import (
        build_confusion_summary,
        format_qualitative_samples,
        summarize_vqa_predictions,
    )

    summary = summarize_vqa_predictions(records)
    summary["qualitative"] = format_qualitative_samples(records, n=qualitative_samples)
    summary["confusions"] = build_confusion_summary(records, top_k=confusion_top_k)
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
    logger.log(f"[eval:{split}] overall_accuracy={overall:.4f}")
    by_answer = summary.get("answer_type_accuracy", {})
    if by_answer:
        logger.log(
            "[eval:{split}] answer_type: ".format(split=split)
            + " ".join(f"{k}={v:.4f}" for (k, v) in sorted(by_answer.items()))
        )
    qtype = summary.get("question_type_accuracy", {})
    if qtype:
        top = sorted(qtype.items(), key=lambda kv: kv[1], reverse=True)[:10]
        logger.log(
            "[eval:{split}] top question-type accuracy: ".format(split=split)
            + " | ".join(f"{k}:{v:.3f}" for (k, v) in top)
        )


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
        f"feature_source={args.vision_feature_source} feature_mode={args.vision_feature_mode}"
    )
    logger.log(
        f"[mm] lm ckpt={args.lm_checkpoint} tokenizer={args.tokenizer_path} vocab={tokenizer.vocab_size}"
    )
    logger.log(
        f"[mm] bridge type={bridge_cfg.bridge_type} visual_tokens={bridge_cfg.num_visual_tokens} "
        f"hidden={bridge_cfg.bridge_hidden_dim} token_reduce={bridge_cfg.token_reduce}"
    )
    logger.log(
        f"[mm] freeze_mode={args.freeze_mode} train_top_lm_layers={args.train_top_lm_layers} "
        f"loss_on_answer_only={int(bool(args.loss_on_answer_only))}"
    )
    logger.log(
        f"[mm] seq_lens q={args.max_question_length} a={args.max_answer_length} text={args.max_text_tokens}"
    )
    logger.log(
        f"[mm] dataloader batch_size={args.batch_size} num_workers={args.num_workers} "
        f"prefetch_factor={args.prefetch_factor} pin_memory={int(bool(args.pin_memory))}"
    )
    logger.log(
        f"[mm] data images_root={args.images_root} annotations_root={args.annotations_root} "
        f"limit_train={args.limit_train} limit_val={args.limit_val} limit_eval={args.limit_eval}"
    )
    logger.log(
        f"[mm] loop epochs={args.epochs} max_steps={args.max_steps} overfit_small_batch={int(bool(args.overfit_small_batch))} "
        f"log_every={args.log_every} eval_every={args.eval_every} eval_batches={args.eval_batches} "
        f"eval_log_every={args.eval_log_every} fixed_eval_count={args.fixed_eval_count} ckpt_every={args.ckpt_every}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", type=str)
    ap.add_argument("--checkpoint", type=int, default=None, help="Resume from logs/<run_id>/step_<N>.tar")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"])

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=35)

    ap.add_argument("--vision_model", type=str, default="vitvae2", choices=["vae", "vaer", "vitvae", "vitvae2"])
    ap.add_argument("--vision_checkpoint", type=str, default=None)
    ap.add_argument("--vision_config", type=str, default=None)
    ap.add_argument("--vision_latent_dim", type=int, default=768)
    ap.add_argument("--vision_cbld", type=int, default=1536)
    ap.add_argument(
        "--vision_feature_source",
        type=str,
        default="posterior_mu",
        choices=["posterior_mu", "encoder", "model_output"],
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

    ap.add_argument("--bridge_type", type=str, default="mlp")
    ap.add_argument("--bridge_hidden_dim", type=int, default=1024)
    ap.add_argument("--num_visual_tokens", type=int, default=8)
    ap.add_argument(
        "--bridge_token_reduce",
        type=str,
        default="adaptive_pool",
        choices=["adaptive_pool", "mean_expand", "all"],
    )

    ap.add_argument(
        "--freeze_mode",
        type=str,
        default="bridge_only",
        choices=["bridge_only", "bridge_plus_top_lm", "full_finetune"],
    )
    ap.add_argument("--train_top_lm_layers", type=int, default=1)

    ap.add_argument("--images_root", type=str, default="images")
    ap.add_argument("--annotations_root", type=str, default="annotations")
    ap.add_argument("--auto_download", action="store_true")
    ap.add_argument("--download_images", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--download_test", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--max_question_length", type=int, default=64)
    ap.add_argument("--max_answer_length", type=int, default=16)
    ap.add_argument("--max_text_tokens", type=int, default=256)
    ap.add_argument("--loss_on_answer_only", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--overfit_small_batch", action="store_true")
    ap.add_argument("--overfit_steps", type=int, default=200)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--eval_log_every", type=int, default=10)
    ap.add_argument("--fixed_eval_count", type=int, default=5)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--qualitative_samples", type=int, default=8)
    ap.add_argument("--confusion_top_k", type=int, default=20)
    ap.add_argument("--save_predictions_jsonl", type=str, default=None)
    ap.add_argument("--debug_shapes", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args = _apply_runtime_defaults(args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    amp_enabled, amp_dtype, use_scaler = resolve_amp(device, args.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    run_dir = os.path.join(LOGDIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(run_id=args.run_id, checkpoint_id=args.checkpoint)
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
    if args.checkpoint is not None:
        ckpt_path = _checkpoint_path(args.run_id, int(args.checkpoint))
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
        resume_payload = _load_checkpoint(ckpt_path, map_location="cpu")
        global_step = int(resume_payload.get("global_step", int(args.checkpoint)))
        start_epoch = int(resume_payload.get("epoch", 0))
        logger.log(f"[mm] resuming from {ckpt_path} (step={global_step}, epoch={start_epoch})")

    model, tokenizer, bridge_cfg = build_runtime_from_args(args, device=device, checkpoint_payload=resume_payload)
    if resume_payload is not None:
        _maybe_initialize_bridge_from_state_dict(model, resume_payload["model_state_dict"])
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
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

    configure_freezing(model, args)
    _set_module_modes(model, args.freeze_mode)
    tr_params, all_params = _trainable_param_count(model)
    logger.log(f"[mm] trainable_params={tr_params:,} / total_params={all_params:,}")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if not optim_params:
        raise SystemExit("No trainable parameters. Check freeze mode configuration.")
    opt = torch.optim.AdamW(optim_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    logger.log(
        f"[mm] optimizer=AdamW trainable_param_tensors={len(optim_params)} "
        f"lr={float(args.lr):.6g} weight_decay={float(args.weight_decay):.6g} grad_clip={float(args.grad_clip):.4g}"
    )
    if resume_payload is not None and "optimizer_state_dict" in resume_payload and not args.eval_only:
        opt.load_state_dict(resume_payload["optimizer_state_dict"])

    if args.eval_only:
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
        )
        summary = evaluate_records(
            records,
            qualitative_samples=int(args.qualitative_samples),
            confusion_top_k=int(args.confusion_top_k),
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
        if args.save_predictions_jsonl:
            with open(args.save_predictions_jsonl, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=True) + "\n")
            logger.log(f"[mm] wrote predictions: {args.save_predictions_jsonl}")
        return

    debug_shape_once = bool(args.debug_shapes)
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

    steps_budget = int(args.max_steps) if int(args.max_steps) > 0 else None
    epoch = start_epoch
    start_time = time.time()

    while True:
        epoch += 1
        if args.overfit_small_batch:
            iter_obj = train_iter
        else:
            if train_loader is None:
                raise RuntimeError("train_loader missing")
            iter_obj = train_loader

        for batch in iter_obj:
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

            opt.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                if float(args.grad_clip) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(optim_params, float(args.grad_clip))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(optim_params, float(args.grad_clip))
                opt.step()

            global_step += 1
            if global_step % int(args.log_every) == 0:
                elapsed = max(1e-6, time.time() - start_time)
                tok_count = int(info.get("loss_tokens", 0.0))
                logger.log(
                    f"[mm] step={global_step} epoch={epoch} loss={float(loss.item()):.4f} "
                    f"loss_tokens={tok_count} steps_per_s={global_step / elapsed:.2f}"
                )

            if int(args.eval_every) > 0 and global_step % int(args.eval_every) == 0:
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

            if int(args.ckpt_every) > 0 and global_step % int(args.ckpt_every) == 0:
                ckpt_path = _checkpoint_path(args.run_id, global_step)
                save_mm_checkpoint(
                    ckpt_path,
                    model,
                    opt,
                    global_step=global_step,
                    epoch=epoch,
                    args=args,
                    bridge_cfg=bridge_cfg,
                )
                logger.log(f"[mm] checkpoint saved: {ckpt_path}")

            if steps_budget is not None and global_step >= steps_budget:
                break

        if steps_budget is not None and global_step >= steps_budget:
            break
        if not args.overfit_small_batch and epoch >= int(args.epochs):
            break
        if args.overfit_small_batch:
            break

    final_ckpt = _checkpoint_path(args.run_id, global_step)
    save_mm_checkpoint(
        final_ckpt,
        model,
        opt,
        global_step=global_step,
        epoch=epoch,
        args=args,
        bridge_cfg=bridge_cfg,
    )
    logger.log(f"[mm] final checkpoint: {final_ckpt}")

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
    if args.save_predictions_jsonl:
        with open(args.save_predictions_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        logger.log(f"[mm] wrote predictions: {args.save_predictions_jsonl}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
