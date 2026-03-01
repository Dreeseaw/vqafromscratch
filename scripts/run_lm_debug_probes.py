"""
Run LM probe-debug metrics from a saved Transformer checkpoint.

Example:
python3 scripts/run_lm_debug_probes.py \
  --ckpt logs/my_run/step_10000.tar \
  --tokenizer logs/my_run/tokenizer.pt \
  --probes data/wiki_tok_256/probes.txt
"""

import argparse
import os

import torch

from models.bpe_tokenizer import ByteBPETokenizer
from models.lm import LMConfig, TransformerV1
from train.lm_probe_debug import parse_probe_layers, parse_probe_prompts, run_debug_probes


def _infer_run_dir(ckpt_path: str, out_dir: str) -> str:
    if out_dir:
        return out_dir
    ckpt_abs = os.path.abspath(ckpt_path)
    parent = os.path.dirname(ckpt_abs)
    if os.path.basename(parent).startswith("logs"):
        return parent
    return parent


def _load_model_from_ckpt(ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg_raw = checkpoint.get("config")
    if not isinstance(cfg_raw, dict):
        raise ValueError("Checkpoint missing config dict.")

    cfg = LMConfig(
        vocab_size=int(cfg_raw["vocab_size"]),
        embed_size=int(cfg_raw["embed_size"]),
        num_heads=int(cfg_raw["num_heads"]),
        mlp_ratio=int(cfg_raw["mlp_ratio"]),
        dropout=float(cfg_raw.get("dropout", 0.1)),
        layers=int(cfg_raw["layers"]),
        max_seq_len=int(cfg_raw["max_seq_len"]),
        tie_embeds=bool(cfg_raw.get("tie_embeds", False)),
        causal_lm=bool(cfg_raw.get("causal_lm", True)),
        activation_checkpointing=bool(cfg_raw.get("activation_checkpointing", False)),
        attn_impl=str(cfg_raw.get("attn_impl", "sdpa")),
        sdp_backend=str(cfg_raw.get("sdp_backend", "auto")),
        cosine_attn=bool(cfg_raw.get("cosine_attn", False)),
        v_rmsnorm=bool(cfg_raw.get("v_rmsnorm", False)),
        layerscale=bool(cfg_raw.get("layerscale", False)),
        layerscale_init=float(cfg_raw.get("layerscale_init", 1e-5)),
    )
    model = TransformerV1(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    step = int(checkpoint.get("global_step", 0))
    return model, cfg, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--probes", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--topk_eigs", type=int, default=5)
    ap.add_argument("--gen_tokens", type=int, default=48)
    ap.add_argument(
        "--probe_attn_entropy",
        dest="probe_attn_entropy",
        action="store_true",
        help="Capture attention entropy metrics during probe runs.",
    )
    ap.add_argument(
        "--no_probe_attn_entropy",
        dest="probe_attn_entropy",
        action="store_false",
        help="Disable attention entropy capture during probe runs.",
    )
    ap.set_defaults(probe_attn_entropy=True)
    ap.add_argument(
        "--probe_layers",
        type=str,
        default="",
        help="Comma-separated layer indices to probe (e.g. 0,1,5). Default uses probe-debug defaults.",
    )
    ap.add_argument("--step", type=int, default=-1, help="Optional override for output step label.")
    ap.add_argument("--log_detailed", action="store_true", help="Also print full per-probe metric lines.")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS requested but not available.")

    model, cfg, ckpt_step = _load_model_from_ckpt(args.ckpt, device=device)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    prompts = parse_probe_prompts(args.probes)
    probe_layers = parse_probe_layers(args.probe_layers, total_layers=int(cfg.layers))

    step = int(args.step) if int(args.step) >= 0 else ckpt_step
    run_dir = _infer_run_dir(args.ckpt, args.out_dir)
    summary = run_debug_probes(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_seq_len=int(cfg.max_seq_len),
        pad_id=int(tokenizer.pad_id),
        run_dir=run_dir,
        step=step,
        topk_eigs=max(1, int(args.topk_eigs)),
        generate_max_new_tokens=max(0, int(args.gen_tokens)),
        probe_layers=probe_layers,
        capture_attn_entropy=bool(args.probe_attn_entropy),
        log_fn=print,
        log_detailed_metrics=bool(args.log_detailed),
    )
    agg = summary.get("aggregate_metrics", {})
    agg_line = " ".join(
        f"{k}={float(v):.6f}" for k, v in agg.items() if isinstance(v, (int, float))
    )
    if agg_line:
        print(f"ProbeDebugSummary Step={summary['step']} probes={summary['num_probes']} {agg_line}")

    print(
        f"Probe debug complete | probes={summary['num_probes']} "
        f"step={summary['step']} artifacts={os.path.join(run_dir, 'probe_debug')}"
    )


if __name__ == "__main__":
    main()
