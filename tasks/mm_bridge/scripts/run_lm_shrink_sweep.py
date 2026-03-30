from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS_ROOT = REPO_ROOT / "logs"
TOKENIZER_PATH = REPO_ROOT / "logs/mix_bpe_16k/tokenizer.pt"
LM_CANONICAL_CKPT = REPO_ROOT / "logs/lm_final/step_45000.tar"
MM_CEMENT_REF = REPO_ROOT / "logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar"
FORMAT_REMAP_CKPT = REPO_ROOT / "logs/mmsemantic_remap_v1_debug/step_500.tar"
PYTHON_DEFAULT = REPO_ROOT / ".venv_local/bin/python"

PRETRAIN_WIKI = "data/pretraining/wikicoco256_cleaned/train"
PRETRAIN_DISTILL = "data/pretraining/distill256_cleaned2/train"
PRETRAIN_VAL = "data/pretraining/distill256_cleaned2/val"
PRETRAIN_TEST = "data/pretraining/wikicoco256_cleaned/test"
PRETRAIN_PROBES = "data/pretraining/wikicoco256_cleaned/probes.txt"
PRETRAIN_MIX_SCHEDULE = json.dumps(
    [
        {"start_step": 1, "end_step": 2000, "weights": {"distill": 0.0, "wiki": 1.0, "coco": 0.0}},
        {"start_step": 2001, "end_step": 12000, "weights": {"distill": 0.1, "wiki": 0.9, "coco": 0.0}},
        {"start_step": 12001, "end_step": None, "weights": {"distill": 0.3, "wiki": 0.7, "coco": 0.0}},
    ]
)

BASE_LM_PARAMS = 39_859_712
BASE_PRETRAIN_STEP_S = 5.46
BASE_PRETRAIN_TOTAL_HOURS = 45_000.0 / BASE_PRETRAIN_STEP_S / 3600.0
BASE_BRIDGE_MIN = 64.0
BASE_COMPRESSION_MIN = 14.0
POST_PHASE_EVAL_MIN = 10.0

REFERENCE_ROW = {
    "name": "original_pretrained_ref",
    "lm_params": BASE_LM_PARAMS,
    "pretrain_loss": None,
    "bridge": {"overall": 0.6163, "yes/no": 0.7589},
    "compression": {"overall": 0.5900, "yes/no": 0.7393, "number": 0.4499, "other": 0.5134},
    "probe": 0.5103,
}


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    d_model: int
    n_heads: int
    layers: int
    params: int
    pretrain: bool
    use_format_teacher: bool


VARIANTS: List[Variant] = [
    Variant("half", "Half", 512, 8, 6, 24_097_280, True, True),
    Variant("randinit", "RandomInit46M", 512, 8, 12, 39_859_712, False, False),
    Variant("quarter", "Quarter", 384, 6, 4, 12_166_272, True, False),
    Variant("tiny", "Tiny", 256, 4, 3, 6_141_952, True, False),
]


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_hours(hours: float) -> str:
    total_min = int(round(hours * 60.0))
    h = total_min // 60
    m = total_min % 60
    return f"{h}h {m:02d}m"


def fmt_minutes(minutes: float) -> str:
    total_min = int(round(minutes))
    h = total_min // 60
    m = total_min % 60
    if h <= 0:
        return f"{m}m"
    return f"{h}h {m:02d}m"


def resolve_python(python_bin: str) -> str:
    cand = Path(python_bin)
    if cand.is_file():
        return str(cand)
    return str(PYTHON_DEFAULT)


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(cmd: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[lmshrink] CMD: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def maybe_empty_cuda_cache(python_bin: str) -> str:
    cmd = [
        python_bin,
        "-c",
        (
            "import gc, torch; "
            "before=(torch.cuda.memory_reserved()/1024/1024 if torch.cuda.is_available() else 0.0); "
            "gc.collect(); "
            "torch.cuda.empty_cache() if torch.cuda.is_available() else None; "
            "after=(torch.cuda.memory_reserved()/1024/1024 if torch.cuda.is_available() else 0.0); "
            "print(f'before_reserved_mib={before:.2f} after_reserved_mib={after:.2f}')"
        ),
    ]
    out = subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True).strip()
    gc.collect()
    return out


def latest_checkpoint_step(run_dir: Path) -> int:
    best = 0
    for path in run_dir.glob("step_*.tar"):
        m = re.match(r"step_(\d+)\.tar$", path.name)
        if not m:
            continue
        best = max(best, int(m.group(1)))
    return best


def phase_complete(run_dir: Path, target_step: int) -> bool:
    return (run_dir / f"step_{target_step}.tar").is_file()


def parse_logged_final_eval(log_path: Path) -> Optional[Dict[str, Any]]:
    if not log_path.is_file():
        return None
    overall_re = re.compile(r"^\[eval:val\] overall_accuracy=([0-9.]+) scorer=([A-Za-z0-9_]+)")
    answer_re = re.compile(
        r"^\[eval:val\] answer_type:\s+yes/no=([0-9.]+)\s+number=([0-9.]+)\s+other=([0-9.]+)"
    )
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        m_overall = overall_re.match(lines[idx].strip())
        if not m_overall:
            continue
        answer_vals: Optional[Dict[str, float]] = None
        for j in range(idx + 1, min(idx + 6, len(lines))):
            m_answer = answer_re.match(lines[j].strip())
            if m_answer:
                answer_vals = {
                    "yes/no": float(m_answer.group(1)),
                    "number": float(m_answer.group(2)),
                    "other": float(m_answer.group(3)),
                }
                break
        if answer_vals is None:
            continue
        return {
            "overall_accuracy": float(m_overall.group(1)),
            "answer_type_accuracy": answer_vals,
            "scorer": str(m_overall.group(2)),
            "source": f"logfile:{log_path.name}",
        }
    return None


def ensure_eval_json_from_log(run_dir: Path, out_json: Path) -> bool:
    parsed = parse_logged_final_eval(run_dir / "logfile.txt")
    if parsed is None:
        return False
    write_text(out_json, json.dumps(parsed, indent=2, ensure_ascii=True) + "\n")
    return True


def probe_layers_for_variant(layers: int) -> str:
    if layers >= 12:
        picks = [0, 3, 7, 11]
    elif layers == 6:
        picks = [0, 1, 3, 5]
    elif layers == 4:
        picks = [0, 1, 2, 3]
    elif layers == 3:
        picks = [0, 1, 2]
    elif layers == 2:
        picks = [0, 1]
    else:
        picks = [0]
    return ",".join(str(x) for x in picks)


def canonical_pretrain_args(variant: Variant) -> List[str]:
    return [
        "--train_data",
        PRETRAIN_WIKI,
        "--val_data",
        PRETRAIN_VAL,
        "--test_data",
        PRETRAIN_TEST,
        "--train_bucket_distill",
        PRETRAIN_DISTILL,
        "--train_bucket_wiki",
        PRETRAIN_WIKI,
        "--mix_schedule",
        PRETRAIN_MIX_SCHEDULE,
        "--tokenizer",
        str(TOKENIZER_PATH),
        "--decoder_only",
        "--d_model",
        str(variant.d_model),
        "--n_heads",
        str(variant.n_heads),
        "--dec_layers",
        str(variant.layers),
        "--ff_mult",
        "2",
        "--dropout",
        "0.1",
        "--precision",
        "bf16",
        "--tie_embeddings",
        "--attn_impl",
        "sdpa",
        "--sdp_backend",
        "flash",
        "--swiglu",
        "--layerscale_init",
        "0.01",
        "--resid_max_norm",
        "0.0",
        "--cap_attn_out_norm",
        "1.5",
        "--cap_mlp_out_norm",
        "2.0",
        "--cap_out_mode",
        "token",
        "--cap_keep_masked",
        "1",
        "--epochs",
        "100",
        "--batch_size",
        "64",
        "--grad_accum_steps",
        "1",
        "--lr",
        "0.001",
        "--weight_decay",
        "0.05",
        "--optimizer",
        "muon",
        "--muon_ns_steps",
        "5",
        "--muon_min_matrix_dim",
        "100",
        "--row_max_norm_c",
        "2.0",
        "--warmup_ratio",
        "0.04",
        "--schedule",
        "cosine",
        "--max_steps",
        "45000",
        "--eval_every_steps",
        "5000",
        "--val_max_tokens",
        "100000",
        "--run_probes",
        "1000",
        "--probe_file",
        PRETRAIN_PROBES,
        "--probe_layers",
        probe_layers_for_variant(variant.layers),
        "--num_workers",
        "1",
        "--prefetch_factor",
        "4",
        "--eval_num_workers",
        "0",
        "--eval_prefetch_factor",
        "2",
        "--eval_pin_memory",
        "0",
        "--log_every",
        "100",
        "--bucket_width",
        "32",
        "--no_activation_checkpointing",
    ]


def canonical_bridge_args(variant: Variant, lm_checkpoint: Optional[Path]) -> List[str]:
    lm_ckpt_arg = str(lm_checkpoint) if lm_checkpoint is not None else ""
    return [
        "--precision",
        "bf16",
        "--num_workers",
        "4",
        "--prefetch_factor",
        "2",
        "--epochs",
        "400",
        "--max_steps",
        "9000",
        "--manual_max_steps",
        "--log_every",
        "20",
        "--eval_every",
        "1000",
        "--eval_batches",
        "100",
        "--final_eval_batches",
        "0",
        "--eval_log_every",
        "20",
        "--eval_fraction",
        "1.0",
        "--ckpt_every",
        "1000",
        "--eval_scorer",
        "official",
        "--final_sanity_count",
        "0",
        "--cuda_empty_cache_after_eval",
        "--eval_use_kv_cache",
        "--eval_kv_cache_mode",
        "batched",
        "--vision_model",
        "siglip_base",
        "--vision_checkpoint",
        "logs/hf_vision/google_siglip_base_patch16_224",
        "--vision_feature_source",
        "encoder",
        "--vision_feature_mode",
        "auto",
        "--lm_checkpoint",
        lm_ckpt_arg,
        "--lm_d_model",
        str(variant.d_model),
        "--lm_num_heads",
        str(variant.n_heads),
        "--lm_layers",
        str(variant.layers),
        "--lm_mlp_ratio",
        "2",
        "--lm_dropout",
        "0.1",
        "--lm_max_seq_len",
        "256",
        "--batch_size",
        "96",
        "--grad_accum_steps",
        "2",
        "--eval_batch_size",
        "96",
        "--num_visual_tokens",
        "49",
        "--bridge_type",
        "perceiver_resampler",
        "--bridge_query_depth",
        "3",
        "--bridge_num_heads",
        "8",
        "--bridge_token_reduce",
        "adaptive_pool",
        "--bridge_add_2d_pos_emb",
        "--bridge_pre_mixer_type",
        "none",
        "--bridge_question_conditioning",
        "--bridge_query_bank_mode",
        "question_hidden_attn",
        "--bridge_question_context_mode",
        "question_only",
        "--bridge_qquery_scale",
        "1.0",
        "--bridge_qcond_scale",
        "0.5",
        "--bridge_token_selector_type",
        "none",
        "--bridge_token_select_k",
        "0",
        "--prefix_calibration",
        "--prefix_calib_layernorm",
        "--prefix_calib_bias",
        "--prefix_calib_gate_init",
        "1.0",
        "--prefix_geom_mlp_ratio",
        "0.5",
        "--prefix_geom_token_mixer_layers",
        "1",
        "--prefix_norm_target_ratio",
        "4.0",
        "--prefix_norm_reg_weight",
        "0.005",
        "--prefix_batchvar_reg_weight",
        "0.0002",
        "--prefix_dropout",
        "0.03",
        "--freeze_mode",
        "bridge_plus_top_lm",
        "--train_top_lm_layers",
        "2",
        "--lm_visual_adapter_type",
        "cross_attn",
        "--lm_visual_adapter_layers",
        "3",
        "--lm_visual_adapter_num_heads",
        "8",
        "--lm_visual_adapter_dropout",
        "0.0",
        "--lm_visual_adapter_gate_init",
        "0.5",
        "--lr",
        "0.0002",
        "--lr_schedule",
        "cosine",
        "--lr_warmup_steps",
        "600",
        "--lr_min_ratio",
        "0.15",
        "--seed",
        "35",
        "--min_train_steps_per_s",
        "0",
    ]


def canonical_compression_args(
    variant: Variant,
    *,
    bridge_checkpoint: Path,
    lm_checkpoint: Optional[Path],
) -> List[str]:
    lm_ckpt_arg = str(lm_checkpoint) if lm_checkpoint is not None else ""
    args = [
        "--precision",
        "bf16",
        "--num_workers",
        "4",
        "--prefetch_factor",
        "2",
        "--epochs",
        "400",
        "--max_steps",
        "3000",
        "--manual_max_steps",
        "--log_every",
        "20",
        "--eval_every",
        "500",
        "--eval_batches",
        "100",
        "--final_eval_batches",
        "0",
        "--eval_log_every",
        "20",
        "--eval_fraction",
        "1.0",
        "--ckpt_every",
        "500",
        "--eval_scorer",
        "official",
        "--final_sanity_count",
        "0",
        "--cuda_empty_cache_after_eval",
        "--eval_use_kv_cache",
        "--eval_kv_cache_mode",
        "batched",
        "--vision_model",
        "siglip_base",
        "--vision_checkpoint",
        "logs/hf_vision/google_siglip_base_patch16_224",
        "--vision_feature_source",
        "encoder",
        "--vision_feature_mode",
        "auto",
        "--lm_checkpoint",
        lm_ckpt_arg,
        "--lm_d_model",
        str(variant.d_model),
        "--lm_num_heads",
        str(variant.n_heads),
        "--lm_layers",
        str(variant.layers),
        "--lm_mlp_ratio",
        "2",
        "--lm_dropout",
        "0.1",
        "--lm_max_seq_len",
        "256",
        "--batch_size",
        "96",
        "--grad_accum_steps",
        "2",
        "--eval_batch_size",
        "96",
        "--num_visual_tokens",
        "49",
        "--bridge_type",
        "perceiver_resampler",
        "--bridge_query_depth",
        "3",
        "--bridge_num_heads",
        "8",
        "--bridge_token_reduce",
        "adaptive_pool",
        "--bridge_add_2d_pos_emb",
        "--bridge_pre_mixer_type",
        "none",
        "--bridge_question_conditioning",
        "--freeze_mode",
        "semantic_bottleneck_only",
        "--bridge_question_context_mode",
        "question_only",
        "--bridge_query_bank_mode",
        "question_hidden_attn",
        "--bridge_qquery_scale",
        "1.0",
        "--bridge_qcond_scale",
        "0.5",
        "--bridge_token_selector_type",
        "none",
        "--bridge_token_select_k",
        "0",
        "--prefix_calibration",
        "--prefix_calib_layernorm",
        "--prefix_calib_bias",
        "--prefix_calib_gate_init",
        "1.0",
        "--prefix_geom_mlp_ratio",
        "0.5",
        "--prefix_geom_token_mixer_layers",
        "1",
        "--prefix_norm_target_ratio",
        "4.0",
        "--prefix_norm_reg_weight",
        "0.005",
        "--prefix_batchvar_reg_weight",
        "0.0002",
        "--prefix_dropout",
        "0.03",
        "--semantic_bottleneck",
        "--semantic_tokens",
        "8",
        "--semantic_latent_dim",
        "256",
        "--semantic_recon_loss_weight",
        "0.1",
        "--semantic_consistency_loss_weight",
        "0.0",
        "--disable_lm_visual_adapters",
        "--lm_visual_adapter_type",
        "cross_attn",
        "--lm_visual_adapter_layers",
        "3",
        "--lm_visual_adapter_num_heads",
        "8",
        "--lm_visual_adapter_dropout",
        "0.0",
        "--lm_visual_adapter_gate_init",
        "0.5",
        "--lr",
        "0.0002",
        "--lr_schedule",
        "cosine",
        "--lr_warmup_steps",
        "200",
        "--init_from_mm_checkpoint",
        str(bridge_checkpoint),
        "--seed",
        "35",
        "--min_train_steps_per_s",
        "0",
    ]
    if variant.use_format_teacher:
        args.extend(
            [
                "--prefix_remap_present",
                "--no-apply_prefix_remap_in_forward",
                "--prefix_remap_checkpoint",
                str(FORMAT_REMAP_CKPT),
                "--semantic_format_loss_weight",
                "0.3",
                "--semantic_format_loss_final_weight",
                "0.0",
                "--semantic_format_anneal_start_step",
                "2250",
                "--semantic_format_anneal_end_step",
                "3000",
            ]
        )
    else:
        args.extend(
            [
                "--semantic_format_loss_weight",
                "0.0",
                "--semantic_format_loss_final_weight",
                "0.0",
                "--semantic_format_anneal_start_step",
                "0",
                "--semantic_format_anneal_end_step",
                "0",
            ]
        )
    return args


def parse_pretrain_metrics(logfile: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"val_ce": None, "val_ppl": None}
    if not logfile.is_file():
        return out
    for line in logfile.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Validation Step=45000" in line:
            m_ce = re.search(r"CE=([0-9.]+)", line)
            m_ppl = re.search(r"PPL=([0-9.]+)", line)
            if m_ce:
                out["val_ce"] = float(m_ce.group(1))
            if m_ppl:
                out["val_ppl"] = float(m_ppl.group(1))
    return out


def greedy_generate_samples(
    *,
    python_bin: str,
    checkpoint: Path,
    output_json: Path,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    prompts = [
        "The capital of France is",
        "Wikipedia is a",
        "Question: What color is the sky? Answer:",
        "The man walked into the",
        "In the image, the sign says",
    ]
    code = r"""
import inspect, json, torch, sys
from models.bpe_tokenizer import ByteBPETokenizer
from models.lm import LMConfig, TransformerDecoderOnlyV1

ckpt_path = sys.argv[1]
tok_path = sys.argv[2]
out_path = sys.argv[3]
max_new = int(sys.argv[4])
prompts = json.loads(sys.argv[5])
payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg_raw = payload['config']
allowed = {k for k in inspect.signature(LMConfig.__init__).parameters.keys() if k != 'self'}
cfg = LMConfig(**{k: v for k, v in cfg_raw.items() if k in allowed})
model = TransformerDecoderOnlyV1(cfg)
model.load_state_dict(payload['model_state_dict'], strict=True)
model.eval()
tok = ByteBPETokenizer.load(tok_path)
rows = []
with torch.no_grad():
    for prompt in prompts:
        ids = tok.encode(prompt, add_bos=True, add_eos=False).tolist()
        seq = torch.tensor([ids], dtype=torch.long)
        for _ in range(max_new):
            if seq.shape[1] >= int(cfg.max_seq_len):
                break
            logits = model(seq)
            nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
            if nxt == tok.eos_id:
                break
        text = tok.decode(seq[0].tolist(), skip_special=True)
        rows.append({"prompt": prompt, "completion": text})
alpha_ok = 0
for row in rows:
    txt = row['completion']
    alpha = sum(1 for ch in txt if ch.isalpha())
    if alpha >= 12:
        alpha_ok += 1
status = 'PASS' if alpha_ok >= 4 else ('QUESTIONABLE' if alpha_ok >= 2 else 'FAIL')
out = {"checkpoint": ckpt_path, "max_new_tokens": max_new, "status": status, "samples": rows}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=True)
print(status)
"""
    cmd = [
        python_bin,
        "-c",
        code,
        str(checkpoint),
        str(TOKENIZER_PATH),
        str(output_json),
        str(max_new_tokens),
        json.dumps(prompts),
    ]
    result = subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True).strip()
    return {"status": result, "output_json": str(output_json)}


def run_mm_eval(
    *,
    python_bin: str,
    checkpoint: Path,
    output_json: Path,
    disable_adapters: bool,
) -> None:
    cmd = [
        python_bin,
        "-m",
        "tasks.mm_bridge.scripts.mm_format_alignment_eval",
        "--checkpoint",
        str(checkpoint),
        "--batch_size",
        "96",
        "--eval_batches",
        "0",
        "--output_json",
        str(output_json),
    ]
    cmd.append("--disable_lm_visual_adapters" if disable_adapters else "--no-disable_lm_visual_adapters")
    run_cmd(cmd, cwd=REPO_ROOT)


def run_probe(
    *,
    python_bin: str,
    checkpoint: Path,
    output_json: Path,
) -> None:
    cmd = [
        python_bin,
        "-m",
        "tasks.mm_bridge.scripts.mm_semantic_probe",
        "--checkpoint",
        str(checkpoint),
        "--batch_size",
        "96",
        "--probe_batch_size",
        "256",
        "--limit_train",
        "10000",
        "--limit_val",
        "5000",
        "--answer_top_k",
        "3000",
        "--epochs",
        "10",
        "--lr",
        "0.001",
        "--feature_pool",
        "flatten",
        "--output_json",
        str(output_json),
    ]
    run_cmd(cmd, cwd=REPO_ROOT)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pretrain_hours_estimate(variant: Variant) -> float:
    if not variant.pretrain:
        return 0.0
    scale = float(variant.params) / float(BASE_LM_PARAMS)
    return BASE_PRETRAIN_TOTAL_HOURS * scale


def block_hours_estimate(variant: Variant) -> float:
    hrs = pretrain_hours_estimate(variant)
    hrs += BASE_BRIDGE_MIN / 60.0
    hrs += BASE_COMPRESSION_MIN / 60.0
    hrs += POST_PHASE_EVAL_MIN / 60.0
    return hrs


def write_inspection_doc(bundle_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# LM Shrink Sweep Codebase Inspection")
    lines.append("")
    lines.append(f"Generated: `{now_ts()}`")
    lines.append("")
    lines.append("## Canonical LM")
    lines.append("")
    lines.append("- checkpoint: `logs/lm_final/step_45000.tar`")
    lines.append("- params: `39,859,712`")
    lines.append("- config: `d_model=512`, `n_heads=8`, `layers=12`, `ff_mult=2`, `vocab=16279`, `max_seq_len=256`")
    lines.append("- optimizer: `Muon`")
    lines.append("- precision: `bf16`")
    lines.append("- data: `wikicoco256_cleaned/train` + `distill256_cleaned2/train` via staged mix schedule")
    lines.append("")
    lines.append("## Canonical Bridge Recipe")
    lines.append("")
    lines.append("- frozen `SigLIP-B/16`")
    lines.append("- `question_only`")
    lines.append("- `question_hidden_attn`")
    lines.append("- perceiver depth `3`, no dynbudget")
    lines.append("- LM adapter depth `3`")
    lines.append("- real Cement freeze recipe: top `2` LM layers trainable")
    lines.append("- `batch=96`, `grad_accum=2`, `eval_batch=96`, `steps=9000`")
    lines.append("")
    lines.append("## Canonical Format-Alignment Recipe")
    lines.append("")
    lines.append("- semantic bottleneck `K=8`, latent dim `256`")
    lines.append("- frozen VM + frozen perceiver + frozen LM")
    lines.append("- adapters disabled in forward")
    lines.append("- `L_vqa + 0.1 * L_distill + L_format` when teacher geometry is compatible")
    lines.append("- `batch=96`, `grad_accum=2`, `steps=3000`, `warmup=200`")
    lines.append("")
    lines.append("## Stage Timing Estimates")
    lines.append("")
    lines.append("- LM pretrain canonical reference: about `2h 17m` to `step_45000`")
    lines.append("  Derived from checkpoint mtimes between `step_5000.tar` and `step_45000.tar` in `logs/lm_final`.")
    lines.append("- Cement bridge block: about `64m`")
    lines.append("  Derived from the successful Seed 42 runs in `logs/mmcement_v1_20260316_175847/timeline.log`.")
    lines.append("- Format-alignment block: about `14m`")
    lines.append("  Derived from `logs/mmsemantic_format_v1_20260324_105300/launcher.log`.")
    lines.append("")
    lines.append("## Variant Table")
    lines.append("")
    lines.append("| Variant | Shape | Params | Pretrain | L_format teacher | Est. block time |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for variant in VARIANTS:
        lines.append(
            f"| {variant.label} | `d={variant.d_model}, h={variant.n_heads}, L={variant.layers}` | "
            f"`{variant.params:,}` | `{int(variant.pretrain)}` | `{int(variant.use_format_teacher)}` | `{fmt_hours(block_hours_estimate(variant))}` |"
        )
    total_hours = sum(block_hours_estimate(v) for v in VARIANTS)
    lines.append("")
    lines.append(f"Estimated total for full overnight execution: `{fmt_hours(total_hours)}`")
    lines.append("")
    lines.append("## Recommended Order")
    lines.append("")
    lines.append("1. `Half`")
    lines.append("2. `RandomInit46M`")
    lines.append("3. `Quarter`")
    lines.append("4. `Tiny`")
    lines.append("")
    lines.append("Why this order:")
    lines.append("- `Half` answers the main shrink question first.")
    lines.append("- `RandomInit46M` is a short, high-value control.")
    lines.append("- `Quarter` and `Tiny` then chase the knee of the curve.")
    lines.append("")
    write_text(bundle_dir / "codebase_inspection.md", "\n".join(lines).rstrip() + "\n")


def progress_header(bundle_dir: Path) -> None:
    if (bundle_dir / "progress.md").is_file():
        return
    lines = [
        f"# LM Shrink Sweep Progress",
        "",
        f"Bundle: `{bundle_dir.name}`",
        f"Started: `{now_ts()}`",
        "",
    ]
    write_text(bundle_dir / "progress.md", "\n".join(lines))


def append_phase_progress(bundle_dir: Path, block_label: str, phase_label: str, items: Dict[str, Any]) -> None:
    lines = [f"## {block_label}", "", f"### {phase_label}"]
    for key, value in items.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    append_text(bundle_dir / "progress.md", "\n".join(lines))


def pretrain_run_id(bundle: str, variant: Variant) -> str:
    return f"{bundle}_pretrain_{variant.name}"


def bridge_run_id(bundle: str, variant: Variant) -> str:
    return f"{bundle}_bridge_{variant.name}"


def compression_run_id(bundle: str, variant: Variant) -> str:
    return f"{bundle}_compression_{variant.name}"


def run_pretrain_phase(bundle_dir: Path, bundle_name: str, variant: Variant, python_bin: str) -> Optional[Path]:
    if not variant.pretrain:
        return None
    run_id = pretrain_run_id(bundle_name, variant)
    run_dir = LOGS_ROOT / run_id
    target_step = 45000
    start = time.time()
    if not phase_complete(run_dir, target_step):
        resume_step = latest_checkpoint_step(run_dir)
        cmd = [python_bin, "-m", "train.train_transformer", run_id]
        if resume_step > 0:
            cmd.extend(["--checkpoint", str(resume_step)])
        cmd.extend(canonical_pretrain_args(variant))
        run_cmd(cmd, cwd=REPO_ROOT)
    metrics = parse_pretrain_metrics(run_dir / "logfile.txt")
    sanity_json = run_dir / "sanity_samples.json"
    sanity: Dict[str, Any]
    try:
        sanity = greedy_generate_samples(
            python_bin=python_bin,
            checkpoint=run_dir / f"step_{target_step}.tar",
            output_json=sanity_json,
        )
    except Exception as exc:
        sanity = {"status": f"ERROR ({type(exc).__name__})", "output_json": str(sanity_json)}
        append_phase_progress(
            bundle_dir,
            f"Block {variant.label}",
            f"Pretrain {variant.label} Sanity Note",
            {
                "Time": now_ts(),
                "Error": f"`{type(exc).__name__}: {exc}`",
            },
        )
    append_phase_progress(
        bundle_dir,
        f"Block {variant.label}",
        f"Pretrain {variant.label}",
        {
            "Started": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
            "LM params": f"{variant.params:,}",
            "Final pretrain CE": f"{metrics.get('val_ce'):.4f}" if metrics.get("val_ce") is not None else "-",
            "Sanity check": sanity["status"],
            "Duration": fmt_minutes((time.time() - start) / 60.0),
            "Run dir": f"`logs/{run_id}`",
        },
    )
    return run_dir / f"step_{target_step}.tar"


def run_bridge_phase(bundle_dir: Path, bundle_name: str, variant: Variant, python_bin: str, lm_checkpoint: Optional[Path]) -> Path:
    run_id = bridge_run_id(bundle_name, variant)
    run_dir = LOGS_ROOT / run_id
    target_step = 9000
    start = time.time()
    if not phase_complete(run_dir, target_step):
        resume_step = latest_checkpoint_step(run_dir)
        cmd = [str(REPO_ROOT / "runmm.sh"), run_id]
        if resume_step > 0:
            cmd.append(str(resume_step))
        cmd.extend(canonical_bridge_args(variant, lm_checkpoint))
        run_cmd(cmd, cwd=REPO_ROOT)
    full_eval_json = run_dir / "bridge_full_eval.json"
    if not full_eval_json.is_file():
        if not ensure_eval_json_from_log(run_dir, full_eval_json):
            run_mm_eval(
                python_bin=python_bin,
                checkpoint=run_dir / f"step_{target_step}.tar",
                output_json=full_eval_json,
                disable_adapters=False,
            )
    metrics = load_json(full_eval_json)
    at = metrics.get("answer_type_accuracy", {})
    append_phase_progress(
        bundle_dir,
        f"Block {variant.label}",
        f"Bridge {variant.label}",
        {
            "Started": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
            "Best eval (step 9000)": (
                f"overall {float(metrics.get('overall_accuracy', 0.0)):.4f}, "
                f"y/n {float(at.get('yes/no', 0.0)):.4f}, "
                f"num {float(at.get('number', 0.0)):.4f}, "
                f"other {float(at.get('other', 0.0)):.4f}"
            ),
            "Duration": fmt_minutes((time.time() - start) / 60.0),
            "Run dir": f"`logs/{run_id}`",
        },
    )
    return run_dir / f"step_{target_step}.tar"


def run_compression_phase(
    bundle_dir: Path,
    bundle_name: str,
    variant: Variant,
    python_bin: str,
    bridge_checkpoint: Path,
    lm_checkpoint: Optional[Path],
) -> Path:
    run_id = compression_run_id(bundle_name, variant)
    run_dir = LOGS_ROOT / run_id
    target_step = 3000
    start = time.time()
    if not phase_complete(run_dir, target_step):
        resume_step = latest_checkpoint_step(run_dir)
        cmd = [str(REPO_ROOT / "runmm.sh"), run_id]
        if resume_step > 0:
            cmd.append(str(resume_step))
        cmd.extend(canonical_compression_args(variant, bridge_checkpoint=bridge_checkpoint, lm_checkpoint=lm_checkpoint))
        run_cmd(cmd, cwd=REPO_ROOT)
    full_eval_json = run_dir / "compression_full_eval.json"
    if not full_eval_json.is_file():
        if not ensure_eval_json_from_log(run_dir, full_eval_json):
            run_mm_eval(
                python_bin=python_bin,
                checkpoint=run_dir / f"step_{target_step}.tar",
                output_json=full_eval_json,
                disable_adapters=True,
            )
    probe_json = run_dir / "tiny_head_probe.json"
    if not probe_json.is_file():
        run_probe(
            python_bin=python_bin,
            checkpoint=run_dir / f"step_{target_step}.tar",
            output_json=probe_json,
        )
    metrics = load_json(full_eval_json)
    probe = load_json(probe_json)
    at = metrics.get("answer_type_accuracy", {})
    append_phase_progress(
        bundle_dir,
        f"Block {variant.label}",
        f"Compression {variant.label}",
        {
            "Started": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
            "L_format used": "yes" if variant.use_format_teacher else "no",
            "Best eval (step 3000)": (
                f"overall {float(metrics.get('overall_accuracy', 0.0)):.4f}, "
                f"y/n {float(at.get('yes/no', 0.0)):.4f}, "
                f"num {float(at.get('number', 0.0)):.4f}, "
                f"other {float(at.get('other', 0.0)):.4f}"
            ),
            "Probe": f"{float(probe.get('best', {}).get('accuracy', 0.0)):.4f}",
            "Duration": fmt_minutes((time.time() - start) / 60.0),
            "Run dir": f"`logs/{run_id}`",
        },
    )
    return run_dir / f"step_{target_step}.tar"


def build_report(bundle_dir: Path, bundle_name: str) -> None:
    rows: List[Dict[str, Any]] = [REFERENCE_ROW]
    for variant in VARIANTS:
        row: Dict[str, Any] = {
            "name": variant.name,
            "label": variant.label,
            "lm_params": variant.params,
            "pretrain_loss": None,
            "bridge": None,
            "compression": None,
            "probe": None,
            "l_format": variant.use_format_teacher,
        }
        pre_dir = LOGS_ROOT / pretrain_run_id(bundle_name, variant)
        if variant.pretrain and (pre_dir / "logfile.txt").is_file():
            row["pretrain_loss"] = parse_pretrain_metrics(pre_dir / "logfile.txt").get("val_ce")
        bridge_eval = LOGS_ROOT / bridge_run_id(bundle_name, variant) / "bridge_full_eval.json"
        bridge_metrics = load_json(bridge_eval) if bridge_eval.is_file() else parse_logged_final_eval(LOGS_ROOT / bridge_run_id(bundle_name, variant) / "logfile.txt")
        if bridge_metrics:
            be = bridge_metrics
            row["bridge"] = {
                "overall": float(be.get("overall_accuracy", 0.0)),
                "yes/no": float(be.get("answer_type_accuracy", {}).get("yes/no", 0.0)),
                "number": float(be.get("answer_type_accuracy", {}).get("number", 0.0)),
                "other": float(be.get("answer_type_accuracy", {}).get("other", 0.0)),
            }
        comp_eval = LOGS_ROOT / compression_run_id(bundle_name, variant) / "compression_full_eval.json"
        comp_metrics = load_json(comp_eval) if comp_eval.is_file() else parse_logged_final_eval(LOGS_ROOT / compression_run_id(bundle_name, variant) / "logfile.txt")
        if comp_metrics:
            ce = comp_metrics
            row["compression"] = {
                "overall": float(ce.get("overall_accuracy", 0.0)),
                "yes/no": float(ce.get("answer_type_accuracy", {}).get("yes/no", 0.0)),
                "number": float(ce.get("answer_type_accuracy", {}).get("number", 0.0)),
                "other": float(ce.get("answer_type_accuracy", {}).get("other", 0.0)),
            }
        probe_json = LOGS_ROOT / compression_run_id(bundle_name, variant) / "tiny_head_probe.json"
        if probe_json.is_file():
            row["probe"] = float(load_json(probe_json).get("best", {}).get("accuracy", 0.0))
        rows.append(row)

    lines: List[str] = []
    lines.append("# LM Shrink Report")
    lines.append("")
    lines.append(f"Bundle: `{bundle_dir.name}`")
    lines.append(f"Generated: `{now_ts()}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| LM variant | LM params | Pretrain loss | Bridge overall | Bridge Y/N | Compressed overall | Compressed Y/N | Compressed Num | Compressed Other | Probe |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        bridge = row.get("bridge") or {}
        comp = row.get("compression") or {}
        lines.append(
            f"| {row.get('label', row['name'])} | {int(row['lm_params']):,} | "
            f"{('-' if row.get('pretrain_loss') is None else f'{float(row['pretrain_loss']):.4f}')} | "
            f"{('-' if not bridge else f'{float(bridge.get('overall', 0.0)):.4f}')} | "
            f"{('-' if not bridge else f'{float(bridge.get('yes/no', 0.0)):.4f}')} | "
            f"{('-' if not comp else f'{float(comp.get('overall', 0.0)):.4f}')} | "
            f"{('-' if not comp else f'{float(comp.get('yes/no', 0.0)):.4f}')} | "
            f"{('-' if not comp else f'{float(comp.get('number', 0.0)):.4f}')} | "
            f"{('-' if not comp else f'{float(comp.get('other', 0.0)):.4f}')} | "
            f"{('-' if row.get('probe') is None else f'{float(row['probe']):.4f}')} |"
        )
    lines.append("")
    lines.append("## Read")
    lines.append("")
    completed = [r for r in rows[1:] if r.get("compression")]
    if not completed:
        lines.append("No variant completed the full three-stage pipeline yet.")
    else:
        best = max(completed, key=lambda x: float(x["compression"]["overall"]))
        lines.append(
            f"- Best completed compressed variant so far: `{best['label']}` at `{float(best['compression']['overall']):.4f}` overall."
        )
        for row in completed:
            bridge = row.get("bridge")
            comp = row.get("compression")
            if bridge and comp:
                delta = float(comp["overall"]) - float(bridge["overall"])
                lines.append(
                    f"- `{row['label']}` bridge->compression delta: `{delta:+.4f}` "
                    f"({float(bridge['overall']):.4f} -> {float(comp['overall']):.4f})."
                )
        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        lines.append("- If bridge accuracy stays high while compressed accuracy drops, the LM can still support VQA but the K=8 alignment path is getting harder.")
        lines.append("- If bridge accuracy already drops with LM size, the LM itself is the bottleneck before compression.")
        lines.append("- The `RandomInit46M` row isolates how much LM pretraining matters independent of raw LM size.")
    write_text(bundle_dir / "lm_shrink_report.md", "\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sequential overnight LM shrink sweep.")
    ap.add_argument("--run_id", type=str, default="")
    ap.add_argument("--python_bin", type=str, default=str(PYTHON_DEFAULT))
    ap.add_argument("--variants", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    python_bin = resolve_python(args.python_bin)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = args.run_id.strip() or f"mmsemantic_lmshrink_v1_{stamp}"
    bundle_dir = LOGS_ROOT / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    selected_names = [x.strip() for x in args.variants.split(",") if x.strip()]
    active_variants = [v for v in VARIANTS if (not selected_names or v.name in selected_names)]

    progress_header(bundle_dir)
    if not (bundle_dir / "codebase_inspection.md").is_file():
        write_inspection_doc(bundle_dir)

    for variant in active_variants:
        try:
            lm_ckpt = run_pretrain_phase(bundle_dir, bundle_name, variant, python_bin) if variant.pretrain else None
            bridge_ckpt = run_bridge_phase(bundle_dir, bundle_name, variant, python_bin, lm_ckpt)
            _ = maybe_empty_cuda_cache(python_bin)
            _ = run_compression_phase(bundle_dir, bundle_name, variant, python_bin, bridge_ckpt, lm_ckpt)
            _ = maybe_empty_cuda_cache(python_bin)
        except Exception as exc:
            append_phase_progress(
                bundle_dir,
                f"Block {variant.label}",
                "Failure",
                {
                    "Time": now_ts(),
                    "Error": f"`{type(exc).__name__}: {exc}`",
                },
            )
            print(f"[lmshrink] ERROR in {variant.label}: {exc}", file=sys.stderr, flush=True)
        finally:
            clear_info = maybe_empty_cuda_cache(python_bin)
            append_text(bundle_dir / "progress.md", f"- VRAM clear: `{clear_info}`\n\n")
            build_report(bundle_dir, bundle_name)

    print(str(bundle_dir))


if __name__ == "__main__":
    main()
