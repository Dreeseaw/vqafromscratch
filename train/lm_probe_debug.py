import json
import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


def parse_probe_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    prompts = [x.strip() for x in raw.split("\n---\n") if x.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}. Expected prompts separated by '\\n---\\n'.")
    return prompts


def resolve_probe_file_path(
    train_data_arg: str,
    resolved_train_path: Optional[str],
    override_probe_file: Optional[str] = None,
) -> str:
    if override_probe_file:
        if not os.path.isfile(override_probe_file):
            raise FileNotFoundError(f"Probe file not found: {override_probe_file}")
        return override_probe_file

    candidates: List[str] = []
    if os.path.isdir(train_data_arg):
        candidates.append(os.path.join(train_data_arg, "probes.txt"))
    if os.path.isfile(train_data_arg):
        candidates.append(os.path.join(os.path.dirname(train_data_arg), "probes.txt"))
    if resolved_train_path:
        candidates.append(os.path.join(os.path.dirname(resolved_train_path), "probes.txt"))
        candidates.append(os.path.join(resolved_train_path, "probes.txt"))

    seen = set()
    for c in candidates:
        c_abs = os.path.abspath(c)
        if c_abs in seen:
            continue
        seen.add(c_abs)
        if os.path.isfile(c_abs):
            return c_abs

    root_hint = None
    if os.path.isdir(train_data_arg):
        root_hint = train_data_arg
    elif os.path.isfile(train_data_arg):
        root_hint = os.path.dirname(train_data_arg)
    elif resolved_train_path:
        root_hint = os.path.dirname(resolved_train_path)
    if not root_hint:
        root_hint = "."
    raise FileNotFoundError(
        f"Could not find probes.txt. Looked near train data. Expected: {os.path.join(root_hint, 'probes.txt')}"
    )


def _fmt(v: float) -> str:
    if v != v:
        return "nan"
    return f"{v:.6f}"


def _to_float(v: Any) -> float:
    if isinstance(v, (float, int)):
        return float(v)
    if torch.is_tensor(v):
        return float(v.item())
    return float(v)


def _hidden_metrics(hidden: torch.Tensor, pad_mask: Optional[torch.Tensor], topk_eigs: int) -> Dict[str, float]:
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden state [B,S,E], got shape={tuple(hidden.shape)}")

    x = hidden.detach()
    if pad_mask is not None:
        keep = ~pad_mask.to(device=x.device, dtype=torch.bool)
        if keep.shape[:2] != x.shape[:2]:
            keep = None
    else:
        keep = None

    if keep is not None:
        x_tok = x[keep]
    else:
        x_tok = x.reshape(-1, x.size(-1))

    out: Dict[str, float] = {}
    if x_tok.size(0) < 2:
        out["pair_cos_mean"] = float("nan")
        out["lambda1_over_trace"] = float("nan")
        for i in range(topk_eigs):
            out[f"spectrum_decay_top{i+1}"] = float("nan")
        return out

    x_float = x_tok.float()
    x_norm = F.normalize(x_float, dim=-1, eps=1e-12)
    sim = torch.matmul(x_norm, x_norm.transpose(0, 1))
    n = sim.size(0)
    pair_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    pair_vals = sim[pair_mask]
    out["pair_cos_mean"] = float(pair_vals.mean().item()) if pair_vals.numel() > 0 else float("nan")

    # Use SVD on centered token vectors to avoid explicit E x E covariance construction on accelerator.
    x_cpu = x_float.cpu()
    x_center = x_cpu - x_cpu.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(x_center)
    denom = max(1, x_center.size(0) - 1)
    eigvals = (svals * svals) / float(denom)
    trace = float(eigvals.sum().item())

    if trace <= 0.0:
        out["lambda1_over_trace"] = float("nan")
        for i in range(topk_eigs):
            out[f"spectrum_decay_top{i+1}"] = float("nan")
    else:
        out["lambda1_over_trace"] = float(eigvals[0].item()) / trace
        for i in range(topk_eigs):
            if i < eigvals.numel():
                out[f"spectrum_decay_top{i+1}"] = float(eigvals[i].item()) / trace
            else:
                out[f"spectrum_decay_top{i+1}"] = float("nan")

    return out


def _serialize_layer_debug(entry: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in entry.items():
        if key in ("attn_prob_grid", "attn_score_grid"):
            if torch.is_tensor(value):
                out[key] = value.detach().cpu().tolist()
            else:
                out[key] = value
        elif key == "layer":
            out[key] = int(value)
        else:
            out[key] = _to_float(value)
    return out


def _line_from_metrics(step: int, metrics: Dict[str, float]) -> str:
    items = [f"Step={step}"]
    for key in sorted(metrics.keys()):
        items.append(f"{key}={_fmt(metrics[key])}")
    return "\nProbeDebug " + " ".join(items)


def _mean_finite(values: List[float]) -> float:
    keep = [float(v) for v in values if isinstance(v, (int, float)) and float(v) == float(v)]
    if not keep:
        return float("nan")
    return float(sum(keep) / len(keep))


def _decode_token_list(tokenizer, token_ids: List[int]) -> List[str]:
    out: List[str] = []
    for tok_id in token_ids:
        try:
            text = tokenizer.decode([int(tok_id)], skip_special=False)
            out.append(str(text))
        except Exception:
            out.append(f"<id:{int(tok_id)}>")
    return out


def _slice_tokens_for_grid(tokenizer, token_ids: List[int], rows: int, cols: int) -> Dict[str, Any]:
    q_ids = token_ids[: max(0, int(rows))]
    k_ids = token_ids[: max(0, int(cols))]
    return {
        "q_token_ids": [int(x) for x in q_ids],
        "k_token_ids": [int(x) for x in k_ids],
        "q_tokens": _decode_token_list(tokenizer, q_ids),
        "k_tokens": _decode_token_list(tokenizer, k_ids),
    }


@torch.no_grad()
def _autoregressive_generate(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    pad_id: int,
    max_seq_len: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    cur = prompt_ids.clone()
    eos_id = int(getattr(tokenizer, "eos_id", -1))
    generated_ids: List[int] = []
    stop_reason = "max_new_tokens"

    for _ in range(max(0, int(max_new_tokens))):
        if cur.size(1) >= int(max_seq_len):
            stop_reason = "max_seq_len"
            break
        pad_mask = cur.eq(int(pad_id))
        logits = model(cur, pad_mask=pad_mask)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        next_tok = torch.tensor([[next_id]], dtype=cur.dtype, device=cur.device)
        cur = torch.cat([cur, next_tok], dim=1)
        generated_ids.append(next_id)
        if eos_id >= 0 and next_id == eos_id:
            stop_reason = "eos"
            break

    prompt_token_ids = prompt_ids[0].detach().cpu().tolist()
    full_token_ids = cur[0].detach().cpu().tolist()

    return {
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_ids,
        "full_token_ids": full_token_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special=True) if generated_ids else "",
        "generated_text_with_special": tokenizer.decode(generated_ids, skip_special=False) if generated_ids else "",
        "full_text": tokenizer.decode(full_token_ids, skip_special=True),
        "full_text_with_special": tokenizer.decode(full_token_ids, skip_special=False),
        "stop_reason": stop_reason,
        "max_new_tokens": int(max_new_tokens),
    }


@torch.no_grad()
def run_debug_probes(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    device: str,
    max_seq_len: int,
    pad_id: int,
    run_dir: str,
    step: int,
    topk_eigs: int = 5,
    generate_max_new_tokens: int = 48,
    log_fn: Optional[Callable[[str], None]] = None,
    log_detailed_metrics: bool = False,
) -> Dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    probe_root = os.path.join(run_dir, "probe_debug")
    step_dir = os.path.join(probe_root, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    model_was_training = bool(model.training)
    model.eval()

    summary: Dict[str, Any] = {"step": int(step), "num_probes": len(prompts), "probes": []}
    latest_attention: Optional[Dict[str, Any]] = None
    attention_entries_for_step: List[Dict[str, Any]] = []
    attention_index_path = os.path.join(probe_root, "attention_index.json")
    agg_values: Dict[str, List[float]] = {}
    generation_records: List[Dict[str, Any]] = []

    for probe_idx, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=True)
        ids = ids[:max_seq_len]
        input_ids = ids.unsqueeze(0).to(device)
        prompt_token_ids = [int(x) for x in ids.detach().cpu().tolist()]
        pad_mask = input_ids.eq(int(pad_id))

        _, debug = model(input_ids, pad_mask=pad_mask, return_debug=True)
        probe_metrics: Dict[str, float] = {}
        generation = _autoregressive_generate(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=input_ids,
            pad_id=int(pad_id),
            max_seq_len=int(max_seq_len),
            max_new_tokens=int(generate_max_new_tokens),
        )
        probe_record: Dict[str, Any] = {
            "probe_idx": int(probe_idx),
            "prompt": prompt,
            "seq_len": int(input_ids.size(1)),
            "generation": generation,
            "encoder_layers": [],
            "decoder_self_layers": [],
            "decoder_cross_layers": [],
            "hidden_metrics": {},
        }

        for scope_key, entries in (
            ("enc", debug.get("encoder_layers", [])),
            ("dec_self", debug.get("decoder_self_layers", [])),
            ("dec_cross", debug.get("decoder_cross_layers", [])),
        ):
            out_entries = []
            for entry in entries:
                layer = int(entry.get("layer", -1))
                prefix = f"probe{probe_idx}_{scope_key}_l{layer}"
                for metric_name in (
                    "q_mag_mean",
                    "q_mag_std",
                    "k_mag_mean",
                    "k_mag_std",
                    "v_mag_mean",
                    "v_mag_std",
                ):
                    if metric_name in entry:
                        metric_val = _to_float(entry[metric_name])
                        probe_metrics[f"{prefix}_{metric_name}"] = metric_val
                        agg_values.setdefault(metric_name, []).append(metric_val)
                for score_name in ("attn_score_mean", "attn_score_std", "attn_score_min", "attn_score_max"):
                    if score_name in entry:
                        score_val = _to_float(entry[score_name])
                        probe_metrics[f"{prefix}_{score_name}"] = score_val
                        agg_values.setdefault(score_name, []).append(score_val)

                if latest_attention is None and "attn_prob_grid" in entry:
                    grid_shape = entry["attn_prob_grid"].shape if torch.is_tensor(entry["attn_prob_grid"]) else ()
                    rows = int(grid_shape[0]) if len(grid_shape) > 0 else 0
                    cols = int(grid_shape[1]) if len(grid_shape) > 1 else 0
                    token_views = _slice_tokens_for_grid(
                        tokenizer=tokenizer,
                        token_ids=prompt_token_ids,
                        rows=rows,
                        cols=cols,
                    )
                    latest_attention = {
                        "step": int(step),
                        "probe_idx": int(probe_idx),
                        "scope": scope_key,
                        "layer": int(layer),
                        "prompt": prompt,
                        "attn_prob_grid": entry["attn_prob_grid"].detach().cpu().tolist(),
                        "attn_score_grid": entry["attn_score_grid"].detach().cpu().tolist()
                        if "attn_score_grid" in entry
                        else None,
                        **token_views,
                    }

                for kind, key in (("attn_prob", "attn_prob_grid"), ("attn_score", "attn_score_grid")):
                    if key not in entry:
                        continue
                    grid_t = entry[key]
                    if not torch.is_tensor(grid_t):
                        continue
                    rows = int(grid_t.size(0))
                    cols = int(grid_t.size(1)) if grid_t.ndim > 1 else 0
                    token_views = _slice_tokens_for_grid(
                        tokenizer=tokenizer,
                        token_ids=prompt_token_ids,
                        rows=rows,
                        cols=cols,
                    )
                    file_name = f"attn_probe{probe_idx}_{scope_key}_l{layer}_{kind}.json"
                    rel_path = os.path.join(f"step_{step}", file_name)
                    out_path = os.path.join(probe_root, rel_path)
                    payload = {
                        "step": int(step),
                        "probe_idx": int(probe_idx),
                        "scope": scope_key,
                        "layer": int(layer),
                        "kind": kind,
                        "prompt": prompt,
                        "grid": grid_t.detach().cpu().tolist(),
                        **token_views,
                    }
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                    attention_entries_for_step.append(
                        {
                            "id": f"step{step}_probe{probe_idx}_{scope_key}_l{layer}_{kind}",
                            "step": int(step),
                            "probe_idx": int(probe_idx),
                            "scope": scope_key,
                            "layer": int(layer),
                            "kind": kind,
                            "prompt": prompt,
                            "file": rel_path,
                        }
                    )

                out_entries.append(_serialize_layer_debug(entry))
            if scope_key == "enc":
                probe_record["encoder_layers"] = out_entries
            elif scope_key == "dec_self":
                probe_record["decoder_self_layers"] = out_entries
            else:
                probe_record["decoder_cross_layers"] = out_entries

        hidden = debug.get("hidden", {})
        for hidden_key, hidden_tensor in hidden.items():
            metrics = _hidden_metrics(hidden_tensor, pad_mask=pad_mask, topk_eigs=max(1, int(topk_eigs)))
            probe_record["hidden_metrics"][hidden_key] = metrics
            prefix = f"probe{probe_idx}_{hidden_key}"
            for metric_name, value in metrics.items():
                probe_metrics[f"{prefix}_{metric_name}"] = float(value)
                agg_values.setdefault(f"{hidden_key}_{metric_name}", []).append(float(value))

        probe_record["flat_metrics"] = probe_metrics
        summary["probes"].append(probe_record)
        generation_records.append(
            {
                "step": int(step),
                "probe_idx": int(probe_idx),
                "prompt": prompt,
                "generated_text": generation.get("generated_text", ""),
                "full_text": generation.get("full_text", ""),
                "stop_reason": generation.get("stop_reason", ""),
                "generated_token_ids": generation.get("generated_token_ids", []),
            }
        )

        if log_fn is not None and log_detailed_metrics:
            log_fn(_line_from_metrics(step=step, metrics=probe_metrics))

        out_json = os.path.join(step_dir, f"probe_{probe_idx}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(probe_record, f)

    if latest_attention is not None:
        latest_path = os.path.join(probe_root, "latest_attention.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(latest_attention, f)

    previous_entries: List[Dict[str, Any]] = []
    if os.path.isfile(attention_index_path):
        try:
            with open(attention_index_path, "r", encoding="utf-8") as f:
                previous_entries = json.load(f) or []
        except Exception:
            previous_entries = []
    previous_entries = [
        x
        for x in previous_entries
        if isinstance(x, dict) and int(x.get("step", -1)) != int(step)
    ]
    attention_index = previous_entries + attention_entries_for_step
    attention_index.sort(key=lambda x: (int(x.get("step", -1)), str(x.get("id", ""))))
    with open(attention_index_path, "w", encoding="utf-8") as f:
        json.dump(attention_index, f)

    aggregate_metrics = {
        "q_mag_mean": _mean_finite(agg_values.get("q_mag_mean", [])),
        "q_mag_std": _mean_finite(agg_values.get("q_mag_std", [])),
        "k_mag_mean": _mean_finite(agg_values.get("k_mag_mean", [])),
        "k_mag_std": _mean_finite(agg_values.get("k_mag_std", [])),
        "v_mag_mean": _mean_finite(agg_values.get("v_mag_mean", [])),
        "v_mag_std": _mean_finite(agg_values.get("v_mag_std", [])),
        "attn_score_mean": _mean_finite(agg_values.get("attn_score_mean", [])),
        "embed_pair_cos_mean": _mean_finite(agg_values.get("embed_pair_cos_mean", [])),
        "enc_l0_pair_cos_mean": _mean_finite(agg_values.get("enc_l0_pair_cos_mean", [])),
        "enc_last_pair_cos_mean": _mean_finite(agg_values.get("enc_last_pair_cos_mean", [])),
        "embed_lambda1_over_trace": _mean_finite(agg_values.get("embed_lambda1_over_trace", [])),
        "enc_last_lambda1_over_trace": _mean_finite(agg_values.get("enc_last_lambda1_over_trace", [])),
    }
    summary["aggregate_metrics"] = aggregate_metrics
    summary["attention_grids"] = attention_entries_for_step
    summary["generations"] = generation_records

    summary_path = os.path.join(step_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f)

    if model_was_training:
        model.train()
    return summary
