import json
import os
import inspect
from contextlib import contextmanager
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


def parse_probe_layers(raw: Optional[str], total_layers: Optional[int] = None) -> Optional[List[int]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None

    parts = [x.strip() for x in txt.split(",") if x.strip() != ""]
    if not parts:
        raise ValueError("--probe_layers was provided but no valid layer indices were found.")

    out: List[int] = []
    seen = set()
    for part in parts:
        idx = int(part)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)

    if total_layers is not None:
        max_idx = int(total_layers) - 1
        for idx in out:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"probe layer index {idx} out of range [0, {max_idx}]")
    return out


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


def _forward_with_optional_cache(
    model: torch.nn.Module,
    seq: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    use_cache: Optional[bool] = None,
    **kwargs,
):
    call_kwargs: Dict[str, Any] = {"pad_mask": pad_mask, **kwargs}
    if use_cache is not None and _model_supports_use_cache(model):
        call_kwargs["use_cache"] = bool(use_cache)
    with _probe_sdpa_ctx(model=model, device=seq.device):
        return model(seq, **call_kwargs)


def _model_supports_use_cache(model: torch.nn.Module) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    return "use_cache" in params


@contextmanager
def _probe_sdpa_ctx(model: torch.nn.Module, device: torch.device):
    if device.type != "cuda":
        yield
        return
    cfg = getattr(model, "_config", None)
    if getattr(cfg, "attn_impl", None) != "sdpa":
        yield
        return

    saved_modules = []
    for module in model.modules():
        attn_impl = getattr(module, "_attn_impl", None)
        if attn_impl is None:
            continue
        saved_modules.append((module, attn_impl))
        module._attn_impl = "eager"

    saved_cfg_attn_impl = None
    if cfg is not None and hasattr(cfg, "attn_impl"):
        saved_cfg_attn_impl = getattr(cfg, "attn_impl")
        cfg.attn_impl = "eager"

    try:
        yield
    finally:
        if saved_cfg_attn_impl is not None:
            cfg.attn_impl = saved_cfg_attn_impl
        for module, attn_impl in saved_modules:
            module._attn_impl = attn_impl


def _print_probe_invariant_failure(
    tag: str,
    tokenizer,
    pad_id: int,
    prompt: Optional[str] = None,
    token_ids: Optional[List[int]] = None,
    max_new_tokens: Optional[int] = None,
) -> None:
    bos_id = int(getattr(tokenizer, "bos_id", -1))
    eos_id = int(getattr(tokenizer, "eos_id", -1))
    msg = [f"[probe-invariant-failed] {tag} bos_id={bos_id} eos_id={eos_id} pad_id={int(pad_id)}"]
    if max_new_tokens is not None:
        msg.append(f"max_new_tokens={int(max_new_tokens)}")
    if prompt is not None:
        msg.append(f"prompt={prompt!r}")
    if token_ids is not None:
        msg.append(f"token_ids_tail={list(token_ids)[-20:]}")
    print(" | ".join(msg))


@torch.inference_mode()
def _autoregressive_generate(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    pad_id: int,
    max_seq_len: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    if hasattr(model, "generate") and callable(getattr(model, "generate")):
        eos_id = int(getattr(tokenizer, "eos_id", -1))
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "num_beams": 1,
            "return_dict_in_generate": False,
            "output_scores": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "pad_token_id": int(pad_id),
        }
        if _model_supports_use_cache(model):
            gen_kwargs["use_cache"] = True
        if eos_id >= 0:
            gen_kwargs["eos_token_id"] = eos_id
        with _probe_sdpa_ctx(model=model, device=prompt_ids.device):
            seq_out = model.generate(prompt_ids, **gen_kwargs)
        if torch.is_tensor(seq_out):
            full_ids_t = seq_out[0]
        else:
            full_ids_t = seq_out.sequences[0]
        full_token_ids = [int(x) for x in full_ids_t.detach().cpu().tolist()]
        prompt_token_ids = [int(x) for x in prompt_ids[0].detach().cpu().tolist()]
        gen_len = max(0, len(full_token_ids) - len(prompt_token_ids))
        generated_ids = full_token_ids[len(prompt_token_ids):]
        if len(prompt_token_ids) + gen_len >= int(max_seq_len):
            stop_reason = "max_seq_len"
        elif eos_id >= 0 and len(generated_ids) > 0 and int(generated_ids[-1]) == eos_id:
            stop_reason = "eos"
        else:
            stop_reason = "max_new_tokens"
        del seq_out, full_ids_t
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

    cur = prompt_ids.clone()
    eos_id = int(getattr(tokenizer, "eos_id", -1))
    generated_ids: List[int] = []
    stop_reason = "max_new_tokens"
    finished = torch.zeros(cur.size(0), dtype=torch.bool, device=cur.device)

    for _ in range(max(0, int(max_new_tokens))):
        if cur.size(1) >= int(max_seq_len):
            stop_reason = "max_seq_len"
            break
        attention_mask = cur.ne(int(pad_id))
        last_nonpad = attention_mask.long().sum(dim=1) - 1
        if bool((last_nonpad < 0).any().item()):
            stop_reason = "invalid_prompt"
            break
        pad_mask = ~attention_mask
        logits = _forward_with_optional_cache(model, cur, pad_mask=pad_mask, use_cache=True)
        gather_idx = last_nonpad.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        step_logits = logits.gather(dim=1, index=gather_idx).squeeze(1)
        next_ids = torch.argmax(step_logits, dim=-1)
        next_tok = next_ids.view(-1, 1).to(dtype=cur.dtype, device=cur.device)
        cur = torch.cat([cur, next_tok], dim=1)
        generated_ids.append(int(next_ids[0].item()))
        if eos_id >= 0:
            finished = finished | next_ids.eq(eos_id)
        del logits, step_logits, next_tok, gather_idx, attention_mask, last_nonpad
        if bool(finished.all().item()):
            stop_reason = "eos"
            break

    prompt_token_ids = prompt_ids[0].detach().cpu().tolist()
    full_token_ids = cur[0].detach().cpu().tolist()
    del cur

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


@torch.inference_mode()
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
    probe_layers: Optional[List[int]] = None,
    capture_attn_entropy: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
    log_detailed_metrics: bool = False,
) -> Dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    probe_root = os.path.join(run_dir, "probe_debug")
    step_dir = os.path.join(probe_root, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    model_was_training = bool(model.training)
    model.eval()

    summary: Dict[str, Any] = {
        "step": int(step),
        "num_probes": len(prompts),
        "probe_layers": [int(x) for x in probe_layers] if probe_layers is not None else None,
        "capture_attn_entropy": bool(capture_attn_entropy),
        "probes": [],
    }
    latest_attention: Optional[Dict[str, Any]] = None
    attention_entries_for_step: List[Dict[str, Any]] = []
    attention_index_path = os.path.join(probe_root, "attention_index.json")
    agg_values: Dict[str, List[float]] = {}
    generation_records: List[Dict[str, Any]] = []
    eos_id = int(getattr(tokenizer, "eos_id", -1))

    try:
        assert int(generate_max_new_tokens) > 0
        assert int(pad_id) != eos_id
    except AssertionError:
        _print_probe_invariant_failure(
            tag="run_config",
            tokenizer=tokenizer,
            pad_id=int(pad_id),
            max_new_tokens=int(generate_max_new_tokens),
        )
        raise

    if len(prompts) >= 2:
        try:
            ids_a = tokenizer.encode(prompts[0], add_bos=True, add_eos=False)[:max_seq_len]
            ids_b = tokenizer.encode(prompts[1], add_bos=True, add_eos=False)[:max_seq_len]
            t = max(int(ids_a.numel()), int(ids_b.numel()))
            assert t > 0
            batch_ids = torch.full((2, t), int(pad_id), dtype=torch.long)
            if ids_a.numel() > 0:
                batch_ids[0, : ids_a.numel()] = ids_a
            if ids_b.numel() > 0:
                batch_ids[1, : ids_b.numel()] = ids_b
            attention_mask = batch_ids.ne(int(pad_id))
            for i in range(2):
                last_nonpad = int(attention_mask[i].sum().item()) - 1
                assert last_nonpad >= 0
                assert int(batch_ids[i, last_nonpad].item()) != int(pad_id)
        except AssertionError:
            print("[probe-invariant-failed] padded_batch_check")
            print(f"shape input_ids={tuple(batch_ids.shape)} attention_mask={tuple(attention_mask.shape)}")
            for i in range(2):
                last_nonpad = int(attention_mask[i].sum().item()) - 1
                tail_ids = batch_ids[i, -20:].tolist()
                tail_mask = attention_mask[i, -20:].long().tolist()
                ln_tok = int(batch_ids[i, last_nonpad].item()) if last_nonpad >= 0 else None
                print(
                    f"sample={i} last_nonpad={last_nonpad} tok_last_nonpad={ln_tok} tok_last={int(batch_ids[i, -1].item())} "
                    f"tail_ids={tail_ids} tail_mask={tail_mask}"
                )
            raise

    for probe_idx, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        ids = ids[:max_seq_len]
        if probe_idx == 0:
            try:
                assert ids.numel() > 0
                assert int(ids[-1].item()) != eos_id
            except AssertionError:
                _print_probe_invariant_failure(
                    tag="prompt_encoding",
                    tokenizer=tokenizer,
                    pad_id=int(pad_id),
                    prompt=prompt,
                    token_ids=[int(x) for x in ids.detach().cpu().tolist()],
                    max_new_tokens=int(generate_max_new_tokens),
                )
                raise
        input_ids = ids.unsqueeze(0).to(device)
        prompt_token_ids = [int(x) for x in ids.detach().cpu().tolist()]
        pad_mask = input_ids.eq(int(pad_id))

        debug_layers = None
        if probe_layers is not None:
            debug_layers = {int(x) for x in probe_layers}
        attn_state = None
        if capture_attn_entropy:
            _, debug, attn_state = _forward_with_optional_cache(
                model=model,
                seq=input_ids,
                pad_mask=pad_mask,
                use_cache=False,
                return_debug=True,
                return_attn_entropy=True,
                debug_layers=debug_layers,
                debug_score_layers=debug_layers,
            )
        else:
            _, debug = _forward_with_optional_cache(
                model=model,
                seq=input_ids,
                pad_mask=pad_mask,
                use_cache=False,
                return_debug=True,
                debug_layers=debug_layers,
                debug_score_layers=debug_layers,
            )

        if capture_attn_entropy and isinstance(attn_state, dict):
            for debug_key, metric_key in (
                ("encoder_layers", "encoder_layers"),
                ("decoder_self_layers", "decoder_self_layers"),
                ("decoder_cross_layers", "decoder_cross_layers"),
            ):
                dbg_rows = debug.get(debug_key, [])
                met_rows = attn_state.get(metric_key, [])
                met_by_layer = {}
                for row in met_rows:
                    if not isinstance(row, dict):
                        continue
                    layer = row.get("layer")
                    value = row.get("attn_entropy")
                    if isinstance(layer, int) and isinstance(value, (float, int)):
                        met_by_layer[int(layer)] = float(value)
                for row in dbg_rows:
                    if not isinstance(row, dict):
                        continue
                    layer = row.get("layer")
                    if isinstance(layer, int) and int(layer) in met_by_layer:
                        row["attn_entropy"] = float(met_by_layer[int(layer)])

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
                    "attn_entropy",
                ):
                    if metric_name in entry:
                        metric_val = _to_float(entry[metric_name])
                        probe_metrics[f"{prefix}_{metric_name}"] = metric_val
                        agg_values.setdefault(metric_name, []).append(metric_val)
                for score_name in (
                    "attn_score_mean",
                    "attn_score_std",
                    "attn_score_min",
                    "attn_score_max",
                    "attn_presoftmax_std",
                    "attn_presoftmax_max",
                    "attn_presoftmax_p99",
                ):
                    if score_name in entry:
                        score_val = _to_float(entry[score_name])
                        probe_metrics[f"{prefix}_{score_name}"] = score_val
                        agg_values.setdefault(score_name, []).append(score_val)
                        if score_name.startswith("attn_presoftmax_"):
                            agg_values.setdefault(f"{scope_key}_{score_name}", []).append(score_val)

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
        del generation, debug, attn_state, input_ids, pad_mask

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
        "attn_presoftmax_std": _mean_finite(agg_values.get("attn_presoftmax_std", [])),
        "attn_presoftmax_max": _mean_finite(agg_values.get("attn_presoftmax_max", [])),
        "attn_presoftmax_p99": _mean_finite(agg_values.get("attn_presoftmax_p99", [])),
        "enc_self_attn_presoftmax_std": _mean_finite(agg_values.get("enc_attn_presoftmax_std", [])),
        "enc_self_attn_presoftmax_max": _mean_finite(agg_values.get("enc_attn_presoftmax_max", [])),
        "enc_self_attn_presoftmax_p99": _mean_finite(agg_values.get("enc_attn_presoftmax_p99", [])),
        "dec_self_attn_presoftmax_std": _mean_finite(agg_values.get("dec_self_attn_presoftmax_std", [])),
        "dec_self_attn_presoftmax_max": _mean_finite(agg_values.get("dec_self_attn_presoftmax_max", [])),
        "dec_self_attn_presoftmax_p99": _mean_finite(agg_values.get("dec_self_attn_presoftmax_p99", [])),
        "cross_attn_presoftmax_std": _mean_finite(agg_values.get("dec_cross_attn_presoftmax_std", [])),
        "cross_attn_presoftmax_max": _mean_finite(agg_values.get("dec_cross_attn_presoftmax_max", [])),
        "cross_attn_presoftmax_p99": _mean_finite(agg_values.get("dec_cross_attn_presoftmax_p99", [])),
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
