from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train.mm import _to_device, build_loader, load_runtime_from_checkpoint, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Linear probe over exported MM visual prefix tokens.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--probe_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--images_root", type=str, default=None)
    ap.add_argument("--annotations_root", type=str, default=None)
    ap.add_argument("--limit_train", type=int, default=10000)
    ap.add_argument("--limit_val", type=int, default=5000)
    ap.add_argument("--answer_top_k", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--feature_pool", type=str, default="flatten", choices=["flatten", "mean"])
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def _build_answer_vocab(dataset: Any, top_k: int) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in getattr(dataset, "items", []):
        ans = str(item.get("answer", "")).strip()
        if ans:
            counts[ans] += 1
    vocab = {ans: idx for idx, (ans, _n) in enumerate(counts.most_common(max(1, int(top_k))))}
    return vocab


@torch.no_grad()
def _extract_features(
    model: Any,
    loader: Any,
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
        raise RuntimeError("No usable probe samples after answer-vocab filtering.")
    return torch.stack(feats, dim=0), torch.stack(labels, dim=0), answer_types


def _eval_probe(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    answer_types: List[str],
) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    by_type_total: Dict[str, int] = defaultdict(int)
    by_type_correct: Dict[str, int] = defaultdict(int)
    offset = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=-1)
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


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    overrides = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "images_root": args.images_root,
        "annotations_root": args.annotations_root,
    }
    model, tokenizer, _bridge_cfg, _payload, run_args = load_runtime_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        args_override=overrides,
    )
    if args.images_root:
        run_args.images_root = args.images_root
    if args.annotations_root:
        run_args.annotations_root = args.annotations_root
    run_args.batch_size = int(args.batch_size)
    run_args.eval_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    run_args.prefetch_factor = int(args.prefetch_factor)
    run_args.pin_memory = bool(args.pin_memory)

    train_loader = build_loader(
        run_args,
        tokenizer=tokenizer,
        split="train",
        train_mode=False,
        limit=max(0, int(args.limit_train)),
    )
    val_loader = build_loader(
        run_args,
        tokenizer=tokenizer,
        split="val",
        train_mode=False,
        limit=max(0, int(args.limit_val)),
    )
    answer_to_idx = _build_answer_vocab(train_loader.dataset, int(args.answer_top_k))
    if not answer_to_idx:
        raise RuntimeError("Probe answer vocabulary is empty.")

    train_x, train_y, _train_types = _extract_features(
        model,
        train_loader,
        device=device,
        answer_to_idx=answer_to_idx,
        feature_pool=str(args.feature_pool),
    )
    val_x, val_y, val_types = _extract_features(
        model,
        val_loader,
        device=device,
        answer_to_idx=answer_to_idx,
        feature_pool=str(args.feature_pool),
    )

    probe = nn.Linear(int(train_x.shape[1]), len(answer_to_idx)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    train_probe_loader = DataLoader(train_ds, batch_size=int(args.probe_batch_size), shuffle=True)
    val_probe_loader = DataLoader(val_ds, batch_size=int(args.probe_batch_size), shuffle=False)

    history: List[Dict[str, Any]] = []
    best_summary: Dict[str, Any] | None = None
    best_acc = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        probe.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_probe_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * int(xb.shape[0])
            train_count += int(xb.shape[0])
        summary = _eval_probe(probe, val_probe_loader, device=device, answer_types=val_types)
        summary["epoch"] = int(epoch)
        summary["train_loss"] = train_loss_sum / max(1, train_count)
        history.append(summary)
        if float(summary["accuracy"]) > best_acc:
            best_acc = float(summary["accuracy"])
            best_summary = dict(summary)
        print(
            f"[probe] epoch={epoch} train_loss={summary['train_loss']:.4f} "
            f"val_acc={summary['accuracy']:.4f}"
        )

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "feature_pool": str(args.feature_pool),
        "answer_vocab_size": int(len(answer_to_idx)),
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "best": best_summary,
        "history": history,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[probe] wrote: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
