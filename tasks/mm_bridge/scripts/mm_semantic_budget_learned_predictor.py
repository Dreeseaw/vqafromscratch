from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train.vqa_data import heuristic_question_category, normalize_text_answer, normalize_vqa_answer


OCR_PATTERNS = [
    "what text",
    "what word",
    "what words",
    "what number is",
    "what does the",
    "what is written",
    "what brand",
    "what is the name",
]
CHART_PATTERNS = [
    "axis",
    "legend",
    "bar",
    "line",
    "trend",
    "value",
    "label",
    "title",
    "percent",
    "chart",
    "graph",
    "plot",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def _stable_bucket(dataset_name: str, question_id: int, mod: int = 10) -> int:
    key = f"{dataset_name}:{int(question_id)}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % int(mod)


def _question_prefix(question: str, n: int = 2) -> str:
    toks = [tok for tok in normalize_vqa_answer(question).split() if tok]
    return " ".join(toks[:n]) if toks else ""


def _question_length_bucket(question: str) -> str:
    n = len([tok for tok in str(question).split() if tok])
    if n <= 4:
        return "short_0_4"
    if n <= 8:
        return "medium_5_8"
    return "long_9_plus"


def _lexical_features(question: str, dataset_name: str) -> List[float]:
    q = normalize_text_answer(question)
    qcat = heuristic_question_category(q)
    words = [tok for tok in q.split() if tok]
    has_digit = 1.0 if any(ch.isdigit() for ch in q) else 0.0
    return [
        float(len(words)),
        float(len(q)),
        1.0 if q.startswith(("is ", "are ", "do ", "does ", "can ", "was ", "were ")) else 0.0,
        1.0 if q.startswith("how many") or " number of " in f" {q} " or qcat == "count" else 0.0,
        1.0 if any(p in q for p in OCR_PATTERNS) else 0.0,
        1.0 if any(p in q for p in CHART_PATTERNS) else 0.0,
        has_digit,
        1.0 if dataset_name == "vqav2" else 0.0,
        1.0 if dataset_name == "chartqa" else 0.0,
        1.0 if dataset_name == "textocr_readout" else 0.0,
        1.0 if dataset_name == "gqa" else 0.0,
    ]


class BudgetMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_standardizer(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-5)
    return mean, std


def _apply_standardizer(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _train_binary_model(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: str,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Tuple[BudgetMLP, Dict[str, torch.Tensor], Dict[str, Any]]:
    mean, std = _fit_standardizer(x_train)
    x_train_std = _apply_standardizer(x_train, mean, std)
    x_val_std = _apply_standardizer(x_val, mean, std)

    model = BudgetMLP(int(x_train.shape[1]), hidden_dim=int(hidden_dim), dropout=float(dropout)).to(device)
    pos_count = float(y_train.sum().item())
    neg_count = float(max(1, int(y_train.numel()) - int(y_train.sum().item())))
    pos_weight = torch.tensor([max(1.0, neg_count / max(1.0, pos_count))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_loader = DataLoader(
        TensorDataset(x_train_std, y_train),
        batch_size=int(batch_size),
        shuffle=True,
    )
    best_state: Dict[str, Any] | None = None
    best_val_acc = -1.0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device=device, dtype=torch.float32)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * int(xb.shape[0])
            train_count += int(xb.shape[0])

        model.eval()
        with torch.no_grad():
            logits = model(x_val_std.to(device))
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= 0.5).to(dtype=torch.float32)
            val_acc = float((preds == y_val).float().mean().item())
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss_sum / max(1, train_count)),
                "val_acc@0.5": float(val_acc),
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "mean": mean.detach().cpu(),
                "std": std.detach().cpu(),
            }

    assert best_state is not None
    model.load_state_dict(best_state["model"])
    model.to(device)
    artifacts = {
        "mean": best_state["mean"],
        "std": best_state["std"],
    }
    return model, artifacts, {"best_val_acc@0.5": best_val_acc, "history": history}


def _predict_probs(model: BudgetMLP, x: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x_std = _apply_standardizer(x, mean, std).to(device)
        return torch.sigmoid(model(x_std)).cpu()


def _load_source(entry: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = str(entry["dataset_name"])
    eval_dir = Path(entry["eval_dir"]).resolve()
    feature_dir = Path(entry["feature_dir"]).resolve()

    fixed_maps: Dict[int, Dict[int, Dict[str, Any]]] = {}
    meta_by_qid: Dict[int, Dict[str, Any]] = {}
    for budget in (2, 4, 8, 16):
        rows: Dict[int, Dict[str, Any]] = {}
        for row in _iter_jsonl(eval_dir / f"fixed_k{budget}_records.jsonl"):
            qid = int(row["question_id"])
            rows[qid] = {
                "accuracy": float(row.get("accuracy", 0.0) or 0.0),
                "answer_type": str(row.get("answer_type", "other")),
                "question_type": str(row.get("question_type", "")),
                "question": str(row.get("question", "")),
                "mean_selected_prob": float(row.get("mean_selected_prob", 0.0) or 0.0),
                "mean_margin": float(row.get("mean_margin", 0.0) or 0.0),
                "mean_entropy": float(row.get("mean_entropy", 0.0) or 0.0),
            }
            if qid not in meta_by_qid:
                meta_by_qid[qid] = {
                    "question": str(row.get("question", "")),
                    "answer_type": str(row.get("answer_type", "other")),
                    "question_type": str(row.get("question_type", "")),
                }
        fixed_maps[int(budget)] = rows

    oracle_budget_by_qid = {
        int(row["question_id"]): int(row["oracle_selected_budget"])
        for row in _iter_jsonl(eval_dir / "oracle_records.jsonl")
    }
    summary = _read_json(eval_dir / "summary.json")

    features: Dict[int, Dict[str, Any]] = {}
    for budget in (2, 4, 8):
        payload = torch.load(feature_dir / f"features_k{budget}.pt", map_location="cpu")
        qids = [int(v) for v in payload["question_ids"].tolist()]
        prefix = payload["prefix_features"].to(dtype=torch.float32)
        question = payload["question_features"].to(dtype=torch.float32)
        prefix_norm = payload["prefix_mean_norm"].to(dtype=torch.float32)
        for idx, qid in enumerate(qids):
            features.setdefault(qid, {})[budget] = {
                "prefix": prefix[idx],
                "question": question[idx],
                "prefix_norm": float(prefix_norm[idx].item()),
            }

    samples: List[Dict[str, Any]] = []
    for qid in sorted(oracle_budget_by_qid.keys()):
        if qid not in features or any(b not in features[qid] for b in (2, 4, 8)):
            continue
        meta = meta_by_qid[qid]
        sample = {
            "dataset_name": dataset_name,
            "question_id": int(qid),
            "question": str(meta["question"]),
            "answer_type": str(meta["answer_type"]),
            "question_type": str(meta["question_type"]),
            "oracle_budget": int(oracle_budget_by_qid[qid]),
            "fixed_accuracy": {budget: float(fixed_maps[budget][qid]["accuracy"]) for budget in (2, 4, 8, 16)},
            "stats": {
                budget: {
                    "mean_selected_prob": float(fixed_maps[budget][qid].get("mean_selected_prob", 0.0)),
                    "mean_margin": float(fixed_maps[budget][qid].get("mean_margin", 0.0)),
                    "mean_entropy": float(fixed_maps[budget][qid].get("mean_entropy", 0.0)),
                }
                for budget in (2, 4, 8)
            },
            "features": features[qid],
        }
        samples.append(sample)
    return {
        "dataset_name": dataset_name,
        "samples": samples,
        "summary": summary,
    }


def _compose_node_features(sample: Dict[str, Any], *, budget: int) -> torch.Tensor:
    feat = sample["features"][int(budget)]
    stats = sample["stats"][int(budget)]
    lexical = torch.tensor(_lexical_features(sample["question"], sample["dataset_name"]), dtype=torch.float32)
    scalars = torch.tensor(
        [
            float(stats.get("mean_selected_prob", 0.0)),
            float(stats.get("mean_margin", 0.0)),
            float(stats.get("mean_entropy", 0.0)),
            float(feat.get("prefix_norm", 0.0)),
        ],
        dtype=torch.float32,
    )
    return torch.cat([feat["prefix"], feat["question"], scalars, lexical], dim=0)


def _split_indices(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    out = {"train": [], "val": [], "report": [], "full": list(range(len(samples)))}
    for idx, sample in enumerate(samples):
        bucket = _stable_bucket(sample["dataset_name"], int(sample["question_id"]))
        if bucket == 8:
            out["val"].append(idx)
        elif bucket == 9:
            out["report"].append(idx)
        else:
            out["train"].append(idx)
    return out


def _score_selection(samples: Sequence[Dict[str, Any]], selected_budgets: Dict[int, int], indices: Sequence[int]) -> Dict[str, Any]:
    total = 0.0
    total_budget = 0.0
    fixed2 = 0.0
    fixed4 = 0.0
    fixed8 = 0.0
    oracle = 0.0
    by_answer_sum = defaultdict(float)
    by_answer_n = defaultdict(int)
    budget_hist = Counter()
    dataset_sum = defaultdict(float)
    dataset_n = defaultdict(int)
    escalated = 0
    for idx in indices:
        sample = samples[int(idx)]
        budget = int(selected_budgets[int(idx)])
        acc = float(sample["fixed_accuracy"][budget])
        total += acc
        total_budget += float(budget)
        fixed2 += float(sample["fixed_accuracy"][2])
        fixed4 += float(sample["fixed_accuracy"][4])
        fixed8 += float(sample["fixed_accuracy"][8])
        oracle += float(sample["fixed_accuracy"][int(sample["oracle_budget"])])
        budget_hist[budget] += 1
        if budget > 2:
            escalated += 1
        at = str(sample["answer_type"])
        by_answer_sum[at] += acc
        by_answer_n[at] += 1
        dataset_sum[str(sample["dataset_name"])] += acc
        dataset_n[str(sample["dataset_name"])] += 1
    denom = float(max(1, len(indices)))
    oracle_gap_from_k2 = float((oracle - fixed2) / denom)
    return {
        "count": int(len(indices)),
        "overall_accuracy": float(total / denom),
        "average_budget": float(total_budget / denom),
        "percent_escalated": float(escalated / denom),
        "selected_budget_histogram": {str(k): int(v) for k, v in sorted(budget_hist.items())},
        "answer_type_accuracy": {
            key: float(by_answer_sum[key] / float(max(1, by_answer_n[key])))
            for key in ("yes/no", "number", "other")
        },
        "dataset_accuracy": {
            key: float(dataset_sum[key] / float(max(1, dataset_n[key])))
            for key in sorted(dataset_sum.keys())
        },
        "delta_vs_fixed_k2": float((total - fixed2) / denom),
        "delta_vs_fixed_k4": float((total - fixed4) / denom),
        "delta_vs_fixed_k8": float((total - fixed8) / denom),
        "oracle_gap_recovered_from_k2": float(((total - fixed2) / denom) / oracle_gap_from_k2) if oracle_gap_from_k2 > 1e-12 else 0.0,
        "fixed_k2_accuracy": float(fixed2 / denom),
        "fixed_k4_accuracy": float(fixed4 / denom),
        "fixed_k8_accuracy": float(fixed8 / denom),
        "oracle_accuracy": float(oracle / denom),
    }


def _tail_breakdown(samples: Sequence[Dict[str, Any]], selected_budgets: Dict[int, int], indices: Sequence[int]) -> Dict[str, Any]:
    tail = [samples[int(idx)] for idx in indices if int(selected_budgets[int(idx)]) > 2]
    hist = Counter(int(selected_budgets[int(idx)]) for idx in indices if int(selected_budgets[int(idx)]) > 2)
    answer_hist = Counter(str(s["answer_type"]) for s in tail)
    dataset_hist = Counter(str(s["dataset_name"]) for s in tail)
    prefix_hist = Counter(_question_prefix(str(s["question"])) for s in tail)
    length_hist = Counter(_question_length_bucket(str(s["question"])) for s in tail)
    return {
        "count": int(len(tail)),
        "fraction": float(len(tail) / float(max(1, len(indices)))),
        "selected_budget_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "answer_type_fraction": {k: float(v / max(1, len(tail))) for k, v in sorted(answer_hist.items())},
        "dataset_fraction": {k: float(v / max(1, len(tail))) for k, v in sorted(dataset_hist.items())},
        "question_length_fraction": {k: float(v / max(1, len(tail))) for k, v in sorted(length_hist.items())},
        "top_question_prefixes": [
            {"prefix": key, "count": int(val)}
            for key, val in prefix_hist.most_common(12)
            if key
        ],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train and evaluate a learned cascade budget predictor from semantic-budget eval artifacts.")
    ap.add_argument("--manifest_json", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--threshold_grid", type=str, default="0.35,0.45,0.5,0.55,0.65")
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    manifest = _read_json(Path(args.manifest_json))
    sources = [_load_source(entry) for entry in manifest["sources"]]
    samples = list(itertools.chain.from_iterable(source["samples"] for source in sources))
    if not samples:
        raise RuntimeError("No predictor samples loaded.")

    split_indices = _split_indices(samples)
    node_features = {
        2: torch.stack([_compose_node_features(sample, budget=2) for sample in samples], dim=0),
        4: torch.stack([_compose_node_features(sample, budget=4) for sample in samples], dim=0),
        8: torch.stack([_compose_node_features(sample, budget=8) for sample in samples], dim=0),
    }
    oracle_budget = torch.tensor([int(sample["oracle_budget"]) for sample in samples], dtype=torch.long)

    node_specs = {
        "node2": {
            "budget": 2,
            "train_selector": lambda y: torch.ones_like(y, dtype=torch.bool),
            "label_fn": lambda y: (y > 2).to(dtype=torch.float32),
        },
        "node4": {
            "budget": 4,
            "train_selector": lambda y: (y >= 4),
            "label_fn": lambda y: (y > 4).to(dtype=torch.float32),
        },
        "node8": {
            "budget": 8,
            "train_selector": lambda y: (y >= 8),
            "label_fn": lambda y: (y > 8).to(dtype=torch.float32),
        },
    }

    trained_nodes: Dict[str, Dict[str, Any]] = {}
    for node_name, spec in node_specs.items():
        budget = int(spec["budget"])
        selector = spec["train_selector"](oracle_budget)
        train_idx = [idx for idx in split_indices["train"] if bool(selector[int(idx)].item())]
        val_idx = [idx for idx in split_indices["val"] if bool(selector[int(idx)].item())]
        if not train_idx or not val_idx:
            selected_idx = [idx for idx in split_indices["full"] if bool(selector[int(idx)].item())]
            if selected_idx:
                const_prob = float(spec["label_fn"](oracle_budget[selected_idx]).float().mean().item())
            else:
                const_prob = 0.0
            trained_nodes[node_name] = {
                "budget": budget,
                "model": None,
                "mean": None,
                "std": None,
                "probs": torch.full((len(samples),), float(const_prob), dtype=torch.float32),
                "train_meta": {
                    "best_val_acc@0.5": None,
                    "history": [],
                    "constant_prob": float(const_prob),
                    "note": "degenerate supervision; falling back to constant node output",
                },
            }
            continue
        x_train = node_features[budget][train_idx]
        y_train = spec["label_fn"](oracle_budget[train_idx])
        x_val = node_features[budget][val_idx]
        y_val = spec["label_fn"](oracle_budget[val_idx])
        if int(torch.unique(y_train).numel()) < 2 or int(torch.unique(y_val).numel()) < 2:
            combined_idx = train_idx + val_idx
            const_prob = float(spec["label_fn"](oracle_budget[combined_idx]).float().mean().item())
            trained_nodes[node_name] = {
                "budget": budget,
                "model": None,
                "mean": None,
                "std": None,
                "probs": torch.full((len(samples),), float(const_prob), dtype=torch.float32),
                "train_meta": {
                    "best_val_acc@0.5": None,
                    "history": [],
                    "constant_prob": float(const_prob),
                    "note": "single-class supervision; falling back to constant node output",
                },
            }
            continue
        model, artifacts, train_meta = _train_binary_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            device=device,
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        probs = _predict_probs(model, node_features[budget], mean=artifacts["mean"], std=artifacts["std"], device=device)
        trained_nodes[node_name] = {
            "budget": budget,
            "model": model,
            "mean": artifacts["mean"],
            "std": artifacts["std"],
            "probs": probs,
            "train_meta": train_meta,
        }

    thresholds = [float(x) for x in str(args.threshold_grid).split(",") if str(x).strip()]
    if not thresholds:
        thresholds = [0.5]

    def select_budgets_for_indices(indices: Sequence[int], t2: float, t4: float, t8: float) -> Dict[int, int]:
        out: Dict[int, int] = {}
        p2 = trained_nodes["node2"]["probs"]
        p4 = trained_nodes["node4"]["probs"]
        p8 = trained_nodes["node8"]["probs"]
        for idx in indices:
            idx = int(idx)
            if float(p2[idx].item()) < t2:
                out[idx] = 2
            elif float(p4[idx].item()) < t4:
                out[idx] = 4
            elif float(p8[idx].item()) < t8:
                out[idx] = 8
            else:
                out[idx] = 16
        return out

    best_cfg: Dict[str, Any] | None = None
    for t2, t4, t8 in itertools.product(thresholds, thresholds, thresholds):
        selected = select_budgets_for_indices(split_indices["val"], t2=t2, t4=t4, t8=t8)
        metrics = _score_selection(samples, selected, split_indices["val"])
        row = {
            "thresholds": {"node2": float(t2), "node4": float(t4), "node8": float(t8)},
            "val": metrics,
        }
        if best_cfg is None:
            best_cfg = row
            continue
        cur = row["val"]
        best = best_cfg["val"]
        if float(cur["overall_accuracy"]) > float(best["overall_accuracy"]) + 1e-12:
            best_cfg = row
        elif abs(float(cur["overall_accuracy"]) - float(best["overall_accuracy"])) <= 1e-12 and float(cur["average_budget"]) < float(best["average_budget"]):
            best_cfg = row

    assert best_cfg is not None
    chosen_thresholds = best_cfg["thresholds"]

    outputs: Dict[str, Any] = {
        "manifest_json": os.path.abspath(args.manifest_json),
        "device": str(device),
        "split_rule": "stable md5(dataset_name:question_id) mod 10 -> train 0-7, val 8, report 9",
        "node_training": {
            node_name: {
                "budget": int(node["budget"]),
                **node["train_meta"],
            }
            for node_name, node in trained_nodes.items()
        },
        "chosen_thresholds": chosen_thresholds,
        "splits": {},
    }
    for split_name in ("report", "full"):
        indices = split_indices["report"] if split_name == "report" else split_indices["full"]
        selected = select_budgets_for_indices(
            indices,
            t2=float(chosen_thresholds["node2"]),
            t4=float(chosen_thresholds["node4"]),
            t8=float(chosen_thresholds["node8"]),
        )
        outputs["splits"][split_name] = {
            "metrics": _score_selection(samples, selected, indices),
            "tail_analysis": _tail_breakdown(samples, selected, indices),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=True)
    report_metrics = outputs["splits"]["report"]["metrics"]
    print(
        f"[learned-budget] report_accuracy={float(report_metrics.get('overall_accuracy', 0.0) or 0.0):.4f} "
        f"avg_budget={float(report_metrics.get('average_budget', 0.0) or 0.0):.4f} "
        f"oracle_gap_recovered_from_k2={float(report_metrics.get('oracle_gap_recovered_from_k2', 0.0) or 0.0):.4f}",
        flush=True,
    )
    print(f"[learned-budget] wrote: {os.path.abspath(args.output_json)}", flush=True)


if __name__ == "__main__":
    main()
