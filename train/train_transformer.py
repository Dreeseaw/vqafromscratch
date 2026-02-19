"""
training code for encoder-decoder Transformer on pre-tokenized sequences

new run
> python3 -m train.train_transformer my_run_id --train_data /path/to/tokens.npy

continue from weights (in this case, from step 4000)
> python3 -m train.train_transformer my_run_id --train_data /path/to/tokens.npy --checkpoint 4000

results saved in
- /logs/<run_id>/logfile.txt
- /logs/<run_id>/step_N.tar
"""
import os
import sys
import math
import time
import json
import random
import argparse
import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from models.lm import TransformerV1, LMConfig
from train.lm_probe_debug import (
    parse_probe_layers,
    parse_probe_prompts,
    resolve_probe_file_path,
    run_debug_probes,
)


LOGDIR = "logs/"
LOGFILE = "logfile.txt"
SEED = 35
RUNNING_CE_WINDOW_TOKENS = 200_000
R_METRIC_GROUP_KEYS = ("embed", "atten", "mlp", "lm_head")
LR_ANNEAL_ALPHA = 0.02
LR_ANNEAL_RATIO_UP = 1.01
LR_ANNEAL_HOLD = 8
LR_ANNEAL_COOLDOWN = 8
LR_ANNEAL_FACTOR = 0.5
LR_ANNEAL_MIN_LR = 1e-6
LR_ANNEAL_CHECK_INTERVAL = 500

# Length buckets (inclusive ranges) for bucketed batching
BUCKET_RANGES = [
    (1, 64),
    (65, 128),
    (129, 256),
]


def make_bucket_ranges(max_seq_len: int, bucket_width: int) -> List[tuple]:
    if bucket_width <= 0:
        raise ValueError("--bucket_width must be > 0")
    ranges = []
    lo = 1
    while lo <= max_seq_len:
        hi = min(max_seq_len, lo + bucket_width - 1)
        ranges.append((lo, hi))
        lo = hi + 1
    return ranges


### Logging

class Logger:
    def __init__(self, run_id: str, checkpoint_id: Optional[int], probe: bool = False):
        self._run_id = run_id
        self._base = os.path.join(LOGDIR, run_id)
        self._ckpt = checkpoint_id
        self._probe = probe

        os.makedirs(self._base, exist_ok=True)

        # fail on duplicate run_id for now (unless cont. training)
        if os.path.isfile(os.path.join(self._base, LOGFILE)) and not checkpoint_id and not probe:
            print("this run_id already exists (no ckpt) - exiting")
            sys.exit(1)

        self._fn = os.path.join(self._base, LOGFILE)
        if self._probe:
            self._fn = os.path.join(self._base, f"logfile_probe{self._ckpt}.txt")
        elif self._ckpt:
            self._fn = os.path.join(self._base, f"logfile_from_{self._ckpt}.txt")

    def log(self, txt: str):
        print(txt)
        with open(self._fn, "a") as f:
            f.write(txt)


def log_params(model: nn.Module, logger: Logger) -> None:
    total_params, total_bytes = 0, 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.nelement() * param.element_size()
    param_size_mb = total_bytes / (1024 ** 2)
    logger.log(f"Total params: {total_params:,}")
    logger.log(f"Total size (MB): {param_size_mb:.4f}")


### Dataset loading

class TokenDataset(Dataset):
    def __init__(self, path: str, max_seq_len: int):
        self.path = path
        self.max_seq_len = max_seq_len
        self.data = self._load(path)
        self.fixed_len = self._is_fixed_length(self.data)
        if self.fixed_len:
            seq_len = int(self.data.shape[1])
            if seq_len != self.max_seq_len:
                # treat as variable-length so collate can pad/truncate to max_seq_len
                self.fixed_len = False

    def _load(self, path: str):
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"Expected integer .npy, got {arr.dtype} for {path}")
            return arr
        if path.endswith(".pt"):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                if "input_ids" in obj:
                    obj = obj["input_ids"]
                elif "sequences" in obj:
                    obj = obj["sequences"]
                else:
                    raise ValueError(f"Unrecognized keys in {path}: {list(obj.keys())}")
            if isinstance(obj, torch.Tensor):
                if obj.dtype not in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
                    raise ValueError(f"Expected integer tensor, got {obj.dtype} for {path}")
                return obj
            if isinstance(obj, list):
                return obj
            raise ValueError(f"Unsupported .pt payload type: {type(obj)}")
        raise ValueError(f"Unsupported data format: {path}")

    def _is_fixed_length(self, data) -> bool:
        if isinstance(data, torch.Tensor) and data.ndim == 2:
            return True
        if isinstance(data, np.ndarray) and data.ndim == 2:
            return True
        return False

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.data[idx]
        if isinstance(item, torch.Tensor):
            if item.dtype != torch.long:
                item = item.to(torch.long)
            return item
        if isinstance(item, np.ndarray):
            if item.dtype != np.int64:
                item = item.astype(np.int64)
            return torch.from_numpy(item)
        # list or other sequence types
        return torch.tensor(item, dtype=torch.long)

    def seq_length(self, idx: int) -> int:
        item = self.data[idx]
        if isinstance(item, (np.ndarray, torch.Tensor, list, tuple)):
            return int(len(item))
        return int(len(self.__getitem__(idx)))


class ManifestShardDataset(Dataset):
    def __init__(self, out_dir: str, max_seq_len: int):
        self.out_dir = out_dir
        self.max_seq_len = max_seq_len
        self.fixed_len = False
        self.meta = None

        meta_path = os.path.join(out_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        manifest_path = os.path.join(out_dir, "manifest.jsonl")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"manifest.jsonl not found in {out_dir}")

        self.shards = []
        self.cum_sizes = []
        total = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                tokens_path = os.path.join(out_dir, entry["tokens"])
                lengths_path = os.path.join(out_dir, entry["lengths"])
                tokens = np.load(tokens_path, mmap_mode="r")
                lengths = np.load(lengths_path, mmap_mode="r")
                num_seqs = int(entry.get("num_seqs", len(lengths)))
                if num_seqs > len(lengths):
                    raise ValueError(f"num_seqs > lengths for {lengths_path}")
                offsets = np.zeros(num_seqs + 1, dtype=np.int64)
                offsets[1:] = np.cumsum(lengths[:num_seqs], dtype=np.int64)
                if tokens.shape[0] < offsets[-1]:
                    raise ValueError(f"Token buffer shorter than lengths in {tokens_path}")
                self.shards.append(
                    {
                        "tokens": tokens,
                        "lengths": lengths,
                        "offsets": offsets,
                        "num_seqs": num_seqs,
                    }
                )
                total += num_seqs
                self.cum_sizes.append(total)

    def __len__(self) -> int:
        return int(self.cum_sizes[-1]) if self.cum_sizes else 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx = int(np.searchsorted(self.cum_sizes, idx, side="right"))
        if shard_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - int(self.cum_sizes[shard_idx - 1])
        shard = self.shards[shard_idx]
        start = int(shard["offsets"][sample_idx])
        end = int(shard["offsets"][sample_idx + 1])
        seq = shard["tokens"][start:end]
        if isinstance(seq, np.ndarray) and seq.dtype != np.int64:
            seq = seq.astype(np.int64)
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        return torch.from_numpy(np.asarray(seq, dtype=np.int64))

    def seq_length(self, idx: int) -> int:
        shard_idx = int(np.searchsorted(self.cum_sizes, idx, side="right"))
        if shard_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - int(self.cum_sizes[shard_idx - 1])
        shard = self.shards[shard_idx]
        return int(shard["lengths"][sample_idx])


class ShardedTokenDataset(Dataset):
    def __init__(self, datasets: List[TokenDataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cum_sizes = np.cumsum(self.lengths)
        self.fixed_len = all(getattr(d, "fixed_len", False) for d in datasets)

    def __len__(self) -> int:
        return int(self.cum_sizes[-1]) if len(self.cum_sizes) else 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        ds_idx = int(np.searchsorted(self.cum_sizes, idx, side="right"))
        if ds_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - int(self.cum_sizes[ds_idx - 1])
        return self.datasets[ds_idx][sample_idx]

    def seq_length(self, idx: int) -> int:
        ds_idx = int(np.searchsorted(self.cum_sizes, idx, side="right"))
        if ds_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - int(self.cum_sizes[ds_idx - 1])
        ds = self.datasets[ds_idx]
        if hasattr(ds, "seq_length"):
            return int(ds.seq_length(sample_idx))
        return int(len(ds[sample_idx]))


def load_dataset(path: str, max_seq_len: int) -> Dataset:
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.jsonl")
        if os.path.isfile(manifest_path):
            return ManifestShardDataset(path, max_seq_len)
        shard_paths = [
            os.path.join(path, p)
            for p in sorted(os.listdir(path))
            if p.endswith(".npy") or p.endswith(".pt")
        ]
        if not shard_paths:
            raise FileNotFoundError(f"No .npy/.pt shards found in {path}")
        datasets = [TokenDataset(p, max_seq_len) for p in shard_paths]
        return ShardedTokenDataset(datasets)
    if os.path.isfile(path):
        return TokenDataset(path, max_seq_len)
    raise FileNotFoundError(f"Data path not found: {path}")


class CollatePad:
    def __init__(
        self,
        max_seq_len: int,
        pad_id: int,
        fixed_len: bool,
    ):
        self.max_seq_len = max_seq_len
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2 for next-token training.")
        self.pad_id = pad_id
        self.fixed_len = fixed_len

    def __call__(self, batch: List[torch.Tensor]):
        tokens = torch.full((len(batch), self.max_seq_len), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(batch):
            if seq.numel() <= 0:
                continue
            length = min(seq.numel(), self.max_seq_len)
            tokens[i, :length] = seq[:length]

        input_ids = tokens
        target_ids = torch.full_like(input_ids, self.pad_id)
        target_ids[:, :-1] = input_ids[:, 1:]

        loss_mask = target_ids.ne(self.pad_id) & input_ids.ne(self.pad_id)
        # Keep token-0 out of loss when using next-token shift.
        loss_mask[:, 0] = False
        return input_ids, target_ids, loss_mask


def build_length_buckets(
    dataset: Dataset,
    max_seq_len: int,
    bucket_ranges: List[tuple],
):
    ranges: List[tuple] = []
    for lo, hi in bucket_ranges:
        if lo > max_seq_len:
            continue
        if hi >= max_seq_len:
            hi = max_seq_len
        if lo <= hi:
            ranges.append((lo, hi))
    if not ranges:
        ranges = [(1, max_seq_len)]

    buckets: List[List[int]] = [[] for _ in ranges]
    token_counts = [0 for _ in ranges]

    for idx in range(len(dataset)):
        if hasattr(dataset, "seq_length"):
            length = int(dataset.seq_length(idx))
        else:
            length = int(len(dataset[idx]))
        if length <= 0:
            length = 1
        if length > max_seq_len:
            length = max_seq_len

        placed = False
        for b_idx, (lo, hi) in enumerate(ranges):
            if lo <= length <= hi:
                buckets[b_idx].append(idx)
                token_counts[b_idx] += length
                placed = True
                break
        if not placed:
            buckets[-1].append(idx)
            token_counts[-1] += length

    return ranges, buckets, token_counts


class BucketBatchSampler:
    def __init__(
        self,
        bucket_indices: List[List[int]],
        bucket_token_counts: List[int],
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        self.bucket_indices = [list(b) for b in bucket_indices]
        self.bucket_token_counts = np.asarray(bucket_token_counts, dtype=np.float64)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.batches_per_bucket = []
        for bucket in self.bucket_indices:
            if drop_last:
                n_batches = len(bucket) // batch_size
            else:
                n_batches = math.ceil(len(bucket) / batch_size) if len(bucket) > 0 else 0
            self.batches_per_bucket.append(int(n_batches))
        self.batches_per_epoch = int(sum(self.batches_per_bucket))

        if self.bucket_token_counts.sum() <= 0:
            raise ValueError("No tokens found for bucket sampling.")
        if self.batches_per_epoch <= 0:
            raise ValueError("No full batches available; reduce --batch_size or adjust data bucketing.")
        self.bucket_probs = self.bucket_token_counts / self.bucket_token_counts.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        all_batches = []
        for bucket, n_batches in zip(self.bucket_indices, self.batches_per_bucket):
            if n_batches <= 0:
                continue
            shuffled = list(bucket)
            rng.shuffle(shuffled)
            if self.drop_last:
                usable = n_batches * self.batch_size
                shuffled = shuffled[:usable]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        rng.shuffle(all_batches)
        for batch in all_batches:
            yield batch


### Scheduling

class WarmupScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, warmup_steps: int, total_steps: int, schedule: str = "cosine"):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.schedule = schedule
        self.step_num = 0

    def get_lr(self) -> float:
        if self.step_num < self.warmup_steps:
            return self.base_lr * float(self.step_num + 1) / float(self.warmup_steps)
        if self.schedule == "stair":
            if self.step_num >= 14000:
                return self.base_lr / 2.0
        if self.schedule == "flat":
            return self.base_lr
        # cosine decay
        progress = (self.step_num - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        self.step_num += 1
        return lr


### Training


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some MPS ops are not deterministic; keep going for stability.
            pass


def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state() -> dict:
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng_state(state: Optional[dict]) -> None:
    if not isinstance(state, dict):
        return
    py_state = state.get("python")
    np_state = state.get("numpy")
    torch_cpu = state.get("torch_cpu")
    torch_cuda = state.get("torch_cuda")
    if py_state is not None:
        random.setstate(py_state)
    if np_state is not None:
        np.random.set_state(np_state)
    if torch_cpu is not None:
        torch.set_rng_state(torch_cpu)
    if torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda)


def _resolve_data_paths(train_data: str, val_data: Optional[str], test_data: Optional[str]):
    train_path = train_data
    val_path = val_data
    test_path = test_data
    if os.path.isdir(train_data):
        candidate_train = os.path.join(train_data, "train")
        candidate_val = os.path.join(train_data, "val")
        candidate_test = os.path.join(train_data, "test")
        if os.path.isdir(candidate_train):
            train_path = candidate_train
            if val_path is None and os.path.isdir(candidate_val):
                val_path = candidate_val
            if test_path is None and os.path.isdir(candidate_test):
                test_path = candidate_test
    return train_path, val_path, test_path


def _example_loss_tokens(dataset: Dataset, idx: int, max_seq_len: int) -> int:
    if hasattr(dataset, "seq_length"):
        length = int(dataset.seq_length(idx))
    else:
        length = int(len(dataset[idx]))
    length = min(max(0, length), max_seq_len)
    return max(0, length - 1)


def build_fixed_eval_subset(
    dataset: Dataset,
    max_seq_len: int,
    max_tokens: int,
    max_steps: int,
    batch_size: int,
    seed: int,
):
    dataset_size = len(dataset)
    if dataset_size <= 0:
        return dataset, {"start_idx": 0, "examples": 0, "tokens": 0, "full_dataset": True}

    if max_tokens <= 0 and max_steps <= 0:
        return dataset, {
            "start_idx": 0,
            "examples": dataset_size,
            "tokens": -1,
            "full_dataset": True,
        }

    rng = np.random.default_rng(seed)
    start_idx = int(rng.integers(0, dataset_size))

    max_examples = dataset_size
    if max_steps > 0:
        max_examples = min(max_examples, max_steps * batch_size)

    indices = []
    token_total = 0
    for offset in range(dataset_size):
        if len(indices) >= max_examples:
            break
        idx = (start_idx + offset) % dataset_size
        indices.append(idx)
        token_total += _example_loss_tokens(dataset, idx, max_seq_len)
        if max_tokens > 0 and token_total >= max_tokens:
            break

    return Subset(dataset, indices), {
        "start_idx": start_idx,
        "examples": len(indices),
        "tokens": token_total,
        "full_dataset": False,
    }


def _load_tokenizer_info(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    from models.bpe_tokenizer import ByteBPETokenizer

    tok = ByteBPETokenizer.load(tokenizer_path)
    return {
        "vocab_size": int(tok.vocab_size),
        "pad_id": int(tok.pad_id),
        "bos_id": int(tok.bos_id),
        "eos_id": int(tok.eos_id),
    }


def _safe_exp(x: float) -> float:
    try:
        return float(math.exp(x))
    except OverflowError:
        return float("inf")


def global_grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return float(math.sqrt(max(total, 0.0)))


def grad_weight_stats(parameters, eps: float = 1e-12, optimizer: Optional[torch.optim.Optimizer] = None):
    grad_sq = 0.0
    weight_sq = 0.0
    m1_sq = 0.0
    have_m1 = False
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        weight = param.detach()
        grad_sq += float(torch.sum(grad * grad).item())
        weight_sq += float(torch.sum(weight * weight).item())
        if optimizer is not None:
            state = optimizer.state.get(param, None)
            if isinstance(state, dict):
                exp_avg = state.get("exp_avg", None)
                if torch.is_tensor(exp_avg):
                    m1 = exp_avg.detach()
                    m1_sq += float(torch.sum(m1 * m1).item())
                    have_m1 = True
    grad_norm = float(math.sqrt(max(grad_sq, 0.0)))
    weight_norm = float(math.sqrt(max(weight_sq, 0.0)))
    r_grad = grad_norm / float(max(weight_norm, eps))
    r_m1 = float("nan")
    if have_m1:
        m1_norm = float(math.sqrt(max(m1_sq, 0.0)))
        r_m1 = m1_norm / float(max(weight_norm, eps))
    return grad_norm, weight_norm, r_grad, r_m1


def compute_r_value(parameters: Sequence[nn.Parameter], optimizer: Optional[torch.optim.Optimizer] = None) -> float:
    has_grad = False
    for param in parameters:
        if param.grad is not None:
            has_grad = True
            break
    if not has_grad:
        return float("nan")
    _grad_norm, _weight_norm, r_grad, r_m1 = grad_weight_stats(parameters, optimizer=optimizer)
    return float(r_m1 if math.isfinite(r_m1) else r_grad)


def _is_attention_param_name(name: str) -> bool:
    attn_markers = (
        "._in_proj.",
        "._out_proj.",
        "._self_in_proj.",
        "._self_out_proj.",
        "._cross_q.",
        "._cross_kv.",
        "._cross_out_proj.",
        "._ls_attn",
        "._ls_self_attn",
        "._ls_cross_attn",
    )
    return any(marker in name for marker in attn_markers)


def _is_mlp_param_name(name: str) -> bool:
    return "._mlp." in name or "._ls_mlp" in name


def _metric_group_for_param(name: str) -> Optional[str]:
    if name.startswith("_embed."):
        return "embed"
    if name.startswith("_unembed."):
        return "lm_head"
    if _is_mlp_param_name(name):
        return "mlp"
    if _is_attention_param_name(name):
        return "atten"
    return None


def build_r_metric_param_groups(model: nn.Module, tie_embeddings: bool) -> Dict[str, List[nn.Parameter]]:
    out: Dict[str, List[nn.Parameter]] = {k: [] for k in R_METRIC_GROUP_KEYS}
    seen_ids: Dict[str, set] = {k: set() for k in R_METRIC_GROUP_KEYS}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_name = _metric_group_for_param(str(name))
        if group_name is None:
            continue
        pid = id(param)
        if pid in seen_ids[group_name]:
            continue
        out[group_name].append(param)
        seen_ids[group_name].add(pid)

    # Keep lm_head metrics available under tied embeddings (shared with _embed.weight).
    if tie_embeddings:
        shared_param = model._unembed.weight
        shared_id = id(shared_param)
        if shared_id not in seen_ids["lm_head"]:
            out["lm_head"].append(shared_param)
            seen_ids["lm_head"].add(shared_id)
    return out


def build_optimizer_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    tie_embeddings: bool,
    half_lr_lm_head: bool,
):
    tied_embed_param_id = id(model._embed.weight) if tie_embeddings else None
    grouped_params: Dict[str, List[nn.Parameter]] = {
        "main_decay": [],
        "lm_head_no_decay": [],
        "embed_decay": [],
        "main_no_decay": [],
        "embed_no_decay": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_name = str(name)
        is_embed = param_name.startswith("_embed.")
        is_lm_head = param_name.startswith("_unembed.")
        is_bias = param_name.endswith(".bias")
        if is_lm_head:
            grouped_params["lm_head_no_decay"].append(param)
            continue
        no_decay = is_bias or (tied_embed_param_id is not None and id(param) == tied_embed_param_id)
        if is_embed:
            key = "embed_no_decay" if no_decay else "embed_decay"
        else:
            key = "main_no_decay" if no_decay else "main_decay"
        grouped_params[key].append(param)

    out = []
    for key in ("main_decay", "lm_head_no_decay", "embed_decay", "main_no_decay", "embed_no_decay"):
        params = grouped_params[key]
        if len(params) == 0:
            continue
        if key.startswith("embed_"):
            lr_scale = 0.5
        elif key == "lm_head_no_decay":
            lr_scale = 0.125 if half_lr_lm_head else 1.0
        else:
            lr_scale = 1.0
        group_weight_decay = 0.0 if key.endswith("_no_decay") or key == "embed_decay" else float(weight_decay)
        out.append(
            {
                "params": params,
                "weight_decay": group_weight_decay,
                "lr_scale": lr_scale,
                "lr": float(base_lr) * lr_scale,
            }
        )
    return out


def load_optimizer_state_compat(optimizer: torch.optim.Optimizer, state_dict: dict) -> str:
    try:
        optimizer.load_state_dict(state_dict)
        return "loaded"
    except ValueError:
        pass

    if not isinstance(state_dict, dict):
        return "failed"
    saved_groups = state_dict.get("param_groups")
    saved_state = state_dict.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        return "failed"

    saved_flat_ids = []
    for group in saved_groups:
        params = group.get("params", [])
        if not isinstance(params, list):
            return "failed"
        saved_flat_ids.extend(params)

    current_state = optimizer.state_dict()
    current_flat_ids = []
    for group in current_state.get("param_groups", []):
        params = group.get("params", [])
        if not isinstance(params, list):
            return "failed"
        current_flat_ids.extend(params)

    if len(saved_flat_ids) != len(current_flat_ids):
        return "failed"

    migrated_state = {}
    for new_id, old_id in zip(current_flat_ids, saved_flat_ids):
        old_entry = saved_state.get(old_id)
        if old_entry is not None:
            migrated_state[new_id] = old_entry

    current_state["state"] = migrated_state
    optimizer.load_state_dict(current_state)
    return "migrated"


def topk_grad_norms(named_parameters, k: int = 5):
    rows = []
    for name, param in named_parameters:
        if param.grad is None:
            continue
        grad_norm = float(torch.linalg.vector_norm(param.grad.detach()).item())
        if not math.isfinite(grad_norm):
            continue
        rows.append((str(name), grad_norm))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[: max(0, int(k))]


def format_top_grad_norms(entries) -> str:
    if not entries:
        return "TopGradNorms=[]"
    pairs = []
    for name, grad_norm in entries:
        safe_name = str(name).replace("@", "_").replace(";", "_").replace("[", "_").replace("]", "_")
        pairs.append(f"{safe_name}@{grad_norm:.6e}")
    return "TopGradNorms=[" + ";".join(pairs) + "]"


def adaptive_clip_grad_(parameters, clip_factor=0.01, eps=1e-3, grad_eps=1e-6):
    """
    Parameter-norm-relative adaptive clipping.
    For each parameter tensor p with gradient g:
        ||g||_2 <= clip_factor * max(||p||_2, eps)
    """
    if clip_factor <= 0:
        return (0, 0)

    clipped = 0
    total = 0
    for param in parameters:
        if param.grad is None:
            continue

        total += 1
        p_norm = float(torch.linalg.vector_norm(param.detach()).item())
        g_norm = float(torch.linalg.vector_norm(param.grad.detach()).item())
        max_norm = clip_factor * max(p_norm, eps)
        clip_coef = min(1.0, max_norm / (g_norm + grad_eps))
        if clip_coef < 1.0:
            param.grad.detach().mul_(clip_coef)
            clipped += 1

    return (clipped, total)


def extract_attn_entropy_metrics(attn_state: Optional[dict]) -> dict:
    out = {}
    if not isinstance(attn_state, dict):
        return out
    for scope_name, scope_key in (
        ("encoder_layers", "enc"),
        ("decoder_self_layers", "dec_self"),
        ("decoder_cross_layers", "dec_cross"),
    ):
        entries = attn_state.get(scope_name, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            layer = entry.get("layer")
            if not isinstance(layer, int):
                continue
            value = entry.get("attn_entropy")
            if not isinstance(value, (float, int)):
                continue
            out[f"attn_entropy_{scope_key}_l{layer}"] = float(value)
    return out


def _collate_shift_sanity(collate_fn: CollatePad, pad_id: int) -> None:
    toy = [
        torch.tensor([11, 12, 13, 14], dtype=torch.long),
        torch.tensor([21, 22], dtype=torch.long),
    ]
    input_ids, target_ids, loss_mask = collate_fn(toy)
    assert int(target_ids[0, 0].item()) == 12
    assert int(target_ids[0, 1].item()) == 13
    assert int(target_ids[0, 2].item()) == 14
    assert not bool(loss_mask[0, 0].item())
    assert not bool(loss_mask[input_ids == pad_id].any().item())


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    pad_id: int,
    vocab_size: int,
    max_tokens: int = 0,
    max_steps: int = 0,
    pin_memory: bool = False,
):
    model_was_training = model.training
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    total_entropy_sum = 0.0
    total_steps = 0

    for batch in loader:
        input_ids, target_ids, loss_mask = batch
        input_ids = input_ids.to(device, non_blocking=pin_memory)
        target_ids = target_ids.to(device, non_blocking=pin_memory)
        loss_mask = loss_mask.to(device, non_blocking=pin_memory)

        logits = model(input_ids, pad_mask=input_ids.eq(pad_id))

        flat_mask = loss_mask.reshape(-1)
        token_count = int(flat_mask.sum().item())
        if token_count <= 0:
            continue
        flat_logits = logits.reshape(-1, vocab_size)[flat_mask]
        flat_targets = target_ids.reshape(-1)[flat_mask]
        loss_sum = F.cross_entropy(flat_logits, flat_targets, reduction="sum")

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_sum = float(entropy[loss_mask].sum().item())

        total_loss_sum += float(loss_sum.item())
        total_tokens += token_count
        total_entropy_sum += entropy_sum
        total_steps += 1

        if max_steps > 0 and total_steps >= max_steps:
            break
        if max_tokens > 0 and total_tokens >= max_tokens:
            break

    ce = total_loss_sum / float(max(1, total_tokens))
    ppl = _safe_exp(ce)
    mean_entropy = total_entropy_sum / float(max(1, total_tokens))

    if model_was_training:
        model.train()
    return {
        "ce": ce,
        "ppl": ppl,
        "entropy": mean_entropy,
        "tokens": total_tokens,
        "steps": total_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)

    # tokens / data
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--pad_id", type=int, default=None)
    parser.add_argument("--bos_id", type=int, default=None)
    parser.add_argument("--eos_id", type=int, default=None)

    # model (models/lm.py)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--enc_layers", type=int, default=6)  # must match dec_layers
    parser.add_argument("--dec_layers", type=int, default=6)  # must match enc_layers
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tie_embeddings", action="store_true")
    parser.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument(
        "--sdp_backend",
        type=str,
        default="auto",
        choices=["auto", "flash", "mem_efficient", "math"],
    )
    parser.add_argument(
        "--cosine_attn",
        action="store_true",
        help="Normalize Q/K before attention dot-product (cosine attention).",
    )
    parser.add_argument(
        "--v_rmsnorm",
        action="store_true",
        help="RMS-normalize projected V per head before attention.",
    )
    parser.add_argument(
        "--layerscale",
        action="store_true",
        help="Enable LayerScale on residual branches.",
    )
    parser.add_argument(
        "--layerscale_init",
        type=float,
        default=1e-5,
        help="Initial LayerScale value (used when --layerscale is set).",
    )

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before optimizer step.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--half_lr_lm_head",
        action="store_true",
        help="Use 0.5x base LR for _unembed (LM head) parameters.",
    )
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "flat", "stair"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--grad_clip_mode",
        type=str,
        default="global",
        choices=["global", "agc", "none"],
        help="Gradient clipping mode: global-norm, adaptive (parameter-relative), or disabled.",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        help="Global gradient clip norm (used when --grad_clip_mode=global).",
    )
    parser.add_argument(
        "--agc_clip_factor",
        type=float,
        default=0.01,
        help="AGC clip factor for ||g|| <= factor * max(||p||, eps).",
    )
    parser.add_argument(
        "--agc_eps",
        type=float,
        default=1e-3,
        help="AGC minimum parameter norm epsilon.",
    )
    parser.add_argument("--eval_every_steps", type=int, default=1000)
    parser.add_argument("--val_max_tokens", type=int, default=200000)
    parser.add_argument("--val_steps", type=int, default=0)
    parser.add_argument("--test_max_tokens", type=int, default=0)
    parser.add_argument("--test_steps", type=int, default=0)
    parser.add_argument("--run_probes", type=int, default=0)
    parser.add_argument("--probe_file", type=str, default=None)
    parser.add_argument(
        "--probe_layers",
        type=str,
        default="",
        help="Comma-separated layer indices to probe (e.g. 0,1,5). Default uses probe-debug defaults.",
    )
    parser.add_argument("--probe_topk_eigs", type=int, default=5)
    parser.add_argument("--probe_gen_tokens", type=int, default=48)
    parser.add_argument(
        "--probe_attn_entropy",
        dest="probe_attn_entropy",
        action="store_true",
        help="Capture attention entropy metrics during probe runs.",
    )
    parser.add_argument(
        "--no_probe_attn_entropy",
        dest="probe_attn_entropy",
        action="store_false",
        help="Disable attention entropy capture during probe runs.",
    )
    parser.set_defaults(probe_attn_entropy=True)
    parser.add_argument(
        "--track_attn_entropy",
        action="store_true",
        help="Track mean attention entropy for encoder/self-cross decoder attention on training batches.",
    )

    # performance
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--bucket_width", type=int, default=64)
    parser.add_argument("--activation_checkpointing", action="store_true")

    args = parser.parse_args()

    # device selection
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    if device == "cpu":
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    set_seed(SEED, deterministic=args.deterministic)

    train_path, val_path, test_path = _resolve_data_paths(
        args.train_data, args.val_data, args.test_data
    )
    train_dataset = load_dataset(train_path, args.max_seq_len)
    train_meta = getattr(train_dataset, "meta", None) or {}
    tok_info = _load_tokenizer_info(args.tokenizer)

    if args.vocab_size is None:
        if tok_info is not None:
            args.vocab_size = int(tok_info["vocab_size"])
        elif train_meta.get("vocab_size") is not None:
            args.vocab_size = int(train_meta["vocab_size"])
        else:
            raise SystemExit("Unable to infer --vocab_size. Provide --tokenizer or --vocab_size.")

    if args.pad_id is None:
        if tok_info is not None:
            args.pad_id = int(tok_info["pad_id"])
        elif train_meta.get("pad_id") is not None:
            args.pad_id = int(train_meta["pad_id"])
        else:
            raise SystemExit("Unable to infer --pad_id. Provide --tokenizer or --pad_id.")
    if args.bos_id is None:
        if tok_info is not None:
            args.bos_id = int(tok_info["bos_id"])
        elif train_meta.get("bos_id") is not None:
            args.bos_id = int(train_meta["bos_id"])
        else:
            args.bos_id = 2
    if args.eos_id is None:
        if tok_info is not None:
            args.eos_id = int(tok_info["eos_id"])
        elif train_meta.get("eos_id") is not None:
            args.eos_id = int(train_meta["eos_id"])
        else:
            raise SystemExit("Unable to infer --eos_id. Provide --tokenizer or --eos_id.")

    if tok_info is not None:
        tok_vocab = int(tok_info["vocab_size"])
        if args.vocab_size != tok_vocab:
            print(f"warning: overriding vocab_size {args.vocab_size} -> tokenizer vocab_size {tok_vocab}")
            args.vocab_size = tok_vocab
        if int(args.pad_id) != int(tok_info["pad_id"]):
            raise SystemExit("pad_id must match tokenizer pad_id.")
        if int(args.eos_id) != int(tok_info["eos_id"]):
            raise SystemExit("eos_id must match tokenizer eos_id.")

    if int(args.pad_id) == int(args.eos_id):
        raise SystemExit("PAD and EOS must be different ids.")
    if args.vocab_size <= int(max(args.pad_id, args.eos_id, args.bos_id)):
        raise SystemExit("vocab_size must be larger than special token ids.")

    probe_tokenizer = None
    probe_prompts = []
    probe_file_path = None
    probe_layers = None
    if int(args.run_probes) > 0:
        if not args.tokenizer:
            raise SystemExit("--run_probes requires --tokenizer so probes can be encoded.")
        from models.bpe_tokenizer import ByteBPETokenizer

        probe_tokenizer = ByteBPETokenizer.load(args.tokenizer)
        if int(probe_tokenizer.pad_id) != int(args.pad_id):
            raise SystemExit("Probe tokenizer pad_id must match training pad_id.")
        probe_file_path = resolve_probe_file_path(
            train_data_arg=args.train_data,
            resolved_train_path=train_path,
            override_probe_file=args.probe_file,
        )
        probe_prompts = parse_probe_prompts(probe_file_path)
        probe_layers = parse_probe_layers(args.probe_layers, total_layers=int(args.enc_layers))

    fixed_len = getattr(train_dataset, "fixed_len", False)
    collate_fn = CollatePad(args.max_seq_len, args.pad_id, fixed_len)
    _collate_shift_sanity(collate_fn, args.pad_id)

    if args.bucket_width > 0:
        active_bucket_ranges = make_bucket_ranges(args.max_seq_len, args.bucket_width)
    else:
        active_bucket_ranges = BUCKET_RANGES

    bucket_ranges, bucket_indices, bucket_token_counts = build_length_buckets(
        train_dataset, args.max_seq_len, active_bucket_ranges
    )
    sampler = BucketBatchSampler(
        bucket_indices=bucket_indices,
        bucket_token_counts=bucket_token_counts,
        batch_size=args.batch_size,
        seed=SEED,
        drop_last=True,
    )

    pin_memory = device == "cuda"
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        if args.persistent_workers:
            loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))

    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **loader_kwargs)

    if val_path is None:
        raise SystemExit("Validation data is required (provide --val_data or use dataset_root/val).")
    if test_path is None:
        raise SystemExit("Test data is required (provide --test_data or use dataset_root/test).")
    val_dataset = load_dataset(val_path, args.max_seq_len)
    test_dataset = load_dataset(test_path, args.max_seq_len)
    val_eval_dataset, val_slice_info = build_fixed_eval_subset(
        dataset=val_dataset,
        max_seq_len=args.max_seq_len,
        max_tokens=max(0, int(args.val_max_tokens)),
        max_steps=max(0, int(args.val_steps)),
        batch_size=args.batch_size,
        seed=SEED,
    )
    eval_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        if args.persistent_workers:
            eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    val_loader = DataLoader(val_eval_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    if args.enc_layers != args.dec_layers:
        raise SystemExit("--enc_layers must equal --dec_layers for TransformerV1.")
    if int(args.grad_accum_steps) <= 0:
        raise SystemExit("--grad_accum_steps must be > 0.")
    if args.grad_clip_mode == "global" and args.clip_grad <= 0:
        raise SystemExit("--clip_grad must be > 0 when --grad_clip_mode=global.")
    if args.grad_clip_mode == "agc" and args.agc_clip_factor <= 0:
        raise SystemExit("--agc_clip_factor must be > 0 when --grad_clip_mode=agc.")
    if args.grad_clip_mode == "agc" and args.agc_eps <= 0:
        raise SystemExit("--agc_eps must be > 0 when --grad_clip_mode=agc.")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise SystemExit("--dropout must be in [0.0, 1.0).")

    cfg = LMConfig(
        vocab_size=args.vocab_size,
        embed_size=args.d_model,
        num_heads=args.n_heads,
        mlp_ratio=args.ff_mult,
        dropout=args.dropout,
        layers=args.enc_layers,
        max_seq_len=args.max_seq_len,
        tie_embeds=args.tie_embeddings,
        activation_checkpointing=args.activation_checkpointing,
        attn_impl=args.attn_impl,
        sdp_backend=args.sdp_backend,
        cosine_attn=args.cosine_attn,
        v_rmsnorm=args.v_rmsnorm,
        layerscale=args.layerscale,
        layerscale_init=args.layerscale_init,
    )
    model = TransformerV1(cfg).to(device)
    r_metric_param_groups = build_r_metric_param_groups(model, tie_embeddings=bool(args.tie_embeddings))

    param_groups = build_optimizer_param_groups(
        model=model,
        base_lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        tie_embeddings=bool(args.tie_embeddings),
        half_lr_lm_head=bool(args.half_lr_lm_head),
    )

    if args.optimizer == "adam":
        opt = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=0.0,
        )

    updates_per_epoch = max(1, math.ceil(len(train_loader) / float(max(1, int(args.grad_accum_steps)))))
    total_steps = args.max_steps if args.max_steps is not None else args.epochs * updates_per_epoch
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = WarmupScheduler(opt, args.lr, warmup_steps, total_steps, schedule=args.schedule)

    global_step = 0
    resume_epoch = 0
    resume_batch_in_epoch = 0
    restored_log_accum_state = None
    restored_lr_anneal_state = None
    optimizer_resume_status = "none"
    if args.checkpoint is not None:
        ckpt_file = os.path.join(LOGDIR, args.run_id, f"step_{args.checkpoint}.tar")
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], weights_only=False)
        optimizer_resume_status = load_optimizer_state_compat(opt, checkpoint["optimizer_state_dict"])
        if optimizer_resume_status == "failed":
            raise RuntimeError("Unable to load optimizer checkpoint with current parameter-group layout.")
        global_step = int(checkpoint.get("global_step", 0))
        scheduler.step_num = int(checkpoint.get("scheduler_step_num", global_step))
        resume_epoch = int(checkpoint.get("epoch", 0))
        resume_batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
        restored_log_accum_state = checkpoint.get("log_accum_state")
        restored_lr_anneal_state = checkpoint.get("lr_anneal_state")
        restore_rng_state(checkpoint.get("rng_state"))

    logger = Logger(args.run_id, args.checkpoint)
    log_params(model, logger)
    logger.log(f"batch size: {args.batch_size}")
    logger.log(
        f"grad accumulation: {int(args.grad_accum_steps)} "
        f"(effective batch size: {int(args.batch_size) * int(args.grad_accum_steps)})"
    )
    logger.log(f"Run start time: {str(datetime.datetime.now())}")
    logger.log(f"Running on {device}\n")
    logger.log(f"Config: {vars(cfg)}\n")
    if args.deterministic:
        logger.log("deterministic mode: enabled (may reduce throughput)")
    if train_meta:
        for key, arg_val in (
            ("pad_id", args.pad_id),
            ("bos_id", args.bos_id),
            ("eos_id", args.eos_id),
        ):
            meta_val = train_meta.get(key)
            if meta_val is not None and int(meta_val) != int(arg_val):
                logger.log(f"warning: meta.{key}={meta_val} != args.{key}={arg_val}")
    logger.log(f"train_data: {train_path}")
    logger.log(f"val_data: {val_path}")
    logger.log(f"test_data: {test_path}")
    logger.log(
        f"eval cadence: every {args.eval_every_steps} steps "
        f"(val_max_tokens={args.val_max_tokens}, val_steps={args.val_steps})"
    )
    if val_slice_info["full_dataset"]:
        logger.log(f"fixed-val-slice: full val set (examples={val_slice_info['examples']}, loss_tokens=all)")
    else:
        logger.log(
            f"fixed-val-slice: start_idx={val_slice_info['start_idx']} "
            f"examples={val_slice_info['examples']} "
            f"loss_tokens={val_slice_info['tokens']}"
        )

    total_bucket_tokens = float(np.sum(np.asarray(bucket_token_counts, dtype=np.float64)))
    if total_bucket_tokens > 0:
        logger.log("Bucket stats (range, seqs, tokens, prob):")
        for (lo, hi), indices, tok_count, prob in zip(
            bucket_ranges,
            bucket_indices,
            bucket_token_counts,
            sampler.bucket_probs.tolist(),
        ):
            logger.log(
                f"  {lo:4d}-{hi:4d}: seqs={len(indices):7d} "
                f"tokens={tok_count:10d} prob={prob:.4f}"
            )
    logger.log(f"bucket width: {args.bucket_width} ({'auto' if args.bucket_width > 0 else 'legacy ranges'})")
    logger.log(f"activation checkpointing: {'on' if args.activation_checkpointing else 'off'}")
    if args.grad_clip_mode == "global":
        logger.log(f"grad clipping: global (max_norm={args.clip_grad})")
    elif args.grad_clip_mode == "agc":
        logger.log(f"grad clipping: adaptive (clip_factor={args.agc_clip_factor}, eps={args.agc_eps})")
    else:
        logger.log("grad clipping: none")
    logger.log("weight decay: all bias parameters excluded from decay.")
    logger.log("weight decay: all _embed and _unembed parameters excluded from decay.")
    logger.log("optimizer lr scaling: _embed parameters use 0.5x base LR.")
    if args.half_lr_lm_head:
        logger.log("optimizer lr scaling: _unembed parameters use 0.5x base LR.")
    logger.log(
        "lr anneal (EWMA plateau): "
        f"alpha={LR_ANNEAL_ALPHA}, ratio_up={LR_ANNEAL_RATIO_UP}, hold={LR_ANNEAL_HOLD}, "
        f"cooldown={LR_ANNEAL_COOLDOWN}, factor={LR_ANNEAL_FACTOR}, min_lr={LR_ANNEAL_MIN_LR}, "
        f"check_interval={LR_ANNEAL_CHECK_INTERVAL}"
    )
    if args.optimizer in ("adam", "adamw"):
        logger.log("R metric: Adam first-moment norm / weight norm (fallback to raw grad ratio before state init).")
    else:
        logger.log("R metric: raw grad norm / weight norm.")
    r_group_counts = {k: len(v) for k, v in r_metric_param_groups.items()}
    logger.log(
        "R metric groups (trainable tensors): "
        f"embed={r_group_counts['embed']}, "
        f"atten={r_group_counts['atten']}, "
        f"mlp={r_group_counts['mlp']}, "
        f"lm_head={r_group_counts['lm_head']}"
    )
    logger.log(
        f"sanity: special tokens pad={args.pad_id}, eos={args.eos_id}, "
        f"bos={args.bos_id}, vocab={args.vocab_size}"
    )
    if int(args.run_probes) > 0:
        probe_layers_label = "default(first,last)" if probe_layers is None else ",".join(str(x) for x in probe_layers)
        logger.log(
            f"probe debug cadence: every {args.run_probes} steps | prompts={len(probe_prompts)} | "
            f"probe_file={probe_file_path} | probe_layers={probe_layers_label}"
        )
        if len(probe_prompts) != 5:
            logger.log(f"warning: expected 5 probes, found {len(probe_prompts)} in {probe_file_path}")
    logger.log(f"attention entropy (train batches): {'on' if args.track_attn_entropy else 'off'}")
    logger.log(f"attention entropy (probe runs): {'on' if args.probe_attn_entropy else 'off'}")
    if args.track_attn_entropy and args.activation_checkpointing:
        logger.log("note: attention entropy capture bypasses activation checkpointing on training forwards.")
    if args.checkpoint is not None:
        logger.log(
            f"resume checkpoint: step={global_step}, epoch={resume_epoch}, "
            f"batch_in_epoch={resume_batch_in_epoch}"
        )
        if optimizer_resume_status == "migrated":
            logger.log("resume checkpoint: optimizer state migrated across parameter-group layout.")

    model.train()
    opt.zero_grad(set_to_none=True)
    tokens_since_log = 0
    loss_sum_since_log = 0.0
    entropy_sum_since_log = 0.0
    pre_grad_norm_sum_since_log = 0.0
    pre_grad_norm_steps_since_log = 0
    post_grad_norm_sum_since_log = 0.0
    post_grad_norm_steps_since_log = 0
    r_sum_since_log = 0.0
    r_steps_since_log = 0
    r_group_sum_since_log = {k: 0.0 for k in R_METRIC_GROUP_KEYS}
    r_group_steps_since_log = {k: 0 for k in R_METRIC_GROUP_KEYS}
    attn_entropy_sum_since_log = defaultdict(float)
    attn_entropy_count_since_log = defaultdict(int)
    agc_clipped_params_since_log = 0
    agc_total_params_since_log = 0
    running_ce_window = deque()
    running_ce_tokens = 0
    running_ce_loss_sum = 0.0
    ema_loss = None
    best_ema = float("inf")
    bad_count = 0
    cooldown_count = 0
    accum_micro_count = 0
    accum_tokens_for_backward = 0
    accum_has_grad = False
    log_start = time.perf_counter()
    did_sanity = False

    batches_per_epoch = max(1, len(train_loader))
    if resume_batch_in_epoch >= batches_per_epoch:
        resume_epoch += int(resume_batch_in_epoch // batches_per_epoch)
        resume_batch_in_epoch = int(resume_batch_in_epoch % batches_per_epoch)

    if isinstance(restored_log_accum_state, dict):
        tokens_since_log = int(restored_log_accum_state.get("tokens_since_log", 0))
        loss_sum_since_log = float(restored_log_accum_state.get("loss_sum_since_log", 0.0))
        entropy_sum_since_log = float(restored_log_accum_state.get("entropy_sum_since_log", 0.0))
        pre_grad_norm_sum_since_log = float(restored_log_accum_state.get("pre_grad_norm_sum_since_log", 0.0))
        pre_grad_norm_steps_since_log = int(restored_log_accum_state.get("pre_grad_norm_steps_since_log", 0))
        post_grad_norm_sum_since_log = float(restored_log_accum_state.get("post_grad_norm_sum_since_log", 0.0))
        post_grad_norm_steps_since_log = int(restored_log_accum_state.get("post_grad_norm_steps_since_log", 0))
        r_sum_since_log = float(restored_log_accum_state.get("r_sum_since_log", 0.0))
        r_steps_since_log = int(restored_log_accum_state.get("r_steps_since_log", 0))
        restored_r_group_sum = restored_log_accum_state.get("r_group_sum_since_log", {})
        restored_r_group_steps = restored_log_accum_state.get("r_group_steps_since_log", {})
        if isinstance(restored_r_group_sum, dict):
            for k in R_METRIC_GROUP_KEYS:
                v = restored_r_group_sum.get(k)
                if isinstance(v, (int, float)):
                    r_group_sum_since_log[k] = float(v)
        if isinstance(restored_r_group_steps, dict):
            for k in R_METRIC_GROUP_KEYS:
                v = restored_r_group_steps.get(k)
                if isinstance(v, (int, float)):
                    r_group_steps_since_log[k] = int(v)
        agc_clipped_params_since_log = int(restored_log_accum_state.get("agc_clipped_params_since_log", 0))
        agc_total_params_since_log = int(restored_log_accum_state.get("agc_total_params_since_log", 0))
        running_ce_tokens = int(restored_log_accum_state.get("running_ce_tokens", 0))
        running_ce_loss_sum = float(restored_log_accum_state.get("running_ce_loss_sum", 0.0))
        running_ce_list = restored_log_accum_state.get("running_ce_window", [])
        if isinstance(running_ce_list, list):
            running_ce_window = deque()
            for row in running_ce_list:
                if not isinstance(row, (list, tuple)) or len(row) != 2:
                    continue
                tok_count = int(row[0])
                loss_sum_val = float(row[1])
                if tok_count <= 0:
                    continue
                running_ce_window.append([tok_count, loss_sum_val])
        attn_sum_map = restored_log_accum_state.get("attn_entropy_sum_since_log", {})
        attn_count_map = restored_log_accum_state.get("attn_entropy_count_since_log", {})
        if isinstance(attn_sum_map, dict):
            for k, v in attn_sum_map.items():
                if isinstance(k, str) and isinstance(v, (float, int)):
                    attn_entropy_sum_since_log[k] = float(v)
        if isinstance(attn_count_map, dict):
            for k, v in attn_count_map.items():
                if isinstance(k, str) and isinstance(v, (float, int)):
                    attn_entropy_count_since_log[k] = int(v)
    if isinstance(restored_lr_anneal_state, dict):
        restored_ema = restored_lr_anneal_state.get("ema_loss")
        if isinstance(restored_ema, (float, int)) and math.isfinite(float(restored_ema)):
            ema_loss = float(restored_ema)
        restored_best = restored_lr_anneal_state.get("best_ema")
        if isinstance(restored_best, (float, int)) and math.isfinite(float(restored_best)):
            best_ema = float(restored_best)
        bad_count = int(restored_lr_anneal_state.get("bad_count", 0))
        cooldown_count = int(restored_lr_anneal_state.get("cooldown_count", 0))

    # Checkpoints are emitted only at optimizer-step boundaries, so resume starts with a clean accumulation window.
    accum_micro_count = 0
    accum_tokens_for_backward = 0
    accum_has_grad = False

    for epoch in range(resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            if epoch == resume_epoch and batch_idx <= resume_batch_in_epoch:
                continue

            input_ids, target_ids, loss_mask = batch
            input_ids = input_ids.to(device, non_blocking=pin_memory)
            target_ids = target_ids.to(device, non_blocking=pin_memory)
            loss_mask = loss_mask.to(device, non_blocking=pin_memory)
            attn_pad_mask = input_ids.eq(args.pad_id)

            if not did_sanity:
                if input_ids.size(1) < 2:
                    raise AssertionError("Need at least two positions for next-token prediction.")
                assert torch.equal(target_ids[:, :-1], input_ids[:, 1:]), "target_ids must be input_ids shifted by one"
                assert not bool(loss_mask[input_ids == args.pad_id].any().item()), "loss_mask must exclude pad positions"
                assert not bool(loss_mask[:, 0].any().item()), "loss_mask position 0 must be False"
                logger.log("sanity: target shift and loss-mask checks passed on first batch")
                did_sanity = True

            step_attn_entropy = {}
            if args.track_attn_entropy:
                logits, attn_state = model(input_ids, pad_mask=attn_pad_mask, return_attn_entropy=True)
                step_attn_entropy = extract_attn_entropy_metrics(attn_state)
            else:
                logits = model(input_ids, pad_mask=attn_pad_mask)
            flat_mask = loss_mask.reshape(-1)
            token_count = int(flat_mask.sum().item())
            loss_sum_value = 0.0
            if token_count > 0:
                flat_logits = logits.reshape(-1, args.vocab_size)[flat_mask]
                flat_targets = target_ids.reshape(-1)[flat_mask]
                loss_sum = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
                loss = loss_sum / float(token_count)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}.")
                loss_sum.backward()
                accum_tokens_for_backward += token_count
                accum_has_grad = True

                with torch.no_grad():
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum(dim=-1)
                    entropy_sum = float(entropy[loss_mask].sum().item())

                loss_sum_value = float(loss_sum.item())
                loss_item = float(loss.item())
                if ema_loss is None:
                    ema_loss = loss_item
                else:
                    ema_loss = (LR_ANNEAL_ALPHA * loss_item) + ((1.0 - LR_ANNEAL_ALPHA) * ema_loss)
                loss_sum_since_log += loss_sum_value
                tokens_since_log += token_count
                entropy_sum_since_log += entropy_sum
                for attn_key, attn_value in step_attn_entropy.items():
                    if not math.isfinite(float(attn_value)):
                        continue
                    attn_entropy_sum_since_log[attn_key] += float(attn_value)
                    attn_entropy_count_since_log[attn_key] += 1
                running_ce_window.append([token_count, loss_sum_value])
                running_ce_tokens += token_count
                running_ce_loss_sum += loss_sum_value

            while running_ce_tokens > RUNNING_CE_WINDOW_TOKENS and len(running_ce_window) > 0:
                overflow = running_ce_tokens - RUNNING_CE_WINDOW_TOKENS
                oldest_tokens, oldest_loss_sum = running_ce_window[0]
                if oldest_tokens <= overflow:
                    running_ce_tokens -= oldest_tokens
                    running_ce_loss_sum -= oldest_loss_sum
                    running_ce_window.popleft()
                    continue

                trim_ratio = float(overflow) / float(max(1, oldest_tokens))
                trimmed_loss = oldest_loss_sum * trim_ratio
                keep_tokens = oldest_tokens - overflow
                keep_loss = oldest_loss_sum - trimmed_loss
                running_ce_window[0] = [keep_tokens, keep_loss]
                running_ce_tokens -= overflow
                running_ce_loss_sum -= trimmed_loss

            accum_micro_count += 1
            do_optimizer_step = (
                accum_micro_count >= int(args.grad_accum_steps)
                or batch_idx >= batches_per_epoch
            )
            if not do_optimizer_step:
                continue

            if not accum_has_grad:
                opt.zero_grad(set_to_none=True)
                accum_micro_count = 0
                accum_tokens_for_backward = 0
                accum_has_grad = False
                continue

            grad_scale = 1.0 / float(max(1, accum_tokens_for_backward))
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach().mul_(grad_scale)

            pre_grad_norm, _pre_weight_norm, r_grad, r_m1 = grad_weight_stats(
                model.parameters(), optimizer=opt
            )
            r_value = r_m1 if math.isfinite(r_m1) else r_grad
            r_values_by_group = {}
            for group_name in R_METRIC_GROUP_KEYS:
                r_values_by_group[group_name] = compute_r_value(
                    r_metric_param_groups.get(group_name, []),
                    optimizer=opt,
                )
            if args.grad_clip_mode == "global":
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                post_grad_norm = global_grad_norm(model.parameters())
            elif args.grad_clip_mode == "agc":
                clipped_count, total_count = adaptive_clip_grad_(
                    model.parameters(),
                    clip_factor=args.agc_clip_factor,
                    eps=args.agc_eps,
                )
                agc_clipped_params_since_log += clipped_count
                agc_total_params_since_log += total_count
                post_grad_norm = global_grad_norm(model.parameters())
            else:
                post_grad_norm = pre_grad_norm
            top_grad_norms_for_log = []
            if (global_step + 1) % args.log_every == 0:
                top_grad_norms_for_log = topk_grad_norms(model.named_parameters(), k=5)
            opt.step()
            lr = scheduler.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            if (
                LR_ANNEAL_CHECK_INTERVAL > 0
                and global_step % LR_ANNEAL_CHECK_INTERVAL == 0
                and ema_loss is not None
            ):
                best_ema = min(best_ema, ema_loss)
                if cooldown_count > 0:
                    cooldown_count -= 1
                else:
                    if ema_loss > (LR_ANNEAL_RATIO_UP * best_ema):
                        bad_count += 1
                    else:
                        bad_count = 0

                    if bad_count >= LR_ANNEAL_HOLD:
                        base_lr_now = max(float(lr), 1e-12)
                        updated_lrs = []
                        for group in opt.param_groups:
                            current_lr = float(group.get("lr", 0.0))
                            new_lr = max(current_lr * LR_ANNEAL_FACTOR, LR_ANNEAL_MIN_LR)
                            group["lr"] = new_lr
                            group["lr_scale"] = new_lr / base_lr_now
                            updated_lrs.append(new_lr)
                        bad_count = 0
                        cooldown_count = LR_ANNEAL_COOLDOWN
                        logger.log(
                            "LR anneal "
                            f"step={global_step} ema_loss={ema_loss:.6f} "
                            f"best_ema={best_ema:.6f} "
                            f"new_lrs={[f'{x:.6e}' for x in updated_lrs]}"
                        )

            pre_grad_norm_sum_since_log += pre_grad_norm
            pre_grad_norm_steps_since_log += 1
            post_grad_norm_sum_since_log += post_grad_norm
            post_grad_norm_steps_since_log += 1
            r_sum_since_log += r_value
            r_steps_since_log += 1
            for group_name, group_r in r_values_by_group.items():
                if not math.isfinite(group_r):
                    continue
                r_group_sum_since_log[group_name] += float(group_r)
                r_group_steps_since_log[group_name] += 1
            accum_micro_count = 0
            accum_tokens_for_backward = 0
            accum_has_grad = False

            if global_step % args.log_every == 0:
                if device == "mps":
                    torch.mps.synchronize()
                now = time.perf_counter()
                elapsed = max(1e-6, now - log_start)
                toks_per_sec = float(tokens_since_log) / elapsed
                epoch_pct = min(100.0, 100.0 * float(batch_idx) / float(batches_per_epoch))
                avg_ce = float(loss_sum_since_log) / float(max(1, tokens_since_log))
                avg_ppl = _safe_exp(avg_ce)
                avg_entropy = float(entropy_sum_since_log) / float(max(1, tokens_since_log))
                avg_pre_grad_norm = float(pre_grad_norm_sum_since_log) / float(max(1, pre_grad_norm_steps_since_log))
                avg_post_grad_norm = float(post_grad_norm_sum_since_log) / float(max(1, post_grad_norm_steps_since_log))
                avg_r = float(r_sum_since_log) / float(max(1, r_steps_since_log))
                avg_r_by_group = {}
                for group_name in R_METRIC_GROUP_KEYS:
                    steps = int(r_group_steps_since_log.get(group_name, 0))
                    if steps <= 0:
                        avg_r_by_group[group_name] = float("nan")
                    else:
                        avg_r_by_group[group_name] = float(r_group_sum_since_log[group_name]) / float(steps)
                train_running_ce = float(running_ce_loss_sum) / float(max(1, running_ce_tokens))

                log_msg = (
                    f"\nEpoch: {epoch + 1}/{args.epochs} ({epoch_pct:.2f}%), "
                    f"Step: {global_step}, Train CE: {avg_ce:.6f} nats/token, "
                    f"train_running_ce: {train_running_ce:.6f} nats/token ({running_ce_tokens} tok), "
                    f"Train PPL: {avg_ppl:.4f}, Train Entropy: {avg_entropy:.6f}, "
                    f"GradNorm(pre={avg_pre_grad_norm:.4f}, post={avg_post_grad_norm:.4f}), "
                    f"R: {avg_r:.6e}, "
                    f"R_embed: {avg_r_by_group['embed']:.6e}, "
                    f"R_atten: {avg_r_by_group['atten']:.6e}, "
                    f"R_mlp: {avg_r_by_group['mlp']:.6e}, "
                    f"R_lm_head: {avg_r_by_group['lm_head']:.6e}, "
                    f"Tokens/s: {toks_per_sec:.2f}, LR: {lr:.6e}"
                )
                if args.grad_clip_mode == "agc":
                    frac = float(agc_clipped_params_since_log) / float(max(1, agc_total_params_since_log))
                    log_msg += f", AGC clipped params: {agc_clipped_params_since_log}/{agc_total_params_since_log} ({frac:.1%})"
                if args.track_attn_entropy and len(attn_entropy_count_since_log) > 0:
                    ent_parts = []
                    for k in sorted(attn_entropy_count_since_log.keys()):
                        count = int(attn_entropy_count_since_log.get(k, 0))
                        if count <= 0:
                            continue
                        avg_v = float(attn_entropy_sum_since_log.get(k, 0.0)) / float(count)
                        ent_parts.append(f"{k}={avg_v:.6f}")
                    if ent_parts:
                        log_msg += ", " + ", ".join(ent_parts)
                log_msg += f", {format_top_grad_norms(top_grad_norms_for_log)}"
                logger.log(log_msg)

                log_start = time.perf_counter()
                tokens_since_log = 0
                loss_sum_since_log = 0.0
                entropy_sum_since_log = 0.0
                pre_grad_norm_sum_since_log = 0.0
                pre_grad_norm_steps_since_log = 0
                post_grad_norm_sum_since_log = 0.0
                post_grad_norm_steps_since_log = 0
                r_sum_since_log = 0.0
                r_steps_since_log = 0
                r_group_sum_since_log = {k: 0.0 for k in R_METRIC_GROUP_KEYS}
                r_group_steps_since_log = {k: 0 for k in R_METRIC_GROUP_KEYS}
                attn_entropy_sum_since_log = defaultdict(float)
                attn_entropy_count_since_log = defaultdict(int)
                agc_clipped_params_since_log = 0
                agc_total_params_since_log = 0

            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                val_metrics = evaluate_model(
                    model=model,
                    loader=val_loader,
                    device=device,
                    pad_id=args.pad_id,
                    vocab_size=args.vocab_size,
                    max_tokens=0,
                    max_steps=0,
                    pin_memory=pin_memory,
                )
                logger.log(
                    "Validation "
                    f"Step={global_step} CE={val_metrics['ce']:.6f} nats/token "
                    f"PPL={val_metrics['ppl']:.4f} Entropy={val_metrics['entropy']:.6f} "
                    f"tokens={val_metrics['tokens']} steps={val_metrics['steps']}"
                )

            if int(args.run_probes) > 0 and global_step % int(args.run_probes) == 0:
                probe_summary = run_debug_probes(
                    model=model,
                    tokenizer=probe_tokenizer,
                    prompts=probe_prompts,
                    device=device,
                    max_seq_len=int(args.max_seq_len),
                    pad_id=int(args.pad_id),
                    run_dir=os.path.join(LOGDIR, args.run_id),
                    step=int(global_step),
                    topk_eigs=max(1, int(args.probe_topk_eigs)),
                    generate_max_new_tokens=max(0, int(args.probe_gen_tokens)),
                    probe_layers=probe_layers,
                    capture_attn_entropy=bool(args.probe_attn_entropy),
                    log_fn=logger.log,
                    log_detailed_metrics=False,
                )
                agg = probe_summary.get("aggregate_metrics", {})
                agg_items = []
                for k in (
                    "q_mag_mean",
                    "k_mag_mean",
                    "v_mag_mean",
                    "attn_score_mean",
                    "embed_pair_cos_mean",
                    "enc_last_lambda1_over_trace",
                ):
                    v = agg.get(k, float("nan"))
                    agg_items.append(f"{k}={v:.6f}" if isinstance(v, (int, float)) else f"{k}=nan")
                logger.log(
                    f"\nProbeDebugSummary Step={global_step} probes={probe_summary['num_probes']} "
                    f"{' '.join(agg_items)} "
                    f"artifacts={os.path.join(LOGDIR, args.run_id, 'probe_debug')}"
                )

            if global_step % args.ckpt_every == 0:
                ckpt_path = os.path.join(LOGDIR, args.run_id, f"step_{global_step}.tar")
                next_epoch = int(epoch)
                next_batch_in_epoch = int(batch_idx)
                if next_batch_in_epoch >= batches_per_epoch:
                    next_epoch += 1
                    next_batch_in_epoch = 0
                torch.save(
                    {
                        "global_step": global_step,
                        "epoch": next_epoch,
                        "batch_in_epoch": next_batch_in_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_step_num": int(scheduler.step_num),
                        "rng_state": capture_rng_state(),
                        "lr_anneal_state": {
                            "ema_loss": None if ema_loss is None else float(ema_loss),
                            "best_ema": float(best_ema),
                            "bad_count": int(bad_count),
                            "cooldown_count": int(cooldown_count),
                        },
                        "log_accum_state": {
                            "tokens_since_log": int(tokens_since_log),
                            "loss_sum_since_log": float(loss_sum_since_log),
                            "entropy_sum_since_log": float(entropy_sum_since_log),
                            "pre_grad_norm_sum_since_log": float(pre_grad_norm_sum_since_log),
                            "pre_grad_norm_steps_since_log": int(pre_grad_norm_steps_since_log),
                            "post_grad_norm_sum_since_log": float(post_grad_norm_sum_since_log),
                            "post_grad_norm_steps_since_log": int(post_grad_norm_steps_since_log),
                            "r_sum_since_log": float(r_sum_since_log),
                            "r_steps_since_log": int(r_steps_since_log),
                            "r_group_sum_since_log": {k: float(v) for k, v in r_group_sum_since_log.items()},
                            "r_group_steps_since_log": {k: int(v) for k, v in r_group_steps_since_log.items()},
                            "agc_clipped_params_since_log": int(agc_clipped_params_since_log),
                            "agc_total_params_since_log": int(agc_total_params_since_log),
                            "running_ce_tokens": int(running_ce_tokens),
                            "running_ce_loss_sum": float(running_ce_loss_sum),
                            "running_ce_window": [list(x) for x in running_ce_window],
                            "attn_entropy_sum_since_log": dict(attn_entropy_sum_since_log),
                            "attn_entropy_count_since_log": dict(attn_entropy_count_since_log),
                        },
                        "config": cfg.__dict__,
                    },
                    ckpt_path,
                )

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        pad_id=args.pad_id,
        vocab_size=args.vocab_size,
        max_tokens=max(0, int(args.test_max_tokens)),
        max_steps=max(0, int(args.test_steps)),
        pin_memory=pin_memory,
    )
    logger.log(
        "Test "
        f"CE={test_metrics['ce']:.6f} nats/token "
        f"PPL={test_metrics['ppl']:.4f} Entropy={test_metrics['entropy']:.6f} "
        f"tokens={test_metrics['tokens']} steps={test_metrics['steps']}"
    )


if __name__ == "__main__":
    main()
