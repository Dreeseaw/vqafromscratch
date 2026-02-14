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
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from models.lm import TransformerV1, LMConfig
from train.lm_probe_debug import parse_probe_prompts, resolve_probe_file_path, run_debug_probes


LOGDIR = "logs/"
LOGFILE = "logfile.txt"
SEED = 35
RUNNING_CE_WINDOW_TOKENS = 200_000

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
        if self.schedule == "flat":
            return self.base_lr
        # cosine decay
        progress = (self.step_num - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
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


def _global_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    return float(math.sqrt(max(total, 0.0)))


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
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "flat"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--eval_every_steps", type=int, default=1000)
    parser.add_argument("--val_max_tokens", type=int, default=200000)
    parser.add_argument("--val_steps", type=int, default=0)
    parser.add_argument("--test_max_tokens", type=int, default=0)
    parser.add_argument("--test_steps", type=int, default=0)
    parser.add_argument("--run_probes", type=int, default=0)
    parser.add_argument("--probe_file", type=str, default=None)
    parser.add_argument("--probe_topk_eigs", type=int, default=5)
    parser.add_argument("--probe_gen_tokens", type=int, default=48)

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

    cfg = LMConfig(
        vocab_size=args.vocab_size,
        embed_size=args.d_model,
        num_heads=args.n_heads,
        mlp_ratio=args.ff_mult,
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

    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=args.weight_decay,
        )

    total_steps = args.max_steps if args.max_steps is not None else args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = WarmupScheduler(opt, args.lr, warmup_steps, total_steps, schedule=args.schedule)

    global_step = 0
    if args.checkpoint is not None:
        ckpt_file = os.path.join(LOGDIR, args.run_id, f"step_{args.checkpoint}.tar")
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = int(checkpoint.get("global_step", 0))
        scheduler.step_num = global_step

    logger = Logger(args.run_id, args.checkpoint)
    log_params(model, logger)
    logger.log(f"batch size: {args.batch_size}")
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
    logger.log(
        f"sanity: special tokens pad={args.pad_id}, eos={args.eos_id}, "
        f"bos={args.bos_id}, vocab={args.vocab_size}"
    )
    if int(args.run_probes) > 0:
        logger.log(
            f"probe debug cadence: every {args.run_probes} steps | prompts={len(probe_prompts)} | "
            f"probe_file={probe_file_path}"
        )
        if len(probe_prompts) != 5:
            logger.log(f"warning: expected 5 probes, found {len(probe_prompts)} in {probe_file_path}")

    model.train()
    tokens_since_log = 0
    loss_sum_since_log = 0.0
    entropy_sum_since_log = 0.0
    grad_norm_sum_since_log = 0.0
    grad_norm_steps_since_log = 0
    pre_clip_sum_since_log = 0.0
    pre_clip_steps_since_log = 0
    running_ce_window = deque()
    running_ce_tokens = 0
    running_ce_loss_sum = 0.0
    log_start = time.perf_counter()
    did_sanity = False

    batches_per_epoch = max(1, len(train_loader))
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.max_steps is not None and global_step >= args.max_steps:
                break

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

            logits = model(input_ids, pad_mask=attn_pad_mask)
            flat_mask = loss_mask.reshape(-1)
            token_count = int(flat_mask.sum().item())
            if token_count <= 0:
                continue
            flat_logits = logits.reshape(-1, args.vocab_size)[flat_mask]
            flat_targets = target_ids.reshape(-1)[flat_mask]
            loss_sum = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
            loss = loss_sum / float(token_count)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}.")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad is not None and args.clip_grad > 0:
                # HACK: pre_clip actually tracks the post clipped value. oopsies!
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad).item())
                pre_clip  = _global_grad_norm(model)
            else:
                grad_norm = _global_grad_norm(model)
                pre_clip  = grad_norm
            opt.step()
            lr = scheduler.step()

            with torch.no_grad():
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                entropy_sum = float(entropy[loss_mask].sum().item())

            global_step += 1
            loss_sum_value = float(loss_sum.item())
            loss_sum_since_log += loss_sum_value
            tokens_since_log += token_count
            entropy_sum_since_log += entropy_sum
            grad_norm_sum_since_log += grad_norm
            grad_norm_steps_since_log += 1
            pre_clip_sum_since_log += pre_clip
            pre_clip_steps_since_log += 1
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
                avg_grad_norm = float(grad_norm_sum_since_log) / float(max(1, grad_norm_steps_since_log))
                avg_pre_clip = float(pre_clip_sum_since_log) / float(max(1, pre_clip_steps_since_log))
                train_running_ce = float(running_ce_loss_sum) / float(max(1, running_ce_tokens))

                log_msg = (
                    f"\nEpoch: {epoch + 1}/{args.epochs} ({epoch_pct:.2f}%), "
                    f"Step: {global_step}, Train CE: {avg_ce:.6f} nats/token, "
                    f"train_running_ce: {train_running_ce:.6f} nats/token ({running_ce_tokens} tok), "
                    f"Train PPL: {avg_ppl:.4f}, Train Entropy: {avg_entropy:.6f}, "
                    f"GradNorm: {avg_pre_clip:.4f} (Pre: {avg_grad_norm:.4f}), Tokens/s: {toks_per_sec:.2f}, LR: {lr:.6e}"
                )
                logger.log(log_msg)

                log_start = time.perf_counter()
                tokens_since_log = 0
                loss_sum_since_log = 0.0
                entropy_sum_since_log = 0.0
                grad_norm_sum_since_log = 0.0
                grad_norm_steps_since_log = 0
                pre_clip_sum_since_log = 0.0
                pre_clip_steps_since_log = 0

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
                if args.checkpoint is not None:
                    ckpt_path = os.path.join(LOGDIR, args.run_id, f"step_{global_step}_from_{args.checkpoint}.tar")
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
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
