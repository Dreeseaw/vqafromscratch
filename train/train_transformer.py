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
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.lm import TransformerV1, LMConfig


LOGDIR = "logs/"
LOGFILE = "logfile.txt"
SEED = 35
IGNORE_INDEX = -100

# Length buckets (inclusive ranges) for bucketed batching
BUCKET_RANGES = [
    (1, 64),
    (65, 256),
    (257, 512),
    (513, 1024),
]


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
        add_bos: bool,
        add_eos: bool,
        bos_id: int,
        eos_id: int,
        ignore_index: int,
    ):
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.fixed_len = fixed_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[torch.Tensor]):
        if self.fixed_len and not (self.add_bos or self.add_eos):
            tokens = torch.stack(batch, dim=0)
        else:
            content_max_len = self.max_seq_len
            if self.add_bos:
                content_max_len -= 1
            if self.add_eos:
                content_max_len -= 1
            if content_max_len <= 0:
                raise ValueError("max_seq_len too small to fit BOS/EOS")

            tokens = torch.full((len(batch), self.max_seq_len), self.pad_id, dtype=torch.long)
            for i, seq in enumerate(batch):
                length = min(seq.numel(), content_max_len)
                if length > 0:
                    offset = 0
                    if self.add_bos:
                        tokens[i, 0] = self.bos_id
                        offset = 1
                    tokens[i, offset:offset + length] = seq[:length]
                    if self.add_eos and (offset + length) < self.max_seq_len:
                        tokens[i, offset + length] = self.eos_id

        tgt_in = tokens[:, :-1]
        labels = tokens[:, 1:].clone()
        labels[(labels == self.pad_id) | (labels == self.bos_id)] = self.ignore_index
        pad_mask = tgt_in.eq(self.pad_id)
        return tgt_in, labels, pad_mask


def build_length_buckets(
    dataset: Dataset,
    max_seq_len: int,
    bucket_ranges: List[tuple],
    add_bos: bool,
    add_eos: bool,
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
        if add_bos:
            length += 1
        if add_eos:
            length += 1
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
        self.bucket_indices = bucket_indices
        self.bucket_token_counts = np.asarray(bucket_token_counts, dtype=np.float64)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        total_seqs = sum(len(b) for b in bucket_indices)
        if drop_last:
            self.batches_per_epoch = total_seqs // batch_size
        else:
            self.batches_per_epoch = math.ceil(total_seqs / batch_size)

        if self.bucket_token_counts.sum() <= 0:
            raise ValueError("No tokens found for bucket sampling.")
        self.bucket_probs = self.bucket_token_counts / self.bucket_token_counts.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for bucket in self.bucket_indices:
            rng.shuffle(bucket)
        cursors = [0 for _ in self.bucket_indices]
        # Sample bucket ids once per epoch using fixed token-proportional probabilities.
        choices = rng.choice(len(self.bucket_indices), size=self.batches_per_epoch, p=self.bucket_probs)

        for bucket_idx in choices:
            bucket = self.bucket_indices[bucket_idx]
            if len(bucket) == 0:
                continue
            if cursors[bucket_idx] + self.batch_size > len(bucket):
                # Reshuffle and reset cursor when a bucket runs out.
                rng.shuffle(bucket)
                cursors[bucket_idx] = 0
            start = cursors[bucket_idx]
            end = start + self.batch_size
            batch = bucket[start:end]
            cursors[bucket_idx] = end
            if len(batch) < self.batch_size:
                # bucket smaller than batch size: sample with replacement from this bucket
                needed = self.batch_size - len(batch)
                extra = rng.choice(bucket, size=needed, replace=True).tolist()
                batch = batch + extra
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=None)

    # tokens / data
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--pad_id", type=int, default=0)
    parser.add_argument("--bos_id", type=int, default=2)
    parser.add_argument("--eos_id", type=int, default=3)
    parser.add_argument("--add_bos", action="store_true")
    parser.add_argument("--add_eos", action="store_true")
    parser.add_argument("--no_add_bos", action="store_false", dest="add_bos")
    parser.add_argument("--no_add_eos", action="store_false", dest="add_eos")

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

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "flat"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    # performance
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--log_eos_acc", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    parser.set_defaults(add_bos=True, add_eos=True)
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

    dataset = load_dataset(args.train_data, args.max_seq_len)
    fixed_len = getattr(dataset, "fixed_len", False)
    collate_fn = CollatePad(
        args.max_seq_len,
        args.pad_id,
        fixed_len,
        args.add_bos,
        args.add_eos,
        args.bos_id,
        args.eos_id,
        IGNORE_INDEX,
    )
    dataset_meta = getattr(dataset, "meta", None)

    bucket_ranges, bucket_indices, bucket_token_counts = build_length_buckets(
        dataset, args.max_seq_len, BUCKET_RANGES, args.add_bos, args.add_eos
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
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)

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
        attn_impl=args.attn_impl,
        sdp_backend=args.sdp_backend,
    )
    model = TransformerV1(cfg).to(device)

    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.max_steps if args.max_steps is not None else args.epochs * len(loader)
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
    if dataset_meta:
        for key, arg_val in (
            ("pad_id", args.pad_id),
            ("bos_id", args.bos_id),
            ("eos_id", args.eos_id),
        ):
            meta_val = dataset_meta.get(key)
            if meta_val is not None and int(meta_val) != int(arg_val):
                logger.log(f"warning: meta.{key}={meta_val} != args.{key}={arg_val}")
        if dataset_meta.get("add_bos") is True and args.add_bos:
            logger.log("warning: dataset meta indicates BOS already added; --add_bos will add another BOS.")
        if dataset_meta.get("add_eos") is True and args.add_eos:
            logger.log("warning: dataset meta indicates EOS already added; --add_eos will add another EOS.")

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

    model.train()
    tokens_since_log = torch.zeros((), device=device)
    log_start = time.perf_counter()
    did_sanity = False

    batches_per_epoch = max(1, len(loader))
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(loader, start=1):
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            # batch returns (tgt_in, labels, pad_mask)
            tgt_in, labels, tgt_key_padding_mask = batch
            tgt_in = tgt_in.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            tgt_key_padding_mask = tgt_key_padding_mask.to(device, non_blocking=pin_memory)

            if not did_sanity:
                if tgt_in.size(1) > 0:
                    bos_ok = tgt_in[:, 0].eq(args.bos_id).all().item()
                else:
                    bos_ok = False
                eos_any = (labels == args.eos_id).any().item()
                if args.add_bos and not bos_ok:
                    logger.log("warning: BOS not found at position 0 for all sequences")
                if not eos_any:
                    logger.log("warning: EOS not found in first batch")
                did_sanity = True

            logits = model(tgt_in, pad_mask=tgt_key_padding_mask)

            loss_sum = F.cross_entropy(
                logits.reshape(-1, args.vocab_size),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            token_count = (labels != IGNORE_INDEX).sum().clamp_min(1)
            loss = loss_sum / token_count

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad is not None and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            lr = scheduler.step()

            global_step += 1
            tokens_since_log += token_count

            if global_step % args.log_every == 0:
                if device == "mps":
                    torch.mps.synchronize()
                now = time.perf_counter()
                elapsed = max(1e-6, now - log_start)
                toks_per_sec = float(tokens_since_log.item()) / elapsed
                epoch_pct = min(100.0, 100.0 * float(batch_idx) / float(batches_per_epoch))

                log_msg = (
                    f"\nEpoch: {epoch + 1}/{args.epochs} ({epoch_pct:.2f}%), "
                    f"Step: {global_step}, Loss/token: {loss.detach():.6f}, "
                    f"Tokens/s: {toks_per_sec:.2f}, LR: {lr:.6e}"
                )
                logger.log(log_msg)

                if args.log_eos_acc:
                    with torch.no_grad():
                        eos_mask = labels == args.eos_id
                        if eos_mask.any():
                            preds = logits.argmax(dim=-1)
                            eos_acc = (preds[eos_mask] == args.eos_id).float().mean().item()
                            logger.log(f"EOS acc: {eos_acc:.4f}")
                        else:
                            logger.log("EOS acc: n/a (no EOS in batch)")

                log_start = time.perf_counter()
                tokens_since_log = torch.zeros((), device=device)

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


if __name__ == "__main__":
    main()
