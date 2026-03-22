from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any, Dict, List

import torch


_SKIP_KEYS = {
    "run_id",
    "checkpoint",
    "eval_only",
    "eval_split",
    "manual_max_steps",
    "max_steps",
    "num_visual_tokens",
    "batch_size",
    "grad_accum_steps",
    "eval_batch_size",
    "seed",
}


def _cli_args(train_args: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in sorted(train_args.keys()):
        if key in _SKIP_KEYS:
            continue
        value = train_args[key]
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            out.append(flag if value else f"--no-{key}")
            continue
        if isinstance(value, (str, int, float)):
            out.extend([flag, str(value)])
            continue
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--run_prefix", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=96)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=100)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(str(args.checkpoint), map_location="cpu")
    train_args = dict(payload.get("train_args") or {})
    if not train_args:
        raise ValueError(f"Checkpoint {args.checkpoint} did not contain train_args.")

    base = _cli_args(train_args)
    for k in (32, 49, 96):
        run_id = f"{args.run_prefix}_k{k}"
        cmd = [
            "./runmm.sh",
            run_id,
            *base,
            "--num_visual_tokens",
            str(int(k)),
            "--batch_size",
            str(int(args.batch_size)),
            "--grad_accum_steps",
            str(int(args.grad_accum_steps)),
            "--eval_batch_size",
            str(int(args.eval_batch_size)),
            "--max_steps",
            str(int(args.max_steps)),
            "--manual_max_steps",
            "--seed",
            str(int(args.seed)),
            "--eval_every",
            str(int(args.eval_every)),
            "--eval_batches",
            str(int(args.eval_batches)),
            "--final_eval_batches",
            "0",
        ]
        print("[cement-query-probe] " + " ".join(cmd))
        if not bool(args.dry_run):
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
