from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from models.bpe_tokenizer import ByteBPETokenizer
from train.vqa_data import VQAv2Dataset


LOGFILE = "logfile.txt"


class Logger:
    def __init__(self, run_id: str) -> None:
        self.run_id = str(run_id)
        self.base = os.path.join("logs", self.run_id)
        os.makedirs(self.base, exist_ok=True)
        self.path = os.path.join(self.base, LOGFILE)

    def log(self, text: str) -> None:
        print(text, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract VQAv2 teacher answer-vocab logits from Qwen2.5-VL-3B.")
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--images_root", type=str, default="images")
    ap.add_argument("--annotations_root", type=str, default="data/vqav2")
    ap.add_argument("--student_tokenizer_path", type=str, default="logs/mix_bpe_16k/tokenizer.pt")
    ap.add_argument("--split", type=str, default="train", choices=["train"])
    ap.add_argument("--answer_top_k", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--min_batch_size", type=int, default=1)
    ap.add_argument("--shard_size", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--progress_every", type=int, default=10000)
    return ap.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _torch_dtype_for_device(device: str) -> torch.dtype:
    if str(device) != "cuda":
        return torch.float32
    if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
        return torch.bfloat16
    return torch.float16


def _build_answer_vocab(items: Sequence[Dict[str, Any]], top_k: int) -> List[str]:
    counts: Counter[str] = Counter()
    for item in items:
        answer = str(item.get("answer", "")).strip()
        if answer:
            counts[answer] += 1
    return [ans for ans, _n in counts.most_common(max(1, int(top_k)))]


def _teacher_first_token_id(tokenizer: Any, answer: str) -> int | None:
    for text in (f" {answer}", answer):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            return int(ids[0])
    return None


def _student_first_token_id(tokenizer: ByteBPETokenizer, answer: str) -> int | None:
    ids = tokenizer.encode(answer, add_bos=False, add_eos=False).tolist()
    if not ids:
        return None
    return int(ids[0])


def _load_progress(progress_path: Path) -> Dict[str, Any]:
    if progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    return {
        "next_index": 0,
        "samples_completed": 0,
        "samples_skipped": 0,
        "shard_index": 0,
        "last_batch_size": None,
    }


def _scan_flushed_state(output_dir: Path) -> Tuple[int, int]:
    total = 0
    shard_count = 0
    for shard_path in sorted(output_dir.glob("shard_*.pt")):
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        qids = payload.get("question_ids")
        if not isinstance(qids, torch.Tensor):
            continue
        total += int(qids.shape[0])
        shard_count += 1
    return total, shard_count


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def _flush_shard(
    output_dir: Path,
    *,
    shard_index: int,
    question_ids: List[int],
    teacher_logits: List[torch.Tensor],
) -> str:
    shard_name = f"shard_{int(shard_index):06d}.pt"
    shard_path = output_dir / shard_name
    tmp_path = output_dir / f"{shard_name}.tmp"
    payload = {
        "question_ids": torch.tensor(question_ids, dtype=torch.long),
        "teacher_logits": torch.stack(teacher_logits, dim=0).to(dtype=torch.float16, device="cpu"),
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, shard_path)
    return shard_name


def _eta_text(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0 or elapsed_s <= 0.0:
        return "unknown"
    rate = float(done) / float(elapsed_s)
    if rate <= 1e-8:
        return "unknown"
    remain_s = max(0.0, float(total - done) / rate)
    return f"{remain_s / 3600.0:.2f}h"


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    logger = Logger(args.run_id)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    meta_path = output_dir / "meta.json"

    logger.log(f"[teacher] run_id={args.run_id} output_dir={str(output_dir)}")
    logger.log(f"[teacher] model_name={args.model_name} device={args.device} requested_batch_size={int(args.batch_size)}")

    ds = VQAv2Dataset(
        images_root=str(args.images_root),
        annotations_root=str(args.annotations_root),
        split="train",
        limit=max(0, int(args.limit)),
        skip_missing_images=True,
    )
    items = list(ds.items)
    total_items = len(items)
    if total_items <= 0:
        raise RuntimeError("No VQAv2 train items found for teacher extraction.")

    student_tokenizer = ByteBPETokenizer.load(str(args.student_tokenizer_path))
    processor = AutoProcessor.from_pretrained(str(args.model_name))
    teacher_tokenizer = processor.tokenizer

    raw_answer_vocab = _build_answer_vocab(items, int(args.answer_top_k))
    answer_vocab: List[str] = []
    teacher_first_token_ids: List[int] = []
    student_first_token_ids: List[int] = []
    dropped_answers = 0
    for answer in raw_answer_vocab:
        teacher_id = _teacher_first_token_id(teacher_tokenizer, answer)
        student_id = _student_first_token_id(student_tokenizer, answer)
        if teacher_id is None or student_id is None:
            dropped_answers += 1
            continue
        answer_vocab.append(str(answer))
        teacher_first_token_ids.append(int(teacher_id))
        student_first_token_ids.append(int(student_id))
    if not answer_vocab:
        raise RuntimeError("Could not build a shared answer vocabulary for teacher distillation.")

    meta: Dict[str, Any]
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {
            "dataset": "vqav2_train",
            "model_name": str(args.model_name),
            "split": "train",
            "images_root": os.path.abspath(str(args.images_root)),
            "annotations_root": os.path.abspath(str(args.annotations_root)),
            "student_tokenizer_path": os.path.abspath(str(args.student_tokenizer_path)),
            "answer_vocab_size": int(len(answer_vocab)),
            "answer_vocab": answer_vocab,
            "teacher_first_token_ids": teacher_first_token_ids,
            "student_first_token_ids": student_first_token_ids,
            "requested_answer_top_k": int(args.answer_top_k),
            "dropped_answers": int(dropped_answers),
            "total_items": int(total_items),
            "shards": [],
        }
        _save_json(meta_path, meta)

    progress = _load_progress(progress_path)
    flushed_count, shard_file_count = _scan_flushed_state(output_dir)
    safe_next_index = int(progress.get("next_index", 0) or 0)
    next_index = int(progress.get("next_index_processed", safe_next_index) or safe_next_index)
    shard_index = max(int(progress.get("shard_index", 0) or 0), int(shard_file_count))
    safe_samples_completed = int(progress.get("samples_completed", 0) or 0)
    samples_completed = int(progress.get("samples_completed_processed", safe_samples_completed) or safe_samples_completed)
    samples_skipped = int(progress.get("samples_skipped", 0) or 0)
    dynamic_batch_size = int(progress.get("last_batch_size") or args.batch_size)
    dynamic_batch_size = max(int(args.min_batch_size), min(int(args.batch_size), dynamic_batch_size))
    if safe_next_index < int(flushed_count) or safe_samples_completed < int(flushed_count):
        safe_next_index = int(flushed_count)
        safe_samples_completed = int(flushed_count)
    if next_index < safe_next_index or samples_completed < safe_samples_completed:
        next_index = int(safe_next_index)
        samples_completed = int(safe_samples_completed)

    logger.log(
        f"[teacher] total_items={total_items} answer_vocab={len(answer_vocab)} dropped_answers={dropped_answers} "
        f"resume_next_index={safe_next_index} shard_index={shard_index}"
    )
    if int(progress.get("next_index", 0) or 0) > int(flushed_count):
        logger.log(
            f"[teacher] note: progress cursor exceeded flushed shards; resuming from safe boundary={flushed_count}"
        )

    if safe_next_index >= total_items:
        logger.log("[teacher] extraction already complete; nothing to do")
        return
    next_index = int(safe_next_index)
    samples_completed = int(safe_samples_completed)

    dtype = _torch_dtype_for_device(str(args.device))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_name),
        torch_dtype=dtype,
    ).to(str(args.device)).eval()
    if str(args.device) == "cuda":
        torch.cuda.empty_cache()

    teacher_first_token_ids_t = torch.tensor(teacher_first_token_ids, dtype=torch.long, device=str(args.device))
    shard_qids: List[int] = []
    shard_logits: List[torch.Tensor] = []
    start_time = time.time()
    last_log_time = start_time
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            logger.log("[teacher] stop requested; will flush current shard buffer before exiting")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def save_progress() -> None:
        payload = {
            "next_index": int(safe_next_index),
            "samples_completed": int(safe_samples_completed),
            "next_index_processed": int(next_index),
            "samples_completed_processed": int(samples_completed),
            "samples_skipped": int(samples_skipped),
            "shard_index": int(shard_index),
            "last_batch_size": int(dynamic_batch_size),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_json(progress_path, payload)

    while next_index < total_items:
        bs = max(int(args.min_batch_size), min(dynamic_batch_size, total_items - next_index))
        batch_items = items[next_index : next_index + bs]
        pil_images: List[Image.Image] = []
        texts: List[str] = []
        batch_qids: List[int] = []
        skipped_in_batch = 0
        for item in batch_items:
            try:
                img = Image.open(str(item["image_path"])).convert("RGB")
            except Exception:
                samples_skipped += 1
                skipped_in_batch += 1
                continue
            prompt = (
                "Answer the following question about the image with a single word or short phrase.\n"
                f"Question: {str(item['question']).strip()}\n"
                "Answer:"
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pil_images.append(img)
            texts.append(text)
            batch_qids.append(int(item["question_id"]))
        next_index += len(batch_items)
        if not batch_qids:
            save_progress()
            continue

        retry_batch = False
        while True:
            try:
                inputs = processor(
                    text=texts,
                    images=pil_images,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {
                    key: (value.to(str(args.device)) if hasattr(value, "to") else value)
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    outputs = model(**inputs)
                attn_mask = inputs.get("attention_mask")
                if not isinstance(attn_mask, torch.Tensor):
                    raise RuntimeError("Teacher processor did not return attention_mask")
                last_pos = attn_mask.long().sum(dim=1) - 1
                batch_index = torch.arange(int(last_pos.shape[0]), device=last_pos.device)
                next_token_logits = outputs.logits[batch_index, last_pos]
                vocab_logits = next_token_logits.index_select(dim=-1, index=teacher_first_token_ids_t)
                cpu_logits = vocab_logits.detach().to(dtype=torch.float16, device="cpu")
                for qid, row in zip(batch_qids, cpu_logits, strict=True):
                    shard_qids.append(int(qid))
                    shard_logits.append(row.contiguous())
                samples_completed += int(len(batch_qids))
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" not in msg or bs <= int(args.min_batch_size):
                    raise
                dynamic_batch_size = max(int(args.min_batch_size), bs // 2)
                logger.log(
                    f"[teacher] CUDA OOM at next_index={next_index} batch_size={bs}; retrying with batch_size={dynamic_batch_size}"
                )
                if str(args.device) == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                next_index -= len(batch_items)
                retry_batch = True
                break
        for img in pil_images:
            try:
                img.close()
            except Exception:
                pass
        if retry_batch:
            save_progress()
            continue

        if len(shard_qids) >= int(args.shard_size) or next_index >= total_items:
            shard_name = _flush_shard(
                output_dir,
                shard_index=shard_index,
                question_ids=shard_qids,
                teacher_logits=shard_logits,
            )
            shard_qids.clear()
            shard_logits.clear()
            shard_index += 1
            safe_next_index = int(next_index)
            safe_samples_completed = int(samples_completed)
            meta["shards"] = list(meta.get("shards") or []) + [shard_name]
            _save_json(meta_path, meta)
            save_progress()

        if stop_requested:
            if shard_qids:
                shard_name = _flush_shard(
                    output_dir,
                    shard_index=shard_index,
                    question_ids=shard_qids,
                    teacher_logits=shard_logits,
                )
                shard_qids.clear()
                shard_logits.clear()
                shard_index += 1
                safe_next_index = int(next_index)
                safe_samples_completed = int(samples_completed)
                meta["shards"] = list(meta.get("shards") or []) + [shard_name]
                _save_json(meta_path, meta)
            save_progress()
            logger.log(
                f"[teacher] graceful stop complete safe_resume_index={safe_next_index} "
                f"safe_samples_completed={safe_samples_completed}"
            )
            return

        elapsed_s = max(1e-6, time.time() - start_time)
        if (
            samples_completed <= len(batch_qids)
            or (samples_completed % max(1, int(args.log_every)) == 0)
            or (time.time() - last_log_time) >= 120.0
            or (samples_completed % max(1, int(args.progress_every)) == 0)
        ):
            sps = float(samples_completed) / elapsed_s
            teacher_line = (
                f"[teacher] step={samples_completed} total={total_items} skipped={samples_skipped} "
                f"samples_per_s={sps:.3f} batch_size={len(batch_qids) or bs} eta={_eta_text(samples_completed, total_items, elapsed_s)}"
            )
            logger.log(teacher_line)
            logger.log(
                f"[mm] step={samples_completed} loss=0.0000 loss_vqa=0.0000 loss_tokens=0 "
                f"lr=0 steps_per_s={sps:.3f}"
            )
            last_log_time = time.time()
            save_progress()
            if str(args.device) == "cuda":
                torch.cuda.empty_cache()

    if shard_qids:
        shard_name = _flush_shard(
            output_dir,
            shard_index=shard_index,
            question_ids=shard_qids,
            teacher_logits=shard_logits,
        )
        shard_index += 1
        meta["shards"] = list(meta.get("shards") or []) + [shard_name]
        _save_json(meta_path, meta)

    save_progress()
    elapsed_s = max(1e-6, time.time() - start_time)
    logger.log(
        f"[teacher] COMPLETE samples={samples_completed} skipped={samples_skipped} "
        f"elapsed_h={elapsed_s / 3600.0:.2f} avg_samples_per_s={float(samples_completed) / elapsed_s:.3f}"
    )
    logger.log(f"[mm] final checkpoint: {str(output_dir)}")


if __name__ == "__main__":
    main()
