import argparse
import json
import os
import random
import string
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from evals.probe import ANNO_DIR_DEFAULT
except Exception:
    ANNO_DIR_DEFAULT = os.path.abspath("./annotations/")

from models.bpe_tokenizer import ByteBPETokenizer


LOGDIR = "logs/"

try:
    import orjson as _orjson  # type: ignore

    def _json_loads(line: str) -> Dict:
        return _orjson.loads(line)

except Exception:  # pragma: no cover - optional speedup
    def _json_loads(line: str) -> Dict:
        return json.loads(line)


def _find_captions_json(anno_dir: str) -> str:
    candidates = [
        os.path.join(anno_dir, "annotations/captions_train2017.json"),
        os.path.join(anno_dir, "annotations/captions_train2014.json"),
        os.path.join(anno_dir, "captions_train2017.json"),
        os.path.join(anno_dir, "captions_train2014.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find COCO captions json under {anno_dir}."
    )


def load_captions(path: str, limit: int = 0) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    captions: List[str] = []
    for ann in data.get("annotations", []):
        cap = ann.get("caption")
        if not cap:
            continue
        captions.append(cap)
        if limit > 0 and len(captions) >= limit:
            break
    return captions


def _word_count(text: str, mode: str) -> int:
    if not text or not text.strip():
        return 0
    if mode == "fast":
        # Fast approximate count: whitespace separators only.
        return text.count(" ") + text.count("\n") + 1
    return len(text.split())


def _sample_captions_by_words(
    captions: List[str], target_words: int, seed: int, word_count_mode: str
) -> List[str]:
    rng = random.Random(seed)
    idxs = list(range(len(captions)))
    rng.shuffle(idxs)
    out: List[str] = []
    total = 0
    for i in idxs:
        cap = captions[i]
        wc = _word_count(cap, word_count_mode)
        if wc <= 0:
            continue
        out.append(cap)
        total += wc
        if total >= target_words:
            break
    return out


def _load_manifest_total_words(articles_jsonl: str) -> int:
    manifest = Path(articles_jsonl).with_name("manifest.json")
    if not manifest.exists():
        return 0
    try:
        with manifest.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("total_words", 0))
    except Exception:
        return 0


def _sample_wiki_by_words(
    path: str,
    target_words: int,
    seed: int,
    word_count_mode: str,
    sample_mode: str,
    total_words: int,
    stop_on_target: bool,
    filter_az: bool,
    workers: int = 0,
    chunk_lines: int = 2000,
) -> Tuple[List[str], Dict]:
    rng = random.Random(seed)
    texts: List[str] = []
    total = 0
    stats: Dict = {"mode": sample_mode}
    accept_prob = None
    if sample_mode == "random":
        if total_words <= 0:
            raise ValueError(
                "random sampling requires --wiki_total_words or a valid manifest.json"
            )
        accept_prob = min(1.0, target_words / float(total_words))
        stats["accept_prob"] = accept_prob
        stats["total_words"] = total_words
    if workers and workers > 1:
        ctx = None
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        def _chunk_iter(fp):
            chunk: List[str] = []
            for line in fp:
                if not line.strip():
                    continue
                chunk.append(line)
                if len(chunk) >= chunk_lines:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

        with open(path, "r", encoding="utf-8") as f:
            pool = ctx.Pool(processes=workers)
            terminated = False
            try:
                tasks = (
                    (
                        chunk,
                        accept_prob,
                        word_count_mode,
                        filter_az,
                        sample_mode,
                        seed + i,
                    )
                    for i, chunk in enumerate(_chunk_iter(f))
                )
                for out_texts, out_words in pool.imap_unordered(
                    _process_wiki_chunk, tasks, chunksize=1
                ):
                    if out_texts:
                        texts.extend(out_texts)
                        total += out_words
                    if stop_on_target and total >= target_words:
                        pool.terminate()
                        terminated = True
                        break
            finally:
                if not terminated:
                    pool.close()
                pool.join()
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = _json_loads(line)
                text = obj.get("text", "")
                if not text:
                    continue
                if filter_az:
                    title = obj.get("title", "")
                    if not title or title[:1].upper() not in string.ascii_uppercase:
                        continue
                if sample_mode == "random" and rng.random() >= accept_prob:
                    continue
                wc = _word_count(text, word_count_mode)
                if wc <= 0:
                    continue
                texts.append(text)
                total += wc
                if stop_on_target and total >= target_words:
                    break

    if not stop_on_target and total > target_words:
        rng.shuffle(texts)
        trimmed: List[str] = []
        t = 0
        for text in texts:
            wc = _word_count(text, word_count_mode)
            if wc <= 0:
                continue
            trimmed.append(text)
            t += wc
            if t >= target_words:
                break
        texts = trimmed
        total = t

    stats["actual_words"] = total
    return texts, stats


def _estimate_tokens_per_word(
    tokenizer: ByteBPETokenizer, texts: List[str], word_count_mode: str
) -> Tuple[float, int, int]:
    total_words = 0
    total_tokens = 0
    for text in texts:
        wc = _word_count(text, word_count_mode)
        if wc <= 0:
            continue
        total_words += wc
        total_tokens += int(
            tokenizer.encode(text, add_bos=False, add_eos=False).numel()
        )
    if total_words <= 0:
        return 0.0, 0, 0
    return total_tokens / float(total_words), total_tokens, total_words


def _process_wiki_chunk(args):
    lines, accept_prob, word_count_mode, filter_az, sample_mode, seed = args
    rng = random.Random(seed)
    out_texts: List[str] = []
    total = 0
    for line in lines:
        if not line.strip():
            continue
        obj = _json_loads(line)
        text = obj.get("text", "")
        if not text:
            continue
        if filter_az:
            title = obj.get("title", "")
            if not title or title[:1].upper() not in string.ascii_uppercase:
                continue
        if sample_mode == "random" and rng.random() >= accept_prob:
            continue
        wc = _word_count(text, word_count_mode)
        if wc <= 0:
            continue
        out_texts.append(text)
        total += wc
    return out_texts, total

def _load_coco_category_letters(anno_dir: str) -> Dict[str, int]:
    paths = [
        os.path.join(anno_dir, "annotations/instances_train2014.json"),
        os.path.join(anno_dir, "annotations/instances_val2014.json"),
        os.path.join(anno_dir, "instances_train2014.json"),
        os.path.join(anno_dir, "instances_val2014.json"),
    ]
    counts = {ch: 0 for ch in string.ascii_uppercase}
    seen = set()
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            for key in ("name", "supercategory"):
                name = cat.get(key)
                if not name:
                    continue
                name = name.strip()
                if not name:
                    continue
                ch = name[0].upper()
                if ch in counts:
                    seen.add((key, name))
    for key, name in seen:
        counts[name[0].upper()] += 1
    return counts


def _load_wiki_articles(
    path: str,
) -> Dict[str, List[Tuple[str, int]]]:
    groups: Dict[str, List[Tuple[str, int]]] = {
        ch: [] for ch in string.ascii_uppercase
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            title = obj.get("title", "")
            text = obj.get("text", "")
            if not title or not text:
                continue
            ch = title[0].upper()
            if ch not in groups:
                continue
            wc = len(text.split())
            if wc <= 0:
                continue
            groups[ch].append((text, wc))
    return groups


def _sample_wiki_texts(
    groups: Dict[str, List[Tuple[str, int]]],
    letter_counts: Dict[str, int],
    target_words: int,
    seed: int,
) -> Tuple[List[str], Dict]:
    total_letters = sum(letter_counts.values())
    if total_letters == 0:
        raise ValueError("No COCO categories/supercategories found to build distribution.")

    random.seed(seed)
    per_letter_target: Dict[str, int] = {}
    for ch in string.ascii_uppercase:
        weight = letter_counts.get(ch, 0)
        per_letter_target[ch] = int(round(target_words * (weight / total_letters)))

    texts: List[str] = []
    stats = {
        "target_words": target_words,
        "per_letter_target": per_letter_target,
        "per_letter_actual": {ch: 0 for ch in string.ascii_uppercase},
        "per_letter_available": {ch: len(groups[ch]) for ch in string.ascii_uppercase},
    }

    for ch in string.ascii_uppercase:
        items = groups[ch]
        random.shuffle(items)
        need = per_letter_target[ch]
        got_words = 0
        for text, wc in items:
            if got_words >= need:
                break
            texts.append(text)
            got_words += wc
        stats["per_letter_actual"][ch] = got_words

    stats["actual_words"] = sum(stats["per_letter_actual"].values())
    return texts, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--num_merges", type=int, default=8000)
    ap.add_argument("--min_pair_freq", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", type=str, choices=["captions", "wiki"], default="captions")
    ap.add_argument("--articles_jsonl", type=str, default="./data/wiki_coco/articles.jsonl")
    ap.add_argument("--target_words", type=int, default=2_500_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--mix_captions_words", type=int, default=500_000)
    ap.add_argument("--mix_wiki_words", type=int, default=7_000_000)
    ap.add_argument("--word_count_mode", type=str, choices=["fast", "split"], default="fast")
    ap.add_argument("--mix_wiki_sample_mode", type=str, choices=["head", "random"], default="random")
    ap.add_argument("--wiki_total_words", type=int, default=0)
    ap.add_argument("--wiki_read_full", action="store_true")
    ap.add_argument("--wiki_workers", type=int, default=0)
    ap.add_argument("--wiki_chunk_lines", type=int, default=2000)
    ap.add_argument("--mix_cache_path", type=str, default="")
    ap.add_argument("--mix_use_cache", action="store_true")
    ap.add_argument("--heldout_words", type=int, default=50000)
    ap.add_argument("--heldout_seed", type=int, default=1337)
    args = ap.parse_args()

    run_dir = os.path.join(LOGDIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    info: Dict = {
        "vocab_size": None,
        "num_merges": args.num_merges,
        "min_pair_freq": args.min_pair_freq,
        "source": args.source,
    }

    if args.mix:
        ann_path = _find_captions_json(ANNO_DIR_DEFAULT)
        captions = load_captions(ann_path, limit=args.limit)
        if args.mix_use_cache and args.mix_cache_path and os.path.isfile(args.mix_cache_path):
            texts = []
            with open(args.mix_cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = _json_loads(line)
                    text = obj.get("text", "")
                    if text:
                        texts.append(text)
            info["mix_cache_path"] = args.mix_cache_path
            info["mix_used_cache"] = True
            info["mix"] = {"cached": True}
        else:
            cap_sample = _sample_captions_by_words(
                captions, args.mix_captions_words, args.seed, args.word_count_mode
            )
            total_words = args.wiki_total_words or _load_manifest_total_words(
                args.articles_jsonl
            )
            wiki_sample, wiki_stats = _sample_wiki_by_words(
                args.articles_jsonl,
                args.mix_wiki_words,
                args.seed + 1,
                args.word_count_mode,
                args.mix_wiki_sample_mode,
                total_words,
                stop_on_target=not args.wiki_read_full,
                filter_az=False,
                workers=args.wiki_workers,
                chunk_lines=args.wiki_chunk_lines,
            )
            texts = cap_sample + wiki_sample
            random.shuffle(texts)
            if args.mix_cache_path:
                with open(args.mix_cache_path, "w", encoding="utf-8") as f:
                    for t in texts:
                        f.write(json.dumps({"text": t}, ensure_ascii=True) + "\n")
            info["mix"] = {
                "captions_words_target": args.mix_captions_words,
                "wiki_words_target": args.mix_wiki_words,
                "captions_count": len(cap_sample),
                "wiki_count": len(wiki_sample),
                "wiki_stats": wiki_stats,
                "word_count_mode": args.word_count_mode,
                "wiki_sample_mode": args.mix_wiki_sample_mode,
                "wiki_total_words": total_words,
                "wiki_read_full": args.wiki_read_full,
                "wiki_workers": args.wiki_workers,
                "wiki_chunk_lines": args.wiki_chunk_lines,
            }
        info["articles_jsonl"] = args.articles_jsonl
        info["num_captions"] = len(captions)
    elif args.source == "captions":
        ann_path = _find_captions_json(ANNO_DIR_DEFAULT)
        captions = load_captions(ann_path, limit=args.limit)
        texts = captions
        info["num_captions"] = len(captions)
    else:
        letter_counts = _load_coco_category_letters(ANNO_DIR_DEFAULT)
        groups = _load_wiki_articles(args.articles_jsonl)
        texts, stats = _sample_wiki_texts(
            groups, letter_counts, target_words=args.target_words, seed=args.seed
        )
        info["articles_jsonl"] = args.articles_jsonl
        info["target_words"] = args.target_words
        info["sampling"] = stats

    if not texts:
        raise ValueError("No training texts found.")

    tokenizer = ByteBPETokenizer()
    tokenizer.train_bpe(
        texts, num_merges=args.num_merges, min_pair_freq=args.min_pair_freq
    )

    tok_path = os.path.join(run_dir, "tokenizer.pt")
    tokenizer.save(tok_path)

    heldout_info: Dict = {}
    if args.heldout_words > 0:
        heldout_texts: List[str] = []
        if args.mix:
            ratio = float(args.mix_captions_words) / float(
                args.mix_captions_words + args.mix_wiki_words
            )
            cap_target = int(round(args.heldout_words * ratio))
            wiki_target = max(0, args.heldout_words - cap_target)
            cap_held = _sample_captions_by_words(
                captions, cap_target, args.heldout_seed, args.word_count_mode
            )
            total_words = args.wiki_total_words or _load_manifest_total_words(
                args.articles_jsonl
            )
            heldout_mode = "random" if total_words > 0 else "head"
            wiki_held, wiki_stats = _sample_wiki_by_words(
                args.articles_jsonl,
                wiki_target,
                args.heldout_seed + 1,
                args.word_count_mode,
                heldout_mode,
                total_words,
                stop_on_target=True,
                filter_az=False,
                workers=0,
                chunk_lines=args.wiki_chunk_lines,
            )
            heldout_texts = cap_held + wiki_held
            heldout_info = {
                "heldout_words_target": args.heldout_words,
                "heldout_captions_target": cap_target,
                "heldout_wiki_target": wiki_target,
                "heldout_wiki_mode": heldout_mode,
                "heldout_wiki_stats": wiki_stats,
            }
        elif args.source == "captions":
            heldout_texts = _sample_captions_by_words(
                captions, args.heldout_words, args.heldout_seed, args.word_count_mode
            )
            heldout_info = {"heldout_words_target": args.heldout_words}
        else:
            total_words = args.wiki_total_words or _load_manifest_total_words(
                args.articles_jsonl
            )
            heldout_mode = "random" if total_words > 0 else "head"
            heldout_texts, wiki_stats = _sample_wiki_by_words(
                args.articles_jsonl,
                args.heldout_words,
                args.heldout_seed,
                args.word_count_mode,
                heldout_mode,
                total_words,
                stop_on_target=True,
                filter_az=True,
                workers=0,
                chunk_lines=args.wiki_chunk_lines,
            )
            heldout_info = {
                "heldout_words_target": args.heldout_words,
                "heldout_wiki_mode": heldout_mode,
                "heldout_wiki_stats": wiki_stats,
            }

        t_per_w, t_count, w_count = _estimate_tokens_per_word(
            tokenizer, heldout_texts, args.word_count_mode
        )
        heldout_info.update(
            {
                "heldout_words_actual": w_count,
                "heldout_tokens_actual": t_count,
                "tokens_per_word": t_per_w,
            }
        )

    info["vocab_size"] = tokenizer.vocab_size
    info["special_tokens"] = tokenizer.special_tokens
    info["special_tokens_immutable"] = True
    info["normalization"] = "NFKC+punct"
    info["max_numeric_token_len"] = tokenizer.max_numeric_token_len
    if heldout_info:
        info["heldout"] = heldout_info
    with open(os.path.join(run_dir, "tokenizer_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"vocab_size: {tokenizer.vocab_size}")
    if heldout_info.get("tokens_per_word") is not None:
        print(f"heldout tokens/word: {heldout_info['tokens_per_word']:.4f}")
    if texts:
        sample = texts[0]
        enc = tokenizer.encode(sample)
        dec = tokenizer.decode(enc)
        print(f"round_trip sample: {sample[:200]}{'...' if len(sample) > 200 else ''}")
        print(f"encoded length: {enc.numel()}")
        print(f"decoded: {dec}")
    else:
        print("round_trip sample: <none>")

    batch = texts[: min(4, len(texts))]
    if batch:
        input_ids, attention_mask = tokenizer(
            batch, max_len=args.max_len, return_attention_mask=True
        )
        print(
            f"batch shapes: input_ids={tuple(input_ids.shape)} "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    else:
        print("batch shapes: <none>")


if __name__ == "__main__":
    main()
