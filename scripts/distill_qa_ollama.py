#!/usr/bin/env python3
"""
This script samples short context paragraphs from the scraped docs, asks Ollama for
  strict JSON extractive QA pairs ({"q","a"} only), and then deterministically verifies
  each answer is actually in the context by computing a_start/a_end locally. If an answer
  span is missing, it does one repair call to force a verbatim substring answer; accepted
  rows go to raw.jsonl (with indices/meta), rejected rows go to rejected.jsonl with
  specific reason buckets, plus stats.json and manifest.json.

Example:
python scripts/distill_qa_ollama.py --in_path data/wiki_clean.jsonl --out_dir data/distill/run001 --num_examples 200000 --workers 8 --answer_max_words 12
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from models.bpe_tokenizer import ByteBPETokenizer

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TEXT_KEYS = ("text", "context", "content", "body", "extract")
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"
BANNED_PHRASES = (
    "see also",
    "references",
    "external links",
    "list of",
)

SYSTEM_PROMPT = "You output only strict JSON. No prose."
STATUS_EVERY_EVAL = 100


@dataclass(frozen=True)
class ContextSample:
    context: str
    source_doc_id: str
    source_offset: int


def _word_count(text: str) -> int:
    return len([x for x in text.strip().split() if x])


def _raw_prompt_word_count(context: str, question: str, answer: str) -> int:
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}\n"
    return _word_count(prompt)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _iter_input_files(in_path: Path) -> List[Path]:
    if in_path.is_file():
        return [in_path]
    if not in_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {in_path}")
    out: List[Path] = []
    for root, _, files in os.walk(in_path):
        for name in files:
            p = Path(root) / name
            lower = p.name.lower()
            if lower.endswith(".jsonl") or lower.endswith(".txt") or lower.endswith(".md") or lower.endswith(".text") or "." not in p.name:
                out.append(p)
    out.sort(key=lambda x: str(x))
    return out


def _safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _extract_text_and_doc_id(obj: Dict[str, Any], fallback_doc_id: str) -> Optional[Tuple[str, str]]:
    text = None
    for k in TEXT_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            text = v
            break
    if text is None:
        return None
    hint = obj.get("doc_id")
    if hint is None:
        hint = obj.get("id")
    if hint is None:
        hint = obj.get("title")
    if isinstance(hint, (str, int)):
        doc_id = f"{fallback_doc_id}::{hint}"
    else:
        doc_id = fallback_doc_id
    return text, doc_id


def _iter_docs(file_path: Path) -> Iterable[Tuple[str, str]]:
    lower = file_path.name.lower()
    if lower.endswith(".jsonl"):
        with file_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                fallback = f"{file_path}::line:{line_idx}"
                extracted = _extract_text_and_doc_id(obj, fallback)
                if extracted is None:
                    continue
                yield extracted[1], extracted[0]
        return

    text = _safe_read_text(file_path)
    if text and text.strip():
        yield str(file_path), text


def _split_paragraphs_with_offsets(text: str) -> List[Tuple[str, int]]:
    pattern = re.compile(r"(.*?)(?:\n\s*\n+|$)", re.S)
    out: List[Tuple[str, int]] = []
    for m in pattern.finditer(text):
        raw = m.group(1)
        if not raw:
            continue
        left = len(raw) - len(raw.lstrip())
        clean = raw.strip()
        if not clean:
            continue
        start = m.start(1) + left
        out.append((clean, start))
    return out


def _contains_banned(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in BANNED_PHRASES)


def _is_quotes_or_punct_only(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    t = re.sub(r"[\"'`“”‘’]+", "", t).strip()
    if not t:
        return True
    return re.fullmatch(r"[^\w]+", t) is not None


def find_span(context: str, answer: str) -> Optional[Tuple[int, int]]:
    # Fast path: exact substring.
    start = context.find(answer)
    if start >= 0:
        return start, start + len(answer)

    # Fallback: whitespace-normalized match with back-mapping to raw indices.
    ans_n = normalize_ws(answer)
    if not ans_n:
        return None

    norm_chars: List[str] = []
    norm_to_raw: List[int] = []
    saw_non_ws = False
    pending_space = False

    for idx, ch in enumerate(context):
        if ch.isspace():
            if saw_non_ws:
                pending_space = True
            continue
        if pending_space and norm_chars:
            norm_chars.append(" ")
            norm_to_raw.append(idx)
            pending_space = False
        norm_chars.append(ch)
        norm_to_raw.append(idx)
        saw_non_ws = True

    ctx_n = "".join(norm_chars)
    if not ctx_n:
        return None
    pos = ctx_n.find(ans_n)
    if pos < 0:
        return None
    end_pos = pos + len(ans_n) - 1
    if pos >= len(norm_to_raw) or end_pos >= len(norm_to_raw):
        return None
    raw_start = norm_to_raw[pos]
    raw_end = norm_to_raw[end_pos] + 1
    if raw_start >= raw_end:
        return None
    return raw_start, raw_end


def _token_count_text(tokenizer: ByteBPETokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    return int(ids.numel())


def _split_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        start = int(m.start())
        end = int(m.end())
        chunk = text[start:end]
        if not chunk.strip():
            continue
        out.append((chunk, start, end))
    if not out and text.strip():
        out.append((text, 0, len(text)))
    return out


def _context_with_sentence_removed(context: str, sent_start: int, sent_end: int) -> str:
    merged = (context[:sent_start] + context[sent_end:]).strip()
    return re.sub(r"\n{3,}", "\n\n", merged)


def _fit_context_to_budget(
    tokenizer: ByteBPETokenizer,
    context: str,
    question: str,
    answer: str,
    qa_token_budget: int,
) -> Optional[str]:
    cur = context.strip()
    if not cur:
        return None

    for _ in range(64):
        c_tok = _token_count_text(tokenizer, cur)
        q_tok = _token_count_text(tokenizer, question)
        a_tok = _token_count_text(tokenizer, answer)
        if (c_tok + q_tok + a_tok) < int(qa_token_budget):
            return cur

        span = find_span(cur, answer)
        if span is None:
            return None
        a_start, a_end = span
        sents = _split_sentences_with_offsets(cur)
        if len(sents) <= 1:
            return None

        first = sents[0]
        last = sents[-1]
        first_contains_answer = (a_start < first[2]) and (a_end > first[1])
        last_contains_answer = (a_start < last[2]) and (a_end > last[1])

        if not last_contains_answer:
            _, rm_start, rm_end = last
        elif not first_contains_answer:
            _, rm_start, rm_end = first
        else:
            return None

        nxt = _context_with_sentence_removed(cur, rm_start, rm_end)
        if not nxt or nxt == cur:
            return None
        cur = nxt

    return None


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if not lines:
        return s
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _build_generation_prompt(context: str, max_q_per_context: int, answer_max_words: int) -> str:
    example = {
        "qas": [
            {"q": "What is the capital of France?", "a": "Paris"}
        ]
    }
    return (
        "CONTEXT:\n"
        f"{context}\n\n"
        "TASK:\n"
        f"Produce up to {max_q_per_context} QA pairs where each answer is copied verbatim from CONTEXT.\n"
        "Output strict JSON with top-level object field \"qas\" (a list).\n"
        "Each QA object must be exactly: {\"q\": \"...\", \"a\": \"...\"}\n"
        "Constraints:\n"
        f"- a length: 1..{answer_max_words} words\n"
        "- q length <= 20 words\n"
        "- ban any q/a that includes: See also, References, External links, List of\n"
        "- no quotes-only answers\n"
        "- no punctuation-only answers\n"
        "- Do not include character indices. Only provide q and a.\n"
        "- If there are no good extractive pairs, return {\"qas\": []}\n"
        "Return JSON only. No markdown. No prose.\n\n"
        "Example:\n"
        "CONTEXT: Paris is the capital of France.\n"
        f"JSON: {json.dumps(example, ensure_ascii=False)}"
    )


def _build_repair_prompt(context: str, question: str, bad_answer: str, answer_max_words: int) -> str:
    return (
        "Return ONLY a JSON object: {\"a\":\"...\"}.\n"
        "a must be an exact substring of CONTEXT.\n"
        f"a length must be 1..{answer_max_words} words.\n"
        "No headings/boilerplate: See also, References, External links, List of.\n"
        "No punctuation-only answers.\n"
        "No empty answers.\n"
        "No markdown, no backticks, no extra keys.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"BAD_ANSWER:\n{bad_answer}\n"
    )


def _ollama_generate(
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    top_p: float,
    seed: int,
    timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
    }
    req = urlrequest.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    out = obj.get("response")
    if not isinstance(out, str):
        raise ValueError("Ollama response missing string field 'response'")
    return out


def _parse_qas_json(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON is not an object")
    qas = obj.get("qas")
    if not isinstance(qas, list):
        raise ValueError("Missing list field 'qas'")
    return obj


def _parse_repair_answer_json(text: str) -> str:
    cleaned = _strip_code_fences(text)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("Repair top-level JSON is not an object")
    a = obj.get("a")
    if not isinstance(a, str):
        raise ValueError("Repair JSON missing string field 'a'")
    return a


def _validate_qa(
    qa: Any,
    answer_max_words: int,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    if not isinstance(qa, dict):
        return None, "schema_invalid"
    q = qa.get("q")
    a = qa.get("a")
    if not isinstance(q, str) or not q.strip():
        return None, "schema_invalid"
    if not isinstance(a, str) or not a.strip():
        return None, "a_empty_or_punct"
    q = q.strip()
    a = a.strip()
    if _contains_banned(q) or _contains_banned(a):
        return None, "banned_phrase"
    if _word_count(q) > 20:
        return None, "q_too_long"
    if _is_quotes_or_punct_only(a):
        return None, "a_empty_or_punct"
    aw = _word_count(a)
    if aw < 1:
        return None, "a_empty_or_punct"
    if aw > answer_max_words:
        return None, "schema_invalid"
    return {"question": q, "answer": a}, None


async def _process_context(
    sample: ContextSample,
    model: str,
    max_q_per_context: int,
    answer_max_words: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout_s: int,
    run_id: str,
    tokenizer: ByteBPETokenizer,
    qa_token_budget: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    reason_counts: Counter = Counter()
    user_prompt = _build_generation_prompt(sample.context, max_q_per_context, answer_max_words)
    try:
        first = await asyncio.to_thread(
            _ollama_generate,
            model,
            SYSTEM_PROMPT,
            user_prompt,
            temperature,
            top_p,
            seed,
            timeout_s,
        )
    except (TimeoutError, urlerror.URLError, urlerror.HTTPError, json.JSONDecodeError, ValueError):
        reason = "ollama_timeout_or_error"
        reason_counts[reason] += 1
        return [], [
            {
                "reason": reason,
                "source_doc_id": sample.source_doc_id,
                "source_offset": sample.source_offset,
                "context": sample.context,
                "raw_response": "",
            }
        ], reason_counts

    try:
        parsed = _parse_qas_json(first)
    except (json.JSONDecodeError, ValueError):
        reason = "json_parse_fail"
        reason_counts[reason] += 1
        return [], [
            {
                "reason": reason,
                "source_doc_id": sample.source_doc_id,
                "source_offset": sample.source_offset,
                "context": sample.context,
                "raw_response": _strip_code_fences(first)[:500],
            }
        ], reason_counts

    qas = parsed.get("qas", []) if isinstance(parsed, dict) else []
    if not isinstance(qas, list):
        reason = "schema_invalid"
        reason_counts[reason] += 1
        return [], [
            {
                "reason": reason,
                "source_doc_id": sample.source_doc_id,
                "source_offset": sample.source_offset,
                "context": sample.context,
                "raw_response": _strip_code_fences(first)[:500],
            }
        ], reason_counts

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for qa in qas[:max_q_per_context]:
        valid, reason = _validate_qa(qa, answer_max_words)
        if reason is not None:
            reason_counts[reason] += 1
            rejected.append(
                {
                    "reason": reason,
                    "source_doc_id": sample.source_doc_id,
                    "source_offset": sample.source_offset,
                    "context": sample.context,
                    "qa_candidate": qa,
                    "raw_response": _strip_code_fences(first)[:500],
                }
            )
            continue
        assert valid is not None
        span = find_span(sample.context, valid["answer"])
        final_answer = valid["answer"]
        if span is None:
            repair_prompt = _build_repair_prompt(
                sample.context,
                valid["question"],
                valid["answer"],
                answer_max_words,
            )
            try:
                repaired = await asyncio.to_thread(
                    _ollama_generate,
                    model,
                    SYSTEM_PROMPT,
                    repair_prompt,
                    temperature,
                    top_p,
                    seed + 1009,
                    timeout_s,
                )
            except (TimeoutError, urlerror.URLError, urlerror.HTTPError, json.JSONDecodeError, ValueError):
                reason = "ollama_timeout_or_error"
                reason_counts[reason] += 1
                rejected.append(
                    {
                        "reason": reason,
                        "source_doc_id": sample.source_doc_id,
                        "source_offset": sample.source_offset,
                        "context": sample.context,
                        "qa_candidate": qa,
                        "raw_response": _strip_code_fences(first)[:500],
                    }
                )
                continue

            try:
                repaired_a = _parse_repair_answer_json(repaired).strip()
            except (json.JSONDecodeError, ValueError):
                reason = "json_parse_fail"
                reason_counts[reason] += 1
                rejected.append(
                    {
                        "reason": reason,
                        "source_doc_id": sample.source_doc_id,
                        "source_offset": sample.source_offset,
                        "context": sample.context,
                        "qa_candidate": qa,
                        "raw_response": _strip_code_fences(repaired)[:500],
                    }
                )
                continue

            if _contains_banned(repaired_a):
                reason = "banned_phrase"
                reason_counts[reason] += 1
                rejected.append(
                    {
                        "reason": reason,
                        "source_doc_id": sample.source_doc_id,
                        "source_offset": sample.source_offset,
                        "context": sample.context,
                        "qa_candidate": qa,
                        "raw_response": _strip_code_fences(repaired)[:500],
                    }
                )
                continue
            if _is_quotes_or_punct_only(repaired_a) or _word_count(repaired_a) < 1:
                reason = "a_empty_or_punct"
                reason_counts[reason] += 1
                rejected.append(
                    {
                        "reason": reason,
                        "source_doc_id": sample.source_doc_id,
                        "source_offset": sample.source_offset,
                        "context": sample.context,
                        "qa_candidate": qa,
                        "raw_response": _strip_code_fences(repaired)[:500],
                    }
                )
                continue
            if _word_count(repaired_a) > answer_max_words:
                reason = "schema_invalid"
                reason_counts[reason] += 1
                rejected.append(
                    {
                        "reason": reason,
                        "source_doc_id": sample.source_doc_id,
                        "source_offset": sample.source_offset,
                        "context": sample.context,
                        "qa_candidate": qa,
                        "raw_response": _strip_code_fences(repaired)[:500],
                    }
                )
                continue
            span = find_span(sample.context, repaired_a)
            if span is None:
                reason = "answer_not_in_context_after_repair"
                reason_counts[reason] += 1
                rejected.append(
                    {
                        "reason": reason,
                        "source_doc_id": sample.source_doc_id,
                        "source_offset": sample.source_offset,
                        "context": sample.context,
                        "qa_candidate": qa,
                        "raw_response": _strip_code_fences(repaired)[:500],
                    }
                )
                continue
            final_answer = repaired_a

        assert span is not None
        orig_a_start, orig_a_end = span
        final_answer = sample.context[orig_a_start:orig_a_end]
        fitted_context = _fit_context_to_budget(
            tokenizer=tokenizer,
            context=sample.context,
            question=valid["question"],
            answer=final_answer,
            qa_token_budget=qa_token_budget,
        )
        if fitted_context is None:
            reason = "qa_triplet_over_token_budget"
            reason_counts[reason] += 1
            rejected.append(
                {
                    "reason": reason,
                    "source_doc_id": sample.source_doc_id,
                    "source_offset": sample.source_offset,
                    "context": sample.context,
                    "qa_candidate": qa,
                    "qa_token_budget": int(qa_token_budget),
                }
            )
            continue
        fitted_span = find_span(fitted_context, final_answer)
        if fitted_span is None:
            reason = "answer_missing_after_context_trim"
            reason_counts[reason] += 1
            rejected.append(
                {
                    "reason": reason,
                    "source_doc_id": sample.source_doc_id,
                    "source_offset": sample.source_offset,
                    "context": sample.context,
                    "trimmed_context": fitted_context,
                    "qa_candidate": qa,
                }
            )
            continue
        a_start, a_end = fitted_span
        item = {
            "context": fitted_context,
            "question": valid["question"],
            "answer": final_answer,
            "a_start": a_start,
            "a_end": a_end,
            "source_doc_id": sample.source_doc_id,
            "source_offset": sample.source_offset,
            "meta": {"model": model, "run_id": run_id},
        }
        accepted.append(item)

    return accepted, rejected, reason_counts


def _git_hash_or_none() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _load_context_pool(
    in_path: Path,
    context_max_chars: int,
    rng: random.Random,
) -> Tuple[List[ContextSample], Dict[str, int]]:
    files = _iter_input_files(in_path)
    docs_seen = 0
    docs_with_eligible = 0
    contexts: List[ContextSample] = []

    for file_path in files:
        for doc_id, text in _iter_docs(file_path):
            docs_seen += 1
            paras = _split_paragraphs_with_offsets(text)
            eligible = [
                (para, offset)
                for para, offset in paras
                if 300 <= len(para) <= context_max_chars
            ]
            if not eligible:
                continue
            docs_with_eligible += 1
            picked_para, picked_offset = rng.choice(eligible)
            contexts.append(
                ContextSample(
                    context=picked_para,
                    source_doc_id=doc_id,
                    source_offset=picked_offset,
                )
            )

    info = {
        "files_seen": len(files),
        "docs_seen": docs_seen,
        "docs_with_eligible_paragraphs": docs_with_eligible,
        "context_pool_size": len(contexts),
    }
    return contexts, info


async def _run(args: argparse.Namespace) -> int:
    run_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    rej_path = out_dir / "rejected.jsonl"
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    q_prefix_tokens = _token_count_text(tokenizer, "\nQuestion:")
    a_prefix_tokens = _token_count_text(tokenizer, "\nAnswer:")
    qa_token_budget = int(args.max_seq_len) - int(args.special_tokens_reserved) - q_prefix_tokens - a_prefix_tokens
    if qa_token_budget <= 0:
        raise ValueError(
            "Invalid QA token budget. Increase --max_seq_len or reduce --special_tokens_reserved."
        )

    rng = random.Random(args.seed)
    contexts, input_info = _load_context_pool(Path(args.in_path), args.context_max_chars, rng)
    if not contexts:
        print("No eligible contexts found after paragraph filtering.", file=sys.stderr)
        return 2

    calls_needed = int(math.ceil(args.num_examples / max(1, args.max_q_per_context)))
    max_calls = max(calls_needed * 6, calls_needed + args.workers * 4)
    base_indices = list(range(len(contexts)))
    call_plan: List[int] = []
    while len(call_plan) < max_calls:
        block = base_indices[:]
        rng.shuffle(block)
        call_plan.extend(block)
    call_plan = call_plan[:max_calls]

    lock = asyncio.Lock()
    shared: Dict[str, Any] = {
        "call_cursor": 0,
        "calls_attempted": 0,
        "calls_completed": 0,
        "accepted": 0,
        "rejected": 0,
        "raw_words_found": 0,
        "last_status_bucket": 0,
        "reasons": Counter(),
    }

    raw_f = raw_path.open("w", encoding="utf-8")
    rej_f = rej_path.open("w", encoding="utf-8")

    async def worker(worker_id: int) -> None:
        del worker_id
        while True:
            async with lock:
                if shared["accepted"] >= args.num_examples:
                    return
                if shared["call_cursor"] >= len(call_plan):
                    return
                call_idx = shared["call_cursor"]
                shared["call_cursor"] += 1
                shared["calls_attempted"] += 1
            ctx = contexts[call_plan[call_idx]]
            seed = int(args.seed) + int(call_idx) * 13
            accepted, rejected, reason_counts = await _process_context(
                sample=ctx,
                model=args.model,
                max_q_per_context=args.max_q_per_context,
                answer_max_words=args.answer_max_words,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
                timeout_s=args.timeout_s,
                run_id=Path(args.out_dir).name,
                tokenizer=tokenizer,
                qa_token_budget=qa_token_budget,
            )
            async with lock:
                shared["calls_completed"] += 1
                shared["reasons"].update(reason_counts)
                for item in rejected:
                    rej_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    shared["rejected"] += 1
                for item in accepted:
                    if shared["accepted"] >= args.num_examples:
                        break
                    raw_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    shared["accepted"] += 1
                    shared["raw_words_found"] += _raw_prompt_word_count(
                        context=item["context"],
                        question=item["question"],
                        answer=item["answer"],
                    )
                evaluated = int(shared["accepted"]) + int(shared["rejected"])
                bucket = evaluated // STATUS_EVERY_EVAL
                if bucket > int(shared["last_status_bucket"]):
                    shared["last_status_bucket"] = bucket
                    print(
                        "status "
                        f"evaluated_tuples={evaluated} "
                        f"accepted={shared['accepted']} "
                        f"rejected={shared['rejected']} "
                        f"raw_words_found={shared['raw_words_found']} "
                        f"calls_completed={shared['calls_completed']}"
                    )

    workers = [asyncio.create_task(worker(i)) for i in range(args.workers)]
    await asyncio.gather(*workers)
    raw_f.close()
    rej_f.close()

    elapsed = time.time() - run_start
    accepted = int(shared["accepted"])
    rejected = int(shared["rejected"])
    total_labeled = accepted + rejected
    acceptance_rate = (float(accepted) / float(total_labeled)) if total_labeled > 0 else 0.0

    stats = {
        "target_num_examples": args.num_examples,
        "accepted_examples": accepted,
        "rejected_examples": rejected,
        "acceptance_rate": acceptance_rate,
        "calls_attempted": int(shared["calls_attempted"]),
        "calls_completed": int(shared["calls_completed"]),
        "raw_words_found": int(shared["raw_words_found"]),
        "duration_s": elapsed,
        "input": input_info,
        "top_rejection_reasons": [
            {"reason": r, "count": c}
            for r, c in shared["reasons"].most_common(20)
        ],
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "settings": {
            "in_path": args.in_path,
            "num_examples": args.num_examples,
            "max_q_per_context": args.max_q_per_context,
            "context_max_chars": args.context_max_chars,
            "answer_max_words": args.answer_max_words,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "workers": args.workers,
            "timeout_s": args.timeout_s,
            "ollama_url": OLLAMA_URL,
            "tokenizer": args.tokenizer,
            "max_seq_len": int(args.max_seq_len),
            "special_tokens_reserved": int(args.special_tokens_reserved),
            "qa_token_budget": int(qa_token_budget),
            "question_prefix_tokens": int(q_prefix_tokens),
            "answer_prefix_tokens": int(a_prefix_tokens),
        },
        "git_hash": _git_hash_or_none(),
        "outputs": {
            "raw_jsonl": str(raw_path),
            "rejected_jsonl": str(rej_path),
            "stats_json": str(out_dir / "stats.json"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if accepted < args.num_examples:
        print(
            (
                f"Warning: generated {accepted} < requested {args.num_examples} accepted examples "
                f"after {shared['calls_attempted']} calls. Increase input size or max_calls factor."
            ),
            file=sys.stderr,
        )
        return 1
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate extractive QA distillation JSONL via Ollama with strict validation."
    )
    ap.add_argument("--in_path", required=True, help="Input file or directory (.jsonl and/or text docs).")
    ap.add_argument("--out_dir", required=True, help="Output directory, e.g. data/distill/run001.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    ap.add_argument("--num_examples", type=int, required=True, help="Total accepted QA pairs to generate.")
    ap.add_argument("--max_q_per_context", type=int, default=2)
    ap.add_argument("--context_max_chars", type=int, default=1800)
    ap.add_argument("--answer_max_words", type=int, default=12, help="Answer cap by whitespace words.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout_s", type=int, default=60)
    ap.add_argument("--tokenizer", type=str, required=True, help="Tokenizer .pt used for token-budget checks.")
    ap.add_argument("--max_seq_len", type=int, default=256, help="Sequence length ceiling for QA budget.")
    ap.add_argument(
        "--special_tokens_reserved",
        type=int,
        default=2,
        help="Reserved tokens (e.g., BOS/EOS) subtracted from max_seq_len.",
    )
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num_examples must be > 0")
    if args.max_q_per_context <= 0:
        raise ValueError("--max_q_per_context must be > 0")
    if args.context_max_chars < 300:
        raise ValueError("--context_max_chars must be >= 300")
    if args.answer_max_words <= 0:
        raise ValueError("--answer_max_words must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.max_seq_len <= 0:
        raise ValueError("--max_seq_len must be > 0")
    if args.special_tokens_reserved < 0:
        raise ValueError("--special_tokens_reserved must be >= 0")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
