#!/usr/bin/env python3
"""
sister script to distill_qa_ollama.py

pretokenizes the data akin to the pretokenization script and appends it in a already-created pretokenization dataset
"""
import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert distill raw.jsonl into plain QA training text.")
    ap.add_argument("--in_raw", required=True, help="Path to distillation raw.jsonl")
    ap.add_argument("--out_path", required=True, help="Output text file path")
    ap.add_argument("--include_context", action="store_true", help="Include context before question/answer.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    in_path = Path(args.in_raw)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("question")
            a = obj.get("answer")
            c = obj.get("context")
            if not isinstance(q, str) or not isinstance(a, str):
                continue
            if args.include_context and isinstance(c, str):
                text = f"Context: {c}\nQuestion: {q}\nAnswer: {a}\n\n"
            else:
                text = f"Question: {q}\nAnswer: {a}\n\n"
            fout.write(text)
            written += 1

    print(f"Wrote {written} examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
