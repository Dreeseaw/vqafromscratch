#!/usr/bin/env python3
"""
Convert distillation raw QA jsonl into pretokenize_corpus-ready JSONL docs.
"""
import argparse
import json
from pathlib import Path
from typing import Optional


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert distill raw.jsonl into pretokenize_corpus input JSONL."
    )
    ap.add_argument("--in_raw", required=True, help="Path to distillation raw.jsonl")
    ap.add_argument("--out_path", required=True, help="Output JSONL path (each row has a 'text' field).")
    return ap


def _format_qa_text(context: Optional[str], question: str, answer: str) -> str:
    ctx = (context or "").strip()
    q = question.strip()
    a = answer.strip()
    return f"{ctx}\nQuestion:{q}\nAnswer:{a}"


def main() -> int:
    args = build_arg_parser().parse_args()
    in_path = Path(args.in_raw)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin, start=1):
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
            text = _format_qa_text(c if isinstance(c, str) else "", q, a)
            out_row = {
                "id": obj.get("id", f"distill:{line_idx}"),
                "text": text,
                "source": "distill_qa",
                "word_count": len(text.split()),
            }
            fout.write(json.dumps(out_row, ensure_ascii=True) + "\n")
            written += 1

    print(f"Wrote {written} examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
