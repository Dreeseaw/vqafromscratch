#!/bin/bash
set -e

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
	echo "Usage: $0 <run_id> [extra distill args...]"
	exit 1
fi
shift || true
EXTRA_ARGS=("$@")
OLLAMA_URL="${OLLAMA_URL:-http://host.docker.internal:11434/api/generate}"

docker run --rm -it --gpus all --add-host=host.docker.internal:host-gateway -e OLLAMA_URL="${OLLAMA_URL}" -v "$(pwd)":/app -w /app myrepo:gpu \
	python3 -m scripts.distill_qa_ollama \
		--in_path data/wiki_coco/articles.jsonl \
		--out_dir "data/distill/${RUN_ID}" \
		--num_examples 200000 \
		--workers 8 \
		--answer_max_words 12 \
		--tokenizer logs/mix_bpe_16k/tokenizer.pt \
		--max_seq_len 256 \
		--special_tokens_reserved 2 \
		"${EXTRA_ARGS[@]}"
