#!/bin/bash
set -e

RUN_ID="$1"
shift || true
EXTRA_ARGS=("$@")

docker run --rm -it --gpus all -v "$(pwd)":/app -w /app myrepo:gpu \
	python -m train.train_transformer "$RUN_ID" \
		--train_data data/pretraining/wikicoco256/train --val_data data/pretraining/wikicoco256/val --test_data data/pretraining/wikicoco256/test \
		--tokenizer logs/mix_bpe_16k/tokenizer.pt \
		"${EXTRA_ARGS[@]}"
