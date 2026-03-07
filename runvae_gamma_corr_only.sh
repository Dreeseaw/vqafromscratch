#!/bin/bash
set -e

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
	echo "Usage: $0 <run_id> [checkpoint_step] [extra train args...]"
	exit 1
fi
shift || true

CKPT_STEP=""
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
	CKPT_STEP="$1"
	shift || true
fi
EXTRA_ARGS=("$@")

mkdir -pv "logs/$RUN_ID"
if [[ -n "${CKPT_STEP}" ]]; then
	cat models/vae.py train/train.py > "logs/$RUN_ID/code_$CKPT_STEP.py"
	docker run --rm -it --gpus all --ipc=host \
		-e PYTORCH_ENABLE_MPS_FALLBACK=1 \
		-v "$(pwd)":/app -w /app myrepo:gpu \
		python -m train.train "$RUN_ID" "$CKPT_STEP" \
			--preset gamma_corr_only \
			"${EXTRA_ARGS[@]}"
else
	cat models/vae.py train/train.py > "logs/$RUN_ID/code.py"
	docker run --rm -it --gpus all --ipc=host \
		-e PYTORCH_ENABLE_MPS_FALLBACK=1 \
		-v "$(pwd)":/app -w /app myrepo:gpu \
		python -m train.train "$RUN_ID" \
			--preset gamma_corr_only \
			"${EXTRA_ARGS[@]}"
fi
