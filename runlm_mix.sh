#!/bin/bash
set -e

RUN_ID="$1"
if [[ -z "${RUN_ID}" ]]; then
	echo "Usage: $0 <run_id> [extra train args...]"
	exit 1
fi
shift || true
EXTRA_ARGS=("$@")

docker run --rm -it --gpus all -v "$(pwd)":/app -w /app myrepo:gpu \
	python -m train.train_transformer "$RUN_ID" \
		--train_data data/pretraining/wikicoco256_cleaned \
    		--train_bucket_wiki data/pretraining/wiki256_cleaned/train \
    		--train_bucket_coco data/pretraining/coco256_cleaned/train \
    		--train_bucket_distill data/pretraining/distill256_cleaned/train \
    		--mix_schedule configs/mix_schedule1.json \
    		--val_data data/pretraining/wikicoco256_cleaned/val \
    		--test_data data/pretraining/wikicoco256_cleaned/test \
		--tokenizer logs/mix_bpe_16k/tokenizer.pt \
		--epochs=100 --warmup_ratio=0.004 \
		--enc_layers=6 --dec_layers=6 --ff_mult=2 --d_model=384 --n_heads=6 \
		--probe_layers=0,1,2,3,4,5 --no_activation_checkpointing \
		--num_workers=8 --persistent_workers --prefetch_factor 8 \
		--run_probes=1000 --probe_after_log_only --eval_every_steps=5000 \
		"${EXTRA_ARGS[@]}"
