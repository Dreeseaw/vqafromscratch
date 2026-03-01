#!/bin/bash
set -e

RUN_ID="$1"
shift || true
EXTRA_ARGS=("$@")

docker run --rm -it --gpus all -v "$(pwd)":/app -w /app myrepo:gpu \
	python -m train.train_transformer "$RUN_ID" \
		--train_data data/pretraining/wikicoco256/train --val_data data/pretraining/wikicoco256/val --test_data data/pretraining/wikicoco256/test \
		--tokenizer logs/mix_bpe_16k/tokenizer.pt \
		--batch_size=64 --num_workers=1 --prefetch_factor=4 --bucket_width=32 \
		--activation_checkpointing --log_every=100 --run_probes=250 \
		--clip_grad=5.0 --lr=0.0001 --warmup_ratio=0.04 --weight_decay=0.05 \
		--row_max_norm_enable --row_max_norm_c=1.0 --layerscale --layerscale_init=0.01 --logit_softcap=0 --resid_max_norm=0 \
		--cap_attn_out_norm 1.5 --cap_mlp_out_norm 2.0 --cap_out_mode token \
		--enc_layers=4 --dec_layers=4 --ff_mult=2 --d_model=512 --n_heads=8 --probe_layers=0,1,2,3 \
		"${EXTRA_ARGS[@]}"
