docker run --rm -it --gpus all -v "$(pwd)":/app -w /app myrepo:gpu \
	python3 -m scripts.distill_to_pretokenize \
    		--in_raw ./data/distill/v3/raw.jsonl \
    		--out_path ./data/pretraining/distill_qa.jsonl 
docker run --rm -it --gpus all -v "$(pwd)":/app -w /app myrepo:gpu \
    python3 -m scripts.pretokenize_corpus \
    	--input ./data/pretraining/distill_qa.jsonl \
    	--out-dir ./data/pretraining/distill256_cleaned \
    	--tokenizer ./logs/mix_bpe_16k/tokenizer.pt \
    	--clean_wikipedia 0 \
    	--min_chars_after_clean 1
