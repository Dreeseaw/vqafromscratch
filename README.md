VQA From Scratch
===================

Grabbing MSCoCo (VAE training, VQA visual component)
----------------------------------------------------
```bash
> mkdir Images && cd Images
> curl -OJL images.cocodataset.org/zips/train2014.zip
> curl -OJL images.cocodataset.org/zips/val2014.zip
> curl -OJL images.cocodataset.org/zips/test2015.zip
> unzip train2014.zip
> unzip val2014.zip
> unzip test2015.zip
> rm *.zip

(repeat similar process with Annotations)
```

Training
--------
```bash
(one time)
> pyenv virtualenv 3.10.14 vqa
> pyenv activate vqa
> python3 -m pip install requirements.txt

(each working session)
> pyenv activate vqa

> ./run.sh <run_id>
> ./run.sh <run_id> (<checkpoint step to begin from>)
```

Running loss logging web app
----------------------------
To visualize the training process a bit better, codex wrote a nice little
bun web app for us to track experiments in both real time and reload old ones.

```bash
> bun run tracker/vm/vmtrackerapp.ts -f logs/<run_id> -p 3000
```
and navigate to `localhost:3000` in your browser. Multiple instances can be run for tab-by-tab comparisons.

Auto-Research Progress Tracker
------------------------------
Minimal Bun + HTML/TS/CSS dashboard that auto-discovers:
- `docs/mm_bridge_diagnostics/*.md`
- sweep timelines under `logs/*/timeline.log`
- run-level `overall_accuracy` from `logs/<run_id>/logfile*.txt`

```bash
bun run tracker/research/researchtrackerapp.ts -p 4090
```

Open `http://localhost:4090`.
For markdown-only view in a separate tab:
- `http://localhost:4090/doc?file=<doc_name>.md`


Probing
-------
Linear probes on mu are used to test downstream task efficiency. Multiple probes may be run in parallel and share the same batch, making them almost 2x as fast when running 3 in parallel, relative to 3 sequential runs.

```bash
> python3 -m evals.probe --ckpt logs/sl_d2_b01/step_10001.tar --use_mu
> python3 -m evals.probe --ckpts logs/model1/step_10001.tar logs/model2/step_10001.tar --use_mu --multi_mode=lockstep
```

Multimodal Bridge Diagnostics
-----------------------------
Docker-first checkpoint diagnostics for the frozen-bridge VQA setup:
- image perturbation sensitivity (`clean`, `shuffle`, `zero`, `noise`, `fixed_image`)
- accuracy deltas vs clean
- prediction agreement vs clean
- visual-prefix geometry stats

```bash
./run_mm_diag.sh mmdiag_example \
  --checkpoint logs/mmbr_basesweep_on_high/step_3466.tar \
  --max_batches 80 \
  --stats_batches 40 \
  --batch_size 256 \
  --modes clean,shuffle,zero,noise,fixed_image \
  --noise_std 0.2
```

Outputs are written to:
- `logs/<diag_run_id>/diag_report.json`
- `logs/<diag_run_id>/diag_report.md`

Prefix Calibration Sweep (MM Training)
--------------------------------------
Shortened iterative training sweep for calibrated bridge interface:

```bash
./scripts/launch_prefix_calib_sweep.sh
```

Key new MM flags:
- `--prefix_calibration`
- `--prefix_calib_layernorm`
- `--prefix_calib_bias`
- `--prefix_calib_gate_init`
- `--prefix_norm_target_ratio`
- `--prefix_norm_reg_weight`
- `--prefix_batchvar_reg_weight`

Next-Gen Bridge Sweep (Night Plan)
----------------------------------
Architecture-focused sequential sweep (Docker via `runmm.sh`) covering:
- learned-query cross-attention reducer
- spatial mixer before reduction
- hybrid constant + image prefix
- perceiver-style resampler
- q-former-lite bridge

Launch:
```bash
./scripts/launch_night_bridge_sweep_v1.sh
```

Common bridge flags:
- `--bridge_type {mlp,learned_tokens,learned_query,perceiver_resampler,qformer_lite,hybrid_const_image}`
- `--bridge_num_heads`
- `--bridge_query_depth`
- `--bridge_refine_layers`
- `--bridge_pre_mixer_type {none,self_attn,conv1d}`
- `--bridge_pre_mixer_layers`
- `--bridge_hybrid_alpha_mode {scalar,token}`
- `--bridge_hybrid_alpha_init`
- `--bridge_hybrid_image_bridge_type {mlp,learned_query,perceiver_resampler,qformer_lite}`

Throughput knobs (night sweeps):
- `--precision bf16` (recommended on modern NVIDIA GPUs)
- `--batch_size`, `--grad_accum_steps`
- `--num_workers`, `--prefetch_factor`
- `--vision_device {auto,cpu,cuda,mps}` (CPU vision is supported but usually slower end-to-end)


Create mp4 of step_nnn.png's 
----------------------------
This was cooler when my goal was focused on pretty reconstructions.

```bash
> cd logs/<run_id>/
> ls step_*.png | sort -V | sed "s/^/file '/; s/$/'/" > frames.txt && \
ffmpeg -y -r 30 -f concat -safe 0 -i frames.txt \
  -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4
```

Gaussian Visualizaton app
-------------------------
Go to Chrome and use 'file:///' in the search bar to pull up the 
file search functionality, and navigate to <project>/gaus/index.html.

Super handy for getting simple 2d visualizations of how gaussians move
under different pressures (loss functions).


Language Modeling
-----------------
To run the coco-flavored wikipedia scraping script (remove --resume for the very first run):
```bash
python3 scripts/scrape_wikipedia_coco.py \
    --annotations-dir ./annotations/annotations \
    --output-dir ./data/wiki_coco \
    --target-words 100000000 \
    --max-rps 2.0 \
    --expand-links \
    --seed-limit=5 \
    --resume
```
Note that eventually wikipedia will rate limit you. Simply wait a while then run with "resume" - the search & scrape states re progressivle saved.

Stalk it's progress with:
```bash
python3 - <<'PY'
import json, collections
counts=collections.Counter()
with open("./data/wiki_coco/articles.jsonl","r",encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        title=json.loads(line).get("title","")
        c=(title[:1] or "#").upper()
        counts[c]+=json.loads(line).get("word_count", 0)
        print(f"A-Z total: {sum([dd for (dk, dd) in counts.items() ])}")
PY
```

Train a tokenizer with a subset of that corpus + MSCoCo image captions (to account for future fintuning):
```bash
python3 -m train.train_tokenizer \
    --run_id mix_bpe_16k \
    --mix \
    --articles_jsonl ./data/wiki_coco/articles.jsonl \
    --mix_captions_words 500000 \
    --mix_wiki_words 7000000 \
    --num_merges 16000 \
    --mix_wiki_sample_mode random \
    --wiki_total_words 71000000 \
    --wiki_read_full \
    --word_count_mode fast \
    --wiki_workers 8 \
    --wiki_chunk_lines 2000
```

Build pre-tokenized train/val/test shards (paragraph-aware, `max_seq_len=256`, `stride=64`):
```bash
python3 scripts/pretokenize_corpus.py \
    --input ./data/wiki_coco/articles.jsonl \
    --out-dir ./data/wiki_tok_256 \
    --tokenizer ./logs/mix_bpe_16k/tokenizer.pt \
    --max_seq_len 256 \
    --stride 64 \
    --split_train 0.95 \
    --split_val 0.04 \
    --split_test 0.01
```
This writes split datasets under `./data/wiki_tok_256/train`, `./data/wiki_tok_256/val`, and `./data/wiki_tok_256/test`, each with shard files + `manifest.jsonl` + `manifest.json`.

Train LM with periodic validation and final test:
```bash
./runlm_best.sh runid1
```
