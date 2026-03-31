# Datasets

This file is the working inventory of the major datasets and corpus products currently present in this repo.

It is organized by training use-case rather than by download order:

- LM pretraining text corpora
- Distilled QA corpora
- VQA / supervised QA corpora
- VM / VLM image and image-text corpora
- Pointing / grounding corpora

Unless otherwise noted, paths are relative to the repo root.

## 1. Wikipedia-Driven LM Corpora

### `data/wiki_coco/`

Primary scraped Wikipedia corpus used for downstream LM mixing and distillation.

- Main file: [data/wiki_coco/articles.jsonl](/home/wdree/percy/vqafromscratch/data/wiki_coco/articles.jsonl)
- State file: [data/wiki_coco/state.json](/home/wdree/percy/vqafromscratch/data/wiki_coco/state.json)
- Current article count: `32,168`
- `state.json` scrape stats:
  - fetched pages: `31,659`
  - total words: `70,326,687`
- Disk usage: about `449M`

What it is for:
- general LM pretraining text
- source text for synthetic QA / distillation generation
- tokenizer training / probe generation

Notes:
- this is the canonical repo-local Wikipedia scrape
- it is already wired into tokenizer and LM training scripts

### `data/pretraining/wikicoco256/`

Tokenized / windowed wiki+coco LM corpus.

- Meta: [data/pretraining/wikicoco256/meta.json](/home/wdree/percy/vqafromscratch/data/pretraining/wikicoco256/meta.json)
- Disk usage: about `1.2G`
- Key preprocessing settings:
  - `max_seq_len=256`
  - `stride=64`
  - `split_train=0.95`
  - `split_val=0.04`
  - `split_test=0.01`

### `data/pretraining/wikicoco256_cleaned/`

Cleaner wiki+coco LM corpus with stronger markdown/wiki cleanup.

- Meta: [data/pretraining/wikicoco256_cleaned/meta.json](/home/wdree/percy/vqafromscratch/data/pretraining/wikicoco256_cleaned/meta.json)
- Disk usage: about `126M`
- Key preprocessing settings:
  - `max_seq_len=256`
  - `window_stride=128`
  - `max_windows_per_doc=4`
  - `clean_wikipedia=true`
  - `clean_markdown=true`
  - list / disambiguation / references-style section dropping enabled

What it is for:
- cleaner LM training runs
- likely stronger signal density than the rawer wiki mix

## 2. Distilled QA Corpora

These are synthetic or distilled QA datasets produced from the Wikipedia corpus and related generation pipelines.

### Top-Level Pretokenization Inputs

- [data/pretraining/distill_qa.jsonl](/home/wdree/percy/vqafromscratch/data/pretraining/distill_qa.jsonl): `51,238` rows
- [data/pretraining/distill_qwen.jsonl](/home/wdree/percy/vqafromscratch/data/pretraining/distill_qwen.jsonl): `102,544` rows

These are the main flattened distillation corpora that feed LM-side pretraining prep.

### Raw Distillation Runs: `data/distill/`

Each subdirectory is a distinct distillation experiment or model family. Most contain:

- `raw.jsonl`
- `rejected.jsonl`
- sometimes `manifest.json`
- sometimes `stats.json`

Current raw-pair counts by run:

- `q1`: `102,544`
- `v3`: `51,238`
- `v5`: `2,941`
- `v1`: `2,000`
- `v2`: `1,369`
- `qwen35_v2`: `1,550`
- `qwen35_9b_v1`: `560`
- `qwen35_4b_v1`: `221`
- `qwen35_9b_v3`: `182`
- `qwen_2bv1`: `121`
- `lfm_v1`: `17`
- `lfm_v2a`: `7`
- `lfm_v2b`: `5`
- `lfm_v2c`: `7`
- `lfm_v2d`: `3`
- `lfm_v3d`: `1`
- `qwen35_v1`: `6`
- `qwen35_9b_v2`: `14`
- `smoke_llama_fix2`: `12`
- `smoke_llama_fix`: `0`
- `v4`: `0`

Approximate total raw distilled QA rows across these run folders: `162,798`

Disk usage for the whole `data/distill/` tree: about `255M`

Example manifests:

- [data/distill/v1/manifest.json](/home/wdree/percy/vqafromscratch/data/distill/v1/manifest.json)
- [data/distill/smoke_llama_fix/manifest.json](/home/wdree/percy/vqafromscratch/data/distill/smoke_llama_fix/manifest.json)
- [data/distill/smoke_llama_fix2/manifest.json](/home/wdree/percy/vqafromscratch/data/distill/smoke_llama_fix2/manifest.json)

What these are for:
- LM QA-style compression / supervision
- synthetic short-form factual QA
- comparing teacher/model families for distillation quality

### Distill Pretokenized Corpora

- [data/pretraining/distill256_cleaned/](/home/wdree/percy/vqafromscratch/data/pretraining/distill256_cleaned)
- [data/pretraining/distill256_cleaned2/](/home/wdree/percy/vqafromscratch/data/pretraining/distill256_cleaned2)

Meta:

- [data/pretraining/distill256_cleaned/meta.json](/home/wdree/percy/vqafromscratch/data/pretraining/distill256_cleaned/meta.json)
- [data/pretraining/distill256_cleaned2/meta.json](/home/wdree/percy/vqafromscratch/data/pretraining/distill256_cleaned2/meta.json)

Disk usage:

- `distill256_cleaned`: about `68M`
- `distill256_cleaned2`: about `91M`

What they are for:
- LM training runs that explicitly bucket distill data apart from wiki data
- e.g. `runlm_mix.sh` style weighted train buckets

### Teacher Distillation Tensor Shards

These are teacher-produced tensor shards for MM/VQA distillation, separate from the text-only LM pretraining corpora above.

- Root: [data/distillation/qwen25vl3b_vqav2_train_v1](/home/wdree/percy/vqafromscratch/data/distillation/qwen25vl3b_vqav2_train_v1)
- Meta: [data/distillation/qwen25vl3b_vqav2_train_v1/meta.json](/home/wdree/percy/vqafromscratch/data/distillation/qwen25vl3b_vqav2_train_v1/meta.json)
- Progress: [data/distillation/qwen25vl3b_vqav2_train_v1/progress.json](/home/wdree/percy/vqafromscratch/data/distillation/qwen25vl3b_vqav2_train_v1/progress.json)
- Teacher model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Source dataset: `vqav2_train`
- Target dataset size in meta: `443,757`
- Current processed count in progress: `430,000`
- Current shard count: `104`
- Disk usage: about `2.38 GiB`

What it is for:
- Qwen-VL teacher distillation over VQAv2 train
- bridge-side supervised distillation experiments
- tensor-sharded teacher targets rather than raw text QA

### Reasoning-Enriched LM Corpora

These were built to supplement the existing wiki + distill LM mix with explicit extractive QA, entailment/verification, and counting/arithmetic signal while keeping the same tokenizer and `max_seq_len=256` pretokenization pipeline.

Canonical outputs:

- mix config: [data/pretraining/reasoning_mix_config.json](/home/wdree/percy/vqafromscratch/data/pretraining/reasoning_mix_config.json)
- report: [data/pretraining/reasoning_corpora_report.md](/home/wdree/percy/vqafromscratch/data/pretraining/reasoning_corpora_report.md)
- report JSON: [data/pretraining/reasoning_corpora_report.json](/home/wdree/percy/vqafromscratch/data/pretraining/reasoning_corpora_report.json)

Raw formatted JSONL inputs:

- [data/pretraining/reasoning_squad.jsonl](/home/wdree/percy/vqafromscratch/data/pretraining/reasoning_squad.jsonl): `10,725` rows
- [data/pretraining/reasoning_multinli.jsonl](/home/wdree/percy/vqafromscratch/data/pretraining/reasoning_multinli.jsonl): `29,297` rows
- [data/pretraining/reasoning_counting.jsonl](/home/wdree/percy/vqafromscratch/data/pretraining/reasoning_counting.jsonl): `51,918` rows

Pretokenized corpora:

- [data/pretraining/squad256_reasoning](/home/wdree/percy/vqafromscratch/data/pretraining/squad256_reasoning)
  - train docs: `10,164`
  - train windows: `11,924`
  - train tokens: `2,272,023`
- [data/pretraining/multinli256_reasoning](/home/wdree/percy/vqafromscratch/data/pretraining/multinli256_reasoning)
  - train docs: `27,878`
  - train windows: `27,882`
  - train tokens: `1,427,194`
- [data/pretraining/counting256_reasoning](/home/wdree/percy/vqafromscratch/data/pretraining/counting256_reasoning)
  - train docs: `49,369`
  - train windows: `49,369`
  - train tokens: `1,425,835`

Combined reasoning supplement:

- total added train tokens: `5,125,052`
- suggested starting mix weights from `reasoning_mix_config.json`:
  - `wiki`: `0.55`
  - `distill`: `0.25`
  - `squad`: `0.0887`
  - `multinli`: `0.0557`
  - `counting`: `0.0556`

What they are for:

- `squad`: extractive evidence-grounded QA
- `multinli`: entailment / contradiction / verification structure
- `counting`: number and simple arithmetic priors for VQA-like reasoning

## 3. VQA / Supervised QA Corpora

### VQAv2

- Root: [data/vqav2](/home/wdree/percy/vqafromscratch/data/vqav2)
- Captions file present: [data/vqav2/captions_train2014.json](/home/wdree/percy/vqafromscratch/data/vqav2/captions_train2014.json)
- Disk usage: about `909M`

What it is for:
- MM / bridge downstream supervised training and eval
- image-question-answer supervision
- caption side-information

### GQA

- Root: [data/gqa](/home/wdree/percy/vqafromscratch/data/gqa)
- Images are extracted under: [data/gqa/raw_images/images](/home/wdree/percy/vqafromscratch/data/gqa/raw_images/images)
- Question archive present: [data/gqa/questions1.2.zip](/home/wdree/percy/vqafromscratch/data/gqa/questions1.2.zip)
- Scene graphs present: [data/gqa/sceneGraphs.zip](/home/wdree/percy/vqafromscratch/data/gqa/sceneGraphs.zip)
- Disk usage: about `23G`

DuckDB-backed QA status from [data/vm_ssl/db/vm_ssl.duckdb](/home/wdree/percy/vqafromscratch/data/vm_ssl/db/vm_ssl.duckdb):

- dataset name: `gqa_questions_1_2`
- labeled QA pairs: `198,553`

What it is for:
- supervised image-question-answer training
- structured visual reasoning supervision
- pseudo-pointing conversion via scene-graph object centroids

### ChartQA

- Root: [data/vm_ssl/raw/chartqa](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/chartqa)
- Stored image tree: [data/vm_ssl/raw/chartqa/images](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/chartqa/images)
- Disk usage: about `766M`

DuckDB-backed QA status from [data/vm_ssl/db/vm_ssl.duckdb](/home/wdree/percy/vqafromscratch/data/vm_ssl/db/vm_ssl.duckdb):

- dataset name: `chartqa`
- stored chart images: `19,321`
- labeled QA pairs: `30,219`
- split breakdown:
  - `train`: `28,299`
  - `val`: `1,920`

Important storage note:

- `ChartQA` is materially one-to-many on the image side
- chart images are deduplicated by content hash and stored once under `data/vm_ssl/raw/chartqa/images/`
- QA rows fan out onto shared `image_id`s in `image_qa_pairs`

What it is for:
- chart-reading QA supervision
- table/axis/value extraction and comparison questions
- later OCR-heavy and diagram-style VQA experiments

## 4. VM / VLM Image and Image-Text Corpora

Canonical registry:

- [data/vm_ssl/db/vm_ssl.duckdb](/home/wdree/percy/vqafromscratch/data/vm_ssl/db/vm_ssl.duckdb)

This DB is the canonical image / image-text / QA manifest layer for the VM recipe work.

### Current DuckDB Composition

#### Images by source

- `chartqa`: `19,321`
- `inat2021`: `500,000`
- `coco_local`: `204,721`
- `gqa`: `148,854`
- `cc3m_subset_50k`: `49,802`
- `flickr30k`: `31,783`
- `midjourney_v6_recap_30k`: `30,000`
- `textocr`: `24,594`
- `coco_text`: `16,171`
- `textcaps`: `7,936`
- `openimages_v7`: `60`
- `mapillary_vistas`: `40`
- `textocr_test`: `30`

Total indexed images: `1,033,312`

#### Valid image-text pairs by dataset

- `flickr30k`: `155,070`
- `coco_captions_2014`: `127,227`
- `coco_text_captions`: `76,701`
- `cc3m_subset_50k`: `49,152`
- `midjourney_v6_recap_llava`: `29,877`
- `midjourney_v6_recap_qwen3`: `24,913`
- `midjourney_v6_recap_gemini`: `23,927`

Total valid image-text pairs: `486,867`

#### Valid labeled image-QA pairs in DuckDB

- `chartqa`: `30,219`
- `gqa_questions_1_2`: `198,553`

Total valid labeled image-QA pairs: `228,772`

### Notable Raw Image Roots

- [data/vm_ssl/raw/inat2021_train_mini](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/inat2021_train_mini)
- [data/vm_ssl/raw/coco_text_materialized](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/coco_text_materialized)
- [data/vm_ssl/raw/textocr_trainval](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/textocr_trainval)
- [data/vm_ssl/raw/textocr_full](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/textocr_full)
- [data/vm_ssl/raw/flickr30k_hf](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/flickr30k_hf)
- [data/vm_ssl/raw/textcaps_hf](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/textcaps_hf)
- [data/vm_ssl/raw/textcaps_materialized](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/textcaps_materialized)
- [data/vm_ssl/raw/cc3m_subset_50k](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/cc3m_subset_50k)
- [data/vm_ssl/raw/midjourney_v6_recap_30k](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/midjourney_v6_recap_30k)
- [data/vm_ssl/raw/chartqa](/home/wdree/percy/vqafromscratch/data/vm_ssl/raw/chartqa)

### Registered Artifact Inventory

Largest registered artifacts:

- `inat2021_train_mini`: `44.64 GB`
- extracted GQA image tree: `21.98 GB`
- `inat2021_val_partial`: `4.71 GB`
- `coco_text_hf_snapshot`: `2.75 GB`
- `textocr_trainval_openimages`: `1.91 GB`
- `textocr_full_repo`: `1.30 GB`

What this family is for:
- VM SSL pretraining
- DINO / cross / SigLIP-style VM recipe runs
- image-text alignment experiments
- OCR-rich and diverse image mixture design

## 5. Pointing / Grounding Corpora

Root:

- [data/pointing](/home/wdree/percy/vqafromscratch/data/pointing)

This is the new pointing / grounding data family for spatial supervision.

### Raw per-source indexes

- [pixmo_points.jsonl](/home/wdree/percy/vqafromscratch/data/pointing/index/pixmo_points.jsonl): `32,012`
- [molmo2_multiimagepoint.jsonl](/home/wdree/percy/vqafromscratch/data/pointing/index/molmo2_multiimagepoint.jsonl): `36,804`
- [gqa_point_sidecar.jsonl](/home/wdree/percy/vqafromscratch/data/pointing/index/gqa_point_sidecar.jsonl): `174,384`
- [molmo2_videopoint.jsonl](/home/wdree/percy/vqafromscratch/data/pointing/index/molmo2_videopoint.jsonl): `0`
  - currently blocked by requester-pays GCS access on the upstream video bucket

Stored image roots:

- [data/pointing/images/pixmo_points](/home/wdree/percy/vqafromscratch/data/pointing/images/pixmo_points)
- [data/pointing/images/molmo2_multiimagepoint](/home/wdree/percy/vqafromscratch/data/pointing/images/molmo2_multiimagepoint)

Disk usage for `data/pointing/`: about `22G`

### Unified training-ready grounding index

- [data/pointing/train_index.jsonl](/home/wdree/percy/vqafromscratch/data/pointing/train_index.jsonl): `118,002`
- [data/pointing/mix_config.json](/home/wdree/percy/vqafromscratch/data/pointing/mix_config.json)

Training-ready source breakdown:

- `gqa`: `49,404`
- `molmo2_multiimagepoint`: `36,620`
- `pixmo_points`: `31,978`

Skipped during build:

- `123,729` non-train-split records
- `1,469` degenerate same-grid-cell records

What it is for:
- grounding supervision for bridge / perceiver cross-attention
- precomputed 14x14 token-grid soft targets
- mixed VQA + grounding supervision

Important representation choice:

- one record per image/question pair
- multi-point images remain one sample with many points
- training target is one combined soft heatmap, not one row per point

## 6. Auxiliary Eval / Support Data

### SSL Eval Bundle

- Root: [data/ssl_eval](/home/wdree/percy/vqafromscratch/data/ssl_eval)
- Archive: [data/ssl_eval/cifar-10-python.tar.gz](/home/wdree/percy/vqafromscratch/data/ssl_eval/cifar-10-python.tar.gz)
- Extracted tree: [data/ssl_eval/cifar-10-batches-py](/home/wdree/percy/vqafromscratch/data/ssl_eval/cifar-10-batches-py)
- Disk usage: about `340 MiB`

What it is for:
- SSL feature quality evaluation via [ssl_knn.py](/home/wdree/percy/vqafromscratch/evals/ssl_knn.py)
- CIFAR-10 kNN probes for VM/SSL sanity checks

### VQAv2 Smoke Annotation Mirror

- Root: [data/vqav2_smoke_ann](/home/wdree/percy/vqafromscratch/data/vqav2_smoke_ann)
- Disk usage: about `661 MiB`

What it is for:
- smoke/debug copies of VQAv2 question and annotation JSONs
- lightweight local experimentation without treating it as the canonical VQAv2 root

## 7. Quick Map By Training Objective

### LM pretraining

- `data/wiki_coco/`
- `data/pretraining/wikicoco256/`
- `data/pretraining/wikicoco256_cleaned/`
- `data/pretraining/distill256_cleaned/`
- `data/pretraining/distill256_cleaned2/`
- `data/pretraining/squad256_reasoning/`
- `data/pretraining/multinli256_reasoning/`
- `data/pretraining/counting256_reasoning/`

### Distilled QA / LM supervision

- `data/distill/*`
- `data/pretraining/distill_qa.jsonl`
- `data/pretraining/distill_qwen.jsonl`

### Teacher distillation / MM supervision shards

- `data/distillation/qwen25vl3b_vqav2_train_v1/`

### VM SSL / VM recipe work

- `data/vm_ssl/db/vm_ssl.duckdb`
- `data/vm_ssl/raw/*`
- `data/gqa/raw_images/images`
- `data/ssl_eval/`
- local COCO image tree and OCR-rich sources represented in DuckDB

### MM / VQA supervised training

- `data/vqav2/`
- GQA QA pairs in DuckDB
- ChartQA QA pairs in DuckDB and chart images under `data/vm_ssl/raw/chartqa/`

### Grounding / pointing supervision

- `data/pointing/index/*.jsonl`
- `data/pointing/train_index.jsonl`

## 8. Canonical Sources of Truth

If you are unsure where to look first:

- VM/VLM image and pair inventory:
  - [data/vm_ssl/db/vm_ssl.duckdb](/home/wdree/percy/vqafromscratch/data/vm_ssl/db/vm_ssl.duckdb)
- Wikipedia scrape status:
  - [data/wiki_coco/state.json](/home/wdree/percy/vqafromscratch/data/wiki_coco/state.json)
- LM reasoning supplement:
  - [data/pretraining/reasoning_mix_config.json](/home/wdree/percy/vqafromscratch/data/pretraining/reasoning_mix_config.json)
  - [data/pretraining/reasoning_corpora_report.md](/home/wdree/percy/vqafromscratch/data/pretraining/reasoning_corpora_report.md)
- Distillation run outputs:
  - `data/distill/*`
- Teacher distillation tensor shards:
  - [data/distillation/qwen25vl3b_vqav2_train_v1](/home/wdree/percy/vqafromscratch/data/distillation/qwen25vl3b_vqav2_train_v1)
- Pointing / grounding training artifact:
  - [data/pointing/train_index.jsonl](/home/wdree/percy/vqafromscratch/data/pointing/train_index.jsonl)
