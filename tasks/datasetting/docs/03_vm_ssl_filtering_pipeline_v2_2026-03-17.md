# VM SSL Filtering Pipeline v2

## Status

This document supersedes the more over-specified v1 filtering spec in [02_vm_ssl_filtering_grading_spec_2026-03-16.md](/home/wdree/percy/vqafromscratch/tasks/datasetting/docs/02_vm_ssl_filtering_grading_spec_2026-03-16.md).

The core changes are:

- simpler scoring for v1 implementation
- per-bucket and per-cluster normalization for novelty and dedup
- explicit document-like classifier instead of OCR-area heuristics alone
- explicit NSFW and watermark filtering
- a small hard-negatives / distribution-shift slice
- fewer schema fields with no clear consumer

The philosophy is unchanged:

- keep the corpus compact
- preserve scenery, OCR-natural-scene, and fine-grained nature
- avoid aesthetic collapse
- enforce diversity instead of selecting a global top-k

## Goal

Build a **small, strong, diverse** image corpus for DINOv2-style ViT pretraining that is materially more useful than:

- raw benchmark dumps
- huge noisy web scrapes
- aesthetic-filtered internet photos

The target is a dataset that is:

- cheap enough to store and train on
- clean enough not to waste compute
- broad enough to support later bridge-style VLM behavior
- deliberately robust to scene messiness, text-in-the-world, and distribution shift

## Storage Model

This plan is explicitly **DuckDB-first**.

Use:

- normal filesystem directories for raw images and derived artifacts
- one DuckDB database as the canonical metadata and pipeline-state layer

Do not require Parquet in the default architecture.

Why:

- image bytes already belong naturally on disk
- the main need is queryable metadata, provenance, scores, clusters, and selection state
- DuckDB gives that cleanly without introducing another mandatory persistence format

Parquet remains optional for interchange or export if a concrete workflow later needs it.

## Guiding Decisions

### 1. Two-stage dedup stays

The first dedup stage should be cheap and blunt:

- exact hash
- perceptual hash

The second stage should be semantic and expensive:

- embedding-space clustering
- cluster-local thresholds

This is the right architecture. Keep it.

### 2. Diversity enforcement is the center of the plan

The most important decision in this pipeline is:

- do **not** rank the whole dataset globally and keep the top examples

Instead:

- bucket the data
- dedup within local neighborhoods
- enforce source and bucket quotas
- fill a target mixture deliberately

That is the mechanism that prevents the pretraining data from collapsing toward a narrow visual mode.

### 3. Technical quality is mostly a gate, not a weighted science fair

For v1, do not use a giant weighted composite with twenty knobs.

Use:

- hard technical filters
- a simple semantic richness score
- a normalized novelty score

That is enough to get a strong first production run and review it honestly.

### 4. "Messy" images are good

The pipeline should not be biased toward clean aesthetic photos.

It should positively preserve:

- cluttered scenes
- odd viewpoints
- street imagery
- signage
- packaging
- embedded text
- non-Western domestic and outdoor contexts
- hard fine-grained natural images

### 5. Distribution-shift imagery should be intentional

A small amount of "hard negative" or distribution-shift imagery should be part of the final corpus.

Candidate sources:

- `ObjectNet`
- `Dollar Street`
- a curated slice of `YFCC100M`

This should stay small, but it should exist.

## Target Output

The pipeline should produce:

1. a cleaned local image store
2. a DuckDB registry of all valid images
3. DuckDB tables for scores, dedup clusters, and selection decisions
4. a SQL view or exported file for the final training subset
5. review grids and drop statistics

Recommended artifact layout:

```text
data/vm_ssl/
  raw/
  staged/
    normalized/
    thumbnails/
    embeddings/
  db/
    vm_ssl.duckdb
  reports/
    filter_stats.json
    dedup_stats.json
    bucket_mix.json
    review_samples/
```

Recommended canonical DuckDB tables:

- `images`
- `image_ocr`
- `image_scores`
- `image_clusters`
- `image_selection`
- `review_samples`

Recommended convenience views:

- `valid_images`
- `dedup_representatives`
- `final_training_images`

## Source Plan

### Core sources

- `Open Images V7`
- `Places365`
- `Mapillary Vistas`
- `iNaturalist 2021`
- `TextOCR` or `COCO-Text`

### Hard-negatives / distribution-shift slice

- `ObjectNet` preferred if license/use case is acceptable
- `Dollar Street` preferred for household and global-context diversity
- curated `YFCC100M` slice only if the cleaner options are insufficient

This slice should be intentionally small:

- target `3-5%` of final images

## Primary Buckets

v2 keeps only primary buckets in the core schema.

Recommended primary buckets:

- `general_web_photo`
- `scene_landscape_architecture`
- `street_signage_navigation`
- `finegrained_nature`
- `people_social_activity`
- `indoor_lived_space`
- `ocr_natural_scene`
- `distribution_shift_hard`
- `document_like`
- `low_information`

There is no secondary tag system in v1 of the implementation. Add it only if a real balancing need appears later.

## Minimal Canonical Schema

The canonical image registry should carry at least:

```text
image_id
source_name
source_native_id
source_split
source_path_or_url
local_path
sha256
width
height
aspect_ratio
file_size_bytes
decode_ok
phash
embedding_path
ocr_token_count
ocr_area_fraction
nsfw_score
watermark_score
document_like_score
semantic_richness_score
novelty_score_raw
novelty_score_norm
primary_bucket
quality_gate_pass
drop_reason
cluster_id
cluster_rank
final_keep
```

Additional fields can live in side tables if that keeps the schema cleaner. The important point is that DuckDB is the canonical query layer.

## Revised Pipeline

The revised pipeline should run in this order:

1. ingest
2. decode validation
3. technical hard filters
4. NSFW and watermark filters
5. cheap dedup
6. normalization and thumbnailing
7. embedding extraction
8. OCR routing signals
9. document-like classifier
10. primary bucket assignment
11. cluster-local dedup
12. novelty normalization
13. quota-aware final selection
14. review and audit

## Stage 1: Ingest

For every source image:

- fetch image and source metadata
- assign stable `image_id`
- preserve source provenance
- compute `sha256`

Recommended id:

```text
{source_name}:{source_native_id}
```

Fallback:

```text
{source_name}:{sha256[:16]}
```

## Stage 2: Decode Validation

Drop immediately if:

- image cannot be decoded
- decoded dimensions are invalid
- file is non-raster or unsupported
- image is obviously corrupted or truncated

Always record `drop_reason`.

## Stage 3: Technical Hard Filters

These are hard gates, not weighted preferences.

### Minimum size

Default:

- short side `>= 256`
- area `>= 256 * 256`

Stricter experimental setting:

- short side `>= 320`

### Aspect ratio

Default:

- keep if aspect ratio is within `[1/3.5, 3.5]`

### Low-information / blank filtering

Drop if:

- image variance is extremely low
- large blank borders dominate
- decoded image is mostly a solid or near-solid field

### Compression / corruption sanity

Drop or flag if:

- bytes-per-pixel is implausibly low
- decode shows strong truncation behavior
- image is visually broken

Technical outcome:

- everything that passes is trainable
- everything that fails is dead weight

## Stage 4: NSFW And Watermark Filters

This stage is mandatory in v2.

### NSFW

Run a lightweight image safety classifier and store `nsfw_score`.

Policy:

- auto-drop high-confidence NSFW
- retain low-confidence ambiguous cases for review if needed

### Watermarks

Run a lightweight watermark detector and store `watermark_score`.

Good enough first pass:

- heuristic edge/text detector near borders
- optional small classifier later

Policy:

- auto-drop clear, large watermarks
- keep mild uncertainty for review

This is cheap insurance even if the current sources are relatively clean.

## Stage 5: Cheap Dedup

This stage exists to reduce work before embeddings.

### Exact dedup

Cluster by:

- `sha256`

Keep:

- highest-resolution exact instance

### Perceptual dedup

Compute:

- `phash`

Default policy:

- Hamming distance `<= 4`: auto-drop duplicate candidate
- Hamming distance `5-8`: keep for semantic-stage confirmation

This stage should be aggressive enough to cut waste, but not so aggressive that it kills legitimate viewpoint variation.

## Stage 6: Normalization

For all surviving images:

- apply EXIF orientation
- decode to RGB
- create a review thumbnail at `224px`
- optionally create a cached larger image for feature extraction

Use deterministic preprocessing.

## Stage 7: Embedding Extraction

Extract one primary embedding for all images.

Recommended default:

- `CLIP ViT-L/14` or `SigLIP`

Optional second embedding:

- pretrained `DINOv2`

For v1 implementation:

- one embedding model is enough

Use the same embedding space for:

- cluster formation
- novelty estimation
- bucket classifier features if needed

## Stage 8: OCR Routing Signals

Run lightweight OCR to extract:

- `ocr_token_count`
- `ocr_area_fraction`
- average token confidence

This stage is not the final document filter.

It exists to:

- identify natural-image OCR candidates
- help route examples into `ocr_natural_scene`
- provide features to the document classifier

## Stage 9: Document-Like Classifier

This is a new mandatory stage in v2.

OCR area fraction alone is not good enough to distinguish:

- storefront with lots of text
- photographed sheet of paper
- slide screenshot
- app UI screenshot

### First-pass implementation

Train a lightweight binary classifier:

- input: image embedding
- optional extra features: `ocr_area_fraction`, `ocr_token_count`, aspect ratio
- model: logistic regression or linear probe

Training set:

- label `300-1000` examples by hand
- include:
  - storefronts
  - street signs
  - posters in scenes
  - packaging
  - photographed documents
  - screenshots
  - slides

Output:

- `document_like_score`

Policy:

- auto-drop high-confidence `document_like`
- route medium-confidence cases to review

This will outperform heuristic OCR thresholds very quickly.

## Stage 10: Primary Bucket Assignment

Assign each image one primary bucket.

Inputs:

- source prior
- OCR signals
- document-like score
- lightweight zero-shot bucket classifier or prompt matching over embeddings

Recommended behavior:

- `document_like` and `low_information` are explicit drop-oriented buckets
- all other buckets are selection buckets

## Stage 11: Cluster-Local Dedup

This is the expensive dedup stage.

### Core idea

Do not use one global cosine threshold across the entire dataset.

Instead:

1. partition images by primary bucket
2. within each bucket, form coarse embedding clusters
3. dedup within those local clusters

Good clustering options:

- `k-means`
- hierarchical mini-batch k-means
- graph connected components from ANN neighborhoods

### Why

Different buckets have different geometry:

- nature images cluster tightly
- street scenes spread out
- indoor lived spaces sit somewhere in between

Global thresholds distort these buckets.

### Threshold bands

Do not hard-code one threshold as truth. Sweep three regimes:

- conservative: cosine `>= 0.985`
- standard: cosine `>= 0.970`
- aggressive: cosine `>= 0.960`

Within each bucket cluster:

- compute local similarity statistics
- normalize thresholding around local cluster density

Practical first pass:

- use `0.970` as the default experiment
- compare retained counts and review grids against `0.985` and `0.960`

### Cluster representative selection

Within each local duplicate cluster, rank by:

1. `quality_gate_pass`
2. `semantic_richness_score`
3. resolution
4. OCR usefulness if the bucket is OCR-relevant

Do not use a large weighted formula here. Lexicographic ranking is good enough to start.

## Stage 12: Novelty Normalization

Novelty is kept, but it is normalized locally.

### Raw novelty

For each image:

- find `k` nearest neighbors in embedding space
- compute local density or mean similarity

This gives `novelty_score_raw`.

### Normalized novelty

Normalize novelty:

- within primary bucket
- optionally within local cluster family

Recommended normalization:

- z-score or percentile rank within bucket

Store:

- `novelty_score_raw`
- `novelty_score_norm`

This prevents:

- tight buckets being unfairly punished
- sparse buckets being unfairly rewarded

## Stage 13: Simplified Selection Score

v2 uses a deliberately simple keep score.

### Hard gate first

An image must pass:

- technical hard filters
- NSFW threshold
- watermark threshold
- document-like threshold

If not, it is out.

### Three core signals

For images that pass the hard gate, compute:

1. `semantic_richness_score`
2. `novelty_score_norm`
3. `quota_pressure_score`

Where:

- `semantic_richness_score` measures useful visual structure
- `novelty_score_norm` measures local distinctiveness
- `quota_pressure_score` measures how badly the bucket/source still needs coverage

Recommended first-pass formula:

```text
keep_score =
  0.45 * semantic_richness_score +
  0.35 * novelty_score_norm +
  0.20 * quota_pressure_score
```

That is enough for v1.

### Semantic richness

Do not overcomplicate it.

It should roughly reward:

- scene complexity
- meaningful text-in-scene
- object/layout richness
- fine-grained detail

It should penalize:

- nearly empty images
- blurry closeups with little structure
- weak generic filler

Good first-pass implementation:

- linear probe or shallow MLP on top of image embeddings
- trained from a few thousand reviewed examples

### Quota pressure

This is a dynamic score, not a static attribute.

Increase it when:

- a primary bucket is below target
- a source is underrepresented relative to target

This is how the final mixture stays balanced.

## Stage 14: Final Mixture Targets

For a first serious production build, set a hard image budget.

Recommended initial target:

- `4M` images

### Target source mix

- `48-50%` Open Images
- `18-22%` Places365
- `10-14%` Mapillary
- `10-14%` iNaturalist 2021
- `4-5%` TextOCR / COCO-Text
- `3-5%` distribution-shift source

The exact values should depend on what survives filtering.

### Protected bucket floors

Suggested initial floors for a `4M` build:

- `scene_landscape_architecture`: `>= 700k`
- `street_signage_navigation`: `>= 300k`
- `finegrained_nature`: `>= 350k`
- `ocr_natural_scene`: `>= 120k`
- `distribution_shift_hard`: `>= 100k`

The point is not the exact numbers. The point is that these buckets are protected by design.

## Final Selection Procedure

Use a greedy quota-aware selection loop.

### Step order

1. drop all failed-gate images
2. dedup within local clusters
3. sort by `keep_score` within each bucket-source slice
4. fill protected bucket floors first
5. fill source targets second
6. fill remaining budget by highest `keep_score`
7. apply local neighborhood suppression while selecting

### Local neighborhood suppression

After selecting an image:

- penalize very similar nearby examples in the same bucket cluster

This keeps the final mixture from filling with visually repetitive near-neighbors.

## Review And Audit

No corpus should be accepted without visual review.

Generate review grids for:

- top selected images by bucket
- borderline selected images by bucket
- dropped duplicates
- dropped document-like images
- dropped NSFW / watermark images
- OCR bucket retained images
- distribution-shift retained images

Minimum review load before freezing v1:

- `100-200` sampled images per strategic bucket

## Metrics To Track

### Throughput

- raw image count
- valid decoded image count
- post-hard-filter count
- post-cheap-dedup count
- post-cluster-dedup count
- final selected count

### Drop reasons

- decode failures
- too small
- aspect ratio
- low information
- NSFW
- watermark
- document-like
- exact duplicate
- phash duplicate
- embedding duplicate
- over bucket quota
- over source quota

### Balance

- source proportions
- bucket proportions
- mean novelty per bucket
- mean semantic richness per bucket
- fraction of final images coming from sparse local neighborhoods

### Efficiency

- bytes per retained image
- raw disk footprint
- final disk footprint
- percent dropped at each stage

## Recommended First Production Settings

### Hard gates

- short side `>= 256`
- area `>= 256^2`
- aspect ratio within `[1/3.5, 3.5]`

### Safety / contamination

- NSFW filter enabled
- watermark filter enabled
- document-like classifier enabled

### Cheap dedup

- exact `sha256`
- `phash <= 4` auto-drop

### Embedding dedup experiment band

Run all three:

- `0.985`
- `0.970`
- `0.960`

Compare:

- retained counts
- bucket fill rates
- review grids

Do not guess this from theory alone.

### Final target size

- `4M`

## Implementation Notes

Suggested stack:

- Python
- `duckdb`
- `polars` or `pandas`
- `Pillow` or `opencv`
- `faiss`
- `torch`
- `open_clip` or equivalent
- lightweight OCR package
- simple sklearn linear models for document-like and semantic-richness probes

Suggested script split:

```text
scripts/vm_ssl/
  ingest_sources.py
  validate_images.py
  filter_nsfw_and_watermarks.py
  dedup_phash.py
  extract_embeddings.py
  run_ocr.py
  train_document_classifier.py
  assign_primary_buckets.py
  dedup_semantic_clusters.py
  compute_novelty.py
  select_final_subset.py
  render_review_grids.py
```

## Immediate Next Steps

1. implement the hard-gate stages first
2. stand up the DuckDB schema and review-grid machinery
3. label the first `300-1000` examples for the document-like classifier
4. run the three semantic dedup thresholds on a medium pilot slice
5. review bucket-level outputs before committing to a full download/build

## Final Position

The revised plan is intentionally simpler and stronger:

- fewer fake-precise weights
- more local normalization
- more explicit contamination filters
- more deliberate distribution shaping

That is the right direction for a first real VM SSL corpus build.
