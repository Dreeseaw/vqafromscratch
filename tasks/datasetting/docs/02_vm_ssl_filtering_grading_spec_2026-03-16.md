# VM SSL Filtering And Grading Spec

## Purpose

This document specifies a filtering, grading, deduplication, and balancing pipeline for building a **small-but-strong** image corpus for DINOv2-style ViT pretraining.

The design target is:

- materially better than just training on raw benchmark dumps
- small enough to fit realistic disk budgets
- clean enough to avoid wasting compute on junk
- diverse enough to avoid collapsing into only pretty web photos or only benchmark-style scenes
- biased toward the needs of a future bridge-style VLM, especially:
  - scene understanding
  - OCR-adjacent natural imagery
  - fine-grained visual detail
  - broad visual robustness

This is not a "max scale at all costs" pipeline. It is a **curated efficiency** pipeline.

## Design Principles

### 1. Prefer quality-density over raw count

For a smaller VM, ten million mediocre or repetitive images are often worse than a few million carefully filtered and balanced ones.

### 2. Do not optimize for aesthetics

The goal is not beautiful internet photos. The goal is useful visual structure.

That means we want to keep:

- clutter
- odd viewpoints
- embedded text
- real-world messiness
- scene diversity
- fine-grained visual distinctions

and remove:

- broken images
- near-duplicates
- low-information junk
- low-value synthetic-looking spam

### 3. Diversity has to be enforced

If you simply keep the top-scoring images globally, the dataset will overconcentrate in a few easy modes:

- centered objects
- pretty outdoor photos
- high-aesthetic consumer photography
- "internet clean" imagery

That is not what you want.

### 4. OCR is a booster, not the core

Scene text should be explicitly included, but only as a small portion of the total mixture.

### 5. Scene-heavy data matters

A weak VM often acts too object-centric and too crop-centric. Scene datasets should be deliberately preserved during curation.

## Target Output

The pipeline should produce:

1. a clean image store on disk
2. a manifest with metadata and scores for every retained image
3. a final training manifest with only the selected subset
4. an audit report explaining what got dropped and why

Recommended artifact layout:

```text
data/vm_ssl/
  raw/
    openimages/
    places365/
    mapillary/
    inat2021/
    textocr/
  staged/
    normalized/
    thumbnails/
    embeddings/
  manifests/
    raw_manifest.parquet
    filtered_manifest.parquet
    dedup_manifest.parquet
    final_manifest.parquet
  reports/
    source_stats.json
    filter_drop_stats.json
    dedup_stats.json
    bucket_balance_stats.json
```

Use `parquet` for manifests unless there is a strong reason not to.

## Source Assumptions

The intended initial source mix is:

- `Open Images V7`
- `Places365`
- `Mapillary Vistas`
- `iNaturalist 2021`
- `TextOCR` or `COCO-Text`

Each source should carry `source_name` as a first-class manifest field. Never lose source provenance.

## Data Model

Every image row in the working manifest should carry at least:

```text
image_id
source_name
source_split
source_path_or_url
local_path
sha256
width
height
aspect_ratio
file_size_bytes
mime_type
decode_ok
is_animated
phash
clip_embedding_path
vit_embedding_path
ocr_token_count
ocr_area_fraction
sharpness_score
exposure_score
compression_score
artifact_score
semantic_richness_score
novelty_score
quality_score
bucket_scene
bucket_ocr
bucket_nature
bucket_people
bucket_object
bucket_street
bucket_indoor
bucket_outdoor
keep_stage
drop_reason
final_keep
```

Not all fields need to exist on day one, but the schema should allow them.

## Pipeline Overview

The full pipeline should run in this order:

1. ingest
2. decode validation
3. technical hard filters
4. image normalization and thumbnailing
5. cheap dedup
6. feature extraction
7. quality grading
8. semantic bucketing
9. embedding-based dedup
10. diversity-aware subset selection
11. final audit

Do not reverse steps 8 through 10. Dedup and balancing should operate on semantic information, not just raw file stats.

## Stage 1: Ingest

The ingestion layer should:

- fetch images and metadata
- assign a stable internal `image_id`
- compute a raw `sha256`
- keep source-native ids separately

Recommended internal id:

```text
{source_name}:{source_native_id}
```

If the source lacks a stable native id, use:

```text
{source_name}:{sha256[:16]}
```

## Stage 2: Decode Validation

Reject immediately if:

- image cannot be decoded
- decoded image is zero-area
- file is not a supported raster image
- animated image handling is unsupported
- file extension and decoded format mismatch in obviously bad ways

Keep a `drop_reason` for every failure. Never silently discard rows.

## Stage 3: Technical Hard Filters

These are non-negotiable drops intended to eliminate obvious waste.

### Minimum resolution

Suggested first-pass thresholds:

- short side `< 256` -> drop
- area `< 256 * 256` -> drop

For a stricter build:

- short side `< 320` -> drop

### Aspect ratio

Suggested thresholds:

- aspect ratio `> 3.5` or `< 1/3.5` -> drop

This removes extreme panoramas and banner-like fragments that add little value for a compact ViT.

### File size sanity

Drop if:

- file size is implausibly small for resolution
- decode shows obvious truncation or corruption

Cheap heuristic:

- if `file_size_bytes / (width * height) < 0.05`, flag for artifact scoring or drop

### Border / blank-image filter

Drop if:

- nearly all pixels are identical
- image is dominated by blank white/black/solid regions
- low-variance border region suggests screenshots, scans, or broken exports

### Transparency and alpha

If alpha exists:

- composite onto a neutral background for evaluation
- if alpha covers too much of the image and the visible content is tiny, drop

## Stage 4: Normalization

For each retained file after hard filtering:

- store original bytes if disk permits
- generate normalized RGB decode
- generate a `224px` thumbnail for fast review
- optionally generate a `518px` or `448px` cache image for feature extraction

Use:

- `RGB`
- consistent EXIF orientation handling
- bicubic or area interpolation depending on resize direction

Do not normalize away all weirdness; just make downstream scoring deterministic.

## Stage 5: Cheap Dedup

Run a first-pass duplicate filter before expensive embedding extraction.

### Exact dedup

Cluster on:

- exact `sha256`

Keep:

- the first successfully decoded file, or
- the file with the highest resolution if identical images somehow appear with different metadata

### Near-exact perceptual dedup

Compute:

- `phash`
- optionally `dhash`

Suggested threshold:

- Hamming distance `<= 4` for aggressive exact-ish duplicates
- `<= 8` for a second looser review queue

Only auto-drop very close matches at this stage.

## Stage 6: Feature Extraction

Extract the following on all remaining images.

### Vision embeddings

Use one strong pretrained embedding model for semantic dedup and coarse bucketing. Good options:

- `CLIP ViT-L/14`
- `SigLIP`
- pretrained `DINOv2`

Recommendation:

- use `CLIP` or `SigLIP` for broad semantic space
- optionally keep a second `DINOv2` embedding for pure vision-space checks

### OCR features

Run a lightweight OCR detector / recognizer, such as:

- `PaddleOCR`
- `EasyOCR`
- a fast DBNet/CRNN-style pipeline

Store:

- `ocr_token_count`
- `ocr_area_fraction`
- `mean_token_confidence`

Do not use OCR confidence as a primary keep metric. Use it as a routing signal.

### Technical signals

Compute:

- Laplacian variance for sharpness
- mean brightness
- brightness histogram spread
- saturation statistics
- JPEG blockiness / compression heuristics
- noise estimate

### Semantic signals

Use zero-shot or lightweight classifiers to estimate:

- scene vs object vs document-like
- indoor vs outdoor
- people presence
- street/road presence
- natural-world / flora-fauna presence
- text/sign/storefront presence

These do not need to be perfect. They only need to be good enough for balancing.

## Stage 7: Quality Grading

Every remaining image gets a scalar `quality_score`, but that score must be decomposable.

Recommended decomposition:

```text
quality_score =
  0.25 * technical_quality +
  0.25 * semantic_richness +
  0.20 * novelty_local +
  0.15 * source_priority +
  0.15 * bucket_priority
```

This should be adjustable, but the key design is:

- technical quality matters
- semantic richness matters
- novelty matters
- source and bucket preferences matter

### 7.1 Technical quality

Technical quality is not beauty. It is:

- not too blurry
- not too dark or blown out
- not obviously broken
- not extremely compressed

Suggested subscore:

```text
technical_quality =
  0.35 * sharpness_component +
  0.20 * exposure_component +
  0.20 * compression_component +
  0.15 * artifact_component +
  0.10 * resolution_component
```

Clamp each component to `[0, 1]`.

### 7.2 Semantic richness

Semantic richness should reward images that contain substantial visual structure.

Proxy signals:

- object/region count from a lightweight detector
- OCR token presence
- scene complexity
- embedding entropy relative to local neighborhood
- color and texture diversity

Suggested heuristic boosts:

- moderate OCR presence: positive
- complex street scene: positive
- fine-grained natural image: positive
- extremely empty sky/wall-only image: negative
- huge close-up blur or macro with little structure: negative unless intentionally retained by nature quota

### 7.3 Novelty

Novelty should be local, not global.

Bad approach:

- "this image looks unusual in the whole dataset"

Better approach:

- "this image is a good representative within its local semantic neighborhood"

Implementation:

- compute `k` nearest neighbors in embedding space
- define `novelty_local` using distance to neighbors and duplicate density
- favor images in sparse areas and strong exemplars in dense areas

### 7.4 Source priority

Recommended initial source priors:

- `Places365`: `1.00`
- `Mapillary`: `1.00`
- `iNaturalist`: `0.95`
- `Open Images`: `0.90`
- `TextOCR`: `0.90`
- `COCO-Text`: `0.80`

This does not mean keep more of smaller sources automatically. It just prevents scene and OCR sources from being drowned by the large base corpus.

### 7.5 Bucket priority

Recommended initial bucket priors:

- scenery / scene-heavy: high
- street / signage / road geometry: high
- fine-grained natural-world: medium-high
- general natural photos: medium
- OCR-heavy natural images: medium
- low-information close-ups: low
- document-like images: very low

## Stage 8: Semantic Bucketing

Every image should receive one primary bucket and optional secondary tags.

Suggested primary buckets:

- `general_web_photo`
- `scene_landscape_architecture`
- `street_signage_navigation`
- `finegrained_nature`
- `indoor_lived_space`
- `people_social_activity`
- `ocr_natural_scene`
- `product_object_closeup`
- `document_like`
- `low_information`

Suggested secondary tags:

- `indoor`
- `outdoor`
- `text_present`
- `vehicle_present`
- `animal_present`
- `plant_present`
- `crowded`
- `wide_view`
- `night`
- `weather`

Bucketing can be produced using a mixture of:

- source priors
- OCR output
- zero-shot labels
- scene classifier
- lightweight object detector

Perfect classification is not required. Stable rough grouping is enough.

## Stage 9: Embedding-Based Dedup

This is the most important space-saving stage after hard filtering.

### Embedding choice

Use normalized embeddings from:

- `CLIP` or `SigLIP` for primary semantic dedup

Optional:

- `DINOv2` embeddings for a second dedup pass focused on visual similarity

### Indexing

Use:

- `faiss`
- cosine similarity or L2 on normalized vectors

Suggested workflow:

1. build ANN index
2. retrieve top `k=20..50` neighbors for each image
3. create duplicate edges above a similarity threshold
4. form connected components or local duplicate clusters

### Thresholds

These must be tuned empirically, but a reasonable starting point:

- semantic near-duplicate candidate if cosine similarity `>= 0.965`
- aggressive duplicate if cosine similarity `>= 0.985`

Auto-drop only the aggressive tier at first.

### Cluster representative selection

Within each duplicate cluster, keep the image with the best:

```text
rep_score =
  0.35 * quality_score +
  0.20 * resolution_component +
  0.20 * semantic_richness +
  0.15 * OCR_bonus_if_bucket_needs_it +
  0.10 * source_priority
```

Do not always pick the sharpest image if it is semantically weaker.

## Stage 10: Diversity-Aware Subset Selection

This is where the final small dataset is actually chosen.

### Global target size

Pick a hard image budget up front.

Example useful scales:

- `2M` images: very compact
- `4M` images: strong practical target
- `6M-8M` images: still manageable if disk allows

For a smaller VM, `4M` well-chosen images is a strong target.

### Per-source ceilings

Do not let `Open Images` dominate just because it is large.

Suggested initial maximum retained proportions:

- `Open Images`: `55%`
- `Places365`: `25%`
- `Mapillary`: `15%`
- `iNaturalist`: `15%`
- `TextOCR/COCO-Text`: `5%`

These percentages can overlap during candidate generation, but the final mixture should stay in this neighborhood.

### Per-bucket floors

Set minimum counts for strategically important buckets.

Example for a `4M` image build:

- `scene_landscape_architecture`: at least `700k`
- `street_signage_navigation`: at least `350k`
- `finegrained_nature`: at least `400k`
- `ocr_natural_scene`: at least `120k`
- `people_social_activity`: at least `300k`
- `indoor_lived_space`: at least `300k`

The exact values depend on what survives filtering, but the principle is important:

- scenes get protected
- OCR gets protected
- nature gets protected

### Selection algorithm

Recommended method:

1. partition by primary bucket
2. within each bucket, partition by source
3. rank by a combined keep score
4. apply local neighborhood suppression so a bucket does not fill with near-clones
5. fill bucket floors first
6. fill remaining budget by highest marginal diversity gain

The simplest usable greedy selection score is:

```text
keep_score =
  0.45 * quality_score +
  0.20 * novelty_score +
  0.20 * bucket_need_score +
  0.10 * source_need_score +
  0.05 * resolution_component
```

Where:

- `bucket_need_score` increases when a bucket is below quota
- `source_need_score` increases when a source is underrepresented relative to target

### Local neighborhood suppression

During final selection:

- after selecting an image, temporarily penalize very close neighbors in embedding space

This avoids filling the final set with multiple almost-identical images from dense clusters.

## Explicit Drop Categories

The pipeline should explicitly track these drop reasons:

- `decode_fail`
- `too_small`
- `extreme_aspect_ratio`
- `blank_or_low_variance`
- `corrupt_or_truncated`
- `exact_duplicate`
- `perceptual_duplicate`
- `embedding_duplicate`
- `document_like`
- `low_information`
- `low_quality_score`
- `over_bucket_quota`
- `over_source_quota`

This matters because you will want to know whether your dataset is getting smaller for good reasons or dumb reasons.

## OCR Handling Policy

OCR should be deliberately preserved without letting it take over.

### Positive OCR cases

Prefer keeping:

- storefronts
- street signs
- packaging
- scoreboards
- public notices in scenes
- shirts / labels / logos in natural photos
- dashboards / industrial labels in real scenes

### Negative OCR cases

Prefer dropping or heavily downweighting:

- scanned pages
- clean document photos
- slide screenshots
- pure UI screenshots
- meme text overlays

### OCR bonus

Suggested small additive term:

```text
ocr_bonus =
  +0.05 if 1 <= ocr_token_count <= 40 and ocr_area_fraction in a natural range
  +0.08 if text_present and bucket == street_signage_navigation
  -0.10 if document_like
```

This is intentionally small. OCR should influence ranking, not dominate it.

## Quality Control And Review

You should not trust the pipeline until you review random samples from each stage.

At minimum, generate human-review grids for:

- top-scoring retained images
- lowest-scoring retained images
- dropped-for-low-quality images
- dropped duplicates
- OCR bucket retained images
- scenery bucket retained images
- nature bucket retained images

Review at least `100-200` sampled images per critical bucket before freezing the first production corpus.

## Metrics To Track

Track these after every full pipeline run.

### Volume metrics

- total raw images
- total decodable images
- total retained after hard filters
- total retained after cheap dedup
- total retained after embedding dedup
- final selected total

### Quality metrics

- average quality score by source
- average quality score by bucket
- OCR prevalence by source
- mean neighbor similarity after final selection

### Diversity metrics

- source mix percentages
- bucket mix percentages
- indoor/outdoor split
- OCR presence rate
- embedding-cluster coverage
- fraction of final set from sparse neighborhoods

### Efficiency metrics

- average bytes per retained image
- total disk footprint of raw corpus
- total disk footprint of final corpus
- dropped-by-stage percentages

## Recommended First Production Settings

If I had to set a strong first-pass configuration now, I would use:

### Hard filters

- short side `>= 256`
- area `>= 256^2`
- aspect ratio within `[1/3.5, 3.5]`

### Dedup

- exact `sha256`
- `phash` Hamming `<= 4` auto-drop
- embedding duplicate threshold `>= 0.985` cosine auto-drop

### Final target size

- `4M` images

### Target source mix

- `50%` Open Images
- `20%` Places365
- `12%` Mapillary
- `13%` iNaturalist
- `5%` TextOCR / COCO-Text

### Protected strategic buckets

- scenes
- street/sign/navigation
- fine-grained nature
- OCR-natural-scene

## Implementation Sketch

One reasonable implementation stack:

- Python for orchestration
- `polars` or `pandas` for manifests
- `pyarrow` / `parquet` for storage
- `Pillow` or `opencv` for decode and basic image stats
- `faiss` for ANN search
- `torch` for embedding extraction
- `open_clip` or `transformers` for CLIP/SigLIP
- `paddleocr` or equivalent for OCR routing signals

Suggested script split:

```text
scripts/vm_ssl/
  ingest_sources.py
  validate_and_normalize.py
  compute_phash.py
  extract_embeddings.py
  run_ocr_signals.py
  score_images.py
  build_duplicate_clusters.py
  assign_buckets.py
  select_final_subset.py
  render_review_grids.py
  export_training_manifest.py
```

## Failure Modes To Watch For

### 1. Aesthetic collapse

Symptom:

- too many pretty landscapes and centered subjects

Fix:

- reduce global-score-only selection
- enforce bucket floors

### 2. Open Images dominance

Symptom:

- everything looks like generic web photography again

Fix:

- stronger source ceilings
- stronger scene and OCR bucket protection

### 3. OCR contamination

Symptom:

- too many documents, screenshots, or text-heavy junk

Fix:

- stronger `document_like` classifier
- lower OCR bucket cap

### 4. Over-deduplication

Symptom:

- dataset becomes too small and loses legitimate variation

Fix:

- raise duplicate thresholds
- dedup within buckets/sources before global suppression

### 5. Under-deduplication

Symptom:

- training set still contains obvious near-clones

Fix:

- add second embedding model
- increase local-neighborhood suppression during final selection

## Final Position

For this VM track, the right move is not "download everything and hope scale wins."

The right move is:

1. build a mixed-source corpus
2. filter aggressively for technical sanity
3. dedup in two stages
4. score for useful visual richness rather than aesthetics
5. explicitly protect scenery, OCR-natural-scene, and fine-grained nature
6. cap the final corpus to a size your disk and training budget can actually exploit

That should produce a much more compute-efficient foundation for a stronger bridge encoder than a raw, bloated, uncurated image pile.
