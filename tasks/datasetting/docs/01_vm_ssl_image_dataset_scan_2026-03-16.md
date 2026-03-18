# VM SSL Image Dataset Scan

## Goal

Pick a high-value image mix for a DINOv2-style ViT pretraining pipeline that is meant to feed the `mm_bridge` direction later.

The desired mix is not just "more images." It should cover:

- broad natural-image diversity
- strong scene/scenery coverage
- OCR-relevant visual text in natural images
- long-tail natural-world detail
- enough distributional spread to make the encoder feel less narrow than the current VM

I also downloaded and spot-checked small sample images from several candidate sets before writing this up.

## Short Answer

If the goal is a serious but still sane first-pass disk download, I would pull this mix:

1. `Open Images V7`
2. `Places365`
3. `Mapillary Vistas`
4. `iNaturalist 2021`
5. `TextOCR`

If `TextOCR` is inconvenient to fetch immediately, use `COCO-Text` as the temporary OCR add-on and replace it later.

## Recommended Roles

### 1. Open Images V7

**Recommendation:** download

This should be the general-purpose base of the mix.

Why it is worth disk:

- very broad object, people, indoor, outdoor, consumer-photo, and web-photo coverage
- much more heterogeneous than a narrow benchmark-style dataset
- good base fuel for generic visual invariances in SSL
- keeps the encoder from collapsing toward only scenes, only OCR, or only nature

Operational note:

- this is a large, URL-based web image corpus, so acquisition is a little annoying
- worth it anyway if you want a strong "general world" component

### 2. Places365

**Recommendation:** download

This is the cleanest scenery/scene-centric addition.

Why it is worth disk:

- explicitly scene-focused rather than object-focused
- useful for landscape, architecture, room-type, and environmental understanding
- fills a real gap that VQA/caption-heavy mixtures usually leave underdeveloped
- especially relevant if you want the VM to stop behaving like a weak object crop encoder

This is probably the highest-value scenery-specific download in the whole list.

### 3. Mapillary Vistas

**Recommendation:** download

This is the best "natural OCR plus spatial layout plus outdoor geometry" dataset in the mix.

Why it is worth disk:

- diverse street-level imagery with signage, road markings, storefront text, vehicles, lane structure, weather, and geography
- very useful for learning text-bearing natural scenes without drifting into scanned-document land
- gives you OCR-adjacent supervision pressure even in image-only SSL, because text and symbols are embedded in real environments
- strong for spatial reasoning, clutter, depth cues, and real-world compositional messiness

Important caveat:

- check the license carefully before treating it as a long-term foundation set
- this is a high-value research download even if you later decide to replace it for commercialization reasons

### 4. iNaturalist 2021

**Recommendation:** download

This is the best long-tail natural-world add-on.

Why it is worth disk:

- excellent for fine-grained texture, shape, species, and background variation
- pushes the encoder to care about subtle visual distinctions instead of only coarse scene/object categories
- adds outdoor imagery that is visually rich but quite different from street scenes and consumer photos
- likely helpful if your current VM is weak on dense visual detail

This should not be the base dataset, but it is a very strong diversity supplement.

### 5. TextOCR

**Recommendation:** download, but as a top-off rather than a core bulk source

Why it is worth disk:

- natural-image scene text, not clean scanned documents
- better aligned with downstream VLM behavior than "document OCR only" corpora
- useful for signs, labels, packaging, scoreboards, shirts, storefronts, and other embedded text
- small enough to manage easily

Important usage note:

- do not overweight it
- this should be a small biasing component, not the backbone of the SSL corpus

## Temporary OCR Substitute

### COCO-Text

**Recommendation:** acceptable stopgap; not the final preferred OCR source

Why it is still useful:

- easy to layer on top of general natural-image data
- scene text is embedded in ordinary photos
- low-risk supplemental download

Why I prefer `TextOCR` long-term:

- `COCO-Text` is older and smaller
- a lot of its text is sparse or incidental
- it feels more like a useful patch than a complete OCR-oriented top-off

## Inspected Samples

I spot-checked small sample downloads from accessible mirrors to avoid purely metadata-driven recommendations.

### Open Images sample

Observed:

- casual human social scene
- low-studio, real-web-photo look
- useful reminder that Open Images brings broad everyday visual messiness rather than benchmark neatness

Takeaway:

- good base-distribution material

### Mapillary sample

Observed:

- road intersection with lane markings, vehicles, traffic lights, sign structures, and outdoor geometry

Takeaway:

- very strong fit for real-world spatial layout and OCR-adjacent vision

### COCO-Text sample

Observed:

- ordinary sports scene with weak but real incidental text in the image

Takeaway:

- confirms the basic value of natural-image OCR data
- also confirms why it should stay a supplement rather than the whole OCR plan

### ADE20K sample

Observed:

- scene-centric images like cathedral interior and archive/library shelving

Takeaway:

- scene datasets really do give a different visual distribution than generic object/photo corpora
- but `ADE20K` is too small to be the main scenery download; it is more of a sanity-check or auxiliary set than a backbone source

## What I Would Actually Download First

### Tier 1: immediate

- `Open Images V7`
- `Places365`
- `TextOCR`

This is the minimum good mix.

What it gives you:

- broad world imagery
- strong scene/scenery pressure
- explicit natural-image text coverage

### Tier 2: add as soon as feasible

- `Mapillary Vistas`
- `iNaturalist 2021`

What this adds:

- road/sign/symbol/layout richness
- long-tail biological and natural-world detail
- harder visual discrimination than the base mix alone

## What I Would Not Center the Pipeline Around

### LAION-style giant web scrapes

**Recommendation:** defer

Why:

- huge scale, but also huge noise
- more duplication, aesthetics bias, meme/junk risk, and license ambiguity
- likely too much cleanup burden for the current stage

### Document OCR datasets

**Recommendation:** do not make them a major ingredient

Why:

- too far from the natural-image OCR you actually want
- can bias the model toward flat page-like text instead of embedded scene text

### ADE20K as a main pretraining source

**Recommendation:** do not prioritize for bulk download

Why:

- useful scene bias
- not enough scale to justify making it a major disk commitment compared with `Places365`

## Suggested Mixture Shape

For a first serious SSL run, I would aim for something roughly like:

- `45-55%` Open Images
- `20-25%` Places365
- `10-15%` iNaturalist 2021
- `10-15%` Mapillary Vistas
- `3-5%` TextOCR / COCO-Text

That is not meant as a law, just a sane starting prior.

The key point is:

- keep the corpus mostly broad and natural
- add explicit scene pressure
- add a smaller but deliberate OCR component
- add long-tail fine-grained nature imagery so the VM has to care about subtle detail

## Final Recommendation

If you want the most defensible first disk plan for this new VM track, I would download:

1. `Open Images V7` as the base corpus
2. `Places365` as the scenery/scenes anchor
3. `Mapillary Vistas` as the street/OCR/layout booster
4. `iNaturalist 2021` as the fine-grained natural-world booster
5. `TextOCR` as the OCR top-off

That mix is much better targeted to your current need than just throwing more VQA/caption data at the problem.

## Source Links

- Open Images: `https://storage.googleapis.com/openimages/web/index.html`
- Open Images paper: `https://arxiv.org/abs/1811.00982`
- Places365 / Places2: `https://places2.csail.mit.edu/`
- Mapillary research: `https://research.mapillary.com/`
- iNaturalist 2021 challenge repo: `https://github.com/visipedia/inat_comp/tree/master/2021`
- TextOCR: `https://textvqa.org/textocr/`
- TextOCR paper: `https://arxiv.org/abs/2003.12462`
- COCO-Text: `https://bgshih.github.io/cocotext/`
