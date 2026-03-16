# Sports Imbalance Audit

Inputs:
- `tasks/mm_bridge/scripts/analyze_sports_bias.py`
- `tasks/mm_bridge/docs/sports_bias_audit_2026-03-15.json`
- LM corpus: `data/wiki_coco/articles.jsonl`, `data/pretraining/wikicoco256_cleaned/clean_stats.jsonl`
- Distill mix: `data/distill/q1/raw.jsonl`
- VQAv2 val annotations/questions
- Representative mm_bridge run logs for original VM, MobileViT, MobileCLIP, and DINOv2-S

## Bottom line

Sports are present in the LM pretraining corpus, but they do **not** look dominant enough to explain the VQA behavior by themselves.

- Conservative estimate, using sports-signaling page titles only:
  - `2.63%` of wiki docs
  - `3.25%` of cleaned characters
  - `2.65%` of sampled training windows
- Broader estimate, allowing strong sports signals in the page intro:
  - `9.19%` of wiki docs
  - `9.22%` of sampled training windows

So the answer is not "sports infected the whole corpus." The answer is closer to:

1. there is a real sports pocket in the data,
2. it is noticeable enough to show up during crawling,
3. but it is still a minority slice of the LM corpus, especially after the `max_windows_per_doc=4` cap dampens giant season-history pages.

## What the corpus actually looks like

The strongest conservative title matches were things like:

- `College football`
- `Super Bowl commercials`
- `European Cup and UEFA Champions League records and statistics`
- `2022 FIFA World Cup`
- `Glossary of baseball terms`

This fits the intuition that the crawl pulled in sports reference/history material. But the important damping effect is the pretokenizer:

- every doc contributes at most `4` sampled windows
- so a huge `NFL season` or `Formula One championship` page cannot flood training in proportion to its raw length

That is visible in the numbers:

- sports are `3.38%` of raw words under the conservative estimate
- but only `2.65%` of sampled training windows

The distill mix follows the same pattern because it is sourced from the same wiki corpus:

- conservative sports share in distill examples: `2.72%`
- broad sports share in distill examples: `9.58%`

## Why `what sport is` is still so strong

On VQAv2 val, `what sport is` is not especially large:

- support: `1086` questions
- support rank: `44`

But it is relatively low-entropy compared with many open-ended question types:

- only `52` distinct majority answers
- top answer `tennis` already covers `30.76%`
- a dumb constant predictor of `tennis` gets `31.25%` official VQA accuracy on this question type

For comparison:

- `what room is` is even easier by answer concentration
- `what animal is` is much harder by answer spread

So sports are partly a **naturally easy classification problem** in VQA: visually distinctive scenes, a modest answer vocabulary, and a strong prior.

## VM effect is larger than corpus effect

The model-side story is stronger than the corpus-skew story.

Representative full-val results:

- Original VM frontier (`Nail`): overall `0.4653`
  - `what sport is` did **not** make the final top-10 question-type list
- MobileViT comparable run: overall `0.5240`
  - `what sport is = 0.831`
- MobileCLIP comparable run: overall `0.5603`
  - `what sport is = 0.877`
- DINOv2-S comparable run: overall `0.5323`
  - `what sport is = 0.880`
- DINOv2-S frontier (`nodynbudget`): overall `0.5762`
  - `what sport is = 0.921`

Interpretation:

- if LM sports skew were the main driver, I would expect the original-VM branch to already show `what sport is` as a standout
- instead, it only becomes a major standout once the VMs get materially better

That points to:

1. sports being an easy/evaluable VQA subtype,
2. stronger VMs extracting the relevant evidence much more cleanly,
3. LM pretraining sports exposure acting at most as a secondary amplifier rather than the primary cause

## Best current read

My best read is:

- **No**, sports did not infect the LM corpus at a high enough level to be the main explanation.
- **Yes**, there is a real sports/reference-history pocket in the wiki data, and your memory of repeated sports pages was not imaginary.
- **But** the dominant reason `what sport is` pops is that it is a comparatively easy VQA subtype, and that ease is being unlocked much more strongly by better VM sources than by LM corpus skew.

