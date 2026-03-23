# Semantic Adapter Fragility Experiment Plan

## Goal

Explain *where* the `K=8` semantic-compression run became unusually dependent on LM visual adapters.

This is an analysis-only experiment:
- no new training
- no new architecture
- no new learned modules

The question is narrow:

`which answer categories, and which GQA reasoning types, become most fragile when adapter support is removed under stronger compression?`

## Fixed Checkpoints

- Cement anchor: `logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar`
- Best compressed run: `logs/mmsemantic_v1_20260322_k32/step_4000.tar`
- Lower-budget frontier: `logs/mmsemantic_v1_20260322_k8/step_4000.tar`

These are the same three checkpoints already used in the completed semantic diagnostics.

## Core Deliverable

Primary table:

| Model | Category | Keep 3 | Keep 2 | Keep 1 | Keep 0 | Delta 3->0 |
|---|---|---:|---:|---:|---:|---:|
| Anchor | Yes/No |  |  |  |  |  |
| Anchor | Number |  |  |  |  |  |
| Anchor | Other |  |  |  |  |  |
| K=32 | Yes/No |  |  |  |  |  |
| K=32 | Number |  |  |  |  |  |
| K=32 | Other |  |  |  |  |  |
| K=8 | Yes/No |  |  |  |  |  |
| K=8 | Number |  |  |  |  |  |
| K=8 | Other |  |  |  |  |  |

Secondary GQA table:

| Model | GQA Group | Keep 3 | Keep 0 | Delta |
|---|---|---:|---:|---:|
| Anchor | Spatial |  |  |  |
| Anchor | Attribute |  |  |  |
| Anchor | Count |  |  |  |
| Anchor | Exist |  |  |  |
| K=8 | Spatial |  |  |  |
| K=8 | Attribute |  |  |  |
| K=8 | Count |  |  |  |
| K=8 | Exist |  |  |  |

Derived metrics:
- fragility ratio per VQAv2 answer category
- `K=8 probe - K=8 keep0` per category

## Experimental Read

The existing diagnostics already established:
- `K=8` keeps almost the same full-system score as the anchor
- `K=8` is more linearly decodable than the anchor
- `K=8` becomes much more fragile when adapters are removed

What is still missing is the cross:

`category x keep_count x compression level`

That cross is the actual explanation experiment.

## Execution Plan

### 1. VQAv2 adapter-ablation grid

Run keep-count ablation on:
- anchor
- `K=32`
- `K=8`

Keep counts:
- `3`
- `2`
- `1`
- `0`

Eval split:
- same VQAv2 val path used by the semantic diagnostics

Logged outputs:
- overall accuracy
- answer-type accuracy
- question-type accuracy
- heuristic question-category accuracy

### 2. GQA subset ablation

Run only keep-`3` and keep-`0` on:
- anchor
- `K=8`

GQA val groups:
- `spatial`
- `attribute`
- `count`
- `exist`

This is intentionally a lighter pass. The goal is not a full GQA benchmark; it is to see whether compression fragility concentrates on a compositional question family.

Local data note:
- the available `val_all_questions.json` slice appears to have very sparse `count` coverage on this machine
- so `count` should be treated as low-sample / qualitative unless a richer GQA split is introduced later

### 3. Derived analysis

Compute:
- `delta_3_to_0` per VQAv2 answer category
- `fragility_ratio = delta_K / delta_anchor`
- `probe_ablation_gap = K8_probe - K8_keep0`

Interpretation targets:
- if `other` dominates, the likely failure mode is compositional diversity
- if `number` dominates, the likely failure mode is entity merging / count collapse
- if ratios are uniform, compression is shifting reasoning load onto the LM globally rather than breaking one specific evidence type

## Implemented Support

Files changed for this experiment:
- [mm_adapter_ablation.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/mm_adapter_ablation.py)
- [vqa_data.py](/home/wdree/percy/vqafromscratch/train/vqa_data.py)
- [mm.py](/home/wdree/percy/vqafromscratch/train/mm.py)

New utilities:
- [analyze_semantic_adapter_fragility.py](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/analyze_semantic_adapter_fragility.py)
- [launch_semantic_adapter_fragility_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_semantic_adapter_fragility_v1.sh)

New support added:
- per-ablation output now retains:
  - `answer_type_accuracy`
  - `question_type_accuracy`
  - `heuristic_category_accuracy`
- GQA eval now supports:
  - `gqa_val`
  - coarse filter groups: `spatial`, `attribute`, `count`, `exist`

## Commands

Main experiment launcher:

```bash
./tasks/mm_bridge/scripts/launch_semantic_adapter_fragility_v1.sh
```

Useful quick-pass override:

```bash
LIMIT_EVAL=5000 GQA_LIMIT_EVAL=500 ./tasks/mm_bridge/scripts/launch_semantic_adapter_fragility_v1.sh
```

## Success Condition

The experiment is successful if it answers one question cleanly:

`why does K=8 need the LM adapters so much more than the anchor?`

That answer can be category-specific or uniform. Either is useful. The important thing is to replace the current aggregate statement:

`K=8 is more adapter-dependent`

with a concrete mechanistic one:

`K=8 is more adapter-dependent mainly because it loses <category/type> reasoning when the adapters are removed`
