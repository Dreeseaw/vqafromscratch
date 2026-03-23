# Compression-Tuning Pipeline Note

## Where It Sits

The new semantic bottleneck sits **after** the perceiver and **before** LM prefix injection:

```text
SigLIP grid tokens [B, 196, D]
-> frozen perceiver evidence latents [B, 49, D]
-> SemanticBottleneck [B, K, D]
-> LM prefix + existing LM visual adapters
```

This is intentionally **post-perceiver**, not pre-perceiver pruning. The perceiver still retrieves from the full dense visual grid. The bottleneck only compresses the LM-facing token export.

## Losses

The training path now supports:

- `L_vqa`: existing generative VQA cross-entropy
- `L_distill`: MSE between decoded bottleneck tokens and detached frozen-teacher perceiver evidence latents
- `L_ground`: masked KL divergence between aggregated perceiver cross-attention over the `14x14` visual grid and provided soft grounding targets

The old VQA-only path still works unchanged when `--use_compression=0` and `--use_grounding_loss=0`.

## Main Files Changed

- [bridge.py](/home/wdree/percy/vqafromscratch/models/bridge.py)
- [mm.py](/home/wdree/percy/vqafromscratch/train/mm.py)
- [vqa_data.py](/home/wdree/percy/vqafromscratch/train/vqa_data.py)
- [launch_compression_tuning_round1_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_compression_tuning_round1_v1.sh)
- [launch_compression_tuning_round2_v1.sh](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/scripts/launch_compression_tuning_round2_v1.sh)

## Example Command

```bash
./tasks/mm_bridge/scripts/launch_compression_tuning_round1_v1.sh
```

Round 2 uses the same pipeline with grounding enabled:

```bash
POINTING_INDEX_PATH=data/pointing/train_index.jsonl \
./tasks/mm_bridge/scripts/launch_compression_tuning_round2_v1.sh
```
