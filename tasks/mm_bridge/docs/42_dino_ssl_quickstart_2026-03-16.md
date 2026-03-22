# DINO SSL Quickstart

Files:
- `models/vit_ssl.py`
- `train/dino_ssl.py`
- `evals/ssl_knn.py`

What this gives you:
- small ViT backbone sized for future multimodal use
- DINO-style student/teacher SSL pretraining
- 2 global + local multi-crop augmentation
- EMA teacher, centering, sharpening
- AdamW + cosine LR and weight-decay schedules
- mixed precision, grad clipping, light drop path
- frozen-feature kNN evaluation

Backbone default:
- `image_size=224`
- `patch_size=16`
- `dim=192`
- `depth=12`
- `heads=3`
- `mlp_ratio=4`

Basic pretraining command:

```bash
./rundino.sh dino_vittiny_scratch \
  --data_dir images/train2014 \
  --epochs 100
```

Wrapper defaults:
- `--batch_size 128`
- `--num_workers 10`
- `--prefetch_factor 1`
- `--pin_memory`
- `--precision bf16`

Resume from the latest checkpoint for the same run:

```bash
./rundino.sh dino_vittiny_scratch
```

Resume from a specific checkpoint step:

```bash
./rundino.sh dino_vittiny_scratch 10000
```

Useful knobs:
- `--local_crops_number`
- `--lr`, `--lr_min`, `--warmup_epochs`
- `--weight_decay`, `--weight_decay_end`
- `--teacher_momentum`
- `--teacher_temp`, `--student_temp`
- `--drop_path`
- `--max_images` for tiny smoke runs

Minimal frozen-feature kNN eval on CIFAR-10:

```bash
python -m evals.ssl_knn \
  --checkpoint logs/dino_vittiny_scratch/step_10000.tar \
  --dataset cifar10 \
  --data_root data/ssl_eval \
  --device cuda
```

Notes:
- the backbone does not depend on a CLS token; downstream patch tokens remain primary
- the training head uses mean-pooled patch tokens for SSL supervision
- the teacher uses the same architecture as the student and is updated by EMA only
