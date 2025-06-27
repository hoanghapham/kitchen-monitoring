#!/bin/bash -l

python scripts/train_yolo.py \
    --base-model yolo11m-cls.pt \
    --data data/classification/dish \
    --image-size 240 \
    --out-model-dir models/dish_classifier \
    --epochs 100 \
    --save true \
    --save-period 10 \
    --batch 10

