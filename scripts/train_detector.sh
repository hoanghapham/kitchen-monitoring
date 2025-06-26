#!/bin/bash -l

python scripts/train_yolo.py \
    --base-model yolo11m.pt \
    --data data/detection/dataset.yaml \
    --out-model-dir models/detector \
    --epochs 100 \
    --save true \
    --save-period 10 \
    --batch 10

