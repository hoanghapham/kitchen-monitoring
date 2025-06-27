#!/bin/bash -l

python scripts/train_yolo.py \
    --base-model yolo11m.pt \
    --data data/detection/dataset.yaml \
    --image-size 640 \
    --out-model-dir models/detector \
    --epochs 200 \
    --save true \
    --save-period 10 \
    --batch 6

