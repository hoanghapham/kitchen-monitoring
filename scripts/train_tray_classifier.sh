#!/bin/bash -l

python scripts/train_yolo.py \
    --base-model yolo11m-cls.pt \
    --data data/classification/tray \
    --image-size 240 \
    --out-model-dir models/tray_classifier \
    --epochs 100 \
    --save true \
    --save-period 10 \
    --batch 10

