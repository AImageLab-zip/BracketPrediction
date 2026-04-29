#!/bin/bash

# Activate conda
source pointcept-brackets-venv/bin/activate

python main.py \
    --seg-config  application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight  application/app_weights/segmentator_best.pth \
    --bond-config application/app_configs/Pt_landmarks_app.py \
    --bond-weight application/app_weights/heatmap_landmarks.pth \
    --teethland \