#!/bin/bash

# Activate conda
source pointcept-brackets-venv/bin/activate

python main.py \
    --samples /work/grana_maxillo/Mlugli/T4M/test.txt \
    --data-folder /work/grana_maxillo/Mlugli/T4M \
    --output-folder /homes/mlugli/BracketPrediction/Mlugli/3dteethland_evaluation \
    --seg-config  application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight  application/app_weights/segmentator_best.pth \
    --bond-config application/app_configs/Pt_landmarks_app.py \
    --bond-weight application/app_weights/heatmap_landmarks.pth \
    --preprocessing /homes/mlugli/BracketPrediction/3dteethland_preprocessing.yaml \
    --cache \
    --vis-seg \
    --save-ply \