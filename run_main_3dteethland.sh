#!/bin/bash

# Activate conda
source .venv/bin/activate
python main.py \
    --samples /work/grana_maxillo/Mlugli/temp/lower.txt /work/grana_maxillo/Mlugli/temp/upper.txt \
    --data-folder /homes/mlugli/BracketPrediction/Mlugli/temp/paper \
    --output-folder /homes/mlugli/BracketPrediction/Mlugli/paper_plots \
    --seg-config  application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight  application/app_weights/segmentator_best.pth \
    --bond-config application/app_configs/Pt_landmarks_app.py \
    --bond-weight /homes/mlugli/BracketPrediction/Mlugli/models/AutoBonding/heatmap_landmarks.pth \
    --preprocessing /homes/mlugli/BracketPrediction/preprocessing/3dteethland_preprocessing.yaml \
    --cache \
    --vis-seg \
    --save-ply

# /homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_debug/testing_lower.txt
# /homes/mlugli/BracketPrediction/Mlugli/temp
#--preprocessing /homes/mlugli/BracketPrediction/preprocessing/3dteethland_preprocessing.yaml \
#/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_debug/testing_lower.txt
# /homes/mlugli/BracketPrediction/exp_brackets/3dteethland_0/model/model_best.pth
# /homes/mlugli/BracketPrediction/exp_brackets/3dteethland_1_softdice/model/model_best.pth 
#--bond-config application/app_configs/Pt_landmarks_app.py \
#/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_test_split_original/testing_lower.txt
#--data-folder /work/grana_maxillo/Mlugli/T4M \
# --bond-weight /homes/mlugli/BracketPrediction/exp_brackets/3dteethland_0/model/model_best.pth \
