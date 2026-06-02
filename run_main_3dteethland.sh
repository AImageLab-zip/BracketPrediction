#!/bin/bash

# Activate conda
source pointcept-brackets-venv/bin/activate

python main.py \
    --samples /homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_test_split_original/testing_lower.txt /homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_test_split_original/testing_upper.txt \
    --data-folder /homes/mlugli/BracketPrediction/Teeth3DS/original_data \
    --output-folder /homes/mlugli/BracketPrediction/Mlugli/3dteethland_evaluation \
    --seg-config  application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight  application/app_weights/segmentator_best.pth \
    --bond-config application/app_configs/Pt_landmarks_app.py \
    --bond-weight /homes/mlugli/BracketPrediction/exp_brackets/3dteethland_0/model/model_best.pth \
    --preprocessing /homes/mlugli/BracketPrediction/3dteethland_preprocessing.yaml \
    --cache \
    --collect-gt \


# /homes/mlugli/BracketPrediction/exp_brackets/3dteethland_0/model/model_best.pth
# /homes/mlugli/BracketPrediction/exp_brackets/3dteethland_1_softdice/model/model_best.pth 
#--bond-config application/app_configs/Pt_landmarks_app.py \
#/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_test_split_original/testing_lower.txt
#--data-folder /work/grana_maxillo/Mlugli/T4M \