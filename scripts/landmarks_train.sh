#!/bin/bash
#SBATCH --job-name=autobonding
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --mem=70GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create necessary directories

mkdir -p logs exp

# Activate your environment
source /homes/mlugli/BracketPrediction/pointcept-brackets-venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=./

# Configuration
CONFIG="configs/brackets/Pt_map_landmarks.py"
EXP_NAME="landmarks_f0_normalized"
NUM_GPU=1
FOLD=0

# Training command
python tools/train.py \
--config-file ${CONFIG} \
--num-gpus ${NUM_GPU} \
--options \
    save_path=exp_brackets/${EXP_NAME} \
    data.train.fold=${FOLD} \
    data.val.fold=${FOLD} \
    data.test.fold=${FOLD}
