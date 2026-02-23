#!/bin/bash
#SBATCH --job-name=IOS_segmentator_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create necessary directories

mkdir -p logs exp

# Activate your environment
source /homes/mlugli/BracketPrediction/pointcept-brackets-venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=./

# Configuration
CONFIG="configs/brackets/Pt_semseg.py"
EXP_NAME="Pt_semseg_train_aug"
NUM_GPU=1

# Training command
python tools/train.py --config-file ${CONFIG} --num-gpus ${NUM_GPU} --options save_path=exp/semseg/${EXP_NAME}
