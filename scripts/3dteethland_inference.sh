#!/bin/bash
#SBATCH --job-name=3dteethland_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --mem=70GB
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

cd "$SLURM_SUBMIT_DIR/.."

bash run_main_3dteethland.sh