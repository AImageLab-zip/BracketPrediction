#!/bin/bash
set -e

# Activate conda
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Run the monitor
rm -f /tmp/.X*-lock /tmp/.X11-unix/X*
xvfb-run --auto-servernum -s "-screen 0 1920x1080x24" \
python /workspace/application/monitor.py \
    --data-root /workspace/application/data/ \
    --seg-config /workspace/application/app_configs/Pt_semseg_app.py \
    --seg-weight /workspace/application/weights/segmentator_best.pth \
    --bond-config /workspace/application/app_configs/Pt_landmarks_app.py \
    --bond-weight /workspace/application/weights/heatmap_landmarks.pth \
    --check-interval 5 \
