#!/bin/bash
set -e

# Activate conda
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Build the optional-flag list from environment variables (set in
# docker-compose.yml / the server's .env) so the container's behavior can be
# tuned without rebuilding the image. See README.md for the full list.
EXTRA_ARGS=()
[ "${CHECK_INTERVAL:-}" != "" ]  && EXTRA_ARGS+=(--check-interval "$CHECK_INTERVAL")
[ "${REMESH:-false}" = "true" ]  && EXTRA_ARGS+=(--remesh)
[ "${CACHE:-false}" = "true" ]   && EXTRA_ARGS+=(--cache)
[ "${SAVE_PLY:-false}" = "true" ] && EXTRA_ARGS+=(--save-ply)
[ "${VIS_SEG:-false}" = "true" ] && EXTRA_ARGS+=(--vis-seg)
[ "${WORKERS:-}" != "" ]         && EXTRA_ARGS+=(--workers "$WORKERS")
[ "${LANDMARKS:-}" != "" ]       && EXTRA_ARGS+=(--landmarks $LANDMARKS)
[ "${PREPROCESSING:-}" != "" ]   && EXTRA_ARGS+=(--preprocessing "$PREPROCESSING")

# Run the monitor
rm -f /tmp/.X*-lock /tmp/.X11-unix/X*
xvfb-run --auto-servernum -s "-screen 0 1920x1080x24" \
python /workspace/application/monitor.py \
    --data-root /workspace/application/data/ \
    --seg-config /workspace/application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight /workspace/application/weights/segmentator_best.pth \
    --bond-config /workspace/application/app_configs/Pt_landmarks_app.py \
    --bond-weight /workspace/application/weights/heatmap_landmarks.pth \
    "${EXTRA_ARGS[@]}"
