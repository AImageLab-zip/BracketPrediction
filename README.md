# Warning (WIP)
Main developement branch. For inference, use the `main.py` or `infer.py` scripts as your entrypoint.

# Input format
The model expects scans (`.stl`/`.obj`) oriented in the reference frame shown below: the occlusal plane normal aligned with `Z`, the mesio-distal axis aligned with `X`, and the arch rotated so the crowns point along `-Z` (i.e. the raw scan, typically captured with the crowns pointing up, must be rotated 180° around `Y`).

![Reference frame](assets/standard_frame.png)

# Model weights
Pretrained `--seg-weight` and `--bond-weight` checkpoints are available [here](https://drive.google.com/drive/folders/1dQYWACZfZUrg0hWHfTeNg7L60KiqFGL-?usp=sharing).

# Inference

## infer.py
Simple entrypoint for a single scan or a folder of scans (scanned recursively). Each scan's filename must contain `lower` or `upper`. Writes a per-scan segmentation mask (and, optionally, a landmarks point cloud / segmentation rendering) plus a combined `landmarks.json` aggregating every scan's predictions.

```bash
python infer.py \
    --input /path/to/scans --output /path/to/predictions \
    --seg-config  application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight  /path/to/seg_weight.pth \
    --bond-config application/app_configs/Pt_landmarks_app.py \
    --bond-weight /path/to/bond_weight.pth \
    --preprocessing 3dteethland_preprocessing.yaml \
    --save-ply
```

Add `--batch` to segment and bond every scan in one pass instead of one scan at a time (faster on large sets), `--vis-seg` to also save a rendering of the segmentation, `--workers N` to parallelize the CPU/IO-bound steps, and `--landmarks ...` to restrict prediction to specific landmark classes. Run `python infer.py --help` for the full list of options.

## main.py
Entrypoint used to run the model on the 3DTeethLand dataset layout (`lower`/`upper` subfolders keyed by patient id), matching predictions against `.txt` sample lists and optionally collecting ground-truth keypoints.

```bash
python main.py \
    --samples /path/to/lower.txt /path/to/upper.txt \
    --data-folder /path/to/dataset \
    --output-folder /path/to/output \
    --seg-config  application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight  /path/to/seg_weight.pth \
    --bond-config application/app_configs/Pt_landmarks_app.py \
    --bond-weight /path/to/bond_weight.pth \
    --preprocessing 3dteethland_preprocessing.yaml \
    --cache \
    --vis-seg \
    --save-ply
```

A ready-to-use version of this command is available in `run_main_3dteethland.sh`. Run `python main.py --help` for the full list of options.
