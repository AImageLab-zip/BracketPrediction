"""
Configuration file for training/testing on BracketsV2 using
the standard heatmap model.
"""

_base_ = ["default_runtime.py"]
# -----------------------------  
# Misc settings
# -----------------------------  
batch_size = 1
num_worker = 0
mix_prob = 0
empty_cache = False
enable_amp = True
  
# -----------------------------
# Wandb settings
# -----------------------------
enable_wandb = True
wandb_project = "bracket_point_prediction"

# -----------------------------
# Model settings
# ----------------------------- 
model = dict(    
    type="LandmarksHeatmapV2",    
    backbone=dict(    
        type="PT-v3m1",    
        in_channels=3,  # xyz only  
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        mlp_ratio=4,
        enable_flash=False,
        cls_mode=False,
    ),    
    backbone_out_channels=64,
)
  
# -----------------------------  
# Dataset settings
# -----------------------------    
dataset_type = "BracketsV2"
data_root = "/work/grana_maxillo/Mlugli/BracketsV2"
feat_keys = ["coord"]
grid_size = 0.005

data = dict( 
    train = dict(),
    val = dict(),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        test_mode=True,
        production=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse = True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index", "inverse"), feat_keys=feat_keys),
            ],
            aug_transform=[ # no TTA
                [dict(type='RandomRotate', angle=[0.0, 0.0], axis='z', p=0.0)]
            ],
        ),
    ),
)  
 
# -----------------------------
# Hooks
# -----------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="HeatmapEvaluatorV2"),
    dict(type="CheckpointSaver", save_freq=None),
]
 
test = dict(
    type="HeatmapTesterV2",
    percentile=97,
    verbose=True
)