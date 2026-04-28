"""    
Configuration file for Bracket Point Prediction with Voxel-based Backbone  
"""
_base_ = ["default_runtime.py"]
# -----------------------------  
# Misc settings
# -----------------------------  
batch_size = 16
num_worker = 4
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
    type="HeatmapRegressor",    
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
dataset_type = "BracketMapDataset"
data_root = "/work/grana_maxillo/Mlugli/BracketsHeatmaps"
feat_keys = ["coord"]
grid_size = 0.005
production = True

data = dict(
    train=dict(),
    
    val=dict(),
 
    test=dict(    
        type="BracketMapDataset",
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
            aug_transform=[
                # test time augmentations
                [dict(type='RandomRotate', angle=[0.0, 0.0], axis='z', p=0.0)],
                #[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],  
                #[dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],  
                #[dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],  
                #[dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)],
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
    dict(type="HeatmapEvaluator"),  
    dict(type="CheckpointSaver", save_freq=None),    
]  
  
test = dict(  
    type="HeatmapTester",  
    verbose=True  
)
