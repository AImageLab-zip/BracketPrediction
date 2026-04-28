"""  
Configuration file for Bracket Point Prediction with Voxel-based Backbone
"""  
  
_base_ = ["default_runtime.py"]
# -----------------------------
# Misc settings
# -----------------------------
num_worker = 4
mix_prob = 0
empty_cache = False
enable_amp = True

# -----------------------------
# Model setting
# -----------------------------
model = dict(
    type="LandmarkPredictor",
    backbone=dict(
        type="PT-v3m1", 
        in_channels=3,
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        dec_channels=(32, 32, 64, 96),
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_num_head=(2, 2, 4, 6),
        enable_flash=False,
        cls_mode=True,
    ),
    backbone_out_channels=512,
    output_dim=3,
)
# -----------------------------
# Dataset settings
# -----------------------------  
dataset_type = "BracketPointDataset"  
data_root = ""  # defined in main program
feat_keys = ["coord"]
grid_size = 0.005
data = dict(
    train=dict(),
    val=dict(),
    # inference
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv", 
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "name"),
                    feat_keys=feat_keys,
                ),
            ],
            aug_transform=[
                [dict(type='CustomRandomRotate', angle=[-0.1, 0.1], axis='x', p=0.5)],
                [dict(type='CustomRandomRotate', angle=[-0.1, 0.1], axis='y', p=0.5)],
                [dict(type='CustomRandomRotate', angle=[-0.1, 0.1], axis='z', p=0.5)],
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
    dict(type="DisplacementEvaluator"), 
    dict(type="CheckpointSaver", save_freq=None),  
]
test = dict(type="BracketTester_v2")