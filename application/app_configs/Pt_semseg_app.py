"""
Configuration to train a IOS semantic segmentator
"""
_base_ = ["default_runtime.py"]
  
# -----------------------------  
# Misc settings
# -----------------------------  
num_classes = 17 # 16 FDI Indices + Gum
ignore_index = -1
num_worker = 8 # for train
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
    type="DefaultSegmentorV2",
    num_classes = num_classes,
    backbone_out_channels = 64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
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
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
    ],
)
# -----------------------------
# Dataset settings
# -----------------------------
dataset_type = "IosDataset"
data_root =  "" # defined in the main program arguments
feat_keys = ["coord"]
grid_size = 0.01
fold = None # not used if running as application
 
data = dict(
    num_classes = num_classes,
    ignore_index = ignore_index,
    names = [
        "Gum", "48-28","47-27", "46-26", "45-25", "44-24", "43-23", "42-22", "41-21",
        "31-11", "32-12", "33-13", "34-14", "35-15", "36-16", "37-17", "38-18"
    ],
    train = dict(),
    val = dict(),
    test=dict(
        type=dataset_type,
        split="test",
        fold=fold,
        data_root=data_root,
        ignore_index = ignore_index,
        load_segment = False,
        transform=[
            dict(type="NormalizeCoord"),
        ],
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
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=feat_keys),
            ],  
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)
   
# -----------------------------
# Hooks
# -----------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
test = dict(type="SemSegTester")