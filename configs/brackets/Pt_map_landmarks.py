"""
Configuration file for training/testing on BracketsV2 using
the standard heatmap model.
"""

_base_ = ["../_base_/default_runtime.py"]

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
    loss_type = "softdice",
)

# -----------------------------  
# Optimizer & Scheduler
# -----------------------------  
epoch = 200
eval_epoch = 200
clip_grad = 1.0

optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.10,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# -----------------------------  
# Dataset settings
# -----------------------------    
dataset_type = "BracketsV2"
data_root = "/work/grana_maxillo/Mlugli/BracketsV2"
feat_keys = ["coord"]
grid_size = 0.005

data = dict(
    train=dict(
        type=dataset_type,
        split="train", 
        debug=False,
        data_root=data_root,
        transform=[
            dict(type='RandomRotate', angle=[-0.1, 0.1], axis='z', p=0.5),
            dict(type='RandomRotate', angle=[1.1, 0.1], axis='x', p=0.5),
            dict(type='RandomRotate', angle=[-0.1, 0.1], axis='y', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            #dict(type='RandomFlip', p=0.5),
            dict(type='RandomShift', shift=((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))),
            dict(
                type='Update',
                keys_dict={"index_valid_keys": ["coord", "normal", "segment", "validity_mask"]}
            ),
            dict(
                type='GridSample',
                grid_size=grid_size,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=['coord', 'grid_coord', 'name', 'segment', 'validity_mask'],
                feat_keys=feat_keys)
        ],    
        test_mode=False
    ),
    
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[  
            dict(type="Copy", keys_dict={"segment": "origin_segment",
                                         "validity_mask": "origin_validity_mask"}),
            dict(
                type='Update',
                keys_dict={"index_valid_keys": ["coord", "normal", "segment", "validity_mask"]}
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord",
                      "grid_coord", 
                      "segment", 
                      "origin_segment", 
                      "inverse", 
                      "name", 
                      "validity_mask",
                      "origin_validity_mask"],
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
 
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        test_mode=True,
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
    verbose=True
)
