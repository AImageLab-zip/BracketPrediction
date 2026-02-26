"""
HeatmapRegressor model using PointTransformerv3 backbone (encoder + decoder).
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
    type="HeatmapRegressor",
    backbone=dict(
        type="PT-v2m2",
        in_channels=3,
        num_classes=64, # HeatmapRegressor takes care of final projection
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=16,
        enc_depths=(2, 6, 2),
        enc_channels=(96, 192, 384),
        enc_groups=(12, 24, 48),
        enc_neighbours=(16, 16, 16),
        dec_depths=(1, 1, 1),
        dec_channels=(48, 96, 192),
        dec_groups=(6, 12, 24),
        dec_neighbours=(16, 16, 16),
        grid_sizes=(0.1, 0.2, 0.4),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="interp",
    ),
    backbone_out_channels=64,
)
  
# -----------------------------  
# Optimizer & Scheduler
# -----------------------------  
epoch = 80
eval_epoch = 80
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
dataset_type = "BracketMapDataset"
data_root = "/work/grana_maxillo/Mlugli/BracketsV1"
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
            dict(type='RandomRotate', angle=[-0.1, 0.1], axis='x', p=0.5),
            dict(type='RandomRotate', angle=[-0.1, 0.1], axis='y', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomShift', shift=((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))),
            dict(
                type='GridSample',
                grid_size=grid_size,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=['coord', 'grid_coord', 'name', 'segment'],
                feat_keys=feat_keys)
        ],    
        test_mode=False
    ),
    
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[  
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),  
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
                keys=["coord", "grid_coord", "segment", "origin_segment", "inverse", "name"],
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
 
    test=dict(    
        type="BracketMapDataset",
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
            aug_transform=[
                # LOW
                [dict(type='RandomRotate', angle=[-0.1, 0.1], axis='z', p=0.0)]
                # MEDIUM
                #[dict(type='RandomRotate', angle=[-0.1, 0.1], axis='z', p=0.5)],
                #[dict(type='RandomRotate', angle=[-0.1, 0.1], axis='x', p=0.5)],
                #[dict(type='RandomRotate', angle=[-0.1, 0.1], axis='y', p=0.5)], 
                # HIGH
                #[dict(type='RandomScale', scale=[0.9, 1.1])],
                #[dict(type='RandomFlip', p=0.5)],
                #[dict(type='RandomShift', shift=((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)))]
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
