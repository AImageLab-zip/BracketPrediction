"""
Simple point regressor with PointTransformerV3 backbone (encoder only).
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
# Make sure to use cls_mode=True so that the 
# decoding part of the backbone architecture (producing
# the segmentation) is not used.
# -----------------------------

model = dict(
    type="LandmarkPredictor",
    backbone=dict(
        type="PT-v3m1", 
        in_channels = 3,
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
dataset_type = "BracketPointDataset"  
data_root = "/work/grana_maxillo/Mlugli/BracketsV1"  
feat_keys = ["coord"]
index_valid_keys = ["coord", "color", "normal", "superpoint", "strength", "segment", "instance", "orientation"]
grid_size = 0.005

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        debug=False,
        data_root=data_root,
        transform=[
            dict(type='CustomRandomRotate', angle=[-0.1, 0.1], axis='x', p=0.5),
            dict(type='CustomRandomRotate', angle=[-0.1, 0.1], axis='y', p=0.5),
            dict(type='CustomRandomRotate', angle=[-0.1, 0.1], axis='z', p=0.5),
            dict(type='CustomRandomScale', scale=[0.9, 1.1]),
            dict(type='CustomRandomFlip', p=0.5),
            dict(
                type='CustomRandomShift',
                shift=((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))),
            dict(
                type="Update",  
                keys_dict={"index_valid_keys": index_valid_keys}  
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
                keys=['coord', 'grid_coord', 'bracket', 'incisal', 'outer', 'name'],
                feat_keys=feat_keys
            )
        ],  
        test_mode=False
    ),
 
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="Update",  
                keys_dict={"index_valid_keys": index_valid_keys}  
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,  # This generates grid_coord
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "grid_coord", "bracket", "incisal", "outer", "name"],
                feat_keys=feat_keys,  
            ),
        ],
        test_mode=False,  
    ),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        test_mode=True,
        transform = [
            dict(
                type="Update",  
                keys_dict={"index_valid_keys": index_valid_keys}  
            ),
        ],
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
# Default ones + DisplacementEvaluator that
# computes the loss of the regression task.
# -----------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),  
    dict(type="InformationWriter"),  
    dict(type="DisplacementEvaluator"), 
    dict(type="CheckpointSaver", save_freq=None),  
]
test = dict(type="BracketTester_v2")
