_base_ = ["../_base_/default_runtime.py"]  
  
# -----------------------------  
# Misc settings
# -----------------------------  
num_classes = 17 # 16 FDI Indices + Gum
ignore_index = -1

batch_size = 16
num_worker = 8 # for train
mix_prob = 0
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="SegLand",
    num_classes=17,  # 16 teeth + 1 background (based on your label mapping)
    segmentor_config=dict(
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
    ),  
    tooth_model_config=dict(
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
    ),
)  
# -----------------------------  
# Dataset Settings
# -----------------------------  
dataset_type = "IosDatasetTeeth3ds"
feat_keys = ["coord"]
grid_size = 0.01

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
        ignore_index = ignore_index,
        load_segment = False,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),  # Add this  
            dict(  
                type="GridSample",  
                grid_size=grid_size,  
                hash_type="fnv",  
                mode="train",  
                return_inverse=True,  # This enables upsampling  
            ),
        ],
        test_mode=True,
        preprocessing = '/homes/mlugli/BracketPrediction/3dteethland_preprocessing.yaml',
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
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)

test = dict(  
    type="SegLandTester",  
    verbose=False,  
)

# Optional: specify two checkpoints for SegLand submodules (config-only)
# Set these to strings with paths to the checkpoints you want to load for
# the segmentor and the tooth/landmark model respectively. Leave as None
# to skip per-submodule loading and rely on the regular `weight` key.
weight_segmentor = "../application/app_weights/segmentator_best.pth"
weight_tooth = "/homes/mlugli/BracketPrediction/exp_brackets/3dteethland_0/model/model_best.pth"