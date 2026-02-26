"""
Intra Oral Scan segmentator dataset.
"""

import os
import json
import numpy as np
from torch.utils.data import Dataset
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose
from pointcept.datasets.defaults import DefaultDataset
from pathlib import Path
import trimesh
from copy import deepcopy
from pointcept.datasets.transform import TRANSFORMS

@DATASETS.register_module()
class IosDataset(DefaultDataset):
    """
    Dataset for predicting bracket_point from STL files.
    """
 
    def __init__(
        self,
        data_root,
        fold=None,
        split="train",
        debug=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        load_segment=True,
        loop=1,
        ignore_index=0,
    ):
        self.fold = fold
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
            ignore_index = ignore_index,
        )
        self.debug = debug
        self.load_segment = load_segment
        self.fallbacks = 0 # counter for malformed stl files
        self.default_mapping = {
            48: 1, 47: 2, 46: 3,
            45: 4, 44: 5, 43: 6,
            42: 7, 41: 8, 31: 9,
            32: 10, 33: 11, 34: 12, 
            35: 13, 36: 14, 37: 15,
            38: 16,
        }
        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)  
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]
 
    def get_data_list(self):
        """Load list of data samples from fold JSON files, with path mapping."""
        # If fold is None and split is test, use all files in data_root
        if self.fold is None and self.split == "test":
            file_names = []
            for root, dirs, files in os.walk(self.data_root):
                for file in files:
                    if file.endswith('.stl'):
                        rel_path = os.path.relpath(os.path.join(root, file), self.data_root)
                        file_names.append(rel_path)
            print(f"Loaded {len(file_names)} samples from all files in {self.data_root}, split {self.split}")
            return file_names
        
        if not os.path.exists(self.fold):
            raise FileNotFoundError(
                f"Split file not found: {self.fold}\n"
                f"Please run the split generation script first to create split_{self.fold}.json"
            )
        with open(self.fold, 'r') as f:
            split_data = json.load(f)
        
        split_mapping = {
            'train': 'train',
            'val': 'validation',
            'test': 'test'
        }
        split_key = split_mapping.get(self.split)
        if split_key not in split_data:
            raise ValueError(f"Invalid split: {self.split}. Must be one of {list(split_mapping.keys())}")

        # Get file paths from the split and apply path mapping
        file_paths = split_data[split_key]['files']
        file_names = []
        for file_path in file_paths:
            if file_path.endswith('.stl'):                
                # Check if the STL file exists in data_root
                full_stl_path = os.path.join(self.data_root, file_path)
                if not os.path.exists(full_stl_path):
                    continue
                file_names.append(file_path)

        print(f"Loaded {len(file_names)} samples from fold {self.fold}, split {self.split}")
        print(f"  Patients: {len(split_data[split_key]['patient_ids'])}")
        
        return file_names
    
    def _load_stl(self, stl_path):
        try:
            # Try loading the primary file
            mesh = trimesh.load(stl_path, force='mesh')
            points = mesh.vertices
            normals = mesh.vertex_normals
            return points.astype(np.float32), normals.astype(np.float32)
        except:
            print(f"Couldn't load sample {stl_path}")
            raise
 

    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        segment = np.array(data['labels'], dtype=np.int32)
        segment[segment <= 28] += 20
        mapped_segment = np.zeros_like(segment)
        for original_label, mapped_label in self.default_mapping.items():
            mapped_segment[segment == original_label] = mapped_label
        return mapped_segment
    
    def get_data(self, idx, testing=False):  
        file_rel_path = self.data_list[idx % len(self.data_list)]  
        stl_path = os.path.join(self.data_root, file_rel_path)  
        json_path = os.path.join(self.data_root, file_rel_path.replace(".stl", ".json"))
 
        coord, normal = self._load_stl(stl_path)
        if self.load_segment:
            segment = self._load_json(json_path)
            assert segment.shape[0] == coord.shape[0], f"Segment shape not matching for sample {stl_path}"
        else:
            segment = np.zeros((coord.shape[0],), dtype=np.int32)
        d = {
            "coord": coord,
            "normal": normal,
            "name": Path(stl_path).stem,
            "full_path": stl_path,
            "segment": segment
        }
        return d
 
    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx, testing=True)
        # apply base transforms (same as training pipeline)
        data_dict = self.transform(data_dict)
 
        # ============Extract ground truth data==============
        result_dict = dict(
            segment=data_dict.pop("segment"), 
            name=data_dict.get("name"),
            full_path = data_dict.get("full_path")
        ) 
        # ==================================================

        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")
    
        # Create fragments with augmentations
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
    
        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:  
                    data_part = [data_part]
                fragment_list += data_part
        # here we can consider sampling just half of the fragments.
        # It should work anyway. 
        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict
 
    def __len__(self):
        if self.debug: return 2 # if debugging, run on just 2 samples
        return len(self.data_list) * self.loop