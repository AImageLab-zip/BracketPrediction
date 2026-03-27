"""
Intra Oral Scan segmentator dataset.
Author: Matteo Lugli (283122@studenti.unimore.it)
Please cite our work if the code is helpful to you.
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
class IosDatasetTeeth3ds(DefaultDataset):
    """
    Dataset for predicting bracket_point from OBJ files.
    """
 
    def __init__(
        self,
        data_root, # path to normalized_data folder containing lower/ and upper/ subfolders
        fold=None, # path to directory containing split files (training_lower.txt, validation_lower.txt, etc.)
        split="train",
        transform=None,
        test_mode=False,
        test_cfg=None,
        load_segment=True,
        loop=1,
        ignore_index=0,
        debug=False,
    ):
        self.fold = fold
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
            ignore_index=ignore_index,
        )
        self.debug = debug
        self.load_segment = load_segment
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
        """Load list of data samples from fold text files."""
        if self.fold is None:
            raise ValueError("fold parameter must be provided (path to directory with split files)")
        
        fold_dir = Path(self.fold)
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {self.fold}")
        
        # Map split to file names
        split_file_mapping = {
            'train': ['training_lower.txt', 'training_upper.txt'],
            'val': ['validation_lower.txt', 'validation_upper.txt'],
            'test': ['testing_lower.txt', 'testing_upper.txt']
        }
        
        #if self.split not in split_file_mapping:
        #    raise ValueError(f"Invalid split: {self.split}. Must be one of {list(split_file_mapping.keys())}")
        # you can specify multiple splits like ["train", "test"] 
        single_splits = self.split.split()
        split_files = []
        for s in single_splits:
            if s not in split_file_mapping:
                raise ValueError(f"Invalid split: {self.split}. Must be one of {list(split_file_mapping.keys())}")
            split_files += split_file_mapping[s]
 
        # Load filenames from split files
        file_list = []
        for split_file in split_files:
            split_file_path = fold_dir / split_file
            if not split_file_path.exists():
                print(f"Warning: Split file not found: {split_file_path}")
                continue
            
            # Determine arch from filename
            arch = 'lower' if 'lower' in split_file else 'upper'
            
            with open(split_file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Extract patient ID from line (e.g., "0EJDI7CW_lower.obj" -> "0EJDI7CW")
                    patient_id = line.split('_')[0]
                    
                    # Construct path: arch/patient_id/patient_id_arch.obj
                    obj_file = f"{patient_id}_{arch}.obj"
                    obj_path = Path(self.data_root) / arch / patient_id / obj_file
                    
                    if obj_path.exists():
                        # Store relative path from data_root
                        rel_path = str(Path(arch) / patient_id / obj_file)
                        file_list.append(rel_path)
                    else:
                        print(f"Warning: File not found: {obj_path}")
        
        print(f"Loaded {len(file_list)} samples from fold {self.fold}, split {self.split}")
  
        return file_list
 
    def _load_obj(self, obj_path):
        """Load OBJ file using trimesh with process=False to preserve vertex order."""
        try:
            mesh = trimesh.load(obj_path, process=False)
            points = mesh.vertices
            normals = mesh.vertex_normals
            return points.astype(np.float32), normals.astype(np.float32)
        except:
            print(f"Couldn't load sample {obj_path}")
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
        obj_path = os.path.join(self.data_root, file_rel_path)  
        json_path = os.path.join(self.data_root, file_rel_path.replace(".obj", ".json"))
 
        coord, normal = self._load_obj(obj_path)
        if self.load_segment:
            segment = self._load_json(json_path)
            assert segment.shape[0] == coord.shape[0], f"Segment shape {segment.shape[0]} != coord shape {coord.shape[0]} for sample {obj_path}"
        else:
            segment = np.zeros((coord.shape[0],), dtype=np.int32)
        
        d = {
            "coord": coord,
            "normal": normal,
            "name": Path(obj_path).stem,
            "full_path": obj_path,
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
            full_path=data_dict.get("full_path")
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
        
        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict
 
    def __len__(self):
        if self.debug: 
            return 2  # if debugging, run on just 2 samples
        return len(self.data_list) * self.loop