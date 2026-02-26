"""
Brackets dataset
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
class BracketMapDataset(DefaultDataset):
    """
    Dataset for predicting bracket_point from STL files.
    """
 
    def __init__(
        self,
        data_root,
        split="train",
        debug=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        fold=None,
        production=False,
    ):
        self.fold = fold
        self.production = production
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )
        self.debug = debug
        self.fallbacks = 0 # counter for malformed stl files
        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)  
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]
 
    def get_data_list(self):

        if self.fold is None:
            # Load all STL files from the data root
            file_names = []
            for file_path in os.listdir(self.data_root):
                if file_path.endswith('.stl'):
                    file_names.append(file_path)
            print(f"Loaded {len(file_names)} samples from data_root (fold=None)")
            return file_names

        """Load list of data samples from fold JSON files, with path mapping."""
        fold_file = os.path.join(self.data_root, f"fold_{self.fold}.json")
        if not os.path.exists(fold_file):
            raise FileNotFoundError(
                f"Split file not found: {fold_file}\n"
                f"Please run the split generation script first to create split_{self.fold}.json"
            )
        with open(fold_file, 'r') as f:
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
                
                # Check if corresponding *_softlabel.npy file exists
                npy_path = os.path.join(self.data_root, file_path.replace('.stl', '_softlabel.npy'))
                if os.path.exists(npy_path):
                    file_names.append(file_path)
        
        print(f"Loaded {len(file_names)} samples from fold {self.fold}, split {self.split}")
        print(f"  Patients: {len(split_data[split_key]['patient_ids'])}")
        print(f"  Total files with softlabel: {len(file_names)}")
        
        return file_names
    
    def _load_stl(self, stl_path):

        mesh = trimesh.load(stl_path, force='mesh')
        points = mesh.vertices
        normals = mesh.vertex_normals
        return points.astype(np.float32), normals.astype(np.float32)

   
    def _load_heatmap(self, npy_path):  
        """Load heatmap values from .npy file"""  
        heatmap = np.load(npy_path)  
        heatmap = heatmap.astype(np.float32) # (N,3)
        return heatmap  

    
    def _load_json(self, json_path):
        # If in production, just load empty vectors
        if self.production:
            bracket, incisal, outer = np.zeros((3,), dtype=np.float32)     
            return bracket, incisal, outer
        with open(json_path, 'r') as f:
            data = json.load(f)
        bracket = np.array(data['bracket'], dtype=np.float32)
        incisal = np.array(data['incisal'], dtype=np.float32)
        outer = np.array(data['outer'], dtype=np.float32)
        return bracket, incisal, outer
    
    def get_data(self, idx, testing=False):  
        file_rel_path = self.data_list[idx % len(self.data_list)]  
        stl_path = os.path.join(self.data_root, file_rel_path)  
        json_path = os.path.join(self.data_root, file_rel_path.replace(".stl", ".json"))
        heatmap_path = os.path.join(self.data_root, file_rel_path.replace(".stl", "_softlabel.npy"))
 
        coord, normal = self._load_stl(stl_path)
        bracket, incisal, outer = self._load_json(json_path)
        if self.production:
            # don't need the GT heatmap when in production
            segment = np.empty_like(coord)
        else:
            segment = self._load_heatmap(heatmap_path)
        assert segment.shape[0] == coord.shape[0], f"Segment shape not matching for sample {stl_path}"
        d = {
            "coord": coord,
            "normal": normal,
            "name": Path(stl_path).stem,
            "full_path": stl_path,
            "bracket": bracket,
            "incisal": incisal,
            "outer": outer,
            "segment": segment # heatmap to predict
        }
        return d
    
    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx, testing=True)  
        data_dict = self.transform(data_dict)
        
        # ============Extract ground truth data==============
        result_dict = dict(
            segment=data_dict.pop("segment"),  
            bracket=data_dict.pop("bracket"),
            incisal = data_dict.pop("incisal"),
            outer = data_dict.pop("outer"),
            name=data_dict.get("name"),
            full_path = data_dict.get("full_path")
        )
        # ==================================================

        if "origin_segment" in data_dict:
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            if "inverse" in result_dict:
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
        if self.debug: return 2 # if debugging, run on just 2 samples
        return len(self.data_list) * self.loop