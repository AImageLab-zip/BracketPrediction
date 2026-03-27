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
class BracketPointDataset(DefaultDataset):
    """
    Dataset for BracketsV1
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
        fold = None, # fold to use
        oversample ={},
    ):
        self.fold = fold
        self.oversample = oversample
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
        """Load list of data samples from fold JSON files or all files if fold is None."""
        if self.fold is None:
            # Load all STL files from the data root
            file_names = []
            for file_path in os.listdir(self.data_root):
                if file_path.endswith('.stl'):
                    file_names.append(file_path)
            print(f"Loaded {len(file_names)} samples from data_root (fold=None)")
            return file_names
        
        fold_file = os.path.join(self.data_root, f"fold_{self.fold}.json")
        if not os.path.exists(fold_file):
            raise FileNotFoundError(
                f"Fold file not found: {fold_file}\n"
                f"Please run the split generation script first to create fold_{self.fold}.json"
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

        # Get file paths from the split
        file_paths = split_data[split_key]['files'] 
        # Filter only .stl files and extract file names without extension
        # Also remove the path prefix to get just the filename
        file_names = []
        for file_path in file_paths:
            if file_path.endswith('.stl'):
                file_names.append(file_path)
        if self.oversample:
            oversampled = []
            for f in file_names:
                FDI_index = Path(f).stem.split("_")[-1]
                if int(FDI_index) in self.oversample:
                    for _ in range(self.oversample[int(FDI_index)]-1):
                        oversampled.append(f)
            file_names = file_names + oversampled
        print(f"Loaded {len(file_names)} samples from fold {self.fold}, split {self.split}")
        print(f"  Patients: {len(split_data[split_key]['patient_ids'])}")
        print(f"  Total files (stl+json): {split_data[split_key]['num_files']}")
        
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

    def _load_orientation_map(self, map_path:str):
        '''
        Loads and returns the orientation heatmap.
        '''
        orientation_map = np.load(map_path)
        return np.expand_dims(orientation_map, 1)
    
    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        if "bracket" in data: 
            bracket_point = np.array(data['bracket'], dtype=np.float32)
        else:
            bracket_point = np.zeros((3,), dtype=np.float32)
        
        # Handle incisal point
        incisal = np.array((3,), dtype=np.float32)
        if 'incisal' in data and 'incisal_2' in data:
            # Both exist: compute midpoint
            incisal_1 = np.array(data['incisal'], dtype=np.float32)
            incisal_2 = np.array(data['incisal_2'], dtype=np.float32)
            incisal = (incisal_1 + incisal_2) / 2.0
        elif 'incisal' in data:
            incisal = np.array(data['incisal'], dtype=np.float32)
        
        # Handle outer point
        outer = np.array((3,), dtype=np.float32)
        if 'outer' in data and 'outer_2' in data:
            # Both exist: compute midpoint
            outer_1 = np.array(data['outer'], dtype=np.float32)
            outer_2 = np.array(data['outer_2'], dtype=np.float32)
            outer = (outer_1 + outer_2) / 2.0
        elif 'outer' in data:
            outer = np.array(data['outer'], dtype=np.float32)
        
        return bracket_point, incisal, outer
    
    def get_data(self, idx, testing=False):  
        file_rel_path = self.data_list[idx % len(self.data_list)]  
        stl_path = os.path.join(self.data_root, file_rel_path)  
        json_path = os.path.join(self.data_root, file_rel_path.replace(".stl", ".json"))  
        orientation_path = os.path.join(self.data_root, file_rel_path.replace(".stl", "")+"_orient.npy")
        
        coord, normal = self._load_stl(stl_path)
        bracket_point, incisal, outer = self._load_json(json_path)
        #orientation_map = self._load_orientation_map(orientation_path)
        d = {
            "coord": coord,
            "normal": normal,
        #    "orientation": orientation_map,
            "name": Path(stl_path).stem,
            "bracket": bracket_point,
            "incisal": incisal,
            "outer": outer
        }
        return d
 
    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx, testing=True)  
        data_dict = self.transform(data_dict)

        # Extract ground truth bracket_point
        result_dict = dict(
            bracket =data_dict.pop("bracket"),  # Add this line 
            outer = data_dict.pop("outer"),
            incisal = data_dict.pop("incisal"),
            name=data_dict.get("name")
        )

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