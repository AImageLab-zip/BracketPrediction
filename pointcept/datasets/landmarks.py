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
class BracketsV2(DefaultDataset):
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
        
        split_mapping = {'train': 'train','val': 'validation','test': 'test'}
        split_key = split_mapping.get(self.split)
        if split_key not in split_data: raise ValueError(f"Invalid split: {self.split}. Must be one of {list(split_mapping.keys())}")

        # Get file paths from the split and apply path mapping
        file_paths = split_data[split_key]['files']
        file_names = []
        for file_path in file_paths:
            if file_path.endswith('.stl'):
                # Check if the STL file exists in data_root
                full_stl_path = os.path.join(self.data_root, file_path)
                if not os.path.exists(full_stl_path): continue
                # Check if corresponding *_softlabel.npy file exists
                npy_path = os.path.join(self.data_root, file_path.replace('.stl', '_softlabel.npy'))
                if os.path.exists(npy_path): file_names.append(file_path)
 
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
        # heathamp has shape (N,4) for brackets, (N,6) for 3dteethland
        heatmap = heatmap.astype(np.float32)
        return heatmap

    def _load_json(self, json_path):
        # Returns a dictionary containing the landmarks
        if self.production: return {}
        return json.load(open(json_path, 'r'))
    
    def _generalize_heatmap(self, heatmap: np.ndarray, stl_path: Path):
        """
        Convert a dataset-specific heatmap into a unified (N, 10) heatmap.
        Returns
        -------
        generalized_heatmap : np.ndarray, shape (N, 10)
        validity_mask       : np.ndarray, shape (N, 10), dtype uint8
        A channel is valid iff:
        1) it exists in the original heatmap
        2) it is not entirely zero
        """
        if heatmap.ndim != 2:
            raise ValueError(f"Expected heatmap with shape (N, C), got {heatmap.shape}")

        dataset = "brackets" if "brackets" in str(stl_path) else "3dteethland"
        N, channels = heatmap.shape
        generalized_heatmap = np.zeros((N, 10), dtype=heatmap.dtype)
        validity_mask = np.zeros((N, 10), dtype=np.uint8)
        if dataset == "brackets":
            if channels != 4:
                raise ValueError(
                    f"Expected brackets heatmap with 4 channels, got shape {heatmap.shape}"
                )

            generalized_heatmap[:, 0:4] = heatmap
            channel_valid = np.any(heatmap != 0, axis=0).astype(np.uint8)
            validity_mask[:, 0:4] = np.broadcast_to(channel_valid[None, :], (N, 4))

        else:  # 3dteethland
            if channels != 6:
                raise ValueError(
                    f"Expected 3dteethland heatmap with 6 channels, got shape {heatmap.shape}"
                )

            generalized_heatmap[:, 4:10] = heatmap
            channel_valid = np.any(heatmap != 0, axis=0).astype(np.uint8)
            validity_mask[:, 4:10] = np.broadcast_to(channel_valid[None, :], (N, 6))

        return generalized_heatmap, validity_mask

    def get_data(self, idx, testing=False):
        file_rel_path = self.data_list[idx % len(self.data_list)]
        stl_path = os.path.join(self.data_root, file_rel_path)
        json_path = os.path.join(self.data_root, file_rel_path.replace(".stl", ".json"))
        heatmap_path = os.path.join(self.data_root, file_rel_path.replace(".stl", "_softlabel.npy"))

        coord, normal = self._load_stl(stl_path)
        landmarks = self._load_json(json_path)

        if self.production:
            segment = np.empty((coord.shape[0], 10), dtype=np.float32)
            validity_mask = np.zeros((coord.shape[0], 10), dtype=np.uint8)
        else:
            dataset_specific_heatmap = self._load_heatmap(heatmap_path)
            segment, validity_mask = self._generalize_heatmap(dataset_specific_heatmap, stl_path)
        assert segment.shape[0] == coord.shape[0], f"Segment shape not matching for sample {stl_path}"

        d = {
            "coord": coord,
            "normal": normal,
            "name": Path(stl_path).stem,
            "full_path": stl_path,
            "landmarks": landmarks,
            "segment": segment,
            "validity_mask": validity_mask,
        }
        return d

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx, testing=True)  
        data_dict = self.transform(data_dict)
 
        # ============Extract ground truth data==============
        result_dict = dict(
            segment=data_dict.pop("segment"),  
            validity_mask=data_dict.pop("validity_mask"),
            landmarks=data_dict.pop("landmarks"),
            name=data_dict.get("name"),
            full_path = data_dict.get("full_path")
        )
        # ==================================================

        if "origin_segment" in data_dict:
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            if "inverse" in result_dict:
                result_dict["inverse"] = data_dict.pop("inverse")
        if "origin_validity_mask" in data_dict:
            result_dict["origin_validity_mask"] = data_dict.pop("origin_validity_mask")
    
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