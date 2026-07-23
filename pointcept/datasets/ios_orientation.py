import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh
from torch.utils.data import Dataset

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose


MESH_EXTENSIONS = {".stl", ".ply", ".obj", ".off"}
ANGLE_MIN_DEG = 1
ANGLE_MAX_DEG = 180


def rotation_matrix_from_angles(angles_deg):
    x, y, z = np.deg2rad(angles_deg.astype(np.float32))
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rz @ ry @ rx).astype(np.float32)


@DATASETS.register_module()
class IOSOrientationDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        max_points=None,
        debug=False,
    ):
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform) if transform is not None else None
        self.test_mode = test_mode
        self.test_cfg = test_cfg
        self.loop = loop
        self.max_points = max_points
        self.debug = debug
        self.data_list = self.get_data_list()

        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]
            self.test_voxelize = Compose([test_cfg.voxelize]) if test_cfg.voxelize else None
            self.test_crop = Compose([test_cfg.crop]) if test_cfg.crop else None

    def get_data_list(self):
        files = []
        split_dir = os.path.join(self.data_root, self.split)
        root = split_dir if os.path.isdir(split_dir) else self.data_root
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if Path(path).suffix.lower() in MESH_EXTENSIONS:
                    files.append(os.path.relpath(path, self.data_root))
        files.sort()
        if not files:
            raise RuntimeError(f"No mesh files found in {root}")
        return files

    def _load_mesh_points(self, path):
        mesh = trimesh.load(path, force="mesh", process=False)
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            raise RuntimeError(f"Could not load vertices from {path}")
        coord = np.asarray(mesh.vertices, dtype=np.float32)
        if self.max_points is not None and coord.shape[0] > self.max_points:
            index = np.random.choice(coord.shape[0], self.max_points, replace=False)
            coord = coord[index]
        return coord

    def _make_sample(self, coord):
        center = coord.mean(axis=0).astype(np.float32)
        target_coord = coord - center
        scale = np.max(np.linalg.norm(target_coord, axis=1)).astype(np.float32)
        target_coord = target_coord / max(float(scale), 1e-8)
        angles = np.random.randint(ANGLE_MIN_DEG, ANGLE_MAX_DEG + 1, size=3).astype(np.int64)
        rotation = rotation_matrix_from_angles(angles)
        input_coord = target_coord @ rotation

        # Store zero-based classes for 1..180 degree bins.
        return input_coord.astype(np.float32), target_coord.astype(np.float32), angles - ANGLE_MIN_DEG

    def get_data(self, idx):
        rel_path = self.data_list[idx % len(self.data_list)]
        path = os.path.join(self.data_root, rel_path)
        coord = self._load_mesh_points(path)
        input_coord, target_coord, angle = self._make_sample(coord)
        return {
            "coord": input_coord,
            "target_coord": target_coord,
            "angle": angle,
            "name": Path(path).stem,
        }

    def prepare_train_data(self, idx):
        data = self.get_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def prepare_test_data(self, idx):
        data = self.get_data(idx)
        result = {
            "name": data["name"],
            "angle": data["angle"],
        }
        if self.transform is not None:
            data = self.transform(data)

        fragment_list = []
        for aug in self.aug_transform:
            aug_data = aug(deepcopy(data))
            data_parts = self.test_voxelize(aug_data) if self.test_voxelize else [aug_data]
            for part in data_parts:
                cropped = self.test_crop(part) if self.test_crop else [part]
                fragment_list += cropped
        result["fragment_list"] = [self.post_transform(fragment) for fragment in fragment_list]
        return result

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        return self.prepare_train_data(idx)

    def __len__(self):
        if self.debug:
            return min(2, len(self.data_list))
        return len(self.data_list) * self.loop
