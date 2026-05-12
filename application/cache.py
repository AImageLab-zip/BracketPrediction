"""Simple in-memory cache for teeth meshes and their metadata."""
import trimesh
from pathlib import Path
import json


class TeethCache:
    """In-memory cache for teeth meshes and transformation metadata."""
    
    def __init__(self):
        self.meshes = {}  # tooth_key -> trimesh object
        self.transforms = {}  # tooth_key -> transform dict
        self.scan_meshes = {}  # scan_name -> trimesh object
    
    def load_mesh(self, teeth_path: Path, tooth_key: str) -> trimesh.Trimesh:
        """Load mesh from cache or disk. Returns trimesh object."""
        if tooth_key in self.meshes:
            return self.meshes[tooth_key]
        
        stl_file = teeth_path / f"{tooth_key}.stl"
        mesh = trimesh.load_mesh(stl_file)
        self.meshes[tooth_key] = mesh
        return mesh
    
    def store_mesh(self, tooth_key: str, mesh: trimesh.Trimesh, transform_data: dict):
        """Store mesh and transformation in cache (skip disk I/O)."""
        self.meshes[tooth_key] = mesh
        self.transforms[tooth_key] = transform_data
    
    def load_transform(self, teeth_path: Path, tooth_key: str) -> dict:
        """Load transformation metadata from cache or disk."""
        if tooth_key in self.transforms:
            return self.transforms[tooth_key]
        
        transform_file = teeth_path / f"{tooth_key}.json"
        with open(transform_file, 'r') as f:
            transform_data = json.load(f)
        self.transforms[tooth_key] = transform_data
        return transform_data
    
    def clear(self):
        """Clear all cached data."""
        self.meshes.clear()
        self.transforms.clear()
        self.scan_meshes.clear()

    def clear_teeth_data(self):
        """Clear only tooth-level cache, keep preloaded scan meshes."""
        self.meshes.clear()
        self.transforms.clear()

    def preload_scan_mesh(self, scan_path: Path):
        """Load a full scan mesh once and keep it in memory."""
        scan_name = scan_path.name
        if scan_name not in self.scan_meshes:
            self.scan_meshes[scan_name] = trimesh.load_mesh(str(scan_path), process=False)

    def get_scan_mesh(self, scan_path: Path):
        """Get a preloaded scan mesh by filename, if available."""
        return self.scan_meshes.get(scan_path.name)
