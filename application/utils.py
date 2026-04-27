import meshlib.mrmeshpy as mr
import numpy as np
from pathlib import Path
import faiss
from timing import *
import json


@timed
def custom_remesh(path:Path,
                  relax_iterations=5,
                  decimate_maxError=0.1,
                  decimate_maxDeletedFaces_ratio=0.5,
                  decimate_subdivideParts=4,
                  decimate_maxTriangleAspectRatio=20.0,
                  ):
    mesh = mr.loadMesh(path)
    mesh.packOptimally()

    relax_params = mr.MeshRelaxParams()
    relax_params.iterations = relax_iterations
    mr.relax(mesh, relax_params)

    settings = mr.DecimateSettings()
    settings.maxError = decimate_maxError
    settings.subdivideParts = decimate_subdivideParts
    settings.maxDeletedFaces = int(mesh.topology.numValidFaces()*decimate_maxDeletedFaces_ratio)
    settings.maxTriangleAspectRatio = decimate_maxTriangleAspectRatio
    mr.decimateMesh(mesh, settings)
    return mesh

@timed
def save_remeshed(mesh, filepath:Path):
    mr.saveMesh(mesh, filepath)

def is_consistent(vertices:np.ndarray, mask:np.ndarray):
    if len(mask) != len(vertices):
        print(f"Warning: Mask length ({len(mask)}) doesn't match points ({len(vertices)})")
        return False
    return True

def rotation_180_y():
    return np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])

def parse_tooth(tooth:str) -> list[str, str, int]:
    # Parse tooth_key: expected format "STEM_lower_0002_FDI_47"
    _, jaw, patient_id, _, fdi = tooth.split("_")
    return jaw, patient_id, int(fdi)

@timed
def fit_segmask(segmask:np.ndarray, source:np.ndarray, dest:np.ndarray):
    d = 3
    index = faiss.IndexFlatL2(d)
    index.add(source)
    _, indices = index.search(dest, 1)
    remeshed_mask = segmask[indices]
    assert remeshed_mask.shape[0] == dest.shape[0]
    return remeshed_mask
    #create_segmentation_visualization(remeshed, remeshed_mask.squeeze(), "reseshed_segmentation.png", Path(OUT_DIR))

def get_edges(faces:np.ndarray) -> np.ndarray:
    edges = np.unique(np.sort(np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [0, 2]],
    ], axis=0), axis=1), axis=0)
    return edges


def load_json(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not read {path}: {e}")
        return {}


def save_json(path: Path, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"⚠️  Could not save {path}: {e}")