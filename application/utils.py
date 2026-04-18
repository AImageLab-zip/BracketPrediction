from functools import wraps
from meshlib import mrmeshnumpy as mr
import time
import numpy as np
from pathlib import Path

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

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
def save_remeshed(mesh):
        mr.saveMesh(mesh, "remeshed.stl")

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