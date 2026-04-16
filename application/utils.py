from functools import wraps
from meshlib import mrmeshnumpy as mr
import time
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
