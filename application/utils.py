import meshlib.mrmeshpy as mr
import numpy as np
from pathlib import Path
import faiss
from timing import *
import json
from pointcept.models import build_model
from pointcept.engines.defaults import default_config_parser, default_setup
import torch
import csv
import trimesh


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

def parse_tooth(tooth:str) -> tuple[str, str, int]:
    # Two naming conventions are supported:
        # "STEM_lower_0002_FDI_47"
        # "0002_lower_FDI_47"
    if tooth.startswith("STEM"):
        _, arch, patient_id, _, fdi = tooth.split("_")
    else:
        patient_id, arch, _, fdi = tooth.split("_")
    return arch, patient_id, int(fdi)

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

def load_model(config: Path, weights: Path):
    cfg   = default_setup(default_config_parser(str(config), {}))
    model = build_model(cfg.model)
    ckpt  = torch.load(str(weights), weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model = model.cuda().eval()
    print("   ✅ Model loaded")
    return cfg, model


def write_rows(rows:list, output_path:str | Path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "coord_x", "coord_y", "coord_z", "class", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics written to {output_path}")

def teethland_output(json_file: str | Path, write_single=False):
    '''
    Writes a csv file next to the given json file.
    '''
    json_path = Path(json_file)
    predictions = json.load(open(json_path))
    output_path = json_path.with_suffix(".csv")
    rows = []
    # don't need gingival for now
    LANDMARK_KEY_MAP = {
        "planar": "Planar",
        "bracket": "Bracket",
        "incisal": "Incisal",
        "cusps": "Cusp",
        "outer": "OuterPoint",
        "mesial": "Mesial",
        "distal": "Distal",
        "inner": "InnerPoint",
        "facial": "FacialPoint",
    }
    for key, vals in predictions.items():
        if key.startswith("STEM"): # old naming
            _, arch, ide, _, fdi = key.split("_")
        else: # new naming
            ide, arch, _, fdi = key.split("_")
        key_base = f"{ide}_{arch}"
        for landmark, coords in vals.items():
            if landmark == "basePlane": continue
            pred_landmark = LANDMARK_KEY_MAP.get(landmark)
            if not pred_landmark: continue
            if landmark in "cusps planar".split():
                for coord in coords:
                    rows.append({
                        "key": key_base,
                        "coord_x": coord[0],
                        "coord_y": coord[1],
                        "coord_z": coord[2],
                        "class": pred_landmark,
                        "score": 1.0
                    })
            else:
                rows.append({
                    "key": key_base,
                    "coord_x": coords[0],
                    "coord_y": coords[1],
                    "coord_z": coords[2],
                    "class": pred_landmark,
                    "score": 1.0
                })
    if write_single: write_rows(rows, output_path)
    return rows

def get_normal_smooth_vector(mesh:trimesh.Trimesh, bracket:np.ndarray, vertices:np.ndarray, scaling:float) -> np.ndarray:
    bracket_mm = bracket / scaling  # Convert to mm space
    vertices_mm = vertices / scaling  # Convert all vertices to mm space
    # Find vertices within 1.5mm radius
    distances = np.linalg.norm(vertices_mm - bracket_mm, axis=1)
    nearby_indices = np.where(distances <= 1.5)[0]
    # Average the vertex normals of nearby vertices
    nearby_normals = mesh.vertex_normals[nearby_indices]
    v_normal = np.mean(nearby_normals, axis=0)
    v_normal = v_normal / np.linalg.norm(v_normal)
    return v_normal

def fit_plane(projected:dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planar_pts = np.array(projected['Planar'])
    center = np.mean(planar_pts, axis=0)
    centered = planar_pts - center
    # SVD to find best fit plane
    U, S, Vt = np.linalg.svd(centered)
    # Normal is the direction with smallest singular value
    v_normal = Vt[2] / np.linalg.norm(Vt[2])
    # Get two orthogonal vectors in the plane
    v1 = Vt[0] / np.linalg.norm(Vt[0])
    v2 = Vt[1] / np.linalg.norm(Vt[1])
    v_io, v_perp = v1, v2
    return v_normal, v_io, v_perp

def get_io_perp(v_normal:np.ndarray, incisal:np.ndarray, outer:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v_io = outer - incisal
    v_io = v_io / np.linalg.norm(v_io)
    # Get perpendicular axis to define plane
    v_perp = np.cross(v_normal, v_io)
    v_perp = v_perp / np.linalg.norm(v_perp)
    return v_io, v_perp