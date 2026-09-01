"""
Modified testing script with postprocessing and visualization.
Operates on the teeth meshes and regresses the pose of the bracket point
and bracket plane (using incisal and buccal/outer point prediction).
Brings back the coordinates of the point to the original one, so that
the points are aligned with the mesh on which only the scanTransformMatrix
is applied.

The bonding is automatically handled by the monitor process, but if you want to try
the script for debugging purposes, here's the command to run on sample patient "2":
python application/bond.py \
    --config-file /homes/mlugli/BracketPrediction/application/configs/Pt_regressor_app.py \
    --options data_folder=/homes/mlugli/BracketPrediction/application/data/2/ \
              weight=/homes/mlugli/BracketPrediction/application/weights/regressor_best.pth
"""

import debugpy
import os
import json
import threading
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch
from visualizers import plot_teeth
from utils import *
from cache import TeethCache
from pointcept.datasets.preprocessing.autobonding.scan_normalizer import ScanNormalizer

# matplotlib's pyplot state is process-global and not thread-safe; serialize
# figure creation when process_tooth_predictions runs across worker threads.
_VIZ_LOCK = threading.Lock()

# ====== GLOBAL VARIABLES ======
SINGLE_LANDMARKS = ['Bracket','Incisal', 'OuterPoint', 'Gingival','Mesial', 'Distal', 'InnerPoint', 'FacialPoint']
MULTI_LANDMARKS = ['Planar', 'Cusp']
# All landmark classes the heatmap tester can decode; valid values for --landmarks.
ALL_LANDMARKS = SINGLE_LANDMARKS + MULTI_LANDMARKS
MOLARS = [16,17,18,26,27,28,36,37,38,46,47,48]
PREMOLARS = [14,15,24,25,34,35,44,45]
# ==============================

def transform_multipoint(points_list, arch, preprocessor):
    """Map a list of points from the model's working frame back to the scan
    frame by inverting the --preprocessing transform (identity if none)."""
    arr = np.array(points_list, dtype=float)
    return preprocessor.apply_inverse(arr, jaw=arch).tolist()

def _denormalization_matrix(translation, scaling, flip):
    """4x4 transform inverting normalize(): undoes unit-sphere scaling, the
    centroid recentre, and (upper only) the 180 deg Y flip."""
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = (1.0 / scaling) if scaling else 1.0
    T = np.eye(4)
    T[:3, 3] = np.asarray(translation, dtype=float)
    R = np.eye(4)
    if flip: R[:3, :3] = rotation_180_y()          # its own inverse
    return R @ T @ S
    
def process_tooth_predictions(mesh, 
                              predictions:dict,
                              patient_id:str, 
                              fdi:int, 
                              output_dir:Path, 
                              tooth_key:str, 
                              teeth_path:Path,
                              visualize:bool = True,
                              cache: TeethCache | None = None):
    """
    Creates three 2D views (XY, XZ, YZ) of the mesh with predicted points.
    Projects predictions onto mesh surface using nearest point method. 
    Args:
        mesh: trimesh object
        predictions: dictionary containing tooth predictions
        patient_id: patient identifier
        fdi: FDI tooth index
        output_dir: directory to save the PNG file
        teeth_path: path to the teeth directory containing transformation JSON files
        tooth_key: key for the tooth
        planar_pred: list of planar points
        cusp_pred: list of cusp points
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vertices = mesh.vertices
 
    # Load transformation parameters from tooth's JSON file
    transform_file = teeth_path / f"{tooth_key}.json"
    translation = np.array([0.0, 0.0, 0.0])
    scaling = 1.0
    if cache:
        translation= np.array(cache.transforms[tooth_key].get("translation", [0.0,0.0,0.0]))
        scaling = float(cache.transforms[tooth_key].get("scaling", 1))
    else:
        with open(transform_file, 'r') as f:
            transform_data = json.load(f)
            translation = np.array(transform_data.get('translation', [0.0, 0.0, 0.0]))
            scaling = float(transform_data.get('scaling', 1.0))
            print(f"  Loaded transformation for {tooth_key}: translation={translation}, scaling={scaling}")

    # ========= Use the predicted points directly (already on mesh) =============
    projected = {}
    single_points = [(lc, predictions.get(lc)) for lc in SINGLE_LANDMARKS]
    for name, pred in single_points:
        if pred is not None: projected[name] = np.array(pred)
    
    if 'Planar' in predictions:
        projected['Planar'] = [np.array(p) for p in predictions["Planar"]]
    if 'Cusp' in predictions:
        projected['Cusp'] = [np.array(p) for p in predictions['Cusp']]

    # Sanity check
    bracket = projected.get('Bracket')
    incisal = projected.get('Incisal')
    outer = projected.get('OuterPoint')
    if bracket is None or incisal is None or outer is None:
        print(f"⚠️ Missing essential points for tooth {fdi}")
        return None

    json_data = None

    # Get the 3 axis of the tooth 
    if fdi in MOLARS and 'Planar' in projected and len(projected['Planar']) == 4:
        v_normal, v_io, v_perp = fit_plane(projected)
    else:
        v_normal = get_normal_smooth_vector(mesh, bracket, vertices, scaling)
        v_io, v_perp = get_io_perp(v_normal, np.array(incisal), np.array(outer))

    # Apply inverse transformation: denormalize the points
    denormalized = {}
    for name, point in projected.items():
        if isinstance(point, list): denormalized[name] = [p/scaling + translation for p in point]
        else: denormalized[name] = point / scaling + translation
    
    # Denormalize vectors
    v_perp_denorm = v_perp / scaling
    v_normal_denorm = v_normal / scaling
    
    if fdi <= 28:
        rotation_matrix = rotation_180_y()
        for name, point in denormalized.items():
            if isinstance(point, list):
                denormalized[name] = [(p @ rotation_matrix.T).tolist() for p in point]
            else:
                denormalized[name] = (point @ rotation_matrix.T).tolist()
        v_perp_denorm = (v_perp_denorm @ rotation_matrix.T).tolist()
        v_normal_denorm = (v_normal_denorm @ rotation_matrix.T).tolist()
    else:
        # Convert arrays to lists (no rotation needed)
        for name, point in denormalized.items():
            if isinstance(point, list):
                denormalized[name] = [p.tolist() for p in point]
            else:
                denormalized[name] = point.tolist()
        v_perp_denorm = v_perp_denorm.tolist() if hasattr(v_perp_denorm, 'tolist') else v_perp_denorm
        v_normal_denorm = v_normal_denorm.tolist() if hasattr(v_normal_denorm, 'tolist') else v_normal_denorm
    
    json_data = {
        "bracket": denormalized.get('Bracket'),
        "incisal": denormalized.get('Incisal'),
        "outer": denormalized.get('OuterPoint'),
        "gingival": denormalized.get('Gingival'),
        "mesial": denormalized.get('Mesial'),
        "distal": denormalized.get('Distal'),
        "inner": denormalized.get('InnerPoint'),
        "facial": denormalized.get('FacialPoint'),
        "basePlane": {
            "origin": denormalized.get('Bracket'),
            "xAxis": denormalized.get('Incisal'),
            "yAxis": (np.array(denormalized.get('Bracket')) + v_perp_denorm).tolist(),
            "zAxis": (np.array(denormalized.get('Bracket')) + v_normal_denorm).tolist(),
        },
    }
    # Tooth-level part only (undoes normalize()); postprocess_predictions left-
    # multiplies the preprocessing inverse to get the full denorm_matrix.
    json_data["denorm_matrix_tooth"] = _denormalization_matrix(translation, scaling, flip=(fdi <= 28)).tolist()
    
    # Add cusps for molars and premolars
    if (fdi in MOLARS or fdi in PREMOLARS) and 'Cusp' in denormalized:
        json_data["cusps"] = denormalized['Cusp']
    
    # Add planar for molars
    if fdi in MOLARS and 'Planar' in denormalized:
        json_data["planar"] = denormalized['Planar']
    # Filter points for visualization based on tooth type
    # Include 'Outer' in single-tooth plots (only exclude 'Facial')
    plot_points = {k: v for k, v in projected.items() if k != 'FacialPoint'}
    # Only show planar for molars
    if fdi not in MOLARS and 'Planar' in plot_points:
        del plot_points['Planar']
    # Only show cusps for molars and premolars
    if fdi not in MOLARS and fdi not in PREMOLARS and 'Cusp' in plot_points:
        del plot_points['Cusp']
    if visualize:
        try:
            with _VIZ_LOCK:
                plot_teeth(plot_points, v_io, v_perp, vertices, patient_id, fdi, output_dir)
        except Exception as e:
            print(f"  ⚠️  Tooth visualization failed: {e}")
    return json_data

def postprocess_predictions(data_folder:Path,
                            visualize:bool = True,
                            cache:TeethCache | None = None,
                            preprocessor:ScanNormalizer | None = None,
                            workers: int = 1):
    """
    Post-processes predictions and creates visualizations.

    Args:
        data_folder: Path to the data folder containing predictions
        visualize: toggles visualization
        cache: cache that stores single teeth mesh objects
        preprocessor: preprocessor that automatically handles scan normalization
        workers: number of teeth to post-process concurrently, for both the
            per-tooth coordinate-frame step and the rotate-back-to-scan step.
            1 (default) processes teeth sequentially. Uses a thread pool, not
            separate processes, so `cache` stays a single shared in-memory
            store visible to every tooth.
    """
    output_reg_path = data_folder / "output_reg" / "results"
    teeth_path = data_folder / "output_seg" / "teeth"
    viz_dir = data_folder /  "output_reg" / "plots"
    print("\n=== Starting Post-Processing and Visualization ===")
    pred_file = output_reg_path / "predictions.json"
    print(f"Loading predictions from: {pred_file.name}")
    with open(pred_file, 'r') as f: all_predictions = json.load(f)
    print(f"Found predictions for {len(all_predictions)} teeth")

    def _process_one(item):
        tooth_key, predictions = item
        arch, patient_id, fdi = parse_tooth(tooth_key)
        mesh = cache.load_mesh(teeth_path, tooth_key) if cache else trimesh.load_mesh(teeth_path / f"{tooth_key}.stl")

        # Process single tooth predictions to
        # bring them back to normalized coordinates
        points_data = process_tooth_predictions(
            mesh=mesh,
            predictions=predictions,
            patient_id=patient_id,
            fdi=fdi,
            output_dir=viz_dir,
            tooth_key=tooth_key,
            teeth_path=teeth_path,
            visualize=visualize,
            cache=cache
        )
        return tooth_key, points_data

    all_points_data = {}
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_process_one, all_predictions.items())
            for tooth_key, points_data in results:
                if points_data:
                    all_points_data[tooth_key] = points_data
    else:
        for item in all_predictions.items():
            tooth_key, points_data = _process_one(item)
            if points_data:
                all_points_data[tooth_key] = points_data

    # Save all points to a single JSON file (pre-rotation, still in the
    # segmentation working frame). Kept for backward compatibility with the
    # old production naming; not consumed by anything in this codebase.
    output_json_path = output_reg_path / "projected_points.json"
    with open(output_json_path, "w") as f: json.dump(all_points_data, f, indent=4)
    print(f"\n💾 Saved all projected points to: {output_json_path}")

    # Map every landmark from the model's working frame back to the scan frame
    # by inverting the --preprocessing transform. For the production monitor
    # that YAML carries the standard-orientation rotations (180Y / 90X / upper
    # extra 180Y); apply_inverse is identity when no --preprocessing is given.

    def _rotate_one(item):
        tooth_key, pdata = item
        arch, patient_id, fdi = parse_tooth(tooth_key)
        try:
            # Collect all scalar points into one (N, 3) array for a single
            # batched apply_inverse call.
            point_map = {
                'bracket':          pdata.get('bracket'),
                'incisal':          pdata.get('incisal'),
                'outer':            pdata.get('outer'),
                'gingival':         pdata.get('gingival'),
                'mesial':           pdata.get('mesial'),
                'distal':           pdata.get('distal'),
                'inner':            pdata.get('inner'),
                'facial':           pdata.get('facial'),
                'basePlane_origin': pdata['basePlane']['origin'],
                'basePlane_xAxis':  pdata['basePlane']['xAxis'],
                'basePlane_yAxis':  pdata['basePlane']['yAxis'],
                'basePlane_zAxis':  pdata['basePlane']['zAxis'],
            }
            valid_keys = [k for k, v in point_map.items() if v is not None]
            pts = np.array([point_map[k] for k in valid_keys], dtype=float)
            # Undo the --preprocessing transform (identity if none given).
            pts = preprocessor.apply_inverse(pts, jaw=arch)
            rotated_scalar = {k: pts[i].tolist() for i, k in enumerate(valid_keys)}

            rotated_entry = {
                'incisal': rotated_scalar.get('incisal'),
                'outer':   rotated_scalar.get('outer'),
                'basePlane': {
                    'origin': rotated_scalar.get('basePlane_origin'),
                    'xAxis':  rotated_scalar.get('basePlane_xAxis'),
                    'yAxis':  rotated_scalar.get('basePlane_yAxis'),
                    'zAxis':  rotated_scalar.get('basePlane_zAxis'),
                },
            }
            for key in ['bracket', 'gingival', 'mesial', 'distal', 'inner', 'facial']:
                if rotated_scalar.get(key):
                    rotated_entry[key] = rotated_scalar[key]
            if (fdi in MOLARS or fdi in PREMOLARS) and pdata.get('cusps'):
                rotated_entry['cusps'] = transform_multipoint(pdata['cusps'], arch, preprocessor)
            if fdi in MOLARS and pdata.get('planar'):
                rotated_entry['planar'] = transform_multipoint(pdata['planar'], arch, preprocessor)
            # Full transform: normalized tooth mesh -> original scan space
            # (after scanTransformMatrix, before the standard-orientation rotations).
            M = np.array(pdata.get("denorm_matrix_tooth", np.eye(4)), dtype=float)
            for mat in reversed(preprocessor.get_matrices(arch)):
                M = np.linalg.inv(mat) @ M
            rotated_entry["denorm_matrix"] = M.tolist()
            return tooth_key, rotated_entry
        except Exception as e:
            print(f"⚠️ Error rotating points for {tooth_key}: {e}")
            return tooth_key, None

    rotated_points = {}
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_rotate_one, all_points_data.items())
            for tooth_key, rotated_entry in results:
                if rotated_entry is not None:
                    rotated_points[tooth_key] = rotated_entry
    else:
        for item in all_points_data.items():
            tooth_key, rotated_entry = _rotate_one(item)
            if rotated_entry is not None:
                rotated_points[tooth_key] = rotated_entry

    # Final result, written under both names: "landmarks.json" is the name
    # used by every current entry point (main.py, infer.py); "projected_points_rotated.json"
    # is kept alongside it, byte-for-byte identical, for full backward
    # compatibility with anything still reading the old production filename
    # (e.g. visualizers.plot_jaw).
    rotated_output_path = output_reg_path / "landmarks.json"
    legacy_output_path = output_reg_path / "projected_points_rotated.json"
    with open(rotated_output_path, 'w') as f: json.dump(rotated_points, f, indent=4)
    with open(legacy_output_path, 'w') as f: json.dump(rotated_points, f, indent=4)
    print(f"\n💾 Saved rotated projected points to: {rotated_output_path} (and legacy alias {legacy_output_path.name})")
    print(f"\n✅ Post-processing complete.")

def run_bond_with_model(cfg, model, data_folder: Path, cache:TeethCache | None = None,
                         target_landmarks: list[str] | None = None) -> bool:
    """
    Run bond prediction with a pre-loaded model.

    Args:
        cfg: Configuration object
        model: Pre-loaded bond prediction model
        data_folder: Path to data folder containing segmentation results
        cache: Optional TeethCache for mesh caching
        target_landmarks: If given, restricts landmark decoding (and its k-means /
            connected-components post-processing) to these classes (see ALL_LANDMARKS).
            'Bracket', 'Incisal' and 'OuterPoint' are always decoded regardless, since
            every landmark's basePlane is expressed relative to the coordinate frame
            they define. None (default) decodes every landmark class.
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        data_folder = Path(data_folder)
        teeth_path = data_folder / "output_seg" / "teeth"
        output_path = data_folder / "output_reg"
 
        # Verify segmentation output exists (or cache is provided)
        if cache is None and not teeth_path.exists():
            raise FileNotFoundError(f"Segmentation output not found: {teeth_path}. Did segmentation complete successfully?")
 
        # Set data paths in config before setup
        cfg._cfg_dict["data_root"] = str(teeth_path)
        cfg._cfg_dict["save_path"] = str(output_path)
        cfg._cfg_dict["data"]["test"]["data_root"] = str(teeth_path)
        
        os.makedirs(output_path, exist_ok=True)
        
        # Set up configuration
        cfg = default_setup(cfg)
        if cache:
            cfg._cfg_dict["data"]["test"]["custom_cache"] = cache
            cfg._cfg_dict["data"]["test"]["type"] = "BracketsV2Cached"
        cfg._cfg_dict["test"]["target_landmarks"] = target_landmarks

        # Build and run tester with cached model
        test_cfg = dict(cfg=cfg, model=model, **cfg.test)
        tester = TESTERS.build(test_cfg)
        tester.test()

        print("\n" + "="*60)
        print("Testing complete. Post-processing deferred to monitor.")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"❌ Error in bond prediction processing: {e}")
        import traceback
        traceback.print_exc()
        return False