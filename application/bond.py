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
import numpy as np
import trimesh
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

# ====== GLOBAL VARIABLES ======
SINGLE_LANDMARKS = ['Bracket','Incisal', 'OuterPoint', 'Gingival','Mesial', 'Distal', 'InnerPoint', 'FacialPoint']
MOLARS = [16,17,18,26,27,28,36,37,38,46,47,48]
PREMOLARS = [14,15,24,25,34,35,44,45]
# ==============================

def rotate_point(pt, shift, seq):
    """Shift a 3D point, then apply the rotation sequence."""
    pt = np.asarray(pt, dtype=float) + shift
    for axis, degrees in seq:
        radians = np.deg2rad(degrees)
        if axis == 'x': rotation = trimesh.transformations.rotation_matrix(radians, [1, 0, 0])
        elif axis == 'y': rotation = trimesh.transformations.rotation_matrix(radians, [0, 1, 0])
        else: rotation = trimesh.transformations.rotation_matrix(radians, [0, 0, 1])
        pt = trimesh.transformations.transform_points([pt], rotation)[0]
    return pt

def rotate_points(points, shift, seq):
    """Rotate a list of 3D points."""
    return [rotate_point(pt, shift, seq).tolist() for pt in points if pt is not None]


def build_rotation_matrix(seq) -> np.ndarray:
    """Build a combined 4x4 rotation matrix from a sequence of (axis, degrees) tuples."""
    AXIS_VECTORS = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    R = np.eye(4)
    for axis, degrees in seq:
        R = trimesh.transformations.rotation_matrix(np.deg2rad(degrees), AXIS_VECTORS[axis]) @ R
    return R


def process_tooth_predictions(mesh, 
                              predictions:dict,
                              patient_id:str, 
                              fdi:int, 
                              output_dir:Path, 
                              tooth_key:str, 
                              teeth_path:Path,
                              visualize:bool = True):
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
    
    bracket = projected.get('Bracket')
    incisal = projected.get('Incisal')
    outer = projected.get('OuterPoint')
    if bracket is None or incisal is None or outer is None:
        print(f"⚠️ Missing essential points for tooth {fdi}")
        return None

    # Find face for bracket to get normal
    bracket_face_id = None
    if bracket is not None:
        try:
            _, _, faces = mesh.nearest.on_surface([bracket])
            bracket_face_id = faces
        except:
            pass

    json_data = None

    # COMPUTATION OF THE VERTEX NORMAL VECTOR
    # Correction is applied for molars
    if fdi in MOLARS:
        bracket_mm = bracket / scaling  # Convert to mm space
        vertices_mm = vertices / scaling  # Convert all vertices to mm space
 
        # Find vertices within 1.5mm radius
        distances = np.linalg.norm(vertices_mm - bracket_mm, axis=1)
        nearby_indices = np.where(distances <= 1.5)[0]
        # Average the vertex normals of nearby vertices
        nearby_normals = mesh.vertex_normals[nearby_indices]
        v_normal = np.mean(nearby_normals, axis=0)
    else:
        if bracket_face_id is not None:
            v_normal = mesh.face_normals[bracket_face_id[0]]
        else:
            v_normal = np.array([0, 0, 1])  # default

    v_normal = v_normal / np.linalg.norm(v_normal)

    # Get axis between incisal and outer points
    v_io = outer - incisal
    v_io = v_io / np.linalg.norm(v_io)

    # Get perpendicular axis to define plane
    v_perp = np.cross(v_normal, v_io)
    v_perp = v_perp / np.linalg.norm(v_perp)
    # For molars, fit plane from planar points instead
    if fdi in MOLARS and 'Planar' in projected and len(projected['Planar']) == 4:
        try:
            planar_pts = np.array(projected['Planar'])
            center = np.mean(planar_pts, axis=0)
            centered = planar_pts - center
            # SVD to find best fit plane
            U, S, Vt = np.linalg.svd(centered)
            # Normal is the direction with smallest singular value
            v_normal = Vt[2]
            v_normal = v_normal / np.linalg.norm(v_normal)
            # Get two orthogonal vectors in the plane
            v1 = Vt[0]
            v1 = v1 / np.linalg.norm(v1)
            v2 = Vt[1]
            v2 = v2 / np.linalg.norm(v2)
            v_io = v1
            v_perp = v2
        except Exception as e:
            print(f"  ⚠️  Plane fitting failed for molar {fdi}: {e}")
    # Apply inverse transformation: denormalize the points
    denormalized = {}
    for name, point in projected.items():
        if isinstance(point, list):
            denormalized[name] = [p / scaling + translation for p in point]
        else:
            denormalized[name] = point / scaling + translation
    
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
    
    # Add cusps for molars and premolars
    if fdi in MOLARS or fdi in PREMOLARS and 'Cusp' in denormalized:
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
            plot_teeth(plot_points, v_io, v_perp, vertices, patient_id, fdi, output_dir)
        except Exception as e:
            print(f"  ⚠️  Tooth visualization failed: {e}")
    return json_data

def postprocess_predictions(data_folder:Path, teethland:bool, visualize:bool = True):
    """
    Post-processes predictions and creates visualizations.
    
    Args:
        data_folder: Path to the data folder containing predictions
        visualize: toggles visualization
    """ 
    output_reg_path = data_folder / "output_reg" / "results"
    teeth_path = data_folder / "output_seg" / "teeth"
    viz_dir = data_folder /  "output_reg" / "plots"
    print("\n=== Starting Post-Processing and Visualization ===")
    pred_file = output_reg_path / "predictions.json"
    print(f"Loading predictions from: {pred_file.name}")
    with open(pred_file, 'r') as f: all_predictions = json.load(f)
    print(f"Found predictions for {len(all_predictions)} teeth")    
    all_points_data = {}
    for tooth_key, predictions in all_predictions.items():
        arch, patient_id, fdi = parse_tooth(tooth_key)
        stl_file = teeth_path / f"{tooth_key}.stl"
        mesh = trimesh.load_mesh(stl_file)
 
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
            visualize=visualize
        )
        if points_data:
            all_points_data[tooth_key] = points_data

    # Save all points to a single JSON file
    output_json_path = output_reg_path / "projected_points.json"
    with open(output_json_path, "w") as f: json.dump(all_points_data, f, indent=4)
    print(f"\n💾 Saved all projected points to: {output_json_path}")

    # Rotated version of points file.
    rotated_points = {}

    for tooth_key, pdata in all_points_data.items():
        arch, patient_id, fdi = parse_tooth(tooth_key)
        # Load shift file before rotation
        shift = np.array([0.0, 0.0, 0.0])
        shift_file_name = f"STEM_{arch}_{patient_id}_shift.json"
        shift_file = data_folder / shift_file_name

        if shift_file.exists():
            try:
                with open(shift_file, 'r') as f:
                    shift_data = json.load(f)
                    shift = np.array(shift_data.get('shift', [0.0, 0.0, 0.0]))
            except Exception as e:
                print(f"  ⚠️ Could not load or parse shift file {shift_file}: {e}")
        else:
            print(f"  ⚠️ Shift file does not exist for patient {patient_id}")
        # Choose rotation sequence based on arch
        if teethland:
            if arch == 'lower': seq = [('z', 180)]
            else: seq = [('z', 180)]
        else:
            if arch == 'lower': seq = [('x', -90), ('y', 180)]
            else: seq = [('y', 180), ('x', -90), ('y', 180)]
        try:
            # load tooth transform (scaling, translation/centroid)
            transform_file = teeth_path / f"{tooth_key}.json"
            scale = 1.0
            centroid = np.array([0.0, 0.0, 0.0])
            with open(transform_file, 'r') as tf:
                tdata = json.load(tf)
                scale = float(tdata.get('scaling', 1.0))
                centroid = np.array(tdata.get('translation', [0.0, 0.0, 0.0]))


            # homogeneous scale (divide by scale)
            S = np.eye(4)
            if scale != 0: S[0, 0] = S[1, 1] = S[2, 2] = 1.0 / float(scale)
            # translation by centroid (from tooth json)
            T_centroid = trimesh.transformations.translation_matrix(centroid.tolist())
            # initial upper rotation applied during denormalization (fdi <= 28)
            if fdi is not None and fdi <= 28: 
                R_upper1 = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
            else: 
                R_upper1 = np.eye(4)
            # shift translation (from shift file)
            T_shift = trimesh.transformations.translation_matrix(shift.tolist())
            # final rotation sequence
            R_seq = build_rotation_matrix(seq)

            # Combined matrix
            M = R_seq @ T_shift @ R_upper1 @ T_centroid @ S

            incisal, outer, origin, xaxis, yaxis, zaxis = rotate_points(
                [pdata['incisal'], pdata['outer'],
                pdata['basePlane']['origin'], pdata['basePlane']['xAxis'],
                pdata['basePlane']['yAxis'],  pdata['basePlane']['zAxis']],
                shift, seq
            )
            rotated_entry = {
                'incisal': incisal,
                'outer': outer,
                'basePlane': {
                    'origin': origin,
                    'xAxis': xaxis,
                    'yAxis': yaxis,
                    'zAxis': zaxis,
                },
            }
            # Add optional scalar points
            for key in ['gingival', 'mesial', 'distal', 'inner', 'facial', 'bracket']:
                if pdata.get(key) is not None:
                    try: rotated_entry[key] = rotate_point(pdata[key], shift, seq).tolist()
                    except Exception: pass
            # Add cusp points for molars and premolars
            if fdi is not None and fdi in MOLARS + PREMOLARS and pdata.get('cusps') is not None:
                try: rotated_entry['cusps'] = rotate_points(pdata['cusps'], shift, seq)
                except Exception: pass
            # Add planar points for molars
            if fdi is not None and fdi in MOLARS and pdata.get('planar') is not None:
                try: rotated_entry['planar'] = rotate_points(pdata['planar'], shift, seq)
                except Exception: pass

            rotated_points[tooth_key] = rotated_entry
        except Exception as e:
            print(f"⚠️ Error rotating points for {tooth_key}: {e}")

    rotated_output_path = output_reg_path / "projected_points_rotated.json"
    with open(rotated_output_path, 'w') as f: json.dump(rotated_points, f, indent=4)
    print(f"\n💾 Saved rotated projected points to: {rotated_output_path}")
    print(f"\n✅ Post-processing complete.")


def run_bond_with_model(cfg, model, data_folder: Path) -> bool:
    """
    Run bond prediction with a pre-loaded model.
    
    Args:
        cfg: Configuration object
        model: Pre-loaded bond prediction model
        data_folder: Path to data folder containing segmentation results
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        data_folder = Path(data_folder)
        teeth_path = data_folder / "output_seg" / "teeth"
        output_path = data_folder / "output_reg"
 
        # Verify segmentation output exists
        if not teeth_path.exists():
            raise FileNotFoundError(f"Segmentation output not found: {teeth_path}. Did segmentation complete successfully?")
        
        # Set data paths in config before setup
        cfg._cfg_dict["data_root"] = str(teeth_path)
        cfg._cfg_dict["save_path"] = str(output_path)
        cfg._cfg_dict["data"]["test"]["data_root"] = str(teeth_path)
 
        os.makedirs(output_path, exist_ok=True)
        
        # Set up configuration
        cfg = default_setup(cfg)
        
        # Build and run tester with cached model
        test_cfg = dict(cfg=cfg, model=model, **cfg.test)
        tester = TESTERS.build(test_cfg)
        tester.test()
        
        # Add post-processing after testing is intentionally skipped here.
        # The monitor will call `postprocess_predictions` after reporting
        # completion to the server so that plotting happens afterward.
        print("\n" + "="*60)
        print("Testing complete. Post-processing deferred to monitor.")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"❌ Error in bond prediction processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main_worker(cfg):
    os.makedirs(cfg.save_path, exist_ok=True)
    cfg = default_setup(cfg)
    test_cfg = dict(cfg=cfg, **cfg.test)
    tester = TESTERS.build(test_cfg)
    tester.test()
 
    # Add post-processing and visualization after testing
    print("\n" + "="*60)
    print("Testing complete. Starting post-processing...")
    print("="*60)
    
    # Extract data_folder from save_path
    data_folder = Path(cfg.save_path).parent
    postprocess_predictions(data_folder, visualize=False)


def main():
    parser = default_argument_parser()
    args = parser.parse_args()
    if args.debug:
        print("Hello, happy debugging.")
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Debugger is listening on port 5681. Waiting for client to attach...")
        debugpy.wait_for_client()
        print(">>> Debugger attached. Resuming execution.")
    cfg = default_config_parser(args.config_file, args.options)
    cfg._cfg_dict["data_root"] = str(Path(args.options["data_folder"]) /  "output_seg" / "teeth")
    cfg._cfg_dict["save_path"] = str(Path(args.options["data_folder"]) / "output_reg") 
    cfg._cfg_dict["data"]["test"]["data_root"] = str(Path(args.options["data_folder"]) /  "output_seg" / "teeth")
    cfg.no_visuals = args.no_visuals
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()