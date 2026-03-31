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
import matplotlib.pyplot as plt
from pathlib import Path
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch
from matplotlib.lines import Line2D
import torch

def process_tooth_predictions(mesh, 
                              bracket_pred:np.ndarray, 
                              incisal_pred:np.ndarray, 
                              outer_pred:np.ndarray, 
                              patient_id:str, 
                              fdi:int, 
                              output_dir:Path, 
                              tooth_key:str, 
                              teeth_path:Path, 
                              gingival_pred:np.ndarray = None,
                              mesial_pred:np.ndarray = None,
                              distal_pred:np.ndarray = None,
                              inner_pred:np.ndarray = None,
                              facial_pred:np.ndarray = None,
                              planar_pred:list = None,
                              cusp_pred:list = None,
                              visualize:bool = True):
    """
    Creates three 2D views (XY, XZ, YZ) of the mesh with predicted points.
    Projects predictions onto mesh surface using nearest point method. 
    Args:
        mesh: trimesh object
        bracket_pred: bracket point coordinates [x, y, z]
        incisal_pred: incisal point coordinates [x, y, z]
        outer_pred: outer point coordinates [x, y, z]
        patient_id: patient identifier
        fdi: FDI tooth index
        output_dir: directory to save the PNG file
        teeth_path: path to the teeth directory containing transformation JSON files
        tooth_key: key for the tooth
        gingival_pred: gingival point coordinates [x, y, z]
        mesial_pred: mesial point coordinates [x, y, z]
        distal_pred: distal point coordinates [x, y, z]
        inner_pred: inner point coordinates [x, y, z]
        facial_pred: facial point coordinates [x, y, z]
        planar_pred: list of planar points
        cusp_pred: list of cusp points
        visualize: whether to generate visualizations
    """
    import trimesh
    from visualizers import plot_teeth
    
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
    single_points = [
        ('Bracket', bracket_pred),
        ('Incisal', incisal_pred),
        ('Outer', outer_pred),
        ('Gingival', gingival_pred),
        ('Mesial', mesial_pred),
        ('Distal', distal_pred),
        ('Inner', inner_pred),
        ('Facial', facial_pred),
    ]
    for name, pred in single_points:
        if pred is not None:
            projected[name] = np.array(pred)
    
    # Use planar and cusp lists directly
    if planar_pred:
        projected['Planar'] = [np.array(p) for p in planar_pred]
    
    if cusp_pred:
        projected['Cusp'] = [np.array(p) for p in cusp_pred]
    
    bracket = projected.get('Bracket')
    incisal = projected.get('Incisal')
    outer = projected.get('Outer')
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
    if fdi in [16, 17, 26, 27, 36, 37, 46, 47]:
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
    molars = [16,17,18,26,27,28,36,37,38,46,47,48]
    if fdi in molars and 'Planar' in projected and len(projected['Planar']) == 4:
        try:
            # Fit plane to the 4 planar points
            planar_pts = np.array(projected['Planar'])
            # Center the points
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
            # Keep as numpy arrays for now
            denormalized[name] = [p / scaling + translation for p in point]
        else:
            denormalized[name] = point / scaling + translation
    
    # Denormalize vectors
    v_perp_denorm = v_perp / scaling
    v_normal_denorm = v_normal / scaling
    
    if fdi <= 28:
        rotation_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
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
        "outer": denormalized.get('Outer'),
        "gingival": denormalized.get('Gingival'),
        "mesial": denormalized.get('Mesial'),
        "distal": denormalized.get('Distal'),
        "inner": denormalized.get('Inner'),
        "facial": denormalized.get('Facial'),
        "basePlane": {
            "origin": denormalized.get('Bracket'),
            "xAxis": denormalized.get('Incisal'),
            "yAxis": (np.array(denormalized.get('Bracket')) + v_perp_denorm).tolist(),
            "zAxis": (np.array(denormalized.get('Bracket')) + v_normal_denorm).tolist(),
        },
    }
    
    # Add cusps for molars and premolars
    molars_premolars = [14,15,16,17,18,24,25,26,27,28,34,35,36,37,38,44,45,46,47,48]
    if fdi in molars_premolars and 'Cusp' in denormalized:
        json_data["cusps"] = denormalized['Cusp']
    
    # Add planar for molars
    molars = [16,17,18,26,27,28,36,37,38,46,47,48]
    if fdi in molars and 'Planar' in denormalized:
        json_data["planar"] = denormalized['Planar']
    # Filter points for visualization based on tooth type
    # Include 'Outer' in single-tooth plots (only exclude 'Facial')
    plot_points = {k: v for k, v in projected.items() if k != 'Facial'}
    # Only show planar for molars
    molars = [16,17,18,26,27,28,36,37,38,46,47,48]
    if fdi not in molars and 'Planar' in plot_points:
        del plot_points['Planar']
    # Only show cusps for molars and premolars
    molars_premolars = [14,15,16,17,18,24,25,26,27,28,34,35,36,37,38,44,45,46,47,48]
    if fdi not in molars_premolars and 'Cusp' in plot_points:
        del plot_points['Cusp']
    
    if visualize:
        try:
            plot_teeth(plot_points, v_io, v_perp,
                       vertices, patient_id, fdi, output_dir)
        except Exception as e:
            print(f"  ⚠️  Tooth visualization failed: {e}")
    return json_data

def postprocess_predictions(data_folder:Path, visualize:bool = True):
    """
    Post-processes predictions and creates visualizations.
    
    Args:
        data_folder: Path to the data folder containing predictions
        visualize: toggles visualization
    """
    import trimesh
    from visualizers import plot_jaw
    
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
        # Parse tooth_key: expected format "STEM_lower_0002_FDI_47"
        parts = tooth_key.split('_')
        patient_idx = parts.index('lower') if 'lower' in parts else parts.index('upper')
        patient_id = parts[patient_idx + 1]
        fdi_idx = parts.index('FDI')
        fdi = int(parts[fdi_idx + 1])
        stl_file = teeth_path / f"{tooth_key}.stl"
        mesh = trimesh.load_mesh(stl_file)
         
        bracket_pred = predictions.get('Bracket')
        incisal_pred = predictions.get('Incisal')
        outer_pred = predictions.get('OuterPoint')
        gingival_pred = predictions.get('Gingival')
        mesial_pred = predictions.get('Mesial')
        distal_pred = predictions.get('Distal')
        inner_pred = predictions.get('InnerPoint')
        facial_pred = predictions.get('FacialPoint')
        planar_pred = predictions.get('Planar')
        cusp_pred = predictions.get('Cusp')
 
        # Process single tooth predictions to
        # bring them back to normalized coordinates
        points_data = process_tooth_predictions(
            mesh=mesh,
            bracket_pred=bracket_pred,
            incisal_pred=incisal_pred,
            outer_pred=outer_pred,
            gingival_pred=gingival_pred,
            mesial_pred=mesial_pred,
            distal_pred=distal_pred,
            inner_pred=inner_pred,
            facial_pred=facial_pred,
            planar_pred=planar_pred,
            cusp_pred=cusp_pred,
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
        # Parse patient id and FDI from tooth_key (expected like "STEM_lower_0002_FDI_47")
        parts = tooth_key.split('_')
        if 'lower' in parts:
            jaw_idx = parts.index('lower')
            jaw_type = 'lower'
        elif 'upper' in parts:
            jaw_idx = parts.index('upper')
            jaw_type = 'upper'
        else:
            jaw_idx = 1
            jaw_type = 'lower' if 'lower' in tooth_key else 'upper'

        try:
            patient_id_local = parts[jaw_idx + 1]
        except Exception:
            patient_id_local = parts[jaw_idx] if jaw_idx < len(parts) else ''

        try:
            fdi_idx = parts.index('FDI')
            fdi_local = int(parts[fdi_idx + 1])
        except Exception:
            # fallback: try to find a numeric token
            fdi_local = None
            for tok in reversed(parts):
                if tok.isdigit():
                    fdi_local = int(tok)
                    break

        # Load shift file before rotation
        shift = np.array([0.0, 0.0, 0.0])
        shift_file_name = f"STEM_{jaw_type}_{patient_id_local}_shift.json"
        shift_file = data_folder / shift_file_name

        if shift_file.exists():
            try:
                with open(shift_file, 'r') as f:
                    shift_data = json.load(f)
                    shift = np.array(shift_data.get('shift', [0.0, 0.0, 0.0]))
            except Exception as e:
                print(f"  ⚠️ Could not load or parse shift file {shift_file}: {e}")
        else:
            print(f"  ⚠️ Shift file does not exist for patient {patient_id_local}")

        # Choose rotation sequence based on jaw
        if jaw_type == 'lower':
            seq = [('x', -90), ('y', 180)]
        else:
            seq = [('y', 180), ('x', -90), ('y', 180)]

        # Compute combined homogeneous transform matrix (4x4):
        # M = R_seq @ T_shift @ R_upper1 @ T_centroid @ S
        # where S = scale inverse, T_centroid = centroid from tooth json,
        # R_upper1 = initial 180deg Y for upper teeth (fdi <= 28),
        # T_shift = shift file translation, R_seq = final rotation sequence.
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
            if scale != 0:
                S[0, 0] = S[1, 1] = S[2, 2] = 1.0 / float(scale)
            # translation by centroid (from tooth json)
            T_centroid = trimesh.transformations.translation_matrix(centroid.tolist())
            # initial upper rotation applied during denormalization (fdi <= 28)
            if fdi_local is not None and fdi_local <= 28:
                R_upper1 = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
            else:
                R_upper1 = np.eye(4)
            # shift translation (from shift file)
            T_shift = trimesh.transformations.translation_matrix(shift.tolist())
            # final rotation sequence
            R_seq = np.eye(4)
            for axis, degrees in seq:
                radians = np.deg2rad(degrees)
                if axis == 'x':
                    R_axis = trimesh.transformations.rotation_matrix(radians, [1, 0, 0])
                elif axis == 'y':
                    R_axis = trimesh.transformations.rotation_matrix(radians, [0, 1, 0])
                else:
                    R_axis = trimesh.transformations.rotation_matrix(radians, [0, 0, 1])
                R_seq = R_axis @ R_seq

            # Combined matrix
            M = R_seq @ T_shift @ R_upper1 @ T_centroid @ S

            # Apply shift to points, then apply rotations using trimesh
            incisal = np.array(pdata['incisal']) + shift
            outer = np.array(pdata['outer']) + shift
            origin = np.array(pdata['basePlane']['origin']) + shift
            xaxis = np.array(pdata['basePlane']['xAxis']) + shift
            yaxis = np.array(pdata['basePlane']['yAxis']) + shift
            zaxis = np.array(pdata['basePlane']['zAxis']) + shift

            # Apply rotation sequence using trimesh transformations
            for axis, degrees in seq:
                radians = np.deg2rad(degrees)
                if axis == 'x':
                    rotation = trimesh.transformations.rotation_matrix(radians, [1, 0, 0])
                elif axis == 'y':
                    rotation = trimesh.transformations.rotation_matrix(radians, [0, 1, 0])
                else:
                    rotation = trimesh.transformations.rotation_matrix(radians, [0, 0, 1])

                # Apply transformation to points
                incisal = trimesh.transformations.transform_points([incisal], rotation)[0]
                outer = trimesh.transformations.transform_points([outer], rotation)[0]
                origin = trimesh.transformations.transform_points([origin], rotation)[0]
                xaxis = trimesh.transformations.transform_points([xaxis], rotation)[0]
                yaxis = trimesh.transformations.transform_points([yaxis], rotation)[0]
                zaxis = trimesh.transformations.transform_points([zaxis], rotation)[0]

            # Copy all new points with rotation (include 'outer' and other landmarks)
            rotated_entry = {
                'incisal': incisal.tolist(),
                'outer': outer.tolist(),
                'basePlane': {
                    'origin': origin.tolist(),
                    'xAxis': xaxis.tolist(),
                    'yAxis': yaxis.tolist(),
                    'zAxis': zaxis.tolist(),
                },
                # store combined homogeneous transform (from normalized prediction -> rotated space)
                'rotation_matrix': M.tolist(),
            }

            # Add optional points if they exist
            for key in ['gingival', 'mesial', 'distal', 'inner', 'facial', 'bracket']:
                if key in pdata and pdata[key] is not None:
                    try:
                        pt = np.array(pdata[key]) + shift
                        for axis, degrees in seq:
                            radians = np.deg2rad(degrees)
                            if axis == 'x':
                                rotation = trimesh.transformations.rotation_matrix(radians, [1, 0, 0])
                            elif axis == 'y':
                                rotation = trimesh.transformations.rotation_matrix(radians, [0, 1, 0])
                            else:
                                rotation = trimesh.transformations.rotation_matrix(radians, [0, 0, 1])
                            pt = trimesh.transformations.transform_points([pt], rotation)[0]
                        rotated_entry[key] = pt.tolist()
                    except Exception:
                        pass

            # Add cusps for molars/premolars and planar for molars
            molars_premolars = [14, 15, 16, 17, 18, 24, 25, 26, 27, 28, 34, 35, 36, 37, 38, 44, 45, 46, 47, 48]
            molars = [16, 17, 18, 26, 27, 28, 36, 37, 38, 46, 47, 48]

            if fdi_local is not None and fdi_local in molars_premolars and 'cusps' in pdata:
                try:
                    cusps_rot = []
                    for cusp_pt in pdata['cusps']:
                        pt = np.array(cusp_pt) + shift
                        for axis, degrees in seq:
                            radians = np.deg2rad(degrees)
                            if axis == 'x':
                                rotation = trimesh.transformations.rotation_matrix(radians, [1, 0, 0])
                            elif axis == 'y':
                                rotation = trimesh.transformations.rotation_matrix(radians, [0, 1, 0])
                            else:
                                rotation = trimesh.transformations.rotation_matrix(radians, [0, 0, 1])
                            pt = trimesh.transformations.transform_points([pt], rotation)[0]
                        cusps_rot.append(pt.tolist())
                    rotated_entry['cusps'] = cusps_rot
                except Exception:
                    pass

            if fdi_local is not None and fdi_local in molars and 'planar' in pdata:
                try:
                    planar_rot = []
                    for planar_pt in pdata['planar']:
                        pt = np.array(planar_pt) + shift
                        for axis, degrees in seq:
                            radians = np.deg2rad(degrees)
                            if axis == 'x':
                                rotation = trimesh.transformations.rotation_matrix(radians, [1, 0, 0])
                            elif axis == 'y':
                                rotation = trimesh.transformations.rotation_matrix(radians, [0, 1, 0])
                            else:
                                rotation = trimesh.transformations.rotation_matrix(radians, [0, 0, 1])
                            pt = trimesh.transformations.transform_points([pt], rotation)[0]
                        planar_rot.append(pt.tolist())
                    rotated_entry['planar'] = planar_rot
                except Exception:
                    pass

            rotated_points[tooth_key] = rotated_entry
        except Exception as e:
            print(f"⚠️ Error rotating points for {tooth_key}: {e}")

    rotated_output_path = output_reg_path / "projected_points_rotated.json"
    with open(rotated_output_path, 'w') as f: json.dump(rotated_points, f, indent=4)
    print(f"\n💾 Saved rotated projected points to: {rotated_output_path}")
    print(f"\n✅ Post-processing complete.")


def run_bond_with_model(cfg, model, data_folder: Path, visualize: bool = True) -> bool:
    """
    Run bond prediction with a pre-loaded model.
    
    Args:
        cfg: Configuration object
        model: Pre-loaded bond prediction model
        data_folder: Path to data folder containing segmentation results
        visualize: Whether to generate visualizations
        
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
        cfg.no_visuals = not visualize
        
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
    parser.add_argument("--no-visuals", action="store_true", help="Do not generate visualizations")
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