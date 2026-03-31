"""
This process segments the scan and extracts a mesh for each single tooth.
It operates on the lower scan in the standard alignment and centered in it's
center of mass, while the upper scan is also rotated around the Y axis of 180 degrees
so that tooth 48 is overlapped with lower's tooth 28.
This rotation will be reversed later to go back to the original reference frame.
This script is automatically handled by the monitor, but if you want to debug it here's
the command to run on sample patient "2":

python application/segment_scan.py \
    --config-file /homes/mlugli/BracketPrediction/application/configs/Pt_semseg_app.py \
    --options data_folder=/homes/mlugli/BracketPrediction/application/data/2/ \
              weight=/homes/mlugli/BracketPrediction/application/weights/segmentator_best.pth
"""
import os

MAPPING = {
    1: 48, 2: 47, 3: 46,
    4: 45, 5: 44, 6: 43,
    7: 42, 8: 41, 9: 31,
    10: 32, 11: 33, 12: 34,
    13: 35, 14: 36, 15: 37,
    16: 38
}
import debugpy
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pathlib import Path
from pointcept.engines.launch import launch
import numpy as np
import json
import torch
from scipy.spatial import cKDTree


def normalize(points: np.ndarray, flip:bool=False) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rotate 180° around y-axis (for upper scan), then normalize points to unit sphere centered at origin.
    Returns: (normalized_points, translation, scale)
    """
    # In production, the model runs on a version of the upper scan that is rotated
    # of an extra 180 degrees around the Y axis, such that the lower and upper jaws are
    # "overlapped", not registered anymore. Since during training we don't make the model
    # robust to this sort of flip, we need to save the tooth meshes oriented as the model
    # is used to. The lower jaw scan is untached, so we don't apply the extra rotation.
    if flip:
        rotation_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        rotated_points = points @ rotation_matrix.T
    else: rotated_points = points
    
    centroid = np.mean(rotated_points, axis=0)
    centered_points = rotated_points - centroid
    distances = np.linalg.norm(centered_points, axis=1)
    max_distance = np.max(distances)
    scale = 1.0 / max_distance if max_distance > 0 else 1.0
    normalized_points = centered_points * scale
    
    return normalized_points, centroid, scale


def compute_dilation_masks(mask: np.ndarray, vertices: np.ndarray) -> dict[int, np.ndarray]:
    tooth_labels = np.unique(mask)
    tooth_labels = tooth_labels[tooth_labels != 0]
    verts = vertices.astype(np.float32)

    result = {}
    for label in tooth_labels:
        tooth_verts = verts[mask == label]
        if len(tooth_verts) == 0:
            continue

        bbox_min = tooth_verts.min(axis=0)
        bbox_max = tooth_verts.max(axis=0)
        tooth_size = np.linalg.norm(bbox_max - bbox_min)

        dilation_radius = 0.05 * tooth_size  # tune this
        tree = cKDTree(tooth_verts)
        dists, _ = tree.query(verts, workers=-1)
        result[label] = dists <= dilation_radius

    return result


def postprocess_segmentation(stl_file: Path, mask_file: Path, output_dir: Path, visualize: bool = True):
    """
    Postprocess segmentation results: split by tooth, normalize, and save.
    
    Args:
        stl_file: Path to original STL file
        mask_file: Path to predicted segmentation mask (.npy)
        output_dir: Output directory for processed teeth
    """
    import pyvista as pv
    from visualizers import create_segmentation_visualization
    
    print(f"Postprocessing {stl_file.name}...")
    
    # Load mesh and mask
    mesh = pv.read(stl_file)
    mask = np.load(mask_file)
    if visualize:
        try:
            create_segmentation_visualization(mesh, mask, stl_file.stem, output_dir)
        except Exception as e:
            print(f"  ⚠️  Visualization failed (continuing anyway): {e}") 
    if len(mask) != len(mesh.points):
        print(f"Warning: Mask length ({len(mask)}) doesn't match points ({len(mesh.points)})")
        return
    
    base_name = stl_file.stem
    teeth_output_dir = output_dir / "teeth"
    teeth_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique FDI indices (excluding 0 which is gum)
    unique_fdi_indices = np.unique(mask)
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Remove the '3' prefix from each face
    print(f"Found {len(unique_fdi_indices)} unique classes: {unique_fdi_indices}")

    # Compute dilation masks for including gum around teeth
    dilation_masks = compute_dilation_masks(mask, points)

    # Process teeth 
    for fdi_index in unique_fdi_indices:
        if fdi_index == 0:
            continue  # Skip gum
        
        # Get points belonging to this tooth and nearby neighbors (gum or other teeth)
        # Use dilation mask directly to include all spatially close vertices
        combined_mask = dilation_masks[fdi_index]
        class_indices = np.where(combined_mask)[0]
        if len(class_indices) == 0:
            continue
        
        class_points = points[combined_mask]
        normalized_class_points, translation, scale = normalize(class_points, "upper" in base_name)
        
        face_mask = np.all(np.isin(faces, class_indices), axis=1)
        class_faces_old_idx = faces[face_mask]
        
        # Remap face indices to new point array
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
        class_faces = np.array([[old_to_new[idx] for idx in face] for face in class_faces_old_idx])
        
        # Create PyVista mesh for this tooth
        # Faces need to be in format: [3, v0, v1, v2, 3, v3, v4, v5, ...]
        try:
            faces_pv = np.hstack([[3] + list(face) for face in class_faces])
            tooth_mesh = pv.PolyData(normalized_class_points, faces_pv)
        except:
            print(f"⚠️ Malformed tooth mesh for class {fdi_index}")

 
        # Save STL file
        if "lower" in base_name: fdi_index = MAPPING[fdi_index]
        if "upper" in base_name: fdi_index = MAPPING[fdi_index]-20
        stl_output_path = teeth_output_dir / f"{base_name}_FDI_{fdi_index}.stl"
        tooth_mesh.save(stl_output_path)
 
        # Save normalization parameters to JSON
        json_output_path = teeth_output_dir / f"{base_name}_FDI_{fdi_index}.json"
        json_data = {
            "translation": translation.tolist(),
            "scaling": float(scale)
        }
        
        with open(json_output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"  Saved FDI {fdi_index}: {len(class_points)} points, {len(class_faces)} faces")
        print(f"    STL: {stl_output_path}")
        print(f"    JSON: {json_output_path}")


def run_segmentation_with_model(cfg, model, data_folder: Path, visualize: bool = True) -> bool:
    """
    Run segmentation with a pre-loaded model.
    
    Args:
        cfg: Configuration object
        model: Pre-loaded segmentation model
        data_folder: Path to data folder containing STL files
        visualize: Whether to generate visualizations
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set data paths in config before setup
        cfg._cfg_dict["data_root"] = str(data_folder)
        cfg._cfg_dict["save_path"] = str(Path(data_folder) / "output_seg")
        cfg._cfg_dict["data"]["test"]["data_root"] = str(data_folder)
        cfg.no_visuals = not visualize
        
        os.makedirs(cfg.save_path, exist_ok=True)
        
        # Set up configuration
        cfg = default_setup(cfg)
        
        # Build and run tester with cached model
        test_cfg = dict(cfg=cfg, model=model, **cfg.test)
        tester = TESTERS.build(test_cfg)
        tester.test()
        
        # Postprocessing: split and normalize teeth
        print("\n" + "="*80)
        print("Starting postprocessing...")
        print("="*80 + "\n")
        
        output_folder = Path(cfg.save_path)
        
        # Find STL files in data folder
        stl_files = list(data_folder.glob("*.stl"))
        
        for stl_file in stl_files:
            # Find corresponding prediction mask
            mask_file = output_folder / "result" / f"{stl_file.stem}_pred.npy"
 
            if not mask_file.exists():
                print(f"Warning: No prediction found for {stl_file.name}, skipping...")
                continue
            
            postprocess_segmentation(stl_file, mask_file, output_folder, visualize=visualize)
        
        print("\n" + "="*80)
        print("Postprocessing complete!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in segmentation processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main_worker(cfg):
    os.makedirs(cfg.save_path, exist_ok=True)
    cfg = default_setup(cfg)
    test_cfg = dict(cfg=cfg, **cfg.test)
    tester = TESTERS.build(test_cfg)
    tester.test()
    
    # Postprocessing: split and normalize teeth
    print("\n" + "="*80)
    print("Starting postprocessing...")
    print("="*80 + "\n")
    
    data_folder = Path(cfg.data_root)
    output_folder = Path(cfg.save_path)
    visualize = not cfg.no_visuals
    
    # Find STL files in data folder
    stl_files = list(data_folder.glob("*.stl"))
    
    for stl_file in stl_files:
        # Find corresponding prediction mask
        mask_file = output_folder / "result" / f"{stl_file.stem}_pred.npy"
        
        if not mask_file.exists():
            print(f"Warning: No prediction found for {stl_file.name}, skipping...")
            continue
        
        postprocess_segmentation(stl_file, mask_file, output_folder, visualize=visualize)
    
    print("\n" + "="*80)
    print("Postprocessing complete!")
    print("="*80)


def segment_scan():
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
    cfg._cfg_dict["data_root"] = args.options["data_folder"]
    cfg._cfg_dict["save_path"] = str(Path(args.options["data_folder"]) / "output_seg") 
    cfg._cfg_dict["data"]["test"]["data_root"] = args.options["data_folder"]
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
    segment_scan()
    print("Finished.")