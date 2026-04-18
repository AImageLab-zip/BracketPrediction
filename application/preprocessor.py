from pathlib import Path
import trimesh
import json
import numpy as np

class Preprocessor:
 
    def __init__(self):
        pass

    def preprocess_raw_scans(self, patient_id: str, patient_dir: Path) -> tuple[bool, list[str]]:
        """
        Processes raw scans from the 'raw_data' subdirectory.
        Applies transformations and saves processed scans to the main patient directory.
        Returns (success_status, list_of_raw_files_handled).
        """
        raw_data_dir = patient_dir / "raw_data"
        if not raw_data_dir.is_dir():
            return True, []

        print(f"\n{'='*80}")
        print(f"🔬 Pre-processing RAW SCANS for patient {patient_id}")
        print(f"{'='*80}")

        try:
            # Case-insensitive glob for config file
            config_files = list(raw_data_dir.glob('[cC][oO][nN][fF][iI][gG]_*.json'))
            if not config_files:
                raise StopIteration
            config_file = config_files[0]
        except StopIteration:
            print(f"❌ Pre-processing failed: config_*.json not found in {raw_data_dir}")
            return False, []

        # Case-insensitive glob for STL files
        raw_stl_files = list(raw_data_dir.glob('[sS][tT][eE][mM]_*.[sS][tT][lL]'))
        if not raw_stl_files:
            print(f"❌ Pre-processing failed: No STEM_*.stl files found in {raw_data_dir}")
            return False, [config_file.name]

        all_raw_files = [f.name for f in raw_stl_files] + [config_file.name]

        with open(config_file, 'r') as f:
            config_data = json.load(f)
        # Reshape the flat list of 16 numbers into a 4x4 matrix
        scan_transform_matrix = np.array(config_data["scanTransformMatrix"]).reshape((4, 4))
        for raw_stl_path in raw_stl_files:
            mesh = trimesh.load_mesh(raw_stl_path)
            # Common rotations for both upper and lower scans
            rot_y_180 = trimesh.transformations.rotation_matrix(angle=np.pi, direction=[0, 1, 0])
            rot_x_90 = trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[1, 0, 0])
            mesh.apply_transform(scan_transform_matrix)
            mesh.apply_transform(rot_y_180)
            mesh.apply_transform(rot_x_90)

            if "upper" in raw_stl_path.name.lower():
                # Additional rotation for the upper scan
                # (The segmentator is trained to segment upper scans
                # rotated so that tooth 48 is "overlapped" with tooth 28)
                rot_y_180_extra = trimesh.transformations.rotation_matrix(angle=np.pi, direction=[0, 1, 0])
                mesh.apply_transform(rot_y_180_extra)
                output_filename = raw_stl_path.name
                print(f"  Applied common rotations + extra 180deg Y-rot to {raw_stl_path.name}")

            elif "lower" in raw_stl_path.name.lower():
                output_filename = raw_stl_path.name
                print(f"  Applied common rotations to {raw_stl_path.name}")
            else:
                print(f"  ⚠️ Skipping {raw_stl_path.name}: does not contain 'upper' or 'lower'")
                continue

            centroid = mesh.centroid
            translation_matrix = trimesh.transformations.translation_matrix(-centroid)
            mesh.apply_transform(translation_matrix)

            # Save shif to file (so that it can be applied later to go back to original space.)
            with open(patient_dir / str("{}_{}.json".format(raw_stl_path.stem, "shift")), "w") as shift_file: 
                json.dump({"shift": list(centroid)}, shift_file)

            output_path = patient_dir / output_filename
            mesh.export(output_path)
            print(f"  ✅ Saved processed mesh to {output_path}")

        return True, all_raw_files