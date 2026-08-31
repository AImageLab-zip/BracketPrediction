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
            name = raw_stl_path.name.lower()
            if "upper" not in name and "lower" not in name:
                print(f"  ⚠️ Skipping {raw_stl_path.name}: does not contain 'upper' or 'lower'")
                continue

            # Apply only the per-patient scanTransformMatrix here. The fixed
            # standard-orientation rotations (180 Y / 90 X / upper extra 180 Y)
            # now live in production_preprocessing.yaml, applied by the
            # segmentation dataset loader and inverted by
            # bond.postprocess_predictions. No centre-of-mass shift: the
            # segmentator re-centres online (NormalizeCoord) and the landmark
            # model works per normalised tooth, so absolute position is
            # irrelevant.
            mesh = trimesh.load_mesh(raw_stl_path)
            mesh.apply_transform(scan_transform_matrix)

            output_path = patient_dir / raw_stl_path.name
            mesh.export(output_path)
            print(f"  ✅ Saved scanTransformMatrix-aligned mesh to {output_path}")

        return True, all_raw_files