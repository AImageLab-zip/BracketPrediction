import os
import sys
sys.path.append(os.path.abspath("application"))
os.environ["VTK_OPENGL_HAS_EGL"] = "0"
import argparse
import debugpy
from pathlib import Path
import traceback
import json
import pickle
from collections import defaultdict


#TESTING_LOWER = "/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_debug/testing_lower.txt"
#TESTING_UPPER = "/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_debug/testing_upper.txt"

#TESTING_LOWER = '/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_test_split_original/testing_lower.txt'
#TESTING_UPPER = '/homes/mlugli/BracketPrediction/Teeth3DS/splits/3DTeethland_challenge_train_test_split_original/testing_upper.txt'
#TEETHLAND_DIR = '/homes/mlugli/BracketPrediction/Teeth3DS/original_data'
DIR = '/work/grana_maxillo/Mlugli/T4M'
TESTING = '/work/grana_maxillo/Mlugli/T4M/test.txt'

from application.segment_scan import run_segmentation_with_model
from application.bond import run_bond_with_model, postprocess_predictions
from application.utils import load_model, teethland_output, write_rows
from application.timing import *
from application.cache import TeethCache
import shutil
from application.visualizers import json_to_ply
from pointcept.datasets.preprocessing.autobonding.scan_normalizer import ScanNormalizer

class LandmarksPredictor:
    def __init__(
        self,
        seg_config: str,
        seg_weight: str,
        bond_config: str,
        bond_weight: str,
        remesh: bool,
        visualize_segmentation: bool,
        save_ply: bool,
        cache: TeethCache | None = None,
        preprocessing = None,
    ):
        self.seg_config = Path(seg_config)
        self.seg_weight= Path(seg_weight)
        self.bond_config = Path(bond_config)
        self.bond_weight = Path(bond_weight)
        self.remesh = remesh
        self.visualize_segmentation = visualize_segmentation
        self.save_ply = save_ply
        self.cache = cache
        self.preprocessing = preprocessing
        if self.preprocessing:
            self.preprocessor = ScanNormalizer(self.preprocessing)
        print("\n🔄 Loading models on GPU …")
        self.seg_cfg,  self.seg_model  = load_model(self.seg_config,  self.seg_weight)
        self.bond_cfg, self.bond_model = load_model(self.bond_config, self.bond_weight)
        print("✅ Both models ready.\n")

    def _clean_outputs(self, directory:Path):
        for name in ("output_reg", "output_seg"):
            target = directory / name
            if target.exists() and target.is_dir():
                shutil.rmtree(target)

    @timed
    def run_segmentation(self, patient_dir: Path) -> tuple[bool, str]:
        print(f"\n{'='*70}\n🦷 SEGMENTATION — {patient_dir.name}\n{'='*70}")
        try:
            ok = run_segmentation_with_model(
                cfg=self.seg_cfg, 
                model=self.seg_model, 
                data_folder=patient_dir, 
                remesh = self.remesh,
                visualize=self.visualize_segmentation,
                cache=self.cache,
                preprocessor=self.preprocessor
            )
            msg = f"Segmentation {'completed' if ok else 'failed'} for {patient_dir.name}"
            return ok, msg
        except Exception as e:
            traceback.print_exc()
            return False, str(e)

    @timed
    def run_bond_prediction(self, patient_dir: Path) -> tuple[bool, str]:
        print(f"\n{'='*70}\n📍 BOND PREDICTION — {patient_dir.name}\n{'='*70}")
        try:
            ok = run_bond_with_model(
                cfg=self.bond_cfg, 
                model=self.bond_model, 
                data_folder=patient_dir,
                cache=self.cache,
            )
            msg = f"Bond prediction {'completed' if ok else 'failed'} for {patient_dir.name}"
            return ok, msg
        except Exception as e:
            traceback.print_exc()
            return False, str(e)
    
    def predict(self, directory:Path, clean_previous=True, postprocess=False):
        if clean_previous: self._clean_outputs(directory)
        if cache: self.cache.clear()
        ok, msg = self.run_segmentation(directory)
        if not ok:
            print("Segmentation failed {}".format(msg))
            return
        ok, msg = self.run_bond_prediction(directory)
        if not ok:
            print("Bond prediction failed {}".format(msg))
            return
        if postprocess:
            postprocess_predictions(directory, 
                visualize=False, 
                cache=self.cache,
                preprocessor=self.preprocessor)
            print("✅ Results saved")
            if self.save_ply:
                json_to_ply(directory / "output_reg" / "results" / "landmarks.json",
                            directory / "output_reg" / "results" / "landmarks.ply")

                #json_to_ply(directory / "output_reg" / "results" / "projected_points_rotated.json",
                #            directory / "output_reg" / "results" / "projected_points_rotated.ply")

def _merge_gold(kpt_path: Path, merged_gold: dict):
    patient_id = kpt_path.stem.replace("__kpt", "")
    with open(kpt_path, "r") as f:
        data = json.load(f)
    for obj in data["objects"]:
        merged_gold[obj["class"]][patient_id].append(obj["coord"])

def test_3dteethland(dataset_path:Path, files:list[str], model:LandmarksPredictor):
    test_samples = set(files)
    rows = []
    merged_gold = defaultdict(lambda: defaultdict(list))
    for arch in ["lower", "upper"]:
        dirpath = dataset_path / arch
        dirs = [d for d in os.listdir(dirpath) if os.path.isdir(dirpath / d)]
        for patient in dirs:
            for ext in (".obj", ".stl"):
                filepath = dirpath / patient / f"{patient}_{arch}{ext}"
                if filepath.exists(): break
            if filepath.exists() and  f"{patient}_{arch}" in test_samples:
                model.predict(dirpath / patient, clean_previous=True, postprocess=True)
                rows += teethland_output(dirpath / patient / "output_reg" / "results" / "landmarks.json")
                kpt_path = dirpath / patient / f"{patient}_{arch}__kpt.json"
                if kpt_path.exists(): _merge_gold(kpt_path, merged_gold)
    write_rows(rows, '/homes/mlugli/BracketPrediction/evaluation/our_predictions.csv')
    final_gold = {cls: dict(patients) for cls, patients in merged_gold.items()}
    output_pickle = Path('/homes/mlugli/BracketPrediction/evaluation/our_predictions_gold.pkl')
    with open(output_pickle, "wb") as f: pickle.dump(final_gold, f)

parser = argparse.ArgumentParser(
    description="Segments and predicts landmarks on a oriented scan."
)
parser.add_argument("--debug",          action="store_true", help="Wait for debugger on port 5681")
parser.add_argument("--seg-config",     required=True, help="Segmentation config file")
parser.add_argument("--seg-weight",     required=True, help="Segmentation model weights")
parser.add_argument("--bond-config",    required=True, help="Bond prediction config file")
parser.add_argument("--bond-weight",    required=True, help="Bond prediction model weights")
parser.add_argument("--remesh",         required=False, action="store_true", help="Enables remeshing of scans")
parser.add_argument("--preprocessing",  required=False, help="Preprocessing")
parser.add_argument("--vis-seg",        required=False, action="store_true", help="Renders the 3D segmentation")
parser.add_argument("--save-ply",       required=False, action="store_true", help="Saves landmarks as point cloud")
parser.add_argument("--cache",          required=False, action="store_true", help="Cache teeth meshes in memory")

args = parser.parse_args()
if args.debug:
    debugpy.listen(("0.0.0.0", 5681))
    print(">>> Waiting for debugger on port 5681 …")
    debugpy.wait_for_client()
    print(">>> Debugger attached.")

cache = TeethCache() if args.cache else None
if cache: print("💾 Teeth caching enabled\n")

model = LandmarksPredictor(args.seg_config,
                           args.seg_weight,
                           args.bond_config,
                           args.bond_weight,
                           args.remesh,
                           args.vis_seg,
                           args.save_ply,
                           cache=cache,
                           preprocessing=args.preprocessing
                           )

#lower_files = open(TESTING_LOWER).read().splitlines()
#upper_files = open(TESTING_UPPER).read().splitlines()
#all_files = lower_files + upper_files
all_files = open(TESTING).read().splitlines()
test_3dteethland(Path(DIR), all_files, model)
timings.report()