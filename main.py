import os
import sys
from typing import Iterator
import uuid
from datetime import datetime
sys.path.append(os.path.abspath("application"))
os.environ["VTK_OPENGL_HAS_EGL"] = "0"
import argparse
import debugpy
from pathlib import Path
import traceback
import json
import pickle
from collections import defaultdict
from application.segment_scan import run_segmentation_with_model
from application.bond import run_bond_with_model, postprocess_predictions
from application.utils import load_model, teethland_output, write_rows
from application.timing import *
from application.cache import TeethCache
import shutil
from application.visualizers import json_to_ply
from pointcept.datasets.preprocessing.autobonding.scan_normalizer import ScanNormalizer
import tempfile

def remove_temp_folder(temp_dir:Path):
    try:
        shutil.rmtree(str(temp_dir))
        print(f"Removed temporary folder {temp_dir}")
    except Exception as e:
        print(f"⚠️ Could not remove temporary folder {temp_dir}: {e}")

def get_samples(L:list[Path]) -> list[str]:
    all_samples = []
    for file in L:
        with open(file) as f:
            samples = f.read().splitlines()
            all_samples += samples
    return all_samples

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

    def _clear_cache(self):
        if self.cache:
            self.cache.clear_teeth_data()
    
    def predict(self, directory:Path, clean_previous=True, postprocess=False):
        if clean_previous: self._clean_outputs(directory)
        self._clear_cache()
        ok, msg = self.run_segmentation(directory)
        if not ok:
            print("Segmentation failed {}".format(msg))
            return
        ok, msg = self.run_bond_prediction(directory)
        if not ok:
            print("Bond prediction failed {}".format(msg))
            return
        if not postprocess: return
        postprocess_predictions(directory, 
            visualize=False, 
            cache=self.cache,
            preprocessor=self.preprocessor)
        print("✅ Results saved")
        if self.save_ply:
            json_to_ply(directory / "output_reg" / "results" / "landmarks.json",
                        directory / "output_reg" / "results" / "landmarks.ply")

def _merge_gold(kpt_path: Path, merged_gold: dict):
    patient_id = kpt_path.stem.replace("__kpt", "")
    with open(kpt_path, "r") as f:
        data = json.load(f)
    for obj in data["objects"]:
        merged_gold[obj["class"]][patient_id].append(obj["coord"])


def _setup_exp_dir(output_folder: Path, model: LandmarksPredictor) -> Path:
    """Create a timestamped experiment directory and copy model configs into it."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = output_folder / f"{timestamp}_{uuid.uuid4().hex[:6]}"
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy(model.bond_config, exp_dir)
    shutil.copy(model.seg_config, exp_dir)
    return exp_dir


def _iter_patient_files(
    dataset_path: Path,
    test_samples: set[str],
    collect_gt: bool,
) -> Iterator[tuple[Path, Path, Path | None]]:
    """
    Yield (patient_dir, mesh_filepath, kpt_path_or_None) for each matching patient.
    Raises FileNotFoundError if collect_gt is True and a GT file is missing.
    """
    for arch in ("lower", "upper"):
        dirpath = dataset_path / arch
        if not dirpath.exists():
            continue
        for patient in (d for d in os.listdir(dirpath) if os.path.isdir(dirpath / d)):
            base = dirpath / patient / f"{patient}_{arch}"
            filepath = next(
                (base.with_suffix(ext) for ext in (".obj", ".stl") if base.with_suffix(ext).exists()),
                None,
            )
            if not filepath or f"{patient}_{arch}" not in test_samples:
                continue
            kpt_path = None
            if collect_gt:
                kpt_path = dirpath / patient / f"{patient}_{arch}__kpt.json"
                if not kpt_path.exists():
                    raise FileNotFoundError(
                        f"GT file missing for {patient}_{arch}. Disable --collect-gt or ensure GT files exist."
                    )
            yield dirpath / patient, filepath, kpt_path

def _save_gold(merged_gold: dict, output_folder: Path) -> None:
    """Serialize the merged gold standard dict to a pickle file."""
    final_gold = {cls: dict(patients) for cls, patients in merged_gold.items()}
    out_path = output_folder / "gold_standard.pkl"
    print("Saving GT pickle to {}".format(out_path))
    with open(out_path, "wb") as f:
        pickle.dump(final_gold, f)


@timed
def _run_prediction_timed(model: LandmarksPredictor, temp_dir: Path) -> None:
    """Measure processing only (segmentation + bonding + postprocessing)."""
    model.predict(temp_dir, clean_previous=False, postprocess=True)

def test_3dteethland_singles(
    dataset_path: Path,
    files: list[str],
    model: LandmarksPredictor,
    output_folder: Path,
    collect_gt: bool = False,
):
    exp_dir = _setup_exp_dir(output_folder, model)
    merged_gold = defaultdict(lambda: defaultdict(list))
    rows = []

    for patient_dir, _, kpt_path in _iter_patient_files(dataset_path, set(files), collect_gt):
        model.predict(patient_dir, clean_previous=True, postprocess=True)
        rows += teethland_output(patient_dir / "output_reg" / "results" / "landmarks.json")
        if kpt_path:
            _merge_gold(kpt_path, merged_gold)

    write_rows(rows, exp_dir / "predictions.csv")
    if collect_gt:
        _save_gold(merged_gold, output_folder)

def test_3dteethland_optimized(
    dataset_path: Path,
    model: LandmarksPredictor,
    files: list[str],
    output_folder: Path,
    collect_gt: bool = False,
):
    exp_dir = _setup_exp_dir(output_folder, model)
    temp_dir = Path(tempfile.mkdtemp(prefix="scans_", dir=exp_dir))
    print(f"Created temporary scans folder: {temp_dir}")
    merged_gold = defaultdict(lambda: defaultdict(list))
    print(f"Pre-loading scans...")
    for _, filepath, kpt_path in _iter_patient_files(dataset_path, set(files), collect_gt):
        if model.cache:
            model.cache.preload_scan_mesh(filepath)
        (temp_dir / filepath.name).symlink_to(filepath)
        if kpt_path:
            _merge_gold(kpt_path, merged_gold)

    _run_prediction_timed(model, temp_dir)
    rows = teethland_output(temp_dir / "output_reg" / "results" / "landmarks.json")
    write_rows(rows, exp_dir / "predictions.csv")
    if collect_gt:
        _save_gold(merged_gold, output_folder)

parser = argparse.ArgumentParser(
    description="Segments and predicts landmarks on a oriented scan."
)
parser.add_argument("--debug",          action="store_true", help="Wait for debugger on port 5681")
# ===================== DATA PATHS ========================
parser.add_argument("--samples",        nargs="+", help="paths to files containing testing filenames")
parser.add_argument("--data-folder",    help="Absolute path of the data folder")
parser.add_argument("--output-folder",  help="Absolute path of the output folder where predictions will be saved.")
# ============= MODEL WEIGHTS AND CONFIGS =================
parser.add_argument("--seg-config",     required=True, help="Segmentation config file")
parser.add_argument("--seg-weight",     required=True, help="Segmentation model weights")
parser.add_argument("--bond-config",    required=True, help="Bond prediction config file")
parser.add_argument("--bond-weight",    required=True, help="Bond prediction model weights")
# ================== OPTIONALS =============================
parser.add_argument("--remesh",         required=False, action="store_true", help="Enables remeshing of scans")
parser.add_argument("--preprocessing",  required=False, help="Preprocessing")
parser.add_argument("--vis-seg",        required=False, action="store_true", help="Renders the 3D segmentation")
parser.add_argument("--save-ply",       required=False, action="store_true", help="Saves landmarks as point cloud")
parser.add_argument("--cache",          required=False, action="store_true", help="Cache teeth meshes in memory")
parser.add_argument("--collect-gt",     required=False, action="store_true", help="Looks for __kpt.json files and stores them in a pickle object.")

args = parser.parse_args()
if args.debug:
    debugpy.listen(("0.0.0.0", 5681))
    print(">>> Waiting for debugger on port 5681 …")
    debugpy.wait_for_client()
    print(">>> Debugger attached.")

cache = TeethCache() if args.cache else None
if cache: print("💾 Teeth caching enabled\n")
flattened = "/homes/mlugli/BracketPrediction/Teeth3DS/original_test_set_flattened"
all_files = get_samples(args.samples)
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

test_3dteethland_optimized(
    Path(args.data_folder),
    model,all_files,
    Path(args.output_folder),
    args.collect_gt)
timings.report()




# Copy outputs into experiment folder for persistence, then cleanup temp dir
#try:
#    for name in ("output_reg", "output_seg"):
#        src = temp_dir / name
#        dst = exp_dir / name
#        if src.exists():
#            if dst.exists():
#                shutil.rmtree(dst)
#            shutil.copytree(src, dst)
#except Exception as e:
#    print(f"⚠️ Could not copy output directories to {exp_dir}: {e}")
#remove_temp_folder(temp_dir)
