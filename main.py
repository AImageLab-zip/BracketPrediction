import os
import sys
sys.path.append(os.path.abspath("application"))
#os.environ["VTK_OPENGL_HAS_EGL"] = "0"
import argparse
import debugpy
from pathlib import Path
import traceback

TEST_PATIENT = '/homes/mlugli/BracketPrediction/application/app_data/0ab54769-02f8-49d7-b13b-b1d7d85db18a'
TEST_RESULTS = '/homes/mlugli/BracketPrediction/application/app_data/0ab54769-02f8-49d7-b13b-b1d7d85db18a/output_reg/results/projected_points.json'
#from pointcept.engines.defaults import default_config_parser, default_setup
#from pointcept.models import build_model
#from preprocessor import Preprocessor
from application.segment_scan import run_segmentation_with_model
from application.bond import run_bond_with_model, postprocess_predictions
#from visualizers import plot_jaw
from application.utils import load_model, prepare_metrics
from application.timing import *
import shutil
from application.visualizers import json_to_ply

class LandmarksPredictor:
    def __init__(
        self,
        data_root: str,
        seg_config: str,
        seg_weight: str,
        bond_config: str,
        bond_weight: str,
    ):
        self.data_root = Path(data_root)
        self.seg_config = Path(seg_config)
        self.seg_weight= Path(seg_weight)
        self.bond_config = Path(bond_config)
        self.bond_weight = Path(bond_weight)
        # Load both models once
        print("\n🔄 Loading models on GPU …")
        self.seg_cfg,  self.seg_model  = load_model(self.seg_config,  self.seg_weight)
        self.bond_cfg, self.bond_model = load_model(self.bond_config, self.bond_weight)
        print("✅ Both models ready.\n")
        print(f"   Data root     : {self.data_root}")

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
                cfg=self.seg_cfg, model=self.seg_model, data_folder=patient_dir
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
                cfg=self.bond_cfg, model=self.bond_model, data_folder=patient_dir
            )
            msg = f"Bond prediction {'completed' if ok else 'failed'} for {patient_dir.name}"
            return ok, msg
        except Exception as e:
            traceback.print_exc()
            return False, str(e)
    
    def predict(self, directory:Path, clean_previous=True, postprocess=False):
        if clean_previous: self._clean_outputs(directory)
        ok, msg = self.run_segmentation(directory)
        if not ok:
            print("Segmentation failed {}".format(msg))
            return
        ok, msg = self.run_bond_prediction(directory)
        if not ok:
            print("Bond prediction failed {}".format(msg))
            return
        if postprocess:
            try:
                postprocess_predictions(directory, visualize=False)
                print("✅ Results saved")
                json_to_ply(directory / "output_reg" / "results" / "projected_points.json",
                            directory / "output_reg" / "results" / "projected_points.ply")
            except Exception as e:
                print("Post-processing failed {}".format(str(e)))
                return


parser = argparse.ArgumentParser(
    description="Segments and predicts landmarks on a oriented scan."
)
parser.add_argument("--debug",          action="store_true", help="Wait for debugger on port 5681")
parser.add_argument("--data-root",      required=True, help="Root directory with patient folders")
parser.add_argument("--seg-config",     required=True, help="Segmentation config file")
parser.add_argument("--seg-weight",     required=True, help="Segmentation model weights")
parser.add_argument("--bond-config",    required=True, help="Bond prediction config file")
parser.add_argument("--bond-weight",    required=True, help="Bond prediction model weights")

args = parser.parse_args()
if args.debug:
    debugpy.listen(("0.0.0.0", 5681))
    print(">>> Waiting for debugger on port 5681 …")
    debugpy.wait_for_client()
    print(">>> Debugger attached.")

#model = LandmarksPredictor(args.data_root,
#                           args.seg_config,
#                           args.seg_weight,
#                           args.bond_config,
#                           args.bond_weight)
#
#model.predict(Path(TEST_PATIENT), clean_previous=True, postprocess=True)
#timings.report()
prepare_metrics(TEST_RESULTS)