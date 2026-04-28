import os
import sys
sys.path.append(os.path.abspath("application"))
#os.environ["VTK_OPENGL_HAS_EGL"] = "0"
import argparse
import debugpy
from pathlib import Path

#from pointcept.engines.defaults import default_config_parser, default_setup
#from pointcept.models import build_model
#from preprocessor import Preprocessor
#from segment_scan import run_segmentation_with_model
#from bond import postprocess_predictions, run_bond_with_model
#from visualizers import plot_jaw
#from utils import *
from utils import load_model

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

model = LandmarksPredictor(args.data_root,
                           args.seg_config,
                           args.seg_weight,
                           args.bond_config,
                           args.bond_weight)