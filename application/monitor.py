"""
Automated dental scan processing monitor.
If it finds raw data in patient folders:
1) Applies scanTransformMatrix (found in the Json file) to the mesh;
2) Rotates the scan by 180 degrees around Y axis;
3) Rotates the scan by 90 degrees around X axis;
4) Shifts the scan towards the center of mass and saves the offset in a file;
Automatically runs segmentation and auto bonding.
Tracks individual files to process new files added to existing patient folders.
Usage:
    python application/monitor.py \
        --data-root /workspace/application/data/ \
        --seg-config /workspace/application/configs/Pt_semseg_app.py \
        --seg-weight /workspace/application/weights/segmentator_best.pth \
        --bond-config /workspace/application/configs/Pt_regressor_app.py \
        --bond-weight /workspace/application/weights/regressor_best.pth \
        --check-interval 10
"""
# CRITICAL: Set rendering environment variables BEFORE any imports that use VTK/graphics
import os
os.environ["VTK_OPENGL_HAS_EGL"] = "0"

import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import debugpy
import requests
import torch

from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models import build_model
from preprocessor import Preprocessor
from segment_scan import run_segmentation_with_model
from bond import postprocess_predictions, run_bond_with_model
from visualizers import plot_jaw
from utils import *

# Job status codes — kept in sync with the remote API
PENDING    = 0
PROCESSING = 1
COMPLETED  = 2
FAILED     = 3

class ScanMonitor:
    def __init__(
        self,
        data_root: Path,
        seg_config: Path,
        seg_weight: Path,
        bond_config: Path,
        bond_weight: Path,
        check_interval: int = 10,
        status_file: str = "processing_status.json",
    ):
        self.data_root      = Path(data_root)
        self.check_interval = check_interval
        self.status_file    = self.data_root / status_file
        self.prep           = Preprocessor()

        # Validate required paths
        for label, p in [
            ("Data root",            Path(data_root)),
            ("Segmentation config",  Path(seg_config)),
            ("Segmentation weights", Path(seg_weight)),
            ("Bond config",          Path(bond_config)),
            ("Bond weights",         Path(bond_weight)),
        ]:
            if not p.exists():
                raise ValueError(f"{label} does not exist: {p}")

        # Load both models once at startup
        print("\n🔄 Loading models onto GPU …")
        self.seg_cfg,  self.seg_model  = self._load_model(seg_config,  seg_weight)
        self.bond_cfg, self.bond_model = self._load_model(bond_config, bond_weight)
        print("✅ Both models ready.\n")

        print(f"✅ Monitor initialised")
        print(f"   Data root     : {self.data_root}")
        print(f"   Check interval: {self.check_interval}s")
        print(f"   Status file   : {self.status_file}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(config: Path, weights: Path):
        cfg   = default_setup(default_config_parser(str(config), {}))
        model = build_model(cfg.model)
        ckpt  = torch.load(str(weights), weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        model = model.cuda().eval()
        print("   ✅ Model loaded")
        return cfg, model

    def __del__(self):
        try:
            del self.seg_model, self.bond_model
            torch.cuda.empty_cache()
        except Exception:
            pass

    def load_status(self) -> dict:
        return load_json(self.status_file)

    def save_status(self, status: dict):
        save_json(self.status_file, status)

    def notify_api(self, job_id: str, status: int, message: str = "") -> bool:
        try:
            url     = f"https://autobonding.ing.unimore.it/api/update/{job_id}/"
            headers = {
                "Authorization": f"Bearer {os.getenv('API_TOKEN')}",
                "Content-Type":  "application/json",
            }
            resp = requests.post(url, json={"status": status, "logs": message},
                                 headers=headers, timeout=5)
            resp.raise_for_status()
            print(f"   ✅ API notified: status={status}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  API notification failed: {e}")
            return False

    def find_pending_patients(self) -> list[Path]:
        """Return patient dirs that have work to do."""
        pending = []
        for item in sorted(self.data_root.iterdir()):
            if not item.is_dir():
                continue
            has_raw  = (item / "raw_data").is_dir()
            has_stls = any(item.glob("*.stl"))
            if has_raw or has_stls:
                pending.append(item)
        return pending

    def has_new_raw_files(self, patient_dir: Path) -> bool:
        raw_dir = patient_dir / "raw_data"
        if not raw_dir.is_dir():
            return False
        for raw_stl in raw_dir.glob("[sS][tT][eE][mM]_*.[sS][tT][lL]"):
            if not (patient_dir / raw_stl.name).exists():
                return True
        return False

    def unprocessed_stls(self, patient_dir: Path, patient_status: dict) -> list[str]:
        """STL files in patient_dir not yet marked processed or failed."""
        all_stls      = {f.name for f in sorted(patient_dir.glob("*.stl"))}
        done          = set(patient_status.get("processed_files", []))
        failed        = set(patient_status.get("failed_files", []))
        return sorted(all_stls - done - failed)

    def is_already_running(self, patient_status: dict) -> bool:
        history = patient_status.get("processing_history", [])
        if not history:
            return False
        last = history[-1]
        return (last.get("status") == "processing"
                and "completed_at" not in last
                and "failed_at"    not in last)

    def needs_processing(self, patient_dir: Path, patient_status: dict) -> bool:
        if self.is_already_running(patient_status):
            print(f"  ⏳ {patient_dir.name}: already running, skipping")
            return False
        if self.has_new_raw_files(patient_dir):
            return True
        return len(self.unprocessed_stls(patient_dir, patient_status)) > 0

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

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

    def make_plots(self, patient_dir: Path):
        """Generate all visualisations. Runs after the API has been notified."""
        try:
            postprocess_predictions(patient_dir, visualize=True)
            print("✅ Per-tooth visualisations done")
        except Exception as e:
            print(f"⚠️  Per-tooth visualisation failed: {e}")

        try:
            plot_jaw(patient_dir, raw_scan=True)
            print("✅ Jaw visualisation done")
        except Exception as e:
            print(f"⚠️  Jaw visualisation failed: {e}")

    # ------------------------------------------------------------------
    # Main patient processor
    # ------------------------------------------------------------------

    def process_patient(self, patient_dir: Path, status: dict):
        patient_id = patient_dir.name

        # Ensure the patient has a status entry
        patient_status = status.setdefault(patient_id, {
            "processed_files": [],
            "failed_files":    [],
            "processing_history": [],
        })

        # ── Stage 1: pre-process raw scans if needed ──────────────────
        if self.has_new_raw_files(patient_dir):
            ok, handled = self.prep.preprocess_raw_scans(patient_id, patient_dir)
            if not ok:
                patient_status["processing_history"].append({
                    "started_at":  datetime.now().isoformat(),
                    "status":      "failed",
                    "failed_at":   datetime.now().isoformat(),
                    "error":       "Pre-processing raw scans failed",
                    "files":       handled,
                })
                patient_status["failed_files"] = list(
                    set(patient_status["failed_files"]) | set(handled)
                )
                self.save_status(status)
                print(f"❌ Pre-processing failed for {patient_id}. Aborting.")
                return

        # ── Stage 2: pick up unprocessed STL files ────────────────────
        todo = self.unprocessed_stls(patient_dir, patient_status)
        if not todo:
            print(f"  ℹ️  Nothing new to process for {patient_id}")
            return

        print(f"\n{'#'*70}")
        print(f"# Patient : {patient_id}")
        print(f"# Files   : {todo}")
        print(f"{'#'*70}")

        # Record start
        entry = {
            "started_at": datetime.now().isoformat(),
            "files":      todo,
            "status":     "processing",
        }
        patient_status["processing_history"].append(entry)
        self.save_status(status)

        def fail(reason: str, message: str):
            self.notify_api(patient_id, FAILED, message)
            entry.update({"status": "failed", "failed_at": datetime.now().isoformat(), "error": reason})
            patient_status["failed_files"] = list(set(patient_status["failed_files"]) | set(todo))
            self.save_status(status)

        # ── Segmentation ──────────────────────────────────────────────
        ok, msg = self.run_segmentation(patient_dir)
        if not ok:
            fail("Segmentation failed", msg)
            return

        # ── Bond prediction ───────────────────────────────────────────
        ok, msg = self.run_bond_prediction(patient_dir)
        if not ok:
            fail("Bond prediction failed", msg)
            return

        try:
            postprocess_predictions(patient_dir, visualize=False)
            print("✅ Results saved")
        except Exception as e:
            fail("Post-processing failed", str(e))
            return

        self.notify_api(patient_id, COMPLETED, "Processing completed!")
        self.make_plots(patient_dir)
        entry.update({"status": "completed", "completed_at": datetime.now().isoformat()})
        patient_status["processed_files"] = list(set(patient_status["processed_files"]) | set(todo))
        self.save_status(status)

        print(f"\n{'='*70}")
        print(f"✅ DONE — {patient_id}  |  files: {todo}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print(f"\n{'='*70}")
        print("🔍 Dental Scan Monitor — running")
        print(f"{'='*70}\n")

        try:
            while True:
                # Re-read status each iteration (picks up manual edits)
                status = self.load_status()

                for patient_dir in self.find_pending_patients():
                    patient_status = status.get(patient_dir.name, {})
                    if not self.needs_processing(patient_dir, patient_status):
                        continue

                    print(f"  ▶ {patient_dir.name}: new work found")
                    self.notify_api(patient_dir.name, PROCESSING, "Processing")
                    self.process_patient(patient_dir, status)

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            n_ok   = sum(len(v.get("processed_files", [])) for v in status.values())
            n_fail = sum(len(v.get("failed_files",    [])) for v in status.values())
            print(f"\n⚠️  Stopped by user  |  processed={n_ok}  failed={n_fail}  patients={len(status)}")
        except Exception as e:
            print(f"\n❌ Monitor crashed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description="Monitor and automatically process dental scan directories"
    )
    parser.add_argument("--data-root",      required=True, help="Root directory with patient folders")
    parser.add_argument("--seg-config",     required=True, help="Segmentation config file")
    parser.add_argument("--seg-weight",     required=True, help="Segmentation model weights")
    parser.add_argument("--bond-config",    required=True, help="Bond prediction config file")
    parser.add_argument("--bond-weight",    required=True, help="Bond prediction model weights")
    parser.add_argument("--check-interval", type=int, default=10, help="Seconds between scans (default: 10)")
    parser.add_argument("--status-file",    default="processing_status.json", help="Status filename")
    parser.add_argument("--debug",          action="store_true", help="Wait for debugger on port 5681")
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Waiting for debugger on port 5681 …")
        debugpy.wait_for_client()
        print(">>> Debugger attached.")

    monitor = ScanMonitor(
        data_root=args.data_root,
        seg_config=args.seg_config,
        seg_weight=args.seg_weight,
        bond_config=args.bond_config,
        bond_weight=args.bond_weight,
        check_interval=args.check_interval,
        status_file=args.status_file,
    )
    monitor.run()


if __name__ == "__main__":
    main()