"""
Segmentation-only inference: runs just the segmentator (no landmark/bond
model) on a single scan or on every scan found recursively in a folder.

For each scan, writes:
  <stem>_seg.npy           per-vertex segmentation mask
  <stem>_segmentation.ply  colored mesh, one color per tooth class (only with --render-segmentation)
  <stem>_debased.stl       scan with the base plate heuristically removed (only with --debase)
Masks are saved next to each input scan, unless --output is given, in which
case the input folder structure is mirrored inside it (a single-scan input
maps directly to --output).

Example:
python segment.py \
    --input /path/to/scans --output /path/to/masks \
    --seg-config application/app_configs/Pt_semseg_teeth3ds_app.py \
    --seg-weight application/app_weights/segmentator_best.pth \
    --preprocessing preprocessing/3dteethland_preprocessing.yaml \
    --render-segmentation --debase
"""
import os
import sys
sys.path.append(os.path.abspath("application"))
os.environ["VTK_OPENGL_HAS_EGL"] = "0"
import argparse
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

from application.segment_scan import run_segmentation_with_model
from application.utils import load_model
from application.cache import TeethCache
from application.timing import timed, timings
from pointcept.datasets.preprocessing.autobonding.scan_normalizer import ScanNormalizer
from infer import collect_scans, output_dir_for, internal_scan_name

# One color per tooth class (up to 16 per arch); gum (label 0) is gray.
GUM_COLOR = (180, 180, 180)
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195),
]


def vertex_colors_from_mask(mask: np.ndarray) -> np.ndarray:
    """One RGB color per vertex, keyed by its tooth class (0 = gum)."""
    colors = np.full((len(mask), 3), GUM_COLOR, dtype=np.uint8)
    for i, label in enumerate(sorted(l for l in np.unique(mask) if l != 0)):
        colors[mask == label] = PALETTE[i % len(PALETTE)]
    return colors


@timed
def render_segmentation_ply(mesh: trimesh.Trimesh, mask: np.ndarray, out_path: Path):
    """Save a colored .ply; each face's color is the average of its 3 vertices' colors."""
    vertex_colors = vertex_colors_from_mask(mask)
    face_colors = vertex_colors[mesh.faces].mean(axis=1).astype(np.uint8)
    colored = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=face_colors, process=False)
    colored.export(str(out_path))


def _keep_by_expansion(mesh: trimesh.Trimesh, mask: np.ndarray, expansion_mm: float) -> np.ndarray:
    """Keep every vertex within `expansion_mm` of some tooth's own boundary
    vertices (i.e. dilate outward from each tooth, not from a single point).

    Simple and precise around present teeth, but leaves a hole wherever a
    tooth is missing: with nothing there to expand from, that stretch of gum
    is dropped too, and it is mesh-connected straight through to the base, so
    nothing separates the "should have kept this" gap from the real base.
    """
    keep = np.zeros(len(mesh.vertices), dtype=bool)
    for label in np.unique(mask):
        if label == 0:
            continue
        tree = cKDTree(mesh.vertices[mask == label])
        dists, _ = tree.query(mesh.vertices, workers=-1)
        keep |= dists <= expansion_mm
    return keep


def _keep_by_arch_spline(mesh: trimesh.Trimesh, mask: np.ndarray, tube_radius_mm: float,
                          n_samples: int = 300) -> np.ndarray | None:
    """Keep every vertex within `tube_radius_mm` of a smooth curve fit through
    the present teeth's centroids, instead of the teeth themselves.

    Since the curve interpolates continuously across whatever gap a missing
    tooth leaves, it bridges that gap instead of dropping it. Returns None if
    there are too few teeth to fit a meaningful arch (< 3).
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) < 3:
        return None

    # Segmentation labels already follow anatomical arch order (see MAPPING in
    # segment_scan.py: label 1 = FDI 48 ... 8 = FDI 41, 9 = FDI 31 ... 16 = FDI
    # 38, i.e. one last molar, around through the front teeth, to the other
    # last molar), so sorting by label recovers arch order directly — no need
    # to infer it geometrically (which broke down at the arch's open ends).
    ordered = np.array([mesh.vertices[mask == label].mean(axis=0) for label in sorted(labels)])

    tck, _ = splprep(ordered.T, k=min(3, len(ordered) - 1), s=0)
    curve = np.array(splev(np.linspace(0, 1, n_samples), tck)).T

    dists, _ = cKDTree(curve).query(mesh.vertices, workers=-1)
    return dists <= tube_radius_mm


@timed
def debase_mesh(mesh: trimesh.Trimesh, mask: np.ndarray, expansion_mm: float,
                 spline_radius_mm: float) -> trimesh.Trimesh | None:
    """
    Heuristic base-plate removal, dropping the base that usually sits at the
    top/bottom of the scan while keeping every tooth plus a rim of gum. Union
    of two keep-masks for consistency: a per-tooth expansion (precise, but
    gapped where a tooth is missing) and an arch-spline tube (bridges those
    gaps, but is only as accurate as a smooth curve through tooth centroids).
    Returns None if the mask has no teeth.
    """
    keep = _keep_by_expansion(mesh, mask, expansion_mm)
    spline_keep = _keep_by_arch_spline(mesh, mask, spline_radius_mm)
    if spline_keep is not None:
        keep |= spline_keep
    else:
        print("⚠️ Not enough teeth to fit an arch spline, using per-tooth expansion only")
    if not keep.any():
        return None
    faces_kept = np.where(keep[mesh.faces].all(axis=1))[0]
    result = mesh.submesh([faces_kept], append=True)
    assert isinstance(result, trimesh.Trimesh)  # guaranteed by append=True, just not in trimesh's stubs
    return result


class SegmentationPipeline:
    """Loads the segmentator once and runs it scan by scan."""

    def __init__(self, args):
        self.cache = TeethCache()  # avoids re-reading scans from disk; per-tooth splits are discarded
        self.preprocessor = ScanNormalizer(args.preprocessing)
        self.render_segmentation = args.render_segmentation
        self.debase = args.debase
        self.expansion_mm = args.expansion
        self.spline_radius_mm = args.spline
        self.preserve_orientation = args.preserve_orientation
        self.workers = args.workers
        self.processing_time = 0.0  # cumulative segmentation time (excludes I/O)
        print("\n🔄 Loading segmentation model on GPU …")
        self.seg_cfg, self.seg_model = load_model(Path(args.seg_config), Path(args.seg_weight))
        print("✅ Model ready.\n")

    def run(self, scan: Path, out_dir: Path):
        """Segment one scan and export its mask (and optional renderings) to out_dir."""
        work_dir = Path(tempfile.mkdtemp(prefix=f"segment_{scan.stem}_"))
        try:
            self.cache.clear()
            internal_name = internal_scan_name(scan)
            symlink_path = work_dir / internal_name
            symlink_path.symlink_to(scan.resolve())
            self.cache.preload_scan_mesh(symlink_path)

            proc_start = time.perf_counter()
            ok = run_segmentation_with_model(
                cfg=self.seg_cfg,
                model=self.seg_model,
                data_folder=work_dir,
                cache=self.cache,
                preprocessor=self.preprocessor,
                workers=self.workers,
            )
            if not ok:
                raise RuntimeError("segmentation failed")
            self.processing_time += time.perf_counter() - proc_start

            self._export(scan, internal_name, symlink_path, work_dir, out_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _export(self, scan: Path, internal_name: str, symlink_path: Path, work_dir: Path, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        internal_stem = Path(internal_name).stem
        mask = np.load(work_dir / "output_seg" / "result" / f"{internal_stem}_pred.npy")
        np.save(out_dir / f"{scan.stem}_seg.npy", mask)

        if self.render_segmentation or self.debase:
            # Rebuild the exact mesh the mask was computed on: same preprocessing
            # (arch-dependent rotation) and vertex merge postprocess_segmentation applied.
            arch = "lower" if "lower" in internal_name else "upper"
            scan_mesh = self.cache.get_scan_mesh(symlink_path)
            assert scan_mesh is not None, f"{symlink_path.name} was not preloaded into the cache"
            mesh = self.preprocessor.apply(scan_mesh, arch)
            assert isinstance(mesh, trimesh.Trimesh)  # apply() also accepts/returns bare point arrays; we pass a mesh
            mesh.merge_vertices()

            if self.preserve_orientation:
                # Undo the --preprocessing rotation so outputs match the raw
                # scan's own orientation. Safe to do here (rather than before
                # the mask was computed): it's a rigid transform on the same
                # vertex array, so `mask` stays aligned with `mesh.vertices`.
                mesh = self.preprocessor.apply_inverse(mesh, arch)
                assert isinstance(mesh, trimesh.Trimesh)

            if self.render_segmentation:
                render_segmentation_ply(mesh, mask, out_dir / f"{scan.stem}_segmentation.ply")

            if self.debase:
                debased = debase_mesh(mesh, mask, self.expansion_mm, self.spline_radius_mm)
                if debased is not None:
                    debased.export(str(out_dir / f"{scan.stem}_debased.stl"))
                else:
                    print(f"⚠️ {scan.name}: no teeth found in mask, skipping --debase output")

        print(f"💾 Saved segmentation for {scan.name} to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Segments scans using only the segmentator.")
    parser.add_argument("--input",       required=True, help="Scan file (.stl/.obj) or folder scanned recursively")
    parser.add_argument("--output",      help="Output folder (mirrors the input folder structure); "
                                               "defaults to saving masks next to each input scan")
    parser.add_argument("--seg-config",  required=True, help="Segmentation config file")
    parser.add_argument("--seg-weight",  required=True, help="Segmentation model weights")
    parser.add_argument("--preprocessing", help="YAML with per-arch scan normalization transforms")
    parser.add_argument("--preserve-orientation", action="store_true",
                        help="Undo the --preprocessing rotation on --render-segmentation/--debase "
                             "outputs, so they match the raw scan's original orientation instead "
                             "of the (rotated) frame the model runs on")
    parser.add_argument("--render-segmentation", action="store_true",
                        help="Also save a colored .ply of the segmentation (each face colored "
                             "by averaging its 3 vertices' colors)")
    parser.add_argument("--debase",      action="store_true",
                        help="Also save a .stl with the base plate heuristically removed, keeping "
                             "every tooth plus surrounding gum: the union of --expansion around "
                             "each tooth and a --spline tube through the arch")
    parser.add_argument("--expansion",   type=float, default=10.0,
                        help="For --debase: how far (mm) to expand outward from each tooth's own "
                             "vertices (default: 10.0)")
    parser.add_argument("--spline",      type=float, default=10.0,
                        help="For --debase: radius (mm) of the tube kept around a spline fit "
                             "through the tooth centroids; bridges the gaps --expansion leaves "
                             "where a tooth is missing (default: 10.0)")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Worker threads for the CPU/IO-bound postprocessing step (mask "
                             "cleanup + per-tooth mesh splitting). Default: 1 (sequential).")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve() if args.output else None

    scans = []
    for scan in collect_scans(input_path):
        if "lower" in scan.stem or "upper" in scan.stem:
            scans.append(scan)
        else:
            print(f"⚠️ Skipping {scan}: filename must contain 'lower' or 'upper'")
    if not scans:
        sys.exit("❌ No valid scans found.")
    print(f"Found {len(scans)} scan(s) to process.")

    pipeline = SegmentationPipeline(args)
    loop_start = time.perf_counter()
    failed = []
    for i, scan in enumerate(scans, 1):
        print(f"\n{'=' * 70}\n🦷 [{i}/{len(scans)}] {scan.name}\n{'=' * 70}")
        out_dir = scan.parent if output_root is None else output_dir_for(scan, input_path, output_root)
        try:
            pipeline.run(scan, out_dir)
        except Exception:
            traceback.print_exc()
            failed.append(scan)
    total_with_io = time.perf_counter() - loop_start

    succeeded = len(scans) - len(failed)
    print(f"\n✅ Done: {succeeded}/{len(scans)} scans processed.")
    for scan in failed:
        print(f"  ❌ Failed: {scan}")

    print(f"\n⏱️  Total time — with I/O:    {total_with_io:.2f}s for {len(scans)} scan(s) "
          f"({total_with_io / len(scans):.2f}s/scan)")
    if succeeded:
        print(f"⏱️  Total time — without I/O: {pipeline.processing_time:.2f}s for {succeeded} scan(s) "
              f"({pipeline.processing_time / succeeded:.2f}s/scan)")
    timings.report()


if __name__ == "__main__":
    main()
