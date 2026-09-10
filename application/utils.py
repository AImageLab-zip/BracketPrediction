import meshlib.mrmeshpy as mr
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import faiss
from timing import *
import json
from pointcept.models import build_model
from pointcept.engines.defaults import default_config_parser, default_setup
import torch
import csv
import trimesh


@timed
def custom_remesh(path:Path,
                  relax_iterations=5,
                  decimate_maxError=0.1,
                  decimate_maxDeletedFaces_ratio=0.5,
                  decimate_subdivideParts=4,
                  decimate_maxTriangleAspectRatio=20.0,
                  ):
    mesh = mr.loadMesh(path)
    mesh.packOptimally()

    relax_params = mr.MeshRelaxParams()
    relax_params.iterations = relax_iterations
    mr.relax(mesh, relax_params)

    settings = mr.DecimateSettings()
    settings.maxError = decimate_maxError
    settings.subdivideParts = decimate_subdivideParts
    settings.maxDeletedFaces = int(mesh.topology.numValidFaces()*decimate_maxDeletedFaces_ratio)
    settings.maxTriangleAspectRatio = decimate_maxTriangleAspectRatio
    mr.decimateMesh(mesh, settings)
    return mesh

@timed
def save_remeshed(mesh, filepath:Path):
    mr.saveMesh(mesh, filepath)

def is_consistent(vertices:np.ndarray, mask:np.ndarray):
    if len(mask) != len(vertices):
        print(f"Error: Mask length ({len(mask)}) doesn't match points ({len(vertices)})")
        return False
    return True

def rotation_180_y():
    return np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])

def parse_tooth(tooth:str) -> tuple[str, str, int]:
    # Two naming conventions are supported:
        # "STEM_lower_0002_FDI_47"
        # "0002_lower_FDI_47"
    if tooth.startswith("STEM"):
        _, arch, patient_id, _, fdi = tooth.split("_")
    else:
        patient_id, arch, _, fdi = tooth.split("_")
    return arch, patient_id, int(fdi)

@timed
def fit_segmask(segmask:np.ndarray, source:np.ndarray, dest:np.ndarray):
    d = 3
    index = faiss.IndexFlatL2(d)
    index.add(source)
    _, indices = index.search(dest, 1)
    remeshed_mask = segmask[indices]
    assert remeshed_mask.shape[0] == dest.shape[0]
    return remeshed_mask
    #create_segmentation_visualization(remeshed, remeshed_mask.squeeze(), "reseshed_segmentation.png", Path(OUT_DIR))

def get_edges(faces:np.ndarray) -> np.ndarray:
    edges = np.unique(np.sort(np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [0, 2]],
    ], axis=0), axis=1), axis=0)
    return edges

def load_json(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not read {path}: {e}")
        return {}

def save_json(path: Path, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"⚠️  Could not save {path}: {e}")

def load_model(config: Path, weights: Path):
    cfg   = default_setup(default_config_parser(str(config), {}))
    model = build_model(cfg.model)
    ckpt  = torch.load(str(weights), weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model = model.cuda().eval()
    print("   ✅ Model loaded")
    return cfg, model


def write_rows(rows:list, output_path:str | Path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "coord_x", "coord_y", "coord_z", "class", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics written to {output_path}")

def teethland_output(json_file: str | Path, write_single=False):
    '''
    Writes a csv file next to the given json file.
    '''
    json_path = Path(json_file)
    predictions = json.load(open(json_path))
    output_path = json_path.with_suffix(".csv")
    rows = []
    # don't need gingival for now
    LANDMARK_KEY_MAP = {
        "planar": "Planar",
        "bracket": "Bracket",
        "incisal": "Incisal",
        "cusps": "Cusp",
        "outer": "OuterPoint",
        "mesial": "Mesial",
        "distal": "Distal",
        "inner": "InnerPoint",
        "facial": "FacialPoint",
        "boundary_mesial": "BoundaryMesial",
        "boundary_distal": "BoundaryDistal",
    }
    for key, vals in predictions.items():
        if key.startswith("STEM"): # old naming
            _, arch, ide, _, fdi = key.split("_")
        else: # new naming
            ide, arch, _, fdi = key.split("_")
        key_base = f"{ide}_{arch}"
        for landmark, coords in vals.items():
            if landmark == "basePlane": continue
            pred_landmark = LANDMARK_KEY_MAP.get(landmark)
            if not pred_landmark: continue
            if landmark in "cusps planar".split():
                for coord in coords:
                    rows.append({
                        "key": key_base,
                        "coord_x": coord[0],
                        "coord_y": coord[1],
                        "coord_z": coord[2],
                        "class": pred_landmark,
                        "score": 1.0
                    })
            else:
                rows.append({
                    "key": key_base,
                    "coord_x": coords[0],
                    "coord_y": coords[1],
                    "coord_z": coords[2],
                    "class": pred_landmark,
                    "score": 1.0
                })
    if write_single: write_rows(rows, output_path)
    return rows

def get_normal_smooth_vector(mesh:trimesh.Trimesh, bracket:np.ndarray, vertices:np.ndarray, scaling:float) -> np.ndarray:
    bracket_mm = bracket / scaling  # Convert to mm space
    vertices_mm = vertices / scaling  # Convert all vertices to mm space
    # Find vertices within 1.5mm radius
    distances = np.linalg.norm(vertices_mm - bracket_mm, axis=1)
    nearby_indices = np.where(distances <= 1.5)[0]
    # Average the vertex normals of nearby vertices
    nearby_normals = mesh.vertex_normals[nearby_indices]
    v_normal = np.mean(nearby_normals, axis=0)
    v_normal = v_normal / np.linalg.norm(v_normal)
    return v_normal

def fit_plane(projected:dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planar_pts = np.array(projected['Planar'])
    center = np.mean(planar_pts, axis=0)
    centered = planar_pts - center
    # SVD to find best fit plane
    U, S, Vt = np.linalg.svd(centered)
    # Normal is the direction with smallest singular value
    v_normal = Vt[2] / np.linalg.norm(Vt[2])
    # Get two orthogonal vectors in the plane
    v1 = Vt[0] / np.linalg.norm(Vt[0])
    v2 = Vt[1] / np.linalg.norm(Vt[1])
    v_io, v_perp = v1, v2
    return v_normal, v_io, v_perp

def get_io_perp(v_normal:np.ndarray, incisal:np.ndarray, outer:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v_io = outer - incisal
    v_io = v_io / np.linalg.norm(v_io)
    # Get perpendicular axis to define plane
    v_perp = np.cross(v_normal, v_io)
    v_perp = v_perp / np.linalg.norm(v_perp)
    return v_io, v_perp


# ===========================================================================
# Mesio-distal width estimation
# ===========================================================================

# Mean mesio-distal crown diameters in mm (Wheeler's Dental Anatomy), used only
# as a catastrophe guard. The tolerance applied to these is deliberately wide
# (~4 SD of real anatomic variation): the job is to catch a sweep that latched
# onto a neighbouring tooth, not to police normal variation.
MD_WIDTH_MM = {
    11: 8.5, 21: 8.5, 12: 6.5, 22: 6.5, 13: 7.5, 23: 7.5,
    14: 7.0, 24: 7.0, 15: 6.7, 25: 6.7, 16: 10.0, 26: 10.0,
    17: 9.0, 27: 9.0, 18: 8.5, 28: 8.5,
    31: 5.0, 41: 5.0, 32: 5.5, 42: 5.5, 33: 7.0, 43: 7.0,
    34: 7.0, 44: 7.0, 35: 7.0, 45: 7.0, 36: 11.0, 46: 11.0,
    37: 10.5, 47: 10.5, 38: 10.0, 48: 10.0,
}

def load_core_points(teeth_path, tooth_key: str, cache=None):
    """Undilated vertices for one tooth, or None if they were not exported.

    These carry the tooth's own segmentation label, without the dilation collar
    of gum and adjacent teeth added by dilate_and_save_teeth.
    """
    if cache is not None:
        return cache.load_core(Path(teeth_path), tooth_key)
    core_file = Path(teeth_path) / f"{tooth_key}.core.npy"
    if not core_file.exists():
        return None
    return np.load(core_file)


def core_vertex_normals(mesh, core_pts) -> np.ndarray:
    """Vertex normals for `core_pts`, looked up on the dilated tooth mesh.

    The core is stored as coordinates, and the STL round-trip re-indexes
    vertices, so the normals are fetched by nearest neighbour rather than by
    index.
    """
    tree = cKDTree(np.asarray(mesh.vertices, dtype=float))
    _, idx = tree.query(np.asarray(core_pts, dtype=float), workers=-1)
    return np.asarray(mesh.vertex_normals, dtype=float)[idx]


def core_submesh(mesh, core_pts, tol_frac: float = 1e-4):
    """The tooth's own surface: faces whose three vertices are all core.

    Projecting onto the full exported mesh would let a mesial/distal point snap
    onto the dilation collar, which on exactly those two sides is the adjacent
    tooth. Returns None when no face survives.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    dist, _ = cKDTree(np.asarray(core_pts, dtype=float)).query(verts, workers=-1)
    is_core = dist <= tol_frac * float(mesh.scale)
    fmask = is_core[np.asarray(mesh.faces)].all(axis=1)
    if not fmask.any():
        return None
    return mesh.submesh([np.flatnonzero(fmask)], append=True)


def plane_section_segments(sub, plane_origin, plane_normal):
    """Cross-section of `sub` as raw (n, 2, 3) line segments, or None.

    Path3D.section is not used here: its .discrete comes back empty unless the
    entities close into loops, which a cross-section of a partial (core-only)
    surface rarely does. mesh_plane hands back the segments directly.
    """
    try:
        segs = np.asarray(trimesh.intersections.mesh_plane(
            sub,
            plane_normal=np.asarray(plane_normal, dtype=float),
            plane_origin=np.asarray(plane_origin, dtype=float)), dtype=float)
    except Exception:
        return None
    if segs.ndim != 3 or len(segs) == 0:
        return None
    return segs


def _closest_on_segments(segs, points):
    """Closest point on a set of line segments.

    Projecting onto the surface freely would pull the two endpoints to
    different heights; confining them to the cross-section at the level the
    chord was measured at keeps them level *and* puts them on the surface.
    """
    a, b = segs[:, 0, :], segs[:, 1, :]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom < 1e-18] = 1e-18

    out = []
    for p in np.atleast_2d(np.asarray(points, dtype=float)):
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out.append(proj[np.argmin(np.linalg.norm(proj - p, axis=1))])
    return np.asarray(out)


def crown_long_axis(incisal=None, gingival=None, fdi=None) -> np.ndarray:
    """Unit vector up the crown, gingival -> incisal.

    NOT +Y. The normalized frame is the segmentator's standard orientation,
    "+Z toward the skull, +X to the patient's right, +Y *outwards*", so +Y is
    the bucco-lingual direction and sweeping along it slices the tooth
    front-to-back instead of up the crown.

    The sign matters as much as the direction: the search band is measured up
    from the low end, so a flipped axis skips the incisal third instead of the
    gingival one. Upper crowns point -Z (the extra 180 deg Y flip in
    normalize()) while lower crowns point +Z, so there is no safe fixed axis.
    Gingival -> Incisal is correctly signed on both arches by construction;
    the per-arch fallback is only for teeth missing either landmark.
    """
    if incisal is not None and gingival is not None:
        v = np.asarray(incisal, dtype=float).reshape(3) - np.asarray(gingival, dtype=float).reshape(3)
        norm = float(np.linalg.norm(v))
        if norm > 1e-9:
            return v / norm
    if fdi is not None:
        return np.array([0.0, 0.0, -1.0]) if int(fdi) <= 28 else np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 0.0, 1.0])


def _nan_movavg(x: np.ndarray, k: int = 3) -> np.ndarray:
    """NaN-aware moving average; windows with no finite sample stay NaN."""
    n = len(x)
    out = np.full(n, np.nan)
    half = k // 2
    for i in range(n):
        seg = x[max(0, i - half): min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def widest_md_chord(core_pts, mesial, distal,
                    core_normals=None, up_axis=None, mesh=None,
                    band_lo_frac=0.30, band_hi_frac=0.95, n_levels=32,
                    slice_frac=0.10, pct=2.0, normal_cos=0.20,
                    slab=None, plane_offset=0.0, plateau_frac=0.02,
                    min_pts=8, fdi=None, mm_per_unit=None, width_tol=0.35,
                    max_shift_frac=None) -> dict:
    """Pull the predicted mesial/distal pair out to the crown's widest chord.

    The heatmap model places Mesial/Distal inside the silhouette, so the pair
    underestimates the mesio-distal crown width. This walks horizontal slices
    up the crown and returns the endpoints of the widest one, both at a single
    height so the chord stays parallel to the predicted M-D line.

    Design notes, in the order they matter:

    * `core_pts` must be the *undilated* vertices. The exported tooth mesh
      carries a collar of adjacent teeth on exactly the mesial and distal
      sides, so measuring the dilated mesh measures the neighbours.
    * The height band is taken from robust percentiles of the core along
      `up_axis`, not from a bounding box, which the collar would inflate.
    * The level is chosen as the centroid of the plateau within
      `plateau_frac` of the smoothed peak, not as a raw argmax over slices.
      An argmax across noisy slices jitters between patients and is biased
      high; the plateau centroid is stable.
    * Optional `normal_cos` gating keeps only vertices whose surface actually
      faces along +/-d, which rejects gingival flare and the occlusal surface
      without a tightly tuned band. It degrades to ungated per slice when it
      would leave too few points.

    Args:
        core_pts: (N, 3) undilated vertices in the normalized tooth frame.
        mesial, distal: predicted landmarks, same frame.
        core_normals: (N, 3) vertex normals for `core_pts`; enables gating.
        up_axis: crown long axis, pointing occlusally. Callers should pass
            crown_long_axis(...): the default +Z is right for the lower arch
            only, and its sign decides whether the band skips the gingival
            third or the incisal one.
        mesh: the exported tooth mesh. When given, the two endpoints are
            projected onto the cross-section of the tooth's own surface at the
            chosen level, so they end up exactly on the mesh without losing
            their common height.
        band_lo_frac, band_hi_frac: search band as fractions of robust height.
        n_levels: horizontal slices swept across the band.
        slice_frac: slice half-thickness as a fraction of robust height.
        pct: tail percentage for the robust per-slice edge.
        normal_cos: min |normal . d| to keep a vertex; 0 disables gating.
        slab: half-thickness of a slab around the M-D plane, as a fraction of
            the M-D length. None (default) measures the true caliper width at
            each height, over every bucco-lingual position.
        plane_offset: shifts that slab bucco-lingually, same units as `slab`.
        plateau_frac: relative tolerance defining the peak plateau.
        min_pts: minimum vertices for a slice edge to be trusted.
        fdi: tooth number; with `mm_per_unit` this enables the width guard.
        mm_per_unit: 1 / scaling for this tooth, i.e. mm per normalized unit.
        width_tol: half-width of the plausible band around MD_WIDTH_MM[fdi],
            as a fraction. A measured width outside it is rejected in favour of
            the prediction, unless the prediction is implausible too.
        max_shift_frac: optional secondary cap on the per-side outward shift,
            as a fraction of the predicted M-D length. Off by default: it is
            scaled by the very quantity that is unreliable, so a badly
            underestimated prediction shrinks its own allowance to correct.

    Returns:
        dict with `boundary_mesial` / `boundary_distal` plus the intermediate
        geometry that debug_shifted_landmarks.py renders.

    Raises:
        ValueError: degenerate input (coincident landmarks, empty core, an up
            axis parallel to the M-D line, no usable slice).
    """
    core = np.asarray(core_pts, dtype=float)
    mesial = np.asarray(mesial, dtype=float).reshape(3)
    distal = np.asarray(distal, dtype=float).reshape(3)
    if core.ndim != 2 or core.shape[0] < min_pts:
        raise ValueError(f"need at least {min_pts} core vertices, got {len(core)}")

    # ---- frame: the crown axis is the reference, d is bent to be level ------
    # Orthogonalising the other way round (up against d) makes "level" mean
    # level with respect to the *predicted* M-D line, which is itself tilted:
    # the leftover height difference is exactly (up . d) * width, several mm on
    # a tooth whose Mesial and Distal come out at different heights. The crown
    # axis is the trustworthy quantity, so it is kept exact and the M-D
    # direction is projected into the plane perpendicular to it - which is also
    # the anatomic definition of mesio-distal crown width.
    up = np.array([0.0, 0.0, 1.0]) if up_axis is None else np.asarray(up_axis, dtype=float).reshape(3)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-9:
        raise ValueError("degenerate crown axis")
    up = up / up_norm

    d = distal - mesial
    md_len_3d = float(np.linalg.norm(d))
    if md_len_3d < 1e-9:
        raise ValueError("mesial and distal predictions coincide")
    d = d / md_len_3d
    tilt = float(d @ up)               # 0 when the prediction is already level
    d = d - tilt * up
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-6:
        raise ValueError("M-D line is parallel to the crown axis")
    d = d / d_norm
    n = np.cross(d, up)
    anchor = 0.5 * (mesial + distal)

    # The predictions no longer sit at py = 0, so measure their half-width in
    # the level plane rather than assuming it.
    half_len = float((distal - anchor) @ d)
    md_len = 2.0 * half_len            # predicted width, projected level
    tilt_deg = float(np.degrees(np.arcsin(min(1.0, abs(tilt)))))

    rel = core - anchor
    pd, py, pn = rel @ d, rel @ up, rel @ n

    # ---- search band from robust core height, never from the bbox ----
    y_lo, y_hi = np.percentile(py, [pct, 100.0 - pct])
    span = float(y_hi - y_lo)
    if span <= 0:
        raise ValueError("degenerate crown height")
    band_lo = y_lo + band_lo_frac * span
    band_hi = y_lo + band_hi_frac * span
    half_slice = 0.5 * slice_frac * span

    # ---- optional bucco-lingual slab around the M-D plane ----
    offset = plane_offset * md_len
    if slab is None:
        delta = None
        in_slab = np.ones(len(core), dtype=bool)
    else:
        delta = slab * md_len
        in_slab = np.abs(pn - offset) <= delta

    # ---- optional normal gating ----
    if core_normals is not None and normal_cos > 0:
        nn = np.asarray(core_normals, dtype=float)
        nn = nn / np.clip(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12, None)
        facing = nn @ d
        gate_mes, gate_dis = facing <= -normal_cos, facing >= normal_cos
    else:
        gate_mes = gate_dis = np.ones(len(core), dtype=bool)

    def _edges_at(level):
        """Robust (mesial, distal) edge along d for the slice at `level`."""
        sl = in_slab & (np.abs(py - level) <= half_slice)
        m_sel, d_sel = sl & gate_mes, sl & gate_dis
        # A gate that empties the slice is worse than no gate at all.
        if m_sel.sum() < min_pts:
            m_sel = sl
        if d_sel.sum() < min_pts:
            d_sel = sl
        if m_sel.sum() < min_pts or d_sel.sum() < min_pts:
            return None
        return (float(np.percentile(pd[m_sel], pct)),
                float(np.percentile(pd[d_sel], 100.0 - pct)),
                m_sel, d_sel)

    # ---- sweep ----
    levels = np.linspace(band_lo, band_hi, n_levels)
    widths = np.full(n_levels, np.nan)
    for i, level in enumerate(levels):
        e = _edges_at(level)
        if e is not None:
            widths[i] = e[1] - e[0]
    if not np.isfinite(widths).any():
        raise ValueError("no slice in the search band held enough core vertices")

    # ---- pick the level: plateau centroid of the smoothed profile ----
    smooth = _nan_movavg(widths, k=3)
    peak = float(np.nanmax(smooth))
    plateau = np.isfinite(smooth) & (smooth >= peak * (1.0 - plateau_frac))
    idx = np.flatnonzero(plateau)
    y0 = float(np.average(levels[idx], weights=smooth[idx]))

    step = levels[1] - levels[0] if n_levels > 1 else 0.0
    band_clipped = bool(y0 <= band_lo + step or y0 >= band_hi - step)

    # ---- edges at the chosen level ----
    e = _edges_at(y0)
    if e is None:
        raise ValueError("chosen level lost its vertices")
    t_mes, t_dis, m_sel, d_sel = e

    # Never pull a point inward: this corrects a known outward bias only.
    t_mes, t_dis = min(t_mes, -half_len), max(t_dis, half_len)
    shift_mes, shift_dis = (-half_len) - t_mes, t_dis - half_len

    # ---- plausibility guards ----
    # Judge the *result*, not the size of the correction. A cap proportional to
    # the predicted width collapses precisely when the prediction is broken,
    # which is the case that most needs correcting.
    fallback, implausible, guard_reason = False, False, ""
    band = None
    if fdi is not None and mm_per_unit:
        mean_mm = MD_WIDTH_MM.get(int(fdi))
        if mean_mm:
            band = (mean_mm * (1.0 - width_tol), mean_mm * (1.0 + width_tol))
            new_mm = (t_dis - t_mes) * mm_per_unit
            pred_mm = md_len * mm_per_unit
            if not (band[0] <= new_mm <= band[1]):
                if band[0] <= pred_mm <= band[1]:
                    fallback = True
                    guard_reason = (f"width {new_mm:.2f} mm outside "
                                    f"[{band[0]:.1f}, {band[1]:.1f}] for FDI {fdi}")
                else:
                    # Prediction is implausible too: keep the measured value,
                    # which at least comes from the geometry, but flag it.
                    implausible = True
                    guard_reason = (f"width {new_mm:.2f} mm and prediction "
                                    f"{pred_mm:.2f} mm both outside "
                                    f"[{band[0]:.1f}, {band[1]:.1f}] for FDI {fdi}")

    if not fallback and max_shift_frac is not None:
        max_shift = max_shift_frac * md_len
        if shift_mes > max_shift or shift_dis > max_shift:
            fallback = True
            guard_reason = f"shift {shift_mes:.3f}/{shift_dis:.3f} over cap {max_shift:.3f}"

    def _pn_at(sel, t):
        """Bucco-lingual position of the vertex nearest the chosen edge, so the
        shifted point hugs the surface instead of the M-D plane."""
        idxs = np.flatnonzero(sel)
        if len(idxs) == 0:
            return offset
        return float(pn[idxs[np.argmin(np.abs(pd[idxs] - t))]])

    if fallback:
        # Hand the predictions back verbatim. They are not level on the crown
        # axis, but fabricating a level pair out of a measurement we just
        # rejected would be worse than saying "could not improve on this".
        t_mes, t_dis = -half_len, half_len
        shift_mes = shift_dis = 0.0
        boundary_mesial, boundary_distal = mesial.copy(), distal.copy()
    else:
        boundary_mesial = anchor + t_mes * d + y0 * up + _pn_at(m_sel, t_mes) * n
        boundary_distal = anchor + t_dis * d + y0 * up + _pn_at(d_sel, t_dis) * n

    # ---- final step: put the points on the mesh ----
    # Confined to the cross-section at y_out, so landing on the surface does not
    # cost the pair its common height. Falls back to a free surface projection,
    # then to the nearest core vertex.
    projected_onto, section = "none", None
    if mesh is not None and not fallback:
        sub = core_submesh(mesh, core)
        if sub is not None:
            pts = np.vstack([boundary_mesial, boundary_distal])
            section = plane_section_segments(sub, anchor + y0 * up, up)
            snapped = _closest_on_segments(section, pts) if section is not None else None
            if snapped is not None:
                projected_onto = "section"
            else:
                try:
                    snapped = trimesh.proximity.closest_point(sub, pts)[0]
                    projected_onto = "surface"
                except Exception:
                    snapped = None
            if snapped is None:
                snapped = core[cKDTree(core).query(pts, workers=-1)[1]]
                projected_onto = "vertex"
            boundary_mesial, boundary_distal = snapped[0], snapped[1]

            # Report what the projected points actually measure.
            t_mes = float((boundary_mesial - anchor) @ d)
            t_dis = float((boundary_distal - anchor) @ d)
            shift_mes = max(0.0, (-half_len) - t_mes)
            shift_dis = max(0.0, t_dis - half_len)

    gap_mes = float(np.min(np.linalg.norm(core - boundary_mesial, axis=1)))
    gap_dis = float(np.min(np.linalg.norm(core - boundary_distal, axis=1)))
    height_delta = float(abs((boundary_mesial - boundary_distal) @ up))

    return {
        "boundary_mesial": boundary_mesial,
        "boundary_distal": boundary_distal,
        "width": float(t_dis - t_mes),
        "md_width_pred": md_len,
        "frame": {"d": d, "up": up, "n": n, "anchor": anchor},
        "offset": offset,
        "delta": delta,
        "band_py": (float(band_lo), float(band_hi)),
        "y0": y0,
        "t_mes": t_mes,
        "t_dis": t_dis,
        "sweep": (levels, smooth),
        "sweep_raw": widths,
        "plateau": plateau,
        "slab": in_slab,
        "shift": (float(shift_mes), float(shift_dis)),
        "surface_gap": (gap_mes, gap_dis),
        "projected_onto": projected_onto,
        "section": section,
        "gate": (gate_mes, gate_dis),
        "half_slice": half_slice,
        "height_delta": height_delta,
        "band_clipped": band_clipped,
        "pred_tilt_deg": tilt_deg,
        "fallback": fallback,
        "implausible": implausible,
        "guard_reason": guard_reason,
        "width_band_mm": band,
    }