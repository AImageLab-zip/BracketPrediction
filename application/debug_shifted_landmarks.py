"""
Visual debugger for the shifted mesial/distal landmarks
(utils.widest_md_chord).

Walks horizontal slices up the *undilated* tooth, picks the level whose robust
span along the mesio-distal axis is widest, and snaps both endpoints onto the
mesh at that one height. Each panel shows one step of that:

  * panel 1 - SIDE VIEW (d, y): what is searched and what is not. Grey is the
    dilation collar (adjacent teeth) which is excluded; pale blue is the core.
    Orange / green are the chosen slice's vertices that survived the surface
    normal gate on the mesial / distal side, i.e. the ones whose 2nd / 98th
    percentile set the two edges (dotted).
  * panel 2 - FROM ABOVE (d, n): the cross-section (dark outline) that the two
    endpoints were projected onto, so they land exactly on the tooth's own
    surface without losing their common height.
  * panel 3 - WIDTH PROFILE: raw and smoothed sweep curves, the plateau the
    level was averaged over (not an argmax), the predicted width for contrast,
    and the plausible width band for this FDI that guards the result.
  * panel 4 - THE CHOSEN SLICE, ZOOMED: only the vertices that set the width,
    with the gate split and the cross-section, so a bad edge is obvious.

Runs on the segmentation output already on disk (output_seg/teeth + the
predictions.json written by the bond model). Nothing here touches production.

Examples
--------
  python application/debug_shifted_landmarks.py \
      --data-folder application/data_debug/<patient> --tooth 16
  python application/debug_shifted_landmarks.py \
      --data-folder application/data_debug/<patient> --all --band-lo-frac 0.4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import trimesh

from application.utils import widest_md_chord, core_vertex_normals, crown_long_axis


def _resolve_keys(all_predictions, tooth, want_all):
    if want_all:
        return list(all_predictions)
    if tooth in all_predictions:
        return [tooth]
    hits = [k for k in all_predictions if k.endswith(f"_FDI_{tooth}")]
    if not hits:
        raise SystemExit(f"no tooth matching '{tooth}' in predictions.json "
                         f"(have: {', '.join(sorted(all_predictions))})")
    return hits


def debug_one(teeth_path: Path, out_dir: Path,
              tooth_key: str, predictions: dict, sweep_kwargs: dict):
    core_file = teeth_path / f"{tooth_key}.core.npy"
    if not core_file.exists():
        print(f"⏭  {tooth_key}: no {core_file.name} "
              f"(re-run segmentation to generate the undilated core vertices)")
        return
    if predictions.get("Mesial") is None or predictions.get("Distal") is None:
        print(f"⏭  {tooth_key}: model produced no Mesial/Distal")
        return

    transform = json.loads((teeth_path / f"{tooth_key}.json").read_text())
    scaling = float(transform.get("scaling", 1.0))
    mm = (1.0 / scaling) if scaling else 1.0

    mesh = trimesh.load_mesh(teeth_path / f"{tooth_key}.stl")
    core = np.load(core_file)
    mesial = np.asarray(predictions["Mesial"], dtype=float)
    distal = np.asarray(predictions["Distal"], dtype=float)
    fdi = int(tooth_key.rsplit("_", 1)[1])

    r = widest_md_chord(core, mesial, distal,
                        core_normals=core_vertex_normals(mesh, core), mesh=mesh,
                        up_axis=crown_long_axis(predictions.get("Incisal"),
                                                predictions.get("Gingival"), fdi),
                        fdi=fdi, mm_per_unit=mm, **sweep_kwargs)

    d, up, n = r["frame"]["d"], r["frame"]["up"], r["frame"]["n"]
    anchor = r["frame"]["anchor"]
    y0, t_mes, t_dis = r["y0"], r["t_mes"], r["t_dis"]
    band_lo, band_hi = r["band_py"]
    half_slice = r["half_slice"]
    gate_mes, gate_dis = r["gate"]
    levels, widths = r["sweep"]

    def to_frame(p):
        rel = np.atleast_2d(np.asarray(p, dtype=float)) - anchor
        return np.column_stack([rel @ d, rel @ up, rel @ n])   # (pd, py, pn)

    all_f = to_frame(mesh.vertices)
    core_f = to_frame(core)
    mes_f, dis_f = to_frame(mesial)[0], to_frame(distal)[0]
    mes_s_f, dis_s_f = to_frame(r["boundary_mesial"])[0], to_frame(r["boundary_distal"])[0]

    # The vertices that actually set the answer: the chosen slice, split by the
    # normal gate into the ones facing mesially and the ones facing distally.
    in_slice = r["slab"] & (np.abs(core_f[:, 1] - y0) <= half_slice)
    sl_mes, sl_dis = in_slice & gate_mes, in_slice & gate_dis

    # The cross-section the endpoints were projected onto, in frame coords.
    sec_f = None
    if r["section"] is not None:
        sec_f = to_frame(r["section"].reshape(-1, 3)).reshape(-1, 2, 3)

    C_DIL, C_CORE, C_SLICE = "0.86", "#cfe3f5", "#4c9be8"
    C_MES, C_DIS, C_SEC = "#e8834c", "#4caf50", "#1a3a5c"

    def draw_pts(ax, i, j):
        ax.scatter(mes_f[i], mes_f[j], s=130, marker="v", c="purple", edgecolors="k",
                   zorder=6, label="Mesial (pred)")
        ax.scatter(dis_f[i], dis_f[j], s=130, marker="^", c="brown", edgecolors="k",
                   zorder=6, label="Distal (pred)")
        ax.scatter(mes_s_f[i], mes_s_f[j], s=165, marker="v", c="magenta", edgecolors="k",
                   zorder=7, label="Mesial (shifted)")
        ax.scatter(dis_s_f[i], dis_s_f[j], s=165, marker="^", c="red", edgecolors="k",
                   zorder=7, label="Distal (shifted)")

    def draw_section(ax, lw=1.6, label=None):
        if sec_f is not None:
            ax.add_collection(LineCollection(sec_f[:, :, [0, 2]], colors=C_SEC,
                                             lw=lw, zorder=5, label=label))

    fig, axes = plt.subplots(2, 2, figsize=(19, 13.5))

    # ---- 1: side view (d, y) -- core vs collar, the gate, the sweep ---------
    ax = axes[0, 0]
    ax.scatter(all_f[:, 0], all_f[:, 1], s=3, c=C_DIL, label="dilated mesh (collar, NOT searched)")
    ax.scatter(core_f[:, 0], core_f[:, 1], s=4, c=C_CORE, label="core (tooth's own label)")
    ax.scatter(core_f[sl_mes, 0], core_f[sl_mes, 1], s=10, c=C_MES, label="chosen slice, mesial-facing")
    ax.scatter(core_f[sl_dis, 0], core_f[sl_dis, 1], s=10, c=C_DIS, label="chosen slice, distal-facing")
    ax.axhspan(band_lo, band_hi, color="orange", alpha=0.10, label="search band")
    ax.axhspan(y0 - half_slice, y0 + half_slice, color="0.35", alpha=0.16, label="slice thickness")
    ax.axhline(y0, c="0.25", lw=1, ls="--", label="chosen level")
    ax.axvline(t_mes, c="0.5", lw=1, ls=":")
    ax.axvline(t_dis, c="0.5", lw=1, ls=":", label="robust edge (2nd/98th pct)")
    ax.plot([mes_s_f[0], dis_s_f[0]], [mes_s_f[1], dis_s_f[1]], c="red", lw=2.5,
            zorder=5, label="widest chord")
    draw_pts(ax, 0, 1)
    ax.set_xlabel("mesio-distal  d"); ax.set_ylabel("up  y")
    ax.set_title("1 · SIDE VIEW — search the core only, gated by surface normal")
    ax.set_aspect("equal", "box"); ax.legend(fontsize=7, loc="best", framealpha=0.85)

    # ---- 2: from above (d, n) -- the surface the points were snapped to -----
    ax = axes[0, 1]
    ax.scatter(all_f[:, 0], all_f[:, 2], s=3, c=C_DIL, label="dilated mesh")
    ax.scatter(core_f[:, 0], core_f[:, 2], s=4, c=C_CORE, label="core")
    ax.scatter(core_f[in_slice, 0], core_f[in_slice, 2], s=9, c=C_SLICE, label="chosen slice")
    draw_section(ax, label="cross-section at the chosen level")
    ax.plot([mes_s_f[0], dis_s_f[0]], [mes_s_f[2], dis_s_f[2]], c="red", lw=2.5,
            zorder=6, label="widest chord")
    draw_pts(ax, 0, 2)
    ax.set_xlabel("mesio-distal  d"); ax.set_ylabel("bucco-lingual  n")
    ax.set_title(f"2 · FROM ABOVE — endpoints projected onto the mesh "
                 f"(via {r['projected_onto']})")
    ax.set_aspect("equal", "box"); ax.legend(fontsize=7, loc="best", framealpha=0.85)

    # ---- 3: width vs height -- how the level was picked --------------------
    ax = axes[1, 0]
    band = r["width_band_mm"]
    if band:
        for i, edge in enumerate(band):
            ax.axvline(edge / mm, c="green", lw=1.1, ls="--", alpha=0.7,
                       label=(f"plausible for FDI {fdi}  "
                              f"[{band[0]:.1f}, {band[1]:.1f}] mm") if i == 0 else None)
    ax.plot(r["sweep_raw"], levels, c="0.75", marker=".", ms=3, lw=1, label="raw width")
    ax.plot(widths, levels, c=C_SLICE, marker="o", ms=3, label="smoothed")
    plateau = r["plateau"]
    if plateau.any():
        ax.scatter(widths[plateau], levels[plateau], s=50, facecolors="none",
                   edgecolors="green", lw=1.6, zorder=4, label="peak plateau")
    ax.axhspan(band_lo, band_hi, color="orange", alpha=0.10, label="search band")
    ax.axhline(y0, c="0.25", lw=1, ls="--", label="chosen level = plateau centroid")
    ax.axvline(r["md_width_pred"], c="purple", lw=1.2, ls="-.", label="predicted M-D width")
    ax.scatter([r["width"]], [y0], s=95, c="red", zorder=5, label="selected width")
    ax.set_xlabel("chord width along d"); ax.set_ylabel("up  y")
    ax.set_title("3 · WIDTH PROFILE — the level is a plateau centroid, not an argmax")
    ax.legend(fontsize=7, loc="best", framealpha=0.85)

    # ---- 4: the chosen slice alone -- where the width actually comes from ---
    ax = axes[1, 1]
    ax.scatter(core_f[in_slice, 0], core_f[in_slice, 2], s=16, c=C_CORE,
               label="slice vertices")
    ax.scatter(core_f[sl_mes, 0], core_f[sl_mes, 2], s=20, c=C_MES,
               label=f"mesial-facing ({int(sl_mes.sum())})")
    ax.scatter(core_f[sl_dis, 0], core_f[sl_dis, 2], s=20, c=C_DIS,
               label=f"distal-facing ({int(sl_dis.sum())})")
    draw_section(ax, lw=1.8, label="cross-section")
    ax.axvline(t_mes, c="0.45", lw=1.2, ls=":")
    ax.axvline(t_dis, c="0.45", lw=1.2, ls=":", label="robust edge")
    ax.plot([mes_s_f[0], dis_s_f[0]], [mes_s_f[2], dis_s_f[2]], c="red", lw=2.5,
            zorder=6, label="widest chord")
    draw_pts(ax, 0, 2)
    ax.set_xlabel("mesio-distal  d"); ax.set_ylabel("bucco-lingual  n")
    ax.set_title("4 · THE CHOSEN SLICE, ZOOMED — only these vertices set the width")
    ax.set_aspect("equal", "box")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.30))

    d_mes = float(np.linalg.norm(r["boundary_mesial"] - mesial))
    d_dis = float(np.linalg.norm(r["boundary_distal"] - distal))
    gap_mes, gap_dis = r["surface_gap"]
    flag = "  ⚠️ band-clipped" if r["band_clipped"] else ""
    if r["fallback"]:
        flag += f"  ⚠️ GUARD TRIPPED, prediction kept ({r['guard_reason']})"
    elif r["implausible"]:
        flag += f"  ⚠️ {r['guard_reason']}"
    fig.suptitle(
        f"{tooth_key}   |   M-D width {r['md_width_pred'] * mm:.2f} -> {r['width'] * mm:.2f} mm   "
        f"move M/D {d_mes * mm:.2f}/{d_dis * mm:.2f} mm   "
        f"on-mesh via {r['projected_onto']} (gap {gap_mes * mm:.3f}/{gap_dis * mm:.3f} mm)"
        f"   core {len(core)}/{len(mesh.vertices)} verts{flag}",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"shifted_{tooth_key}.png"
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {tooth_key}: mesial {d_mes * mm:.2f} mm / distal {d_dis * mm:.2f} mm  ->  {out_file}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-folder", required=True, type=Path,
                    help="patient dir containing output_seg/ and output_reg/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tooth", help="FDI number (e.g. 16) or full tooth_key")
    g.add_argument("--all", action="store_true", help="every tooth in predictions.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for PNGs (default: <data-folder>/output_reg/plots)")
    # see utils.widest_md_chord
    ap.add_argument("--slab", type=float, default=None,
                    help="restrict the search to a slab around the M-D plane, half-thickness "
                         "as a fraction of the M-D length. Default: no slab, i.e. the true "
                         "caliper width over every bucco-lingual position")
    ap.add_argument("--band-lo-frac", type=float, default=0.30,
                    help="bottom of the search band, fraction of crown height (skips the gingival third)")
    ap.add_argument("--band-hi-frac", type=float, default=0.95,
                    help="top of the search band (1.0 = incisal/occlusal edge)")
    ap.add_argument("--n-levels", type=int, default=32, help="horizontal slices swept across the band")
    ap.add_argument("--slice-frac", type=float, default=0.10,
                    help="slice half-thickness, fraction of the crown height span")
    ap.add_argument("--pct", type=float, default=2.0, help="tail %% for the robust edge")
    ap.add_argument("--normal-cos", type=float, default=0.20,
                    help="min |vertex normal . M-D axis| to keep a vertex; 0 disables gating")
    ap.add_argument("--plateau-frac", type=float, default=0.02,
                    help="relative tolerance defining the peak plateau")
    ap.add_argument("--width-tol", type=float, default=0.35,
                    help="plausible-width band around the FDI norm, as a fraction")
    ap.add_argument("--max-shift-frac", type=float, default=None,
                    help="optional secondary per-side shift cap, fraction of the "
                         "predicted M-D length (off by default)")
    ap.add_argument("--plane-offset", type=float, default=0.0,
                    help="shift the slab off the M-D plane, bucco-lingual (fraction)")
    args = ap.parse_args()

    teeth_path = args.data_folder / "output_seg" / "teeth"
    pred_file = args.data_folder / "output_reg" / "results" / "predictions.json"
    if not pred_file.exists():
        raise SystemExit(f"missing {pred_file}")
    all_predictions = json.loads(pred_file.read_text())

    out_dir = args.out or (args.data_folder / "output_reg" / "plots")
    sweep_kwargs = dict(slab=args.slab, band_lo_frac=args.band_lo_frac,
                        band_hi_frac=args.band_hi_frac, n_levels=args.n_levels,
                        slice_frac=args.slice_frac, pct=args.pct,
                        normal_cos=args.normal_cos, plateau_frac=args.plateau_frac,
                        width_tol=args.width_tol, max_shift_frac=args.max_shift_frac,
                        plane_offset=args.plane_offset)

    for key in _resolve_keys(all_predictions, args.tooth, args.all):
        try:
            debug_one(teeth_path, out_dir, key, all_predictions[key], sweep_kwargs)
        except Exception as e:
            print(f"❌ {key}: {e}")


if __name__ == "__main__":
    main()
