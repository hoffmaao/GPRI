#!/usr/bin/env python3
"""Audit the stable-ground reference against the Randolph Glacier Inventory.

    python examples/baker_rgi.py --scene 20170803 --decimate 16

The pipeline's "bedrock" reference is coherence-chosen, and high coherence
does not mean stationary — slowly moving ice stays coherent at a 2-minute
pair spacing.  If part of the reference mask sits on glacier, every
bedrock-referenced product is tied to moving ground: real motion is
subtracted out of the maps and an artificial signal is pushed onto rock.

This script puts the RGI outlines into the radar's map frame and reports how
much of the coherence-only mask the inventory contradicts, writes the overlay
figure (which doubles as a check on the scan heading — with the heading wrong
the outlines land in the wrong place against the backscatter), and prints the
corrected mask statistics.  RGI file location comes from ``GPRI_RGI`` in
``site.env`` or defaults to ``data/rgi/rgi_61.zip``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "gpri_site", Path(__file__).resolve().parent / "site.py")
_site = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_site)
_site.load_site()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baker_aps import SCENES                                       # noqa: E402
from baker_north_side import decimated_par, read_backdrop          # noqa: E402

from gpri.geocode import BAKERBEND1_HEADING, RadarGeometry, geocode  # noqa: E402
from gpri.glaciers import (glacier_mask, load_outlines, outline_paths,  # noqa: E402
                           stable_ground_mask)
from gpri.stack import DiffStack                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--heading", type=float, default=BAKERBEND1_HEADING)
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--buffer", type=float, default=100.0,
                    help="metres to stand clear of the RGI outlines")
    ap.add_argument("--cc-stride", type=int, default=10,
                    help="use every Nth pair for the mean-coherence map")
    ap.add_argument("--rgi", type=Path,
                    default=Path(_os.environ.get("GPRI_RGI",
                                                 "data/rgi/rgi_61.zip")))
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene) or args.scene)
    day = scene.name

    # ---- mean coherence from a stride of pairs -----------------------------
    stack = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    dec = args.decimate
    nc = stack.shape[1] // dec
    t0 = time.time()
    acc = np.zeros((stack.shape[0], nc), np.float64)
    used = 0
    for p in range(0, stack.n_pairs, max(1, args.cc_stride)):
        c = stack.read_coherence(p)
        if c is None:
            continue
        acc += c[:, ::dec][:, :nc]
        used += 1
    mean_cc = (acc / max(used, 1)).astype(np.float32)
    print(f"mean coherence from {used} pairs in {time.time() - t0:.0f} s")

    geom = RadarGeometry(decimated_par(stack.par, dec), heading=args.heading)

    # ---- RGI ---------------------------------------------------------------
    lat, lon = geom.geodetic(rows=[0, geom.shape[0] - 1],
                             cols=[0, geom.shape[1] - 1])
    pad = 0.02
    bbox = (min(lon.min(), geom.lon0) - pad, min(lat.min(), geom.lat0) - pad,
            max(lon.max(), geom.lon0) + pad, max(lat.max(), geom.lat0) + pad)
    gdf = load_outlines(args.rgi, bbox=bbox)
    print(f"{len(gdf)} RGI glaciers intersect the footprint")

    t0 = time.time()
    on_ice = glacier_mask(geom, gdf, buffer_m=0.0)
    stable, contested = stable_ground_mask(mean_cc, geom, gdf,
                                           threshold=args.stable_coherence,
                                           buffer_m=args.buffer)
    print(f"glacier rasterisation in {time.time() - t0:.0f} s")

    old = mean_cc >= args.stable_coherence
    shown = mean_cc >= 0.5
    print(f"\ncoherence-only reference (cc >= {args.stable_coherence}): "
          f"{old.sum():,} px")
    print(f"  on RGI glacier (+{args.buffer:.0f} m buffer): "
          f"{contested.sum():,} px ({100 * contested.sum() / max(old.sum(), 1):.1f}%)")
    print(f"  corrected reference: {stable.sum():,} px")
    print(f"\nof everything the maps display (cc >= 0.5): "
          f"{100 * (shown & on_ice).mean() / max(shown.mean(), 1e-9):.1f}% "
          f"is RGI glacier — the rest is rock, moraine and forest, where "
          f"'deformation' means error")

    # ---- overlay figure ----------------------------------------------------
    spacing = 40.0
    bg = read_backdrop(scene, stack, dec)
    bg_map, transform = geocode(bg.astype(np.float32), geom, spacing=spacing)
    cls = np.full(mean_cc.shape, np.nan, np.float32)
    cls[stable] = 0.0                       # kept reference: rock
    cls[contested] = 1.0                    # coherent but on ice: dropped
    cls[shown & on_ice & ~old] = 2.0        # glacier surface in the maps
    cls_map, _ = geocode(cls, geom, spacing=spacing, order="nearest")

    xmin, sx, _, ymax, _, sy = transform
    ny, nx = bg_map.shape
    extent = [xmin / 1000, (xmin + sx * nx) / 1000,
              (ymax + sy * ny) / 1000, ymax / 1000]

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.imshow(bg_map, cmap="gray", extent=extent, origin="upper",
              vmin=np.nanpercentile(bg_map, 2), vmax=np.nanpercentile(bg_map, 98),
              interpolation="bilinear")
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#2f7ed8", "#d62728", "#9edae5"])
    ax.imshow(cls_map, cmap=cmap, vmin=0, vmax=2, extent=extent,
              origin="upper", interpolation="nearest", alpha=0.75)

    for x, y in outline_paths(gdf, geom):
        ax.plot(x / 1000, y / 1000, "-", color="w", lw=0.9)
    x0, y0 = geom.origin_xy()
    ax.plot(x0 / 1000, y0 / 1000, "^", ms=10, mfc="w", mec="k", mew=1.5)

    named = gdf[gdf.get("Name").notna()] if "Name" in gdf.columns else gdf[:0]
    if len(named):
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:4326", geom.crs, always_xy=True)
        for _, r in named.nlargest(8, "Area").iterrows():
            fx, fy = tf.transform(r.CenLon, r.CenLat)
            ax.annotate(str(r.Name).replace(" WA", ""), (fx / 1000, fy / 1000),
                        fontsize=7, color="w", ha="center",
                        path_effects=None)

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#2f7ed8", label="reference kept (rock, coherent)"),
        mpatches.Patch(color="#d62728",
                       label=f"reference DROPPED: coherent but on RGI ice "
                             f"({100 * contested.sum() / max(old.sum(), 1):.0f}% of old mask)"),
        mpatches.Patch(color="#9edae5", label="glacier surface in the maps"),
    ], loc="lower right", fontsize=8)
    ax.set_xlabel("easting (km)")
    ax.set_ylabel("northing (km)")
    ax.set_aspect("equal")
    ok = np.isfinite(cls_map)
    if ok.any():
        rr, ccx = np.nonzero(ok)
        padp = int(1500 / spacing)
        ax.set_xlim((xmin + sx * max(ccx.min() - padp, 0)) / 1000,
                    (xmin + sx * min(ccx.max() + padp, nx)) / 1000)
        ax.set_ylim((ymax + sy * min(rr.max() + padp, ny)) / 1000,
                    (ymax + sy * max(rr.min() - padp, 0)) / 1000)
    ax.set_title(f"{day}: stable-ground reference audited against RGI 6.0\n"
                 f"white outlines: glacier inventory; heading "
                 f"{geom.heading:.1f}° (outline/backscatter fit checks it)",
                 fontsize=10)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"16_rgi_reference_{day}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
