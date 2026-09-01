#!/usr/bin/env python3
"""A movie of the LOS deformation field evolving through a GPRI day.

    python examples/baker_movie.py --scene 20170803 --decimate 16

Each frame is the cumulative LOS displacement since the first acquisition,
geocoded to the local stereographic frame over the mean-backscatter backdrop,
with the real UTC clock in the corner.

The correction recipe is the one the held-out-bedrock validation chose
(``docs/atmosphere.md``): reference + per-epoch drift removal + per-epoch
turbulence screen, and **no per-pair parametric screens** — those were
measured to inject as much random-walk noise as the atmosphere they remove.
All bedrock feeds the corrections here (this is a product, not a test, so
there is nothing to hold out).

Two kinds of smoothing are applied *for display only*, and the frame says so:
a rolling temporal mean (default 7 epochs, ~14 min) and a light spatial
Gaussian.  Without them a per-pixel movie of single-look data is snow — the
√t random-walk pixel noise is several times the spatially coherent signal.
The smoothing windows are printed on the frame rather than hidden.

The ``--anomaly`` views (departure from the pixel's day mean, or from its
linear trend) get a second panel: the **reference displacement rate** the
anomaly is measured against -- the per-pixel linear LOS rate over the day,
in mm/hr -- and a time strip with the median anomaly over the ice and a
cursor at the current frame, so the diurnal swing is readable as a curve and
not only as a colour.

Geocoding is done once: the inverse map from the output grid to fractional
radar coordinates is precomputed, and every frame is just a bilinear gather.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, load, integrate                      # noqa: E402
from baker_north_side import read_backdrop                         # noqa: E402

from gpri.aps import epoch_screen_correction, turbulence_screen    # noqa: E402
from gpri.geocode import BAKERBEND1_HEADING, RadarGeometry, map_grid  # noqa: E402
from gpri.timeseries import los_displacement                       # noqa: E402


def decimated_geom(stack, dec, heading):
    from baker_north_side import decimated_par
    return RadarGeometry(decimated_par(stack.par, dec), heading=heading)


class Resampler:
    """Radar grid -> map grid, mapping computed once, applied per frame."""

    def __init__(self, geom, spacing):
        x, y, self.transform = map_grid(geom, spacing=spacing)
        X, Y = np.meshgrid(x, y)
        row, col = geom.radar_coordinates(X, Y)
        na, nr = geom.shape
        self.inside = ((row >= 0) & (row <= na - 1) &
                       (col >= 0) & (col <= nr - 1) &
                       np.isfinite(row) & np.isfinite(col))
        r = np.clip(row[self.inside], 0, na - 1)
        c = np.clip(col[self.inside], 0, nr - 1)
        self.r0 = np.floor(r).astype(np.int32)
        self.c0 = np.floor(c).astype(np.int32)
        self.r1 = np.minimum(self.r0 + 1, na - 1)
        self.c1 = np.minimum(self.c0 + 1, nr - 1)
        self.fr = (r - self.r0).astype(np.float32)
        self.fc = (c - self.c0).astype(np.float32)
        self.shape = X.shape

    def __call__(self, img, fill=np.nan):
        out = np.full(self.shape, fill, np.float32)
        v = (img[self.r0, self.c0] * (1 - self.fr) * (1 - self.fc)
             + img[self.r1, self.c0] * self.fr * (1 - self.fc)
             + img[self.r0, self.c1] * (1 - self.fr) * self.fc
             + img[self.r1, self.c1] * self.fr * self.fc)
        out[self.inside] = v
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--heading", type=float, default=BAKERBEND1_HEADING)
    ap.add_argument("--spacing", type=float, default=40.0)
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--show-coherence", type=float, default=0.4,
                    help="mask displayed pixels below this mean coherence")
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0),
                    metavar=("AZ", "RG"), help="turbulence kernel, pixels")
    ap.add_argument("--t-smooth", type=int, default=7,
                    help="rolling-mean window in epochs, display only")
    ap.add_argument("--s-smooth", type=float, nargs=2, default=(1.0, 2.0),
                    help="spatial gaussian in radar pixels, display only")
    ap.add_argument("--rgi", action="store_true",
                    help="tie the reference to true rock: coherent pixels "
                         "outside the RGI outlines (+100 m); needs GPRI_RGI")
    ap.add_argument("--anomaly", default="none", choices=("none", "mean", "trend"),
                    help="show each frame as an anomaly: 'mean' subtracts the "
                         "pixel's time-mean over the day, 'trend' also removes "
                         "its linear (secular) rate, leaving the transient part")
    ap.add_argument("--rate-hours", type=float, default=0.0,
                    help="render LOS motion over the trailing N hours instead "
                         "of cumulative displacement -- bounded noise, and the "
                         "right view for a diurnal signal")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"),
                    help="which receive antenna; 'lower' is formed from the "
                         "SLCs (GAMMA only processed the upper)")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene, args.scene))
    day = scene.name + ("" if args.antenna == "upper" else f"_{args.antenna}")

    stack, net, phase, cc, r, az, n = load(scene, args.decimate, args.pairs,
                                           antenna=args.antenna)
    lam = stack.wavelength
    mean_cc = cc.mean(axis=0)
    del cc
    stable = mean_cc >= args.stable_coherence
    show = mean_cc >= args.show_coherence
    if args.rgi:
        import os as _os
        from gpri.glaciers import load_outlines, stable_ground_mask
        _g = decimated_geom(stack, args.decimate, args.heading)
        la, lo = _g.geodetic(rows=[0, _g.shape[0] - 1],
                             cols=[0, _g.shape[1] - 1])
        bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
        gdf = load_outlines(_os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                            bbox=bbox)
        stable, contested = stable_ground_mask(mean_cc, _g, gdf,
                                               threshold=args.stable_coherence)
        print(f"RGI reference: dropped {contested.sum():,} coherent-but-glacier px")
    print(f"{day}: {n} pairs, bedrock {stable.sum():,} px, "
          f"showing {100 * show.mean():.1f}% of the swath")

    # ---------------- corrections: the validated recipe (A + C + D) --------
    d, times = integrate(los_displacement(phase, lam), net, n)
    del phase
    d, _ = epoch_screen_correction(d, stable, r, model="linear",
                                   weights=mean_cc)
    t0 = time.time()
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], stable, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        d[k] -= scr
    print(f"drift + turbulence corrections in {time.time() - t0:.0f} s")

    # ---------------- display smoothing (declared on the frame) ------------
    W = max(1, args.t_smooth)
    if W > 1:
        from scipy.ndimage import uniform_filter1d
        d = uniform_filter1d(d, W, axis=0, mode="nearest")
    from scipy.ndimage import gaussian_filter
    if max(args.s_smooth) > 0:
        good = show & np.isfinite(d).all(axis=0)
        den = gaussian_filter(good.astype(np.float32), args.s_smooth)
        for k in range(d.shape[0]):
            num = gaussian_filter(np.where(good, d[k], 0.0).astype(np.float32),
                                  args.s_smooth)
            with np.errstate(invalid="ignore", divide="ignore"):
                d[k] = np.where(den > 0.05, num / den, np.nan)
    d[:, ~show] = np.nan

    # ---------------- cumulative vs anomaly vs trailing-rate view ----------
    if args.anomaly != "none" and args.rate_hours > 0:
        raise SystemExit("--anomaly is a view of the cumulative series; "
                         "it does not combine with --rate-hours")
    if args.anomaly != "none":
        # anomaly from the day's mean state.  On flowing ice the cumulative
        # series is dominated by the secular rate, so the mean anomaly is
        # mostly "before the middle of the day / after it"; the trend anomaly
        # removes that too and shows what departs from steady flow -- which
        # is where a diurnal hydrological response would appear.
        tt = (times - times[0]).astype(np.float64)          # days
        finite = np.isfinite(d)
        cnt = finite.sum(axis=0)
        dz = np.where(finite, d, 0.0)
        mean = dz.sum(axis=0) / np.maximum(cnt, 1)
        tm = (np.where(finite, tt[:, None, None], 0.0).sum(axis=0)
              / np.maximum(cnt, 1))
        tc = np.where(finite, tt[:, None, None] - tm, 0.0)
        rate = (tc * (dz - mean)).sum(axis=0) / np.maximum((tc ** 2).sum(axis=0), 1e-12)
        # the reference the anomaly is read against: the day's linear LOS
        # rate per pixel, shown beside the animation in mm/hr
        ref_mm_h = np.where((cnt >= 2) & show, rate * 1000 / 24.0, np.nan)
        if args.anomaly == "trend":
            d = d - mean - rate * (tt[:, None, None] - tm)
            label, title = "Trend anomaly (mm)", "departure from the day's linear trend"
            ref_label = "Reference rate (mm/hr)"
            ref_title = "reference removed: linear LOS rate over the day"
        else:
            d = d - mean
            label, title = "Mean anomaly (mm)", "departure from the day's mean"
            ref_label = "Reference rate (mm/hr)"
            ref_title = "reference: linear LOS rate over the day"
        d[:, cnt < 2] = np.nan
        # the same anomaly as a curve: median over the moving (non-reference)
        # pixels, with the interquartile band, so the frame's place in the
        # diurnal swing is readable at a glance
        moving = show & ~stable & (cnt >= 2)
        q25, q50, q75 = np.nanpercentile(d[:, moving], [25, 50, 75], axis=1) * 1000
        tag = f"anom{args.anomaly}_"
    elif args.rate_hours > 0:
        # motion over the trailing window: d(t) - d(t - w).  Unlike the
        # cumulative view this does not accumulate the random walk, so the
        # noise is bounded for the whole movie -- the right way to look for a
        # signal that comes and goes with the melt day.
        w = args.rate_hours / 24.0
        lag = np.searchsorted(times, times - w)
        dd = np.full_like(d, np.nan)
        for k in range(d.shape[0]):
            if times[k] - times[0] >= w:
                dd[k] = d[k] - d[lag[k]]
        d = dd
        label = "LOS motion (mm)"
        title = f"motion over the trailing {args.rate_hours:g} h"
        tag = f"rate{args.rate_hours:g}h_"
    else:
        label = "LOS displacement (mm)"
        title = "cumulative since the first acquisition"
        tag = ""

    # ---------------- geocode once, sample per frame -----------------------
    geom = decimated_geom(stack, args.decimate, args.heading)
    rs = Resampler(geom, args.spacing)
    bg = read_backdrop(scene, stack, args.decimate)
    bg_map = None if bg is None else rs(bg.astype(np.float32))
    x0, y0 = geom.origin_xy()

    valid0 = next((k for k in range(d.shape[0])
                   if np.isfinite(d[k]).any()), 0)
    frames = range(valid0, d.shape[0], max(1, args.stride))
    # limits from the whole series, not the last frame: cumulative displacement
    # grows all day, and end-of-day limits would flatten the first hours
    lim = float(np.nanpercentile(np.abs(d[::max(1, d.shape[0] // 40)]) * 1000, 97))
    lim = max(lim, 1.0)
    print(f"colour limits ±{lim:.1f} mm, {len(frames)} frames")

    # crop to where there is something to see
    first = rs(d[len(d) // 2])
    ok = np.isfinite(first)
    rr, ccx = np.nonzero(ok)
    xmin, sx, _, ymax, _, sy = rs.transform
    pad = int(1200 / args.spacing)
    x_lo = (xmin + sx * max(rr.min() * 0, ccx.min() - pad)) / 1000
    x_hi = (xmin + sx * min(ccx.max() + pad, first.shape[1])) / 1000
    y_lo = (ymax + sy * min(rr.max() + pad, first.shape[0])) / 1000
    y_hi = (ymax + sy * max(rr.min() - pad, 0)) / 1000
    extent = [xmin / 1000, (xmin + sx * first.shape[1]) / 1000,
              (ymax + sy * first.shape[0]) / 1000, ymax / 1000]

    def map_panel(ax_, img_mm, lim_, label_, title_):
        if bg_map is not None:
            ax_.imshow(bg_map, cmap="gray", extent=extent, origin="upper",
                       vmin=np.nanpercentile(bg_map, 2),
                       vmax=np.nanpercentile(bg_map, 98),
                       interpolation="bilinear")
        im_ = ax_.imshow(img_mm, cmap="RdBu_r", vmin=-lim_, vmax=lim_,
                         extent=extent, origin="upper", interpolation="nearest")
        ax_.plot(x0 / 1000, y0 / 1000, "^", ms=9, mfc="w", mec="k", mew=1.4)
        ax_.set_xlim(x_lo, x_hi)
        ax_.set_ylim(y_lo, y_hi)
        ax_.set_xlabel("easting (km)")
        ax_.set_ylabel("northing (km)")
        ax_.set_aspect("equal")
        cb_ = fig.colorbar(im_, ax=ax_, pad=0.02, fraction=0.046)
        cb_.set_label(label_)
        ax_.set_title(f"{title_}  (+ toward radar)", fontsize=9)
        return im_

    cursor = None
    if args.anomaly != "none":
        # anomaly | reference rate, with the anomaly curve underneath
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[4.6, 1.0])
        ax = fig.add_subplot(gs[0, 0])
        ax_ref = fig.add_subplot(gs[0, 1], sharex=ax, sharey=ax)
        ax_ts = fig.add_subplot(gs[1, :])
        ref_map = rs(ref_mm_h.astype(np.float32))
        lim_ref = max(float(np.nanpercentile(np.abs(ref_map), 98)), 0.1)
        map_panel(ax_ref, ref_map, lim_ref, ref_label, ref_title)
        ax_ref.set_ylabel("")
        hh = tt * 24
        ax_ts.fill_between(hh, q25, q75, color="0.6", alpha=0.35, lw=0,
                           label="interquartile range")
        ax_ts.plot(hh, q50, color="k", lw=1.2, label="median")
        ax_ts.axhline(0, color="0.5", lw=0.6)
        cursor = ax_ts.axvline(hh[0], color="r", lw=1.4)
        ax_ts.set_xlim(hh[0], hh[-1])
        ax_ts.set_xlabel("Elapsed time (hr)")
        ax_ts.set_ylabel("Ice anomaly (mm)")
        ax_ts.set_title(f"median and interquartile range over the moving "
                        f"pixels, from {net.epochs[0]:%Y-%m-%d %H:%M} UTC",
                        fontsize=9)
        ax_ts.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
        ax_ts.grid(alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(7.68, 6.40), dpi=100)
    im = map_panel(ax, first * 1000, lim, label, title)
    clock = ax.text(0.02, 0.975, "", transform=ax.transAxes, va="top",
                    fontsize=11, family="monospace",
                    bbox=dict(fc="w", alpha=0.8, ec="none"))
    ax.text(0.02, 0.02,
            f"smoothed for display: {W} epochs (~{W * 2} min), "
            f"gaussian {args.s_smooth} px",
            transform=ax.transAxes, fontsize=7, color="0.35",
            bbox=dict(fc="w", alpha=0.7, ec="none"))
    if cursor is None:
        fig.tight_layout()

    out = args.outdir / f"14_los_movie_{tag}{day}.mp4"
    args.outdir.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, codec="h264",
                          extra_args=["-crf", "24", "-pix_fmt", "yuv420p"])
    t0 = time.time()
    with writer.saving(fig, str(out), dpi=100):
        for i, k in enumerate(frames):
            im.set_data(rs(d[k]) * 1000)
            ep = net.epochs[k]
            hours = (times[k] - times[0]) * 24
            clock.set_text(f"{ep:%Y-%m-%d %H:%M} UTC   +{hours:5.2f} hr")
            if cursor is not None:
                cursor.set_xdata([hours, hours])
            writer.grab_frame()
            if i % 100 == 0:
                print(f"  frame {i + 1}/{len(frames)}  ({time.time() - t0:.0f} s)")
    plt.close(fig)
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(frames)} frames, {size:.1f} MB, "
          f"{len(frames) / args.fps:.0f} s at {args.fps} fps)")


if __name__ == "__main__":
    main()
