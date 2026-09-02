#!/usr/bin/env python3
"""End-to-end: BakerBend1 GAMMA interferograms -> LOS velocity on a map.

Runs the whole package against the real ``diff0`` stack and writes the
figures in ``docs/figures/``.  No GAMMA binaries involved — this reads GAMMA's
output rasters directly.

    python examples/baker_north_side.py --pairs 120 --decimate 8

The radar sat NW of Mount Baker at 48.82132 N, 121.92018 W, 1252 m, looking
south-east at the north-side glaciers.  ``GPRI_scan_heading`` was never
surveyed (it is 0.0 in every parameter file), so the heading here is the
``BAKERBEND1_HEADING`` estimate — good enough to see that the fan covers the
right mountain, not good enough to publish a map from.  Tie it to a real
feature with ``gpri.geocode.heading_from_tiepoint`` first.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpri import atmosphere, plot
from gpri.gamma import ParFile
from gpri.geocode import BAKERBEND1_HEADING, RadarGeometry, geocode
from gpri.refractivity import invert_refractivity, screens_to_delta_n
from gpri.stack import DiffStack
from gpri.timeseries import los_displacement, stack_velocity

import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "gpri_site", Path(__file__).resolve().parent / "site.py")
_site = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_site)
_site.load_site()

#: Set GPRI_SCENE_20170803 in site.env (see site.env.example).
SCENE = Path(_os.environ.get("GPRI_SCENE_20170803", "unset-see-site.env"))

#: North-side features, for checking the scan heading points at the right place.
FEATURES = {
    "Baker summit":  (48.7767, -121.8144),
    "Colfax Pk":     (48.7714, -121.8300),
    "Coleman Gl":    (48.7900, -121.8400),
    "Roosevelt Gl":  (48.7930, -121.8200),
    "Mazama Gl":     (48.7960, -121.7980),
    "Rainbow Gl":    (48.7880, -121.7800),
}


def decimated_par(par: ParFile, stride: int) -> ParFile:
    """A copy of the parameter file describing a range-decimated grid.

    Decimating in range without saying so is how a geocoded product ends up
    silently stretched by the decimation factor: the pixel spacing has to grow
    with the stride or every range coordinate is wrong.
    """
    e = {k: list(v) for k, v in par.entries.items()}
    n = par.range_samples // stride
    e["range_samples"] = [str(n)]
    e["range_pixel_spacing"] = [str(par.range_pixel_spacing * stride), "m"]
    return ParFile(e, par.header)


def read_backdrop(scene: Path, stack: DiffStack, stride: int):
    """Backscatter in dB, for plotting terrain under the measurements.

    Not from the interferograms: GAMMA normalises ``.diff`` magnitude to unity,
    so ``abs(ifg)`` is 0 dB everywhere and makes a uniformly black backdrop.
    The MLIs carry the real backscatter, on the identical 396 x 22101 grid —
    ``baker_mli_upper.ave`` is the whole-day average and is the best of them.
    """
    from gpri.gamma import read_image

    # which antenna an SLC-formed stack is looking through: the id suffix
    ant = "u"
    if getattr(stack, "images", None):
        ant = Path(stack.images[0]).name.split(".")[0][-1]
    for name in ("baker_mli_upper.ave", "baker_mli.ave", f"mli_mean_{ant}.ave"):
        if (scene / name).exists():
            path = scene / name
            break
    else:
        mlis = sorted((scene / "mli").glob(f"*{ant}.mli"))
        if mlis:
            path = mlis[0]
        elif hasattr(stack, "mean_intensity"):
            # focused here, not by GAMMA: no MLIs, so average some SLCs and
            # keep the result beside them for the next script (written under
            # a temporary name so a script running alongside never reads a
            # half-written file)
            from gpri.gamma import write_image
            path = scene / f"mli_mean_{ant}.ave"
            a = stack.mean_intensity()
            try:
                tmp = path.with_suffix(f".ave.{os.getpid()}")
                write_image(tmp, a, image_format="FLOAT")
                os.replace(tmp, path)
            except OSError:
                pass
            a = a[:, ::stride][:, : stack.shape[1] // stride]
            return 10.0 * np.log10(np.maximum(a, 1e-9))
        else:
            return None
    a = read_image(path, shape=stack.shape, image_format="FLOAT")
    a = a[:, ::stride][:, : stack.shape[1] // stride]
    return 10.0 * np.log10(np.maximum(a, 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--decimate", type=int, default=8, help="range decimation")
    ap.add_argument("--heading", type=float, default=BAKERBEND1_HEADING)
    ap.add_argument("--spacing", type=float, default=25.0, help="map pixel, m")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--coherence", type=float, default=0.4,
                    help="coherence floor for fitting atmospheric screens")
    ap.add_argument("--mask-coherence", type=float, default=0.5,
                    help="mean-coherence floor below which rate is masked")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- load
    t0 = time.time()
    stack = DiffStack.from_directory(args.scene / "diff0",
                                     slc_tab=args.scene / "SLCu_tab")
    print(stack)
    n = min(args.pairs, stack.n_pairs)
    dec = args.decimate

    ifg = np.empty((n, stack.shape[0], stack.shape[1] // dec), np.complex64)
    cc = np.empty(ifg.shape, np.float32)
    for p in range(n):
        ifg[p] = stack.read_pair(p)[:, ::dec][:, :ifg.shape[2]]
        c = stack.read_coherence(p)
        cc[p] = np.abs(ifg[p]) if c is None else c[:, ::dec][:, :ifg.shape[2]]
        if p % 20 == 0:
            print(f"  read {p + 1}/{n} pairs  ({time.time() - t0:.0f} s)")
    print(f"read {n} pairs in {time.time() - t0:.0f} s, "
          f"mean coherence {cc.mean():.3f}")

    par = decimated_par(stack.par, dec)
    geom = RadarGeometry(par, heading=args.heading)
    print(geom)

    net = stack.network
    net.pairs = net.pairs[:n]
    if net.paths:
        net.paths = net.paths[:n]

    r = par.slant_range()
    az = np.deg2rad(np.linspace(par.float("GPRI_az_start_angle"),
                                par.float("GPRI_az_start_angle")
                                + par.float("GPRI_az_angle_step") * (par.azimuth_lines - 1),
                                par.azimuth_lines))

    # ------------------------------------------------- atmospheric screens
    print("estimating refractivity screens ...")
    phase = np.angle(ifg).astype(np.float32)
    screens, corrected = [], np.empty_like(phase)
    for p in range(n):
        mask = atmosphere.stable_mask(cc[p], threshold=args.coherence)
        w = np.where(mask, cc[p], 0.0)
        try:
            s = atmosphere.fit_screen(phase[p], slant_range=r, azimuth=np.rad2deg(az),
                                      weights=w, model="linear",
                                      wavelength=par.wavelength)
        except Exception as exc:                      # a pair too decorrelated to fit
            print(f"  pair {p}: screen fit failed ({exc}); left uncorrected")
            screens.append(None)
            corrected[p] = phase[p]
            continue
        screens.append(s)
        corrected[p] = atmosphere.remove_screen(phase[p], s)

    dN = screens_to_delta_n(screens, wavelength=par.wavelength)
    ok = np.isfinite(dN)
    print(f"screens fitted for {ok.sum()}/{n} pairs; "
          f"dN range {np.nanmin(dN):+.3f} .. {np.nanmax(dN):+.3f} N-units")

    # ----------------------------------------------------------- velocity
    d = los_displacement(corrected, par.wavelength)
    v = stack_velocity(d, net, weights=cc, min_pairs=max(2, n // 4))

    mean_cc = cc.mean(axis=0)
    amplitude = read_backdrop(args.scene, stack, dec)

    # Beyond about 8 km the beam is behind the mountain, and coherence sits at
    # the noise floor.  Rate estimated there is not a small number, it is a
    # meaningless one -- mask it rather than let it set the colour scale.
    good = mean_cc >= args.mask_coherence
    v_shown = np.where(good, v, np.nan)
    span = float(net.times[n] - net.times[0])          # days actually observed
    disp = v_shown * span                              # metres over the window
    print(f"observed window {span * 24:.2f} h over {n} pairs; "
          f"{100 * good.mean():.1f}% of the swath above coherence "
          f"{args.mask_coherence}")
    print(f"LOS displacement over the window, mm (5/50/95): "
          f"{np.nanpercentile(disp * 1000, [5, 50, 95]).round(2)}")

    # ----------------------------------------------------------- geocode
    print(f"geocoding at {args.spacing:g} m ...")
    v_map, transform = geocode(v, geom, spacing=args.spacing)
    d_map, _ = geocode(disp, geom, spacing=args.spacing)
    cc_map, _ = geocode(mean_cc, geom, spacing=args.spacing)
    amp_map = (None if amplitude is None
               else geocode(amplitude, geom, spacing=args.spacing)[0])
    x0, y0 = geom.origin_xy()
    print(f"map grid {v_map.shape}, {np.isfinite(v_map).mean() * 100:.0f}% illuminated")

    # crop the display to where there is actually something to see
    xs, sx, _, ymax, _, sy = transform
    ny, nx = d_map.shape
    ok_xy = np.isfinite(d_map)
    if ok_xy.any():
        rr, ccx = np.nonzero(ok_xy)
        pad = int(1500 / args.spacing)
        xlim = ((xs + sx * max(ccx.min() - pad, 0)) / 1000,
                (xs + sx * min(ccx.max() + pad, nx)) / 1000)
        ylim = ((ymax + sy * min(rr.max() + pad, ny)) / 1000,
                (ymax + sy * max(rr.min() - pad, 0)) / 1000)
    else:
        xlim = ylim = None

    # ------------------------------------------------------------ figures
    out = args.outdir

    plot.coverage_map(geom, spacing=args.spacing * 2, features=FEATURES)
    plt.tight_layout(); plt.savefig(out / "01_coverage.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    plot.radar_image(phase[0], geom, kind="phase", ax=axes[0],
                     title="pair 0, raw phase — the atmosphere, not the glacier")
    plot.radar_image(corrected[0], geom, kind="phase", ax=axes[1],
                     title=f"after removing a linear range ramp "
                           f"($\\Delta N$ = {dN[0]:+.2f} N-units)")
    plt.tight_layout(); plt.savefig(out / "02_atmosphere.png", dpi=150); plt.close()

    plot.map_image(cc_map, transform, kind="coherence", origin_xy=(x0, y0),
                   title=f"mean coherence over {n} pairs — beyond the flank "
                         f"the beam is in shadow")
    plt.tight_layout(); plt.savefig(out / "03_coherence.png", dpi=150); plt.close()

    if amp_map is not None:
        ax = plot.map_image(amp_map, transform, kind="amplitude",
                            origin_xy=(x0, y0), title="mean backscatter (dB)")
        if xlim:
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        plt.tight_layout()
        plt.savefig(out / "03b_backscatter.png", dpi=150); plt.close()

    lim = float(np.nanpercentile(np.abs(d_map[np.isfinite(d_map)] * 1000), 96))
    ax = plot.map_image(d_map * 1000, transform, kind="displacement",
                        origin_xy=(x0, y0), background=amp_map,
                        vmin=-lim, vmax=lim,
                        title=f"LOS displacement over {span * 24:.1f} h, "
                              f"coherence $\\geq$ {args.mask_coherence}")
    if xlim:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    plt.tight_layout(); plt.savefig(out / "04_displacement.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    plot.radar_image(disp * 1000, geom, kind="displacement", ax=axes[0],
                     title="radar geometry — shape is not real",
                     vmin=-lim, vmax=lim)
    ax = plot.map_image(d_map * 1000, transform, kind="displacement", ax=axes[1],
                        origin_xy=(x0, y0), background=amp_map,
                        vmin=-lim, vmax=lim,
                        title="local stereographic — shape is real")
    if xlim:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    plt.tight_layout(); plt.savefig(out / "05_radar_vs_map.png", dpi=150); plt.close()

    plot.network_plot(net, values=cc.reshape(n, -1).mean(axis=1),
                      label="mean coherence")
    plt.tight_layout(); plt.savefig(out / "06_network.png", dpi=150); plt.close()

    if ok.sum() > 2:
        N = invert_refractivity(np.where(ok, dN, 0.0), net,
                                weights=ok.astype(float))
        plot.refractivity_plot(N, times=net.times)
        plt.tight_layout(); plt.savefig(out / "07_refractivity.png", dpi=150)
        plt.close()

    print(f"\nwrote {len(list(out.glob('*.png')))} figures to {out}/")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
