#!/usr/bin/env python3
"""Diurnal analysis of the full BakerBend1 day — and whether it is ice or air.

    python examples/baker_diurnal.py --decimate 16

BakerBend1/20170803 is 723 acquisitions on a 2-minute cadence spanning 24.18
hours: one complete diurnal cycle, sampled 723 times.  The experiment was
designed to catch sub-daily velocity and uplift variation driven by water
pressure in the subglacial drainage system.

This script runs the whole chain — atmospheric screens, network inversion,
harmonic fit — and then runs the three tests in :mod:`gpri_tools.diurnal` that decide
whether the recovered diurnal is glaciological or is the atmosphere's own
diurnal cycle leaking through.  It prints the verdict either way.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpri_tools import atmosphere, diurnal, plot
from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry, geocode
from gpri_tools.heading import scene_heading
from gpri_tools.refractivity import invert_refractivity, screens_to_delta_n
from gpri_tools.stack import DiffStack
from gpri_tools.timeseries import los_displacement, reference_to_stable

from baker_north_side import SCENE, decimated_par, read_backdrop


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--pairs", type=int, default=0, help="0 = all 722")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--heading", type=float, default=None,
                    help="scan heading, deg true (default: the scene's "
                         "heading.json from `gpri heading --write`)")
    ap.add_argument("--spacing", type=float, default=40.0)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--stable-coherence", type=float, default=0.85,
                    help="coherence above which ground is treated as bedrock")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.heading is None:
        args.heading = scene_heading(args.scene, default=BAKERBEND1_HEADING)

    t0 = time.time()
    stack = DiffStack.from_directory(args.scene / "diff0",
                                     slc_tab=args.scene / "SLCu_tab")
    n = stack.n_pairs if args.pairs <= 0 else min(args.pairs, stack.n_pairs)
    dec = args.decimate
    net = stack.network
    net.pairs = net.pairs[:n]
    span = float(net.times[n] - net.times[0])
    print(f"{stack}\nusing {n} pairs over {span * 24:.2f} h "
          f"(decimate {dec} in range)")
    if span < 1.0:
        print(f"WARNING: {span * 24:.1f} h is less than one diurnal cycle; "
              f"the harmonic fit will refuse.")

    nc = stack.shape[1] // dec
    ifg = np.empty((n, stack.shape[0], nc), np.complex64)
    cc = np.empty(ifg.shape, np.float32)
    for p in range(n):
        ifg[p] = stack.read_pair(p)[:, ::dec][:, :nc]
        c = stack.read_coherence(p)
        cc[p] = np.abs(ifg[p]) if c is None else c[:, ::dec][:, :nc]
        if p % 100 == 0:
            print(f"  read {p + 1}/{n}  ({time.time() - t0:.0f} s)")
    print(f"read in {time.time() - t0:.0f} s, mean coherence {cc.mean():.3f}")

    par = decimated_par(stack.par, dec)
    geom = RadarGeometry(par, heading=args.heading)
    r = par.slant_range()
    az_deg = np.rad2deg(np.deg2rad(
        par.float("GPRI_az_start_angle")
        + par.float("GPRI_az_angle_step") * np.arange(par.azimuth_lines)))

    # -------------------------------------------------- atmospheric screens
    print("estimating refractivity screens ...")
    phase = np.angle(ifg).astype(np.float32)
    del ifg
    screens = []
    for p in range(n):
        w = np.where(atmosphere.stable_mask(cc[p], threshold=0.4), cc[p], 0.0)
        try:
            s = atmosphere.fit_screen(phase[p], slant_range=r, azimuth=az_deg,
                                      weights=w, model="linear",
                                      wavelength=par.wavelength)
            phase[p] = atmosphere.remove_screen(phase[p], s)
        except Exception:
            s = None
        screens.append(s)
    dN = screens_to_delta_n(screens, wavelength=par.wavelength)
    ok = np.isfinite(dN)
    print(f"screens fitted for {ok.sum()}/{n}; "
          f"dN {np.nanmin(dN):+.3f} .. {np.nanmax(dN):+.3f} N-units")

    # ------------------------------------------- pair phase -> per-epoch series
    # This network is a daisy chain, so the inversion is a cumulative sum: the
    # displacement of epoch k+1 relative to epoch 0 is the running total of the
    # pair displacements up to it.  No least squares needed, and no ambiguity.
    d_pair = los_displacement(phase, par.wavelength)
    del phase
    disp = np.concatenate([np.zeros((1,) + d_pair.shape[1:], np.float32),
                           np.cumsum(d_pair, axis=0, dtype=np.float32)], axis=0)
    del d_pair
    times = net.times[:n + 1]
    print(f"time series: {disp.shape[0]} epochs x {disp.shape[1]}x{disp.shape[2]}")

    N_epoch = invert_refractivity(np.where(ok, dN, 0.0), net,
                                  weights=ok.astype(float)).ravel()

    # ------------------------------------------------------- harmonic fit
    mean_cc = cc.mean(axis=0)
    del cc
    ice = mean_cc >= args.ice_coherence
    stable = mean_cc >= args.stable_coherence
    print(f"ice mask {100 * ice.mean():.2f}% of swath, "
          f"stable-ground mask {100 * stable.mean():.2f}%")

    # Split the stable ground in two: half ties the series down, half is held
    # out to test it.  Referencing on the same pixels you then run the null test
    # on is circular -- they were forced to zero by construction.
    idx = np.flatnonzero(stable.ravel())
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    ref_mask = np.zeros(stable.size, bool)
    ref_mask[idx[: idx.size // 2]] = True
    null_mask = np.zeros(stable.size, bool)
    null_mask[idx[idx.size // 2:]] = True
    ref_mask = ref_mask.reshape(stable.shape)
    null_mask = null_mask.reshape(stable.shape)

    raw_null = diurnal.stable_ground_null(disp, times, null_mask)
    disp, common = reference_to_stable(disp, ref_mask, return_offset=True)
    print(f"referenced to {ref_mask.sum():,} bedrock pixels "
          f"({null_mask.sum():,} held out for the null test)")
    print(f"common mode removed: peak-to-peak "
          f"{(common.max() - common.min()) * 1000:.1f} mm")

    fit = diurnal.fit_harmonics(disp, times, periods=(diurnal.DIURNAL,))
    amp = fit.amplitude(diurnal.DIURNAL)
    origin_hour = (net.epochs[0].hour + net.epochs[0].minute / 60.0)
    peak = fit.peak_time(diurnal.DIURNAL, origin_hour=origin_hour)
    print(f"\n{fit!r}")
    print(f"diurnal amplitude on ice: median "
          f"{np.nanmedian(amp[ice]) * 1000:.3f} mm, "
          f"p95 {np.nanpercentile(amp[ice], 95) * 1000:.3f} mm")

    # ================= the three tests ==================================
    print("\n" + "=" * 70)
    print("IS THIS ICE OR IS IT AIR?")
    print("=" * 70)

    print("\n1. range dependence (residual refractivity is linear in range)")
    rd = diurnal.range_dependence(amp, r, mask=ice)
    print(f"   correlation with slant range: r = {rd['correlation']:+.3f} "
          f"over {rd['n']:,} pixels")
    print(f"   slope {rd['slope'] * 1e6:+.4f} mm amplitude per km of range")
    print(f"   -> {rd['verdict']}")

    print("\n2. atmospheric coherence (variance explained by the N series)")
    frac = diurnal.atmospheric_coherence(disp, times, N_epoch)
    print(f"   median over ice: {100 * np.nanmedian(frac[ice]):.1f}% of the "
          f"time-series variance is explained by refractivity alone")

    print("\n3. stable-ground null (bedrock is not moving)")
    null = diurnal.stable_ground_null(disp, times, null_mask)
    print(f"   before referencing, held-out bedrock showed "
          f"{raw_null['amplitude_median'] * 1000:.2f} mm at phase "
          f"concentration {raw_null['phase_concentration']:.3f}")
    print(f"   {null['n']:,} held-out stable pixels: median amplitude "
          f"{null['amplitude_median'] * 1000:.3f} mm, "
          f"p95 {null['amplitude_p95'] * 1000:.3f} mm")
    print(f"   phase concentration {null['phase_concentration']:.3f} "
          f"(near 1 = a shared systematic error)")
    ratio = np.nanmedian(amp[ice]) / max(null["amplitude_median"], 1e-12)
    print(f"   ice/bedrock amplitude ratio: {ratio:.2f}")

    print("\n" + "-" * 70)
    if ratio < 2.0:
        print("VERDICT: the diurnal on ice does not clear the bedrock error")
        print("floor. This is not a detection of subglacial hydrology.")
    elif abs(rd["correlation"]) >= 0.5:
        print("VERDICT: the diurnal is dominated by residual refractivity.")
        print("Correct the atmosphere further before interpreting it.")
    else:
        print("VERDICT: the diurnal clears the bedrock floor and is not")
        print("range-dependent. Worth interpreting -- but note the LOS")
        print("geometry limit below before calling any of it uplift.")
    print("-" * 70)

    vs = diurnal.vertical_sensitivity(geom)
    print(f"\nLOS geometry: beam elevation {geom.elevation:.1f} deg, so LOS "
          f"sensitivity to\nuplift is {vs:.3f} against "
          f"{np.cos(np.deg2rad(geom.elevation)):.3f} for horizontal motion — "
          f"uplift is\nsuppressed {np.cos(np.deg2rad(geom.elevation)) / vs:.1f}x, "
          f"and one line of sight cannot separate them.")

    # ------------------------------------------------------------ figures
    out = args.outdir
    plot.diurnal_summary(fit, times, slant_range=r, mask=ice,
                         origin_hour=origin_hour, stable=stable)
    plt.tight_layout(); plt.savefig(out / "08_diurnal.png", dpi=140); plt.close()

    amp_map, transform = geocode(np.where(ice, amp * 1000, np.nan), geom,
                                 spacing=args.spacing)
    peak_map, _ = geocode(np.where(ice, peak, np.nan), geom, spacing=args.spacing)
    bg = read_backdrop(args.scene, stack, dec)
    bg_map = None if bg is None else geocode(bg, geom, spacing=args.spacing)[0]
    x0, y0 = geom.origin_xy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot.map_image(amp_map, transform, kind="coherence", ax=axes[0],
                   origin_xy=(x0, y0), background=bg_map, cmap="magma",
                   vmin=0, vmax=float(np.nanpercentile(amp_map, 98)),
                   title="diurnal amplitude (mm LOS)")
    plot.map_image(peak_map, transform, kind="phase", ax=axes[1],
                   origin_xy=(x0, y0), background=bg_map, cmap="twilight",
                   vmin=0, vmax=24, title="hour of diurnal peak")
    ok_xy = np.isfinite(amp_map)
    if ok_xy.any():
        rr, ccx = np.nonzero(ok_xy)
        xs, sx, _, ymax, _, sy = transform
        pad = int(1500 / args.spacing)
        for ax in axes:
            ax.set_xlim((xs + sx * max(ccx.min() - pad, 0)) / 1000,
                        (xs + sx * min(ccx.max() + pad, amp_map.shape[1])) / 1000)
            ax.set_ylim((ymax + sy * min(rr.max() + pad, amp_map.shape[0])) / 1000,
                        (ymax + sy * max(rr.min() - pad, 0)) / 1000)
    plt.tight_layout(); plt.savefig(out / "09_diurnal_map.png", dpi=150); plt.close()

    plt.figure(figsize=(7.5, 4))
    plt.plot(times * 24, common * 1000, "-", lw=1.2, color="C3")
    plt.xlabel("hours from first acquisition")
    plt.ylabel("common-mode LOS (mm)")
    plt.title("Scene-wide common mode removed by referencing to bedrock",
              fontsize=10)
    plt.axhline(0, color="0.6", lw=0.8, zorder=0)
    plt.tight_layout(); plt.savefig(out / "11_common_mode.png", dpi=150)
    plt.close()

    plot.refractivity_plot(N_epoch, times=net.times[:len(N_epoch)])
    plt.tight_layout(); plt.savefig(out / "10_refractivity_day.png", dpi=150)
    plt.close()

    print(f"\nwrote figures to {out}/   total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
