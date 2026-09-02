#!/usr/bin/env python3
"""The population time series: what the whole glacier did, hour by hour.

    python examples/baker_population.py --scene 20170827 --decimate 16 --rgi

Per pixel the corrected series is single-look noise growing as sqrt(t); the
median over thirty thousand RGI ice pixels is not, and neither is the median
over the held-out bedrock that never saw a correction.  This script draws
both against a UTC clock, as departures from each pixel's own secular trend,
on the same corrected displacement the pair-domain fits and the movies use
(reference + drift removal + turbulence, ``--rgi`` masks).

The trend is the point.  A least-squares line through a day of ice motion
absorbs part of whatever repeats each day unless that waveform happens to be
symmetric about the record's middle — a night-time trough with a sharp
morning recovery is not — so a "linear trend anomaly" is the waveform with a
tilt taken out of it and the rate it reports is biased by the same tilt.  On
a record longer than one cycle the separation can be made without assuming
any shape: the displacement between an epoch and the one 24 h later contains
no 24 h-periodic part at all (:func:`gpri.diurnal.secular_slope`).  Where the
record allows it (20170803's 24.2 h just does; 20170827's 44.9 h comfortably)
the secular rate is taken from those same-hour differences and the anomaly
is measured from that line; the linear version is drawn beside it for
comparison, and a record under one cycle keeps the line.  For two or more
UTC days the hour-of-day composite (:func:`gpri.diurnal.hour_composite`) is
the shape-agnostic estimate of the cycle itself, and what is left after it
is what did not repeat.

It answers a question the harmonic fits cannot: *what shape* is the
non-secular motion — a smooth afternoon-peaking oscillation, which is what
melt forcing gives, or a single event, which a 24 h harmonic will happily
render as a sinusoid of the right period and the wrong meaning.  The
held-out bedrock series is the control: whatever it does at the same hour
is the atmosphere and the reference, not the ice, and their correlation is
printed.

A diurnal harmonic is also fitted to each population series inside each
full-day window, in the epoch domain — with the population's noise this low
the ordinary least squares fit is fine — and reported next to the
pair-domain population phasor from ``baker_repeat.py`` for comparison.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES, integrate, load, split_mask          # noqa: E402

from gpri.aps import epoch_screen_correction, turbulence_screen    # noqa: E402
from gpri.diurnal import (DIURNAL, MIN_CYCLES, hour_composite,     # noqa: E402
                          m_per_yr, secular_slope)
from gpri.timeseries import los_displacement                       # noqa: E402


def population_path(scene: Path, antenna: str, dec: int) -> Path:
    """Where the population series of one scene/antenna are cached."""
    import os
    root = Path(os.environ.get("GPRI_WORK_ROOT", "work"))
    return root / scene.name / f"population_{antenna[0].lower()}_dec{dec}.npz"


def detrend_pixels(t, d):
    """Every pixel's series minus its own least-squares line; the rates too.

    The per-pixel trend, not the population's: glaciers flow at different
    speeds in different places, and a common trend would leave that
    difference in the anomaly and swamp the interquartile band with it.
    """
    G = np.column_stack([np.ones_like(t), t])
    flat = d.reshape(d.shape[0], -1)
    finite = np.isfinite(flat)
    x = np.linalg.lstsq(G, np.where(finite, flat, 0.0), rcond=None)[0]
    anom = (flat - G @ x).reshape(d.shape)
    return anom, x[1].reshape(d.shape[1:])


def harmonic(t, y):
    """OLS ``offset + rate t + a cos + b sin`` at one day: amplitude, peak."""
    ok = np.isfinite(y)
    w = 2 * np.pi / DIURNAL
    G = np.column_stack([np.ones(ok.sum()), t[ok], np.cos(w * t[ok]),
                         np.sin(w * t[ok])])
    x, *_ = np.linalg.lstsq(G, y[ok], rcond=None)
    resid = y[ok] - G @ x
    return np.hypot(x[2], x[3]), np.arctan2(x[3], x[2]), x[1], np.std(resid)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170827")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--rgi", action="store_true",
                    help="reference/null masks exclude RGI glacier outlines")
    ap.add_argument("--window", type=float, default=24.0,
                    help="length of each single-day window, hours")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene, args.scene))
    day = scene.name + ("" if args.antenna == "upper" else f"_{args.antenna}")

    stack, net, phase, cc, r, az, n = load(scene, args.decimate, args.pairs,
                                           antenna=args.antenna)
    lam = stack.wavelength
    mean_cc = cc.mean(axis=0)
    usable = mean_cc >= args.ice_coherence
    del cc
    stable = mean_cc >= args.stable_coherence
    if args.rgi:
        import os as _os
        from baker_north_side import decimated_par
        from gpri.geocode import BAKERBEND1_HEADING, RadarGeometry
        from gpri.heading import scene_heading
        from gpri.glaciers import glacier_mask, load_outlines, stable_ground_mask
        geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                             heading=scene_heading(scene, default=BAKERBEND1_HEADING))
        la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1],
                               cols=[0, geom.shape[1] - 1])
        bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
        gdf = load_outlines(_os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                            bbox=bbox)
        stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                       threshold=args.stable_coherence)
        ice = usable & glacier_mask(geom, gdf)
    else:
        ice = usable & ~stable
    fit_m, held_m = split_mask(stable)
    span = float(net.times[-1] * 24)
    print(f"{day}: {n} pairs over {span:.1f} h; ice {ice.sum():,} px, "
          f"bedrock {fit_m.sum():,} fit + {held_m.sum():,} held out")

    # ---- corrected series, exactly as baker_pairlsq.py --------------------
    d, times = integrate(los_displacement(phase, lam), net, n)
    del phase
    d, _ = epoch_screen_correction(d, fit_m, r, model="linear", weights=mean_cc)
    t0 = time.time()
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], fit_m, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        d[k] -= scr
    print(f"corrections in {time.time() - t0:.0f} s")

    t = np.asarray(times, float)                          # days
    hours = t * 24
    pops = {"ice": ice, "held-out bedrock": held_m}
    series = {k: np.nanmedian(d[:, m], axis=1) * 1000 for k, m in pops.items()}

    # ---- the secular signal: where did each population end up? -----------
    # (rock the corrections never saw must sit at zero; ice need not)
    last = d[-1] * 1000
    print(f"\ncumulative LOS after {hours[-1]:.1f} h, mm, positive towards the radar")
    print(f"  {'population':18s} {'px':>7s} {'median':>8s} {'mean':>8s} {'p16':>6s} {'p84':>6s}")
    for k, m in pops.items():
        v = last[m]
        v = v[np.isfinite(v)]
        p16, p84 = np.percentile(v, [16, 84])
        print(f"  {k:18s} {v.size:7,d} {np.median(v):+8.1f} {v.mean():+8.1f} "
              f"{p16:+6.0f} {p84:+6.0f}")
    v, ri = last[ice], np.broadcast_to(r, last.shape)[ice]
    ok = np.isfinite(v)
    print(f"  corr(ice displacement, slant range) {np.corrcoef(v[ok], ri[ok])[0, 1]:+.2f}"
          f"  -- near zero means motion, not a screen extrapolated over the ice\n")

    da, rate_px = detrend_pixels(t, d)
    del d
    linear = {k: np.nanmedian(da[:, m], axis=1) * 1000 for k, m in pops.items()}
    rate_lin = {k: np.nanmedian(rate_px[m]) * 1000 for k, m in pops.items()}  # mm/day
    origin = net.epochs[0].hour + net.epochs[0].minute / 60.0

    # ---- the secular line without a shape: same-hour differences ---------
    # the line's tilt is common to every pixel of a population that shares
    # the waveform, so it is measured once on the population median and
    # taken out of every pixel's anomaly (and put back into its rate)
    tilt = {}                      # mm/day the linear rate under-states by
    tol = (1 - MIN_CYCLES) * DIURNAL      # the harmonic fits' allowance
    try:
        for k in pops:
            tilt[k] = secular_slope(t, linear[k], tolerance=tol)[0]
        detrend = "periodic"
    except ValueError as e:        # under one cycle: the line is all there is
        print(f"secular rate from same-hour differences: {e}")
        tilt = {k: 0.0 for k in pops}
        detrend = "linear"
    tc = t - t.mean()
    anom = {k: linear[k] - tilt[k] * tc for k in pops}
    rates = {k: rate_lin[k] + tilt[k] for k in pops}
    da[:, ice] -= (tilt["ice"] / 1000) * tc[:, None]
    q_anom = np.nanpercentile(da[:, ice], [25, 75], axis=1) * 1000
    del da
    if detrend == "periodic":
        n_same = secular_slope(t, linear["ice"], tolerance=tol)[1]
        print(f"secular rate by same-hour differences ({n_same} pairs of epochs "
              f"24 h apart): ice {m_per_yr(rates['ice'], 'mm'):+.2f} m/yr, "
              f"held-out bedrock {m_per_yr(rates['held-out bedrock'], 'mm'):+.2f} m/yr "
              f"(ground that does not move: the estimator's error on this record)")
        print(f"  the linear fit gave ice {m_per_yr(rate_lin['ice'], 'mm'):+.2f} m/yr: "
              f"the waveform tilts the line by {m_per_yr(-tilt['ice'], 'mm'):+.2f} m/yr, "
              f"{-tilt['ice'] * t[-1] / 2:+.1f} mm at either end of the record")
    else:
        print(f"median linear rate over the record: ice "
              f"{m_per_yr(rates['ice'], 'mm'):+.2f} m/yr, held-out bedrock "
              f"{m_per_yr(rates['held-out bedrock'], 'mm'):+.2f} m/yr")
    # the shape correlation is taken with both lines removed: the same-hour
    # tilt on bedrock is the estimator's noise, a ramp that would correlate
    # with the ice ramp and say nothing about whether the waveforms match
    print(f"trend anomaly RMS ({detrend} detrend): ice {np.nanstd(anom['ice']):.2f} mm, "
          f"bedrock {np.nanstd(anom['held-out bedrock']):.2f} mm; shape correlation "
          f"{np.corrcoef(linear['ice'], linear['held-out bedrock'])[0, 1]:.2f}")

    # ---- the cycle itself, if there is more than one: hour-of-day composite
    hod = (origin + hours) % 24
    composite = None
    if t[-1] >= 1.5 * DIURNAL:
        composite = {k: hour_composite(hod, anom[k])[0] for k in pops}
        n_days = len({e.date() for e in net.epochs})
        print(f"\nhour-of-day composite over the record's days, {detrend} detrend, "
              f"mm by UTC hour")
        print(f"{'':18s}" + "".join(f"{h:>5d}" for h in range(24)))
        for k in pops:
            row = "".join("    ." if not np.isfinite(v) else f"{v:5.1f}"
                          for v in composite[k])
            print(f"{k:18s}{row}")
        for k in pops:
            c = composite[k][hod.astype(int)]
            resid = anom[k] - c
            print(f"  {k}: composite RMS {np.nanstd(composite[k]):.2f} mm, "
                  f"what did not repeat {np.nanstd(resid):.2f} mm")

    # ---- what a 24 h harmonic makes of each window ------------------------
    win = args.window / 24.0
    windows = {"both": (0.0, t[-1])}
    if t[-1] >= win + 1 / 24:
        windows = {"day 1": (0.0, win), "day 2": (t[-1] - win, t[-1]), **windows}
    diurnal = t[-1] >= DIURNAL * MIN_CYCLES
    if not diurnal:
        # a sub-cycle record: the anomaly series above is still the product
        # (baker_seasons.py overlays it); a 24 h harmonic of it is not
        windows = {}
        print(f"\n{day} spans {hours[-1]:.1f} h ({t[-1]:.2f} cycles); the 24 h "
              f"harmonic needs {MIN_CYCLES:g} -- skipped")
    else:
        print(f"\n{'window':8s} {'population':18s} {'amp':>8s} {'peak UTC':>9s} "
              f"{'rate':>11s} {'resid':>8s}")
    fits = {}
    for wname, (lo, hi) in windows.items():
        sel = (t >= lo) & (t <= hi)
        for k in series:
            a, ph, rate, sd = harmonic(t[sel], series[k][sel])
            peak = np.mod(origin + ph / (2 * np.pi) * 24, 24)
            fits[wname, k] = (a, peak)
            print(f"{wname:8s} {k:18s} {a:6.2f} mm {peak:7.1f} h "
                  f"{m_per_yr(rate, 'mm'):+8.2f} m/yr {sd:6.2f} mm")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2]})
    ax = axes[0]
    ax.fill_between(hours, q_anom[0], q_anom[1], color="0.8",
                    label="ice interquartile range (per-pixel anomalies)")
    if detrend == "periodic":
        ax.plot(hours, linear["ice"], color="0.45", lw=0.8, ls="--",
                label=f"ice median, linear detrend ({m_per_yr(rate_lin['ice'], 'mm'):+.1f} m/yr)")
    ax.plot(hours, anom["ice"], "k", lw=1.2,
            label=f"ice median ({m_per_yr(rates['ice'], 'mm'):+.1f} m/yr removed)")
    ax.plot(hours, anom["held-out bedrock"], color="tab:red", lw=1.0,
            label="held-out bedrock median")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Trend anomaly (mm)")
    how = ("secular rate from same-hour differences" if detrend == "periodic"
           else "each pixel's linear trend")
    ax.set_title(f"{day}: departure from {how}, population medians, from "
                 f"{net.epochs[0]:%Y-%m-%d %H:%M} UTC", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    if composite is not None:
        ax.plot(hours, composite["ice"][hod.astype(int)], color="tab:blue",
                lw=1.6, drawstyle="steps-mid",
                label=f"ice hour-of-day composite over {n_days} UTC days")
    for wname, (lo, hi) in windows.items():
        if wname == "both" and len(windows) > 1:
            continue          # one-day record: the whole-record fit is the day
        sel = (t >= lo) & (t <= hi)
        for k, colour in (("ice", "k"), ("held-out bedrock", "tab:red")):
            a, peak = fits[wname, k]
            ph = (peak - origin) / 24 * 2 * np.pi
            ax.plot(hours[sel], a * np.cos(2 * np.pi * t[sel] - ph), color=colour,
                    lw=1.2 if k == "ice" else 0.8,
                    label=f"{k} {wname}: {a:.1f} mm, peak {peak:.1f} h UTC")
    if not diurnal:
        # too short for a harmonic: show the median series with its trend
        for k, colour in (("ice", "k"), ("held-out bedrock", "tab:red")):
            ax.plot(hours, series[k], color=colour, lw=1.2 if k == "ice" else 0.8,
                    label=f"{k} median: {m_per_yr(rates[k], 'mm'):+.1f} m/yr")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("24 h harmonic (mm)" if diurnal else "Median LOS (mm)")
    ax.set_xlabel("Elapsed time (hr)")
    # the legend sits under the panel: with a composite and two days' fits
    # there is no corner of the axes it would not cover
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8,
              ncol=3, frameon=False)
    # a UTC clock along the top, every six hours
    top = axes[0].secondary_xaxis("top")
    utc = np.arange(np.ceil(origin / 6) * 6, origin + hours[-1] + 1e-9, 6)
    top.set_xticks(utc - origin)
    top.set_xticklabels([f"{int(u) % 24:02d}" for u in utc])
    top.set_xlabel("UTC (hr)")
    for u in utc[(utc % 24) == 0]:
        for ax in axes:
            ax.axvline(u - origin, color="0.6", lw=0.6, ls=":")
    plt.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"19_population_{day}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"\nwrote {out}")

    # the population series themselves, for baker_seasons.py to overlay
    npz = population_path(scene, args.antenna, args.decimate)
    # rates in m/yr; ``detrend`` says which line the anomalies are from
    np.savez(npz, hours=hours, origin=origin,
             epoch0=np.datetime64(net.epochs[0]).astype("datetime64[s]"),
             detrend=detrend,
             ice=anom["ice"], rock=anom["held-out bedrock"],
             ice_linear=linear["ice"], rock_linear=linear["held-out bedrock"],
             q25=q_anom[0], q75=q_anom[1],
             ice_series=series["ice"], rock_series=series["held-out bedrock"],
             ice_rate=m_per_yr(rates["ice"], "mm"),
             rock_rate=m_per_yr(rates["held-out bedrock"], "mm"),
             ice_rate_linear=m_per_yr(rate_lin["ice"], "mm"),
             rock_rate_linear=m_per_yr(rate_lin["held-out bedrock"], "mm"))
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
