#!/usr/bin/env python3
"""Two antennas, one day: the replicate that a single GPRI record never had.

    python examples/baker_antennas.py --scene 20170803 --decimate 16 --rgi

The GPRI-II receives on two antennas, 25 cm apart on the same mast, sampled
in the same sweep.  GAMMA only ever processed the upper one.  Forming the
lower antenna's pairs from its SLCs (:class:`gpri.stack.SlcPairStack`) and
running them through the identical chain gives a second, independent
realisation of the same 24 hours — same atmosphere, same ice, same
geometry to within the antenna spacing, but independent thermal noise and
independently decorrelating speckle.

What that buys, in order of importance:

1.  **A noise floor that is measured, not assumed.**  Everything common to
    the antennas cancels in ``upper - lower`` — deformation and atmosphere
    both — so the difference series is the pipeline's own noise, separately
    on ice and on rock, and the residual on held-out bedrock can finally be
    split into "atmosphere" and "noise".
2.  **A replication test for the diurnal signal.**  Noise does not replicate
    with the same phase in an independent channel; a forced diurnal
    response does.  Where both antennas fit the same amplitude *and* the same
    hour of peak, the fit is a measurement.  Where they disagree, it is not.
3.  **A better product.**  Averaging the two channels lowers the noise by
    ``sqrt(2)`` on everything the difference test says is noise.

Honesty protocol as in ``baker_aps.py``: RGI bedrock is split, corrections
see one half, statistics come from the other.
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
from gpri.diurnal import DIURNAL                                   # noqa: E402
from gpri.pairlsq import fit_pairs                                 # noqa: E402
from gpri.timeseries import los_displacement                       # noqa: E402


def masks(stack, mean_cc, args):
    """(stable, ice) with or without the RGI audit, as in the other examples."""
    import os as _os
    usable = mean_cc >= args.ice_coherence
    stable = mean_cc >= args.stable_coherence
    if not args.rgi:
        return stable, usable & ~stable
    from baker_north_side import decimated_par
    from gpri.geocode import BAKERBEND1_HEADING, RadarGeometry
    from gpri.glaciers import glacier_mask, load_outlines, stable_ground_mask
    geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                         heading=BAKERBEND1_HEADING)
    la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1],
                           cols=[0, geom.shape[1] - 1])
    bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
    gdf = load_outlines(_os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                        bbox=bbox)
    stable, _ = stable_ground_mask(mean_cc, geom, gdf,
                                   threshold=args.stable_coherence)
    return stable, usable & glacier_mask(geom, gdf)


def corrected_series(phase, net, n, lam, fit_m, r, mean_cc, sigma):
    """The full ladder of ``baker_aps.py`` (A -> D) on one antenna."""
    d, times = integrate(los_displacement(phase, lam), net, n)
    d, _ = epoch_screen_correction(d, fit_m, r, model="linear", weights=mean_cc)
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], fit_m, sigma=sigma, weights=mean_cc,
                                   wrapped=False)
        d[k] -= scr
    return d, times


def rms_mm(a, mask):
    v = a[:, mask]
    return float(np.sqrt(np.nanmean(v ** 2)) * 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--rgi", action="store_true")
    ap.add_argument("--snr-threshold", type=float, default=3.0)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene, args.scene))
    day = scene.name
    sigma = tuple(args.sigma)

    # ---- both antennas, one mask set --------------------------------------
    series, fits, cc_maps, wts = {}, {}, {}, {}
    common = None
    for ant in ("upper", "lower"):
        stack, net, phase, cc, r, az, n = load(scene, args.decimate, args.pairs,
                                               antenna=ant)
        lam = stack.wavelength
        mean_cc = cc.mean(axis=0)
        cc_maps[ant] = mean_cc
        usable = mean_cc >= args.ice_coherence
        wts[ant] = np.median(cc[:, usable], axis=1).astype(float)
        del cc
        if common is None:
            # masks from the upper antenna's coherence, shared by both, so the
            # comparison is pixel-for-pixel on identical ground
            stable, ice = masks(stack, mean_cc, args)
            fit_m, held_m = split_mask(stable)
            common = (stable, ice, fit_m, held_m, r, lam, net, n)
            print(f"{day}: ice {ice.sum():,} px, bedrock {fit_m.sum():,} fit "
                  f"+ {held_m.sum():,} held out")
        stable, ice, fit_m, held_m, r, lam, net, n = common
        t0 = time.time()
        d, times = corrected_series(phase, net, n, lam, fit_m, r, mean_cc, sigma)
        del phase
        series[ant] = d
        obs = d[1:] - d[:-1]
        fits[ant] = fit_pairs(obs, net, periods=(DIURNAL,), weights=wts[ant])
        print(f"  {ant}: mean coherence ice {np.nanmean(mean_cc[ice]):.3f} "
              f"rock {np.nanmean(mean_cc[held_m]):.3f}; chain + fit in "
              f"{time.time() - t0:.0f} s")

    du, dl = series["upper"], series["lower"]
    diff = du - dl
    mean = 0.5 * (du + dl)
    hours = (times - times[0]) * 24

    # ---- 1. the noise floor -----------------------------------------------
    print("\nRMS over the day, mm LOS  (difference/sqrt2 = one antenna's noise)")
    print(f"  {'':18s} {'upper':>8s} {'lower':>8s} {'mean':>8s} {'u-l':>8s} {'noise':>8s}")
    rows = {}
    for label, m in (("held-out bedrock", held_m), ("ice", ice)):
        vals = (rms_mm(du, m), rms_mm(dl, m), rms_mm(mean, m), rms_mm(diff, m))
        noise = vals[3] / np.sqrt(2)
        rows[label] = vals + (noise,)
        print(f"  {label:18s} {vals[0]:8.2f} {vals[1]:8.2f} {vals[2]:8.2f} "
              f"{vals[3]:8.2f} {noise:8.2f}")
    rock = rows["held-out bedrock"]
    atm = np.sqrt(max(rock[0] ** 2 - rock[4] ** 2, 0.0))
    print(f"\n  held-out bedrock residual {rock[0]:.2f} mm = noise {rock[4]:.2f} mm"
          f" (+) common-mode error {atm:.2f} mm\n  — the common part is what the "
          f"two antennas share: un-modelled atmosphere and reference error, "
          f"not measurement noise.")

    # correlation of the two channels' residual on rock: 1 = all atmosphere
    ru = du[:, held_m].ravel(); rl = dl[:, held_m].ravel()
    ok = np.isfinite(ru) & np.isfinite(rl)
    print(f"  corr(upper, lower) on held-out bedrock: "
          f"{np.corrcoef(ru[ok], rl[ok])[0, 1]:.3f}")

    # ---- 2. does the diurnal fit replicate? --------------------------------
    thr = args.snr_threshold
    fu, fl = fits["upper"], fits["lower"]
    au, al = fu.amplitude(DIURNAL) * 1000, fl.amplitude(DIURNAL) * 1000
    pu, pl = fu.peak_time(DIURNAL), fl.peak_time(DIURNAL)     # hours
    su, sl = fu.snr(DIURNAL), fl.snr(DIURNAL)
    both = ice & (su > thr) & (sl > thr)
    dphase_h = (pu - pl + 12) % 24 - 12
    print(f"\ndiurnal fit replication, SNR > {thr:g} in both antennas:")
    for label, m in (("ice", ice), ("held-out bedrock", held_m)):
        sel = m & np.isfinite(au) & np.isfinite(al)
        c = np.corrcoef(au[sel], al[sel])[0, 1] if sel.sum() > 2 else np.nan
        det_u = 100 * np.nanmean(su[m] > thr)
        det_l = 100 * np.nanmean(sl[m] > thr)
        det_b = 100 * np.nanmean((su[m] > thr) & (sl[m] > thr))
        print(f"  {label:18s} amp upper {np.nanmedian(au[m]):5.2f} lower "
              f"{np.nanmedian(al[m]):5.2f} mm  corr(amp) {c:5.2f}  "
              f"SNR>{thr:g}: upper {det_u:4.1f}%  lower {det_l:4.1f}%  "
              f"both {det_b:4.1f}%")
    if both.any():
        print(f"  ice pixels significant in both: {both.sum():,}; peak-time "
              f"difference median {np.nanmedian(dphase_h[both]):+.2f} h, "
              f"IQR {np.nanpercentile(dphase_h[both], 75) - np.nanpercentile(dphase_h[both], 25):.2f} h"
              f"  (independent noise would give a uniform spread, IQR 12 h)")
        agree = np.abs(dphase_h[both]) < 2
        print(f"  within 2 h of each other: {100 * agree.mean():.1f}%")
    det_rock_both = 100 * np.nanmean((su[held_m] > thr) & (sl[held_m] > thr))
    print(f"  false alarms surviving the replication test on rock: "
          f"{det_rock_both:.2f}%")

    # ---- 3. the combined product -----------------------------------------
    w = 0.5 * (wts["upper"] + wts["lower"])
    fm = fit_pairs(mean[1:] - mean[:-1], net, periods=(DIURNAL,), weights=w)
    am, sm = fm.amplitude(DIURNAL) * 1000, fm.snr(DIURNAL)
    print(f"\ntwo-antenna mean: ice amp {np.nanmedian(am[ice]):.2f} mm, median "
          f"SNR {np.nanmedian(sm[ice]):.2f} (upper alone {np.nanmedian(su[ice]):.2f}); "
          f"rock SNR>{thr:g} {100 * np.nanmean(sm[held_m] > thr):.1f}%")

    # ---- figure ------------------------------------------------------------
    usable = cc_maps["upper"] >= args.ice_coherence
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    vmax = np.nanpercentile(np.where(usable, au, np.nan), 98)
    for ax, a, t in ((axes[0, 0], au, "upper antenna: diurnal amplitude"),
                     (axes[0, 1], al, "lower antenna: diurnal amplitude")):
        im = ax.imshow(np.where(usable, a, np.nan), cmap="magma", vmin=0,
                       vmax=vmax, aspect="auto")
        ax.set_title(t, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                     label="Diurnal amplitude (mm)")
    nmap = np.sqrt(np.nanmean(diff ** 2, axis=0)) * 1000 / np.sqrt(2)
    im = axes[0, 2].imshow(np.where(usable, nmap, np.nan), cmap="viridis",
                           vmin=0, vmax=np.nanpercentile(nmap[usable], 98),
                           aspect="auto")
    axes[0, 2].set_title("single-antenna noise, RMS(u-l)/sqrt2", fontsize=10)
    fig.colorbar(im, ax=axes[0, 2], fraction=0.04, pad=0.02,
                 label="Noise RMS (mm)")
    for ax in axes[0]:
        ax.set_xlabel("range sample"); ax.set_ylabel("azimuth line")

    ax = axes[1, 0]
    sel = ice & np.isfinite(au) & np.isfinite(al)
    ax.hexbin(au[sel], al[sel], gridsize=45, bins="log", cmap="Greys",
              extent=(0, vmax, 0, vmax))
    ax.plot([0, vmax], [0, vmax], "r-", lw=0.8)
    ax.set_xlabel("Upper amplitude (mm)"); ax.set_ylabel("Lower amplitude (mm)")
    ax.set_title("ice: amplitude, antenna vs antenna", fontsize=10)

    ax = axes[1, 1]
    if both.any():
        ax.hist(dphase_h[both], bins=48, range=(-12, 12), color="#2f7ed8")
    ax.axvspan(-2, 2, color="k", alpha=0.08)
    ax.set_xlabel("Peak-time difference (hr)")
    ax.set_title(f"ice, SNR>{thr:g} in both: peak time, upper minus lower",
                 fontsize=10)

    ax = axes[1, 2]
    for label, m, c in (("ice", ice, "#d62728"),
                        ("held-out bedrock", held_m, "#2f7ed8")):
        ax.plot(hours, np.nanmedian(du[:, m], axis=1) * 1000, "-", color=c,
                lw=1.2, label=f"{label}, upper")
        ax.plot(hours, np.nanmedian(dl[:, m], axis=1) * 1000, "--", color=c,
                lw=1.2, label=f"{label}, lower")
    ax.set_xlabel("Elapsed time (hr)"); ax.set_ylabel("Median LOS (mm)")
    ax.legend(fontsize=7, ncol=2); ax.set_title("median series, both channels (+ toward radar)",
                                                 fontsize=10)
    plt.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"17_antennas_{day}.png"
    plt.savefig(out, dpi=140); plt.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
