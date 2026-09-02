#!/usr/bin/env python3
"""Ohenhen-style single-step WLS on a GPRI day, with honest error bars.

    python examples/baker_pairlsq.py --scene 20170803 --decimate 16

Fits rate + diurnal directly to the corrected pair observations
(:func:`gpri.pairlsq.fit_pairs`) with per-pair coherence weights, and puts the
result beside the two-step (integrate-then-fit) estimate.  The headline
product is the per-pixel **SNR map**: amplitude over its formal standard
error, which turns "is there a diurnal signal?" into a number with a
threshold instead of an eyeball judgement.

Honesty protocol as in ``baker_aps.py``: the bedrock mask is split, the
corrections and the atmospheric covariate see one half, and every statistic
quoted on stable ground comes from the held-out half.  A calibration check
falls out for free: held-out bedrock is not moving, so the fraction of it
exceeding SNR 3 is the real false-alarm rate of the whole chain — if the
error bars were honest and the model complete it would be well under 1 %, and
the amount by which it is not measures the un-modelled atmosphere.
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
from gpri.diurnal import DIURNAL, MIN_CYCLES, fit_harmonics        # noqa: E402
from gpri.pairlsq import fit_pairs                                 # noqa: E402
from gpri.refractivity import invert_refractivity, screens_to_delta_n  # noqa: E402
from gpri.timeseries import los_displacement                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170803")
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--ice-coherence", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 25.0))
    ap.add_argument("--rgi", action="store_true",
                    help="reference/null masks exclude RGI glacier outlines")
    ap.add_argument("--snr-threshold", type=float, default=3.0)
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"),
                    help="which receive antenna; 'lower' is formed from the "
                         "SLCs (GAMMA only processed the upper)")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene, args.scene))
    day = scene.name + ("" if args.antenna == "upper" else f"_{args.antenna}")

    stack, net, phase, cc, r, az, n = load(scene, args.decimate, args.pairs,
                                           antenna=args.antenna)
    # the record fitted ends at the last epoch a used pair touches, not at
    # the last epoch of the campaign (``--pairs`` truncates the network)
    last = float(net.times[max(j for _, j in net.pairs[:n])])
    span = last * 24
    if last < DIURNAL * MIN_CYCLES:
        print(f"{day} spans {span:.1f} h ({span / 24:.2f} cycles); a diurnal fit "
              f"needs {MIN_CYCLES:g} -- nothing to estimate, see the population "
              f"and aps steps for the rates")
        return
    lam = stack.wavelength
    mean_cc = cc.mean(axis=0)
    # per-pair scalar weight: median coherence over the usable swath.
    # coherence -> phase variance is monotone; the median is robust and cheap.
    usable = mean_cc >= args.ice_coherence
    w_pair = np.median(cc[:, usable], axis=1).astype(float)
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
        stable, contested = stable_ground_mask(mean_cc, geom, gdf,
                                               threshold=args.stable_coherence)
        on_ice = glacier_mask(geom, gdf)
        ice = usable & on_ice                  # ice now MEANS ice
        print(f"RGI: reference drops {contested.sum():,} coherent-but-glacier px; "
              f"ice mask is now RGI-defined ({ice.sum():,} px)")
    else:
        ice = usable & ~stable
    fit_m, held_m = split_mask(stable)
    print(f"{day}: {n} pairs; ice {ice.sum():,} px, "
          f"bedrock {fit_m.sum():,} fit + {held_m.sum():,} held out")

    # ---- corrected series, fit-half bedrock only (as in the aps ladder) ----
    d, times = integrate(los_displacement(phase, lam), net, n)
    del phase
    d, _ = epoch_screen_correction(d, fit_m, r, model="linear",
                                   weights=mean_cc)
    t0 = time.time()
    for k in range(d.shape[0]):
        scr, _ = turbulence_screen(d[k], fit_m, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        d[k] -= scr
    print(f"corrections in {time.time() - t0:.0f} s")

    # ---- the two estimators on identical data -----------------------------
    # re-difference so the epoch-domain corrections land on the pairs without
    # touching the pair noise structure
    obs = d[1:] - d[:-1] if all(j == i + 1 for i, j in net.pairs[:n]) else \
        np.stack([d[j] - d[i] for i, j in net.pairs[:n]])

    t0 = time.time()
    single = fit_pairs(obs, net, periods=(DIURNAL,), weights=w_pair)
    print(f"pair-domain WLS in {time.time() - t0:.1f} s: {single!r}")
    two = fit_harmonics(d, times)

    amp1 = single.amplitude(DIURNAL)
    snr = single.snr(DIURNAL)
    amp2 = two.amplitude(DIURNAL)

    # ---- report ------------------------------------------------------------
    thr = args.snr_threshold

    def stats(mask, label):
        a1 = np.nanmedian(amp1[mask]) * 1000
        a2 = np.nanmedian(amp2[mask]) * 1000
        s = np.nanmedian(snr[mask])
        det = 100 * np.nanmean(snr[mask] > thr)
        print(f"  {label:16s} amp(single) {a1:6.2f} mm  amp(two-step) "
              f"{a2:6.2f} mm  median SNR {s:5.2f}  SNR>{thr:g}: {det:5.1f}%")
        return det

    print(f"\ndiurnal fit, {day}:")
    stats(ice, "ice")
    fa = stats(held_m, "held-out bedrock")
    print(f"\n  held-out bedrock is not moving, so its {fa:.1f}% above "
          f"SNR {thr:g} is the real\n  false-alarm rate of the whole chain — "
          f"the gap above ~0.5% is un-modelled\n  atmosphere, not estimator "
          f"optimism.")

    # ---- covariate run: project the refractivity series out ----------------
    from gpri import atmosphere
    dN = []
    for p in range(min(n, obs.shape[0])):
        try:
            s = atmosphere.fit_screen(
                np.angle(np.exp(1j * obs[p] / (-lam / (4 * np.pi)))),
                slant_range=r, weights=np.where(fit_m, mean_cc, 0.0),
                model="linear", wavelength=lam)
            dN.append(s)
        except Exception:
            dN.append(None)
    dn_pair = screens_to_delta_n(dN, wavelength=lam)
    okp = np.isfinite(dn_pair)
    N_epoch = invert_refractivity(np.where(okp, dn_pair, 0.0), net,
                                  weights=okp.astype(float)).ravel()
    cov = fit_pairs(obs, net, periods=(DIURNAL,), weights=w_pair,
                    covariates={"N": N_epoch[: len(times)]})
    a_cov = np.nanmedian(cov.amplitude(DIURNAL)[ice]) * 1000
    s_cov = np.nanmedian(cov.snr(DIURNAL)[ice])
    print(f"\n  with the refractivity covariate inside the fit: ice amp "
          f"{a_cov:.2f} mm, median SNR {s_cov:.2f}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    a = np.where(usable, amp1 * 1000, np.nan)
    im = axes[0].imshow(a, cmap="magma", vmin=0,
                        vmax=np.nanpercentile(a, 98), aspect="auto")
    axes[0].set_title("diurnal amplitude, single-step WLS (mm)")
    fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)

    s = np.where(usable, snr, np.nan)
    im = axes[1].imshow(s, cmap="viridis", vmin=0, vmax=2 * thr, aspect="auto")
    axes[1].contour(np.nan_to_num(s), levels=[thr], colors="r",
                    linewidths=0.4)
    axes[1].set_title(f"amplitude / sigma  (red contour: SNR = {thr:g})")
    fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)
    for ax in axes:
        ax.set_xlabel("Range (samples)")
        ax.set_ylabel("Azimuth (lines)")
    plt.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"15_pairlsq_{day}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
