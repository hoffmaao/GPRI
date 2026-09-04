#!/usr/bin/env python3
"""Does the diurnal signal repeat?  The question two cycles can answer.

    python examples/baker_repeat.py --scene 20170827 --decimate 16 --rgi

One day of data fits a diurnal harmonic; it cannot say whether the harmonic
is a *forced response* (melt input peaking every afternoon) or a one-off
(one warm afternoon, one storm, a drift in the reference).  20170827 spans
44.9 h — 1.87 cycles — so the same pair-domain WLS as ``baker_pairlsq.py`` is
run three times on identically corrected observations:

* **day 1** — pairs inside the first 24 h,
* **day 2** — pairs inside the last 24 h (the two windows overlap by ~3 h,
  which is what 1.87 cycles allows two full cycles to do),
* **both** — every pair, the fit the campaign was designed for.

Then day 1 against day 2.  A forced diurnal has the same amplitude and the
same phase on consecutive days; an atmospheric residual, or a one-off, does
not.  Per pixel the SNR is about 1 and almost nothing is detected on either
day, so the comparison is made at the population level: the mean of the
per-pixel phasors ``a + ib`` over all RGI ice (the glacier's response is
spatially coherent, single-look noise is not), the same mean over the
held-out bedrock as its null, and peak-hour maps of block-averaged phasors
where the ice beats what the same averaging leaves on bedrock.  The bedrock
half that never sees the corrections (``split_mask``) also reports the
false-alarm rate of each fit, as in ``baker_pairlsq.py``.

The other thing two cycles buy is written into the covariance: over exactly
one cycle the rate and the harmonic are nearly collinear, and the fit
reports the parameter correlation between them for each window.
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

from gpri_tools.aps import epoch_screen_correction, turbulence_screen    # noqa: E402
from gpri_tools.diurnal import DIURNAL, m_per_yr                         # noqa: E402
from gpri_tools.network import Network                                   # noqa: E402
from gpri_tools.pairlsq import fit_pairs                                 # noqa: E402
from gpri_tools.timeseries import los_displacement                       # noqa: E402


def window_fit(obs, net, w_pair, t_lo, t_hi, label):
    """WLS on the pairs whose epochs both fall in ``[t_lo, t_hi]`` days."""
    t = net.times
    keep = (t[net.pairs[:, 0]] >= t_lo) & (t[net.pairs[:, 1]] <= t_hi)
    sub = Network(net.epochs, net.pairs[keep])
    fit = fit_pairs(obs[keep], sub, periods=(DIURNAL,), weights=w_pair[keep])
    ia, ib = fit._ab(DIURNAL)
    ir = fit._idx("t^1")
    c = fit.cov_unit
    rho = max(abs(c[ir, ia]) / np.sqrt(c[ir, ir] * c[ia, ia]),
              abs(c[ir, ib]) / np.sqrt(c[ir, ir] * c[ib, ib]))
    print(f"  {label:6s} {keep.sum():5d} pairs over {(t_hi - t_lo) * 24:5.1f} h;  "
          f"|corr(rate, harmonic)| = {rho:.3f}")
    return fit, rho


def circ_diff_hours(a, b):
    return (a - b + 12.0) % 24.0 - 12.0


def phasor(fit):
    """``a + ib`` per pixel: the diurnal term is ``Re[(a + ib) exp(-i w t)]``."""
    ia, ib = fit._ab(DIURNAL)
    return fit.params[ia] + 1j * fit.params[ib]


def peak_of(z, origin_hour):
    """Hour of day at which the phasor's cosine peaks."""
    return np.mod(origin_hour + np.angle(z) / (2 * np.pi) * 24.0, 24.0)


def smooth_phasor(z, mask, size):
    """Block mean of the phasor field over ``mask`` (normalised convolution)."""
    from scipy.ndimage import uniform_filter
    w = uniform_filter(mask.astype(float), size)
    num = (uniform_filter(np.where(mask, z.real, 0.0), size) +
           1j * uniform_filter(np.where(mask, z.imag, 0.0), size))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0.25, num / w, np.nan)


def phasor_corr(z1, z2):
    """Complex correlation of two phasor fields over the same pixels.

    ``|c|`` is 1 for identical waveforms at every pixel and ~1/sqrt(N) for
    independent noise; ``arg(c)`` is the systematic phase shift between them.
    """
    ok = np.isfinite(z1) & np.isfinite(z2)
    z1, z2 = z1[ok], z2[ok]
    return np.sum(z1 * np.conj(z2)) / np.sqrt(np.sum(np.abs(z1) ** 2) *
                                              np.sum(np.abs(z2) ** 2))


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
    ap.add_argument("--snr-threshold", type=float, default=3.0)
    ap.add_argument("--window", type=float, default=24.0,
                    help="length of each single-day window, hours")
    ap.add_argument("--block", type=int, default=15,
                    help="block size (px) for the peak-hour maps")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene, args.scene))
    day = scene.name + ("" if args.antenna == "upper" else f"_{args.antenna}")

    stack, net, phase, cc, r, az, n = load(scene, args.decimate, args.pairs,
                                           antenna=args.antenna)
    lam = stack.wavelength
    span = float(net.times[-1] * 24)
    win = args.window / 24.0
    if net.times[-1] < win + 1.0 / 24:
        # a one-day record has nothing to compare: not an error, just no figure
        print(f"{day} spans {span:.1f} h; two {args.window:g} h windows "
              f"need more than that -- nothing to compare")
        return
    mean_cc = cc.mean(axis=0)
    usable = mean_cc >= args.ice_coherence
    w_pair = np.median(cc[:, usable], axis=1).astype(float)
    del cc
    stable = mean_cc >= args.stable_coherence
    if args.rgi:
        import os as _os
        from baker_north_side import decimated_par
        from gpri_tools.geocode import BAKERBEND1_HEADING, RadarGeometry
        from gpri_tools.heading import scene_heading
        from gpri_tools.glaciers import glacier_mask, load_outlines, stable_ground_mask
        geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                             heading=scene_heading(scene, default=BAKERBEND1_HEADING))
        la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1],
                               cols=[0, geom.shape[1] - 1])
        bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
        gdf = load_outlines(_os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                            bbox=bbox)
        stable, contested = stable_ground_mask(mean_cc, geom, gdf,
                                               threshold=args.stable_coherence)
        ice = usable & glacier_mask(geom, gdf)
    else:
        ice = usable & ~stable
    fit_m, held_m = split_mask(stable)
    print(f"{day}: {n} pairs over {span:.1f} h ({span / 24:.2f} cycles); "
          f"ice {ice.sum():,} px, bedrock {fit_m.sum():,} fit + "
          f"{held_m.sum():,} held out")

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
    obs = np.stack([d[j] - d[i] for i, j in net.pairs[:n]])
    del d
    net.pairs = net.pairs[:n]

    # ---- three fits --------------------------------------------------------
    t_end = float(net.times[-1])
    print("fits:")
    f1, rho1 = window_fit(obs, net, w_pair, 0.0, win, "day 1")
    f2, rho2 = window_fit(obs, net, w_pair, t_end - win, t_end, "day 2")
    fb, rhob = window_fit(obs, net, w_pair, 0.0, t_end, "both")
    origin = net.epochs[0].hour + net.epochs[0].minute / 60.0     # UTC
    fits = {"day 1": f1, "day 2": f2, "both": fb}
    amp = {k: f.amplitude(DIURNAL) * 1000 for k, f in fits.items()}
    snr = {k: f.snr(DIURNAL) for k, f in fits.items()}
    peak = {k: f.peak_time(DIURNAL, origin_hour=origin) for k, f in fits.items()}
    rate = {k: m_per_yr(f.secular) for k, f in fits.items()}       # m/yr

    z = {k: phasor(f) * 1000 for k, f in fits.items()}                # mm

    # the peak hour is circular, so the population's is that of its mean
    # phasor, not the median of per-pixel hours
    thr = args.snr_threshold
    print(f"\n{'fit':6s} {'ice amp':>8s} {'ice SNR':>8s} {'ice peak UTC':>13s} "
          f"{'ice rate':>9s} {'bedrock FA':>11s}")
    for k in fits:
        a = np.nanmedian(amp[k][ice]); s = np.nanmedian(snr[k][ice])
        p = peak_of(np.nanmean(z[k][ice]), origin); v = np.nanmedian(rate[k][ice])
        fa = 100 * np.nanmean(snr[k][held_m] > thr)
        print(f"{k:6s} {a:6.2f} mm {s:8.2f} {p:10.1f} h  {v:+6.2f} m/yr"
              f"  {fa:8.1f}%")

    # ---- does it repeat? --------------------------------------------------
    det = ice & (snr["day 1"] > thr) & (snr["day 2"] > thr)
    a1, a2 = amp["day 1"][det], amp["day 2"][det]
    dp = circ_diff_hours(peak["day 1"][det], peak["day 2"][det])
    r_amp = np.corrcoef(a1, a2)[0, 1] if det.sum() > 2 else np.nan
    ratio = np.nanmedian(a2 / a1) if det.any() else np.nan
    print(f"\nice pixels detected (SNR > {thr:g}) on both days: {det.sum():,} "
          f"of {ice.sum():,}")
    print(f"  amplitude day 2 / day 1: median {ratio:.2f}  (corr {r_amp:.2f})")
    print(f"  peak hour day 1 - day 2: median {np.nanmedian(dp):+.1f} h, "
          f"MAD {np.nanmedian(np.abs(dp - np.nanmedian(dp))):.1f} h")
    print(f"  rate/harmonic correlation: day 1 {rho1:.3f}, day 2 {rho2:.3f}, "
          f"both {rhob:.3f}")
    sig1 = np.nanmedian(f1.amplitude_sigma(DIURNAL)[ice]) * 1000
    sigb = np.nanmedian(fb.amplitude_sigma(DIURNAL)[ice]) * 1000
    print(f"  amplitude sigma on ice: one day {sig1:.2f} mm, both {sigb:.2f} mm")

    # ---- population level: the mean phasor --------------------------------
    # Per-pixel SNR is about 1, so the per-pixel test above can only ever
    # count a handful of pixels.  A glacier's diurnal response is spatially
    # coherent and single-look noise is not, so the mean of the per-pixel
    # phasors a + ib over the ice population has a far higher SNR; the same
    # mean over the held-out bedrock is its null (what a spatially smooth
    # residual, atmosphere or reference drift, does to a population).
    print("\npopulation mean phasor (amplitude, peak UTC):")
    print(f"{'fit':6s} {'ice':>22s} {'held-out bedrock':>22s}  ice/bedrock")
    for k in fits:
        zi, zr = np.nanmean(z[k][ice]), np.nanmean(z[k][held_m])
        print(f"{k:6s} {abs(zi):8.2f} mm {peak_of(zi, origin):6.1f} h "
              f"{abs(zr):8.2f} mm {peak_of(zr, origin):6.1f} h "
              f"{abs(zi) / abs(zr):10.1f}")
    zi1, zi2 = np.nanmean(z["day 1"][ice]), np.nanmean(z["day 2"][ice])
    print(f"  ice mean phasor day 2 / day 1: amplitude {abs(zi2) / abs(zi1):.2f}, "
          f"peak shift {circ_diff_hours(peak_of(zi1, origin), peak_of(zi2, origin)):+.1f} h")
    for name, m in (("ice", ice), ("bedrock", held_m)):
        c = phasor_corr(z["day 1"][m], z["day 2"][m])
        print(f"  {name:8s} day 1 x day 2 phasor correlation |c| = {abs(c):.3f} "
              f"(noise floor ~{1 / np.sqrt(m.sum()):.3f}), "
              f"systematic shift {np.angle(c) / (2 * np.pi) * 24:+.1f} h")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    vmax = np.nanpercentile(np.where(usable, amp["both"], np.nan), 98)
    for ax, k in zip(axes[0], fits):
        im = ax.imshow(np.where(usable, amp[k], np.nan), cmap="magma", vmin=0,
                       vmax=vmax, aspect="auto")
        ax.set_title(f"{k}: diurnal amplitude, all coherent ground", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Amplitude (mm)")
    # Peak hour of the block-averaged phasor: per pixel the SNR is ~1 and
    # the map would be empty, so average over `size` px blocks and show it
    # where the ice amplitude beats what the same averaging leaves on the
    # held-out bedrock (its 95th percentile -- the block-level false alarm).
    size = args.block
    for ax, k in zip(axes[1][:2], ("day 1", "day 2")):
        zi = smooth_phasor(z[k], ice, size)
        zr = smooth_phasor(z[k], held_m, size)
        null = np.nanpercentile(np.abs(zr[held_m]), 95)
        show = ice & (np.abs(zi) > null)
        im = ax.imshow(np.where(show, peak_of(zi, origin), np.nan), cmap="twilight",
                       vmin=0, vmax=24, aspect="auto")
        ax.set_title(f"{k}: peak hour on ice, {size}x{size} px blocks above "
                     f"the bedrock null ({null:.1f} mm)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Peak time (UTC hr)")
        print(f"  {k}: {size}x{size} block amplitude above the bedrock null "
              f"({null:.2f} mm) on {100 * show.sum() / ice.sum():.0f}% of ice")
    for ax in axes.ravel()[:5]:
        ax.set_xlabel("Range (samples)"); ax.set_ylabel("Azimuth (lines)")
    ax = axes[1][2]
    hours = np.linspace(0, 24, 241)
    for k, style in (("day 1", "-"), ("day 2", "-"), ("both", ":")):
        zi = np.nanmean(z[k][ice])
        ax.plot(hours, np.real(zi * np.exp(-1j * 2 * np.pi * (hours - origin) / 24)),
                style, lw=1.6, label=f"ice {k}: {abs(zi):.1f} mm at {peak_of(zi, origin):.1f} h")
    for k in ("day 1", "day 2"):
        zr = np.nanmean(z[k][held_m])
        ax.plot(hours, np.real(zr * np.exp(-1j * 2 * np.pi * (hours - origin) / 24)),
                "--", lw=1.0, color="0.5",
                label=f"bedrock {k}: {abs(zr):.1f} mm" if k == "day 1" else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("UTC (hr)"); ax.set_ylabel("Mean diurnal (mm)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("population mean diurnal term, ice vs held-out bedrock",
                 fontsize=10)
    plt.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"18_repeat_{day}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
