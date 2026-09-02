#!/usr/bin/env python3
"""Atmospheric-correction ladder, validated on held-out bedrock.

    python examples/baker_aps.py --scene <dir> --decimate 4

Runs one scene through four correction stages and scores each the only honest
way available without independent truth: the RMS of the displacement time
series on **held-out** stable ground.  Bedrock is not moving, so whatever
remains there is error.  The stable mask is split in half — one half feeds the
corrections, the other is never touched by them and does the scoring — because
correcting and scoring on the same pixels is circular.

The ladder:

    A  reference only            per-epoch constant tied to bedrock
    B  + per-pair screens        matched-filter ramp + robust linear fit,
                                 per interferogram (the original pipeline)
    C  + drift removal           gpri.aps.epoch_screen_correction: the screen
                                 model refitted per epoch on the integrated
                                 displacement over bedrock, killing the
                                 random-walk drift that B integrates
    D  + turbulence              gpri.aps.turbulence_screen on each epoch's
                                 residual displacement: the non-parametric,
                                 spatially smooth part no polynomial catches

Every stage includes the ones above it.  A is deliberately in the table: the
gap between A and B is what per-pair screens buy, and on a bad day it is less
than you'd hope.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpri import atmosphere
from gpri.aps import (displacement_ramp_to_delta_n, epoch_screen_correction,
                      turbulence_screen)
from gpri.stack import DiffStack
from gpri.timeseries import invert_network, los_displacement

# examples/site.py -- loaded by explicit path so the stdlib `site` module
# cannot shadow it. All machine-specific paths live in site.env (gitignored).
import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "gpri_site", Path(__file__).resolve().parent / "site.py")
_site = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_site)
_site.load_site()
SCENES = {d: _os.environ.get(f"GPRI_SCENE_{d}", "")
          for d in ("20170827", "20170803", "20170713", "20170713_full",
                    "20160826", "20170913", "20180709")}


def open_stack(scene: Path, antenna: str = "upper", lags=(1,), looks=(1, 1)):
    """The pair stack for one antenna of a scene.

    GAMMA only ever formed the upper-antenna daisy chain (``diff0``), so that
    exact product is used when it is what was asked for.  Anything else — the
    lower antenna, or a network with i->i+2 / i->i+3 pairs, or multilooked
    products — is formed from the SLCs with :class:`gpri.stack.SlcPairStack`,
    which reproduces GAMMA's ``.diff`` to 1e-7 rad and its ``.cc`` to 0.014.
    If ``gpri coregister --write`` left an ``azimuth_offsets.json`` for the
    scene, the SLCs are shifted onto its reference grid as they are read.
    """
    from gpri.stack import SlcPairStack
    letter = antenna[0].lower()
    if letter not in "ul":
        raise ValueError(f"antenna must be upper or lower, not {antenna!r}")
    tab = scene / "SLCu_tab"
    if letter == "u" and tuple(lags) == (1,) and tuple(looks) == (1, 1) \
            and (scene / "diff0").is_dir():
        return DiffStack.from_directory(scene / "diff0",
                                        slc_tab=tab if tab.exists() else None)
    if letter == "u" and tab.exists():
        st = SlcPairStack.from_tab(tab, lags=lags, looks=looks)
    else:
        st = SlcPairStack.from_directory(scene / "slc", antenna=letter,
                                         lags=lags, looks=looks)
    # a tripod that turned (20180709: 5 deg in five hours) is read on one grid
    from gpri.coregister import scene_azimuth_offsets
    return st.apply_azimuth_offsets(scene_azimuth_offsets(scene))


def _cache_path(scene: Path, antenna: str, lags, looks, dec: int):
    root = Path(_os.environ.get("GPRI_WORK_ROOT", "work"))
    tag = (f"pairs_{antenna[0].lower()}_lag{''.join(map(str, lags))}"
           f"_lk{looks[0]}x{looks[1]}_dec{dec}.npz")
    return root / scene.name / tag


def load(scene: Path, dec: int, n_max: int = 0, antenna: str = "upper",
         lags=(1,), looks=(1, 1), cache: bool = True):
    """Phase and coherence for every pair, range-decimated by ``dec``.

    Reading a full day is 50 GB of interferograms off a network mount (or 50
    GB of SLCs plus a coherence estimate per pair); the decimated arrays are
    cached under ``GPRI_WORK_ROOT/<day>/`` so that the second script to ask
    for the same product gets it in seconds.
    """
    stack = open_stack(scene, antenna, lags, looks)
    n = stack.n_pairs if n_max <= 0 else min(n_max, stack.n_pairs)
    net = stack.network
    net.pairs = net.pairs[:n]

    nc = stack.shape[1] // dec
    par = stack.par
    r = par.slant_range()[::dec][:nc]
    step = par.float("GPRI_az_angle_step", 0.0)
    az = (par.float("GPRI_az_start_angle", 0.0)
          + step * np.arange(par.azimuth_lines)) if step else None

    # the azimuth grid the pairs were formed on is part of the cache's
    # identity: a sidecar written since (or remeasured) means reading again
    shifts = getattr(stack, "azimuth_shifts", None)    # GAMMA's diff0 has none
    shifts = (np.zeros(stack.n_epochs) if shifts is None
              else np.asarray(shifts, float))
    cpath = _cache_path(scene, antenna, lags, looks, dec)
    if cache and cpath.exists():
        with np.load(cpath) as z:
            was = z["azimuth_shifts"] if "azimuth_shifts" in z.files else np.zeros(stack.n_epochs)
            same_grid = was.shape == shifts.shape and np.allclose(was, shifts, atol=0.01)
            if (int(z["n_pairs"]) >= n and tuple(z["shape"]) == (stack.shape[0], nc)
                    and same_grid):
                phase, cc = z["phase"][:n], z["cc"][:n]
                print(f"loaded {n} pairs from {cpath}")
                return stack, net, phase, cc, r, az, n
            if not same_grid:
                by = (f"by up to {np.abs(was - shifts).max():.2f} lines"
                      if was.shape == shifts.shape else "in number")
                print(f"{cpath} was formed on another azimuth grid (offsets "
                      f"changed {by}): reading the pairs again")

    phase = np.empty((n, stack.shape[0], nc), np.float32)
    cc = np.empty(phase.shape, np.float32)
    t0 = time.time()
    for p in range(n):
        ifg = stack.read_pair(p)[:, ::dec][:, :nc]
        phase[p] = np.angle(ifg)
        c = stack.read_coherence(p)
        cc[p] = np.abs(ifg) if c is None else c[:, ::dec][:, :nc]
        if p % 100 == 0:
            print(f"  read {p + 1}/{n}  ({time.time() - t0:.0f} s)")
    print(f"read {n} pairs in {time.time() - t0:.0f} s")
    if hasattr(stack, "close"):
        stack.close()

    if cache and n == stack.n_pairs:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cpath, phase=phase, cc=cc, n_pairs=n,
                 shape=np.array(phase.shape[1:]), pairs=net.pairs,
                 times=net.times, azimuth_shifts=shifts)
        print(f"cached to {cpath}")
    return stack, net, phase, cc, r, az, n


def integrate(pair_disp, net, n):
    """Daisy chain -> cumulative sum; anything else -> network inversion."""
    chain = all(j == i + 1 for i, j in net.pairs[:n])
    if chain:
        d = np.concatenate([np.zeros((1,) + pair_disp.shape[1:], np.float32),
                            np.cumsum(pair_disp, axis=0, dtype=np.float32)])
        return d, net.times[:n + 1]
    ts = invert_network(pair_disp, net, method="lstsq")
    return ts.displacement.astype(np.float32), net.times


def split_mask(stable, seed=0):
    idx = np.flatnonzero(stable.ravel())
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    fit = np.zeros(stable.size, bool)
    fit[idx[: idx.size // 2]] = True
    held = np.zeros(stable.size, bool)
    held[idx[idx.size // 2:]] = True
    return fit.reshape(stable.shape), held.reshape(stable.shape)


def score(disp, held):
    """RMS over held-out bedrock, per the whole series, in mm."""
    v = disp[:, held]
    return float(np.sqrt(np.nanmean(v ** 2)) * 1000.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="20170713",
                    help="a key of SCENES, or a scene directory path")
    ap.add_argument("--decimate", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--screen-coherence", type=float, default=0.4)
    ap.add_argument("--stable-coherence", type=float, default=0.85)
    ap.add_argument("--rgi", action="store_true",
                    help="drop reference pixels that fall on RGI glacier "
                         "outlines (+100 m); needs GPRI_RGI in site.env")
    ap.add_argument("--screens-on-bedrock", action="store_true",
                    help="fit the per-pair screens on the bedrock fit-mask "
                         "only, instead of everything above --screen-coherence")
    ap.add_argument("--sigma", type=float, nargs=2, default=(5.0, 40.0),
                    metavar=("AZ", "RG"), help="turbulence kernel, pixels")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"),
                    help="which receive antenna; 'lower' is formed from the "
                         "SLCs (GAMMA only processed the upper)")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    scene = Path(SCENES.get(args.scene) or args.scene)
    if not scene.is_dir():
        raise SystemExit(f"scene directory {scene} not found -- set "
                         f"GPRI_SCENE_{args.scene} in site.env or pass a path")
    day = scene.name + ("" if args.antenna == "upper" else f"_{args.antenna}")

    stack, net, phase, cc, r, az, n = load(scene, args.decimate, args.pairs,
                                           antenna=args.antenna)
    lam = stack.wavelength
    span = float((net.times[min(n, net.n_epochs - 1)] - net.times[0]) * 24)
    print(f"{day}: {n} pairs over {span:.2f} h, grid {phase.shape[1:]}, "
          f"wavelength {lam * 100:.3f} cm")

    mean_cc = cc.mean(axis=0)
    stable = mean_cc >= args.stable_coherence
    if args.rgi:
        from baker_north_side import decimated_par
        from gpri.geocode import BAKERBEND1_HEADING, RadarGeometry
        from gpri.heading import scene_heading
        from gpri.glaciers import load_outlines, stable_ground_mask
        geom = RadarGeometry(decimated_par(stack.par, args.decimate),
                             heading=scene_heading(scene, default=BAKERBEND1_HEADING))
        la, lo = geom.geodetic(rows=[0, geom.shape[0] - 1],
                               cols=[0, geom.shape[1] - 1])
        bbox = (lo.min() - .02, la.min() - .02, lo.max() + .02, la.max() + .02)
        gdf = load_outlines(_os.environ.get("GPRI_RGI", "data/rgi/rgi_61.zip"),
                            bbox=bbox)
        stable, contested = stable_ground_mask(mean_cc, geom, gdf,
                                               threshold=args.stable_coherence)
        print(f"RGI audit: dropped {contested.sum():,} coherent-but-glacier "
              f"px ({100 * contested.sum() / max(contested.sum() + stable.sum(), 1):.0f}% "
              f"of the coherence-only mask)")
    fit_m, held_m = split_mask(stable)
    print(f"stable ground {100 * stable.mean():.2f}% of swath -> "
          f"{fit_m.sum():,} fit px, {held_m.sum():,} held out")

    results, series = {}, {}

    # ---- stage A: reference only --------------------------------------
    d0, times = integrate(los_displacement(phase, lam), net, n)
    dA, _ = epoch_screen_correction(d0, fit_m, r, model="constant",
                                    weights=mean_cc)
    results["A reference only"] = score(dA, held_m)
    series["A"] = dA

    # ---- stage B: + per-pair screens ----------------------------------
    t0 = time.time()
    corrected = np.empty_like(phase)
    screens = []
    for p in range(n):
        if args.screens_on_bedrock:
            w = np.where(fit_m, cc[p], 0.0)
        else:
            w = np.where(atmosphere.stable_mask(cc[p], args.screen_coherence),
                         cc[p], 0.0)
        try:
            s = atmosphere.fit_screen(phase[p], slant_range=r, azimuth=az,
                                      weights=w, model="linear",
                                      wavelength=lam)
            corrected[p] = atmosphere.remove_screen(phase[p], s)
        except Exception:
            s, corrected[p] = None, phase[p]
        screens.append(s)
    print(f"per-pair screens: {sum(s is not None for s in screens)}/{n} "
          f"fitted in {time.time() - t0:.0f} s")

    dB0, _ = integrate(los_displacement(corrected, lam), net, n)
    dB, _ = epoch_screen_correction(dB0, fit_m, r, model="constant",
                                    weights=mean_cc)
    results["B + pair screens"] = score(dB, held_m)
    series["B"] = dB

    # ---- stage C: + drift removal -------------------------------------
    dC, coeffs = epoch_screen_correction(dB0, fit_m, r, model="linear",
                                         weights=mean_cc)
    results["C + drift removal"] = score(dC, held_m)
    series["C"] = dC
    drift_dn = displacement_ramp_to_delta_n(coeffs[:, 1])
    print(f"accumulated ramp drift removed: {np.ptp(drift_dn):.3f} N-units "
          f"peak to peak across the series")

    # ---- stage D: + per-epoch turbulence ------------------------------
    t0 = time.time()
    dD = np.empty_like(dC)
    covered = None
    for k in range(dC.shape[0]):
        scr, q = turbulence_screen(dC[k], fit_m, sigma=tuple(args.sigma),
                                   weights=mean_cc, wrapped=False)
        dD[k] = dC[k] - scr
        covered = (q > 0) if covered is None else covered
    print(f"turbulence screens for {dC.shape[0]} epochs in "
          f"{time.time() - t0:.0f} s; supported on "
          f"{100 * covered.mean():.1f}% of the grid")
    results["D + turbulence"] = score(dD, held_m)
    series["D"] = dD

    # ------------------------------------------------------------ report
    print("\n" + "=" * 58)
    print(f"held-out bedrock RMS, {day} ({span:.1f} h, {n} pairs)")
    print("=" * 58)
    base = results["A reference only"]
    for k, v in results.items():
        print(f"  {k:24s} {v:8.2f} mm   ({v / base * 100:5.1f}% of A)")
    print("=" * 58)

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    hours = (times[: series['A'].shape[0]] - times[0]) * 24
    for k in "ABCD":
        axes[0].plot(hours, np.nanmean(series[k][:, held_m], axis=1) * 1000,
                     lw=1.0, label=f"stage {k}: {results[[x for x in results if x.startswith(k)][0]]:.1f} mm rms")
    axes[0].axhline(0, color="0.6", lw=0.8, zorder=0)
    axes[0].set_xlabel("Elapsed time (hr)")
    axes[0].set_ylabel("Mean LOS (mm)")
    axes[0].set_title(f"{day}: what each correction stage leaves on held-out bedrock",
                      fontsize=10)
    axes[0].legend(fontsize=8)

    rms_t = {k: np.sqrt(np.nanmean(series[k][:, held_m] ** 2, axis=1)) * 1000
             for k in "ABCD"}
    for k in "ABCD":
        axes[1].plot(hours, rms_t[k], lw=1.0, label=f"stage {k}")
    axes[1].set_xlabel("Elapsed time (hr)")
    axes[1].set_ylabel("Rock RMS (mm)")
    axes[1].set_title("error growth through the series, held-out bedrock", fontsize=10)
    axes[1].legend(fontsize=8)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"12_aps_{day}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
