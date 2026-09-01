"""Command line interface: ``gpri <command> <scene>``.

A GPRI "scene" here is a GAMMA processing directory as it comes off the
instrument — an ``SLC_tab``, a ``diff0/`` full of interferograms, and the
``.par`` files beside them.  Every command takes one and writes NumPy ``.npz``
plus, optionally, GAMMA-format ``FLOAT`` rasters you can hand straight back to
``rasdt_pwr`` or ``disras``.

    gpri info       scene                 what is in it, and how coherent
    gpri screens    scene                 per-pair refractivity screens
    gpri velocity   scene -o vel.npz      coherence-weighted stacked rate
    gpri timeseries scene -o ts.npz       network-inverted LOS displacement
    gpri phaselink  scene -o pl.npz       EVD/eigenSAR/EMI/ML on a window
    gpri closure    scene                 closure-phase bias against baseline
    gpri unwrap     scene -o unw.npz      PS-interpolation unwrapping of a pair
    gpri geocode    scene vel.npz -o .tif reproject a product to a map frame

``velocity`` and ``timeseries`` also take ``--geotiff``, which geocodes the
result to a local stereographic GeoTIFF in one step.  That needs a scan
heading: ``GPRI_scan_heading`` is 0.0 in the BakerBend1 files, so pass
``--heading`` or the map will be rotated by an unknown amount.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .atmosphere import fit_screen, remove_screen, stable_mask
from .covariance import coherence_from_slcs
from .gamma import ParFile, read_slc, write_image
from .network import Network, read_slc_tab
from .phaselink import phase_link, temporal_coherence
from .stack import DiffStack, SlcPairStack
from .timeseries import invert_network, los_displacement, stack_velocity


# ------------------------------------------------------------------- plumbing
def _open(args):
    scene = Path(args.scene)
    antenna = getattr(args, "antenna", "upper")
    lags = tuple(getattr(args, "lags", None) or ())
    looks = tuple(getattr(args, "looks_pairs", None) or (1, 1))
    tab = scene / args.slc_tab
    if antenna != "upper" or lags or looks != (1, 1):
        # Pairs formed from the SLCs: the lower antenna, a lag network with
        # closed triangles, or multilooked products -- none of which GAMMA
        # wrote to diff0.  See gpri.stack.SlcPairStack.
        lags = lags or (1,)
        letter = antenna[0].lower()
        if letter == "u" and tab.exists():
            st = SlcPairStack.from_tab(tab, lags=lags, looks=looks)
        else:
            slc_dir = scene / getattr(args, "slc_dir", "slc")
            if not slc_dir.is_dir():
                raise SystemExit(f"no SLC directory at {slc_dir}")
            st = SlcPairStack.from_directory(slc_dir, antenna=letter,
                                             lags=lags, looks=looks)
        if args.max_pairs and st.n_pairs > args.max_pairs:
            keep = np.arange(args.max_pairs)
            st.network = Network(st.network.epochs, st.network.pairs[keep])
            st._pairs = [st._pairs[k] for k in keep]
        return st
    diff = scene / args.diff_dir
    if not diff.is_dir():
        raise SystemExit(f"no interferogram directory at {diff}")
    st = DiffStack.from_directory(diff, slc_tab=tab if tab.exists() else None,
                                  suffix=args.suffix)
    if args.max_pairs and st.n_pairs > args.max_pairs:
        # A *contiguous* run, not an evenly spaced sample.  Taking every k-th
        # pair out of a daisy chain shatters it into isolated two-epoch
        # fragments, and the network inversion then has nothing to connect.
        st = _subset(st, np.arange(args.max_pairs))
    if not st.network.is_connected():
        n = len(st.network.components())
        print(f"warning: network has {n} disconnected components; displacement "
              f"is only resolved within each one", file=sys.stderr)
    return st


def _subset(st, keep):
    """Restrict a stack to a subset of its pairs, renumbering the network."""
    keep = np.asarray(keep, int)
    pairs = st.network.pairs[keep]
    used = sorted({int(e) for pr in pairs for e in pr})
    remap = {e: k for k, e in enumerate(used)}
    net = Network([st.network.epochs[e] for e in used],
                  [(remap[int(i)], remap[int(j)]) for i, j in pairs])
    return DiffStack([st.paths[k] for k in keep], st.par, network=net,
                     cc_paths=None if st.cc_paths is None else [st.cc_paths[k] for k in keep])


def _rows(args, st):
    if args.rows is None:
        return slice(0, st.shape[0])
    a, _, b = args.rows.partition(":")
    return slice(int(a or 0), int(b) if b else int(a) + 1)


def _cols(args, st):
    if args.cols is None:
        return slice(0, st.shape[1])
    a, _, b = args.cols.partition(":")
    return slice(int(a or 0), int(b) if b else int(a) + 1)


def _save(path, arrays, par=None, rasters=()):
    path = Path(path)
    np.savez_compressed(path, **arrays)
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    for name in rasters:
        if name not in arrays:
            continue
        r = path.with_suffix("") .with_name(path.stem + f".{name}")
        write_image(r, np.asarray(arrays[name], np.float32), "FLOAT")
        print(f"wrote {r}  (GAMMA FLOAT {arrays[name].shape})")


# -------------------------------------------------------------------- commands
def cmd_info(args):
    st = _open(args)
    par = st.par
    r, az = st.slant_range(), st.azimuth_angles()
    print(f"scene            {args.scene}")
    print(f"sensor           {par.str('sensor')}   {par.date}")
    print(f"raster           {st.shape[0]} azimuth x {st.shape[1]} range "
          f"({st.shape[0] * st.shape[1] * 8 / 1e6:.0f} MB per FCOMPLEX pair)")
    print(f"wavelength       {st.wavelength * 100:.4f} cm  "
          f"(one fringe = {st.wavelength / 2 * 1000:.3f} mm of range)")
    print(f"slant range      {r[0]:.1f} -> {r[-1]:.1f} m  ({r[1] - r[0]:.4f} m/sample)")
    print(f"azimuth sweep    {az[0]:.3f} -> {az[-1]:.3f} deg")
    print(f"network          {st.network}")
    print(f"stack on disk    {st.n_pairs * st.shape[0] * st.shape[1] * 8 / 2 ** 30:.1f} GiB")

    dt = st.network.temporal_baselines() * 86400
    print(f"temporal baseline {dt.min():.0f} - {dt.max():.0f} s "
          f"(median {np.median(dt):.0f} s)")

    n = min(args.sample, st.n_pairs)
    idx = np.linspace(0, st.n_pairs - 1, n).astype(int)
    print(f"\ncoherence over {n} sampled pairs:")
    for p in idx:
        cc = st.read_coherence(p)
        i, j = st.network.pairs[p]
        cc = np.abs(st.read_pair(p)) if cc is None else cc
        print(f"  pair {p:4d}  {i:4d}->{j:4d}  "
              f"median {np.median(cc):.3f}  >0.6 {100 * (cc > 0.6).mean():5.1f}%")
    return 0


def cmd_screens(args):
    st = _open(args)
    r, lam = st.slant_range(), st.wavelength
    n = min(args.sample, st.n_pairs)
    idx = np.linspace(0, st.n_pairs - 1, n).astype(int)
    print(f"refractivity screens, model={args.model}, coherence>={args.threshold}")
    print(f"{'pair':>6} {'dt/s':>7} {'ramp rad/m':>12} {'dN':>9} "
          f"{'fringes':>8} {'quality':>8}")
    out = {}
    for p in idx:
        ifg = st.read_pair(p)
        cc = st.read_coherence(p)
        cc = np.abs(ifg) if cc is None else cc
        scr = fit_screen(ifg, par=st.par, weights=cc,
                         mask=stable_mask(cc, args.threshold), model=args.model)
        i, j = st.network.pairs[p]
        dt = (st.network.times[j] - st.network.times[i]) * 86400
        fr = abs(scr.ramp) * (r[-1] - r[0]) / (2 * np.pi)
        print(f"{p:6d} {dt:7.0f} {scr.ramp:+12.4e} {scr.delta_n * 1e6:+9.3f} "
              f"{fr:8.2f} {scr.quality:8.3f}")
        out[f"pair_{p}"] = np.array([scr.ramp, scr.delta_n, scr.quality, dt])
    if args.output:
        _save(args.output, out)
    return 0


def _pair_displacement(st, args, rows, cols, report=True):
    """Every pair over a window: atmosphere removed, converted to LOS metres."""
    nr = len(range(*rows.indices(st.shape[0])))
    nc = len(range(*cols.indices(st.shape[1])))
    disp = np.empty((st.n_pairs, nr, nc), np.float32)
    wgt = np.empty((st.n_pairs, nr, nc), np.float32)
    r_full = st.slant_range()
    t0 = time.time()
    for p in range(st.n_pairs):
        ifg = st.read_pair(p, rows, cols)
        cc = st.read_coherence(p, rows, cols)
        cc = np.abs(ifg) if cc is None else cc
        phase = np.angle(ifg)
        if not args.no_atmosphere:
            scr = fit_screen(ifg, slant_range=r_full[cols],
                             azimuth=st.azimuth_angles()[rows],
                             wavelength=st.wavelength, weights=cc,
                             mask=stable_mask(cc, args.threshold),
                             model=args.model)
            phase = remove_screen(phase, scr)
        disp[p] = los_displacement(phase, st.wavelength)
        wgt[p] = cc
        if report and (p + 1) % max(1, st.n_pairs // 10) == 0:
            print(f"  {p + 1}/{st.n_pairs} pairs  ({time.time() - t0:.0f}s)",
                  file=sys.stderr)
    return disp, wgt


def cmd_velocity(args):
    st = _open(args)
    rows, cols = _rows(args, st), _cols(args, st)
    print(f"stacking {st.n_pairs} pairs over rows {rows.start}:{rows.stop}, "
          f"cols {cols.start}:{cols.stop}")
    disp, wgt = _pair_displacement(st, args, rows, cols)
    w = np.where(wgt >= args.threshold, wgt, 0.0)
    v = stack_velocity(disp, st.network, weights=w, min_pairs=args.min_pairs)
    good = np.isfinite(v)
    print(f"velocity: {100 * good.mean():.1f}% of pixels solved")
    if good.any():
        print(f"  median {np.nanmedian(v) * 1000:+.4f} mm/day, "
              f"5-95% [{np.nanpercentile(v[good], 5) * 1000:+.4f}, "
              f"{np.nanpercentile(v[good], 95) * 1000:+.4f}]")
    if args.output:
        _save(args.output,
              {"velocity": v, "n_pairs": np.sum(w > 0, axis=0),
               "rows": np.array([rows.start, rows.stop]),
               "cols": np.array([cols.start, cols.stop]),
               "wavelength": np.array(st.wavelength)},
              rasters=("velocity",) if args.rasters else ())
        if getattr(args, "geotiff", False):
            _geocode_and_write(_place(v, st, rows, cols), st, args,
                               Path(args.output).with_suffix(""))
    return 0


def cmd_timeseries(args):
    st = _open(args)
    rows, cols = _rows(args, st), _cols(args, st)
    print(f"inverting {st.n_pairs} pairs -> {st.n_epochs} epochs, "
          f"rows {rows.start}:{rows.stop}, cols {cols.start}:{cols.stop}")
    disp, wgt = _pair_displacement(st, args, rows, cols)
    w = np.median(np.where(wgt >= args.threshold, wgt, 0.0), axis=(1, 2))
    ts = invert_network(disp, st.network, weights=w, method=args.method,
                        reference=args.reference, smoothing=args.smoothing,
                        wavelength=st.wavelength, incremental=args.incremental)
    print(f"{ts!r}")
    rms = ts.rms_residual()
    if rms is not None:
        print(f"  residual rms: median {np.nanmedian(rms) * 1000:.4f} mm")
    v = ts.velocity()
    print(f"  rate: median {np.nanmedian(v) * 1000:+.4f} mm/day")
    if args.output:
        _save(args.output,
              {"displacement": ts.displacement.astype(np.float32),
               "times": ts.times, "velocity": v.astype(np.float32),
               "pairs": st.network.pairs,
               "rows": np.array([rows.start, rows.stop]),
               "cols": np.array([cols.start, cols.stop]),
               "wavelength": np.array(st.wavelength)},
              rasters=("velocity",) if args.rasters else ())
        if getattr(args, "geotiff", False):
            _geocode_and_write(_place(v, st, rows, cols), st, args,
                               Path(args.output).with_suffix(""))
    return 0


def _place(a, st, rows, cols):
    """Put a windowed result back on the full radar grid, NaN elsewhere.

    Geocoding needs the full grid: the map transform is derived from the whole
    fan, so handing it a window would place the window's pixels at the fan's
    coordinates and silently put the answer in the wrong place.
    """
    if a.shape == st.shape:
        return a
    full = np.full(st.shape, np.nan, float)
    full[rows, cols] = a
    return full


def _geocode_and_write(array, st, args, path, kind="velocity"):
    """Reproject a radar-geometry product and write it as a GeoTIFF."""
    from .geocode import RadarGeometry, geocode, write_geotiff

    geom = RadarGeometry(st.par, heading=args.heading)
    out, transform = geocode(np.asarray(array, float), geom,
                             spacing=args.map_spacing)
    path = Path(path).with_suffix(".tif")
    write_geotiff(path, out, transform, geom.crs)
    filled = np.isfinite(out).mean()
    print(f"wrote {path}  ({out.shape[0]}x{out.shape[1]} at "
          f"{args.map_spacing:g} m, {100 * filled:.0f}% illuminated, "
          f"heading {geom.heading:.1f} deg)")
    return path


def cmd_closure(args):
    """Closure-phase bias against temporal baseline.

    Needs triangles.  A daisy-chain ``itab`` — which is what BakerBend1 ships —
    has none, and this says so rather than returning zeros.
    """
    from .closure import closure_rms, correct_bias, estimate_bias
    from .timeseries import triplets

    st = _open(args)
    rows, cols = _rows(args, st), _cols(args, st)
    trip = triplets(st.network)
    print(f"{st.n_pairs} pairs, {st.n_epochs} epochs, {len(trip)} closed triangles")
    if len(trip) == 0:
        raise SystemExit(
            "this network is a daisy chain and contains no closed triangles, so "
            "the closure bias is not estimable. Form the i->i+2 interferograms "
            "(GAMMA SLC_intf, or gpri.covariance from the SLCs) and try again.")

    phase = np.empty((st.n_pairs,
                      len(range(*rows.indices(st.shape[0]))),
                      len(range(*cols.indices(st.shape[1])))), np.float32)
    for p in range(st.n_pairs):
        phase[p] = np.angle(st.read_pair(p, rows, cols))

    before = closure_rms(phase, st.network, trip)
    model = estimate_bias(phase, st.network, trip=trip, robust=args.robust,
                          wavelength=st.wavelength)
    after = closure_rms(correct_bias(phase, model), st.network, trip)
    print(f"{model!r}")
    print(f"closure rms: {before:.4f} -> {after:.4f} rad")
    print("\nNOTE: a bias linear in temporal baseline closes perfectly and is "
          "invisible here.\n      That is exactly a constant velocity, so this "
          "does not validate any rate.")
    print(f"\n{'baseline/s':>11} {'bias/rad':>10} {'bias/mm LOS':>12}")
    b = model.bias
    b = b.reshape(b.shape[0], -1).mean(axis=1) if b.ndim > 1 else b
    for c, v in zip(model.centers, b):
        print(f"{c * 86400:11.0f} {v:+10.4f} "
              f"{-st.wavelength / (4 * np.pi) * v * 1000:+12.4f}")
    if args.output:
        _save(args.output, {"bias": model.bias, "centers": model.centers,
                            "index": model.index,
                            "closure_rms": np.array([before, after])})
    return 0


def cmd_unwrap(args):
    """PS-interpolation unwrapping of one interferogram (Chen et al. 2015)."""
    from .psinterp import unwrap_with_ps

    st = _open(args)
    rows, cols = _rows(args, st), _cols(args, st)
    p = args.pair
    if not 0 <= p < st.n_pairs:
        raise SystemExit(f"pair {p} out of range (0..{st.n_pairs - 1})")

    ifg = st.read_pair(p, rows, cols)
    cc = st.read_coherence(p, rows, cols)
    cc = np.abs(ifg) if cc is None else cc

    # physical coordinates, because a range bin is 0.75 m and an azimuth bin at
    # 10 km is 35 m -- a graph built in pixel units would connect the wrong pairs
    gr = st.par.slant_range()[cols] * np.cos(
        np.deg2rad(st.par.float("GPRI_ant_elev_angle", 0.0)))
    ang = np.deg2rad(st.azimuth_angles()[rows])
    Y = (gr[None, :] * np.sin(ang[:, None])).ravel()
    X = (gr[None, :] * np.cos(ang[:, None])).ravel()
    coords = np.column_stack([X, Y])

    res = unwrap_with_ps(ifg, coherence=cc, min_coherence=args.min_coherence,
                         max_dispersion=np.inf, max_count=args.max_ps,
                         coords=coords, method=args.interp)
    print(f"pair {p}: {res!r}")
    print(f"  PS density {100 * res.n_ps / cc.size:.2f}% of the window")
    print(f"  {100 * res.suspect_fraction:.2f}% of pixels have a residual near "
          f"pi -- the interpolation failed there")
    if res.n_unresolved:
        print(f"  {res.n_unresolved} PS could not be tied to the reference "
              f"(returned NaN, not guessed)")
    if args.output:
        _save(args.output,
              {"unwrapped": res.unwrapped.astype(np.float32),
               "interpolated": res.interpolated.astype(np.float32),
               "residual": res.residual.astype(np.float32),
               "ps_mask": res.mask, "coherence": cc.astype(np.float32)},
              rasters=("unwrapped",) if args.rasters else ())
        if args.geotiff:
            _geocode_and_write(res.unwrapped, st, args,
                               Path(args.output).with_name(
                                   Path(args.output).stem + "_unwrapped"))
    return 0


def cmd_geocode(args):
    """Reproject an array from a previous command's .npz into a map frame."""
    st = _open(args)
    data = np.load(args.npz)
    if args.field not in data:
        raise SystemExit(f"{args.npz} has no array {args.field!r}; "
                         f"it holds {sorted(data.files)}")
    a = data[args.field]
    if a.ndim == 3:
        print(f"{args.field} is a stack of {a.shape[0]}; geocoding band "
              f"{args.band}")
        a = a[args.band]
    if a.shape != st.shape:
        raise SystemExit(
            f"{args.field} is {a.shape} but the scene is {st.shape}; it was "
            f"probably written from a --rows/--cols window, which cannot be "
            f"geocoded without the same window")
    _geocode_and_write(a, st, args, args.output or Path(args.npz).with_suffix(""),
                       kind=args.field)
    return 0


def cmd_phaselink(args):
    """Phase linking needs SLCs, not interferograms: it wants the full matrix."""
    scene = Path(args.scene)
    tab = scene / args.slc_tab
    if not tab.exists():
        raise SystemExit(f"phaselink needs an SLC table; {tab} not found")
    imgs, pars = read_slc_tab(tab)
    par = ParFile.load(scene / pars[0])

    idx = np.arange(len(imgs))
    if args.max_epochs and len(idx) > args.max_epochs:
        idx = np.linspace(0, len(idx) - 1, args.max_epochs).astype(int)
    rows = _rows(args, type("S", (), {"shape": par.shape})())
    cols = _cols(args, type("S", (), {"shape": par.shape})())
    print(f"phase linking {len(idx)} epochs, method={args.method}, "
          f"rows {rows.start}:{rows.stop}, cols {cols.start}:{cols.stop}")

    from .gamma import map_image
    slcs = np.empty((len(idx),
                     len(range(*rows.indices(par.shape[0]))),
                     len(range(*cols.indices(par.shape[1])))), np.complex64)
    for k, e in enumerate(idx):
        slcs[k] = np.asarray(map_image(scene / imgs[e], par=par)[rows, cols])

    G = coherence_from_slcs(slcs, looks=(args.looks[0], args.looks[1]),
                            max_gib=args.max_gib)
    print(f"  coherence matrices: {G.shape}")
    theta = phase_link(G, method=args.method, reference=args.reference)
    tcoh = temporal_coherence(G, theta)
    epochs = [imgs[e] for e in idx]
    net = Network([__import__("gpri.network", fromlist=["parse_epoch"]).parse_epoch(e)
                   for e in epochs], [(i, i + 1) for i in range(len(idx) - 1)])
    from .timeseries import displacement_from_phases
    d = displacement_from_phases(np.moveaxis(theta, -1, 0), par.wavelength,
                                 reference=args.reference, axis=0)
    print(f"  temporal coherence: median {np.median(tcoh):.3f}, "
          f">0.6 {100 * (tcoh > 0.6).mean():.1f}%")
    print(f"  displacement range: {np.nanmin(d) * 1000:+.3f} to "
          f"{np.nanmax(d) * 1000:+.3f} mm")
    if args.output:
        _save(args.output,
              {"displacement": d.astype(np.float32),
               "temporal_coherence": tcoh.astype(np.float32),
               "phase": np.angle(theta).astype(np.float32),
               "times": net.times, "epochs": np.array(epochs),
               "wavelength": np.array(par.wavelength)},
              rasters=("temporal_coherence",) if args.rasters else ())
    return 0


# ---------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(
        prog="gpri", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"gpri {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, window=True):
        sp.add_argument("scene", help="GAMMA scene directory")
        sp.add_argument("--diff-dir", default="diff0")
        sp.add_argument("--slc-tab", default="SLCu_tab")
        sp.add_argument("--suffix", default=".diff",
                        help="'.diff' or '.adf.diff' (default: .diff)")
        sp.add_argument("--max-pairs", type=int, default=0,
                        help="use only the first N pairs (kept contiguous so "
                             "the network stays connected)")
        sp.add_argument("--antenna", default="upper", choices=("upper", "lower"),
                        help="'lower' forms that antenna's pairs from the SLCs "
                             "in --slc-dir (GAMMA only wrote the upper's)")
        sp.add_argument("--slc-dir", default="slc")
        sp.add_argument("--lags", type=int, nargs="+", default=None,
                        help="form i->i+lag pairs from the SLCs instead of "
                             "reading diff0, e.g. --lags 1 2 3 for closure")
        sp.add_argument("--looks-pairs", type=int, nargs=2, default=None,
                        metavar=("AZ", "RG"),
                        help="multilook SLC-formed pairs (default 1 1)")
        if window:
            sp.add_argument("--rows", help="azimuth window, e.g. 180:220")
            sp.add_argument("--cols", help="range window, e.g. 0:8000")
        sp.add_argument("-o", "--output", help="write results to this .npz")
        sp.add_argument("--rasters", action="store_true",
                        help="also write GAMMA FLOAT rasters beside the .npz")

    def mapping(sp, tif=True):
        sp.add_argument("--heading", type=float, default=None,
                        help="true bearing of azimuth zero, degrees. "
                             "GPRI_scan_heading is 0.0 in the BakerBend1 files "
                             "and is not a survey; ~105 fits the north side. "
                             "Solve it properly with "
                             "gpri.geocode.heading_from_tiepoint")
        sp.add_argument("--map-spacing", type=float, default=25.0,
                        help="output pixel size in metres (default 25)")
        if tif:
            sp.add_argument("--geotiff", action="store_true",
                            help="also write a geocoded GeoTIFF beside the .npz")

    def atmo(sp):
        sp.add_argument("--model", default="linear",
                        help="screen model: linear, quadratic, planar, bilinear, full")
        sp.add_argument("--threshold", type=float, default=0.6,
                        help="coherence floor for stable ground (default 0.6)")
        sp.add_argument("--no-atmosphere", action="store_true",
                        help="skip refractivity correction entirely")

    s = sub.add_parser("info", help="summarise a scene")
    common(s, window=False)
    s.add_argument("--sample", type=int, default=5)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("screens", help="per-pair refractivity screens")
    common(s, window=False)
    atmo(s)
    s.add_argument("--sample", type=int, default=10)
    s.set_defaults(func=cmd_screens)

    s = sub.add_parser("velocity", help="coherence-weighted stacked rate")
    common(s)
    atmo(s)
    mapping(s)
    s.add_argument("--min-pairs", type=int, default=2)
    s.set_defaults(func=cmd_velocity)

    s = sub.add_parser("closure", help="closure-phase bias against baseline")
    common(s)
    s.add_argument("--robust", type=int, default=0,
                   help="Huber reweighting sweeps against outlier triangles")
    s.set_defaults(func=cmd_closure)

    s = sub.add_parser("unwrap", help="PS-interpolation unwrapping of one pair")
    common(s)
    mapping(s)
    s.add_argument("--pair", type=int, default=0)
    s.add_argument("--min-coherence", type=float, default=0.6,
                   help="coherence floor defining a persistent scatterer")
    s.add_argument("--max-ps", type=int, default=20000)
    s.add_argument("--interp", default="linear",
                   choices=["linear", "idw", "nearest"])
    s.set_defaults(func=cmd_unwrap)

    s = sub.add_parser("geocode", help="reproject a product to a map frame")
    common(s, window=False)
    mapping(s, tif=False)
    s.add_argument("npz", help=".npz written by velocity/timeseries/unwrap")
    s.add_argument("--field", default="velocity", help="array to geocode")
    s.add_argument("--band", type=int, default=0,
                   help="which band, if the array is a stack")
    s.set_defaults(func=cmd_geocode)

    s = sub.add_parser("timeseries", help="network-inverted LOS displacement")
    common(s)
    atmo(s)
    mapping(s)
    s.add_argument("--method", default="lstsq",
                   choices=["lstsq", "wls", "l1", "smooth"])
    s.add_argument("--reference", type=int, default=0)
    s.add_argument("--smoothing", type=float, default=0.0)
    s.add_argument("--incremental", action="store_true")
    s.set_defaults(func=cmd_timeseries)

    s = sub.add_parser("phaselink", help="EVD/eigenSAR/EMI/ML phase linking")
    common(s)
    s.add_argument("--method", default="emi",
                   choices=["evd", "eigensar", "emi", "mle"])
    s.add_argument("--looks", type=int, nargs=2, default=(10, 10),
                   metavar=("AZ", "RG"))
    s.add_argument("--max-epochs", type=int, default=30)
    s.add_argument("--reference", type=int, default=0)
    s.add_argument("--max-gib", type=float, default=4.0)
    s.set_defaults(func=cmd_phaselink)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
