"""Figures: radar geometry, map-projected products, and diagnostics.

Two kinds of figure here, and the distinction is the point.

**Radar geometry** (:func:`radar_image`) shows the array as it is stored,
azimuth line against range sample.  Use it for diagnostics — it is the frame
the algorithms work in, and a stripe or a bad line is obvious there and
invisible once resampled.  Never use it to make a claim about where something
is on the ground: the fan is anisotropic by a factor of 60 across the swath.

**Map geometry** (:func:`map_image`, :func:`velocity_map`) shows the same array
after :func:`gpri_tools.geocode.geocode`, in a north-up projected frame with a scale
bar and a north arrow.  Everything spatial goes here.

Colour
------
Displacement and velocity get a diverging map centred on zero, because the sign
means something and a sequential map hides it.  Coherence gets a sequential
map, floored at zero.  Phase gets a cyclic map — ``twilight`` — because phase
wraps and any non-cyclic map puts a false discontinuity at the wrap.  These are
not preferences; using a non-cyclic map on wrapped phase is simply wrong.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "radar_image", "map_image", "velocity_map", "displacement_series",
    "network_plot", "closure_bias_plot", "refractivity_plot", "ps_map",
    "coverage_map", "scalebar", "north_arrow",
]

#: Sensible defaults per quantity: colormap, whether it is symmetric about zero.
STYLES = {
    "phase":        ("twilight", False, "phase (rad)"),
    "displacement": ("RdBu_r", True, "LOS displacement (mm)"),
    "velocity":     ("RdBu_r", True, "LOS velocity (m/yr)"),
    "coherence":    ("magma", False, "coherence"),
    "amplitude":    ("gray", False, "amplitude (dB)"),
    "refractivity": ("RdBu_r", True, "$\\Delta N$ (N-units)"),
}


def _mpl():
    import matplotlib.pyplot as plt
    return plt


def _limits(a, kind, vmin=None, vmax=None, percentile=2.0):
    """Robust colour limits: percentile-clipped, and symmetric where the sign matters."""
    if vmin is not None and vmax is not None:
        return vmin, vmax
    finite = np.asarray(a, float)[np.isfinite(a)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo = np.percentile(finite, percentile)
    hi = np.percentile(finite, 100.0 - percentile)
    symmetric = STYLES.get(kind, (None, False, None))[1]
    if kind == "phase":
        lo, hi = -np.pi, np.pi
    elif symmetric:
        m = max(abs(lo), abs(hi)) or 1.0
        lo, hi = -m, m
    return (lo if vmin is None else vmin, hi if vmax is None else vmax)


# --------------------------------------------------------------- radar frame
def radar_image(data, geom=None, kind="phase", ax=None, title=None, cmap=None,
                vmin=None, vmax=None, ground_range=True, colorbar=True):
    """Plot an array in radar geometry, with physical axes where possible.

    With a :class:`gpri_tools.geocode.RadarGeometry` the axes become ground range in
    kilometres and azimuth angle in degrees rather than pixel indices, which at
    least makes the distortion legible even though it does not remove it.
    """
    plt = _mpl()
    a = np.asarray(data)
    if np.iscomplexobj(a):
        a = np.angle(a)
        kind = "phase"
    a = np.asarray(a, float)

    cmap = cmap or STYLES.get(kind, ("viridis",))[0]
    lo, hi = _limits(a, kind, vmin, vmax)

    ax = ax or plt.subplots(figsize=(11, 3.2))[1]
    if geom is not None:
        r = (geom.ground_range() if ground_range else geom.slant_range()) / 1000.0
        az = geom.par.float("GPRI_az_start_angle", 0.0) + \
            geom.par.float("GPRI_az_angle_step", 0.0) * np.arange(a.shape[0])
        extent = [r[0], r[-1], az[-1], az[0]]
        ax.set_xlabel("Ground range (km)" if ground_range else "Slant range (km)")
        ax.set_ylabel("Azimuth angle (deg)")
    else:
        extent = None
        ax.set_xlabel("Range (samples)")
        ax.set_ylabel("Azimuth (lines)")

    im = ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, extent=extent,
                   aspect="auto", interpolation="nearest", origin="upper")
    if title:
        ax.set_title(title)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label(STYLES.get(kind, (None, None, ""))[2])
    return ax


# ----------------------------------------------------------------- map frame
def map_image(data, transform, kind="velocity", ax=None, title=None, cmap=None,
              vmin=None, vmax=None, colorbar=True, origin_xy=None,
              scale=True, north=True, background=None):
    """Plot a geocoded array in the map frame, north up.

    ``transform`` is the affine tuple :func:`gpri_tools.geocode.geocode` returns.
    ``origin_xy`` marks the radar position.  ``background`` is an optional
    array plotted underneath in grey — a geocoded MLI amplitude makes a good
    one, so the reader can see the terrain the measurements sit on.
    """
    plt = _mpl()
    a = np.asarray(data)
    if np.iscomplexobj(a):
        a, kind = np.angle(a), "phase"
    a = np.asarray(a, float)

    xmin, sx, _, ymax, _, sy = transform
    ny, nx = a.shape
    extent = [xmin / 1000.0, (xmin + sx * nx) / 1000.0,
              (ymax + sy * ny) / 1000.0, ymax / 1000.0]

    ax = ax or plt.subplots(figsize=(7.5, 6.5))[1]
    if background is not None:
        b = np.asarray(background, float)
        ax.imshow(b, cmap="gray", extent=extent, origin="upper",
                  vmin=np.nanpercentile(b, 2), vmax=np.nanpercentile(b, 98),
                  interpolation="bilinear")

    cmap = cmap or STYLES.get(kind, ("viridis",))[0]
    lo, hi = _limits(a, kind, vmin, vmax)
    im = ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, extent=extent,
                   origin="upper", interpolation="nearest")

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if origin_xy is not None:
        ax.plot(origin_xy[0] / 1000.0, origin_xy[1] / 1000.0, marker="^",
                ms=9, mfc="w", mec="k", mew=1.4, zorder=5, label="radar")
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(STYLES.get(kind, (None, None, ""))[2])
    if scale:
        scalebar(ax)
    if north:
        north_arrow(ax)
    return ax


def velocity_map(velocity, transform, wavelength=None, **kwargs):
    """LOS velocity in m/yr on a map, diverging about zero.

    ``velocity`` is metres per day out of :func:`gpri_tools.timeseries.stack_velocity`
    or :meth:`gpri_tools.timeseries.TimeSeries.velocity`; it is scaled to metres
    per year here, the unit glacier velocities are quoted in.
    """
    from .diurnal import m_per_yr
    v = m_per_yr(velocity)
    kwargs.setdefault("title", "LOS velocity (positive toward radar)")
    return map_image(v, transform, kind="velocity", **kwargs)


def coverage_map(geom, spacing=25.0, ax=None, crs=None, features=None):
    """The illuminated fan on the ground, before any data is involved.

    The first figure to make and the first to check: it shows where the radar
    can see, and whether the scan heading you supplied points it at the right
    mountain.  ``features`` is an optional ``{name: (lat, lon)}`` mapping —
    plot the summit and a couple of glaciers on it and the heading is either
    obviously right or obviously wrong.
    """
    plt = _mpl()
    from .geocode import map_grid

    ax = ax or plt.subplots(figsize=(7.5, 6.5))[1]
    x, y, transform = map_grid(geom, spacing=spacing, crs=crs)
    X, Y = np.meshgrid(x, y)
    row, col = geom.radar_coordinates(X, Y, crs=crs)
    na, nr = geom.shape
    inside = ((row >= 0) & (row <= na - 1) & (col >= 0) & (col <= nr - 1)).astype(float)

    xmin, sx, _, ymax, _, sy = transform
    extent = [xmin / 1000, (xmin + sx * X.shape[1]) / 1000,
              (ymax + sy * X.shape[0]) / 1000, ymax / 1000]
    ax.imshow(np.where(inside > 0, inside, np.nan), cmap="Blues", vmin=0, vmax=1.4,
              extent=extent, origin="upper", interpolation="nearest")

    x0, y0 = geom.origin_xy(crs=crs)
    ax.plot(x0 / 1000, y0 / 1000, "^", ms=10, mfc="w", mec="k", mew=1.5,
            zorder=5, label="radar")

    if features:
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:4326", crs or geom.crs, always_xy=True)
        for name, (lat, lon) in features.items():
            fx, fy = tf.transform(lon, lat)
            ax.plot(fx / 1000, fy / 1000, "o", ms=5, color="firebrick", zorder=6)
            ax.annotate(name, (fx / 1000, fy / 1000), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="firebrick")

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")
    ax.set_title(f"Illuminated fan, scan heading {geom.heading:.1f}$\\degree$")
    ax.legend(loc="upper right", fontsize=8)
    scalebar(ax)
    north_arrow(ax)
    return ax


# ------------------------------------------------------------------ overlays
def scalebar(ax, length_km=None, loc=(0.06, 0.06), color="k"):
    """A scale bar in kilometres, sized to a round fraction of the axis."""
    x0, x1 = ax.get_xlim()
    span = abs(x1 - x0)
    if length_km is None:
        raw = span / 5.0
        mag = 10 ** np.floor(np.log10(max(raw, 1e-9)))
        length_km = float(min([1, 2, 5, 10], key=lambda m: abs(m * mag - raw)) * mag)
    y0, y1 = ax.get_ylim()
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    ax.plot([bx, bx + length_km], [by, by], color=color, lw=3,
            solid_capstyle="butt", zorder=6)
    ax.annotate(f"{length_km:g} km", ((bx + length_km / 2), by),
                textcoords="offset points", xytext=(0, 5), ha="center",
                fontsize=8, color=color, zorder=6)
    return ax


def north_arrow(ax, loc=(0.93, 0.10), size=0.08, color="k"):
    """A north arrow.  Only honest in a north-up projected frame."""
    ax.annotate("N", xy=loc, xytext=(loc[0], loc[1] + size),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="bottom", fontsize=9, color=color,
                arrowprops=dict(arrowstyle="<-", color=color, lw=1.4), zorder=6)
    return ax


# -------------------------------------------------------------- diagnostics
def displacement_series(ts, pixels=None, ax=None, unit="mm", label=None):
    """LOS displacement against time for one or more pixels.

    ``ts`` is a :class:`gpri_tools.timeseries.TimeSeries`.  ``pixels`` is a list of
    index tuples into the spatial axes; omit it for a spatially averaged series.
    """
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(7.5, 4))[1]
    scale = {"m": 1.0, "mm": 1000.0, "cm": 100.0}[unit]
    t = ts.times * 24.0                      # hours: a GPRI day is the timescale

    d = np.asarray(ts.displacement, float)
    if pixels is None:
        series = [(np.nanmean(d.reshape(d.shape[0], -1), axis=1), label or "scene mean")]
    else:
        series = [(d[(slice(None),) + tuple(p)], label or f"{tuple(p)}") for p in pixels]

    for y, lab in series:
        ax.plot(t, y * scale, marker="o", ms=2.5, lw=1.0, label=lab)
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel("Elapsed time (hr)")
    ax.set_ylabel(f"LOS displacement ({unit})")
    ax.set_title("positive toward the radar", fontsize=9)
    ax.legend(fontsize=8)
    return ax


def network_plot(network, ax=None, values=None, cmap="viridis", label=None):
    """The interferogram network: epochs on the time axis, pairs as arcs.

    Colour the arcs by anything per-pair — coherence, closure residual, the
    fitted bias — and a bad subset of the network shows up immediately.
    """
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(9, 3.6))[1]
    t = network.times * 24.0
    dt = network.temporal_baselines() * 24.0

    if values is not None:
        v = np.asarray(values, float)
        norm = plt.Normalize(np.nanpercentile(v, 2), np.nanpercentile(v, 98))
        colours = plt.get_cmap(cmap)(norm(v))
    else:
        colours = ["0.4"] * network.n_pairs

    for k, (i, j) in enumerate(network.pairs):
        ax.plot([t[i], t[j]], [dt[k], dt[k]], "-", color=colours[k], lw=1.0,
                alpha=0.85)
    ax.plot(t, np.zeros_like(t), "|", color="k", ms=8, mew=0.8)
    ax.set_xlabel("Elapsed time (hr)")
    ax.set_ylabel("Temporal baseline (hr)")
    ax.set_title(f"{network.n_epochs} epochs, {network.n_pairs} pairs, "
                 f"{'connected' if network.is_connected() else 'DISCONNECTED'}")
    if values is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = ax.figure.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label(label or "")
    return ax


def closure_bias_plot(model, ax=None, wavelength=None, unit="mm"):
    """The fitted closure bias against temporal baseline.

    Plots the bias curve, and states in the title that the correction is blind
    to a velocity — because a reader looking at this figure is exactly the
    reader about to over-claim from it.
    """
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(6.5, 4))[1]
    b = np.asarray(model.bias, float)
    if b.ndim > 1:
        b = np.nanmean(b.reshape(b.shape[0], -1), axis=1)
    x = model.centers * 24.0

    wl = wavelength if wavelength is not None else model.wavelength
    if wl is not None and unit != "rad":
        from .timeseries import los_displacement
        y = los_displacement(b, wl) * {"mm": 1000.0, "cm": 100.0, "m": 1.0}[unit]
        ylab = f"Closure bias ({unit})"
    else:
        y, ylab = b, "Closure bias (rad)"

    ax.plot(x, y, "o-", ms=4, lw=1.2, color="C3")
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel("Temporal baseline (hr)")
    ax.set_ylabel(ylab)
    ax.set_title(f"Closure bias, {model.n_triplets} triplets\n"
                 "(blind to a bias linear in baseline, i.e. to velocity)",
                 fontsize=9)
    return ax


def refractivity_plot(N, times=None, ax=None, met=None, label="estimated"):
    """Per-epoch refractivity, optionally against met-derived values.

    ``N`` is N-units from :func:`gpri_tools.refractivity.invert_refractivity`; ``met``
    is an optional companion series computed from a weather station via
    :func:`gpri_tools.refractivity.refractivity`.  Plotting them together is the only
    independent check there is on an empirically estimated screen.
    """
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(7.5, 4))[1]
    n = np.asarray(N, float)
    if n.ndim > 1:
        n = np.nanmean(n.reshape(n.shape[0], -1), axis=1)
    t = np.arange(len(n)) if times is None else np.asarray(times, float) * 24.0

    ax.plot(t, n, "o-", ms=3, lw=1.0, label=label)
    if met is not None:
        m = np.asarray(met, float)
        ax.plot(t[:len(m)], m - np.nanmean(m) + np.nanmean(n), "s--", ms=3,
                lw=1.0, color="C1", label="from met data")
        ax.legend(fontsize=8)
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel("Elapsed time (hr)" if times is not None else "Epoch (index)")
    ax.set_ylabel("$\\Delta N$ (N-units)")
    return ax


def ps_map(mask, transform=None, geom=None, ax=None, background=None,
           title=None):
    """Where the persistent scatterers are.

    Worth looking at before trusting anything from :mod:`gpri_tools.psinterp`: if the
    PS all sit on one rock rib, the interpolation across the glacier is
    extrapolation and the time series there is a guess.
    """
    plt = _mpl()
    m = np.asarray(mask, bool)
    ax = ax or plt.subplots(figsize=(7.5, 6.5))[1]
    frac = 100.0 * m.sum() / m.size

    if transform is not None:
        xmin, sx, _, ymax, _, sy = transform
        extent = [xmin / 1000, (xmin + sx * m.shape[1]) / 1000,
                  (ymax + sy * m.shape[0]) / 1000, ymax / 1000]
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.set_aspect("equal")
    else:
        extent = None
        ax.set_xlabel("Range (samples)")
        ax.set_ylabel("Azimuth (lines)")

    if background is not None:
        b = np.asarray(background, float)
        ax.imshow(b, cmap="gray", extent=extent, origin="upper",
                  vmin=np.nanpercentile(b, 2), vmax=np.nanpercentile(b, 98),
                  aspect="auto" if transform is None else "equal")
    ax.imshow(np.where(m, 1.0, np.nan), cmap="autumn", extent=extent,
              origin="upper", vmin=0, vmax=1, interpolation="nearest",
              aspect="auto" if transform is None else "equal")
    ax.set_title(title or f"persistent scatterers: {m.sum():,} px ({frac:.2f}%)")
    return ax


def diurnal_summary(fit, times, slant_range=None, mask=None, origin_hour=0.0,
                    stable=None, axes=None):
    """Four-panel diurnal diagnostic: amplitude, phase, range test, stacked cycle.

    The bottom-left panel is the one that decides whether the signal is real.
    It plots diurnal amplitude against slant range: residual refractivity is
    linear in range, ice motion is not, so a rising trend there means the
    diurnal is atmospheric.  See :func:`gpri_tools.diurnal.range_dependence`.
    """
    plt = _mpl()
    from .diurnal import DIURNAL, range_dependence

    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = np.asarray(axes).ravel()

    amp = fit.amplitude(DIURNAL) * 1000.0
    peak = fit.peak_time(DIURNAL, origin_hour=origin_hour)
    m = np.ones(np.shape(amp), bool) if mask is None else np.asarray(mask, bool)

    a = np.where(m, amp, np.nan)
    im = axes[0].imshow(a, cmap="magma", aspect="auto", origin="upper",
                        vmin=0, vmax=np.nanpercentile(a, 98) if np.isfinite(a).any() else 1)
    axes[0].set_title("diurnal amplitude")
    axes[0].figure.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02,
                            label="mm LOS")

    # phase is cyclic in the hour of day, so it needs a cyclic colormap
    im = axes[1].imshow(np.where(m, peak, np.nan), cmap="twilight",
                        aspect="auto", origin="upper", vmin=0, vmax=24)
    axes[1].set_title("hour of diurnal peak")
    axes[1].figure.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02,
                            label="hour of day")
    for ax in axes[:2]:
        ax.set_xlabel("Range (samples)")
        ax.set_ylabel("Azimuth (lines)")

    ax = axes[2]
    if slant_range is not None:
        R = np.broadcast_to(np.asarray(slant_range, float), amp.shape)
        ok = m & np.isfinite(amp)
        if ok.sum() > 50:
            r_km = R[ok] / 1000.0
            ax.plot(r_km, amp[ok], ".", ms=1, alpha=0.15, color="0.4")
            res = range_dependence(amp / 1000.0, np.asarray(slant_range, float),
                                   mask=m)
            xs = np.linspace(r_km.min(), r_km.max(), 2)
            ax.plot(xs, 1000.0 * (res["intercept"] + res["slope"] * xs * 1000),
                    "-", color="C3", lw=2,
                    label=f"r = {res['correlation']:+.2f}")
            ax.legend(fontsize=8)
            ax.set_title("amplitude vs range — atmosphere is linear here",
                         fontsize=10)
    ax.set_xlabel("Slant range (km)")
    ax.set_ylabel("Diurnal amplitude (mm)")

    ax = axes[3]
    t = np.asarray(times, float)
    hours = np.mod(origin_hour + t * 24.0, 24.0)
    model = fit.evaluate()
    flat = model.reshape(model.shape[0], -1)
    sel = m.ravel() if m.size == flat.shape[1] else slice(None)
    ax.plot(hours, 1000.0 * np.nanmean(flat[:, sel], axis=1), ".", ms=2,
            color="C0", label="ice (masked mean)")
    if stable is not None:
        s = np.asarray(stable, bool).ravel()
        if s.any():
            ax.plot(hours, 1000.0 * np.nanmean(flat[:, s], axis=1), ".", ms=2,
                    color="C3", label="stable ground (null)")
    ax.set_xlabel("Clock hour (hr)")
    ax.set_ylabel("Modelled LOS (mm)")
    ax.set_title("stacked diurnal cycle, relative to the reference epoch", fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    return axes


__all__.append("diurnal_summary")
