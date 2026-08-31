"""Diurnal signals: harmonic analysis, and telling ice from atmosphere.

BakerBend1/20170803 is 723 acquisitions on a 2-minute cadence spanning 24.18
hours — one full diurnal cycle, sampled 723 times.  That is not an incidental
property of the dataset, it is the experiment: the target is the sub-daily
velocity and uplift variation driven by water pressure in the subglacial
drainage system, as the hydrological system reorganises through a melt day.

What a diurnal signal looks like
--------------------------------
Melt-driven input peaks in the afternoon.  Where the drainage system cannot
carry it, water pressure rises, the glacier partly separates from its bed, and
the surface both **lifts** and **speeds up**, relaxing overnight as the system
drains.  In LOS displacement that is a roughly sinusoidal term at 24 hours
riding on the secular flow rate, and its *phase* — the hour of peak — is the
diagnostic quantity, because it says how long the bed takes to respond to
surface melt.  :func:`fit_harmonics` estimates amplitude and phase per pixel by
weighted least squares, which handles the 2-minute sampling and the one
6-minute gap without special pleading.

The problem: the atmosphere is diurnal too
------------------------------------------
This is the whole methodological difficulty, and it is worth being blunt about.
Air temperature and humidity on a mountain flank cycle with a 24-hour period.
A residual refractivity error therefore appears in the time series **at exactly
the period being looked for, and roughly in phase with it** — melt and warming
peak together.  From :mod:`gpri.refractivity`, 1 % of relative humidity is 0.8
fringes across the swath; the diurnal humidity swing is tens of percent.  The
atmospheric diurnal is, before correction, one to two orders of magnitude
larger than the glaciological one.

So a diurnal signal in the data is not evidence of subglacial hydrology.  Three
tests here, and a claim should survive all three:

1. :func:`range_dependence` — **the sharp one**.  Residual refractivity puts a
   phase ramp that is *linear in slant range*.  Ice motion has no reason to
   correlate with distance from a tripod.  If diurnal amplitude grows linearly
   with range, it is atmosphere.  This is a strong discriminator and it costs
   nothing.
2. :func:`atmospheric_coherence` — regress the per-pixel diurnal against the
   independently estimated per-epoch refractivity series
   (:func:`gpri.refractivity.invert_refractivity`).  What is left after
   projecting that out is what can be defended.
3. :func:`stable_ground_null` — run the same fit on bedrock, which is not
   moving.  Any diurnal amplitude recovered there is the error floor, and no
   signal on ice below it means anything.

Reference the series first
--------------------------
Before any of that: an interferogram fixes phase only up to an additive
constant, so integrating a network accumulates one arbitrary offset per pair
into a scene-wide drift.  On a mountain flank that drift is diurnal, because
the atmosphere is, and it appears on ice and bedrock alike at the same phase.
Run :func:`gpri.timeseries.reference_to_stable` before fitting, and hold out
reference pixels from the null test — testing on the pixels used to reference
is circular, since they were forced to zero by construction.  Skipping this
step produces a large, clean, entirely spurious diurnal signal; it is the
first thing to check when one appears.

The geometry limits what can be claimed
---------------------------------------
A tripod radar looks nearly horizontally: at BakerBend1 the beam elevation is
10 degrees, so LOS sensitivity to **vertical** motion is ``sin(10 deg) = 0.17``
while sensitivity to horizontal motion along the look direction is
``cos(10 deg) = 0.98``.  Uplift is suppressed by a factor of six relative to
speed-up, and one line of sight cannot separate them at all.
:func:`vertical_sensitivity` and :func:`decompose_los` make that explicit
rather than letting a LOS time series be read as uplift.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "HarmonicFit", "harmonic_design", "fit_harmonics", "diurnal_amplitude",
    "diurnal_phase", "range_dependence", "atmospheric_coherence",
    "stable_ground_null", "look_vector", "vertical_sensitivity",
    "decompose_los", "DIURNAL", "SEMIDIURNAL",
]

#: Periods in days.
DIURNAL = 1.0
SEMIDIURNAL = 0.5


# ------------------------------------------------------------------- the fit
def harmonic_design(times, periods=(DIURNAL,), degree=1):
    """Design matrix for ``offset + secular + sum_k (cos, sin)`` at each period.

    Columns are ``[1, t, t^2, ..., cos(w1 t), sin(w1 t), cos(w2 t), ...]`` with
    ``t`` in days.  ``degree=1`` (offset plus linear rate) is right for a single
    day: a higher polynomial competes with the harmonic for the same variance
    and the fit stops meaning anything.

    Raises
    ------
    ValueError
        If the record is shorter than the longest period requested.  Fitting a
        24-hour harmonic to six hours of data returns a number, and that number
        is meaningless — the amplitude and the secular rate are not separable
        over a fraction of a cycle.
    """
    t = np.asarray(times, float)
    span = t.max() - t.min() if t.size else 0.0
    for p in periods:
        if span < p:
            raise ValueError(
                f"record spans {span * 24:.2f} h but a {p * 24:.0f} h harmonic "
                f"was requested; amplitude and secular rate are not separable "
                f"over less than one cycle")
    cols = [t ** k for k in range(degree + 1)]
    for p in periods:
        w = 2.0 * np.pi / float(p)
        cols += [np.cos(w * t), np.sin(w * t)]
    return np.column_stack(cols)


class HarmonicFit:
    """Secular rate plus harmonics fitted to a displacement time series."""

    def __init__(self, coeffs, times, periods, degree, residual_rms=None,
                 total_rms=None, shape=()):
        self.coeffs = np.asarray(coeffs, float)
        self.times = np.asarray(times, float)
        self.periods = tuple(periods)
        self.degree = int(degree)
        self.residual_rms = residual_rms
        self.total_rms = total_rms
        self.shape = tuple(shape)

    def _slot(self, period):
        try:
            k = self.periods.index(period)
        except ValueError:
            raise ValueError(f"no harmonic at period {period}; "
                             f"fitted {self.periods}") from None
        return self.degree + 1 + 2 * k

    @property
    def offset(self):
        return self.coeffs[0]

    @property
    def secular(self):
        """Linear rate, metres per day (the units of the input over time)."""
        if self.degree < 1:
            return np.zeros(self.shape)
        return self.coeffs[1]

    def amplitude(self, period=DIURNAL):
        """Peak-to-mean amplitude ``sqrt(A^2 + B^2)`` of one harmonic."""
        i = self._slot(period)
        return np.hypot(self.coeffs[i], self.coeffs[i + 1])

    def phase(self, period=DIURNAL):
        """Phase of the harmonic in radians, ``atan2(B, A)``."""
        i = self._slot(period)
        return np.arctan2(self.coeffs[i + 1], self.coeffs[i])

    def peak_time(self, period=DIURNAL, origin_hour=0.0):
        """Hour of day at which the harmonic peaks, in [0, period).

        ``origin_hour`` is the clock time of ``times[0]``; pass it and the
        answer is a real time of day, which is the number that means something
        — an afternoon peak says melt input, an early-morning one does not.
        """
        ph = self.phase(period)
        hours = (ph / (2.0 * np.pi)) * period * 24.0
        return np.mod(origin_hour + hours, period * 24.0)

    def explained_variance(self):
        """Fraction of the signal variance the model accounts for, per pixel."""
        if self.residual_rms is None or self.total_rms is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            r = 1.0 - (np.asarray(self.residual_rms) ** 2
                       / np.asarray(self.total_rms) ** 2)
        return np.clip(np.nan_to_num(r, nan=0.0), 0.0, 1.0)

    def evaluate(self, times=None):
        """The fitted model, sampled at ``times`` (defaults to the input times)."""
        t = self.times if times is None else np.asarray(times, float)
        G = harmonic_design(t, self.periods, self.degree)
        flat = self.coeffs.reshape(self.coeffs.shape[0], -1)
        return (G @ flat).reshape((len(t),) + self.shape)

    def __repr__(self):
        a = np.nanmedian(self.amplitude())
        return (f"HarmonicFit(periods={self.periods}, degree={self.degree}, "
                f"median diurnal amplitude={a * 1000:.3f} mm, "
                f"pixels={self.shape})")


def fit_harmonics(displacement, times, periods=(DIURNAL,), degree=1,
                  weights=None, rcond=None):
    """Weighted least-squares harmonic fit to a displacement time series.

    Parameters
    ----------
    displacement : array (n_epochs, ...)
        LOS displacement in metres, from
        :attr:`gpri.timeseries.TimeSeries.displacement` or
        :func:`gpri.timeseries.displacement_from_phases`.
    times : (n_epochs,) array
        Days from the first acquisition — ``network.times``.
    periods : tuple
        Periods to fit, in days.  ``(1.0,)`` for diurnal alone; ``(1.0, 0.5)``
        adds the semi-diurnal, which picks up an asymmetric rise-and-relax
        shape that a single sinusoid smears.
    weights : (n_epochs,) or (n_epochs, ...) array, optional
        Per-epoch or per-pixel-per-epoch weights.  A shared 1-D weight vector
        is solved once for the whole image; per-pixel weights cost a solve per
        pixel and are much slower.

    Returns
    -------
    :class:`HarmonicFit`
    """
    d = np.asarray(displacement, float)
    t = np.asarray(times, float)
    if d.shape[0] != t.size:
        raise ValueError(f"{d.shape[0]} epochs of displacement but {t.size} times")

    shape = d.shape[1:]
    Y = d.reshape(d.shape[0], -1)
    G = harmonic_design(t, periods, degree)

    finite = np.isfinite(Y)
    Y0 = np.where(finite, Y, 0.0)

    if weights is None and finite.all():
        X = np.linalg.lstsq(G, Y0, rcond=rcond)[0]
    else:
        w = np.ones_like(Y) if weights is None else np.broadcast_to(
            np.asarray(weights, float).reshape(t.size, -1)
            if np.ndim(weights) > 1 else np.asarray(weights, float)[:, None],
            Y.shape).copy()
        w = np.where(finite & np.isfinite(w), np.maximum(w, 0.0), 0.0)
        X = np.empty((G.shape[1], Y.shape[1]))
        shared = np.ndim(weights) <= 1 and finite.all()
        if shared:
            sw = np.sqrt(w[:, 0])
            X = np.linalg.lstsq(G * sw[:, None], Y0 * sw[:, None], rcond=rcond)[0]
        else:
            for c in range(Y.shape[1]):
                sw = np.sqrt(w[:, c])
                if np.count_nonzero(sw) < G.shape[1]:
                    X[:, c] = np.nan
                    continue
                X[:, c] = np.linalg.lstsq(G * sw[:, None], Y0[:, c] * sw,
                                          rcond=rcond)[0]

    resid = Y0 - G @ np.nan_to_num(X)
    n_ok = np.maximum(finite.sum(axis=0), 1)
    rms = np.sqrt((np.where(finite, resid, 0.0) ** 2).sum(axis=0) / n_ok)
    mean = (Y0.sum(axis=0) / n_ok)
    tot = np.sqrt((np.where(finite, Y0 - mean, 0.0) ** 2).sum(axis=0) / n_ok)

    return HarmonicFit(X.reshape((G.shape[1],) + shape), t, periods, degree,
                       residual_rms=rms.reshape(shape),
                       total_rms=tot.reshape(shape), shape=shape)


def diurnal_amplitude(displacement, times, **kwargs):
    """Shorthand: diurnal amplitude in metres, per pixel."""
    return fit_harmonics(displacement, times, **kwargs).amplitude(DIURNAL)


def diurnal_phase(displacement, times, origin_hour=0.0, **kwargs):
    """Shorthand: hour of day of the diurnal peak, per pixel."""
    return fit_harmonics(displacement, times, **kwargs).peak_time(
        DIURNAL, origin_hour=origin_hour)


# ------------------------------------------------- is it ice or is it air?
def range_dependence(amplitude, slant_range, mask=None, min_pixels=50):
    """Regress diurnal amplitude against slant range.

    The sharpest test available, and the cheapest.  A residual refractivity
    error puts phase that is **linear in slant range**, so its diurnal
    signature grows linearly with distance from the radar.  Ice motion has no
    reason whatever to correlate with distance from a tripod parked on a
    moraine.  A significant positive slope means the diurnal is atmospheric.

    Parameters
    ----------
    amplitude : 2-D array
        Diurnal amplitude per pixel, metres.
    slant_range : 1-D array
        Slant range per range sample, metres — broadcast across azimuth.
    mask : bool array, optional
        Where to trust the amplitude (coherence, stable ground).

    Returns
    -------
    dict with ``slope`` (metres of amplitude per metre of range),
    ``intercept``, ``correlation``, ``n``, and ``verdict`` — a plain-language
    reading of the correlation.
    """
    a = np.asarray(amplitude, float)
    r = np.asarray(slant_range, float)
    R = np.broadcast_to(r, a.shape) if r.shape != a.shape else r

    ok = np.isfinite(a) & np.isfinite(R)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    n = int(ok.sum())
    if n < min_pixels:
        return {"slope": np.nan, "intercept": np.nan, "correlation": np.nan,
                "n": n, "verdict": f"only {n} usable pixels; no test possible"}

    x, y = R[ok], a[ok]
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])

    if abs(corr) < 0.2:
        verdict = ("no range dependence (|r| < 0.2): consistent with a real "
                   "surface signal, though not proof of one")
    elif abs(corr) < 0.5:
        verdict = ("moderate range dependence: some of this diurnal is "
                   "residual atmosphere; correct further before interpreting")
    else:
        verdict = ("strong range dependence (|r| >= 0.5): this diurnal is "
                   "dominated by residual refractivity, not ice motion")
    return {"slope": float(slope), "intercept": float(intercept),
            "correlation": corr, "n": n, "verdict": verdict}


def atmospheric_coherence(displacement, times, refractivity, weights=None):
    """How much of the time series the atmosphere alone explains.

    Regresses each pixel's displacement onto the independently estimated
    per-epoch refractivity series (N-units, from
    :func:`gpri.refractivity.invert_refractivity`) plus an offset and rate.
    Returns the fraction of variance the refractivity term accounts for.

    A pixel where this is high has a time series driven by the atmosphere,
    whatever period it happens to be at.  Report it beside any diurnal
    amplitude: a 3 mm diurnal with 80 % atmospheric variance is not a
    measurement of the bed.
    """
    d = np.asarray(displacement, float)
    t = np.asarray(times, float)
    N = np.asarray(refractivity, float)
    if N.ndim > 1:
        N = np.nanmean(N.reshape(N.shape[0], -1), axis=1)
    if N.size != t.size or d.shape[0] != t.size:
        raise ValueError("displacement, times and refractivity must share an "
                         "epoch axis")

    Y = d.reshape(d.shape[0], -1)
    finite = np.isfinite(Y)
    Y0 = np.where(finite, Y, 0.0)
    N0 = np.nan_to_num(N)

    full = np.column_stack([np.ones_like(t), t, N0])
    base = np.column_stack([np.ones_like(t), t])

    def rss(G):
        X = np.linalg.lstsq(G, Y0, rcond=None)[0]
        return (np.where(finite, Y0 - G @ X, 0.0) ** 2).sum(axis=0)

    r_full, r_base = rss(full), rss(base)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(r_base > 0, 1.0 - r_full / r_base, 0.0)
    return np.clip(np.nan_to_num(frac), 0.0, 1.0).reshape(d.shape[1:])


def stable_ground_null(displacement, times, stable_mask, periods=(DIURNAL,),
                       **kwargs):
    """Run the diurnal fit on ground that is not moving.

    Bedrock and moraine have no subglacial hydrology.  Whatever diurnal
    amplitude comes back from them is the method's error floor — atmosphere,
    unwrapping, thermal expansion of the tripod, all of it.  A diurnal signal
    on ice that does not clear this floor is not a detection.

    Returns
    -------
    dict with ``amplitude_median``, ``amplitude_p95``, ``phase_concentration``
    and ``n``.  ``phase_concentration`` is the circular resultant length of the
    stable-ground phases, in [0, 1]: near 0 means the null pixels are
    incoherent noise (good), near 1 means they share a phase, which is a
    systematic error the ice pixels are getting too.
    """
    m = np.asarray(stable_mask, bool)
    d = np.asarray(displacement, float)
    if not m.any():
        return {"amplitude_median": np.nan, "amplitude_p95": np.nan,
                "phase_concentration": np.nan, "n": 0}

    sub = d.reshape(d.shape[0], -1)[:, m.ravel()]
    fit = fit_harmonics(sub, times, periods=periods, **kwargs)
    a = fit.amplitude(periods[0])
    ph = fit.phase(periods[0])
    good = np.isfinite(a) & np.isfinite(ph)
    if not good.any():
        return {"amplitude_median": np.nan, "amplitude_p95": np.nan,
                "phase_concentration": np.nan, "n": 0}
    return {
        "amplitude_median": float(np.median(a[good])),
        "amplitude_p95": float(np.percentile(a[good], 95)),
        "phase_concentration": float(np.abs(np.mean(np.exp(1j * ph[good])))),
        "n": int(good.sum()),
    }


# ------------------------------------------------------------- LOS geometry
def look_vector(geom, rows=None, cols=None):
    """Unit vector from radar to target, as ``(east, north, up)``.

    Uses the antenna elevation angle for the vertical component, which is the
    beam centre and not the true look angle to a specific target — that needs a
    DEM, and none accompanies the data.  Over a flank at 5-12 degrees the
    approximation costs a few percent on the vertical component; it is not
    good enough to invert uplift from, which is the point of
    :func:`vertical_sensitivity`.
    """
    b = np.deg2rad(geom.bearings() if rows is None else geom.bearings()[rows])
    el = np.deg2rad(geom.elevation)
    east = np.sin(b) * np.cos(el)
    north = np.cos(b) * np.cos(el)
    up = np.full(b.shape, np.sin(el))
    return np.stack([east, north, up], axis=-1)


def vertical_sensitivity(geom):
    """LOS sensitivity to vertical motion: ``sin(beam elevation)``.

    At BakerBend1 this is ``sin(10 deg) = 0.174``.  So 1 mm of uplift produces
    0.17 mm of LOS, while 1 mm of horizontal motion toward the radar produces
    0.98 mm.  A tripod radar is a **horizontal-motion instrument** that is
    nearly blind to uplift, and a single line of sight cannot separate the two
    in any case.  Quote this number next to any uplift claim.
    """
    return float(np.sin(np.deg2rad(geom.elevation)))


def decompose_los(los, geom, flow_azimuth, uplift_ratio=0.0, rows=None):
    """Convert LOS displacement to along-flow displacement, under an assumption.

    One line of sight measures one number per pixel; flow and uplift are two.
    They cannot be separated without another viewing geometry or an external
    constraint.  This function makes the constraint explicit: you supply the
    flow azimuth (degrees true, the direction the ice moves) and, optionally,
    the ratio of uplift to along-flow motion, and it divides the LOS by the
    resulting projection factor.

    Parameters
    ----------
    los : array
        LOS displacement, positive toward the radar.
    flow_azimuth : float or array
        Direction of ice flow, degrees clockwise from true north.
    uplift_ratio : float
        Vertical motion as a fraction of along-flow motion.  0 assumes pure
        horizontal flow.

    Returns
    -------
    along_flow : array
        Displacement along the flow direction.  **NaN where the flow direction
        is within 10 degrees of perpendicular to the look direction** — there
        the projection factor is near zero and dividing by it manufactures
        enormous numbers out of noise.
    """
    b = geom.bearings() if rows is None else geom.bearings()[rows]
    el = np.deg2rad(geom.elevation)
    az = np.deg2rad(np.asarray(flow_azimuth, float) - np.asarray(b, float)[..., None]
                    if np.ndim(flow_azimuth) else
                    np.asarray(flow_azimuth, float) - np.asarray(b, float))
    # motion toward the radar is -cos(angle between flow and look), since the
    # look vector points away from the radar
    factor = -np.cos(az) * np.cos(el) + uplift_ratio * np.sin(el)
    if np.ndim(factor) and np.ndim(los) > np.ndim(factor):
        factor = factor[..., None] if factor.shape[0] == los.shape[0] else factor
    factor = np.where(np.abs(factor) < np.cos(np.deg2rad(80.0)), np.nan, factor)
    return np.asarray(los, float) / factor
