"""Refractivity (atmospheric path-delay) screens for ground-based radar.

For a tripod radar looking horizontally through the boundary layer, the
dominant error is not topography or orbits — it is the change in refractive
index between two acquisitions.  A uniform change ``dn`` along the whole path
puts a phase ramp on the interferogram that is **linear in slant range**:

    phi_atm(r) = (4 pi / lambda) * dn * r          (two-way)

The numbers matter here.  BakerBend1 spans 300 m to 16.9 km at
lambda = 1.743 cm, so ``4 pi / lambda = 721`` rad per metre of range change.
A very ordinary ``dn = 1e-6`` — a fraction of a degree of temperature, or a
few tenths of a millibar of water vapour — puts **12 radians**, almost two
full fringes, across the swath.  At ``dn = 1e-5`` it is 120 radians.  Nothing
in a GPRI interferogram is interpretable until this is removed.

Estimating it on *wrapped* phase is the whole difficulty, and it is why this
module does not simply least-squares a plane.  Instead:

1. :func:`estimate_range_ramp` finds the dominant linear-in-range term by
   maximising ``|sum w exp(i(phi - k r))|`` over ``k`` — a matched filter, run
   directly on the wrapped phase, with no unwrapping and no sensitivity to how
   many fringes there are.
2. :func:`fit_screen` then fits the remaining (now small, sub-radian) structure
   with robust iteratively-reweighted least squares, which handles azimuth
   tilts and range curvature.

Both stages are weighted, so coherence drives the fit and decorrelated pixels
are ignored rather than averaged in.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "PhaseScreen", "estimate_range_ramp", "ramp_objective", "fit_screen",
    "remove_screen", "delta_refractivity", "design_matrix", "stable_mask",
    "MODELS",
]

#: Model name -> the terms it fits.  ``r`` is slant range in metres (referenced
#: to near range), ``a`` is azimuth angle in degrees (referenced to boresight).
#: Terms that need an azimuth axis.  Matched exactly -- testing ``"a" in term``
#: would also fire on any term whose name merely contains the letter.
AZIMUTH_TERMS = frozenset({"a", "a2", "ra"})

MODELS = {
    "constant":  ("1",),
    "linear":    ("1", "r"),
    "quadratic": ("1", "r", "r2"),
    "planar":    ("1", "r", "a"),
    "bilinear":  ("1", "r", "a", "ra"),
    "full":      ("1", "r", "r2", "a", "ra", "a2"),
}

#: Every term any model may name.
_KNOWN_TERMS = frozenset({"1", "r", "r2", "a", "a2", "ra"})


# --------------------------------------------------------------- design matrix
def design_matrix(model, slant_range, azimuth=None):
    """Columns of the phase-screen model evaluated on the image grid.

    Parameters
    ----------
    model : str or sequence of str
        A key of :data:`MODELS`, or an explicit term list.
    slant_range : array (nr,)
        Slant range per range sample, metres.  Use
        :meth:`gpri_tools.gamma.ParFile.slant_range`.
    azimuth : array (na,), optional
        Azimuth angle per line, degrees.  Required by any model with an ``a``
        term; use :func:`gpri_tools.gamma.azimuth_angles`.

    Returns
    -------
    A : array (na * nr, n_terms)
        Ready for least squares against a flattened image.
    """
    terms = MODELS[model] if isinstance(model, str) else tuple(model)
    r = np.asarray(slant_range, float)
    nr = r.size
    # centre the predictors: keeps the normal equations well conditioned and
    # makes the constant term mean phase rather than phase at r = 0
    rc = r - r.mean()
    if azimuth is None:
        unknown = [t for t in terms if t not in _KNOWN_TERMS]
        if unknown:
            raise ValueError(f"unknown screen term {unknown[0]!r}; "
                             f"known terms are {sorted(_KNOWN_TERMS)}")
        if AZIMUTH_TERMS.intersection(terms):
            raise ValueError(f"model {model!r} needs azimuth angles")
        a = np.zeros(1)
    else:
        a = np.asarray(azimuth, float)
        a = a - a.mean()
    na = a.size

    R = np.broadcast_to(rc, (na, nr))
    A_ = np.broadcast_to(a[:, None], (na, nr))
    cols = {
        "1": np.ones((na, nr)), "r": R, "r2": R ** 2,
        "a": A_, "a2": A_ ** 2, "ra": R * A_,
    }
    try:
        stack = [np.asarray(cols[t], float).ravel() for t in terms]
    except KeyError as exc:
        raise ValueError(f"unknown screen term {exc.args[0]!r}; "
                         f"known terms are {sorted(cols)}") from None
    return np.column_stack(stack)


class PhaseScreen:
    """A fitted atmospheric screen, and what it implies physically."""

    def __init__(self, coeffs, model, slant_range, azimuth=None,
                 wavelength=None, ramp=0.0, quality=None):
        self.coeffs = np.asarray(coeffs, float)
        self.model = model
        self.slant_range = np.asarray(slant_range, float)
        self.azimuth = None if azimuth is None else np.asarray(azimuth, float)
        self.wavelength = wavelength
        #: linear-in-range term from the matched filter, rad/m
        self.ramp = float(ramp)
        #: matched-filter objective in [0, 1]; ~1 means a clean ramp
        self.quality = quality

    def evaluate(self, shape=None):
        """The screen as a 2-D phase image, radians."""
        A = design_matrix(self.model, self.slant_range, self.azimuth)
        na = 1 if self.azimuth is None else self.azimuth.size
        nr = self.slant_range.size
        phi = (A @ self.coeffs).reshape(na, nr)
        phi = phi + self.ramp * (self.slant_range - self.slant_range.mean())
        if shape is not None and phi.shape != tuple(shape):
            phi = np.broadcast_to(phi, shape)
        return phi

    @property
    def delta_n(self):
        """Path-averaged refractive-index change implied by the range ramp.

        ``dn = ramp * lambda / (4 pi)``.  Dimensionless; multiply by 1e6 for
        the N-units meteorologists use.
        """
        if self.wavelength is None:
            raise ValueError("wavelength not known for this screen")
        return delta_refractivity(self.ramp, self.wavelength)

    def __repr__(self):
        q = "?" if self.quality is None else f"{self.quality:.3f}"
        s = f"PhaseScreen({self.model}, ramp={self.ramp:.4e} rad/m, quality={q}"
        if self.wavelength is not None:
            s += f", dN={self.delta_n * 1e6:+.3f}"
        return s + ")"


def delta_refractivity(ramp, wavelength):
    """Refractive-index change from a range-phase slope (rad/m -> dimensionless)."""
    return ramp * wavelength / (4.0 * np.pi)


# ----------------------------------------------------------- the matched filter
def _collapse_azimuth(phase, weights, slant_range):
    """Sum ``w exp(i phi)`` down the azimuth axis, one complex value per range bin.

    The whole point of the matched filter is that the ramp depends on **range
    only**, and range is constant along an azimuth column.  So

        sum_{a,r} w exp(i phi) exp(-i k r)  =  sum_r [sum_a w exp(i phi)] exp(-i k r)

    and the inner bracket can be computed once.  That turns a
    22101 x 396 x n_k problem into a 22101 x n_k one, and then into a single
    FFT.  Returns ``(s, total_weight)`` with ``s`` of length ``nr``, or
    ``(None, None)`` if range is not constant along azimuth (in which case the
    caller must take the general path).
    """
    phase = np.asarray(phase)
    r = np.asarray(slant_range, float)
    if r.ndim != 1 or phase.shape[-1] != r.size:
        return None, None
    z = np.exp(1j * phase) if not np.iscomplexobj(phase) else _unit_or_zero(phase)
    w = np.ones(z.shape, np.float64) if weights is None else np.asarray(weights, float)
    w = np.where(np.isfinite(w) & np.isfinite(z.real) & np.isfinite(z.imag), w, 0.0)
    w = np.broadcast_to(w, z.shape)
    zw = np.where(w > 0, w * np.nan_to_num(z), 0.0)
    axes = tuple(range(zw.ndim - 1))
    return zw.sum(axis=axes), float(w.sum())


def _unit_or_zero(z):
    m = np.abs(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m > 0, z / np.where(m > 0, m, 1.0), 0.0)


def ramp_objective(phase, slant_range, k, weights=None, _collapsed=None):
    """Matched-filter response ``|<w exp(i(phi - k r))>|`` at slope(s) ``k``.

    1.0 means every weighted pixel agrees on the same ramp; 0 means no ramp is
    present.  ``k`` may be an array, in which case the result has its shape.
    """
    scalar = np.ndim(k) == 0
    kk = np.atleast_1d(np.asarray(k, float))

    if _collapsed is None:
        _collapsed = _collapse_azimuth(phase, weights, slant_range)
    s, tot = _collapsed

    if s is not None:
        if not tot:
            out = np.zeros(kk.shape)
        else:
            r = np.asarray(slant_range, float)
            # (n_k, nr) would be huge; accumulate in chunks over k instead
            out = np.empty(kk.size, float)
            step = max(1, int(4e7 // max(r.size, 1)))
            for a in range(0, kk.size, step):
                b = min(a + step, kk.size)
                out[a:b] = np.abs(np.exp(-1j * kk.ravel()[a:b, None] * r[None, :]) @ s) / tot
            out = out.reshape(kk.shape)
        return out[0] if scalar else out

    # general path: range varies per pixel
    z = np.exp(1j * np.asarray(phase)) if not np.iscomplexobj(phase) else _unit_or_zero(phase)
    r = np.broadcast_to(np.asarray(slant_range, float), z.shape)
    w = np.ones(z.shape) if weights is None else np.broadcast_to(np.asarray(weights, float), z.shape)
    w = np.where(np.isfinite(w) & np.isfinite(z.real) & np.isfinite(z.imag), w, 0.0)
    tot = w.sum()
    if tot <= 0:
        out = np.zeros(kk.shape)
        return out[0] if scalar else out
    zw, rr = (w * np.nan_to_num(z)).ravel(), r.ravel()
    out = np.array([np.abs(np.dot(zw, np.exp(-1j * ki * rr))) / tot for ki in kk.ravel()])
    out = out.reshape(kk.shape)
    return out[0] if scalar else out


def estimate_range_ramp(phase, slant_range, weights=None, max_delta_n=1e-4,
                        wavelength=None, oversample=4, refine=True):
    """Find the linear-in-range phase slope without unwrapping.

    Scans ``k`` over the range of physically plausible refractivity changes and
    returns the maximiser of :func:`ramp_objective`, polished by parabolic
    interpolation on the three samples around the peak.

    On a uniformly sampled range axis — which every GAMMA slant-range product
    has — the scan is a zero-padded FFT of the azimuth-collapsed signal, so the
    cost is ``O(nr log nr)`` regardless of how many trial slopes are implied.

    Parameters
    ----------
    max_delta_n : float
        Half-width of the search, as a refractive-index change.  ``1e-4`` is far
        beyond anything the boundary layer does.
    wavelength : float, optional
        Needed to turn ``max_delta_n`` into a slope bound.  Defaults to Ku band
        (1.743 cm), what a GPRI-II transmits.

    Returns
    -------
    ramp : float
        Slope in rad/m.
    quality : float
        Matched-filter response at the peak, in ``[0, 1]``.
    """
    lam = 1.743e-2 if wavelength is None else float(wavelength)
    r = np.asarray(slant_range, float)
    span = float(np.ptp(r))
    if span <= 0:
        return 0.0, 0.0
    k_max = 4.0 * np.pi * max_delta_n / lam

    s, tot = _collapse_azimuth(phase, weights, slant_range)
    if s is not None and not tot:
        return 0.0, 0.0        # nothing weighted in: no ramp, no confidence
    uniform = False
    if s is not None and tot and r.size > 2:
        dr = np.diff(r)
        uniform = bool(np.allclose(dr, dr[0], rtol=1e-9, atol=0) and dr[0] != 0)

    if uniform:
        dr = float(r[1] - r[0])
        nr = r.size
        # spectrum resolution 2*pi/(M*dr) must beat the 2*pi/span peak width
        M = int(2 ** np.ceil(np.log2(max(8, nr * max(1, oversample)))))
        spec = np.fft.fftshift(np.fft.fft(s, n=M))
        kgrid = np.fft.fftshift(np.fft.fftfreq(M, d=dr)) * 2.0 * np.pi
        # fft gives sum_n s_n exp(-i k dr n); restore the exp(-i k r[0]) phase
        # (irrelevant to the magnitude, so only the magnitude is used here)
        resp = np.abs(spec) / tot
        sel = np.abs(kgrid) <= k_max
        if not sel.any():
            sel = np.ones_like(kgrid, bool)
        kgrid, resp = kgrid[sel], resp[sel]
    else:
        dk = 2.0 * np.pi / span / max(1, oversample)
        n = min(int(np.ceil(2.0 * k_max / dk)) + 1, 200_000)
        kgrid = np.linspace(-k_max, k_max, n)
        resp = ramp_objective(phase, slant_range, kgrid, weights, _collapsed=(s, tot))

    i = int(np.argmax(resp))
    k_hat, q_hat = float(kgrid[i]), float(resp[i])

    if refine and 0 < i < len(kgrid) - 1:
        dk = float(kgrid[1] - kgrid[0])
        y0, y1, y2 = resp[i - 1], resp[i], resp[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
            if abs(shift) <= 1.0:
                k_hat = float(kgrid[i] + shift * dk)
        # The parabola only gets to ~0.5% of the grid step, because the
        # objective is not actually parabolic near its peak.  A golden-section
        # search on the exact objective costs ~40 one-dimensional dot products
        # and takes the error to the numerical floor.
        k_hat, q_hat = _golden_max(phase, slant_range, weights, (s, tot),
                                   k_hat - dk, k_hat + dk)
    return k_hat, q_hat


def _golden_max(phase, slant_range, weights, collapsed, lo, hi, tol=1e-12,
                max_iter=200):
    """Maximise :func:`ramp_objective` on ``[lo, hi]`` by golden-section search."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    f = lambda k: float(ramp_objective(phase, slant_range, k, weights,
                                       _collapsed=collapsed))
    c, d = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(hi - lo) <= tol * max(abs(lo) + abs(hi), 1e-12):
            break
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = f(d)
    k = c if fc > fd else d
    return float(k), float(max(fc, fd))


# --------------------------------------------------------------- the full fit
def _terms_of(model):
    return MODELS[model] if isinstance(model, str) else tuple(model)


def _design_at(terms, r, a):
    """Design columns evaluated at explicit, already-centred predictor values."""
    cols = {"1": np.ones_like(r), "r": r, "r2": r ** 2,
            "a": a, "a2": a ** 2, "ra": r * a}
    try:
        return np.column_stack([cols[t] for t in terms])
    except KeyError as exc:
        raise ValueError(f"unknown screen term {exc.args[0]!r}; "
                         f"known terms are {sorted(cols)}") from None


def fit_screen(phase, par=None, slant_range=None, azimuth=None, weights=None,
               mask=None, model="linear", wavelength=None, robust=True,
               max_delta_n=1e-4, iterations=5, huber=1.345, max_samples=200_000,
               seed=0):
    """Estimate the atmospheric screen on a wrapped interferogram.

    Runs the matched filter first, subtracts the ramp it finds, and fits the
    remainder with robust weighted least squares.  Because the residual after
    the ramp is small, the least-squares stage can work on
    ``angle(exp(i phi))`` with no unwrapping at all.

    Parameters
    ----------
    phase : array (na, nr)
        Wrapped interferometric phase in radians, or the complex interferogram
        itself (its phase is used and its magnitude becomes the default weight).
    par : :class:`gpri_tools.gamma.ParFile`, optional
        Supplies ``slant_range``, ``azimuth`` and ``wavelength`` if not given.
    weights : array (na, nr), optional
        Per-pixel weight — coherence is the natural choice.
    mask : array (na, nr) of bool, optional
        True where a pixel may be used.  Pair with :func:`stable_mask` to tie
        the screen to ground you believe is not moving.
    model : str
        Key of :data:`MODELS`, or an explicit term list.
    robust : bool
        Iteratively reweight with a Huber loss, so a moving glacier tongue or
        an unwrapping error cannot drag the screen.
    max_samples : int
        Fit on at most this many pixels, drawn at random from those allowed.
        Six coefficients do not need eight million points, and subsampling is
        what keeps this interactive on a full-swath GPRI scene.

    Returns
    -------
    screen : :class:`PhaseScreen`
    """
    phase = np.asarray(phase)
    if np.iscomplexobj(phase):
        if weights is None:
            weights = np.abs(phase)
        phase = np.angle(phase)
    phase = phase.astype(np.float64, copy=False)

    terms = _terms_of(model)
    if par is not None:
        from .gamma import azimuth_angles
        if slant_range is None:
            slant_range = par.slant_range()
        if azimuth is None and AZIMUTH_TERMS.intersection(terms):
            azimuth = azimuth_angles(par)
        if wavelength is None:
            wavelength = par.wavelength
    if slant_range is None:
        raise ValueError("need slant_range, or a par file to take it from")
    unknown = [t for t in terms if t not in _KNOWN_TERMS]
    if unknown:
        raise ValueError(f"unknown screen term {unknown[0]!r}; "
                         f"known terms are {sorted(_KNOWN_TERMS)}")
    if azimuth is None and AZIMUTH_TERMS.intersection(terms):
        raise ValueError(f"model {model!r} needs azimuth angles")

    w = (np.ones(phase.shape, np.float64) if weights is None
         else np.asarray(weights, dtype=np.float64))
    w = np.broadcast_to(w, phase.shape).copy()
    w[~np.isfinite(w)] = 0.0
    w[~np.isfinite(phase)] = 0.0
    if mask is not None:
        w = np.where(np.asarray(mask, bool), w, 0.0)

    ramp, quality = estimate_range_ramp(
        phase, slant_range, weights=w, max_delta_n=max_delta_n,
        wavelength=wavelength)

    r = np.asarray(slant_range, float)
    resid = np.angle(np.exp(1j * (phase - ramp * np.broadcast_to(r, phase.shape))))

    # --- fit the remainder on a subsample of the usable pixels only ----------
    rows, cols = np.nonzero(w > 0)
    if rows.size == 0:
        return PhaseScreen(np.zeros(len(terms)), model, slant_range, azimuth,
                           wavelength=wavelength, ramp=ramp, quality=quality)
    if max_samples and rows.size > max_samples:
        pick = np.random.default_rng(seed).choice(rows.size, max_samples, replace=False)
        rows, cols = rows[pick], cols[pick]

    rc = r - r.mean()
    ac = (np.zeros(phase.shape[0]) if azimuth is None
          else np.asarray(azimuth, float) - np.asarray(azimuth, float).mean())
    A = _design_at(terms, rc[cols], ac[rows])
    y = resid[rows, cols]
    w0 = w[rows, cols]
    ww = w0.copy()

    coeffs = np.zeros(A.shape[1])
    n_iter = max(1, iterations) if robust else 1
    for it in range(n_iter):
        good = ww > 0
        if good.sum() < A.shape[1]:
            break
        sw = np.sqrt(ww[good])
        sol, *_ = np.linalg.lstsq(A[good] * sw[:, None], y[good] * sw, rcond=None)
        coeffs = sol
        if not robust or it == n_iter - 1:
            break
        res = np.angle(np.exp(1j * (y - A @ coeffs)))
        med = np.median(res[good])
        scale = 1.4826 * np.median(np.abs(res[good] - med))
        if not np.isfinite(scale) or scale <= 0:
            break
        u = np.abs(res - med) / (huber * scale)
        ww = np.where(w0 > 0, w0 * np.minimum(1.0, 1.0 / np.maximum(u, 1e-9)), 0.0)

    return PhaseScreen(coeffs, model, slant_range, azimuth,
                       wavelength=wavelength, ramp=ramp, quality=quality)


def remove_screen(phase, screen):
    """Subtract a screen, returning wrapped phase (or a complex interferogram).

    Accepts and returns the same kind of array it was given: real phase in,
    wrapped real phase out; complex interferogram in, complex out with its
    magnitude preserved.
    """
    phase = np.asarray(phase)
    scr = screen.evaluate() if isinstance(screen, PhaseScreen) else np.asarray(screen)
    scr = np.broadcast_to(scr, phase.shape)
    if np.iscomplexobj(phase):
        return phase * np.exp(-1j * scr)
    return np.angle(np.exp(1j * (phase - scr)))


def stable_mask(coherence, threshold=0.6, amplitude=None, amplitude_percentile=None):
    """Pixels trustworthy enough to estimate a screen on.

    Coherence above ``threshold``, optionally further restricted to the
    brightest pixels — on a glacier the bright, coherent returns are rock and
    moraine, which is exactly the ground you want the screen tied to.
    """
    m = np.asarray(coherence, float) >= threshold
    m &= np.isfinite(np.asarray(coherence, float))
    if amplitude is not None and amplitude_percentile is not None:
        a = np.asarray(amplitude, float)
        finite = np.isfinite(a)
        if finite.any():
            m &= finite & (a >= np.percentile(a[finite], amplitude_percentile))
    return m
