"""From interferometric phase to a line-of-sight displacement time series.

Sign convention
---------------
Worth being pedantic about, because it is the easiest thing in InSAR to get
backwards and the hardest to notice.

The SLC phase of epoch ``i`` is ``phi_i = -4 pi r_i / lambda`` (two-way path).
GAMMA's ``SLC_intf`` forms the interferogram for pair ``(i, j)`` as
``z_i * conj(z_j)``, so its phase is

    psi_ij = phi_i - phi_j = +4 pi (r_j - r_i) / lambda

A **positive** ``psi`` therefore means the target got **further away** between
epoch ``i`` and epoch ``j``.  Displacement here is reported the conventional
way round — **positive toward the radar** — so

    d_ij = -(lambda / 4 pi) * psi_ij                (:func:`los_displacement`)

and since ``d_ij`` is the motion of ``j`` relative to ``i``, it is exactly the
``d_j - d_i`` quantity that :meth:`gpri.network.Network.design_matrix` builds
its rows for.  That is why :func:`invert_network` takes displacements (or
negated phase) rather than raw phase: converting first makes the design matrix
and the observations agree, instead of hiding a sign flip inside the solver.

Phases coming out of :mod:`gpri.phaselink` are per-epoch ``theta``, already in
the ``psi_ij = theta_i - theta_j`` convention, so
:func:`displacement_from_phases` converts them directly with no inversion at
all — phase linking has done the network inversion implicitly.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "TimeSeries", "los_displacement", "phase_from_los", "displacement_from_phases",
    "invert_network", "stack_velocity", "triplets", "closure_phase",
    "closure_residual_mask", "wrap",
]


def wrap(phase):
    """Wrap to ``(-pi, pi]``."""
    return np.angle(np.exp(1j * np.asarray(phase)))


# ------------------------------------------------------- phase <-> displacement
def los_displacement(phase, wavelength):
    """GAMMA interferometric phase -> LOS displacement, positive toward the radar.

    ``d = -(lambda / 4 pi) * psi``.  For BakerBend1 (Ku band,
    lambda = 1.743 cm) one full fringe is 8.7 mm of range change — which is why
    a GPRI can see millimetre motion, and why the atmosphere in
    :mod:`gpri.atmosphere` is such a problem.

    Give it **unwrapped** phase.  Wrapped phase in gives displacement modulo
    lambda/2 out, which is rarely what you want.
    """
    return -(np.asarray(wavelength, float) / (4.0 * np.pi)) * np.asarray(phase, float)


def phase_from_los(displacement, wavelength):
    """Inverse of :func:`los_displacement`."""
    return -(4.0 * np.pi / np.asarray(wavelength, float)) * np.asarray(displacement, float)


def displacement_from_phases(theta, wavelength, reference=0, axis=-1):
    """Per-epoch phase from :mod:`gpri.phaselink` -> LOS displacement series.

    ``theta`` may be the complex unit-modulus vector the estimators return, or
    real angles.  Displacement is relative to the ``reference`` epoch and
    positive toward the radar:  ``d_i = +(lambda / 4 pi) (theta_i - theta_ref)``.

    The sign is opposite to :func:`los_displacement` only because ``psi_ij``
    is ``theta_i - theta_j`` while ``d_i`` is referenced *from* the reference
    epoch — the two are consistent, and ``tests`` asserts exactly that.
    """
    theta = np.asarray(theta)
    ang = np.angle(theta) if np.iscomplexobj(theta) else theta.astype(float)
    ang = np.moveaxis(ang, axis, -1)
    ang = np.unwrap(ang, axis=-1)
    ang = ang - ang[..., reference, np.newaxis]
    d = (np.asarray(wavelength, float) / (4.0 * np.pi)) * ang
    return np.moveaxis(d, -1, axis)


# ------------------------------------------------------------------- container
class TimeSeries:
    """A LOS displacement time series and the network it came from."""

    def __init__(self, times, displacement, network=None, reference=0,
                 residual=None, quality=None, wavelength=None):
        #: epoch times in days from the first acquisition
        self.times = np.asarray(times, float)
        #: displacement, ``(n_epochs, ...)``, metres, positive toward the radar
        self.displacement = np.asarray(displacement)
        self.network = network
        self.reference = reference
        #: per-pair misfit left over by the inversion, metres
        self.residual = residual
        self.quality = quality
        self.wavelength = wavelength

    @property
    def n_epochs(self):
        return self.displacement.shape[0]

    def velocity(self, weights=None):
        """Least-squares linear rate through the series, metres per day."""
        t = self.times - self.times.mean()
        d = self.displacement.reshape(self.n_epochs, -1)
        w = np.ones_like(t) if weights is None else np.asarray(weights, float)
        denom = np.sum(w * t * t)
        if denom <= 0:
            return np.zeros(self.displacement.shape[1:])
        v = (w * t) @ np.nan_to_num(d) / denom
        return v.reshape(self.displacement.shape[1:])

    def rms_residual(self):
        if self.residual is None:
            return None
        return np.sqrt(np.nanmean(np.asarray(self.residual) ** 2, axis=0))

    def __repr__(self):
        span = self.times[-1] - self.times[0] if len(self.times) else 0.0
        return (f"TimeSeries({self.n_epochs} epochs, {span:.3f} d, "
                f"pixels={self.displacement.shape[1:]})")


# ------------------------------------------------------------------- inversion
def invert_network(observations, network, weights=None, method="lstsq",
                   reference=0, smoothing=0.0, iterations=10, rcond=None,
                   wavelength=None, incremental=False):
    """Solve the SBAS system for per-epoch displacement.

    Parameters
    ----------
    observations : array (n_pairs, ...)
        One value per interferogram, in the ``d_j - d_i`` convention — i.e.
        **LOS displacement**, from :func:`los_displacement` applied to unwrapped
        phase.  Passing raw phase inverts the sign of the answer.
    network : :class:`gpri.network.Network`
    weights : array, optional
        ``(n_pairs,)`` for weights shared by every pixel — one factorisation,
        fast.  ``(n_pairs, ...)`` for per-pixel weights, which needs one solve
        per pixel and is far slower; use it only on a masked subset.
    method : {'lstsq', 'wls', 'l1', 'smooth'}
        ``lstsq``  minimum-norm least squares (SVD); tolerates a rank-deficient
        or disconnected network.
        ``wls``    weighted least squares; identical to ``lstsq`` when
        ``weights`` is None.
        ``l1``     IRLS approximation to an L1 fit — the one to reach for when
        you suspect unwrapping errors, since it will not let a single bad pair
        smear across the series.
        ``smooth`` Tikhonov regularisation penalising the second time
        difference, i.e. preferring a smooth velocity.  Needs ``smoothing > 0``.
    reference : int
        Epoch pinned to zero displacement.
    incremental : bool
        Solve for increments between consecutive epochs rather than for
        displacement relative to ``reference``.  Better conditioned on a daisy
        chain, which is what the BakerBend1 ``itab`` is.

    Returns
    -------
    ts : :class:`TimeSeries`
    """
    obs = np.asarray(observations, float)
    if obs.shape[0] != network.n_pairs:
        raise ValueError(
            f"{obs.shape[0]} observations but the network has {network.n_pairs} pairs")
    if not network.is_connected():
        warnings.warn(
            f"network has {len(network.components())} disconnected components; "
            "displacement is only determined within each one",
            stacklevel=2)

    G = (network.incremental_design_matrix() if incremental
         else network.design_matrix(reference))
    spatial = obs.shape[1:]
    Y = obs.reshape(obs.shape[0], -1)
    npix = Y.shape[1]
    finite = np.isfinite(Y)
    Y = np.nan_to_num(Y)

    if method == "smooth" and smoothing > 0:
        G, Y, finite = _append_smoothing(G, Y, finite, network, smoothing, incremental)

    w = None if weights is None else np.asarray(weights, float)
    per_pixel = w is not None and w.ndim > 1

    if method in ("lstsq", "wls", "smooth") and not per_pixel:
        X, resid = _solve_shared(G, Y, w, finite, rcond)
    elif method == "l1":
        X, resid = _solve_l1(G, Y, w, finite, iterations, rcond, per_pixel)
    else:
        X, resid = _solve_per_pixel(G, Y, w, finite, rcond)

    D = _expand(X, network, reference, incremental)
    D = D.reshape((network.n_epochs,) + spatial)
    if resid is not None:
        # 'smooth' appends penalty rows to G; those are not observations
        resid = resid[:network.n_pairs].reshape((network.n_pairs,) + spatial)
    return TimeSeries(network.times, D, network=network, reference=reference,
                      residual=resid, wavelength=wavelength)


def _append_smoothing(G, Y, finite, network, smoothing, incremental):
    """Stack a second-difference penalty under the design matrix."""
    m = G.shape[1]
    if m < 3:
        return G, Y, finite
    L = np.zeros((m - 2, m))
    for k in range(m - 2):
        L[k, k], L[k, k + 1], L[k, k + 2] = 1.0, -2.0, 1.0
    G2 = np.vstack([G, smoothing * L])
    Y2 = np.vstack([Y, np.zeros((m - 2, Y.shape[1]))])
    f2 = np.vstack([finite, np.ones((m - 2, Y.shape[1]), bool)])
    return G2, Y2, f2


def _solve_shared(G, Y, w, finite, rcond):
    """One factorisation for every pixel — the fast path."""
    if w is None:
        A, B = G, Y
    else:
        s = np.sqrt(np.maximum(np.asarray(w, float).ravel(), 0.0))
        if s.size != G.shape[0]:
            s = np.concatenate([s, np.ones(G.shape[0] - s.size)])   # smoothing rows
        A, B = G * s[:, None], Y * s[:, None]
    X, *_ = np.linalg.lstsq(A, B, rcond=rcond)
    resid = np.where(finite, Y - G @ X, np.nan)
    return X, resid


def _solve_per_pixel(G, Y, w, finite, rcond):
    """Per-pixel weights: normal equations, solved as a stack."""
    m, npix = G.shape[1], Y.shape[1]
    nrow = G.shape[0]
    if w is None:
        W = np.ones((nrow, npix))
    else:
        W = np.maximum(np.asarray(w, float).reshape(-1, npix), 0.0)
        if W.shape[0] < nrow:      # smoothing rows were appended to G
            W = np.vstack([W, np.ones((nrow - W.shape[0], npix))])
    W = np.where(finite, W, 0.0)
    need = npix * m * m * 8 / 2 ** 30
    if need > 4.0:
        raise MemoryError(
            f"per-pixel weighting needs {need:.1f} GiB for {npix} pixels x {m}^2 "
            "normal equations; mask down to the pixels you care about, or use "
            "shared (1-D) weights")
    GtG = np.einsum("pm,pn,pk->kmn", G, G, W, optimize=True)
    Gty = np.einsum("pm,pk,pk->km", G, W, Y, optimize=True)
    eps = 1e-10 * np.trace(GtG, axis1=1, axis2=2)[:, None, None] * np.eye(m)
    X = np.linalg.solve(GtG + eps, Gty[..., None])[..., 0].T
    resid = np.where(finite, Y - G @ X, np.nan)
    return X, resid


def _solve_l1(G, Y, w, finite, iterations, rcond, per_pixel, tol=1e-10):
    """L1 fit by iteratively reweighted least squares.

    Reweighting by ``1 / |residual|`` drives the L2 solution toward the L1 one,
    which is what makes it shrug off an interferogram carrying a whole-cycle
    unwrapping error instead of spreading that error over every epoch.

    The residual floor is fixed to a robust scale taken from the **initial**
    least-squares fit rather than recomputed each sweep.  That matters: as the
    good pairs converge their residuals go to zero, and a floor that shrinks
    with them sends the weights to infinity and the normal equations to
    garbage.  Pinning it makes the iteration stable for any ``iterations``.
    """
    X, resid = _solve_shared(G, Y, None if per_pixel else w, finite, rcond)
    base = (np.ones(G.shape[0]) if w is None or per_pixel
            else np.asarray(w, float).ravel())
    if base.size != G.shape[0]:
        base = np.concatenate([base, np.ones(G.shape[0] - base.size)])

    r0 = np.abs(np.nan_to_num(resid))
    nz = r0[r0 > 0]
    scale = float(np.median(nz)) if nz.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    floor = 1e-3 * scale                      # fixed, so weights stay bounded

    for _ in range(max(0, iterations - 1)):
        r = np.abs(np.nan_to_num(resid))
        ww = np.where(finite, base[:, None] / np.maximum(r, floor), 0.0)
        X_new, resid = _solve_per_pixel(G, Y, ww, finite, rcond)
        step = np.max(np.abs(X_new - X)) if X.size else 0.0
        X = X_new
        if step <= tol * max(float(np.max(np.abs(X))), 1e-300):
            break
    return X, resid


def _expand(X, network, reference, incremental):
    """Put the reference epoch back in, or integrate increments."""
    n = network.n_epochs
    if incremental:
        D = np.zeros((n, X.shape[1]))
        D[1:] = np.cumsum(X, axis=0)
        return D - D[reference][None, :]
    D = np.zeros((n, X.shape[1]))
    cols = [e for e in range(n) if e != reference]
    D[cols] = X
    return D


# --------------------------------------------------------------------- stacking
def stack_velocity(observations, network, weights=None, min_pairs=2):
    """Coherence-weighted stacking: a single rate, no time series.

    Fits ``d_p = v * dt_p`` across every pair by weighted least squares, giving

        v = sum(w dt d) / sum(w dt^2)

    Stacking is the right tool when you want a robust mean rate and do not care
    about the shape of the time series: it needs no network connectivity, no
    unwrapping across epochs, and averages down atmospheric noise as
    ``1/sqrt(n_pairs)`` because the screens are uncorrelated between pairs while
    the signal accumulates with ``dt``.

    Parameters
    ----------
    observations : array (n_pairs, ...)
        Pair LOS displacement, from :func:`los_displacement`.
    weights : array, optional
        ``(n_pairs,)`` or ``(n_pairs, ...)`` — coherence is the usual choice.
    min_pairs : int
        Pixels contributing fewer than this many finite, positively weighted
        pairs come back as NaN.

    Returns
    -------
    velocity : array (...)
        Metres per day, positive toward the radar.
    """
    obs = np.asarray(observations, float)
    if obs.shape[0] != network.n_pairs:
        raise ValueError(
            f"{obs.shape[0]} observations but the network has {network.n_pairs} pairs")
    dt = network.temporal_baselines()
    shape = (network.n_pairs,) + (1,) * (obs.ndim - 1)
    dt_b = dt.reshape(shape)

    w = np.ones(shape) if weights is None else np.asarray(weights, float)
    w = np.broadcast_to(np.reshape(w, shape) if w.ndim == 1 else w, obs.shape)
    good = np.isfinite(obs) & np.isfinite(w) & (w > 0) & (dt_b != 0)
    w = np.where(good, w, 0.0)
    o = np.nan_to_num(obs)

    num = np.sum(w * dt_b * o, axis=0)
    den = np.sum(w * dt_b ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(den > 0, num / den, np.nan)
    return np.where(good.sum(axis=0) >= min_pairs, v, np.nan)


# ---------------------------------------------------------------- closure phase
def triplets(network, max_triplets=None):
    """Every ``(ij, jk, ik)`` triangle present in the network.

    Returns an ``(n_triplets, 3)`` array of **pair** indices.  A pure daisy
    chain — which is what BakerBend1's ``itab_mr`` is — contains no triangles at
    all, so this returns empty; add the ``i -> i+2`` pairs to get closure.
    """
    index = {(int(i), int(j)): p for p, (i, j) in enumerate(network.pairs)}
    out = []
    for (i, j), p_ij in index.items():
        for (j2, k), p_jk in index.items():
            if j2 != j or k == i:
                continue
            p_ik = index.get((i, k))
            if p_ik is not None:
                out.append((p_ij, p_jk, p_ik))
                if max_triplets and len(out) >= max_triplets:
                    return np.asarray(out, int)
    return np.asarray(out, int).reshape(-1, 3)


def closure_phase(pair_phase, network=None, trip=None):
    """Wrapped closure ``psi_ij + psi_jk - psi_ik`` for each triangle.

    Identically zero for consistent phase, because
    ``(theta_i - theta_j) + (theta_j - theta_k) - (theta_i - theta_k) = 0``.
    What is left is unwrapping error, or a genuine physical closure bias — the
    systematic, non-closing phase that soil moisture and short-baseline
    decorrelation introduce and that biases any time series built from short
    pairs.

    ``pair_phase`` must be **phase**, not displacement, and is used wrapped.
    """
    if trip is None:
        if network is None:
            raise ValueError("need a network, or an explicit triplet array")
        trip = triplets(network)
    trip = np.asarray(trip, int).reshape(-1, 3)
    p = np.asarray(pair_phase)
    if trip.size == 0:
        return np.zeros((0,) + p.shape[1:])
    return wrap(p[trip[:, 0]] + p[trip[:, 1]] - p[trip[:, 2]])


def closure_residual_mask(pair_phase, network=None, trip=None, threshold=1.0):
    """Pixels whose closure phase stays below ``threshold`` radians everywhere.

    A cheap, unwrapping-free reliability mask: a pixel that closes across every
    triangle it takes part in is one whose phase you can trust.
    """
    c = closure_phase(pair_phase, network, trip)
    if c.shape[0] == 0:
        return np.ones(np.shape(pair_phase)[1:], bool)
    return np.all(np.abs(c) <= threshold, axis=0)


# ------------------------------------------------------- common-mode removal
def reference_to_stable(displacement, reference_mask, method="median",
                        return_offset=False):
    """Remove the scene-wide common mode by tying every epoch to stable ground.

    **This is not optional, and leaving it out quietly ruins a time series.**

    An interferogram determines phase only up to an additive constant: nothing
    in ``z_i * conj(z_j)`` fixes the absolute phase.  Each pair therefore
    carries its own arbitrary offset, and integrating a network — a cumulative
    sum, for the daisy chain in BakerBend1 — accumulates 722 of them into a
    scene-wide drift.  Any range-independent atmospheric term the screen model
    could not absorb lands in the same place.

    The result looks exactly like a signal: spatially smooth, temporally
    coherent, and on a mountain flank it is *diurnal*, because that is what the
    atmosphere does.  It appears on bedrock at the same amplitude and phase as
    on ice, which is the tell — and the reason
    :func:`gpri.diurnal.stable_ground_null` exists.

    The fix is to subtract, at each epoch, a robust average over ground known
    not to be moving.  Displacement is then relative to that ground, which is
    what a reader assumes it already was.

    Parameters
    ----------
    displacement : array (n_epochs, ...)
    reference_mask : bool array
        Stable ground — bedrock, moraine.  High mean coherence is the usual
        proxy.
    method : {'median', 'mean'}
        Median by default: a few moving pixels leaking into the mask should not
        drag the reference with them.

    Returns
    -------
    referenced : array
        Same shape, with the per-epoch common mode removed.
    offset : (n_epochs,) array
        Only if ``return_offset``.  What was subtracted — plot it, because it
        *is* the common-mode signal and is worth looking at in its own right.

    Notes
    -----
    If you then run a null test on stable ground, hold out reference pixels
    from it.  Testing on the same pixels used to reference is circular: they
    were forced to zero by construction.
    """
    d = np.asarray(displacement, float)
    m = np.asarray(reference_mask, bool)
    if m.shape != d.shape[1:]:
        raise ValueError(f"reference mask {m.shape} does not match the "
                         f"spatial shape {d.shape[1:]}")
    if not m.any():
        raise ValueError("reference mask selects no pixels; there is nothing "
                         "to tie the series to")

    flat = d.reshape(d.shape[0], -1)[:, m.ravel()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN epochs
        agg = np.nanmedian(flat, axis=1) if method == "median" \
            else np.nanmean(flat, axis=1)
    agg = np.nan_to_num(agg)
    out = d - agg.reshape((-1,) + (1,) * (d.ndim - 1))
    return (out, agg) if return_offset else out


__all__.append("reference_to_stable")
