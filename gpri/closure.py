"""Closure-phase bias: estimating it, and what it can and cannot fix.

Around every triangle of epochs ``(i, j, k)`` the interferometric phases ought
to sum to nothing:

    Xi_ijk = psi_ij + psi_jk - psi_ik
           = (th_i - th_j) + (th_j - th_k) - (th_i - th_k) = 0

Any per-epoch phase, however wrong, cancels.  So a non-zero closure phase is
proof that the pairwise phases are **not** differences of a single per-epoch
quantity — the interferograms carry something that is not epoch-separable.

Two things do that.  Unwrapping errors, which are multiples of ``2 pi`` and are
handled by :func:`gpri.timeseries.closure_residual_mask`.  And a genuine
physical bias: short-baseline interferograms are systematically biased by
changes in the scattering medium itself (De Zan et al. 2014; Zheng, Zebker &
Michaelides 2022), so that a chain of short pairs does not sum to the long pair
that spans it.  On a glacier the medium is snow — its wetness and density
change through the day, and at the four-minute cadence here almost every pair
*is* a short-baseline pair.  Left alone, this bias accumulates through the time
series and looks exactly like motion.

The model
---------
To first order the bias depends only on how long the pair spans, not on when:

    psi_observed(i, j) = psi_true(i, j) + b(dt_ij)

Substituting into the closure gives a linear system in ``b``:

    Xi_ijk = b(dt_ij) + b(dt_jk) - b(dt_ik)

which :func:`estimate_bias` solves for ``b`` sampled on temporal-baseline bins.

The one thing this cannot do
----------------------------
That system has a null space, and it is exactly one dimensional:
``b(dt) = c * dt``.  Because ``dt_ij + dt_jk = dt_ik``, a bias proportional to
the temporal baseline closes perfectly and leaves **no trace in any closure
phase**.  But a bias proportional to ``dt`` is precisely a constant velocity
offset.

So: closure phase can tell you the *nonlinear-in-time* part of the bias, and
:func:`estimate_bias` returns that part.  It can never tell you whether your
mean velocity is biased.  Anyone claiming a closure correction validated their
rate is mistaken, and this module refuses to pretend otherwise —
:attr:`BiasModel.velocity_blind` says so in the object itself.
"""
from __future__ import annotations

import numpy as np

from .timeseries import closure_phase, triplets, wrap

__all__ = [
    "BiasModel", "baseline_bins", "closure_design_matrix", "estimate_bias",
    "correct_bias", "closure_rms",
]


# ---------------------------------------------------------------- binning
def baseline_bins(network, bins=None, n_bins=None, tolerance=1e-6):
    """Assign each pair to a temporal-baseline bin.

    GPRI acquires on a fixed cadence, so temporal baselines come out quantised
    — at BakerBend1 every pair is an integer number of ~4-minute steps.  When
    that is the case (few enough distinct baselines) each distinct baseline
    gets its own bin, which is both exact and what the physics wants.
    Otherwise the baselines are cut into ``n_bins`` quantile bins.

    Returns
    -------
    index : (n_pairs,) int
    centers : (n_bins,) float
        Representative baseline for each bin, in days.
    """
    dt = np.abs(network.temporal_baselines())
    if bins is not None:
        edges = np.asarray(bins, float)
        index = np.clip(np.digitize(dt, edges) - 1, 0, len(edges) - 2)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return index, centers

    # Group by a rounded key, but keep the *exact* mean baseline as the centre.
    # Rounding the centres themselves would be a quiet disaster: the null-space
    # argument in the module docstring needs dt_ij + dt_jk = dt_ik to hold to
    # machine precision, and centres snapped to a 1e-6 d grid break it at 1e-6,
    # which is enough to leak a spurious velocity back into the fit.
    key = np.round(dt / tolerance).astype(np.int64)
    uniq = np.unique(key)
    if n_bins is None and len(uniq) <= max(2, len(dt) // 2):
        index = np.searchsorted(uniq, key)
        centers = np.array([dt[index == m].mean() for m in range(len(uniq))])
        return index, centers

    n_bins = n_bins or min(20, max(2, len(uniq)))
    edges = np.quantile(dt, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(len(dt), int), np.array([dt.mean() if dt.size else 0.0])
    index = np.clip(np.digitize(dt, edges) - 1, 0, len(edges) - 2)
    centers = np.array([dt[index == m].mean() if np.any(index == m)
                        else 0.5 * (edges[m] + edges[m + 1])
                        for m in range(len(edges) - 1)])
    return index, centers


def closure_design_matrix(index, trip, n_bins=None):
    """Map a per-bin bias vector onto the closure phase of each triangle.

    Row ``t`` for triangle ``(p_ij, p_jk, p_ik)`` is ``+1`` in the bins of the
    two short legs and ``-1`` in the bin of the long one, summed — so a triangle
    whose legs share a bin gets ``+2`` or ``+1`` there rather than three
    separate entries.
    """
    index = np.asarray(index, int)
    trip = np.asarray(trip, int).reshape(-1, 3)
    n_bins = int(index.max()) + 1 if n_bins is None else int(n_bins)
    C = np.zeros((len(trip), n_bins))
    rows = np.arange(len(trip))
    np.add.at(C, (rows, index[trip[:, 0]]), 1.0)
    np.add.at(C, (rows, index[trip[:, 1]]), 1.0)
    np.add.at(C, (rows, index[trip[:, 2]]), -1.0)
    return C


# -------------------------------------------------------------- the estimate
class BiasModel:
    """A closure-phase bias curve ``b(dt)``, and its honest limits."""

    #: Closure phase is identically blind to a bias linear in temporal
    #: baseline, which is exactly a constant velocity.  Always ``True``.
    velocity_blind = True

    def __init__(self, bias, centers, index, n_triplets, residual_rms=None,
                 wavelength=None):
        #: bias per baseline bin, radians; shape ``(n_bins, ...)``
        self.bias = np.asarray(bias, float)
        #: bin centre temporal baselines, days
        self.centers = np.asarray(centers, float)
        #: bin index of each pair
        self.index = np.asarray(index, int)
        self.n_triplets = int(n_triplets)
        #: closure phase still unexplained after the fit, radians
        self.residual_rms = residual_rms
        self.wavelength = wavelength

    @property
    def n_bins(self):
        return self.bias.shape[0]

    def per_pair(self):
        """The bias of every pair, in the pair ordering of the network."""
        return self.bias[self.index]

    def displacement(self):
        """The bias expressed as LOS displacement, metres.  Needs a wavelength."""
        if self.wavelength is None:
            raise ValueError("wavelength not known for this bias model")
        from .timeseries import los_displacement
        return los_displacement(self.bias, self.wavelength)

    def __repr__(self):
        rms = "?" if self.residual_rms is None else f"{np.mean(self.residual_rms):.3f}"
        peak = np.nanmax(np.abs(self.bias)) if self.bias.size else 0.0
        return (f"BiasModel({self.n_bins} bins, {self.n_triplets} triplets, "
                f"peak={peak:.3f} rad, closure residual rms={rms} rad, "
                f"velocity_blind=True)")


def _project_out_linear(C, centers):
    """Remove the ``b ~ dt`` null direction from the solution space.

    ``C @ centers`` is zero by construction (``dt_ij + dt_jk = dt_ik``), so the
    least-squares problem cannot see that direction at all.  Rather than let
    ``lstsq`` pick an arbitrary point on the null line, pin the answer to the
    one orthogonal to it: the bias with no linear-in-baseline component.  That
    is the only choice that does not quietly inject a velocity.
    """
    v = np.asarray(centers, float)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        return np.eye(C.shape[1])
    v = v / nrm
    return np.eye(C.shape[1]) - np.outer(v, v)


def estimate_bias(pair_phase, network, trip=None, bins=None, n_bins=None,
                  weights=None, robust=0, rcond=None, wavelength=None):
    """Fit ``b(dt)`` to the observed closure phases.

    Parameters
    ----------
    pair_phase : array (n_pairs, ...)
        Interferometric **phase** (not displacement).  Used wrapped, so it does
        not need unwrapping — which is the point: the bias is estimable before
        any unwrapping decision has been made.
    network : :class:`gpri.network.Network`
    trip : (n_triplets, 3) int, optional
        Triangles to use.  Defaults to every triangle in the network.  A pure
        daisy chain has none — add the ``i -> i+2`` pairs to get closure.
    bins, n_bins : see :func:`baseline_bins`
    weights : (n_triplets,) or (n_triplets, ...) array, optional
        Down-weight triangles you trust less.
    robust : int
        Iterations of Huber reweighting against closure outliers.  A few help
        when some triangles carry unwrapping errors rather than physical bias.

    Returns
    -------
    :class:`BiasModel`

    Raises
    ------
    ValueError
        If the network has no closed triangles — in which case the bias is not
        estimable at all, and saying so beats returning zeros.
    """
    if trip is None:
        trip = triplets(network)
    trip = np.asarray(trip, int).reshape(-1, 3)
    if trip.size == 0:
        raise ValueError(
            "no closed triangles in this network, so closure bias is not "
            "estimable; a daisy chain needs the i->i+2 pairs adding")

    index, centers = baseline_bins(network, bins=bins, n_bins=n_bins)
    C = closure_design_matrix(index, trip, n_bins=len(centers))
    P = _project_out_linear(C, centers)
    Cp = C @ P

    Xi = closure_phase(np.asarray(pair_phase), network=network, trip=trip)
    lead = Xi.shape[1:]
    Y = Xi.reshape(len(trip), -1)

    w = None
    if weights is not None:
        w = np.asarray(weights, float)
        w = np.broadcast_to(w.reshape(len(trip), -1) if w.ndim > 1
                            else w[:, None], Y.shape)

    w0 = w
    beta = _wlstsq(Cp, Y, w, rcond)
    for _ in range(int(robust)):
        r = Y - Cp @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)
        s = np.where(s > 0, s, 1.0)
        hub = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-12))
        # recompute from the *original* weights each sweep; multiplying the
        # previous iteration's weights compounds the down-weighting and walks
        # good triangles out of the fit entirely
        w = hub if w0 is None else w0 * hub
        beta = _wlstsq(Cp, Y, w, rcond)

    resid = Y - Cp @ beta
    rms = np.sqrt(np.mean(resid ** 2, axis=0)).reshape(lead)

    bias = (P @ beta).reshape((len(centers),) + lead)
    return BiasModel(bias, centers, index, len(trip),
                     residual_rms=rms if lead else float(rms),
                     wavelength=wavelength)


def _wlstsq(A, Y, w, rcond):
    """Least squares, shared design if unweighted, per-column if weighted."""
    if w is None:
        return np.linalg.lstsq(A, Y, rcond=rcond)[0]
    out = np.empty((A.shape[1], Y.shape[1]))
    for c in range(Y.shape[1]):
        sw = np.sqrt(np.maximum(w[:, c], 0.0))
        out[:, c] = np.linalg.lstsq(A * sw[:, None], Y[:, c] * sw, rcond=rcond)[0]
    return out


# -------------------------------------------------------------- application
def correct_bias(pair_phase, model, wrap_result=True):
    """Subtract the fitted bias from each pair's phase.

    Returns wrapped phase by default, since that is what the input was.  Pass
    ``wrap_result=False`` if you are correcting already-unwrapped phase.
    """
    p = np.asarray(pair_phase)
    b = model.per_pair()
    if b.ndim < p.ndim:
        b = b.reshape(b.shape + (1,) * (p.ndim - b.ndim))
    if np.iscomplexobj(p):
        return p * np.exp(-1j * b)
    out = p - b
    return wrap(out) if wrap_result else out


def closure_rms(pair_phase, network=None, trip=None):
    """Root-mean-square closure phase, radians — the before/after number.

    Report it either side of :func:`correct_bias`.  If the fit did anything,
    this drops; if it did not, say so rather than showing the corrected time
    series alone.
    """
    c = closure_phase(pair_phase, network=network, trip=trip)
    if c.shape[0] == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(c, float) ** 2)))
