"""Phase linking: one phase per epoch from the full N x N coherence matrix.

Classical InSAR reads each pair independently, which throws away the redundancy
in a stack and leaves the network vulnerable to any single bad interferogram.
Phase linking instead fits a *single* phase vector ``theta`` of length N to
**all** N(N-1)/2 observed coherences at once, under the rank-one model

    Gamma_ij  ~=  |Gamma_ij| * exp(i (theta_i - theta_j))

which is exactly GAMMA's ``SLC_intf`` convention (pair ``(i, j)`` carries
``theta_i - theta_j``).  ``theta`` is only determined up to a common additive
constant, so every estimator here returns it referenced to one epoch.

Estimators
----------
``evd``
    Principal eigenvector of ``Gamma``.  The CAESAR estimator of Fornaro et al.
    Cheap, and optimal only when all pairs are equally coherent.
``eigensar``
    EVD-based reconstruction hardened for scenes with very low PS/DS density:
    the coherence matrix is shrunk toward the identity, entries below a
    coherence floor are down-weighted rather than trusted, and the leading
    eigenvector is refined by inverse iteration.  Falls back gracefully where
    the eigen-gap is too small to trust.
``emi``
    Eigendecomposition Maximum-likelihood-estimator of Interferometric phase
    (Ansari et al.).  Eigenvector of the **smallest** eigenvalue of
    ``inv(|Gamma|) * Gamma`` (Hadamard product).  A closed-form relaxation of
    the true ML problem, and much better than EVD when coherence varies.
``mle``
    The exact maximum-likelihood / phase-triangulation solution, reached by
    coordinate descent on the ML cost from an ``emi`` start.  Monotone by
    construction; see :func:`ml_cost`.

All of them accept ``Gamma`` with any number of leading spatial axes —
``(..., N, N)`` in, ``(..., N)`` out — so a whole patch is estimated in one
call.  Memory is the binding constraint: see :mod:`gpri_tools.covariance`.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "phase_link", "evd", "eigensar", "emi", "mle",
    "ml_cost", "temporal_coherence", "coherence_magnitude",
]


# --------------------------------------------------------------------- helpers
def _as_matrix_stack(Gamma):
    Gamma = np.asarray(Gamma)
    if Gamma.ndim < 2 or Gamma.shape[-1] != Gamma.shape[-2]:
        raise ValueError(f"expected (..., N, N) coherence matrices, got {Gamma.shape}")
    return Gamma


def _reference(theta, reference):
    """Rotate a unit-modulus phase vector so ``theta[reference]`` is zero."""
    if reference is None:
        return theta
    ref = theta[..., reference, np.newaxis]
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(np.abs(ref) > 0, ref / np.abs(ref), 1.0)
    return theta * np.conj(ref)


def coherence_magnitude(Gamma, floor=0.0, shrink=0.0):
    """``|Gamma|`` conditioned for inversion.

    ``inv(|Gamma|)`` appears in both :func:`emi` and the ML cost, but a sample
    coherence matrix estimated from fewer looks than epochs is rank deficient
    and ``|Gamma|`` for a perfectly coherent stack is the all-ones matrix,
    which is singular outright.  ``shrink`` mixes toward the identity;
    ``floor`` keeps near-zero coherences from dominating the inverse.
    """
    mag = np.abs(_as_matrix_stack(Gamma))
    if floor > 0.0:
        mag = np.maximum(mag, floor)
    n = mag.shape[-1]
    eye = np.eye(n, dtype=mag.dtype)
    mag = mag * (1.0 - eye) + eye          # unit diagonal regardless of floor
    if shrink > 0.0:
        mag = (1.0 - shrink) * mag + shrink * eye
    return mag


def _safe_inverse(mag, shrink):
    """Invert ``|Gamma|``, escalating the shrinkage until it is well conditioned."""
    n = mag.shape[-1]
    eye = np.eye(n, dtype=mag.dtype)
    for extra in (0.0, 1e-6, 1e-4, 1e-2, 1e-1):
        m = (1.0 - extra) * mag + extra * eye if extra else mag
        try:
            inv = np.linalg.inv(m)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(inv)):
            return inv
    # last resort: pseudo-inverse never raises
    return np.linalg.pinv(mag)


# ------------------------------------------------------------------ estimators
def evd(Gamma, reference=0):
    """Principal-eigenvector (CAESAR) phase estimate.

    Returns
    -------
    theta : array (..., N) complex
        Unit modulus; ``np.angle(theta)`` gives the phase in radians.
    """
    Gamma = _as_matrix_stack(Gamma)
    # eigh returns ascending eigenvalues, so the principal vector is the last
    _, vecs = np.linalg.eigh(Gamma)
    theta = vecs[..., :, -1]
    return _reference(_unit(theta), reference)


def eigensar(Gamma, reference=0, floor=0.2, shrink=1e-3, refine=2,
             min_eigen_gap=0.0):
    """EVD phase reconstruction hardened for low PS/DS density (eigenSAR).

    Three departures from plain :func:`evd`, all aimed at scenes where few
    pixels are reliable — which is the normal case for a glacier surface:

    1. Coherences below ``floor`` are treated as uninformative and pulled up to
       the floor, so a handful of decorrelated pairs cannot steer the
       eigenvector.
    2. The matrix is shrunk toward the identity by ``shrink``, which keeps the
       decomposition stable when the number of looks is below N.
    3. The leading eigenvector is polished by ``refine`` steps of inverse
       iteration, which sharpens it when the top two eigenvalues are close.

    ``min_eigen_gap`` optionally rejects pixels where the rank-one model is not
    supported by the data: if ``(l1 - l2) / l1`` falls below it, that pixel's
    estimate is returned as NaN rather than as confident nonsense.
    """
    Gamma = _as_matrix_stack(Gamma)
    n = Gamma.shape[-1]
    eye = np.eye(n, dtype=Gamma.dtype)

    mag = coherence_magnitude(Gamma, floor=floor)
    phase = _unit_or_zero(Gamma)
    G = mag * phase                                  # re-weighted coherence
    G = (1.0 - shrink) * G + shrink * eye

    vals, vecs = np.linalg.eigh(G)
    theta = vecs[..., :, -1]

    # inverse iteration about the leading eigenvalue sharpens a shallow peak
    for _ in range(max(0, refine)):
        shift = vals[..., -1, np.newaxis, np.newaxis] * (1.0 + 1e-3)
        try:
            theta = np.linalg.solve(G - shift * eye, theta[..., :, np.newaxis])[..., 0]
        except np.linalg.LinAlgError:
            break
        theta = _unit(theta)

    theta = _unit(theta)
    if min_eigen_gap > 0.0:
        l1, l2 = vals[..., -1], vals[..., -2]
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.where(l1 > 0, (l1 - l2) / l1, 0.0)
        theta = np.where((gap >= min_eigen_gap)[..., np.newaxis], theta, np.nan)
    return _reference(theta, reference)


def emi(Gamma, reference=0, floor=0.0, shrink=1e-3):
    """EMI: eigenvector of the smallest eigenvalue of ``inv(|Gamma|) * Gamma``.

    The Hadamard product of a real symmetric matrix with a Hermitian one is
    Hermitian, so this is a well-posed symmetric eigenproblem.  EMI is the
    closed-form relaxation of the ML estimator in :func:`mle` and is usually
    within a few tenths of a radian of it at a fraction of the cost.
    """
    Gamma = _as_matrix_stack(Gamma)
    mag = coherence_magnitude(Gamma, floor=floor, shrink=shrink)
    M = _safe_inverse(mag, shrink) * Gamma
    M = 0.5 * (M + np.conj(np.swapaxes(M, -1, -2)))   # kill round-off asymmetry
    _, vecs = np.linalg.eigh(M)
    return _reference(_unit(vecs[..., :, 0]), reference)   # smallest eigenvalue


def ml_cost(Gamma, theta, floor=0.0, shrink=1e-3):
    """The maximum-likelihood cost minimised by :func:`mle`.

    ``J(theta) = Lambda^H (inv(|Gamma|) * Gamma) Lambda`` with
    ``Lambda = exp(i theta)``.  Real-valued because the matrix is Hermitian.
    Exposed so callers (and the tests) can confirm an estimate actually
    improves the fit.
    """
    Gamma = _as_matrix_stack(Gamma)
    Lam = _unit(np.asarray(theta))
    mag = coherence_magnitude(Gamma, floor=floor, shrink=shrink)
    M = _safe_inverse(mag, shrink) * Gamma
    q = np.einsum("...i,...ij,...j->...", np.conj(Lam), M, Lam)
    return q.real


def mle(Gamma, reference=0, init=None, max_sweeps=10, tol=1e-5,
        floor=0.0, shrink=1e-3):
    """Exact ML phase linking by coordinate descent (phase triangulation).

    Minimises :func:`ml_cost` over unit-modulus ``Lambda``.  Holding every
    other element fixed, the cost seen by element ``n`` is
    ``2 Re(conj(Lambda_n) g_n) + const`` with
    ``g_n = sum_{m != n} M_nm Lambda_m``, which is minimised at
    ``Lambda_n = -g_n / |g_n|``.  Each element update therefore cannot increase
    the cost, so the sweep is monotone and converges.

    Started from :func:`emi` unless ``init`` is given.  One sweep costs N
    mat-vecs, so this is the expensive estimator — use it on a subset of
    epochs, or on pixels that :func:`temporal_coherence` says are worth it.
    """
    Gamma = _as_matrix_stack(Gamma)
    n = Gamma.shape[-1]
    mag = coherence_magnitude(Gamma, floor=floor, shrink=shrink)
    M = _safe_inverse(mag, shrink) * Gamma
    M = 0.5 * (M + np.conj(np.swapaxes(M, -1, -2)))

    Lam = _unit(np.asarray(init)) if init is not None else emi(
        Gamma, reference=None, floor=floor, shrink=shrink)
    Lam = np.array(Lam, dtype=np.complex128, copy=True)

    prev = ml_cost(Gamma, Lam, floor=floor, shrink=shrink)
    for _ in range(max_sweeps):
        for k in range(n):
            # g_k = sum_{m != k} M[k, m] Lambda_m
            g = np.einsum("...m,...m->...", M[..., k, :], Lam) - M[..., k, k] * Lam[..., k]
            mod = np.abs(g)
            with np.errstate(invalid="ignore", divide="ignore"):
                upd = np.where(mod > 0, -g / np.where(mod > 0, mod, 1.0), Lam[..., k])
            Lam[..., k] = upd
        cost = ml_cost(Gamma, Lam, floor=floor, shrink=shrink)
        if np.all(np.abs(prev - cost) <= tol * np.maximum(np.abs(prev), 1e-12)):
            break
        prev = cost
    return _reference(_unit(Lam), reference)


_ESTIMATORS = {"evd": evd, "eigensar": eigensar, "emi": emi, "mle": mle}


def phase_link(Gamma, method="emi", reference=0, **kwargs):
    """Estimate one phase per epoch from a stack of coherence matrices.

    Parameters
    ----------
    Gamma : array (..., N, N) complex
        Hermitian coherence matrices, e.g. from
        :func:`gpri_tools.covariance.coherence_from_slcs`.
    method : {'evd', 'eigensar', 'emi', 'mle'}
    reference : int or None
        Epoch held at zero phase.  ``None`` leaves the arbitrary global phase
        as the estimator produced it.

    Returns
    -------
    theta : array (..., N) complex
        Unit-modulus per-epoch phase.  Take ``np.angle`` for radians, and feed
        straight into :func:`gpri_tools.timeseries.displacement_from_phases`.
    """
    try:
        fn = _ESTIMATORS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_ESTIMATORS)}") from None
    return fn(Gamma, reference=reference, **kwargs)


# -------------------------------------------------------------------- quality
def temporal_coherence(Gamma, theta, mask=None):
    """Goodness of fit of the rank-one model, in ``[0, 1]``.

    The standard phase-linking quality metric: how well the estimated phase
    differences reproduce the observed interferometric phases,

        gamma_temp = | mean_{i != j} exp(i (angle(Gamma_ij) - theta_i + theta_j)) |

    1 means every observed pair is explained exactly.  Values below ~0.6 are
    conventionally masked out; on a decorrelating surface most pixels will be.
    ``mask`` (``(N, N)`` bool, as returned by
    :func:`gpri_tools.covariance.coherence_from_interferograms`) restricts the average
    to pairs that were actually observed.
    """
    Gamma = _as_matrix_stack(Gamma)
    Lam = _unit(np.asarray(theta))
    n = Gamma.shape[-1]

    model = Lam[..., :, np.newaxis] * np.conj(Lam[..., np.newaxis, :])
    resid = _unit_or_zero(Gamma) * np.conj(model)

    off = ~np.eye(n, dtype=bool)
    if mask is not None:
        off = off & np.asarray(mask, bool)
    count = off.sum()
    if count == 0:
        return np.zeros(Gamma.shape[:-2])
    return np.abs(resid[..., off].sum(axis=-1) / count)


# ------------------------------------------------------------------ utilities
def _unit(z):
    """Normalise to unit modulus, leaving exact zeros alone."""
    z = np.asarray(z)
    mod = np.abs(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mod > 0, z / np.where(mod > 0, mod, 1.0), z)


def _unit_or_zero(z):
    """Unit modulus where defined, zero where the input vanished."""
    z = np.asarray(z)
    mod = np.abs(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mod > 0, z / np.where(mod > 0, mod, 1.0), 0.0)
