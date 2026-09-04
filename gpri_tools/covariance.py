"""Sample covariance / coherence matrices for a stack of coregistered scenes.

Phase-linking estimators work on the full N x N complex coherence matrix at each
output pixel, so this module is where the memory goes: one matrix costs
``N^2 * 16`` bytes, which for the 723-epoch BakerBend1 stack is 8.4 MB *per
pixel*.  Both entry points therefore multilook onto a coarse output grid and
accept an epoch subset, and both refuse to silently allocate more than
``max_gib``.

Two sources are supported behind the same output convention:

* :func:`coherence_from_slcs` — the real thing, a spatially averaged sample
  covariance.  Needed for genuine phase linking.
* :func:`coherence_from_interferograms` — assembled from interferograms GAMMA
  has already formed.  Only the pairs present in the network are filled; the
  rest are marked missing so estimators can down-weight them.
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    from scipy.ndimage import uniform_filter
except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
    uniform_filter = None


def _boxcar(a, window):
    """Separable boxcar mean over the trailing two axes."""
    if uniform_filter is not None:
        return uniform_filter(a.real, size=window, mode="nearest") + (
            1j * uniform_filter(a.imag, size=window, mode="nearest")
            if np.iscomplexobj(a) else 0.0
        )
    # cumulative-sum fallback
    wa, wr = window
    k = np.ones((wa, wr), a.dtype) / (wa * wr)
    from numpy.fft import irfft2, rfft2
    return irfft2(rfft2(a, a.shape) * rfft2(k, a.shape), a.shape)


def _check_budget(n_out, n_epochs, max_gib, what):
    need = n_out * n_epochs * n_epochs * 16 / 2**30
    if need > max_gib:
        raise MemoryError(
            f"{what} would need {need:.1f} GiB for {n_out} output pixels x "
            f"{n_epochs} epochs (limit {max_gib} GiB). Increase `looks`, pass a "
            f"smaller `epochs` subset, or process in patches."
        )
    return need


def coherence_from_slcs(slcs, looks=(10, 10), stride=None, epochs=None,
                        normalize=True, max_gib=4.0):
    """Spatially averaged sample coherence matrix from a stack of SLCs.

    Parameters
    ----------
    slcs : array (N, A, R) complex
        Coregistered single-look complex scenes.  GPRI is tripod-mounted, so
        scenes from one deployment are already coregistered.
    looks : (int, int)
        Boxcar window in (azimuth, range) used to estimate each ensemble average.
    stride : (int, int), optional
        Output grid step.  Defaults to ``looks`` (non-overlapping, the usual
        multilooking convention).
    epochs : sequence of int, optional
        Subset of scenes to use.
    normalize : bool
        Divide by ``sqrt(C_ii C_jj)`` to get coherence rather than covariance.

    Returns
    -------
    Gamma : array (na, nr, N, N) complex
        Hermitian, unit diagonal when ``normalize``.
    """
    slcs = np.asarray(slcs)
    if slcs.ndim != 3:
        raise ValueError(f"expected (N, A, R) stack, got shape {slcs.shape}")
    if epochs is not None:
        slcs = slcs[np.asarray(epochs, int)]
    n, a, r = slcs.shape
    stride = tuple(stride or looks)

    sl = (slice(looks[0] // 2, a, stride[0]), slice(looks[1] // 2, r, stride[1]))
    na = len(range(*sl[0].indices(a)))
    nr = len(range(*sl[1].indices(r)))
    _check_budget(na * nr, n, max_gib, "coherence_from_slcs")

    Gamma = np.zeros((na, nr, n, n), np.complex64)
    # diagonal first: intensities, reused for normalisation
    power = np.empty((n, na, nr), np.float32)
    for i in range(n):
        power[i] = _boxcar(np.abs(slcs[i]) ** 2, looks).real[sl]

    for i in range(n):
        Gamma[:, :, i, i] = 1.0 if normalize else power[i]
        for j in range(i + 1, n):
            cross = _boxcar(slcs[i] * np.conj(slcs[j]), looks)[sl]
            if normalize:
                denom = np.sqrt(power[i] * power[j])
                with np.errstate(invalid="ignore", divide="ignore"):
                    cross = np.where(denom > 0, cross / denom, 0)
            Gamma[:, :, i, j] = cross
            Gamma[:, :, j, i] = np.conj(cross)
    return Gamma


def coherence_from_interferograms(ifgs, pairs, n_epochs, coherence=None):
    """Assemble a coherence matrix from interferograms GAMMA already formed.

    Only network pairs are observed, so the returned matrix is generally sparse
    off the diagonal.  ``mask`` reports which entries are real observations —
    estimators must not treat the unobserved zeros as measured decorrelation.

    Parameters
    ----------
    ifgs : array (P, ...) complex
        One interferogram per pair, conjugate convention ``z_ref * conj(z_sec)``
        as produced by GAMMA's ``SLC_intf``.
    pairs : array (P, 2) int
        0-based epoch indices, e.g. from :func:`gpri_tools.network.read_itab`.
    coherence : array (P, ...) float, optional
        Per-pair coherence magnitude.  If omitted, magnitudes come from ``ifgs``.

    Returns
    -------
    Gamma : array (..., N, N) complex
    mask : array (N, N) bool
        True where the pair was actually observed.
    """
    ifgs = np.asarray(ifgs)
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if len(pairs) != len(ifgs):
        raise ValueError(f"{len(ifgs)} interferograms but {len(pairs)} pairs")

    spatial = ifgs.shape[1:]
    Gamma = np.zeros(spatial + (n_epochs, n_epochs), np.complex64)
    mask = np.zeros((n_epochs, n_epochs), bool)
    idx = np.arange(n_epochs)
    Gamma[..., idx, idx] = 1.0
    mask[idx, idx] = True

    for p, (i, j) in enumerate(pairs):
        z = ifgs[p]
        mag = np.abs(z)
        if coherence is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.where(mag > 0, z / mag, 0) * coherence[p]
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.where(mag > 0, z / mag, 0)
        Gamma[..., i, j] = z
        Gamma[..., j, i] = np.conj(z)
        mask[i, j] = mask[j, i] = True

    if mask.sum() < n_epochs * n_epochs:
        frac = mask.sum() / n_epochs ** 2
        warnings.warn(
            f"coherence matrix is {frac:.1%} observed; phase-linking estimators "
            "assume a full matrix. Build from SLCs where possible.",
            stacklevel=2,
        )
    return Gamma, mask


def regularize(Gamma, epsilon=1e-3):
    """Nudge a coherence matrix toward positive definiteness.

    Sample matrices formed from fewer looks than epochs are rank deficient, and
    EMI needs to invert ``|Gamma|``.  Shrinks toward the identity.
    """
    n = Gamma.shape[-1]
    eye = np.eye(n, dtype=Gamma.dtype)
    return (1.0 - epsilon) * Gamma + epsilon * eye
