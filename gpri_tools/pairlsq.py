"""Single-step weighted least squares: interferograms straight to a model.

After the estimation approach of Ohenhen et al.'s subsidence mapping
(Ohenhen, Shirzaei et al., e.g. *Hidden vulnerability of US Atlantic coasts*,
2023; *Disappearing cities on US coasts*, 2024): a parametric temporal model —
secular rate plus sinusoids — fitted per pixel by weighted least squares, with
formal parameter uncertainties carried through so every map of amplitude comes
with a map of its standard error.  Adapted to GPRI in two ways: the periods
are diurnal rather than annual, and the model is fitted **directly to the
pair observations** rather than to an integrated time series.

Why single-step beats integrate-then-fit
----------------------------------------
The pipeline so far integrates the network (a cumulative sum, for a daisy
chain) and fits harmonics to the result.  Three things are statistically wrong
with that, all fixed here:

1. **Integration correlates the noise.**  Independent per-pair errors become a
   random walk, and ordinary least squares on a random walk is not the best
   estimator — the early epochs are quiet and the late ones noisy, and OLS
   weights them equally.  Fitting the *differences*, whose errors are
   independent, is the correctly whitened problem.
2. **Per-pair quality is usable.**  A cumulative sum has no way to down-weight
   a low-coherence pair; the pair-domain fit takes one weight per
   interferogram.
3. **The uncertainties are real.**  OLS error bars on an integrated series
   assume white noise and are badly optimistic on a random walk.  Here the
   parameter covariance ``sigma^2 (G^T W G)^-1`` is exact under the stated
   model, so an amplitude map comes with an honest SNR map — and a "diurnal
   detection" can be required to mean ``amplitude > 3 sigma``, per pixel,
   rather than eyeballed against a bedrock ratio.

One more structural gain: the constant term of the model **cancels in the
differencing**, so no reference epoch, no reference pixel, and no
connectedness requirement — a network split into disconnected components
still constrains rate and harmonics, because every component sees the same
clock.  Only the absolute offset is lost, and it was never observable
interferometrically anyway.

What the error bars will tell you, bluntly
------------------------------------------
On a pure short-pair chain the correct variance is often *large*: a pair
spanning ``dt`` sees only ``A * omega * dt`` of a smooth harmonic — 3.5e-4 of
a 1 cm diurnal at 8-minute pairs — so most of the sensitivity to slow signals
lives in the long-baseline combinations a daisy chain does not have.  The
single-step fit does not beat that physics; it *reports* it, where the
integrate-then-fit error bars quietly understate it.  The practical
consequence is the same one the campaign inventory reached from the other
side: sensitivity to the diurnal is bought with longer and redundant pairs
(``i -> i+k``), which is what the 20170827 ``itab`` has and the 20170803 one
does not.

What this does not fix
----------------------
The model must still be right.  A diurnal atmospheric residual fits a diurnal
harmonic beautifully, single-step or not — the ice-vs-air tests in
:mod:`gpri_tools.diurnal` (range dependence, refractivity regression, bedrock null)
apply to these estimates exactly as before.  Pass the per-epoch refractivity
series as a ``covariates`` column to project the atmosphere out *inside* the
fit rather than after it.
"""
from __future__ import annotations

import numpy as np

from .diurnal import DIURNAL, MIN_CYCLES, SEMIDIURNAL  # periods in days

__all__ = ["PairModelFit", "temporal_design", "pair_design", "fit_pairs"]


# ------------------------------------------------------------------ designs
def temporal_design(times, periods=(DIURNAL,), degree=1, covariates=None):
    """Per-epoch model matrix ``F`` and the parameter names, in order.

    Columns: ``1, t, t^2, ..., cos/sin per period, covariates...`` with ``t``
    in days.  ``covariates`` is an optional ``(n_epochs, k)`` array (or dict
    of name -> series) of extra regressors known per epoch — the per-epoch
    refractivity series from :func:`gpri_tools.refractivity.invert_refractivity` is
    the intended use.
    """
    t = np.asarray(times, float)
    cols = [t ** k for k in range(degree + 1)]
    names = [f"t^{k}" if k else "1" for k in range(degree + 1)]
    for p in periods:
        w = 2.0 * np.pi / float(p)
        cols += [np.cos(w * t), np.sin(w * t)]
        names += [f"cos{p:g}d", f"sin{p:g}d"]
    if covariates is not None:
        if isinstance(covariates, dict):
            for k, v in covariates.items():
                cols.append(np.asarray(v, float))
                names.append(str(k))
        else:
            c = np.atleast_2d(np.asarray(covariates, float))
            if c.shape[0] == t.size and c.ndim == 2:
                c = c.T
            for k, v in enumerate(c):
                cols.append(np.asarray(v, float))
                names.append(f"cov{k}")
    F = np.column_stack(cols)
    if F.shape[0] != t.size:
        raise ValueError("covariates must have one value per epoch")
    return F, names


def pair_design(times, pairs, periods=(DIURNAL,), degree=1, covariates=None,
                span=None):
    """Differenced design ``G[p] = F[j] - F[i]`` with unobservable columns cut.

    The constant column differences to exactly zero and is dropped — that is
    the mathematics saying an interferometric network cannot see an absolute
    offset.  Any covariate that happens to difference to zero everywhere is
    dropped for the same reason, and the returned ``names`` say what survived.

    Refuses a record shorter than the longest period, for the same reason
    :func:`gpri_tools.diurnal.harmonic_design` does: amplitude and rate are not
    separable over a fraction of a cycle, and a returned number would be
    meaningless.
    """
    t = np.asarray(times, float)
    total = (t.max() - t.min()) if span is None else float(span)
    for p in periods:
        if total < p * MIN_CYCLES:
            raise ValueError(
                f"record spans {total * 24:.2f} h but a {p * 24:.0f} h "
                f"harmonic was requested; amplitude and rate are not "
                f"separable over less than one cycle")
    F, names = temporal_design(t, periods, degree, covariates)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    G = F[pr[:, 1]] - F[pr[:, 0]]
    keep = np.ptp(G, axis=0) > 0
    return G[:, keep], [n for n, k in zip(names, keep) if k]


# ------------------------------------------------------------------- result
class PairModelFit:
    """Model parameters per pixel, with the covariance to stand behind them."""

    def __init__(self, params, names, cov_unit, sigma2, periods, degree,
                 n_obs, shape=()):
        #: ``(n_params, ...)`` fitted parameters
        self.params = np.asarray(params, float)
        self.names = list(names)
        #: unit-weight parameter covariance ``(G^T W G)^-1``
        self.cov_unit = np.asarray(cov_unit, float)
        #: per-pixel residual variance of unit weight
        self.sigma2 = np.asarray(sigma2, float)
        self.periods = tuple(periods)
        self.degree = int(degree)
        self.n_obs = n_obs
        self.shape = tuple(shape)

    def _idx(self, name):
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(f"no parameter {name!r}; fitted {self.names}") \
                from None

    def param(self, name):
        return self.params[self._idx(name)]

    def param_sigma(self, name):
        i = self._idx(name)
        return np.sqrt(np.maximum(self.cov_unit[i, i] * self.sigma2, 0.0))

    @property
    def secular(self):
        """Linear rate, in input units per day."""
        return self.param("t^1")

    @property
    def secular_sigma(self):
        return self.param_sigma("t^1")

    def _ab(self, period):
        return (self._idx(f"cos{period:g}d"), self._idx(f"sin{period:g}d"))

    def amplitude(self, period=DIURNAL):
        ia, ib = self._ab(period)
        return np.hypot(self.params[ia], self.params[ib])

    def phase(self, period=DIURNAL):
        ia, ib = self._ab(period)
        return np.arctan2(self.params[ib], self.params[ia])

    def peak_time(self, period=DIURNAL, origin_hour=0.0):
        """Hour of day of the harmonic peak, as in :class:`gpri_tools.diurnal.HarmonicFit`."""
        hours = (self.phase(period) / (2.0 * np.pi)) * period * 24.0
        return np.mod(origin_hour + hours, period * 24.0)

    def amplitude_sigma(self, period=DIURNAL):
        """Standard error of the amplitude, by covariance propagation.

        ``A = hypot(a, b)`` gives ``var(A) = (a^2 Vaa + b^2 Vbb + 2ab Vab)/A^2``
        to first order.  Where the amplitude itself is ~0 the linearisation is
        meaningless; those pixels return the conservative
        ``sqrt(max(Vaa, Vbb) * sigma2)`` instead.
        """
        ia, ib = self._ab(period)
        a, b = self.params[ia], self.params[ib]
        A = np.hypot(a, b)
        vaa = self.cov_unit[ia, ia] * self.sigma2
        vbb = self.cov_unit[ib, ib] * self.sigma2
        vab = self.cov_unit[ia, ib] * self.sigma2
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (a * a * vaa + b * b * vbb + 2 * a * b * vab) / (A * A)
        floor = np.maximum(vaa, vbb)
        var = np.where(A > 0, var, floor)
        return np.sqrt(np.maximum(var, 0.0))

    def snr(self, period=DIURNAL):
        """Amplitude / its standard error.  The per-pixel detection statistic.

        The honest bar: under the null (no harmonic) the amplitude is
        Rayleigh-distributed, so demand ``snr > 3`` before calling anything a
        detection, and remember that a diurnal *atmosphere* passes this test
        too — significance is not attribution.
        """
        s = self.amplitude_sigma(period)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(s > 0, self.amplitude(period) / s, 0.0)

    def __repr__(self):
        a = np.nanmedian(self.amplitude(self.periods[0])) if self.periods else 0
        return (f"PairModelFit({self.names}, "
                f"median diurnal amp={a * 1000:.2f} mm, pixels={self.shape})")


# ---------------------------------------------------------------------- fit
def fit_pairs(observations, network, periods=(DIURNAL,), degree=1,
              weights=None, covariates=None, rcond=None, max_gib=4.0):
    """Weighted least squares from pair observations to model parameters.

    Parameters
    ----------
    observations : array (n_pairs, ...)
        Pair values in the ``d_j - d_i`` convention — LOS displacement from
        :func:`gpri_tools.timeseries.los_displacement`, exactly as
        :func:`gpri_tools.timeseries.invert_network` takes.  For a corrected series
        that only exists in epoch form, re-difference it:
        ``obs[p] = d[j] - d[i]`` puts the epoch-domain corrections onto the
        pairs without disturbing the pair noise structure.
    network : :class:`gpri_tools.network.Network`
    weights : (n_pairs,) or (n_pairs, ...) array, optional
        Inverse-variance up to a scale — per-pair coherence-derived weights
        are the point of the exercise.  1-D weights share one factorisation
        across all pixels (fast); per-pixel weights solve normal equations per
        pixel (memory-guarded by ``max_gib``).
    covariates : array or dict, optional
        Extra per-epoch regressors (see :func:`temporal_design`) — the
        refractivity series, to absorb the atmosphere inside the fit.

    Returns
    -------
    :class:`PairModelFit`
    """
    obs = np.asarray(observations, float)
    if obs.shape[0] != network.n_pairs:
        raise ValueError(f"{obs.shape[0]} observations but the network has "
                         f"{network.n_pairs} pairs")
    G, names = pair_design(network.times, network.pairs, periods, degree,
                           covariates)
    m = G.shape[1]
    shape = obs.shape[1:]
    Y = obs.reshape(obs.shape[0], -1)
    npix = Y.shape[1]
    finite = np.isfinite(Y)
    Y0 = np.where(finite, Y, 0.0)

    w = None if weights is None else np.asarray(weights, float)
    per_pixel = (w is not None and w.ndim > 1) or not finite.all()

    if not per_pixel:
        sw = np.ones(G.shape[0]) if w is None else np.sqrt(np.maximum(w, 0.0))
        A = G * sw[:, None]
        X, *_ = np.linalg.lstsq(A, Y0 * sw[:, None], rcond=rcond)
        GtWG = A.T @ A
        cov_unit = np.linalg.pinv(GtWG)
        r = Y0 - G @ X
        wr = r * (sw ** 2)[:, None]
        dof = max(G.shape[0] - m, 1)
        sigma2 = np.einsum("pk,pk->k", r, wr) / dof
    else:
        W = np.ones((G.shape[0], npix)) if w is None else \
            np.maximum(w.reshape(G.shape[0], -1), 0.0) * 1.0
        W = np.where(finite, np.broadcast_to(W, (G.shape[0], npix)), 0.0)
        need = npix * m * m * 8 / 2 ** 30
        if need > max_gib:
            raise MemoryError(
                f"per-pixel weights need {need:.1f} GiB of normal equations "
                f"for {npix} pixels x {m} params (limit {max_gib}); mask down "
                f"or use shared (n_pairs,) weights")
        GtG = np.einsum("pm,pn,pk->kmn", G, G, W, optimize=True)
        Gty = np.einsum("pm,pk,pk->km", G, W, Y0, optimize=True)
        eps = 1e-12 * np.trace(GtG, axis1=1, axis2=2)[:, None, None] * np.eye(m)
        X = np.linalg.solve(GtG + eps, Gty[..., None])[..., 0].T
        cov_unit = np.linalg.pinv(GtG + eps)          # (npix, m, m)
        r = np.where(finite, Y0 - G @ X, 0.0)
        n_eff = np.maximum((W > 0).sum(axis=0), m + 1)
        sigma2 = np.einsum("pk,pk->k", r, r * W) / (n_eff - m)
        # reshape per-pixel covariance to (m, m, ...) on the way out
        cov_unit = np.moveaxis(cov_unit, 0, -1).reshape((m, m) + shape)

    params = X.reshape((m,) + shape)
    sigma2 = sigma2.reshape(shape)
    if not per_pixel:
        pass                                           # cov_unit is (m, m)
    n_obs = int(G.shape[0])
    return PairModelFit(params, names, cov_unit, sigma2, periods, degree,
                        n_obs, shape=shape)
