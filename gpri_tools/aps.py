"""Network-consistent atmospheric correction: per-epoch screens plus turbulence.

Why per-pair screens are not enough
-----------------------------------
:func:`gpri_tools.atmosphere.fit_screen` estimates each interferogram's screen
independently.  Three things are wrong with stopping there, and the full-day
BakerBend1 run exposed all of them:

1. **The fits do not close.**  The screen of pair ``(i, j)`` is physically
   ``S_j - S_i`` — a difference of two per-epoch atmospheres — but independent
   fits are under no obligation to satisfy that, and their inconsistency is
   pure error injected into the time series.
2. **The noise accumulates.**  On a daisy chain the network inversion is a
   cumulative sum, so per-pair estimation noise integrates into a random walk.
   That walk was most of the 99.4 mm common mode found on the 20170803 day.
3. **Each fit sees one pair's worth of data.**  Epoch ``k`` appears in every
   pair that touches it; fitting per-pair throws that redundancy away.

Two corrections, for two different errors
----------------------------------------
:func:`invert_screens` treats the fitted per-pair screen *coefficients* as
network observations — exactly the quantities
:func:`gpri_tools.timeseries.invert_network` was built to invert — and solves for one
coefficient vector **per epoch**.  Screen coefficients are linear in the phase,
so this is exact, and the reconstructed pair screens ``e_j - e_i`` close by
construction.  Its real value is on networks with redundancy (``i -> i+2``
pairs and beyond), where the joint solve averages the per-pair noise down.
Be honest about its limit on a pure daisy chain: the system is square there,
so consistency alone changes nothing, and the temporal-smoothness penalty was
*measured* (synthetic 40-epoch chains) to help by only ~3 % at its best setting
and to hurt beyond it — because the error that matters is the low-frequency
random-walk drift of the integrated series, which a curvature penalty cannot
see.

:func:`epoch_screen_correction` is what actually kills the drift.  It works in
the **displacement domain**, after the network is integrated: at each epoch the
accumulated atmospheric error is, on stable ground, directly visible — bedrock
is not moving, so whatever spatial structure the series has there *is* the
error.  Fitting the screen model to each epoch's displacement over stable
ground and subtracting it everywhere generalises
:func:`gpri_tools.timeseries.reference_to_stable` from a constant to the full model:
the constant term is the common mode that function removes, and the range term
is the accumulated ramp drift it cannot.  No wrapping issues arise, because
displacement is already integrated.

What the parametric model can never catch
-----------------------------------------
A low-order polynomial in range and azimuth describes the *mixed, homogeneous*
part of the refractivity change.  The rest — moist convection rolling up the
flank, a cloud edge crossing the fan — is **turbulent**: spatially smooth on
scales of hundreds of metres, but not polynomial.  After the 20170803 day was
corrected with linear screens and referenced to bedrock, 41 % of the remaining
time-series variance was still explained by the refractivity series; that
residual is this term.

:func:`turbulence_screen` estimates it non-parametrically: the wrapped residual
phase *on stable ground* is smoothed by normalised convolution and read off as
a screen for the whole scene.  Working on the complex phasor means no
unwrapping; weighting by the stable mask means ice motion cannot leak into the
estimate — the screen over ice is interpolated from the rock around it, never
from the ice itself.  Where there is no stable ground within reach of the
kernel, the screen is zero and the phase is left alone, with the
``quality`` map saying so.

One correction that is *not* here, and why
------------------------------------------
A stratified (height-dependent) term, standard in spaceborne InSAR, is
**unidentifiable for a GPRI without a DEM**: every pixel shares the same
antenna elevation angle, so beam height is exactly ``alt + r sin(elev)`` —
perfectly linear in slant range and therefore absorbed, indistinguishably,
by the uniform-mixing ramp the matched filter already estimates.  Separating
them needs per-pixel terrain height, and no DEM accompanies the data.
"""
from __future__ import annotations

import numpy as np

from .atmosphere import MODELS, PhaseScreen, _terms_of, delta_refractivity
from .timeseries import invert_network

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
    gaussian_filter = None

__all__ = ["EpochScreens", "invert_screens", "epoch_screen_correction",
           "displacement_ramp_to_delta_n", "turbulence_screen",
           "remove_turbulence"]


# ------------------------------------------------------------ epoch screens
class EpochScreens:
    """One atmospheric screen per epoch, relative to a reference epoch."""

    def __init__(self, params, network, model, slant_range, azimuth=None,
                 wavelength=None, reference=0, residual_rms=None):
        #: ``(n_epochs, 1 + n_terms)`` — column 0 is the range ramp (rad/m),
        #: the rest are the model coefficients, all relative to ``reference``.
        self.params = np.asarray(params, float)
        self.network = network
        self.model = model
        self.slant_range = np.asarray(slant_range, float)
        self.azimuth = None if azimuth is None else np.asarray(azimuth, float)
        self.wavelength = wavelength
        self.reference = int(reference)
        #: per-pair rms misfit between fitted and reconstructed coefficients
        self.residual_rms = residual_rms

    @property
    def n_epochs(self):
        return self.params.shape[0]

    def _screen(self, p):
        return PhaseScreen(p[1:], self.model, self.slant_range, self.azimuth,
                           wavelength=self.wavelength, ramp=p[0])

    def epoch_screen(self, k):
        """Epoch ``k``'s screen relative to the reference epoch."""
        return self._screen(self.params[k])

    def pair_screen(self, p):
        """The network-consistent screen of pair ``p``: ``e_j - e_i``.

        This is what to hand to :func:`gpri_tools.atmosphere.remove_screen` in place
        of the independently fitted per-pair screen.  Around any closed
        triangle these reconstructions sum to zero by construction, which the
        independent fits never did.
        """
        i, j = self.network.pairs[p]
        return self._screen(self.params[j] - self.params[i])

    def delta_n(self):
        """Per-epoch refractivity relative to the reference, N-units.

        The physical series to plot against a weather station — the same
        quantity :func:`gpri_tools.refractivity.invert_refractivity` produces from
        per-pair values, but estimated jointly with the rest of the screen.
        """
        if self.wavelength is None:
            raise ValueError("wavelength not known for these screens")
        return delta_refractivity(self.params[:, 0], self.wavelength) * 1e6

    def __repr__(self):
        n = "?" if self.wavelength is None else \
            f"{np.ptp(self.delta_n()):.3f}"
        return (f"EpochScreens({self.n_epochs} epochs, model={self.model!r}, "
                f"dN span={n} N-units)")


def invert_screens(screens, network, smoothing=0.0, weights=None,
                   reference=0):
    """Per-epoch screens from independently fitted per-pair screens.

    Parameters
    ----------
    screens : sequence of :class:`gpri_tools.atmosphere.PhaseScreen` or None
        One per pair, in network order — the output of running
        :func:`gpri_tools.atmosphere.fit_screen` over the stack.  ``None`` marks a
        pair whose fit failed; it is excluded (weight zero), not zero-filled.
    network : :class:`gpri_tools.network.Network`
    smoothing : float
        Second-difference temporal penalty, passed to
        :func:`gpri_tools.timeseries.invert_network`.  0 reproduces the per-pair
        fits exactly on a daisy chain (the system is square there).  Keep it
        small: measured on synthetic chains, a light penalty buys a few
        percent and a heavy one biases the recovered atmosphere toward a
        straight line.  The drift this cannot remove is
        :func:`epoch_screen_correction`'s job.
    weights : (n_pairs,) array, optional
        Per-pair confidence.  Defaults to each screen's matched-filter
        ``quality``; multiplied with it if given.

    Returns
    -------
    :class:`EpochScreens`
    """
    screens = list(screens)
    if len(screens) != network.n_pairs:
        raise ValueError(f"{len(screens)} screens for {network.n_pairs} pairs")

    ref = next((s for s in screens if s is not None), None)
    if ref is None:
        raise ValueError("every screen is None; nothing to invert")
    terms = _terms_of(ref.model)
    n_terms = len(terms)

    obs = np.zeros((network.n_pairs, 1 + n_terms))
    w = np.zeros(network.n_pairs)
    for p, s in enumerate(screens):
        if s is None:
            continue
        if _terms_of(s.model) != terms:
            raise ValueError(
                f"pair {p} was fitted with model {s.model!r}, others with "
                f"{ref.model!r}; invert one model at a time")
        obs[p, 0] = s.ramp
        obs[p, 1:] = s.coeffs
        w[p] = 1.0 if s.quality is None else max(float(s.quality), 1e-6)
    if weights is not None:
        w = w * np.asarray(weights, float)

    method = "smooth" if smoothing > 0 else "wls"
    ts = invert_network(obs, network, weights=w, method=method,
                        reference=reference, smoothing=smoothing)

    recon = np.stack([ts.displacement[j] - ts.displacement[i]
                      for i, j in network.pairs])
    rms = np.sqrt(np.mean((recon - obs) ** 2, axis=1))

    return EpochScreens(ts.displacement, network, ref.model, ref.slant_range,
                        ref.azimuth, wavelength=ref.wavelength,
                        reference=reference, residual_rms=rms)


# -------------------------------------------------- displacement-domain fit
def displacement_ramp_to_delta_n(slope):
    """Displacement range-slope (m per m of range) -> refractivity, N-units.

    ``d = -(lambda / 4 pi) * phi`` and ``phi = (4 pi / lambda) * dn * r``, so
    the wavelength cancels and ``d(r) = -dn * r``: a displacement ramp of
    -1 mm/km is exactly +1 N-unit of accumulated refractivity error.
    """
    return -np.asarray(slope, float) * 1e6


def epoch_screen_correction(displacement, mask, slant_range, azimuth=None,
                            model="linear", weights=None, rcond=None):
    """Fit-and-remove a screen per epoch, on the displacement series itself.

    The drift killer.  After network integration, each epoch's accumulated
    atmospheric error is written plainly on stable ground — bedrock is not
    moving, so any spatial structure the displacement has there is error.
    This fits the screen model to every epoch's displacement over the masked
    pixels (one shared factorisation, so it is one ``lstsq`` for the whole
    series) and subtracts the evaluated screen from the whole scene.

    With ``model="constant"`` this *is*
    :func:`gpri_tools.timeseries.reference_to_stable` (with a weighted-mean rather
    than a median).  ``"linear"`` additionally removes the accumulated
    range-ramp drift — the spurious range-linear displacement that per-pair
    ramp noise integrates into and that spatial referencing cannot touch.

    Parameters
    ----------
    displacement : array (n_epochs, na, nr)
        LOS displacement, metres.
    mask : bool array (na, nr)
        Stable ground defining the fit.  As with referencing: hold pixels out
        of this mask if you intend to test the result on stable ground,
        or the test is circular.
    slant_range : (nr,) array
    azimuth : (na,) array, optional — required by models with azimuth terms.
    model : str or term sequence, from :data:`gpri_tools.atmosphere.MODELS`.
    weights : array (na, nr), optional
        Quality inside the mask (mean coherence is the natural choice).

    Returns
    -------
    corrected : array, same shape
    coeffs : array (n_epochs, n_terms)
        Per-epoch fitted coefficients, in metres, for the centred predictors.
        Feed the ``"r"`` column to :func:`displacement_ramp_to_delta_n` for
        the accumulated refractivity series it implies.
    """
    from .atmosphere import _design_at

    d = np.asarray(displacement, float)
    m = np.asarray(mask, bool)
    if d.ndim != 3 or m.shape != d.shape[1:]:
        raise ValueError(f"displacement {d.shape} and mask {m.shape} do not "
                         f"describe an (n_epochs, na, nr) series")
    if not m.any():
        raise ValueError("mask selects no pixels; nothing to fit the screen on")

    terms = _terms_of(model)
    r = np.asarray(slant_range, float)
    rc = r - r.mean()
    ac = (np.zeros(d.shape[1]) if azimuth is None
          else np.asarray(azimuth, float) - np.asarray(azimuth, float).mean())
    if azimuth is None and any(t in ("a", "a2", "ra") for t in terms):
        raise ValueError(f"model {model!r} needs azimuth angles")

    rows, cols = np.nonzero(m)
    A = _design_at(terms, rc[cols], ac[rows])

    w = np.ones(rows.size) if weights is None else \
        np.clip(np.nan_to_num(np.asarray(weights, float)[rows, cols]), 0.0, None)
    Y = np.nan_to_num(d[:, rows, cols]).T                # (n_masked, n_epochs)
    finite = np.isfinite(d[:, rows, cols]).all(axis=0)
    w = w * finite

    sw = np.sqrt(w)[:, None]
    coeffs, *_ = np.linalg.lstsq(A * sw, Y * sw, rcond=rcond)  # (n_terms, n_epochs)

    A_full = _design_at(terms, np.broadcast_to(rc, d.shape[1:]).ravel(),
                        np.broadcast_to(ac[:, None], d.shape[1:]).ravel())
    screen = (A_full @ coeffs).T.reshape(d.shape)
    return d - screen, coeffs.T


# --------------------------------------------------------------- turbulence
def turbulence_screen(phase, mask, sigma, weights=None, min_support=0.02,
                      min_coherence=0.25, wrapped=True):
    """Non-parametric residual screen from stable ground, no unwrapping.

    Normalised convolution of the complex phasor: the wrapped residual phase at
    stable-ground pixels is smoothed with a Gaussian kernel and the smoothed
    phasor's angle becomes the screen.  Because the weight is the stable mask,
    the screen over moving ice is interpolated from the rock around it — ice
    motion cannot contaminate its own correction.

    Run this **after** the parametric screens, when the residual on stable
    ground is small; wrapped arithmetic is then safe because averaging phasors
    within a fraction of a cycle is linear to first order.

    Parameters
    ----------
    phase : array (na, nr)
        Wrapped residual phase (radians), or the complex interferogram.
    mask : bool array
        Stable ground — the pixels allowed to *define* the screen.
    sigma : (float, float)
        Gaussian kernel in **pixels**, ``(azimuth, range)``.  Pick it from the
        physical correlation length of the turbulence (hundreds of metres) and
        the pixel spacing — for full-resolution GPRI at mid-range that is far
        more range pixels than azimuth pixels, because a range pixel is 0.75 m
        and an azimuth pixel tens of metres.
    weights : array, optional
        Per-pixel quality inside the mask (coherence).
    min_support : float
        Minimum local weighted density of stable ground.  Below it the screen
        is 0 — no correction — rather than an extrapolation from nothing.
    min_coherence : float
        Minimum magnitude of the averaged phasor.  If the stable pixels inside
        the kernel disagree with each other the mean phasor shrinks toward
        zero and its angle is noise; those places also get screen 0.  Only
        meaningful for wrapped input.
    wrapped : bool
        True (default) treats the input as wrapped phase and averages the
        complex phasor — right for per-pair interferogram residuals.  False
        averages the values directly — right for already-integrated
        **displacement** residuals (metres), where wrapping is not a concern
        and the screen comes back in the input's own units.

    Returns
    -------
    screen : array (na, nr)
        Radians (wrapped) or input units (unwrapped); exactly 0 where the
        estimate is not supported.
    quality : array (na, nr)
        Where supported: magnitude of the averaged phasor (wrapped), or the
        local weighted density of stable ground (unwrapped).  0 elsewhere.
        A map of where the correction can be believed.
    """
    if gaussian_filter is None:  # pragma: no cover
        raise ImportError("turbulence_screen needs scipy")

    p = np.asarray(phase)
    w = np.asarray(mask, bool).astype(float)
    if weights is not None:
        w = w * np.clip(np.nan_to_num(np.asarray(weights, float)), 0.0, None)

    if not wrapped and not np.iscomplexobj(p):
        v = p.astype(float)
        w[~np.isfinite(v)] = 0.0
        v = np.nan_to_num(v)
        num = gaussian_filter(w * v, sigma, mode="nearest")
        den = gaussian_filter(w, sigma, mode="nearest")
        good = den > min_support
        with np.errstate(invalid="ignore", divide="ignore"):
            screen = np.where(good, num / np.where(good, den, 1.0), 0.0)
        return screen, np.where(good, den, 0.0)

    z = p / np.maximum(np.abs(p), 1e-30) if np.iscomplexobj(p) \
        else np.exp(1j * p.astype(float))
    w[~np.isfinite(z)] = 0.0
    z = np.nan_to_num(z)

    num = (gaussian_filter(w * z.real, sigma, mode="nearest")
           + 1j * gaussian_filter(w * z.imag, sigma, mode="nearest"))
    den = gaussian_filter(w, sigma, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        zs = np.where(den > min_support, num / den, 0.0)
    mag = np.abs(zs)
    good = (den > min_support) & (mag > min_coherence)

    screen = np.where(good, np.angle(np.where(good, zs, 1.0)), 0.0)
    quality = np.where(good, mag, 0.0)
    return screen, quality


def remove_turbulence(phase, mask, sigma, **kwargs):
    """Estimate and subtract the turbulence screen in one step.

    Returns ``(corrected, screen, quality)``; corrected is wrapped phase (or
    complex, matching the input) exactly as
    :func:`gpri_tools.atmosphere.remove_screen` would return it.
    """
    from .atmosphere import remove_screen

    screen, quality = turbulence_screen(phase, mask, sigma, **kwargs)
    return remove_screen(phase, screen), screen, quality
