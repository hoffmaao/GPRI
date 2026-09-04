"""Refractivity screens: the matched filter, the robust fit, and the physics."""
import numpy as np
import pytest

from gpri_tools.atmosphere import (MODELS, PhaseScreen, delta_refractivity,
                             design_matrix, estimate_range_ramp, fit_screen,
                             ramp_objective, remove_screen, stable_mask)

LAM = 0.017430
NA, NR = 40, 2000
R = 300.0 + 0.750349 * np.arange(NR)         # BakerBend1 range axis, shortened
AZ = -27.955 + 0.2 * np.arange(NA)


def _wrapped(k, noise=0.0, seed=0, extra=0.0):
    rng = np.random.default_rng(seed)
    phi = k * np.broadcast_to(R, (NA, NR)) + extra
    if noise:
        phi = phi + rng.normal(0, noise, (NA, NR))
    return np.angle(np.exp(1j * phi))


# ------------------------------------------------------------------- physics
def test_delta_refractivity_conversion():
    """dn = ramp * lambda / (4 pi), the two-way path relation."""
    assert delta_refractivity(4 * np.pi / LAM * 1e-6, LAM) == pytest.approx(1e-6)


def test_a_realistic_refractivity_change_is_several_fringes():
    """dn = 1e-6 over the real BakerBend1 swath must be ~12 rad, not ~0."""
    k = 4 * np.pi * 1e-6 / LAM
    span = 16882.851434 - 300.139581
    assert k * span == pytest.approx(12.0, rel=0.1)


# ------------------------------------------------------------ matched filter
def test_objective_peaks_at_the_true_ramp():
    k = 3e-3
    grid = np.linspace(k - 5e-4, k + 5e-4, 201)
    resp = ramp_objective(_wrapped(k), R, grid)
    assert grid[int(np.argmax(resp))] == pytest.approx(k, abs=2e-5)


def test_objective_is_one_for_a_perfect_ramp_and_small_off_peak():
    k = 2e-3
    assert ramp_objective(_wrapped(k), R, k) == pytest.approx(1.0, abs=1e-6)
    assert ramp_objective(_wrapped(k), R, k + 0.05) < 0.1


@pytest.mark.parametrize("k", [0.0, 1e-4, -1e-3, 5e-3, -1.2e-2])
def test_recovers_ramps_of_many_fringes_without_unwrapping(k):
    """The whole point: tens of fringes recovered from wrapped phase alone."""
    got, q = estimate_range_ramp(_wrapped(k), R, wavelength=LAM)
    assert got == pytest.approx(k, abs=2e-6)
    assert q > 0.99


def test_recovers_a_ramp_under_heavy_phase_noise():
    """Precision is set by SNR, so the tolerance is stated as phase across the swath."""
    k, sigma = -4e-3, 1.5
    got, q = estimate_range_ramp(_wrapped(k, noise=sigma, seed=2), R, wavelength=LAM)
    assert abs(got - k) * (R[-1] - R[0]) < 0.1          # < 0.1 rad end to end
    # the matched-filter response should land on the theoretical coherence
    assert q == pytest.approx(np.exp(-sigma ** 2 / 2), rel=0.1)


def test_weights_steer_the_estimate():
    """Half the scene ramps one way, half the other; weights pick the winner."""
    phi = np.empty((NA, NR))
    phi[: NA // 2] = np.angle(np.exp(1j * 2e-3 * R))
    phi[NA // 2:] = np.angle(np.exp(1j * -5e-3 * R))
    w = np.zeros((NA, NR))
    w[NA // 2:] = 1.0
    got, _ = estimate_range_ramp(phi, R, weights=w, wavelength=LAM)
    assert got == pytest.approx(-5e-3, abs=1e-5)


def test_zero_weight_everywhere_is_handled():
    got, q = estimate_range_ramp(_wrapped(1e-3), R, weights=np.zeros((NA, NR)),
                                 wavelength=LAM)
    assert (got, q) == (0.0, 0.0)


def test_accepts_a_complex_interferogram():
    k = 1.5e-3
    z = np.exp(1j * k * np.broadcast_to(R, (NA, NR))) * 3.0
    assert estimate_range_ramp(z, R, wavelength=LAM)[0] == pytest.approx(k, abs=2e-6)


def test_non_uniform_range_axis_still_works():
    r = np.sort(np.random.default_rng(0).uniform(300, 5000, 400))
    k = 2e-3
    phi = np.angle(np.exp(1j * k * np.broadcast_to(r, (10, 400))))
    got, _ = estimate_range_ramp(phi, r, wavelength=LAM, oversample=8)
    assert got == pytest.approx(k, abs=5e-5)


# ------------------------------------------------------------- design matrix
@pytest.mark.parametrize("model", sorted(MODELS))
def test_every_model_builds(model):
    A = design_matrix(model, R, AZ)
    assert A.shape == (NA * NR, len(MODELS[model]))
    assert np.all(np.isfinite(A))


def test_predictors_are_centred():
    A = design_matrix("linear", R)
    assert A[:, 1].mean() == pytest.approx(0.0, abs=1e-9)


def test_azimuth_model_without_azimuth_is_rejected():
    with pytest.raises(ValueError, match="azimuth"):
        design_matrix("planar", R)


def test_unknown_term_is_rejected():
    with pytest.raises(ValueError, match="unknown screen term"):
        design_matrix(["1", "banana"], R)


# ----------------------------------------------------------------- full fit
def test_fit_recovers_ramp_and_offset():
    k, c = 3e-3, 0.4
    scr = fit_screen(_wrapped(k, extra=c), slant_range=R, wavelength=LAM,
                     model="linear", robust=False)
    assert scr.ramp == pytest.approx(k, abs=2e-6)
    assert scr.coeffs[0] == pytest.approx(c, abs=0.05)
    assert scr.delta_n == pytest.approx(delta_refractivity(k, LAM))


def test_removing_the_screen_flattens_the_phase():
    phi = _wrapped(2.5e-3, extra=0.3, noise=0.2, seed=1)
    scr = fit_screen(phi, slant_range=R, wavelength=LAM, model="linear")
    after = remove_screen(phi, scr)
    assert np.abs(np.mean(np.exp(1j * after))) > np.abs(np.mean(np.exp(1j * phi)))
    assert np.abs(np.mean(np.exp(1j * after))) > 0.95


def test_remove_screen_preserves_complex_magnitude():
    z = 7.0 * np.exp(1j * _wrapped(1e-3))
    scr = fit_screen(z, slant_range=R, wavelength=LAM)
    assert np.allclose(np.abs(remove_screen(z, scr)), 7.0)


def test_azimuth_tilt_is_captured():
    tilt = 0.01 * (AZ - AZ.mean())[:, None]
    phi = np.angle(np.exp(1j * (1e-3 * R[None, :] + tilt)))
    scr = fit_screen(phi, slant_range=R, azimuth=AZ, wavelength=LAM,
                     model="planar", robust=False)
    resid = remove_screen(phi, scr)
    assert np.abs(np.mean(np.exp(1j * resid))) > 0.99


def test_robust_fit_ignores_a_moving_tongue():
    """A fast-moving glacier tongue must not drag the atmospheric screen.

    The tongue occupies part of the azimuth sweep at all ranges, which is the
    real geometry: a moving target sits at a bearing, not at a range band.  (A
    displacement confined to a *range* band is genuinely indistinguishable from
    an atmospheric range ramp, and no estimator can separate the two.)
    """
    phi = 1e-3 * np.broadcast_to(R, (NA, NR)).copy()
    phi[: NA // 4] += 2.5                          # 25% of the sweep is moving
    phi = np.angle(np.exp(1j * phi))
    loose = fit_screen(phi, slant_range=R, wavelength=LAM, robust=False)
    tight = fit_screen(phi, slant_range=R, wavelength=LAM, robust=True, iterations=8)
    assert abs(loose.coeffs[0]) > 0.5              # least squares is dragged
    assert abs(tight.coeffs[0]) < 0.05             # the robust fit is not
    assert tight.ramp == pytest.approx(1e-3, abs=1e-6)


def test_robust_fit_reduces_bias_from_a_compact_moving_block():
    phi = 1e-3 * np.broadcast_to(R, (NA, NR)).copy()
    phi[5:15, 200:400] += 2.5
    phi = np.angle(np.exp(1j * phi))
    loose = fit_screen(phi, slant_range=R, wavelength=LAM, robust=False)
    tight = fit_screen(phi, slant_range=R, wavelength=LAM, robust=True, iterations=8)
    assert abs(tight.coeffs[0]) < abs(loose.coeffs[0])


def test_mask_restricts_the_fit():
    phi = np.angle(np.exp(1j * 1e-3 * np.broadcast_to(R, (NA, NR))))
    mask = np.zeros((NA, NR), bool)
    mask[:, NR // 2:] = True
    scr = fit_screen(phi, slant_range=R, wavelength=LAM, mask=mask)
    assert scr.ramp == pytest.approx(1e-3, abs=1e-5)


def test_fit_without_usable_pixels_returns_a_null_screen():
    scr = fit_screen(_wrapped(1e-3), slant_range=R, wavelength=LAM,
                     mask=np.zeros((NA, NR), bool))
    assert np.allclose(scr.coeffs, 0.0)
    assert np.allclose(scr.evaluate(), 0.0)


def test_screen_evaluate_has_the_image_shape():
    scr = fit_screen(_wrapped(1e-3), slant_range=R, azimuth=AZ, wavelength=LAM,
                     model="bilinear")
    assert scr.evaluate().shape == (NA, NR)
    assert "dN" in repr(scr)


def test_screen_needs_a_wavelength_for_delta_n():
    scr = PhaseScreen([0.0], "constant", R, ramp=1e-3)
    with pytest.raises(ValueError, match="wavelength"):
        scr.delta_n


def test_missing_range_axis_is_rejected():
    with pytest.raises(ValueError, match="slant_range"):
        fit_screen(_wrapped(1e-3))


# --------------------------------------------------------------- stable mask
def test_stable_mask_thresholds_coherence():
    cc = np.array([[0.1, 0.7], [0.9, 0.5]])
    assert stable_mask(cc, 0.6).tolist() == [[False, True], [True, False]]


def test_stable_mask_can_also_require_brightness():
    cc = np.full((2, 4), 0.9)
    amp = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    m = stable_mask(cc, 0.6, amplitude=amp, amplitude_percentile=50)
    assert m.sum() == 4 and m[:, :2].sum() == 0


def test_stable_mask_drops_nans():
    cc = np.array([[np.nan, 0.8]])
    assert stable_mask(cc, 0.6).tolist() == [[False, True]]
