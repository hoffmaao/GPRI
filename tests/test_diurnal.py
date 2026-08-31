"""Diurnal harmonic analysis, and the tests that separate ice from atmosphere."""
import numpy as np
import pytest

from gpri.diurnal import (DIURNAL, SEMIDIURNAL, atmospheric_coherence,
                          decompose_los, diurnal_amplitude, diurnal_phase,
                          fit_harmonics, harmonic_design, look_vector,
                          range_dependence, stable_ground_null,
                          vertical_sensitivity)


def _times(hours=24.18, cadence_min=2.0):
    """The real BakerBend1 sampling: 723 epochs, 2 min apart, over 24.18 h."""
    n = int(hours * 60 / cadence_min) + 1
    return np.arange(n) * (cadence_min / 1440.0)


def _signal(t, amp=0.003, phase=0.0, rate=0.0, offset=0.0):
    """Secular rate plus a 24-hour sinusoid, in metres."""
    return offset + rate * t + amp * np.cos(2 * np.pi * t / DIURNAL - phase)


# --------------------------------------------------------------- design matrix
def test_design_has_offset_rate_and_a_pair_per_period():
    t = _times()
    G = harmonic_design(t, periods=(DIURNAL, SEMIDIURNAL), degree=1)
    assert G.shape == (t.size, 2 + 4)
    assert np.allclose(G[:, 0], 1.0)
    assert np.allclose(G[:, 1], t)


def test_a_record_shorter_than_one_cycle_is_refused():
    """Six hours cannot constrain a 24-hour amplitude, and pretending it can is worse
    than failing."""
    t = _times(hours=6.0)
    with pytest.raises(ValueError, match="not separable"):
        harmonic_design(t, periods=(DIURNAL,))
    with pytest.raises(ValueError, match="not separable"):
        fit_harmonics(np.zeros((t.size, 2)), t)


# ----------------------------------------------------------------- recovery
def test_recovers_amplitude_phase_and_secular_rate():
    t = _times()
    d = _signal(t, amp=0.004, phase=1.1, rate=0.05, offset=0.01)
    fit = fit_harmonics(d, t)
    assert fit.amplitude(DIURNAL) == pytest.approx(0.004, abs=1e-9)
    assert fit.phase(DIURNAL) == pytest.approx(1.1, abs=1e-9)
    assert fit.secular == pytest.approx(0.05, abs=1e-9)
    assert fit.offset == pytest.approx(0.01, abs=1e-9)


def test_peak_time_is_a_real_hour_of_day():
    """A signal peaking 15 h after a 22:21 start peaks at about 13:00."""
    t = _times()
    peak_h = 15.0
    d = _signal(t, phase=2 * np.pi * peak_h / 24.0)
    fit = fit_harmonics(d, t)
    assert fit.peak_time(DIURNAL, origin_hour=22.36) == pytest.approx(
        (22.36 + peak_h) % 24.0, abs=1e-6)


def test_semidiurnal_term_is_separable_from_the_diurnal():
    t = _times()
    d = (_signal(t, amp=0.003)
         + 0.001 * np.cos(2 * np.pi * t / SEMIDIURNAL - 0.7))
    fit = fit_harmonics(d, t, periods=(DIURNAL, SEMIDIURNAL))
    assert fit.amplitude(DIURNAL) == pytest.approx(0.003, abs=1e-9)
    assert fit.amplitude(SEMIDIURNAL) == pytest.approx(0.001, abs=1e-9)


def test_asking_for_an_unfitted_period_is_an_error():
    t = _times()
    fit = fit_harmonics(_signal(t), t)
    with pytest.raises(ValueError, match="no harmonic at period"):
        fit.amplitude(SEMIDIURNAL)


def test_fit_broadcasts_over_pixels():
    t = _times()
    amps = np.array([[0.001, 0.002], [0.003, 0.004]])
    d = _signal(t[:, None, None], amp=amps)
    fit = fit_harmonics(d, t)
    assert fit.amplitude(DIURNAL).shape == (2, 2)
    assert np.allclose(fit.amplitude(DIURNAL), amps, atol=1e-9)
    assert "HarmonicFit" in repr(fit)


def test_noise_free_fit_explains_everything():
    t = _times()
    fit = fit_harmonics(_signal(t, amp=0.003, rate=0.02), t)
    assert fit.explained_variance() == pytest.approx(1.0, abs=1e-6)
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-12)


def test_evaluate_reproduces_the_input():
    t = _times()
    d = _signal(t, amp=0.003, rate=0.01)
    assert np.allclose(fit_harmonics(d, t).evaluate(), d, atol=1e-12)


def test_nan_epochs_are_ignored_not_propagated():
    t = _times()
    d = _signal(t, amp=0.004)
    d[10:20] = np.nan
    fit = fit_harmonics(d, t, weights=np.ones_like(t))
    assert fit.amplitude(DIURNAL) == pytest.approx(0.004, abs=1e-6)


def test_shorthands_agree_with_the_full_fit():
    t = _times()
    d = _signal(t, amp=0.0025, phase=1.0)
    assert diurnal_amplitude(d, t) == pytest.approx(0.0025, abs=1e-9)
    assert diurnal_phase(d, t, origin_hour=0.0) == pytest.approx(
        fit_harmonics(d, t).peak_time(DIURNAL), abs=1e-9)


def test_epoch_count_mismatch_is_caught():
    with pytest.raises(ValueError, match="epochs of displacement"):
        fit_harmonics(np.zeros((10, 3)), _times())


# ------------------------------------------------- ice or atmosphere: test 1
def test_range_dependence_flags_an_atmospheric_diurnal():
    """A diurnal that grows linearly with range is refractivity, not ice."""
    r = np.linspace(300.0, 16883.0, 200)
    amp = 1e-7 * r                       # pure range ramp
    out = range_dependence(np.broadcast_to(amp, (50, 200)), r)
    assert out["correlation"] > 0.9
    assert "dominated by residual refractivity" in out["verdict"]
    assert out["slope"] == pytest.approx(1e-7, rel=1e-6)


def test_range_dependence_clears_a_signal_with_no_range_structure():
    rng = np.random.default_rng(0)
    r = np.linspace(300.0, 16883.0, 200)
    amp = rng.normal(0.003, 0.0005, (50, 200))
    out = range_dependence(amp, r)
    assert abs(out["correlation"]) < 0.2
    assert "no range dependence" in out["verdict"]


def test_range_dependence_honours_a_mask_and_refuses_when_too_sparse():
    r = np.linspace(300.0, 16883.0, 100)
    amp = np.full((20, 100), 0.002)
    mask = np.zeros_like(amp, bool)
    mask[0, :5] = True
    out = range_dependence(amp, r, mask=mask)
    assert out["n"] == 5
    assert "no test possible" in out["verdict"]


# ------------------------------------------------- ice or atmosphere: test 2
def test_atmospheric_coherence_catches_a_purely_atmospheric_pixel():
    t = _times()
    rng = np.random.default_rng(1)
    N = np.cos(2 * np.pi * t / DIURNAL) + 0.1 * rng.normal(size=t.size)
    atmospheric = 0.002 * N                     # driven entirely by refractivity
    independent = _signal(t, amp=0.002, phase=np.pi / 2)

    d = np.stack([atmospheric, independent], axis=1)
    frac = atmospheric_coherence(d, t, N)
    assert frac[0] > 0.95
    assert frac[1] < frac[0]


def test_atmospheric_coherence_requires_a_shared_epoch_axis():
    t = _times()
    with pytest.raises(ValueError, match="share an"):
        atmospheric_coherence(np.zeros((t.size, 2)), t, np.zeros(5))


# ------------------------------------------------- ice or atmosphere: test 3
def test_stable_ground_null_reports_the_error_floor():
    t = _times()
    rng = np.random.default_rng(2)
    d = rng.normal(0, 0.0005, (t.size, 40, 40))
    mask = np.zeros((40, 40), bool)
    mask[:10] = True
    out = stable_ground_null(d, t, mask)
    assert out["n"] == 400
    assert out["amplitude_median"] > 0
    # independent noise should not share a phase
    assert out["phase_concentration"] < 0.3


def test_stable_ground_null_detects_a_shared_systematic_phase():
    """If bedrock all peaks at the same hour, that is systematic error."""
    t = _times()
    rng = np.random.default_rng(3)
    d = (_signal(t, amp=0.002)[:, None, None]
         + rng.normal(0, 0.0002, (t.size, 20, 20)))
    mask = np.ones((20, 20), bool)
    out = stable_ground_null(d, t, mask)
    assert out["phase_concentration"] > 0.9
    assert out["amplitude_median"] == pytest.approx(0.002, abs=2e-4)


def test_stable_ground_null_with_an_empty_mask_says_so():
    t = _times()
    out = stable_ground_null(np.zeros((t.size, 5, 5)), t, np.zeros((5, 5), bool))
    assert out["n"] == 0 and np.isnan(out["amplitude_median"])


# --------------------------------------------------------------- LOS geometry
class _Geom:
    """Minimal stand-in for RadarGeometry."""

    def __init__(self, bearings, elevation):
        self._b = np.asarray(bearings, float)
        self.elevation = elevation

    def bearings(self):
        return self._b


def test_vertical_sensitivity_is_the_sine_of_the_beam_elevation():
    assert vertical_sensitivity(_Geom([100.0], 10.0)) == pytest.approx(0.17365, abs=1e-5)


def test_a_tripod_radar_is_six_times_less_sensitive_to_uplift():
    """The number to quote beside any uplift claim."""
    g = _Geom([105.0], 10.0)
    horizontal = np.cos(np.deg2rad(10.0))
    assert horizontal / vertical_sensitivity(g) == pytest.approx(5.67, abs=0.05)


def test_look_vector_is_a_unit_vector_pointing_along_the_bearing():
    g = _Geom([0.0, 90.0, 180.0, 270.0], 0.0)
    v = look_vector(g)
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0)
    assert np.allclose(v[0], [0, 1, 0], atol=1e-12)      # due north
    assert np.allclose(v[1], [1, 0, 0], atol=1e-12)      # due east


def test_look_vector_tilts_up_with_the_beam():
    v = look_vector(_Geom([0.0], 10.0))
    assert v[0, 2] == pytest.approx(np.sin(np.deg2rad(10.0)))


def test_decompose_los_recovers_flow_toward_the_radar():
    """Ice flowing straight at the radar: LOS is the full horizontal motion."""
    g = _Geom([90.0], 0.0)                    # radar looks due east
    flow = 270.0                              # ice moves due west, at the radar
    los = np.array([0.010])
    assert decompose_los(los, g, flow)[0] == pytest.approx(0.010, abs=1e-9)


def test_decompose_los_refuses_a_near_perpendicular_flow_direction():
    """Dividing by a projection factor near zero manufactures numbers."""
    g = _Geom([90.0], 0.0)
    out = decompose_los(np.array([0.001]), g, flow_azimuth=0.0)   # flow due north
    assert np.isnan(out[0])


def test_decompose_los_accounts_for_an_uplift_component():
    g = _Geom([90.0], 10.0)
    pure = decompose_los(np.array([0.01]), g, 270.0, uplift_ratio=0.0)[0]
    with_uplift = decompose_los(np.array([0.01]), g, 270.0, uplift_ratio=0.5)[0]
    assert with_uplift != pure
