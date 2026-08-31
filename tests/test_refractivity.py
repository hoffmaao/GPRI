"""Refractivity from meteorology, and the per-epoch refractivity series."""
import numpy as np
import pytest
from datetime import datetime, timedelta

from gpri.atmosphere import PhaseScreen, delta_refractivity
from gpri.network import Network
from gpri.refractivity import (MetRecord, delta_n_from_met, dry_refractivity,
                               invert_refractivity, ramp_from_delta_n,
                               refractivity, refractivity_phase,
                               saturation_vapour_pressure, screens_to_delta_n,
                               sensitivity, vapour_pressure, wet_refractivity)

LAMBDA = 0.017430          # GPRI-II Ku band, metres
SWATH = 16882.851 - 300.140


# ------------------------------------------------------------- moist air
def test_saturation_vapour_pressure_matches_known_values():
    # Buck (1981): 6.112 hPa at 0 C, 23.4 hPa at 20 C
    assert saturation_vapour_pressure(0.0) == pytest.approx(6.112, abs=0.01)
    assert saturation_vapour_pressure(20.0) == pytest.approx(23.39, abs=0.05)
    assert saturation_vapour_pressure(100.0) == pytest.approx(1013.0, rel=0.01)


def test_ice_curve_is_below_the_water_curve_when_it_should_be():
    """At -10 C the ice value is about 9 % lower; getting it wrong is 0.2 N."""
    w = saturation_vapour_pressure(-10.0, over="water")
    i = saturation_vapour_pressure(-10.0, over="ice")
    assert i < w
    assert (w - i) / w == pytest.approx(0.09, abs=0.02)
    # they coincide at the triple point
    assert saturation_vapour_pressure(0.0, "water") == pytest.approx(
        saturation_vapour_pressure(0.0, "ice"), abs=0.01)


def test_relative_humidity_accepts_fraction_or_percent():
    assert vapour_pressure(10.0, 0.5) == pytest.approx(vapour_pressure(10.0, 50.0))


def test_refractivity_at_standard_conditions():
    """N ~ 315 at sea level, 15 C, 60 % RH -- the textbook number."""
    assert refractivity(1013.25, 15.0, 0.60) == pytest.approx(315.0, abs=5.0)


def test_dry_term_dominates_but_wet_term_is_the_variable_one():
    N = refractivity(880.0, 5.0, 0.70)
    dry = dry_refractivity(880.0, 5.0)
    wet = wet_refractivity(5.0, 0.70)
    assert dry + wet == pytest.approx(N)
    assert wet / N < 0.12                       # small in absolute terms
    # but far more responsive: 10 % RH moves N more than 1 hPa of pressure
    assert abs(refractivity(880.0, 5.0, 0.80) - N) > abs(
        refractivity(881.0, 5.0, 0.70) - N)


def test_wet_refractivity_needs_humidity_or_vapour_pressure():
    with pytest.raises(ValueError, match="relative_humidity or vapour_hpa"):
        wet_refractivity(5.0)


def test_vapour_pressure_can_be_given_directly():
    e = vapour_pressure(5.0, 0.70)
    assert wet_refractivity(5.0, vapour_hpa=e) == pytest.approx(
        wet_refractivity(5.0, 0.70))


# --------------------------------------------------------------- phase link
def test_ramp_and_delta_refractivity_are_inverses():
    """ramp_from_delta_n must undo gpri.atmosphere.delta_refractivity."""
    dN = 2.5                                     # N-units
    ramp = ramp_from_delta_n(dN, LAMBDA)
    assert delta_refractivity(ramp, LAMBDA) * 1e6 == pytest.approx(dN)


def test_one_N_unit_puts_twelve_radians_across_the_bakerbend_swath():
    """The number that motivates the whole atmosphere module."""
    phi = ramp_from_delta_n(1.0, LAMBDA) * SWATH
    assert phi == pytest.approx(11.96, abs=0.1)


def test_refractivity_phase_is_zero_at_the_reference_range():
    r = np.linspace(300.0, 16883.0, 64)
    phi = refractivity_phase(1.0, r, LAMBDA)
    assert phi[0] == pytest.approx(0.0)
    assert phi[-1] > 0
    assert np.allclose(np.diff(phi, 2), 0.0, atol=1e-9)     # strictly linear


def test_refractivity_phase_broadcasts_over_epochs():
    r = np.linspace(300.0, 16883.0, 32)
    dN = np.array([0.0, 1.0, -2.0])
    phi = refractivity_phase(dN, r, LAMBDA)
    assert phi.shape == (3, 32)
    assert np.allclose(phi[0], 0.0)
    assert np.allclose(phi[2], -2.0 * phi[1])


def test_sensitivity_reports_the_documented_numbers():
    s = sensitivity(880.0, 5.0, 0.70, LAMBDA, SWATH)
    assert s["dN_dT"] == pytest.approx(1.01, abs=0.05)
    assert s["dN_dT_dry"] == pytest.approx(-1.09, abs=0.05)
    assert s["dN_dP"] == pytest.approx(0.279, abs=0.01)
    assert s["dN_dRH1"] == pytest.approx(0.421, abs=0.02)
    assert s["dN_dRH10"] == pytest.approx(4.21, abs=0.1)
    assert s["phase_dN_dRH10"] == pytest.approx(50.3, abs=1.0)


def test_temperature_sensitivity_flips_sign_with_the_humidity_assumption():
    """Warming at fixed RH lengthens the path; warming at fixed vapour shortens it.

    The single easiest way to make an atmospheric correction worse than none.
    """
    s = sensitivity(880.0, 5.0, 0.70, LAMBDA, SWATH)
    assert s["dN_dT"] > 0 > s["dN_dT_dry"]


def test_humidity_dominates_temperature_and_pressure():
    s = sensitivity(880.0, 5.0, 0.70, LAMBDA, SWATH)
    assert abs(s["phase_dN_dRH10"]) > abs(s["phase_dN_dP"])
    assert abs(s["phase_dN_dRH10"]) > abs(s["phase_dN_dT"])
    assert abs(s["phase_dN_dRH10"]) > 2 * np.pi        # multiple fringes


# -------------------------------------------------------------- MetRecord
def test_met_record_splits_dry_and_wet():
    m = MetRecord(880.0, 5.0, 0.70)
    assert m.N == pytest.approx(m.N_dry + m.N_wet)
    assert m.N_dry > m.N_wet
    assert "MetRecord" in repr(m)


def test_delta_n_from_met_signs_a_wetter_atmosphere_positive():
    dry = MetRecord(880.0, 5.0, 0.40)
    wet = MetRecord(880.0, 5.0, 0.90)
    assert delta_n_from_met(dry, wet) > 0
    assert delta_n_from_met(wet, dry) == pytest.approx(-delta_n_from_met(dry, wet))


def test_delta_n_from_met_accepts_plain_tuples():
    assert delta_n_from_met((880.0, 5.0, 0.40), (880.0, 5.0, 0.90)) > 0


# -------------------------------------------------- per-epoch N from a network
def _network(n_epochs=6, max_gap=2):
    t0 = datetime(2017, 8, 3, 22, 0, 0)
    ep = [t0 + timedelta(minutes=4 * k) for k in range(n_epochs)]
    pairs = [(i, j) for i in range(n_epochs) for j in range(i + 1, n_epochs)
             if j - i <= max_gap]
    return Network(ep, pairs)


def test_per_epoch_refractivity_is_recovered_from_pair_observations():
    net = _network()
    truth = np.array([0.0, 1.2, -0.4, 0.9, 2.1, -1.5])
    obs = np.array([truth[j] - truth[i] for i, j in net.pairs])
    got = invert_refractivity(obs, net)
    assert got.shape[0] == net.n_epochs
    assert np.allclose(got.ravel() - got.ravel()[0], truth - truth[0], atol=1e-8)


def test_per_epoch_inversion_averages_noise_down():
    """Solving n_epochs unknowns from n_pairs observations beats taking pairs raw."""
    rng = np.random.default_rng(0)
    net = _network(n_epochs=12, max_gap=4)
    truth = rng.normal(0, 1.5, net.n_epochs)
    truth -= truth[0]
    clean = np.array([truth[j] - truth[i] for i, j in net.pairs])
    obs = clean + rng.normal(0, 0.3, net.n_pairs)

    got = invert_refractivity(obs, net).ravel()
    got -= got[0]
    resid = np.array([got[j] - got[i] for i, j in net.pairs]) - clean
    assert np.std(resid) < np.std(obs - clean)


def test_per_epoch_refractivity_keeps_spatial_axes():
    net = _network()
    truth = np.array([0.0, 1.0, -1.0, 0.5, 0.25, -0.75])
    obs = np.array([truth[j] - truth[i] for i, j in net.pairs])
    got = invert_refractivity(np.broadcast_to(obs[:, None, None], (net.n_pairs, 3, 4)),
                              net)
    assert got.shape == (net.n_epochs, 3, 4)


# ------------------------------------------------------- screens -> N-units
def test_screens_convert_to_n_units():
    r = np.linspace(300.0, 16883.0, 16)
    dN = 1.7
    screen = PhaseScreen([0.0], "constant", r, wavelength=LAMBDA,
                         ramp=ramp_from_delta_n(dN, LAMBDA))
    assert screens_to_delta_n([screen])[0] == pytest.approx(dN)


def test_missing_screens_come_back_as_nan_not_zero():
    r = np.linspace(300.0, 16883.0, 8)
    s = PhaseScreen([0.0], "constant", r, wavelength=LAMBDA, ramp=0.0)
    out = screens_to_delta_n([s, None, s])
    assert np.isnan(out[1]) and np.isfinite(out[0])


def test_screen_without_a_wavelength_is_an_error():
    r = np.linspace(300.0, 16883.0, 8)
    s = PhaseScreen([0.0], "constant", r, ramp=1e-3)
    with pytest.raises(ValueError, match="no wavelength"):
        screens_to_delta_n([s])
