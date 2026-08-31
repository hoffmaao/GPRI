"""Single-step pair-domain weighted LSQ, against the integrate-then-fit way."""
import numpy as np
import pytest
from datetime import datetime, timedelta

from gpri.diurnal import fit_harmonics
from gpri.network import Network
from gpri.pairlsq import (DIURNAL, SEMIDIURNAL, PairModelFit, fit_pairs,
                          pair_design, temporal_design)


def _network(n_epochs=200, cadence_min=8.0, max_gap=1):
    t0 = datetime(2017, 8, 3, 22, 0, 0)
    epochs = [t0 + timedelta(minutes=cadence_min * k) for k in range(n_epochs)]
    pairs = [(i, j) for i in range(n_epochs) for j in range(i + 1, n_epochs)
             if j - i <= max_gap]
    return Network(epochs, pairs)


def _model(t, rate=0.01, amp=0.004, phase=1.2, offset=0.03):
    return offset + rate * t + amp * np.cos(2 * np.pi * t / DIURNAL - phase)


def _pair_obs(net, series, noise=0.0, rng=None, sig=None):
    d = np.array([series[j] - series[i] for i, j in net.pairs])
    if noise and rng is not None:
        s = np.ones(net.n_pairs) if sig is None else np.asarray(sig)
        d = d + noise * s * rng.normal(size=d.shape[0])
    return d


# ------------------------------------------------------------------ designs
def test_constant_cancels_in_the_differencing():
    net = _network(50, cadence_min=30.0)
    G, names = pair_design(net.times, net.pairs)
    assert "1" not in names
    assert names == ["t^1", "cos1d", "sin1d"]
    assert G.shape == (net.n_pairs, 3)


def test_short_record_is_refused():
    net = _network(20, cadence_min=8.0)          # 2.5 h
    with pytest.raises(ValueError, match="not separable"):
        pair_design(net.times, net.pairs)


def test_covariates_ride_along_and_zero_ones_are_dropped():
    net = _network(50, cadence_min=30.0)
    N = np.sin(np.pi * net.times)
    G, names = pair_design(net.times, net.pairs,
                           covariates={"refractivity": N, "flat": np.ones(50)})
    assert "refractivity" in names
    assert "flat" not in names                    # constant covariate: gone too


# ----------------------------------------------------------------- recovery
def test_exact_recovery_and_offset_unobservability():
    net = _network()
    truth = _model(net.times)
    fit = fit_pairs(_pair_obs(net, truth), net)
    assert isinstance(fit, PairModelFit)
    assert fit.secular == pytest.approx(0.01, abs=1e-10)
    assert fit.amplitude(DIURNAL) == pytest.approx(0.004, abs=1e-10)
    assert fit.phase(DIURNAL) == pytest.approx(1.2, abs=1e-8)
    assert "1" not in fit.names                   # the offset never existed


def test_matches_two_step_fit_in_the_equal_weight_case():
    """Same model, same data: both estimators must agree on a clean chain."""
    net = _network()
    truth = _model(net.times)
    obs = _pair_obs(net, truth)
    single = fit_pairs(obs, net)
    d = np.concatenate([[0.0], np.cumsum(obs)])
    two = fit_harmonics(d, net.times)
    assert single.amplitude() == pytest.approx(float(two.amplitude()), abs=1e-9)
    assert single.secular == pytest.approx(float(two.secular), abs=1e-9)


def test_weighted_pair_fit_beats_integrate_then_fit_under_uneven_noise():
    """The Ohenhen argument: use the per-pair quality, whiten the problem."""
    err_single, err_two = [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        net = _network()
        truth = _model(net.times)
        sig = np.where(rng.random(net.n_pairs) < 0.3, 5.0, 1.0)   # bad pairs
        obs = _pair_obs(net, truth, noise=0.002, rng=rng, sig=sig)

        fit = fit_pairs(obs, net, weights=1.0 / sig ** 2)
        err_single.append(abs(fit.amplitude() - 0.004))

        d = np.concatenate([[0.0], np.cumsum(obs)])
        err_two.append(abs(float(fit_harmonics(d, net.times).amplitude()) - 0.004))
    assert np.mean(err_single) < 0.6 * np.mean(err_two)


def test_disconnected_network_still_constrains_rate_and_harmonic():
    """No reference epoch is needed: each component sees the same clock."""
    net = _network(200)
    pairs = [pr for pr in map(tuple, net.pairs) if pr != (99, 100)]  # cut it
    broken = Network(net.epochs, pairs)
    assert not broken.is_connected()
    truth = _model(broken.times)
    fit = fit_pairs(_pair_obs(broken, truth), broken)
    assert fit.amplitude() == pytest.approx(0.004, abs=1e-9)
    assert fit.secular == pytest.approx(0.01, abs=1e-9)


def test_semidiurnal_is_separable():
    net = _network()
    t = net.times
    truth = _model(t) + 0.001 * np.cos(2 * np.pi * t / SEMIDIURNAL - 0.4)
    fit = fit_pairs(_pair_obs(net, truth), net,
                    periods=(DIURNAL, SEMIDIURNAL))
    assert fit.amplitude(DIURNAL) == pytest.approx(0.004, abs=1e-9)
    assert fit.amplitude(SEMIDIURNAL) == pytest.approx(0.001, abs=1e-9)


def test_covariate_absorbs_an_atmosphere_shaped_signal():
    """Project the refractivity series out inside the fit."""
    rng = np.random.default_rng(3)
    net = _network()
    t = net.times
    N = np.cos(2 * np.pi * t / DIURNAL + 0.3) + 0.2 * rng.normal(size=t.size)
    truth = _model(t, amp=0.002) + 0.005 * N       # atmosphere rides on top
    naked = fit_pairs(_pair_obs(net, truth), net)
    fitted = fit_pairs(_pair_obs(net, truth), net, covariates={"N": N})
    assert abs(naked.amplitude() - 0.002) > 0.002  # badly contaminated
    assert fitted.amplitude() == pytest.approx(0.002, abs=2e-4)
    assert fitted.param("N") == pytest.approx(0.005, abs=5e-4)


# -------------------------------------------------------------- uncertainty
def test_reported_sigma_matches_the_empirical_scatter():
    """The error bars must mean what they say."""
    net = _network()
    truth = _model(net.times)
    amps, sigs = [], []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        obs = _pair_obs(net, truth, noise=0.0005, rng=rng)
        fit = fit_pairs(obs, net)
        amps.append(fit.amplitude())
        sigs.append(fit.amplitude_sigma())
    empirical = np.std(amps)
    predicted = np.mean(sigs)
    assert predicted == pytest.approx(empirical, rel=0.35)


def test_snr_flags_a_real_signal_and_not_noise():
    rng = np.random.default_rng(4)
    net = _network()
    strong = _pair_obs(net, _model(net.times, amp=0.01), noise=0.0005, rng=rng)
    silent = _pair_obs(net, _model(net.times, amp=0.0), noise=0.0005, rng=rng)
    assert fit_pairs(strong, net).snr() > 5.0
    assert fit_pairs(silent, net).snr() < 3.0


def test_broadcasts_over_pixels_with_shared_weights():
    net = _network(120, cadence_min=15.0)
    t = net.times
    amps = np.array([[0.001, 0.003], [0.005, 0.002]])
    series = 0.01 * t[:, None, None] + amps * np.cos(2 * np.pi * t / DIURNAL)[:, None, None]
    obs = np.stack([series[j] - series[i] for i, j in net.pairs])
    fit = fit_pairs(obs, net, weights=np.ones(net.n_pairs))
    assert fit.amplitude().shape == (2, 2)
    assert np.allclose(fit.amplitude(), amps, atol=1e-9)
    assert "PairModelFit" in repr(fit)


def test_per_pixel_weights_and_nans_take_the_slow_path():
    rng = np.random.default_rng(5)
    net = _network(120, cadence_min=15.0)
    truth = _model(net.times)
    obs = np.stack([_pair_obs(net, truth, noise=0.0005, rng=rng)
                    for _ in range(6)], axis=1)
    obs[10, 2] = np.nan                            # a hole
    w = np.ones(obs.shape)
    w[:, 4] *= 0.1
    fit = fit_pairs(obs, net, weights=w)
    assert fit.amplitude().shape == (6,)
    assert np.all(np.isfinite(fit.amplitude()))
    # per-pixel sigma on this chain is ~1 mm; test the ensemble, not each draw
    assert np.abs(fit.amplitude() - 0.004).mean() < 1.5e-3
    assert np.abs(fit.amplitude().mean() - 0.004) < 1e-3


def test_memory_guard_refuses_an_oversized_per_pixel_solve():
    net = _network(60, cadence_min=30.0)
    obs = np.zeros((net.n_pairs, 400, 400))
    w = np.ones(obs.shape)
    with pytest.raises(MemoryError, match="per-pixel weights"):
        fit_pairs(obs, net, weights=w, max_gib=0.001)


def test_pair_count_mismatch_is_caught():
    net = _network(50, cadence_min=30.0)
    with pytest.raises(ValueError, match="observations but the network"):
        fit_pairs(np.zeros(3), net)


def test_unknown_parameter_name_is_an_error():
    net = _network()
    fit = fit_pairs(_pair_obs(net, _model(net.times)), net)
    with pytest.raises(ValueError, match="no parameter"):
        fit.param("annual")
