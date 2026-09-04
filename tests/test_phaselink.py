"""Phase-linking estimators: EVD, eigenSAR, EMI and the exact ML solution."""
import numpy as np
import pytest

from gpri_tools.phaselink import (coherence_magnitude, emi, eigensar, evd, ml_cost,
                            mle, phase_link, temporal_coherence)

METHODS = ["evd", "eigensar", "emi", "mle"]


def _truth(n=8):
    return np.linspace(0.0, 6.0, n)


def _rank_one(theta):
    """Noise-free coherence matrix in the GAMMA convention (Gamma_ij = theta_i - theta_j)."""
    lam = np.exp(1j * np.asarray(theta))
    return lam[:, None] * np.conj(lam[None, :])


def _speckle(theta, looks=200, tau=3.0, seed=0):
    """Sample coherence from `looks` realisations with exponential temporal decorrelation."""
    rng = np.random.default_rng(seed)
    n = len(theta)
    i = np.arange(n)
    mag = np.exp(-np.abs(i[:, None] - i[None, :]) / tau)
    C = mag * _rank_one(theta)
    L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
    z = L @ (rng.normal(size=(n, looks)) + 1j * rng.normal(size=(n, looks))) / np.sqrt(2)
    G = (z @ z.conj().T) / looks
    d = np.sqrt(np.abs(np.diag(G)))
    return G / np.outer(d, d)


def _rms_deg(est, theta):
    ref = np.exp(1j * (np.asarray(theta) - theta[0]))
    return np.rad2deg(np.sqrt(np.mean(np.angle(est * np.conj(ref)) ** 2)))


@pytest.mark.parametrize("method", METHODS)
def test_exact_on_noiseless_rank_one(method):
    """With a perfect rank-one matrix every estimator must return the truth."""
    th = _truth()
    assert _rms_deg(phase_link(_rank_one(th), method=method), th) < 1e-8


@pytest.mark.parametrize("method", METHODS)
def test_reference_epoch_has_zero_phase(method):
    est = phase_link(_speckle(_truth()), method=method, reference=2)
    assert np.angle(est[2]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_returns_unit_modulus(method):
    est = phase_link(_speckle(_truth()), method=method)
    assert np.allclose(np.abs(est), 1.0)


@pytest.mark.parametrize("method", METHODS)
def test_recovers_phase_through_speckle(method):
    th = _truth()
    assert _rms_deg(phase_link(_speckle(th), method=method), th) < 15.0


def test_emi_beats_evd_when_coherence_varies():
    """EVD weights every pair equally; EMI does not, and that is the point."""
    th = _truth()
    G = _speckle(th, looks=60, tau=2.0, seed=3)
    assert _rms_deg(emi(G), th) < _rms_deg(evd(G), th)


def test_mle_does_not_increase_the_ml_cost():
    G = _speckle(_truth(), looks=60, tau=2.0, seed=1)
    start = emi(G, reference=None)
    assert ml_cost(G, mle(G, reference=None)) <= ml_cost(G, start) + 1e-9


def test_mle_improves_a_deliberately_bad_start():
    rng = np.random.default_rng(5)
    G = _speckle(_truth(), looks=60, seed=2)
    bad = np.exp(1j * rng.uniform(-np.pi, np.pi, G.shape[-1]))
    assert ml_cost(G, mle(G, reference=None, init=bad)) < ml_cost(G, bad)


def test_temporal_coherence_is_one_for_a_perfect_fit():
    th = _truth()
    G = _rank_one(th)
    assert temporal_coherence(G, phase_link(G, method="emi")) == pytest.approx(1.0)


def test_temporal_coherence_falls_for_random_phase():
    rng = np.random.default_rng(0)
    n = 12
    G = _rank_one(rng.uniform(-np.pi, np.pi, n))
    wrong = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    assert temporal_coherence(G, wrong) < 0.6


def test_temporal_coherence_honours_the_observed_mask():
    th = _truth(6)
    G = _rank_one(th)
    mask = np.zeros((6, 6), bool)
    mask[0, 1] = mask[1, 0] = True
    assert temporal_coherence(G, phase_link(G, method="evd"), mask=mask) == pytest.approx(1.0)


@pytest.mark.parametrize("method", METHODS)
def test_batched_over_spatial_axes(method):
    th = _truth(6)
    G = np.broadcast_to(_speckle(th, seed=4), (3, 4, 6, 6)).copy()
    est = phase_link(G, method=method)
    assert est.shape == (3, 4, 6)
    assert temporal_coherence(G, est).shape == (3, 4)
    assert np.allclose(est[0, 0], est[2, 3])


def test_eigensar_rejects_pixels_without_an_eigen_gap():
    G = _speckle(_truth(), looks=60, seed=7)
    assert np.all(np.isnan(eigensar(G, min_eigen_gap=0.999)))


def test_eigensar_survives_a_singular_all_ones_magnitude():
    """|Gamma| is exactly singular for a perfectly coherent stack."""
    est = eigensar(_rank_one(_truth()))
    assert np.all(np.isfinite(est))


def test_coherence_magnitude_has_a_unit_diagonal():
    G = _speckle(_truth())
    m = coherence_magnitude(G, floor=0.3, shrink=0.01)
    assert np.allclose(np.diag(m), 1.0)
    assert m.min() >= 0.0


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        phase_link(_rank_one(_truth()), method="nope")


def test_non_square_input_is_rejected():
    with pytest.raises(ValueError, match="N, N"):
        phase_link(np.zeros((4, 5), complex))
