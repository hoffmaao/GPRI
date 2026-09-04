"""Sample coherence matrices, and the memory guards that keep them tractable."""
import numpy as np
import pytest

from gpri_tools.covariance import (coherence_from_interferograms, coherence_from_slcs,
                             regularize)
from gpri_tools.phaselink import phase_link


def _stack(n=4, a=40, r=40, theta=None, seed=0):
    """A coregistered SLC stack whose pairs carry a known phase difference."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2.0, n) if theta is None else np.asarray(theta)
    common = (rng.normal(size=(a, r)) + 1j * rng.normal(size=(a, r)))
    return np.stack([common * np.exp(1j * t) for t in theta]), theta


def test_shape_and_hermitian_unit_diagonal():
    slcs, _ = _stack()
    G = coherence_from_slcs(slcs, looks=(8, 8))
    assert G.shape[-2:] == (4, 4)
    assert np.allclose(G, np.conj(np.swapaxes(G, -1, -2)))
    assert np.allclose(np.diagonal(G, axis1=-2, axis2=-1), 1.0)


def test_recovers_the_imposed_phase():
    """Fully correlated scenes: the coherence phase is exactly theta_i - theta_j."""
    slcs, theta = _stack(n=4)
    G = coherence_from_slcs(slcs, looks=(8, 8))
    est = phase_link(G[2, 2], method="evd")
    assert np.allclose(np.angle(est), theta - theta[0], atol=1e-5)


def test_convention_matches_slc_i_times_conj_slc_j():
    slcs, theta = _stack(n=3)
    G = coherence_from_slcs(slcs, looks=(8, 8))
    assert np.angle(G[2, 2, 0, 1]) == pytest.approx(theta[0] - theta[1], abs=1e-5)


def test_unnormalised_diagonal_is_intensity():
    slcs, _ = _stack(n=2)
    G = coherence_from_slcs(slcs, looks=(8, 8), normalize=False)
    assert np.all(np.diagonal(G, axis1=-2, axis2=-1).real > 0)


def test_epoch_subset_is_honoured():
    slcs, _ = _stack(n=6)
    assert coherence_from_slcs(slcs, looks=(8, 8), epochs=[0, 2, 4]).shape[-1] == 3


def test_rejects_a_non_stack():
    with pytest.raises(ValueError, match="N, A, R"):
        coherence_from_slcs(np.zeros((4, 4)))


def test_memory_guard_fires_before_allocating():
    slcs, _ = _stack(n=60, a=200, r=200)
    with pytest.raises(MemoryError, match="GiB"):
        coherence_from_slcs(slcs, looks=(2, 2), max_gib=0.01)


def test_from_interferograms_fills_only_observed_pairs():
    n = 4
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    ifgs = np.stack([np.full((3, 3), np.exp(1j * 0.5))] * 3)
    with pytest.warns(UserWarning, match="observed"):
        G, mask = coherence_from_interferograms(ifgs, pairs, n)
    assert G.shape == (3, 3, n, n)
    assert mask[0, 1] and mask[1, 0] and not mask[0, 3]
    assert np.allclose(np.abs(G[..., 0, 1]), 1.0)
    assert np.allclose(G[..., 0, 3], 0.0)


def test_from_interferograms_is_hermitian():
    pairs = np.array([[0, 1], [0, 2]])
    ifgs = np.stack([np.full((2, 2), np.exp(1j * 0.3)),
                     np.full((2, 2), np.exp(-1j * 0.7))])
    with pytest.warns(UserWarning):
        G, _ = coherence_from_interferograms(ifgs, pairs, 3)
    assert np.allclose(G, np.conj(np.swapaxes(G, -1, -2)))
    assert np.angle(G[0, 0, 0, 1]) == pytest.approx(0.3)


def test_from_interferograms_applies_supplied_coherence():
    # n=2 with one pair is a fully observed matrix, so this must not warn
    pairs = np.array([[0, 1]])
    ifgs = np.full((1, 2, 2), np.exp(1j * 0.4))
    coh = np.full((1, 2, 2), 0.25)
    G, mask = coherence_from_interferograms(ifgs, pairs, 2, coherence=coh)
    assert mask.all()
    assert np.abs(G[0, 0, 0, 1]) == pytest.approx(0.25)
    assert np.angle(G[0, 0, 0, 1]) == pytest.approx(0.4)


def test_from_interferograms_checks_the_count():
    with pytest.raises(ValueError, match="interferograms"):
        coherence_from_interferograms(np.zeros((2, 2, 2), complex),
                                      np.array([[0, 1]]), 2)


def test_a_complete_network_does_not_warn():
    pairs = np.array([[0, 1]])
    ifgs = np.ones((1, 2, 2), complex)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coherence_from_interferograms(ifgs, pairs, 2)


def test_regularize_makes_a_singular_matrix_invertible():
    G = np.ones((5, 5), complex)          # perfectly coherent -> rank 1
    assert np.linalg.matrix_rank(G) == 1
    R = regularize(G, epsilon=1e-2)
    assert np.linalg.matrix_rank(R) == 5
    assert np.allclose(np.diagonal(R), 1.0)
