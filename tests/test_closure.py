"""Closure-phase bias estimation, and the velocity blindness it cannot escape."""
import numpy as np
import pytest

from gpri.closure import (BiasModel, baseline_bins, closure_design_matrix,
                          closure_rms, correct_bias, estimate_bias)
from gpri.network import Network
from gpri.timeseries import closure_phase, triplets, wrap
from datetime import datetime, timedelta


def _network(n_epochs=8, max_gap=3, cadence_min=4.0):
    """Epochs on a fixed cadence, every pair out to `max_gap` steps."""
    t0 = datetime(2017, 8, 3, 22, 0, 0)
    epochs = [t0 + timedelta(minutes=cadence_min * k) for k in range(n_epochs)]
    pairs = [(i, j) for i in range(n_epochs) for j in range(i + 1, n_epochs)
             if j - i <= max_gap]
    return Network(epochs, pairs)


def _true_phase(net, rng, scale=0.3):
    """Pair phase from a per-epoch phase: closes exactly, by construction."""
    theta = rng.normal(0, scale, net.n_epochs)
    return np.array([theta[i] - theta[j] for i, j in net.pairs])


# ------------------------------------------------------------------ binning
def test_regular_cadence_gives_one_bin_per_distinct_baseline():
    net = _network(n_epochs=8, max_gap=3)
    index, centers = baseline_bins(net)
    assert len(centers) == 3                       # 1, 2 and 3 steps
    dt = np.abs(net.temporal_baselines())
    assert np.allclose(centers[index], dt)


def test_explicit_bin_edges_are_honoured():
    net = _network()
    edges = [0.0, 0.005, 0.02]
    index, centers = baseline_bins(net, bins=edges)
    assert len(centers) == 2
    assert index.min() >= 0 and index.max() <= 1


# ------------------------------------------------------------ design matrix
def test_design_matrix_rows_are_plus_plus_minus():
    net = _network(n_epochs=5, max_gap=4)
    index, centers = baseline_bins(net)
    trip = triplets(net)
    C = closure_design_matrix(index, trip, n_bins=len(centers))
    assert C.shape == (len(trip), len(centers))
    # each row sums to +1: two legs in, one long leg out
    assert np.allclose(C.sum(axis=1), 1.0)


def test_bias_linear_in_baseline_is_invisible_to_closure():
    """The null space, asserted directly: b ~ dt produces zero closure."""
    net = _network(n_epochs=6, max_gap=5)
    index, centers = baseline_bins(net)
    trip = triplets(net)
    C = closure_design_matrix(index, trip, n_bins=len(centers))
    assert np.allclose(C @ centers, 0.0, atol=1e-12)


# --------------------------------------------------------------- estimation
def test_recovers_an_injected_nonlinear_bias():
    rng = np.random.default_rng(0)
    net = _network(n_epochs=10, max_gap=4)
    index, centers = baseline_bins(net)

    # a bias that saturates with baseline -- the classic short-baseline shape --
    # with its linear component removed, since that part is unknowable
    truth = 0.4 * (1.0 - np.exp(-centers / centers.mean()))
    truth = truth - centers * (truth @ centers) / (centers @ centers)

    psi = _true_phase(net, rng) + truth[index]
    model = estimate_bias(psi, net, wavelength=0.01743)

    assert isinstance(model, BiasModel)
    assert np.allclose(model.bias, truth, atol=1e-8)


def test_correction_drives_closure_to_zero():
    rng = np.random.default_rng(1)
    net = _network(n_epochs=10, max_gap=4)
    index, centers = baseline_bins(net)
    truth = 0.5 * np.sqrt(centers / centers.max())
    truth = truth - centers * (truth @ centers) / (centers @ centers)

    psi = wrap(_true_phase(net, rng) + truth[index])
    before = closure_rms(psi, net)
    after = closure_rms(correct_bias(psi, estimate_bias(psi, net)), net)
    assert before > 0.05
    assert after < 1e-6


def test_velocity_bias_is_not_recovered_and_we_say_so():
    """A purely linear-in-baseline bias is a velocity, and closure cannot see it."""
    rng = np.random.default_rng(2)
    net = _network(n_epochs=10, max_gap=4)
    index, centers = baseline_bins(net)

    linear = 3.0 * centers                      # exactly a constant velocity
    psi = _true_phase(net, rng) + linear[index]
    model = estimate_bias(psi, net)

    assert model.velocity_blind is True
    assert np.allclose(model.bias, 0.0, atol=1e-8)   # nothing recovered
    assert closure_rms(psi, net) < 1e-10             # because it closes perfectly


def test_per_pixel_estimation_broadcasts():
    rng = np.random.default_rng(3)
    net = _network(n_epochs=8, max_gap=3)
    index, centers = baseline_bins(net)
    truth = np.array([0.3, -0.1, 0.05])
    truth = truth - centers * (truth @ centers) / (centers @ centers)

    psi = np.empty((net.n_pairs, 4, 5))
    for a in range(4):
        for b in range(5):
            psi[:, a, b] = _true_phase(net, rng) + truth[index]

    model = estimate_bias(psi, net)
    assert model.bias.shape == (3, 4, 5)
    assert np.allclose(model.bias, truth[:, None, None], atol=1e-8)

    corrected = correct_bias(psi, model)
    assert corrected.shape == psi.shape


def test_complex_input_is_corrected_in_place_on_the_phase():
    rng = np.random.default_rng(4)
    net = _network(n_epochs=8, max_gap=3)
    psi = _true_phase(net, rng)
    z = 2.5 * np.exp(1j * psi)
    model = estimate_bias(psi, net)
    out = correct_bias(z, model)
    assert np.iscomplexobj(out)
    assert np.allclose(np.abs(out), 2.5)          # magnitude untouched


def test_daisy_chain_has_no_closure_and_refuses_loudly():
    net = _network(n_epochs=6, max_gap=1)          # sequential only
    assert triplets(net).size == 0
    with pytest.raises(ValueError, match="no closed triangles"):
        estimate_bias(np.zeros(net.n_pairs), net)


def test_robust_iterations_survive_an_outlier_triangle():
    rng = np.random.default_rng(5)
    net = _network(n_epochs=12, max_gap=4)
    index, centers = baseline_bins(net)
    truth = np.array([0.4, 0.2, 0.05, -0.1])
    truth = truth - centers * (truth @ centers) / (centers @ centers)

    psi = _true_phase(net, rng) + truth[index]
    psi[3] += 2.0 * np.pi * 0.4                    # one corrupted interferogram

    plain = estimate_bias(psi, net)
    robust = estimate_bias(psi, net, robust=4)
    assert (np.abs(robust.bias - truth).max()
            <= np.abs(plain.bias - truth).max() + 1e-9)


def test_bias_model_reports_displacement_and_repr():
    rng = np.random.default_rng(6)
    net = _network(n_epochs=8, max_gap=3)
    model = estimate_bias(_true_phase(net, rng), net, wavelength=0.01743)
    d = model.displacement()
    assert d.shape == model.bias.shape
    assert "velocity_blind=True" in repr(model)
