"""Sign conventions, network inversion, stacking, and closure phase.

The sign tests here are the point of this file.  A flipped sign in InSAR
produces a perfectly plausible-looking time series that says the glacier is
advancing when it is retreating, so the convention is pinned down from first
principles rather than asserted.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from gpri_tools.network import Network
from gpri_tools.timeseries import (TimeSeries, closure_phase, closure_residual_mask,
                             displacement_from_phases, invert_network,
                             los_displacement, phase_from_los,
                             reference_to_stable, stack_velocity, triplets,
                             wrap)

LAM = 0.017430          # GPRI-II Ku band, metres


def _net(n=5, pairs=None, step_minutes=2):
    t0 = datetime(2017, 8, 3, 22, 0, 0)
    epochs = [t0 + timedelta(minutes=step_minutes * i) for i in range(n)]
    if pairs is None:
        pairs = [(i, i + 1) for i in range(n - 1)]
    return Network(epochs, pairs)


def _forward(d, net, lam=LAM):
    """Truth displacement -> GAMMA-convention pair phase.

    Derived from first principles: a target moving toward the radar by ``d``
    shortens the range by ``d``, so its SLC phase ``-4 pi r / lambda`` rises by
    ``4 pi d / lambda``.  GAMMA's pair phase is ``theta_i - theta_j``.
    """
    theta = 4.0 * np.pi * np.asarray(d) / lam
    return np.array([theta[i] - theta[j] for i, j in net.pairs])


# ------------------------------------------------------------ sign convention
def test_motion_toward_the_radar_is_positive():
    d = np.array([0.0, 0.004])                       # 4 mm toward the radar
    net = _net(2)
    psi = _forward(d, net)
    assert los_displacement(psi, LAM) == pytest.approx([0.004])


def test_motion_away_from_the_radar_is_negative():
    d = np.array([0.0, -0.004])
    psi = _forward(d, _net(2))
    assert los_displacement(psi, LAM) == pytest.approx([-0.004])


def test_one_fringe_is_half_a_wavelength():
    assert los_displacement(2 * np.pi, LAM) == pytest.approx(-LAM / 2)


def test_phase_and_displacement_are_inverses():
    d = np.array([-0.01, 0.0, 0.003])
    assert np.allclose(los_displacement(phase_from_los(d, LAM), LAM), d)


def test_displacement_from_phases_matches_the_pairwise_route():
    """Phase linking output and network inversion must agree exactly."""
    net = _net(6)
    d = np.array([0.0, 1e-3, 2.5e-3, 2e-3, 4e-3, 5e-3])
    theta = 4.0 * np.pi * d / LAM
    direct = displacement_from_phases(np.exp(1j * theta), LAM, reference=0)
    ts = invert_network(los_displacement(_forward(d, net), LAM), net, reference=0)
    assert np.allclose(direct, d)
    assert np.allclose(ts.displacement, d)


# ---------------------------------------------------------------- inversion
def test_inversion_recovers_a_known_series():
    net = _net(6)
    d = np.array([0.0, 1e-3, 2.5e-3, 2e-3, 4e-3, 5e-3])
    ts = invert_network(los_displacement(_forward(d, net), LAM), net)
    assert np.allclose(ts.displacement, d, atol=1e-12)
    assert np.allclose(np.nan_to_num(ts.residual), 0.0, atol=1e-12)


def test_inversion_broadcasts_over_pixels():
    net = _net(5)
    d = np.array([0.0, 1e-3, 2e-3, 3e-3, 4e-3])
    obs = los_displacement(_forward(d, net), LAM)
    ts = invert_network(np.repeat(obs[:, None], 7, axis=1), net)
    assert ts.displacement.shape == (5, 7)
    assert np.allclose(ts.displacement, d[:, None])


def test_incremental_parameterisation_agrees():
    net = _net(6)
    d = np.array([0.0, 1e-3, 2.5e-3, 2e-3, 4e-3, 5e-3])
    obs = los_displacement(_forward(d, net), LAM)
    a = invert_network(obs, net, incremental=False).displacement
    b = invert_network(obs, net, incremental=True).displacement
    assert np.allclose(a, b)


def test_l1_shrugs_off_one_corrupted_pair():
    """A whole-cycle unwrapping error must not smear across the whole series."""
    net = _net(9, pairs=[(i, j) for i in range(9) for j in range(i + 1, 9) if j - i <= 3])
    d = np.linspace(0, 8e-3, 9)
    obs = los_displacement(_forward(d, net), LAM)
    obs[4] += LAM / 2                      # one full fringe of error
    l2 = invert_network(obs, net, method="lstsq").displacement
    l1 = invert_network(obs, net, method="l1", iterations=25).displacement
    assert np.max(np.abs(l1 - d)) < np.max(np.abs(l2 - d))


def test_weights_are_honoured():
    net = _net(4, pairs=[(0, 1), (1, 2), (2, 3), (0, 3)])
    d = np.array([0.0, 1e-3, 2e-3, 3e-3])
    obs = los_displacement(_forward(d, net), LAM)
    obs[3] += 5e-3                                    # bad long pair
    w = np.array([1.0, 1.0, 1.0, 1e-6])               # down-weight it
    ts = invert_network(obs, net, weights=w, method="wls")
    assert np.allclose(ts.displacement, d, atol=1e-6)


def test_disconnected_network_warns():
    net = Network([datetime(2017, 1, 1) + timedelta(days=i) for i in range(4)],
                  [(0, 1), (2, 3)])
    with pytest.warns(UserWarning, match="disconnected"):
        invert_network(np.zeros(2), net)


def test_wrong_observation_count_is_rejected():
    with pytest.raises(ValueError, match="observations"):
        invert_network(np.zeros(3), _net(5))


def test_smoothing_damps_a_noisy_series():
    rng = np.random.default_rng(0)
    net = _net(12)
    d = np.linspace(0, 1e-2, 12)
    obs = los_displacement(_forward(d, net), LAM) + rng.normal(0, 3e-4, net.n_pairs)
    rough = invert_network(obs, net, method="lstsq").displacement
    smooth = invert_network(obs, net, method="smooth", smoothing=50.0).displacement
    curv = lambda x: np.sum(np.diff(x, 2) ** 2)
    assert curv(smooth) < curv(rough)


# ----------------------------------------------------------------- stacking
def test_stack_velocity_recovers_a_known_rate():
    net = _net(10, step_minutes=60)
    rate = 2e-3                                       # m/day toward the radar
    d = rate * net.times
    v = stack_velocity(los_displacement(_forward(d, net), LAM), net)
    assert v == pytest.approx(rate, rel=1e-9)


def test_stack_velocity_masks_thin_pixels():
    net = _net(4)
    obs = np.full((net.n_pairs, 2), np.nan)
    obs[:, 0] = los_displacement(_forward(1e-3 * net.times, net), LAM)
    v = stack_velocity(obs, net, min_pairs=2)
    assert np.isfinite(v[0]) and np.isnan(v[1])


def test_timeseries_velocity_matches_a_linear_trend():
    net = _net(8, step_minutes=180)
    rate = 1.5e-3
    ts = TimeSeries(net.times, rate * net.times)
    assert ts.velocity() == pytest.approx(rate, rel=1e-9)


# ------------------------------------------------------------ closure phase
def _triangle_net():
    return _net(3, pairs=[(0, 1), (1, 2), (0, 2)])


def test_daisy_chain_has_no_triangles():
    assert triplets(_net(5)).shape == (0, 3)


def test_triangle_is_found():
    assert triplets(_triangle_net()).tolist() == [[0, 1, 2]]


def test_closure_is_zero_for_consistent_phase():
    net = _triangle_net()
    psi = _forward(np.array([0.0, 1e-3, 2e-3]), net)
    assert np.allclose(closure_phase(psi, net), 0.0, atol=1e-9)


def test_closure_detects_an_unwrapping_error():
    net = _triangle_net()
    psi = _forward(np.array([0.0, 1e-3, 2e-3]), net)
    psi[1] += 1.0                            # 1 rad of inconsistency
    assert np.abs(closure_phase(psi, net))[0] == pytest.approx(1.0, abs=1e-9)


def test_closure_mask_flags_the_bad_pixel():
    net = _triangle_net()
    good = _forward(np.array([0.0, 1e-3, 2e-3]), net)
    psi = np.stack([good, good], axis=1)
    psi[1, 1] += 2.0
    assert closure_residual_mask(psi, net, threshold=1.0).tolist() == [True, False]


def test_wrap_is_in_range():
    x = np.array([-3 * np.pi, 0.0, 3 * np.pi, 7.0])
    assert np.all(wrap(x) > -np.pi - 1e-12) and np.all(wrap(x) <= np.pi + 1e-12)


# ------------------------------------------------------- common-mode removal
class TestReferenceToStable:
    """Tying a series to stable ground -- the step whose absence looks like signal."""

    @staticmethod
    def _scene(n_epochs=48, shape=(20, 20), common=None, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(n_epochs) / 24.0
        if common is None:
            common = 0.03 * np.cos(2 * np.pi * t)      # a diurnal common mode
        d = np.broadcast_to(common[:, None, None], (n_epochs,) + shape).copy()
        stable = np.zeros(shape, bool)
        stable[:8] = True
        # real motion, on the moving half only
        d[:, 8:] += (0.01 * t)[:, None, None]
        d += rng.normal(0, 1e-5, d.shape)
        return d, stable, common, t

    def test_removes_a_scene_wide_common_mode(self):
        d, stable, common, _ = self._scene()
        out = reference_to_stable(d, stable)
        assert np.abs(out[:, :8]).max() < 1e-3
        assert np.abs(d[:, :8]).max() > 0.02      # it was there before

    def test_returns_the_offset_it_subtracted(self):
        d, stable, common, _ = self._scene()
        out, offset = reference_to_stable(d, stable, return_offset=True)
        assert offset.shape == (d.shape[0],)
        assert np.allclose(offset, common, atol=1e-4)
        assert np.allclose(out, d - offset[:, None, None])

    def test_real_differential_motion_survives(self):
        """Referencing must remove the common mode, not the signal."""
        d, stable, _, t = self._scene()
        out = reference_to_stable(d, stable)
        moving = out[:, 8:].mean(axis=(1, 2))
        assert np.polyfit(t, moving, 1)[0] == pytest.approx(0.01, rel=1e-2)

    def test_median_resists_a_few_moving_pixels_in_the_reference(self):
        d, stable, _, t = self._scene()
        d[:, 0, :3] += (5.0 * t)[:, None]          # 3 bad pixels of 160
        med = reference_to_stable(d, stable, method="median")
        mean = reference_to_stable(d, stable, method="mean")
        truth = reference_to_stable(d, stable)[:, 8:]
        assert (np.abs(med[:, 8:] - truth).max()
                < np.abs(mean[:, 8:] - truth).max())

    def test_an_empty_reference_mask_is_refused(self):
        d, _, _, _ = self._scene()
        with pytest.raises(ValueError, match="selects no pixels"):
            reference_to_stable(d, np.zeros(d.shape[1:], bool))

    def test_mask_shape_mismatch_is_caught(self):
        d, _, _, _ = self._scene()
        with pytest.raises(ValueError, match="does not match"):
            reference_to_stable(d, np.ones((3, 3), bool))

    def test_all_nan_epoch_does_not_poison_the_series(self):
        d, stable, _, _ = self._scene()
        d[5] = np.nan
        out = reference_to_stable(d, stable)
        assert np.isfinite(out[np.arange(len(d)) != 5]).all()
