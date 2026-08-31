"""Network-consistent epoch screens, and the non-parametric turbulence screen."""
import numpy as np
import pytest
from datetime import datetime, timedelta

from gpri.aps import (EpochScreens, displacement_ramp_to_delta_n,
                      epoch_screen_correction, invert_screens,
                      remove_turbulence, turbulence_screen)
from gpri.atmosphere import PhaseScreen, remove_screen
from gpri.network import Network
from gpri.timeseries import wrap

R = np.linspace(300.0, 10400.0, 64)
AZ = np.linspace(-28.0, 51.0, 40)
LAMBDA = 0.017430


def _network(n_epochs=10, max_gap=1, cadence_min=2.0):
    t0 = datetime(2017, 8, 3, 22, 0, 0)
    epochs = [t0 + timedelta(minutes=cadence_min * k) for k in range(n_epochs)]
    pairs = [(i, j) for i in range(n_epochs) for j in range(i + 1, n_epochs)
             if j - i <= max_gap]
    return Network(epochs, pairs)


def _epoch_params(net, rng, ramp_scale=2e-4, coeff_scale=0.5, n_coeffs=2):
    """Smooth per-epoch [ramp, coeffs] with epoch 0 at zero."""
    t = net.times
    e = np.zeros((net.n_epochs, 1 + n_coeffs))
    e[:, 0] = ramp_scale * np.sin(2 * np.pi * t / max(t[-1], 1e-9))
    for c in range(n_coeffs):
        e[:, 1 + c] = coeff_scale * np.cos((c + 1) * np.pi * t / max(t[-1], 1e-9))
    return e - e[0]


def _pair_screens(net, e, noise=0.0, rng=None, quality=0.9):
    out = []
    for i, j in net.pairs:
        p = e[j] - e[i]
        if noise and rng is not None:
            p = p + noise * rng.normal(size=p.shape) * [2e-5, 0.05, 0.05][:p.size]
        out.append(PhaseScreen(p[1:], "linear", R, wavelength=LAMBDA,
                               ramp=p[0], quality=quality))
    return out


# ------------------------------------------------------------ invert_screens
def test_exact_recovery_on_a_daisy_chain():
    """With no noise and no smoothing the inversion is exact (square system)."""
    net = _network()
    e = _epoch_params(net, None)
    es = invert_screens(_pair_screens(net, e), net)
    assert isinstance(es, EpochScreens)
    assert np.allclose(es.params, e, atol=1e-10)
    assert np.allclose(es.residual_rms, 0.0, atol=1e-10)


def test_pair_screens_close_by_construction():
    """Reconstructed screens around a triangle sum to zero; independent fits don't."""
    rng = np.random.default_rng(0)
    net = _network(max_gap=2)                       # has triangles
    e = _epoch_params(net, rng)
    fitted = _pair_screens(net, e, noise=1.0, rng=rng)
    es = invert_screens(fitted, net)

    index = {tuple(pr): p for p, pr in enumerate(map(tuple, net.pairs))}
    p01, p12, p02 = index[(0, 1)], index[(1, 2)], index[(0, 2)]

    def closure(get):
        a, b, c = (get(p) for p in (p01, p12, p02))
        return np.abs(np.r_[a.ramp + b.ramp - c.ramp,
                            a.coeffs + b.coeffs - c.coeffs]).max()

    assert closure(es.pair_screen) < 1e-10
    assert closure(lambda p: fitted[p]) > 1e-4      # the noisy fits do not close


def test_redundant_pairs_average_the_noise_down():
    """Where invert_screens actually earns its keep: an i->i+2 network.

    A daisy chain gives a square system, so the inversion is a no-op there.
    With redundant pairs the joint solve is overdetermined and beats reading
    the chain pairs alone.
    """
    err = {}
    for max_gap in (1, 3):
        errs = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            net = _network(n_epochs=30, max_gap=max_gap)
            e = _epoch_params(net, rng)
            es = invert_screens(_pair_screens(net, e, noise=1.0, rng=rng), net)
            errs.append(np.abs(es.params - e).mean())
        err[max_gap] = np.mean(errs)
    assert err[3] < 0.75 * err[1]


def test_light_smoothing_helps_a_little_and_heavy_smoothing_hurts():
    """Documents the measured behaviour honestly: the drift is low-frequency,
    so a curvature penalty buys a few percent at best, and a heavy one biases
    the recovered atmosphere toward a straight line."""
    errs = {s: [] for s in (0.0, 2.0, 100.0)}
    for seed in range(6):
        rng = np.random.default_rng(seed)
        net = _network(n_epochs=40)
        e = _epoch_params(net, rng)
        noisy = _pair_screens(net, e, noise=1.0, rng=rng)
        for s in errs:
            es = invert_screens(noisy, net, smoothing=s)
            errs[s].append(np.abs(es.params - e).mean())
    mean = {s: np.mean(v) for s, v in errs.items()}
    assert mean[2.0] <= mean[0.0] * 1.02
    assert mean[100.0] > mean[0.0]


def test_failed_fits_are_excluded_not_zero_filled():
    net = _network(max_gap=2)
    e = _epoch_params(net, None)
    screens = _pair_screens(net, e)
    screens[3] = None
    es = invert_screens(screens, net)
    # redundancy from the i->i+2 pairs covers the hole exactly
    assert np.allclose(es.params, e, atol=1e-8)


def test_all_none_is_refused():
    net = _network()
    with pytest.raises(ValueError, match="every screen is None"):
        invert_screens([None] * net.n_pairs, net)


def test_mixed_models_are_refused():
    net = _network()
    e = _epoch_params(net, None)
    screens = _pair_screens(net, e)
    screens[0] = PhaseScreen([0.0], "constant", R, ramp=0.0)
    with pytest.raises(ValueError, match="one model at a time"):
        invert_screens(screens, net)


def test_length_mismatch_is_caught():
    net = _network()
    with pytest.raises(ValueError, match="screens for"):
        invert_screens([], net)


def test_delta_n_matches_the_injected_refractivity():
    net = _network()
    e = _epoch_params(net, None)
    es = invert_screens(_pair_screens(net, e), net)
    expected = e[:, 0] * LAMBDA / (4 * np.pi) * 1e6
    assert np.allclose(es.delta_n(), expected, atol=1e-10)
    assert "EpochScreens" in repr(es)


def test_correcting_with_epoch_screens_removes_the_atmosphere():
    """End to end: synthetic pair phases, corrected with reconstructed screens."""
    net = _network()
    e = _epoch_params(net, None)
    es = invert_screens(_pair_screens(net, e), net)
    for p, (i, j) in enumerate(net.pairs):
        atm = es.epoch_screen(j).evaluate() - es.epoch_screen(i).evaluate()
        corrected = remove_screen(atm.copy(), es.pair_screen(p))
        assert np.abs(corrected).max() < 1e-8


# --------------------------------------------------------------- turbulence
def _turb_field(shape=(60, 80), scale=0.8, seed=0):
    """A smooth random screen, well under a cycle."""
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.normal(size=shape), (6, 8))
    return scale * f / np.abs(f).max()


def test_turbulence_screen_recovers_a_smooth_field():
    truth = _turb_field()
    mask = np.zeros(truth.shape, bool)
    mask[::2, ::2] = True                          # dense stable ground
    screen, quality = turbulence_screen(truth, mask, sigma=(3, 4))
    good = quality > 0
    assert good.mean() > 0.9
    assert np.abs(screen - truth)[good].mean() < 0.1


def test_no_stable_ground_means_no_correction_not_extrapolation():
    truth = _turb_field()
    mask = np.zeros(truth.shape, bool)
    mask[:, :10] = True                            # rock only at near range
    screen, quality = turbulence_screen(truth, mask, sigma=(3, 4))
    assert np.all(screen[:, 40:] == 0.0)
    assert np.all(quality[:, 40:] == 0.0)
    assert np.any(screen[:, :10] != 0.0)


def test_ice_motion_cannot_contaminate_its_own_correction():
    """A moving patch off the mask must not appear in the screen."""
    truth = _turb_field(scale=0.3)
    phase = truth.copy()
    phase[25:35, 35:45] += 2.0                     # strong local motion on ice
    mask = np.ones(truth.shape, bool)
    mask[20:40, 30:50] = False                     # ice excluded from the mask
    screen, _ = turbulence_screen(phase, mask, sigma=(3, 4))
    # the screen under the patch comes from the rock around it
    assert np.abs(screen[28:32, 38:42] - truth[28:32, 38:42]).max() < 0.35
    assert np.abs(screen[28:32, 38:42] - 2.0).min() > 1.0


def test_disagreeing_phases_are_rejected_by_the_coherence_floor():
    rng = np.random.default_rng(2)
    phase = rng.uniform(-np.pi, np.pi, (40, 40))   # incoherent everywhere
    mask = np.ones(phase.shape, bool)
    screen, quality = turbulence_screen(phase, mask, sigma=(5, 5),
                                        min_coherence=0.5)
    assert (quality > 0).mean() < 0.2              # almost nowhere believed


def test_complex_input_and_the_one_step_wrapper():
    truth = _turb_field(scale=0.5)
    z = 3.0 * np.exp(1j * truth)
    mask = np.ones(truth.shape, bool)
    corrected, screen, quality = remove_turbulence(z, mask, sigma=(3, 4))
    assert np.iscomplexobj(corrected)
    assert np.allclose(np.abs(corrected), 3.0)     # magnitude untouched
    good = quality > 0
    assert np.abs(np.angle(corrected))[good].mean() < np.abs(truth)[good].mean()


def test_wrapped_input_stays_wrapped():
    truth = _turb_field(scale=0.5)
    mask = np.ones(truth.shape, bool)
    corrected, _, _ = remove_turbulence(wrap(truth), mask, sigma=(3, 4))
    assert np.all(np.abs(corrected) <= np.pi + 1e-9)


# ---------------------------------------------- displacement-domain correction
def _series(n_epochs=30, shape=(40, 60), seed=0):
    """Displacement with real motion on ice, drift error everywhere."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_epochs) / 30.0
    r = np.linspace(300.0, 10400.0, shape[1])
    rc = r - r.mean()
    # accumulated atmospheric drift: per-epoch constant + range ramp
    drift = np.zeros((n_epochs,) + shape)
    drift += (0.02 * np.cumsum(rng.normal(size=n_epochs)))[:, None, None]
    drift += (2e-6 * np.cumsum(rng.normal(size=n_epochs)))[:, None, None] \
        * rc[None, None, :]
    stable = np.zeros(shape, bool)
    stable[:12] = True
    d = drift.copy()
    d[:, 12:] += (0.05 * t)[:, None, None]         # real motion on ice
    return d, stable, drift, r, t


def test_epoch_screen_correction_removes_constant_and_ramp_drift():
    d, stable, drift, r, t = _series()
    out, coeffs = epoch_screen_correction(d, stable, r, model="linear")
    assert out.shape == d.shape
    assert coeffs.shape == (d.shape[0], 2)
    # stable ground is flat again
    assert np.abs(out[:, :12]).max() < 1e-9
    # and the real ice motion survived
    ice = out[:, 12:].mean(axis=(1, 2))
    assert np.polyfit(t, ice, 1)[0] == pytest.approx(0.05, rel=1e-6)


def test_constant_model_reproduces_reference_to_stable_mean():
    from gpri.timeseries import reference_to_stable
    d, stable, _, r, _ = _series(seed=1)
    out, _ = epoch_screen_correction(d, stable, r, model="constant")
    ref = reference_to_stable(d, stable, method="mean")
    assert np.allclose(out, ref, atol=1e-10)


def test_ramp_drift_is_invisible_to_plain_referencing():
    """The failure mode this function exists for."""
    from gpri.timeseries import reference_to_stable
    d, stable, drift, r, _ = _series(seed=2)
    ref = reference_to_stable(d, stable)
    scr, _ = epoch_screen_correction(d, stable, r, model="linear")
    # after only referencing, stable ground still carries the ramp drift
    assert np.abs(ref[:, :12]).max() > 1e-3
    assert np.abs(scr[:, :12]).max() < 1e-9


def test_displacement_ramp_to_delta_n_sign_and_scale():
    # -1 mm per km of range is +1 N-unit
    assert displacement_ramp_to_delta_n(-1e-6) == pytest.approx(1.0)


def test_epoch_screen_correction_input_checks():
    d, stable, _, r, _ = _series()
    with pytest.raises(ValueError, match="do not describe"):
        epoch_screen_correction(d[0], stable, r)
    with pytest.raises(ValueError, match="selects no pixels"):
        epoch_screen_correction(d, np.zeros_like(stable), r)
    with pytest.raises(ValueError, match="needs azimuth"):
        epoch_screen_correction(d, stable, r, model="planar")


def test_unwrapped_turbulence_smooths_values_not_phasors():
    """Displacement residuals are metres, not phase; no wrapping arithmetic."""
    truth = 0.5 + _turb_field(scale=0.2, seed=3)   # mean far from zero phase
    mask = np.ones(truth.shape, bool)
    screen, quality = turbulence_screen(truth, mask, sigma=(3, 4), wrapped=False)
    good = quality > 0
    assert np.abs(screen - truth)[good].mean() < 0.05
    # the wrapped path would have aliased a 0.5-radian offset differently near pi
    big = 3.0 * np.ones(truth.shape)               # |values| > pi
    screen2, _ = turbulence_screen(big, mask, sigma=(3, 4), wrapped=False)
    assert screen2[20, 40] == pytest.approx(3.0, abs=1e-6)
