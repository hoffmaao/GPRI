"""PS interpolation unwrapping (Chen, Zebker & Knight 2015)."""
import numpy as np
import pytest

from gpri.psinterp import (amplitude_dispersion, interpolate_ps, ps_density,
                           select_ps, unwrap_sparse, unwrap_with_ps)
from gpri.timeseries import wrap


def _smooth_field(shape=(60, 80), amplitude=14.0, seed=0):
    """A smooth deformation field spanning several fringes."""
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    return amplitude * (np.sin(2.5 * np.pi * x / nx) * np.cos(1.5 * np.pi * y / ny)
                        + 0.4 * x / nx)


def _scene(shape=(60, 80), ps_fraction=0.25, noise=2.5, seed=0):
    """Smooth truth, reliable at the PS, garbage everywhere else."""
    rng = np.random.default_rng(seed)
    truth = _smooth_field(shape)
    ps = rng.random(shape) < ps_fraction
    phase = truth.copy()
    phase[~ps] += rng.normal(0, noise, size=(~ps).sum())
    return truth, ps, wrap(phase)


# ------------------------------------------------------------- PS selection
def test_amplitude_dispersion_is_low_for_a_stable_target():
    rng = np.random.default_rng(0)
    stable = 100.0 + rng.normal(0, 1.0, (40, 5, 5))
    noisy = np.abs(rng.normal(0, 50.0, (40, 5, 5))) + 1.0
    assert amplitude_dispersion(stable).mean() < 0.05
    assert amplitude_dispersion(noisy).mean() > 0.4


def test_zero_mean_amplitude_can_never_be_selected():
    a = np.zeros((10, 4, 4))
    d = amplitude_dispersion(a)
    assert np.all(np.isinf(d))
    assert not select_ps(dispersion=d, max_dispersion=1e9).any()


def test_select_ps_honours_dispersion_coherence_and_cap():
    rng = np.random.default_rng(1)
    d = rng.random((30, 30))
    c = rng.random((30, 30))
    m = select_ps(dispersion=d, max_dispersion=0.3)
    assert np.all(d[m] <= 0.3)

    m2 = select_ps(dispersion=d, coherence=c, max_dispersion=0.3, min_coherence=0.8)
    assert np.all(c[m2] >= 0.8) and m2.sum() <= m.sum()

    m3 = select_ps(dispersion=d, max_dispersion=1.0, max_count=25)
    assert m3.sum() == 25
    assert d[m3].max() <= d[~m3].min() + 1e-12    # it kept the best ones


def test_select_ps_needs_something_to_go_on():
    with pytest.raises(ValueError, match="need amplitudes"):
        select_ps()


def test_ps_density_is_a_fraction():
    assert ps_density(np.array([[True, False], [False, False]])) == 0.25


# -------------------------------------------------------- sparse unwrapping
def test_unwrap_sparse_recovers_a_smooth_field_up_to_a_constant():
    truth = _smooth_field((40, 50), amplitude=9.0)
    ps = np.zeros(truth.shape, bool)
    ps[::2, ::2] = True                        # dense enough for sub-pi steps
    out = unwrap_sparse(wrap(truth), ps)
    err = (out - truth[ps]) - (out[0] - truth[ps][0])
    assert np.allclose(err, 0.0, atol=1e-9)


def test_unwrap_sparse_leaves_an_unreachable_component_as_nan():
    """A PS island beyond max_edge has an undetermined 2 pi offset -- say NaN."""
    mask = np.zeros((20, 40), bool)
    mask[10, 0:3] = True
    mask[10, 35:38] = True                     # a long way off
    phase = np.zeros(mask.shape)
    out = unwrap_sparse(phase, mask, reference=0, max_edge=5.0)
    assert np.isfinite(out[:3]).all()
    assert np.isnan(out[3:]).all()


def test_unwrap_sparse_handles_a_single_ps():
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    out = unwrap_sparse(np.full((5, 5), 0.7), mask)
    assert out.shape == (1,) and np.isclose(out[0], 0.7)


def test_unwrap_sparse_accepts_physical_coordinates():
    """GPRI pixels are 0.75 m in range and tens of metres in azimuth."""
    truth = _smooth_field((30, 40), amplitude=6.0)
    ps = np.zeros(truth.shape, bool)
    ps[::2, ::2] = True
    rows, cols = np.nonzero(ps)
    coords = np.column_stack([rows * 30.0, cols * 0.75])
    out = unwrap_sparse(wrap(truth), ps, coords=coords)
    assert np.isfinite(out).all()


# ------------------------------------------------------------ interpolation
@pytest.mark.parametrize("method", ["linear", "idw", "nearest"])
def test_interpolation_reproduces_a_smooth_field(method):
    truth = _smooth_field((40, 50), amplitude=3.0)
    mask = np.zeros(truth.shape, bool)
    mask[::3, ::3] = True
    out = interpolate_ps(truth[mask], mask, method=method)
    assert out.shape == truth.shape
    assert np.isfinite(out).all()
    assert np.abs(out - truth).mean() < 0.5


def test_interpolation_is_exact_at_the_ps_themselves():
    truth = _smooth_field((30, 30), amplitude=4.0)
    mask = np.zeros(truth.shape, bool)
    mask[::4, ::4] = True
    out = interpolate_ps(truth[mask], mask, method="linear")
    assert np.allclose(out[mask], truth[mask], atol=1e-8)


def test_interpolation_rejects_a_length_mismatch():
    mask = np.zeros((5, 5), bool)
    mask[0, :3] = True
    with pytest.raises(ValueError, match="values for"):
        interpolate_ps(np.zeros(7), mask)


def test_nan_values_are_dropped_not_propagated():
    truth = _smooth_field((20, 20), amplitude=2.0)
    mask = np.zeros(truth.shape, bool)
    mask[::2, ::2] = True
    v = truth[mask].copy()
    v[0] = np.nan
    out = interpolate_ps(v, mask, method="linear")
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- the workflow
def test_interpolated_field_recovers_deformation_over_decorrelated_ground():
    """The actual claim: the smooth field is right *where the phase is garbage*.

    The method cannot denoise a decorrelated pixel -- nothing can, the
    information is not there.  What it delivers is a deformation estimate over
    those pixels, carried in from the PS around them.  So that is what gets
    asserted, at the pixels that decorrelated.
    """
    truth, ps, wrapped = _scene(noise=2.5, seed=3)
    res = unwrap_with_ps(wrapped, mask=ps, method="linear")

    ref = np.unravel_index(np.argmax(ps), ps.shape)
    got = res.interpolated - res.interpolated[ref]
    exp = truth - truth[ref]
    assert np.abs(got - exp)[~ps].mean() < 0.5       # well under a fringe


def test_unwrapped_phase_beats_naive_unwrapping_at_the_ps():
    """Where truth is knowable -- the PS -- it beats a plain 2-D unwrap."""
    truth, ps, wrapped = _scene(noise=2.5, seed=3)
    res = unwrap_with_ps(wrapped, mask=ps, method="linear")

    ref = np.unravel_index(np.argmax(ps), ps.shape)
    got = (res.unwrapped - res.unwrapped[ref])[ps]
    exp = (truth - truth[ref])[ps]

    naive = np.unwrap(np.unwrap(wrapped, axis=0), axis=1)
    naive = (naive - naive[ref])[ps]

    assert np.abs(got - exp).mean() < 0.1
    assert np.abs(got - exp).mean() < np.abs(naive - exp).mean()


def test_result_reports_where_the_assumption_broke():
    truth, ps, wrapped = _scene(noise=3.0, seed=4)
    res = unwrap_with_ps(wrapped, mask=ps)
    assert 0.0 <= res.suspect_fraction <= 1.0
    assert res.suspect.shape == wrapped.shape
    assert res.n_ps == int(ps.sum())
    assert "PS" in repr(res)


def test_residual_is_wrapped_and_the_sum_is_consistent():
    truth, ps, wrapped = _scene(seed=5)
    res = unwrap_with_ps(wrapped, mask=ps)
    assert np.all(np.abs(res.residual) <= np.pi + 1e-9)
    assert np.allclose(res.unwrapped, res.interpolated + res.residual)
    # the answer still wraps back to the observation -- nothing was invented
    assert np.allclose(wrap(res.unwrapped), wrap(wrapped), atol=1e-9)


def test_complex_interferogram_input_is_accepted():
    truth, ps, wrapped = _scene(seed=6)
    z = 3.0 * np.exp(1j * wrapped)
    res = unwrap_with_ps(z, mask=ps)
    assert np.allclose(wrap(res.unwrapped), wrap(wrapped), atol=1e-9)


def test_ps_are_selected_from_coherence_when_no_mask_is_given():
    truth, ps, wrapped = _scene(seed=7)
    cc = np.where(ps, 0.95, 0.1)
    res = unwrap_with_ps(wrapped, coherence=cc, min_coherence=0.5,
                         max_dispersion=np.inf)
    assert res.n_ps == int(ps.sum())


def test_empty_selection_refuses_rather_than_returning_nonsense():
    truth, ps, wrapped = _scene(seed=8)
    with pytest.raises(ValueError, match="no persistent scatterers"):
        unwrap_with_ps(wrapped, mask=np.zeros_like(ps))


def test_mask_shape_mismatch_is_caught():
    truth, ps, wrapped = _scene(seed=9)
    with pytest.raises(ValueError, match="does not match"):
        unwrap_with_ps(wrapped, mask=np.ones((3, 3), bool))
