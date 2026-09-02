"""Azimuth co-registration of SLCs whose tripod turned."""
import json

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from gpri.coregister import (acquisition_id, azimuth_offset, campaign_offsets,
                             scene_azimuth_offsets, shift_azimuth, shifts_for,
                             texture, write_azimuth_offsets)
from gpri.gamma import write_image
from gpri.stack import SlcPairStack

PAR = """Gamma Interferometric SAR Processor (ISP) - Image Parameter File
title: synthetic
sensor: GPRI 2.0
range_samples:    400
azimuth_lines:    120
image_format:          FCOMPLEX
range_pixel_spacing:   0.750349  m
azimuth_pixel_spacing: 0.000000  m
near_range_slc:        300.139581  m
radar_frequency:       1.72e+10  Hz
GPRI_az_start_angle:    -12.0  degrees
GPRI_az_angle_step:   0.2  degrees
GPRI_scan_heading:      0.00000  degrees
"""


def scene_slc(seed=0, shape=(140, 400)):
    """Speckle under a terrain pattern, smoothed over two lines like a beam."""
    rng = np.random.default_rng(seed)
    na, nr = shape
    l, r = np.mgrid[:na, :nr]
    # bright ridges and dark shadows that cross the lines at an angle
    terrain = 1.0 + 0.9 * np.sin(2 * np.pi * (l / 17.0 + r / 230.0)) \
        + 0.8 * np.sin(2 * np.pi * (l / 7.3 - r / 90.0))
    terrain = np.exp(terrain)
    s = np.sqrt(terrain) * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    s = uniform_filter1d(s.real, 2, axis=0) + 1j * uniform_filter1d(s.imag, 2, axis=0)
    return s.astype(np.complex64)


def window(big, lines, n=120, margin=10):
    """A 120-line scan of ``big`` whose ground lies ``lines`` later than the
    reference scan's -- cut from the interior, as a real scan is, with no
    zeroed edge."""
    return np.ascontiguousarray(shift_azimuth(big, -lines)[margin:margin + n])


def coherence(a, b, w=(5, 5)):
    from scipy.ndimage import uniform_filter
    pr = a * np.conj(b)
    num = uniform_filter(pr.real, w) + 1j * uniform_filter(pr.imag, w)
    den = np.sqrt(uniform_filter(np.abs(a) ** 2, w) * uniform_filter(np.abs(b) ** 2, w))
    return np.abs(num) / np.maximum(den, 1e-12)


def test_integer_shift_is_a_roll_with_the_wrap_zeroed():
    s = scene_slc()
    out = shift_azimuth(s, 3)
    assert np.allclose(out[:-3], s[3:], atol=1e-4)
    assert not out[-3:].any()
    back = shift_azimuth(s, -3)
    assert np.allclose(back[3:], s[:-3], atol=1e-4)
    assert not back[:3].any()
    assert shift_azimuth(s, 0) is s


def test_fractional_shift_inverts_and_keeps_the_phase():
    s = scene_slc()
    there = shift_azimuth(s, 0.4)
    back = shift_azimuth(there, -0.4)
    err = np.abs(back - s)[10:-10]
    assert np.sqrt((err ** 2).mean() / (np.abs(s[10:-10]) ** 2).mean()) < 0.03
    # a phase ramp is invisible to a shift: the interferogram survives it
    twin = s * np.exp(1j * 0.7)
    cc = coherence(shift_azimuth(s, 0.4)[2:-2], shift_azimuth(twin, 0.4)[2:-2])
    assert np.median(cc) > 0.999
    real = shift_azimuth(np.abs(s), 1.5)
    assert real.dtype == np.abs(s).dtype and not np.iscomplexobj(real)
    # a quarter line or less of wrap is tolerated, more is zeroed
    assert shift_azimuth(s, 0.2)[-1].any() and not shift_azimuth(s, 0.3)[-1].any()
    assert shift_azimuth(s, -1.2)[1].any() and not shift_azimuth(s, -1.3)[1].any()


def test_offset_is_recovered_and_coherence_restored():
    big = scene_slc()
    ref = window(big, 0.0)
    truth = 2.3
    # the same ground caught 2.3 lines later, with fresh receiver noise
    rng = np.random.default_rng(5)
    moved = window(big, truth) + 0.3 * (rng.normal(size=ref.shape)
                                        + 1j * rng.normal(size=ref.shape))
    moved = moved.astype(np.complex64)
    d, c = azimuth_offset(texture(moved, range_looks=4), texture(ref, range_looks=4),
                          search=10)
    assert d == pytest.approx(truth, abs=0.15)
    assert c > 0.5
    before = np.median(coherence(moved[6:-6], ref[6:-6]))
    after = np.median(coherence(shift_azimuth(moved, d)[6:-6], ref[6:-6]))
    assert before < 0.6 and after > 0.9


def test_no_shift_finds_zero():
    a = window(scene_slc(), 0.0)
    d, c = azimuth_offset(texture(a, 4), texture(a, 4), search=5)
    assert d == pytest.approx(0.0, abs=1e-9) and c == pytest.approx(1.0)


@pytest.fixture
def drifting_scene(tmp_path):
    """Three acquisitions; the ground of the last lies 4 and 1.5 lines later
    in the first two, as when the mount turns towards lower headings."""
    big = scene_slc(seed=2)
    (tmp_path / "slc").mkdir()
    ids = ["20180710_133506", "20180710_153506", "20180710_173506"]
    for sid, lines in zip(ids, (4.0, 1.5, 0.0)):
        for ant in "ul":
            write_image(tmp_path / "slc" / f"{sid}{ant}.slc", window(big, lines))
            (tmp_path / "slc" / f"{sid}{ant}.slc.par").write_text(PAR)
    return tmp_path, ids


def test_campaign_offsets_against_the_last(drifting_scene):
    root, ids = drifting_scene
    images = [root / "slc" / f"{i}u.slc" for i in ids]
    res = campaign_offsets(images, search=8, range_looks=4, ranges=(300.0, 600.0))
    assert res.reference == ids[-1]
    assert res.offsets[2] == 0.0 and res.corr[2] == 1.0
    assert res.offsets[0] == pytest.approx(4.0, abs=0.15)
    assert res.offsets[1] == pytest.approx(1.5, abs=0.15)
    assert res.span == pytest.approx(4.0, abs=0.2)
    assert res.span_deg == pytest.approx(0.8, abs=0.05)
    by_id = campaign_offsets(images, reference=ids[0], search=8, range_looks=4,
                             ranges=(300.0, 600.0))
    assert by_id.reference == ids[0]
    assert by_id.offsets[2] == pytest.approx(-4.0, abs=0.15)


def test_stack_reads_a_drifting_campaign_on_one_grid(drifting_scene):
    root, ids = drifting_scene
    raw = SlcPairStack.from_directory(root / "slc", antenna="u")
    lost = np.median(raw.read_coherence(0)[8:-8])
    res = campaign_offsets(raw.images, search=8, range_looks=4, ranges=(300.0, 600.0))
    aligned = SlcPairStack.from_directory(root / "slc", antenna="u",
                                          azimuth_shifts=res.offsets)
    kept = np.median(aligned.read_coherence(0)[8:-8])
    assert lost < 0.6 and kept > 0.95
    # the lower antenna takes the same table, keyed by acquisition
    lower = SlcPairStack.from_directory(root / "slc", antenna="l")
    lower.apply_azimuth_offsets(dict(zip(res.ids, res.offsets)))
    assert np.median(lower.read_coherence(1)[8:-8]) > 0.95
    assert lower.apply_azimuth_offsets(None).azimuth_shifts is None
    with pytest.raises(ValueError, match="one azimuth shift per image"):
        SlcPairStack.from_directory(root / "slc", antenna="u", azimuth_shifts=[1.0])


def test_sidecar_round_trip_and_missing_ids(tmp_path, monkeypatch, drifting_scene):
    monkeypatch.setenv("GPRI_WORK_ROOT", str(tmp_path / "work"))
    root, ids = drifting_scene
    images = [root / "slc" / f"{i}u.slc" for i in ids]
    res = campaign_offsets(images, search=8, range_looks=4, ranges=(300.0, 600.0))
    assert scene_azimuth_offsets("/archive/20180709") is None
    out = write_azimuth_offsets("/archive/20180709", res, extra={"antenna": "u"})
    assert out == tmp_path / "work" / "20180709" / "azimuth_offsets.json"
    rec = json.loads(out.read_text())
    assert rec["method"] == "texture" and rec["antenna"] == "u"
    assert rec["reference"] == ids[-1] and rec["step_deg"] == 0.2
    table = scene_azimuth_offsets("/elsewhere/20180709")
    assert table[ids[0]] == pytest.approx(res.offsets[0], abs=0.001)
    # the table is by acquisition: either antenna's file finds its row
    assert acquisition_id(root / "slc" / f"{ids[1]}l.slc") == ids[1]
    with pytest.warns(UserWarning, match="without a recorded azimuth offset"):
        sh = shifts_for(images + [root / "slc" / "20180710_193506u.slc"], table)
    assert sh[-1] == 0.0 and sh[0] == pytest.approx(res.offsets[0], abs=0.001)
