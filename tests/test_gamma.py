"""GAMMA parameter-file parsing and binary raster IO."""
import numpy as np
import pytest

from gpri_tools.gamma import (DTYPES, ParFile, azimuth_angles, dtype_for, ground_range,
                        map_image, read_image, read_slc, write_image)

PAR = """Gamma Interferometric SAR Processor (ISP) - Image Parameter File
title: 2017-08-03 22:21:49.676416+00:00 CH2 upper
sensor: GPRI 2.0
date:  2017 08 03
range_samples:    12
azimuth_lines:    5
image_format:          FCOMPLEX
range_pixel_spacing:   0.750349  m
near_range_slc:        300.139581  m
radar_frequency:       1.720000e+10  Hz
prf:                   2.4999500D+01  Hz
GPRI_az_start_angle:    -27.955467  degrees
GPRI_az_angle_step:   2.000040e-01  degrees
GPRI_ant_elev_angle:     10.000000  degrees
GPRI_tx_coord:     0.22181    0.00000   -0.36936  m m m
"""


def test_parses_keys_units_and_shape():
    p = ParFile.loads(PAR)
    assert p.range_samples == 12 and p.azimuth_lines == 5
    assert p.shape == (5, 12)
    assert p.image_format == "FCOMPLEX"
    assert p.str("sensor") == "GPRI 2.0"
    # the unit word must not be parsed as part of the number
    assert p.float("range_pixel_spacing") == pytest.approx(0.750349)


def test_title_containing_colons_survives():
    p = ParFile.loads(PAR)
    assert p.str("title").startswith("2017-08-03 22:21:49.676416+00:00")
    assert "CH2 upper" in p.str("title")


def test_fortran_d_exponent():
    assert ParFile.loads(PAR).float("prf") == pytest.approx(24.9995)


def test_multi_value_entry():
    v = ParFile.loads(PAR).floats("GPRI_tx_coord")
    assert np.allclose(v, [0.22181, 0.0, -0.36936])


def test_date_and_wavelength():
    p = ParFile.loads(PAR)
    assert p.date == "20170803"
    # Ku band: c / 17.2 GHz
    assert p.wavelength == pytest.approx(0.0174298, abs=1e-6)


def test_slant_range_axis():
    p = ParFile.loads(PAR)
    r = p.slant_range()
    assert r.shape == (12,)
    assert r[0] == pytest.approx(300.139581)
    assert r[1] - r[0] == pytest.approx(0.750349)


def test_azimuth_angles_use_gpri_sweep():
    p = ParFile.loads(PAR)
    a = azimuth_angles(p)
    assert a.shape == (5,)
    assert a[0] == pytest.approx(-27.955467)
    assert a[1] - a[0] == pytest.approx(0.200004)


def test_azimuth_angles_fall_back_without_gpri_keys():
    p = ParFile.loads(PAR.replace("GPRI_az_angle_step", "unused_key"))
    assert np.allclose(azimuth_angles(p), np.arange(5))


def test_ground_range_foreshortens_by_elevation():
    p = ParFile.loads(PAR)
    assert np.allclose(ground_range(p), p.slant_range() * np.cos(np.deg2rad(10.0)))


def test_all_dtypes_are_big_endian_except_byte():
    for name, dt in DTYPES.items():
        assert dt.byteorder in (">", "|"), name


def test_roundtrip_complex_raster(tmp_path):
    a = (np.arange(60).reshape(5, 12) + 1j * np.arange(60).reshape(5, 12)).astype(np.complex64)
    f = tmp_path / "x.int"
    write_image(f, a)
    back = read_image(f, shape=(5, 12), image_format="FCOMPLEX")
    assert np.allclose(back, a)
    # written big-endian on disk, whatever the host order
    assert f.stat().st_size == 5 * 12 * 8


def test_read_image_finds_sidecar_par(tmp_path):
    (tmp_path / "x.slc.par").write_text(PAR)
    a = np.ones((5, 12), np.complex64)
    write_image(tmp_path / "x.slc", a)
    assert read_slc(tmp_path / "x.slc").shape == (5, 12)


def test_read_image_rejects_wrong_size(tmp_path):
    f = tmp_path / "short.int"
    write_image(f, np.ones((2, 3), np.complex64))
    with pytest.raises(ValueError, match="samples"):
        read_image(f, shape=(5, 12), image_format="FCOMPLEX")


def test_map_image_matches_read_image(tmp_path):
    a = np.random.default_rng(0).normal(size=(5, 12)).astype(np.float32)
    f = tmp_path / "y.mli"
    write_image(f, a, image_format="FLOAT")
    m = map_image(f, shape=(5, 12), image_format="FLOAT")
    assert np.allclose(np.asarray(m), read_image(f, shape=(5, 12), image_format="FLOAT"))


def test_map_image_rejects_size_mismatch(tmp_path):
    f = tmp_path / "z.mli"
    write_image(f, np.ones((2, 2), np.float32), image_format="FLOAT")
    with pytest.raises(ValueError, match="bytes"):
        map_image(f, shape=(5, 12), image_format="FLOAT")


def test_unknown_format_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        dtype_for("MYSTERY")
