"""The scan heading from a DEM's shadows and facing slopes."""
import json

import numpy as np
import pytest

from gpri.gamma import ParFile
from gpri.heading import (heading_from_dem, polar_terrain, scene_heading,
                          simulate_intensity, write_scene_heading)

LAT0, LON0 = 48.8213, -121.9202

PAR_TEXT = """Gamma Interferometric SAR Processor (ISP) - Image Parameter File
title: synthetic
sensor: GPRI 2.0
range_samples:    8000
azimuth_lines:    300
image_format:          SCOMPLEX
range_pixel_spacing:   0.750349  m
azimuth_pixel_spacing: 0.000000  m
near_range_slc:        300.139581  m
GPRI_az_start_angle:    -20.0  degrees
GPRI_az_angle_step:   0.2  degrees
GPRI_ref_north:      48.8213
GPRI_ref_east:     -121.9202
GPRI_ref_alt:         1250.0  m
GPRI_scan_heading:      0.00000  degrees
"""


def hills(lat, lon):
    """A bench with two hills to the east: shadows behind, bright faces in front."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = (lon - LON0) * 73000.0            # metres east
    y = (lat - LAT0) * 111000.0           # metres north
    z = 1250.0 - 0.02 * np.hypot(x, y)    # falls away from the radar
    z += 900 * np.exp(-((x - 4000) ** 2 + (y + 1500) ** 2) / 1200.0 ** 2)
    z += 600 * np.exp(-((x - 6500) ** 2 + (y - 500) ** 2) / 900.0 ** 2)
    return z


@pytest.fixture(scope="module")
def terrain():
    return polar_terrain(hills, LAT0, LON0, rmax=8000.0, daz=0.2, dr=15.0)


def synthetic_image(terrain, par, heading, seed=0):
    """What the radar would record at ``heading``, speckled."""
    r0, dr = par.float("near_range_slc"), par.float("range_pixel_spacing")
    ml = 20
    nr = par.range_samples // ml
    bearings, img = simulate_intensity(terrain, r0, dr * ml, nr)
    start, step = par.float("GPRI_az_start_angle"), par.float("GPRI_az_angle_step")
    theta = start + step * np.arange(par.azimuth_lines)
    b = np.rint(((theta + heading) % 360.0) / 0.2).astype(int) % bearings.size
    fine = np.repeat(img[b], ml, axis=1)[:, :par.range_samples]
    rng = np.random.default_rng(seed)
    return (fine + 1e-3 * fine.max()) * rng.exponential(1.0, fine.shape)


def test_shadow_falls_behind_the_hill(terrain):
    """Along the bearing of the big hill, the far slope is dark."""
    b = int(np.argmin(np.abs(terrain.bearings - np.degrees(np.arctan2(4000, -1500)) % 360)))
    lit = terrain.lit[b]
    g = terrain.ground_range
    assert lit[(g > 2000) & (g < 3800)].all()          # the face
    assert not lit[(g > 4600) & (g < 5500)].any()      # behind the crest


def test_recovers_the_heading(terrain):
    par = ParFile.loads(PAR_TEXT)
    truth = 83.0
    meas = synthetic_image(terrain, par, truth)
    fit = heading_from_dem(par, meas, hills, rmax=8000.0, terrain=terrain,
                           headings=np.arange(0.0, 360.0, 0.2))
    assert fit.heading == pytest.approx(truth, abs=0.3)
    assert fit.corr > 0.5
    # nothing else comes close: the sidelobes are an order of magnitude down
    far = np.abs(fit.headings - truth) > 3
    assert fit.curve[far].max() < 0.3 * fit.corr


def test_a_wrong_position_still_finds_the_peak_but_lower(terrain):
    par = ParFile.loads(PAR_TEXT)
    meas = synthetic_image(terrain, par, 83.0)
    right = heading_from_dem(par, meas, hills, rmax=8000.0, terrain=terrain,
                             headings=np.arange(60.0, 110.0, 0.2))
    off = heading_from_dem(par, meas, hills, rmax=8000.0,
                           lat0=LAT0 + 0.004, headings=np.arange(60.0, 110.0, 0.2))
    assert off.corr < right.corr


def test_sidecar_round_trip(tmp_path, monkeypatch, terrain):
    monkeypatch.setenv("GPRI_WORK_ROOT", str(tmp_path))
    par = ParFile.loads(PAR_TEXT)
    fit = heading_from_dem(par, synthetic_image(terrain, par, 83.0), hills,
                           rmax=8000.0, terrain=terrain,
                           headings=np.arange(70.0, 100.0, 0.2))
    out = write_scene_heading("/somewhere/20170913", fit, extra={"dem": "x.tif"})
    assert out == tmp_path / "20170913" / "heading.json"
    rec = json.loads(out.read_text())
    assert rec["method"] == "dem" and rec["dem"] == "x.tif"
    assert scene_heading("/elsewhere/20170913") == pytest.approx(fit.heading, abs=0.01)
    with pytest.warns(UserWarning, match="guess"):
        assert scene_heading("/elsewhere/20990101", default=105.0) == 105.0
    with pytest.raises(FileNotFoundError):
        scene_heading("/elsewhere/20990101")
