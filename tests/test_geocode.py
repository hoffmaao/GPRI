"""Radar polar geometry to a local stereographic map frame."""
import numpy as np
import pytest

from gpri.gamma import ParFile
from gpri.geocode import (BAKERBEND1_HEADING, RadarGeometry, geocode,
                          heading_from_tiepoint, local_stereographic, map_grid)

# The real BakerBend1 upper-antenna geometry, trimmed to the keys geocoding uses.
PAR_TEXT = """Gamma Interferometric SAR Processor (ISP) - Image Parameter File
title: 2017-08-03 22:21:49 CH2 upper
sensor: GPRI 2.0
date:  2017 08 03
range_samples:    221
azimuth_lines:    396
image_format:          FCOMPLEX
range_pixel_spacing:   75.0349  m
azimuth_pixel_spacing: 0.000000  m
near_range_slc:        300.139581  m
radar_frequency:       1.720000e+10  Hz
GPRI_az_start_angle:    -27.955467  degrees
GPRI_az_angle_step:   2.000040e-01  degrees
GPRI_ant_elev_angle:     10.000000  degrees
GPRI_ref_north:      48.82132167
GPRI_ref_east:     -121.92018167
GPRI_ref_alt:         1252.20000  m
GPRI_scan_heading:      0.00000  degrees
"""

BAKER_SUMMIT = (48.7767, -121.8144)
FEATURES = {                      # north-side targets, from the findings survey
    "Baker summit":    (48.7767, -121.8144),
    "Coleman Glacier": (48.7900, -121.8400),
    "Mazama Glacier":  (48.7960, -121.7980),
    "Colfax Peak":     (48.7714, -121.8300),
}


@pytest.fixture
def par():
    return ParFile.loads(PAR_TEXT)


@pytest.fixture
def geom(par):
    return RadarGeometry(par, heading=BAKERBEND1_HEADING)


# --------------------------------------------------------------- the warning
def test_unsurveyed_scan_heading_warns_rather_than_silently_pointing_north(par):
    """GPRI_scan_heading is 0.0 in every BakerBend1 file. Never take that quietly."""
    with pytest.warns(UserWarning, match="GPRI_scan_heading is 0.0"):
        g = RadarGeometry(par)
    assert g.heading == 0.0


def test_an_explicit_heading_does_not_warn(par, recwarn):
    RadarGeometry(par, heading=105.0)
    assert not [w for w in recwarn if "scan_heading" in str(w.message)]


# ------------------------------------------------------------------ geometry
def test_reads_the_radar_position_from_the_parameter_file(geom):
    assert geom.lat0 == pytest.approx(48.82132167)
    assert geom.lon0 == pytest.approx(-121.92018167)
    assert geom.alt0 == pytest.approx(1252.2)
    assert geom.elevation == pytest.approx(10.0)


def test_ground_range_is_slant_range_foreshortened(geom):
    assert np.allclose(geom.ground_range(),
                       geom.slant_range() * np.cos(np.deg2rad(10.0)))
    assert np.all(geom.ground_range() < geom.slant_range())


def test_beam_rises_with_range_at_the_antenna_elevation(geom):
    h = geom.height()
    assert h[0] == pytest.approx(1252.2 + 300.139581 * np.sin(np.deg2rad(10.0)))
    assert np.all(np.diff(h) > 0)


def test_azimuth_resolution_spans_two_orders_of_magnitude(geom):
    """The reason radar geometry cannot be read as a map."""
    a = geom.azimuth_resolution()
    assert a[0] == pytest.approx(1.03, abs=0.1)       # metres at near range
    assert a[-1] / a[0] > 50


def test_bearings_are_heading_plus_azimuth_angle(geom):
    b = geom.bearings()
    assert b[0] == pytest.approx((105.0 - 27.955467) % 360.0)
    assert len(b) == 396
    assert np.all((b >= 0) & (b < 360))


# ------------------------------------------------------------ heading solve
def test_heading_from_tiepoint_inverts_the_forward_geometry(par):
    """Put a feature on a known row, solve, and get the heading back."""
    truth = 105.0
    g = RadarGeometry(par, heading=truth)
    row = 90
    lat, lon = g.geodetic(rows=[row], cols=[150])
    got = heading_from_tiepoint(par, lat[0, 0], lon[0, 0], row=row)
    assert got == pytest.approx(truth, abs=1e-6)


def test_heading_from_tiepoint_accepts_a_subpixel_row(par):
    g = RadarGeometry(par, heading=105.0)
    lat, lon = g.geodetic(rows=[90], cols=[100])
    a = heading_from_tiepoint(par, lat[0, 0], lon[0, 0], row=90.0)
    b = heading_from_tiepoint(par, lat[0, 0], lon[0, 0], row=90.5)
    step = par.float("GPRI_az_angle_step")            # 0.200004, not 0.2
    assert b == pytest.approx(a - 0.5 * step, abs=1e-9)


def test_the_bakerbend_default_heading_puts_the_north_side_in_the_fan(par):
    """A sanity check on BAKERBEND1_HEADING, not a survey."""
    g = RadarGeometry(par, heading=BAKERBEND1_HEADING)
    lo, hi = g.bearings().min(), g.bearings().max()
    for name, (lat, lon) in FEATURES.items():
        h = heading_from_tiepoint(par, lat, lon, row=0)
        bearing = (h + g.bearings()[0] - BAKERBEND1_HEADING) % 360.0
        assert lo <= bearing <= hi, f"{name} at {bearing:.1f} deg outside the fan"


# ------------------------------------------------------------- map transform
def test_local_stereographic_is_centred_on_its_origin():
    crs = local_stereographic(48.82132167, -121.92018167)
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(-121.92018167, 48.82132167)
    assert abs(x) < 1e-6 and abs(y) < 1e-6


def test_map_coordinates_agree_with_bearing_and_ground_range(geom):
    """Sanity: a target due east of the radar lands at +x, y ~ 0."""
    x, y = geom.map_coordinates()
    b = geom.bearings()
    east = int(np.argmin(np.abs(b - 90.0)))
    assert abs(y[east, -1]) < 0.02 * abs(x[east, -1])
    assert x[east, -1] > 0


def test_map_coordinate_distance_matches_ground_range(geom):
    x, y = geom.map_coordinates()
    x0, y0 = geom.origin_xy()
    d = np.hypot(x - x0, y - y0)
    gr = geom.ground_range()
    assert np.allclose(d, np.broadcast_to(gr, d.shape), rtol=2e-4)


def test_radar_coordinates_invert_map_coordinates(geom):
    x, y = geom.map_coordinates()
    row, col = geom.radar_coordinates(x, y)
    na, nr = geom.shape
    assert np.allclose(row, np.arange(na)[:, None], atol=1e-3)
    assert np.allclose(col, np.arange(nr)[None, :], atol=1e-3)


def test_bounds_enclose_the_fan_and_the_radar(geom):
    xmin, ymin, xmax, ymax = geom.bounds()
    x0, y0 = geom.origin_xy()
    assert xmin <= x0 <= xmax and ymin <= y0 <= ymax
    x, y = geom.map_coordinates()
    assert xmin <= x.min() and x.max() <= xmax


def test_map_grid_is_north_up(geom):
    x, y, transform = map_grid(geom, spacing=100.0)
    assert np.all(np.diff(y) < 0)             # y descends down the raster
    assert np.all(np.diff(x) > 0)
    assert transform[1] > 0 and transform[5] < 0


# --------------------------------------------------------------- resampling
def test_geocode_preserves_a_constant(geom):
    img = np.full(geom.shape, 7.0)
    out, _ = geocode(img, geom, spacing=200.0)
    inside = np.isfinite(out)
    assert inside.any()
    assert np.allclose(out[inside], 7.0)


def test_outside_the_fan_is_nan_not_zero(geom):
    out, _ = geocode(np.ones(geom.shape), geom, spacing=200.0)
    assert np.isnan(out).any(), "a rectangular grid must have corners off the fan"
    assert not (out == 0).any()


def test_geocode_recovers_a_ramp_in_range(geom):
    """A field that varies only with range must come back varying with distance."""
    na, nr = geom.shape
    img = np.broadcast_to(np.arange(nr, dtype=float), (na, nr)).copy()
    out, transform = geocode(img, geom, spacing=150.0, order="linear")

    x, y, _ = map_grid(geom, spacing=150.0)
    X, Y = np.meshgrid(x, y)
    x0, y0 = geom.origin_xy()
    d = np.hypot(X - x0, Y - y0)
    expected = (d / np.cos(np.deg2rad(geom.elevation))
                - geom.par.near_range) / geom.par.range_pixel_spacing
    m = np.isfinite(out)
    assert np.abs(out[m] - expected[m]).max() < 0.05


def test_nearest_and_linear_broadly_agree(geom):
    rng = np.random.default_rng(0)
    img = rng.normal(size=geom.shape)
    a, _ = geocode(img, geom, spacing=300.0, order="nearest")
    b, _ = geocode(img, geom, spacing=300.0, order="linear")
    m = np.isfinite(a) & np.isfinite(b)
    assert np.corrcoef(a[m], b[m])[0, 1] > 0.5


def test_complex_input_stays_complex(geom):
    z = np.exp(1j * np.zeros(geom.shape))
    out, _ = geocode(z, geom, spacing=300.0)
    assert np.iscomplexobj(out)
    m = np.isfinite(out.real)
    assert np.allclose(out[m], 1.0 + 0j)


def test_shape_mismatch_is_caught(geom):
    with pytest.raises(ValueError, match="geometry says"):
        geocode(np.zeros((3, 3)), geom)


def test_geocoded_grid_is_isotropic_unlike_the_radar_grid(geom):
    """The whole point: equal metres per pixel in both directions."""
    _, transform = geocode(np.ones(geom.shape), geom, spacing=100.0)
    assert abs(transform[1]) == pytest.approx(abs(transform[5]))
