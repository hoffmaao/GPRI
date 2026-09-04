"""RGI glacier masks: the reference selector that coherence alone cannot be."""
import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon

from gpri_tools.gamma import ParFile
from gpri_tools.geocode import RadarGeometry
from gpri_tools.glaciers import glacier_mask, outline_paths, stable_ground_mask

PAR_TEXT = """Gamma Interferometric SAR Processor (ISP) - Image Parameter File
sensor: GPRI 2.0
date:  2017 08 03
range_samples:    221
azimuth_lines:    396
image_format:          FCOMPLEX
range_pixel_spacing:   75.0349  m
near_range_slc:        300.139581  m
radar_frequency:       1.720000e+10  Hz
GPRI_az_start_angle:    -27.955467  degrees
GPRI_az_angle_step:   2.000040e-01  degrees
GPRI_ant_elev_angle:     10.000000  degrees
GPRI_ref_north:      48.82132167
GPRI_ref_east:     -121.92018167
GPRI_ref_alt:         1252.20000  m
"""


@pytest.fixture
def geom():
    return RadarGeometry(ParFile.loads(PAR_TEXT), heading=105.0)


def _square_at(geom, row, col, half_km=0.6):
    """An RGI-like polygon centred on a known radar pixel."""
    lat, lon = geom.geodetic(rows=[row], cols=[col])
    la, lo = float(lat[0, 0]), float(lon[0, 0])
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * np.cos(np.radians(la)))
    poly = Polygon([(lo - dlon, la - dlat), (lo + dlon, la - dlat),
                    (lo + dlon, la + dlat), (lo - dlon, la + dlat)])
    return gpd.GeoDataFrame({"Name": ["Test Glacier"]}, geometry=[poly],
                            crs="EPSG:4326")


def test_mask_lands_on_the_pixel_the_polygon_was_built_around(geom):
    gdf = _square_at(geom, row=200, col=120)
    m = glacier_mask(geom, gdf)
    assert m.shape == geom.shape
    assert m[200, 120]
    assert not m[10, 10]                     # far corner of the fan
    assert 0 < m.mean() < 0.5                # a patch, not the world


def test_buffer_grows_the_mask(geom):
    gdf = _square_at(geom, row=200, col=120)
    assert glacier_mask(geom, gdf, buffer_m=400.0).sum() > \
        glacier_mask(geom, gdf).sum()


def test_stable_ground_mask_splits_coherent_pixels_by_the_inventory(geom):
    gdf = _square_at(geom, row=200, col=120)
    cc = np.full(geom.shape, 0.9)            # everything is coherent
    stable, contested = stable_ground_mask(cc, geom, gdf, threshold=0.85,
                                           buffer_m=0.0)
    assert contested[200, 120]               # coherent AND glacier: dropped
    assert stable[10, 10]                    # coherent, off-ice: kept
    assert not (stable & contested).any()    # a pixel is one or the other
    assert (stable | contested).sum() == cc.size


def test_outline_paths_project_into_the_map_frame(geom):
    gdf = _square_at(geom, row=200, col=120)
    paths = outline_paths(gdf, geom)
    assert len(paths) == 1
    x, y = paths[0]
    gx, gy = geom.map_coordinates(rows=[200], cols=[120])
    assert x.min() < gx[0, 0] < x.max()
    assert y.min() < gy[0, 0] < y.max()
