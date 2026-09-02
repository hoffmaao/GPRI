"""Radar polar geometry to a map frame: local stereographic, UTM, or lat/lon.

A GPRI image is not a picture of the ground.  Its axes are *slant range* and
*antenna azimuth angle* — a polar fan anchored at the tripod — so a velocity
map in radar geometry is stretched by ``1/cos(elevation)`` in one direction and
by ``r * d(theta)`` in the other.  At BakerBend1 a range bin is 0.75 m
throughout while an azimuth bin is 1.0 m at the near edge of the swath and
59 m at the far edge.  Nothing spatial can be read off it honestly until it is
projected.

What the parameter file gives you
---------------------------------
Everything except one number:

===========================  ==========================================
``GPRI_ref_north/east/alt``  radar position, WGS84 degrees and metres
``GPRI_az_start_angle``      first azimuth sample, degrees
``GPRI_az_angle_step``       azimuth increment, degrees
``GPRI_ant_elev_angle``      antenna elevation above horizontal, degrees
``near_range_slc``           slant range of the first sample, metres
``range_pixel_spacing``      slant range increment, metres
``GPRI_scan_heading``        true bearing of azimuth zero  <-- the problem
===========================  ==========================================

**``GPRI_scan_heading`` is 0.0 in every BakerBend1 parameter file** — it was
never surveyed, and a scan heading of exactly zero would point the fan due
north, at nothing.  So the absolute orientation is not in the data and has to
be supplied.  Three ways to get it:

* :func:`gpri.heading.heading_from_dem` (``gpri heading``) — the terrain's
  shadows and facing slopes, simulated from a DEM and correlated with the
  mean backscatter, fix it to 0.03 deg with no field survey and no tie point.
  This is what the examples use, through :func:`gpri.heading.scene_heading`.
* :func:`heading_from_tiepoint` — give it one identifiable feature (a summit,
  a nunatak, a rock rib) with known coordinates and the pixel it lands on, and
  it solves for the heading.
* Pass ``heading=`` yourself if it was recorded in the field notes.

For the BakerBend1 north-side geometry, the radar at 48.82132 N, 121.92018 W,
1252 m sees Baker's summit at bearing 122.5 deg and 9.2 km, Coleman Glacier at
120.6 deg, Mazama at 107.4 deg and Colfax Peak at 129.9 deg.  The 79-degree
fan (-27.96 to +51.05 deg) therefore needs a heading near **105 deg** to cover
them; :data:`BAKERBEND1_HEADING` holds that as a starting guess.  The DEM
says each campaign pointed its own way — 111.4 deg on 2017-07-13, 107.4 on
08-03, 100.1 on 08-27, 108.4 on 09-15 and 122.8 in 2018 — so the guess is
only a fallback for a scene that has not been measured, and
:class:`RadarGeometry` built on it warns.

Sign and datum conventions
--------------------------
Bearings are degrees clockwise from true north.  Ground range is the
horizontal projection of slant range, ``r * cos(elevation)``.  Heights are
ellipsoidal unless you subtract ``GPRI_geoid`` yourself.  The default map frame
is a **local stereographic projection centred on the radar** — conformal,
scale-true at the origin, and better than a degree of latitude short of any
distortion worth worrying about over a 17 km swath.
"""
from __future__ import annotations

import warnings

import numpy as np

from .gamma import ParFile, azimuth_angles

try:
    from pyproj import CRS, Geod, Transformer
    _GEOD = Geod(ellps="WGS84")
except ImportError:  # pragma: no cover - pyproj is a hard dependency in practice
    CRS = Geod = Transformer = None
    _GEOD = None

__all__ = [
    "RadarGeometry", "local_stereographic", "geocode", "heading_from_tiepoint",
    "map_grid", "write_geotiff", "BAKERBEND1_HEADING",
]

#: Starting-guess scan heading for the BakerBend1 north-side fan, degrees true.
#: Derived from the bearings to the north-side glaciers, not from a survey;
#: the measured headings (``gpri heading``) range from 100.1 to 122.8 deg.
BAKERBEND1_HEADING = 105.0


def local_stereographic(lat0, lon0, datum="WGS84"):
    """A stereographic projection centred on ``(lat0, lon0)``, metres.

    Conformal and scale-true at the centre, which for a 17 km swath means the
    scale error at the far edge is under 2 parts per million — far below
    anything the radar geometry itself contributes.  Use this when you want a
    clean local Cartesian frame rather than a national grid.
    """
    if CRS is None:  # pragma: no cover
        raise ImportError("local_stereographic needs pyproj")
    return CRS.from_proj4(
        f"+proj=stere +lat_0={float(lat0)} +lon_0={float(lon0)} "
        f"+k_0=1 +x_0=0 +y_0=0 +datum={datum} +units=m +no_defs")


class RadarGeometry:
    """The mapping between a GPRI polar grid and the ground.

    >>> geom = RadarGeometry.from_par("20170803_222136u.slc.par", heading=105.0)
    >>> geom
    RadarGeometry(48.82132 N, -121.92018 E, 1252.2 m, heading 105.0 deg,
                  elev 10.0 deg, 396 az x 22101 rg)
    >>> east, north = geom.map_coordinates()      # local stereographic, metres
    """

    def __init__(self, par, heading=None, elevation=None, lat0=None, lon0=None,
                 alt0=None, crs=None):
        self.par = par if isinstance(par, ParFile) else ParFile.load(par)

        self.lat0 = self.par.float("GPRI_ref_north", 0.0) if lat0 is None else float(lat0)
        self.lon0 = self.par.float("GPRI_ref_east", 0.0) if lon0 is None else float(lon0)
        self.alt0 = self.par.float("GPRI_ref_alt", 0.0) if alt0 is None else float(alt0)

        if heading is None:
            heading = self.par.float("GPRI_scan_heading", 0.0)
            if heading == 0.0:
                warnings.warn(
                    "GPRI_scan_heading is 0.0 in this parameter file, which "
                    "points the fan due north and is almost certainly a "
                    "placeholder rather than a survey. Pass heading=, or solve "
                    "for it with heading_from_tiepoint(); any map made from "
                    "this is rotated by an unknown amount.", stacklevel=2)
        self.heading = float(heading)

        self.elevation = (self.par.float("GPRI_ant_elev_angle", 0.0)
                          if elevation is None else float(elevation))
        self._crs = crs

    @classmethod
    def from_par(cls, par, **kwargs):
        return cls(par, **kwargs)

    # ------------------------------------------------------------- properties
    @property
    def crs(self):
        """Target map CRS; a local stereographic on the radar unless set."""
        if self._crs is None:
            self._crs = local_stereographic(self.lat0, self.lon0)
        return self._crs

    @crs.setter
    def crs(self, value):
        self._crs = None if value is None else CRS.from_user_input(value)

    @property
    def shape(self):
        return self.par.shape

    def slant_range(self):
        return self.par.slant_range()

    def ground_range(self):
        """Horizontal distance from the radar, metres."""
        return self.slant_range() * np.cos(np.deg2rad(self.elevation))

    def height(self):
        """Ellipsoidal height of the beam centre at each range sample, metres."""
        return self.alt0 + self.slant_range() * np.sin(np.deg2rad(self.elevation))

    def bearings(self):
        """True bearing of every azimuth line, degrees clockwise from north."""
        return (self.heading + azimuth_angles(self.par)) % 360.0

    def azimuth_resolution(self):
        """Cross-range sample spacing at each range, metres.

        The number that makes the case for geocoding: 1.0 m at near range,
        59 m at far range for BakerBend1.
        """
        step = np.deg2rad(self.par.float("GPRI_az_angle_step", 0.0))
        return self.ground_range() * step

    # ------------------------------------------------------- forward mapping
    def geodetic(self, rows=None, cols=None):
        """``(lat, lon)`` of each radar pixel, degrees.

        Uses a proper geodesic forward solution on the WGS84 ellipsoid rather
        than a flat-earth offset, so it stays right at the far edge of the
        swath and at high latitude.
        """
        if _GEOD is None:  # pragma: no cover
            raise ImportError("geodetic() needs pyproj")
        az = self.bearings() if rows is None else self.bearings()[rows]
        gr = self.ground_range() if cols is None else self.ground_range()[cols]
        AZ, GR = np.meshgrid(np.asarray(az, float), np.asarray(gr, float),
                             indexing="ij")
        lon, lat, _ = _GEOD.fwd(
            np.full(AZ.size, self.lon0), np.full(AZ.size, self.lat0),
            AZ.ravel(), GR.ravel())
        return lat.reshape(AZ.shape), lon.reshape(AZ.shape)

    def map_coordinates(self, rows=None, cols=None, crs=None):
        """``(x, y)`` of each radar pixel in the target CRS, metres.

        For the default local stereographic these are metres east and north of
        the radar.
        """
        lat, lon = self.geodetic(rows, cols)
        tf = Transformer.from_crs("EPSG:4326", crs or self.crs, always_xy=True)
        x, y = tf.transform(lon.ravel(), lat.ravel())
        return x.reshape(lat.shape), y.reshape(lat.shape)

    def origin_xy(self, crs=None):
        """The radar position in the target CRS."""
        tf = Transformer.from_crs("EPSG:4326", crs or self.crs, always_xy=True)
        return tf.transform(self.lon0, self.lat0)

    # ------------------------------------------------------- inverse mapping
    def radar_coordinates(self, x, y, crs=None):
        """Map ``(x, y)`` -> fractional ``(row, col)`` in the radar grid.

        The inverse mapping is what :func:`geocode` actually uses: rather than
        scatter 8.7 million radar pixels into a grid and interpolate, it asks
        each output pixel which radar sample it came from.  Exact, and linear
        in the size of the output rather than the input.

        Returns ``(row, col)`` as floats, which may fall outside the grid;
        :func:`geocode` masks those.
        """
        tf = Transformer.from_crs(crs or self.crs, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(np.asarray(x, float), np.asarray(y, float))
        az, _, dist = _GEOD.inv(np.full(np.shape(lon), self.lon0),
                                np.full(np.shape(lat), self.lat0), lon, lat)

        slant = dist / max(np.cos(np.deg2rad(self.elevation)), 1e-12)
        col = (slant - self.par.near_range) / self.par.range_pixel_spacing

        rel = (np.asarray(az, float) - self.heading + 180.0) % 360.0 - 180.0
        start = self.par.float("GPRI_az_start_angle", 0.0)
        step = self.par.float("GPRI_az_angle_step", 0.0)
        row = (rel - start) / step if step else np.zeros_like(rel)
        return row, col

    # ------------------------------------------------------------- extent
    def bounds(self, crs=None, pad=0.0):
        """``(xmin, ymin, xmax, ymax)`` of the illuminated fan in the map frame."""
        na, nr = self.shape
        rows = np.arange(na)
        cols = np.array([0, nr - 1])
        x, y = self.map_coordinates(rows, cols, crs=crs)
        x0, y0 = self.origin_xy(crs=crs)
        xs = np.concatenate([x.ravel(), [x0]])
        ys = np.concatenate([y.ravel(), [y0]])
        return (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)

    def __repr__(self):
        na, nr = self.shape
        return (f"RadarGeometry({self.lat0:.5f} N, {self.lon0:.5f} E, "
                f"{self.alt0:.1f} m, heading {self.heading:.1f} deg, "
                f"elev {self.elevation:.1f} deg, {na} az x {nr} rg)")


# ------------------------------------------------------------- heading solve
def heading_from_tiepoint(par, lat, lon, row, elevation=None, lat0=None,
                          lon0=None):
    """Solve ``GPRI_scan_heading`` from one identified feature.

    Give it a feature whose coordinates you know and the azimuth **row** it
    falls on in the radar image; the range coordinate is not needed, since
    heading is a pure rotation about the radar.

    >>> heading_from_tiepoint(par, 48.7767, -121.8144, row=88)   # Baker summit
    105.4...

    One tie-point fixes the heading exactly.  Two or more let you check it —
    take the spread across several features as your uncertainty, and expect a
    fraction of a degree if the tripod was stable.  0.5 degrees of heading
    error is 80 m of position error at 9 km, which matters for overlaying a
    DEM but not for the phase.
    """
    if _GEOD is None:  # pragma: no cover
        raise ImportError("heading_from_tiepoint needs pyproj")
    par = par if isinstance(par, ParFile) else ParFile.load(par)
    lat0 = par.float("GPRI_ref_north", 0.0) if lat0 is None else float(lat0)
    lon0 = par.float("GPRI_ref_east", 0.0) if lon0 is None else float(lon0)

    true_bearing, _, _ = _GEOD.inv(lon0, lat0, float(lon), float(lat))
    az = azimuth_angles(par)
    r = np.asarray(row, float)
    # linear interpolation so a sub-pixel row is honoured
    local = np.interp(r, np.arange(len(az)), az)
    return float((true_bearing - local) % 360.0)


# ------------------------------------------------------------------ gridding
def map_grid(geom, spacing=10.0, bounds=None, crs=None):
    """Regular map grid covering the radar fan.

    Returns ``(x, y, transform)`` where ``x`` and ``y`` are 1-D coordinate axes
    and ``transform`` is the affine tuple GDAL/rasterio wants:
    ``(xmin, spacing, 0, ymax, 0, -spacing)``.  ``y`` descends, north-up, which
    is the convention every GIS expects.
    """
    if bounds is None:
        bounds = geom.bounds(crs=crs)
    xmin, ymin, xmax, ymax = bounds
    sx = sy = float(spacing) if np.isscalar(spacing) else None
    if sx is None:
        sx, sy = float(spacing[0]), float(spacing[1])
    nx = max(1, int(np.ceil((xmax - xmin) / sx)))
    ny = max(1, int(np.ceil((ymax - ymin) / sy)))
    x = xmin + sx * (np.arange(nx) + 0.5)
    y = ymax - sy * (np.arange(ny) + 0.5)
    return x, y, (xmin, sx, 0.0, ymax, 0.0, -sy)


def geocode(image, geom, spacing=10.0, bounds=None, crs=None, order="linear",
            fill=np.nan):
    """Resample a radar-geometry image onto a north-up map grid.

    Parameters
    ----------
    image : 2-D array
        Radar geometry, ``(azimuth_lines, range_samples)``.  Real or complex —
        complex is interpolated as real and imaginary parts, which is correct
        for an interferogram provided the fringe rate is under the output
        sampling.  For phase, geocode ``exp(1j * phi)`` rather than ``phi``.
    geom : :class:`RadarGeometry`
    spacing : float or (float, float)
        Output pixel size in map units (metres).  10 m is a sensible default
        for BakerBend1: finer than the 59 m azimuth sampling at far range, and
        coarser than the 1 m at near range where oversampling buys nothing.
    order : {'linear', 'nearest'}
    fill : scalar
        Value outside the illuminated fan.  NaN, so the empty area is visibly
        empty rather than silently zero.

    Returns
    -------
    out : 2-D array
        ``(ny, nx)``, north-up.
    transform : tuple
        Affine geotransform, ready for :func:`write_geotiff`.
    """
    img = np.asarray(image)
    na, nr = geom.shape
    if img.shape != (na, nr):
        raise ValueError(f"image is {img.shape}, geometry says {(na, nr)}")

    x, y, transform = map_grid(geom, spacing=spacing, bounds=bounds, crs=crs)
    X, Y = np.meshgrid(x, y)
    row, col = geom.radar_coordinates(X, Y, crs=crs)

    inside = (row >= 0) & (row <= na - 1) & (col >= 0) & (col <= nr - 1)
    inside &= np.isfinite(row) & np.isfinite(col)

    out = np.full(X.shape, fill, dtype=complex if np.iscomplexobj(img) else float)
    if not inside.any():
        return out, transform

    r, c = row[inside], col[inside]
    if order == "nearest":
        out[inside] = img[np.rint(r).astype(int), np.rint(c).astype(int)]
        return out, transform

    r0 = np.clip(np.floor(r).astype(int), 0, na - 1)
    c0 = np.clip(np.floor(c).astype(int), 0, nr - 1)
    r1 = np.minimum(r0 + 1, na - 1)
    c1 = np.minimum(c0 + 1, nr - 1)
    fr = r - r0
    fc = c - c0
    v = (img[r0, c0] * (1 - fr) * (1 - fc) + img[r1, c0] * fr * (1 - fc)
         + img[r0, c1] * (1 - fr) * fc + img[r1, c1] * fr * fc)
    out[inside] = v
    return out, transform


def write_geotiff(path, array, transform, crs, nodata=np.nan):
    """Write a geocoded array as a GeoTIFF.

    Complex input is written as two bands, real then imaginary — GeoTIFF has a
    complex type but almost nothing downstream reads it.
    """
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:  # pragma: no cover
        raise ImportError("write_geotiff needs rasterio") from exc

    a = np.asarray(array)
    bands = [a.real.astype("float32"), a.imag.astype("float32")] \
        if np.iscomplexobj(a) else [a.astype("float32")]
    xmin, sx, _, ymax, _, sy = transform
    aff = Affine(sx, 0.0, xmin, 0.0, sy, ymax)
    crs_obj = crs if isinstance(crs, str) else crs.to_wkt()
    with rasterio.open(path, "w", driver="GTiff", height=bands[0].shape[0],
                       width=bands[0].shape[1], count=len(bands),
                       dtype="float32", crs=crs_obj, transform=aff,
                       nodata=nodata, compress="deflate") as dst:
        for i, b in enumerate(bands, start=1):
            dst.write(b, i)
    return path
