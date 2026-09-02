"""Scan heading from the terrain: which way was the tripod pointing?

``GPRI_scan_heading`` is 0.0 in every BakerBend1 parameter file — the
instrument never knew, and a tripod set up by hand on a different day points
a different way.  :data:`gpri.geocode.BAKERBEND1_HEADING` was a bearing-
table guess (105 deg) shared by every campaign; the terrain says otherwise,
and says it differently for each campaign.

The measurement here needs no field survey and no tie point.  A DEM, the
radar's position and a mean-intensity image are enough: the terrain a
ground-based radar sees is dominated by *shadow* — everything behind the
first ridge along each bearing is black — and by *facing* — slopes that lean
toward the radar are bright.  Both are geometry, both are in the DEM, and
neither cares about the heading except as a rotation.  So:

1. resample the DEM onto a polar grid around the radar
   (:func:`polar_terrain`), run a running-max elevation-angle test out along
   each bearing for shadow, and take the cosine of the local incidence angle
   for brightness;
2. bin that into a (true bearing, slant range) image
   (:func:`simulate_intensity`);
3. for each trial heading, pick the bearings the antenna angles land on and
   correlate the high-passed simulated image with the high-passed measured
   one (:func:`heading_from_dem`).

The correlation is ~0 everywhere and 0.5–0.7 within a degree of the answer.
It is sharp because the shadow edges are sharp: a degree of heading moves
a ridge a full cross-range cell at 5 km.

Half a degree of heading is 80 m at 9 km, which the RGI ice/rock masks
notice; five degrees is 400 m at 5 km, which puts glacier margins on the
wrong side of the reference.  Run this before anything that uses the map.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .gamma import ParFile, azimuth_angles, read_slc

__all__ = [
    "PolarTerrain", "polar_terrain", "simulate_intensity", "mean_intensity",
    "HeadingFit", "heading_from_dem", "scene_heading", "write_scene_heading",
    "HEADING_FILE",
]

#: Sidecar written beside a scene's caches by ``gpri heading --write``.
HEADING_FILE = "heading.json"


# ------------------------------------------------------------------ terrain
@dataclass
class PolarTerrain:
    """A DEM resampled onto bearings x ground ranges around the radar."""
    bearings: np.ndarray        # degrees true, (nb,)
    ground_range: np.ndarray    # metres, (nr,)
    height: np.ndarray          # metres, (nb, nr)
    lit: np.ndarray             # bool, (nb, nr): not in radar shadow
    cos_incidence: np.ndarray   # (nb, nr): 0 where the slope faces away
    alt0: float                 # radar height used, metres

    @property
    def slant_range(self):
        return np.hypot(self.ground_range[None, :], self.height - self.alt0)


def _dem_sampler(dem, lat0, lon0, half_width_deg=0.25):
    """``(sample(lat, lon) -> height, height_at_radar)`` from a raster DEM.

    Reads only a window around the radar; the window is clipped to the tile
    so its transform stays honest when the radar sits near a tile edge.
    """
    import rasterio
    from rasterio.windows import from_bounds

    d = rasterio.open(dem)
    bb = d.bounds
    win = from_bounds(max(lon0 - half_width_deg, bb.left),
                      max(lat0 - half_width_deg * 0.7, bb.bottom),
                      min(lon0 + half_width_deg, bb.right),
                      min(lat0 + half_width_deg * 0.7, bb.top), d.transform)
    win = win.round_offsets().round_lengths()
    z = d.read(1, window=win).astype(float)
    nodata = d.nodata
    if nodata is not None:
        z[z == nodata] = np.nan
    tr = d.window_transform(win)

    def sample(lat, lon):
        col = (np.atleast_1d(np.asarray(lon, float)) - tr.c) / tr.a - 0.5
        row = (np.atleast_1d(np.asarray(lat, float)) - tr.f) / tr.e - 0.5
        return map_coordinates(z, [row, col], order=1, mode="nearest")

    return sample, float(sample(lat0, lon0)[0])


def polar_terrain(dem, lat0, lon0, alt0=None, rmax=12000.0, daz=0.2, dr=15.0,
                  antenna_height=2.0, shadow_tolerance=0.02):
    """Resample a DEM around the radar and find what the radar can see.

    Parameters
    ----------
    dem : path or callable
        A raster the ``rasterio`` package can open (e.g. a Copernicus 30 m
        tile), or a function ``(lat, lon) -> height`` for testing.
    lat0, lon0 : float
        Radar position, degrees.
    alt0 : float, optional
        Radar phase-centre height, metres.  Default: the DEM height at the
        radar plus ``antenna_height`` — safer than a hand-held GPS altitude
        on a ridge, and the two agree to a metre at BakerBend1.
    rmax, daz, dr : float
        Extent and spacing of the polar grid: metres, degrees, metres.
    shadow_tolerance : float
        Degrees below the running-max elevation angle a cell may sit and
        still count as lit; absorbs DEM noise on a flat facing slope.
    """
    from pyproj import Transformer

    from .geocode import local_stereographic

    if callable(dem):
        sample, z0 = dem, float(np.ravel(dem(lat0, lon0))[0])
    else:
        sample, z0 = _dem_sampler(dem, lat0, lon0)
    if alt0 is None:
        alt0 = z0 + antenna_height

    bearings = np.arange(0.0, 360.0, daz)
    ground = np.arange(dr, rmax, dr)
    B, G = np.meshgrid(np.deg2rad(bearings), ground, indexing="ij")
    x, y = G * np.sin(B), G * np.cos(B)
    to_ll = Transformer.from_crs(local_stereographic(lat0, lon0), "EPSG:4326",
                                 always_xy=True)
    lon, lat = to_ll.transform(x, y)
    Z = np.asarray(sample(lat, lon), float)

    # shadow: out along each bearing, a cell is visible only if it rises
    # above every cell before it as seen from the radar
    elev = np.degrees(np.arctan2(Z - alt0, G))
    lit = elev >= np.maximum.accumulate(elev, axis=1) - shadow_tolerance

    # facing: cosine of the local incidence angle from the polar gradients
    dz_r = np.gradient(Z, dr, axis=1)
    dz_a = np.gradient(Z, axis=0) / (G * np.deg2rad(daz))
    nrm = np.stack([-dz_r, -dz_a, np.ones_like(Z)])
    nrm /= np.linalg.norm(nrm, axis=0)
    los = np.stack([-G, np.zeros_like(Z), alt0 - Z])       # cell -> radar
    los /= np.linalg.norm(los, axis=0)
    cos_inc = np.clip((nrm * los).sum(0), 0.0, None)
    cos_inc[~np.isfinite(cos_inc)] = 0.0

    return PolarTerrain(bearings, ground, Z, lit, cos_inc, float(alt0))


def simulate_intensity(terrain, r0, dr, nr, daz=None):
    """What the DEM says the radar sees: (true bearing, slant range) image.

    Returns ``(bearings, image)``; ``image[b, k]`` is the lit, radar-facing
    terrain area that lands at slant range ``r0 + k*dr`` from bearing
    ``bearings[b]``, so layover (two heights, one range) simply adds.
    """
    t = terrain
    daz_t = float(t.bearings[1] - t.bearings[0])
    daz = daz_t if daz is None else float(daz)
    nb = int(round(360.0 / daz))
    dr_t = float(t.ground_range[1] - t.ground_range[0])
    area = t.ground_range[None, :] * np.deg2rad(daz_t) * dr_t
    w = t.cos_incidence * t.lit * area
    bi = np.rint(t.bearings / daz).astype(int)[:, None] % nb
    ri = ((t.slant_range - r0) / dr).astype(int)
    ok = (ri >= 0) & (ri < nr) & (w > 0)
    img = np.zeros((nb, nr))
    np.add.at(img, (np.broadcast_to(bi, ri.shape)[ok], ri[ok]), w[ok])
    return np.arange(nb) * daz, img


# ---------------------------------------------------------------- measured
def mean_intensity(slcs, n=8, which="first", offsets=None):
    """Average |SLC|^2 over ``n`` images and the par of the first one.

    ``offsets`` is a campaign's ``{acquisition id: lines}`` table
    (:func:`gpri.coregister.scene_azimuth_offsets`): each image is shifted
    onto the reference grid before it is added, so a campaign whose tripod
    turned still averages to one sharp picture — on the reference's grid,
    which is the grid the heading then describes.
    """
    slcs = sorted(str(s) for s in slcs)
    slcs = slcs[:n] if which == "first" else slcs[-n:]
    shifts = np.zeros(len(slcs))
    if offsets:
        from .coregister import shifts_for
        shifts = shifts_for(slcs, offsets)
    acc = None
    for f, d in zip(slcs, shifts):
        a = np.abs(read_slc(f)).astype(np.float32) ** 2
        if d:
            from .coregister import shift_azimuth
            a = shift_azimuth(a, d)
        acc = a if acc is None else acc + a
    return acc / len(slcs), ParFile.load(slcs[0] + ".par"), slcs


def _coarsen(a, ml_a, ml_r):
    nl = a.shape[0] // ml_a * ml_a
    nr = a.shape[1] // ml_r * ml_r
    return a[:nl, :nr].reshape(nl // ml_a, ml_a, nr // ml_r, ml_r).mean((1, 3))


def _db(x, floor=1e-3):
    return 10 * np.log10(np.maximum(x, x.max() * floor))


def _highpass(x, sigma):
    return x - gaussian_filter(x, sigma)


# --------------------------------------------------------------------- fit
@dataclass
class HeadingFit:
    heading: float                 # degrees true, best
    corr: float                    # correlation there
    headings: np.ndarray           # the scan
    curve: np.ndarray              # correlation at each
    width: float                   # degrees over which corr > 0.9 * peak
    measured: np.ndarray = field(repr=False)    # dB, comparison grid
    simulated: np.ndarray = field(repr=False)   # dB, at the best heading
    antenna_angles: np.ndarray = field(repr=False)

    def as_dict(self):
        return {"heading": round(float(self.heading), 2),
                "corr": round(float(self.corr), 3),
                "width_deg": round(float(self.width), 2)}


def heading_from_dem(par, intensity, dem, headings=None, rmax=12000.0,
                     az_bin=0.2, range_bin=15.0, min_range=1000.0,
                     highpass=(1.5, 150.0), lat0=None, lon0=None,
                     alt0=None, terrain=None):
    """Find the scan heading that lays the DEM's shadows on the image.

    Parameters
    ----------
    par : ParFile or path
        SLC parameter file: azimuth angles, range sampling, radar position.
    intensity : ndarray (azimuth_lines, range_samples)
        Mean |SLC|^2 (see :func:`mean_intensity`); a single image works too.
    dem : path or callable
        As for :func:`polar_terrain`.
    headings : array, optional
        Trial headings, degrees true.  Default 0..360 by 0.2.
    az_bin, range_bin : float
        Comparison grid.  ``az_bin`` should be a multiple of the line step.
    min_range : float
        Metres; nearer samples (the tripod's own bench) are ignored.
    highpass : (float, float)
        Gaussian sigma removed from both images before correlating, in
        degrees and metres: keeps ridge edges and shadow boundaries, drops
        the brightness gradient with range and the speckle.
    terrain : PolarTerrain, optional
        Reuse a previous :func:`polar_terrain` (the expensive half).
    """
    par = par if isinstance(par, ParFile) else ParFile.load(par)
    lat0 = par.float("GPRI_ref_north") if lat0 is None else lat0
    lon0 = par.float("GPRI_ref_east") if lon0 is None else lon0
    if terrain is None:
        terrain = polar_terrain(dem, lat0, lon0, alt0=alt0, rmax=rmax, daz=az_bin)

    r0 = par.float("near_range_slc")
    dr = par.float("range_pixel_spacing")
    step = par.float("GPRI_az_angle_step")
    ml_a = max(1, int(round(az_bin / step)))
    ml_r = max(1, int(round(range_bin / dr)))
    nr = int((rmax - r0) / (dr * ml_r))
    meas = _coarsen(np.asarray(intensity, float)[:, :nr * ml_r], ml_a, ml_r)
    nr = meas.shape[1]
    angles = azimuth_angles(par)
    theta = angles[:meas.shape[0] * ml_a].reshape(-1, ml_a).mean(1)

    bearings, sim = simulate_intensity(terrain, r0, dr * ml_r, nr, daz=az_bin)
    sim_db, meas_db = _db(sim), _db(meas)
    sigma = (highpass[0] / az_bin, highpass[1] / (dr * ml_r))
    M = _highpass(meas_db, sigma)
    S_all = _highpass(sim_db, sigma)                # high-pass once, in bearing
    k0 = int((min_range - r0) / (dr * ml_r))
    sel = (slice(2, -2), slice(max(k0, 0), None))
    m = M[sel].ravel()
    m = (m - m.mean()) / m.std()

    headings = np.arange(0.0, 360.0, az_bin) if headings is None else np.asarray(headings, float)
    curve = np.empty(headings.size)
    nb = bearings.size
    for i, h in enumerate(headings):
        b = np.rint(((theta + h) % 360.0) / az_bin).astype(int) % nb
        s = S_all[b][sel].ravel()
        s = (s - s.mean()) / (s.std() + 1e-12)
        curve[i] = (m * s).mean()
    i = int(np.nanargmax(curve))
    # parabolic refinement between neighbours
    best = headings[i]
    if 0 < i < curve.size - 1:
        y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            best = headings[i] + 0.5 * (y0 - y2) / den * (headings[1] - headings[0])
    width = float((curve > 0.9 * curve[i]).sum() * (headings[1] - headings[0]))
    b = np.rint(((theta + best) % 360.0) / az_bin).astype(int) % nb
    return HeadingFit(float(best), float(curve[i]), headings, curve, width,
                      meas_db, sim_db[b], theta)


# ---------------------------------------------------------------- sidecar
def _work_dir(scene):
    import os
    return Path(os.environ.get("GPRI_WORK_ROOT", "work")) / Path(scene).name


def write_scene_heading(scene, fit, extra=None):
    """Record a fit as ``heading.json`` under ``GPRI_WORK_ROOT/<scene>/``."""
    d = _work_dir(scene)
    d.mkdir(parents=True, exist_ok=True)
    rec = {**fit.as_dict(), "method": "dem", **(extra or {})}
    (d / HEADING_FILE).write_text(json.dumps(rec, indent=1) + "\n")
    return d / HEADING_FILE


def scene_heading(scene, default=None):
    """The measured heading for a scene, or ``default`` with a warning.

    Looks for ``heading.json`` under ``GPRI_WORK_ROOT/<scene>/`` (the work
    directory, so a scene on a read-only archive still gets one).
    """
    import warnings
    f = _work_dir(scene) / HEADING_FILE
    if f.exists():
        return float(json.loads(f.read_text())["heading"])
    if default is None:
        raise FileNotFoundError(f"{f}: run `gpri heading {scene} --write` first")
    warnings.warn(f"{f} missing: using heading {default} deg, a guess -- run "
                  f"`gpri heading {Path(scene).name} --write`", stacklevel=2)
    return float(default)
