"""Glacier outlines from the Randolph Glacier Inventory, in radar geometry.

Why this matters here, and not just for cartography: the pipeline's stable
"bedrock" reference is chosen by *coherence* — but high coherence does not
mean stationary.  Ice moving tens of millimetres per day decorrelates slowly
at a 2-minute pair spacing, so a coherence-only mask can quietly include
moving glacier surface.  Every correction and every null test is then
referenced to ground that moves: real deformation leaks out of the maps, and
an artificial signal leaks onto true bedrock.  The RGI is the independent
answer to "where is the ice?" that coherence cannot give.

Workflow
--------
1. :func:`load_outlines` reads RGI polygons (shapefile, or the nested
   region zips inside the global RGI distribution) clipped to a bounding box.
2. :func:`glacier_mask` rasterises them onto the radar grid through
   :class:`gpri.geocode.RadarGeometry` — so it inherits the scan-heading
   caveat: with the heading wrong the mask lands in the wrong place, and the
   overlay figure is itself a check on the heading.
3. :func:`stable_ground_mask` is the corrected reference selector:
   coherent **and** outside the (buffered) glacier outlines.

The buffer matters: RGI outlines are digitised from optical imagery at a
different epoch and carry their own uncertainty; a margin of ice-cored
moraine can move too.  Default 100 m keeps the reference clear of all of it.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

__all__ = ["load_outlines", "glacier_mask", "stable_ground_mask",
           "outline_paths"]


def _require_geopandas():
    try:
        import geopandas as gpd
        return gpd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("RGI handling needs geopandas") from exc


def load_outlines(source, bbox=None, region_prefix="02_"):
    """RGI glacier polygons, optionally clipped to ``bbox`` (WGS84).

    Parameters
    ----------
    source : path
        Any of: a shapefile (``.shp``), a single-region RGI zip, or the
        global RGI distribution zip that contains per-region zips — for the
        last of these the inner zip whose name starts with ``region_prefix``
        (default region 02, Western Canada & USA) is extracted next to the
        source on first use and read thereafter.
    bbox : (lon_min, lat_min, lon_max, lat_max), optional
        Keep only glaciers intersecting this box.  Pass one; reading a whole
        RGI region (~18k polygons) when you want one mountain is slow.

    Returns
    -------
    geopandas.GeoDataFrame in EPSG:4326.
    """
    gpd = _require_geopandas()
    source = Path(source)

    if source.suffix == ".zip" and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            inner = [n for n in z.namelist()
                     if Path(n).name.startswith(region_prefix)
                     and n.endswith(".zip")]
            if inner:                        # global distribution: extract once
                target = source.parent / Path(inner[0]).name
                if not target.exists():
                    with z.open(inner[0]) as f, open(target, "wb") as out:
                        out.write(f.read())
                source = target
        source = Path(source)
        read_path = f"zip://{source}"
    elif source.suffix == ".zip":
        read_path = f"zip://{source}"
    else:
        read_path = str(source)

    gdf = gpd.read_file(read_path, bbox=bbox)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def outline_paths(gdf, geom):
    """Outline vertex arrays in the radar's map frame, for plotting.

    Returns a list of ``(x, y)`` vertex arrays (metres in ``geom.crs``), one
    per exterior ring, ready for ``ax.plot(x/1000, y/1000)``.
    """
    _require_geopandas()
    proj = gdf.to_crs(geom.crs)
    out = []
    for g in proj.geometry:
        if g is None or g.is_empty:
            continue
        polys = g.geoms if hasattr(g, "geoms") else [g]
        for poly in polys:
            x, y = poly.exterior.xy
            out.append((np.asarray(x), np.asarray(y)))
    return out


def glacier_mask(geom, outlines, buffer_m=0.0):
    """Boolean radar-grid mask: True where the pixel centre falls on a glacier.

    Works in the radar's own map frame (metres), so ``buffer_m`` is a real
    distance.  Inherits the scan-heading uncertainty of ``geom``: half a
    degree of heading is ~80 m of position at 9 km, which is comparable to
    the default reference buffer — one more reason to tie the heading to a
    feature before trusting fine mask edges.
    """
    from shapely import vectorized
    from shapely.ops import unary_union

    proj = outlines.to_crs(geom.crs)
    union = unary_union(list(proj.geometry.values))
    if buffer_m:
        union = union.buffer(float(buffer_m))
    x, y = geom.map_coordinates()
    return vectorized.contains(union, x, y)


def stable_ground_mask(coherence, geom, outlines, threshold=0.85,
                       buffer_m=100.0):
    """The reference selector done right: coherent AND off the ice.

    Returns ``(stable, on_glacier)`` where ``stable`` is the corrected
    reference mask and ``on_glacier`` is the part of the coherence-only mask
    that the RGI says is glacier — the pixels the old mask was wrong about.
    A large ``on_glacier`` fraction means every earlier bedrock-referenced
    product was tied to moving ground.
    """
    cc = np.asarray(coherence, float)
    coherent = cc >= threshold
    ice = glacier_mask(geom, outlines, buffer_m=buffer_m)
    if ice.shape != coherent.shape:
        raise ValueError(f"glacier mask {ice.shape} does not match "
                         f"coherence {coherent.shape}")
    return coherent & ~ice, coherent & ice
