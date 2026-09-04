"""Persistent-scatterer interpolation for phase over decorrelated ground.

After Chen, Zebker & Knight (2015), *A persistent scatterer interpolation for
retrieving accurate ground deformation over InSAR-decorrelated agricultural
fields*, Geophys. Res. Lett. 42, 9284-9291.

The problem it solves
---------------------
A GPRI scene at Baker Bend is two populations of pixel.  Rock, moraine and
bare ice hold their phase for hours; fresh snow, crevasse fields and shadowed
slopes decorrelate between consecutive four-minute acquisitions.  Conventional
unwrapping fails on the second population — not because the *deformation* there
is unknowable, but because the noise between reliable pixels exceeds pi and the
unwrapper has no path across the gap.

Chen et al.'s observation is that this is the wrong way round.  Deformation is
driven by a continuous physical process (aquifer head in the original paper, ice
flow here), so the deformation field is **smooth even where the phase is not**.
So: unwrap only at the persistent scatterers, interpolate that sparse, reliable
field across the whole scene, and subtract it.  What is left is small — under a
fringe if the interpolation is any good — and needs no unwrapping at all, because
wrapping a sub-pi residual is the identity.  Add the two back together and every
pixel has an unwrapped phase, including the ones that decorrelated.

The workflow
------------
1. :func:`amplitude_dispersion` / :func:`select_ps` pick the persistent
   scatterers, by Ferretti's ``D_A = sigma_A / mu_A`` and/or by coherence.
2. :func:`unwrap_sparse` unwraps the PS set alone, integrating wrapped
   differences along a minimum spanning tree.  With only reliable pixels in the
   graph there is no path through a decorrelated region to go wrong on.
3. :func:`interpolate_ps` carries the sparse field to the full grid.
4. :func:`unwrap_with_ps` runs 1-3 and does the subtract-wrap-add step.

:func:`unwrap_with_ps` also reports where the assumption failed — the fraction
of pixels whose residual came back near pi, which is where the interpolation
was not good enough and the answer should not be trusted.
"""
from __future__ import annotations

import numpy as np

from .timeseries import wrap

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
    cKDTree = None

__all__ = [
    "amplitude_dispersion", "select_ps", "interpolate_ps", "unwrap_sparse",
    "unwrap_with_ps", "PSResult", "ps_density",
]


# ------------------------------------------------------------- PS selection
def amplitude_dispersion(amplitudes, axis=0):
    """Ferretti's amplitude dispersion index ``D_A = sigma_A / mu_A``.

    Computed over the epoch axis of a stack of amplitude images.  For a bright,
    stable target the phase standard deviation is well approximated by ``D_A``
    itself, which is why a cut at 0.25 is the conventional PS threshold — it
    corresponds to roughly 14 degrees of phase noise.

    Pixels with non-positive mean amplitude come back as ``inf``, so they can
    never be selected.
    """
    a = np.asarray(amplitudes, float)
    mu = np.nanmean(a, axis=axis)
    sigma = np.nanstd(a, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(mu > 0, sigma / mu, np.inf)
    return np.where(np.isfinite(d), d, np.inf)


def select_ps(amplitudes=None, dispersion=None, coherence=None,
              max_dispersion=0.25, min_coherence=None, max_count=None):
    """Boolean mask of persistent scatterers.

    Give it either a stack of ``amplitudes`` (epoch axis first) or a
    precomputed ``dispersion`` image, and optionally a ``coherence`` image —
    temporal coherence from :func:`gpri_tools.phaselink.temporal_coherence` is the
    natural companion, since it measures exactly the thing the amplitude
    dispersion only proxies for.

    ``max_count`` caps the selection at the best N pixels (lowest dispersion,
    or highest coherence where no dispersion is available).  Worth using: the
    minimum spanning tree in :func:`unwrap_sparse` is ``O(n log n)`` in the
    number of PS, and tens of thousands is plenty to constrain a smooth field.
    """
    if dispersion is None and amplitudes is not None:
        dispersion = amplitude_dispersion(amplitudes)

    if dispersion is not None:
        d = np.asarray(dispersion, float)
        mask = d <= max_dispersion
        rank = d
    elif coherence is not None:
        mask = np.ones(np.shape(coherence), bool)
        rank = -np.asarray(coherence, float)
    else:
        raise ValueError("need amplitudes, dispersion, or coherence")

    if coherence is not None and min_coherence is not None:
        c = np.asarray(coherence, float)
        mask = mask & np.isfinite(c) & (c >= min_coherence)

    mask = mask & np.isfinite(rank)

    if max_count is not None and mask.sum() > max_count:
        order = np.argsort(np.where(mask, rank, np.inf), axis=None)
        keep = np.zeros(mask.size, bool)
        keep[order[:max_count]] = True
        mask = keep.reshape(mask.shape)
    return mask


def ps_density(mask):
    """Fraction of the scene selected as persistent scatterers."""
    m = np.asarray(mask, bool)
    return float(m.sum()) / float(m.size) if m.size else 0.0


# ------------------------------------------------------- sparse unwrapping
def _edges(coords, neighbours=8):
    """Candidate graph edges over a scattered point set.

    Delaunay first, because it is connected by construction.  A
    k-nearest-neighbour graph is not, and fails in exactly the case that
    matters here: GPRI coordinates are wildly anisotropic — 0.75 m per range
    sample against tens of metres per azimuth line — so every one of a point's
    k nearest neighbours lies along the range direction and the graph falls
    apart into disconnected range lines that never talk to each other.
    Delaunay has no such preferred direction.

    Falls back to kNN for point sets Delaunay cannot triangulate: fewer than
    three points, or perfectly collinear ones.
    """
    from scipy.spatial import Delaunay, QhullError

    n = len(coords)
    try:
        tri = Delaunay(coords)
        simp = tri.simplices
        pairs = np.vstack([simp[:, [0, 1]], simp[:, [1, 2]], simp[:, [0, 2]]])
    except (QhullError, ValueError):
        k = min(neighbours + 1, n)
        _, nbr = cKDTree(coords).query(coords, k=k)
        pairs = np.column_stack([np.repeat(np.arange(n), k - 1),
                                 nbr[:, 1:].ravel()])
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    keep = lo != hi
    uniq = np.unique(np.column_stack([lo[keep], hi[keep]]), axis=0)
    return uniq[:, 0], uniq[:, 1]


def unwrap_sparse(phase, mask, coords=None, reference=None, weights=None,
                  max_edge=None, neighbours=8):
    """Unwrap phase at the PS pixels only, along a minimum spanning tree.

    The classic 2-D unwrapping failure is a path that crosses a noisy region and
    picks up a spurious ``2 pi``.  Restricting the graph to persistent
    scatterers removes those paths entirely: every edge joins two pixels whose
    phase is individually reliable, so integrating ``wrap(phi_b - phi_a)`` along
    a spanning tree of that graph accumulates no error, provided the true phase
    difference across each edge is under ``pi``.

    Edges come from a Delaunay triangulation of the PS (see :func:`_edges`),
    which is connected by construction.  Edge cost is distance, optionally
    divided by the mean quality of the two endpoints (``weights``), so the tree
    prefers to route through good pixels even when that means going further.

    Parameters
    ----------
    phase : array
        Wrapped phase, full grid, or a 1-D vector already restricted to the PS.
    mask : bool array
        The PS mask, same shape as the full grid.
    coords : (n_ps, 2) array, optional
        Pixel coordinates.  Defaults to ``(row, col)`` in pixels; pass
        ``(azimuth_metres, range_metres)`` to make distances physical, which
        matters for GPRI because a range bin is 0.75 m while an azimuth bin at
        10 km is 30 m.
    reference : int, optional
        Index into the PS set to hold at its wrapped value.  Defaults to the
        highest-weight PS, or the first one.
    max_edge : float, optional
        Refuse to connect PS further apart than this.  The graph may then split,
        and only the reference's component is unwrapped — everything else comes
        back NaN, because its offset really is unknown.  Silence about that
        would be a lie.
    neighbours : int
        Only used for the k-nearest-neighbour fallback when the point set
        cannot be triangulated (fewer than three PS, or collinear ones).

    Returns
    -------
    unwrapped : (n_ps,) float array
        Unwrapped phase at each PS, in the order ``np.flatnonzero(mask)``.
        NaN where a component could not be tied to the reference.
    """
    if cKDTree is None:  # pragma: no cover
        raise ImportError("unwrap_sparse needs scipy")

    mask = np.asarray(mask, bool)
    idx = np.flatnonzero(mask.ravel())
    n = idx.size
    if n == 0:
        return np.zeros(0)

    p = np.asarray(phase, float)
    values = p.ravel()[idx] if p.shape == mask.shape else p.ravel()
    if values.size != n:
        raise ValueError(f"phase has {values.size} values for {n} PS pixels")
    values = wrap(values)

    if coords is None:
        rows, cols = np.unravel_index(idx, mask.shape)
        coords = np.column_stack([rows, cols]).astype(float)
    coords = np.asarray(coords, float).reshape(n, -1)

    if n == 1:
        return values.copy()

    w = None
    if weights is not None:
        w = np.asarray(weights, float)
        w = w.ravel()[idx] if w.shape == mask.shape else w.ravel()
        w = np.clip(np.nan_to_num(w, nan=0.0), 1e-6, None)

    src, dst = _edges(coords, neighbours)
    d = np.linalg.norm(coords[src] - coords[dst], axis=1)

    good = np.isfinite(d)
    if max_edge is not None:
        good &= d <= max_edge
    src, dst, d = src[good], dst[good], d[good]
    if src.size == 0:
        out = np.full(n, np.nan)
        ref = 0 if reference is None else int(reference)
        out[ref] = values[ref]
        return out

    cost = d + 1e-9
    if w is not None:
        cost = cost / (0.5 * (w[src] + w[dst]))

    graph = coo_matrix((cost, (src, dst)), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(graph)
    # symmetrise: the tree is directed, but integration goes both ways
    mst = mst + mst.T
    if reference is None:
        reference = int(np.argmax(w)) if w is not None else 0
    reference = int(reference)

    out = np.full(n, np.nan)
    mst = mst.tocsr()
    # breadth-first integration outward from the reference.  Anything in another
    # component stays NaN: its 2 pi offset relative to the reference is
    # genuinely not determined by the data, and guessing it would be a lie.
    out[reference] = values[reference]
    seen = np.zeros(n, bool)
    seen[reference] = True
    frontier = [reference]
    while frontier:
        nxt = []
        for a in frontier:
            lo, hi = mst.indptr[a], mst.indptr[a + 1]
            for b in mst.indices[lo:hi]:
                if seen[b]:
                    continue
                seen[b] = True
                out[b] = out[a] + wrap(values[b] - values[a])
                nxt.append(b)
        frontier = nxt
    return out


# -------------------------------------------------------------- interpolation
def interpolate_ps(values, mask, shape=None, coords=None, method="linear",
                   power=2.0, neighbours=12, fill="nearest", smooth=0.0):
    """Carry a sparse PS field onto the full grid.

    Parameters
    ----------
    values : (n_ps,) array
        Unwrapped phase (or displacement, or anything smooth) at the PS.
    mask : bool array
        PS mask defining where those values sit.
    method : {'linear', 'idw', 'nearest'}
        ``'linear'`` is Delaunay-based linear interpolation — the closest thing
        to the natural-neighbour scheme of the original paper, and the right
        default for a field with real spatial structure.  ``'idw'`` is inverse
        distance weighting over the ``neighbours`` nearest PS, which is more
        robust when the PS are clustered along a few rock ribs and the Delaunay
        triangulation grows slivers.
    fill : {'nearest', 'nan'}
        What to do outside the convex hull of the PS.  ``'nearest'`` extends the
        edge value; ``'nan'`` admits that it is extrapolation.
    smooth : float
        Optional Gaussian smoothing of the result, in pixels.  A little helps
        when the PS are dense and noisy; too much eats real gradients.

    Returns
    -------
    field : array
        Same shape as ``mask``.
    """
    if cKDTree is None:  # pragma: no cover
        raise ImportError("interpolate_ps needs scipy")

    mask = np.asarray(mask, bool)
    shape = mask.shape if shape is None else tuple(shape)
    idx = np.flatnonzero(mask.ravel())
    v = np.asarray(values, float).ravel()
    if v.size != idx.size:
        raise ValueError(f"{v.size} values for {idx.size} PS pixels")

    finite = np.isfinite(v)
    idx, v = idx[finite], v[finite]
    if idx.size == 0:
        return np.full(shape, np.nan)

    if coords is None:
        rows, cols = np.unravel_index(idx, shape)
        pts = np.column_stack([rows, cols]).astype(float)
        gr, gc = np.meshgrid(np.arange(shape[0], dtype=float),
                             np.arange(shape[1], dtype=float), indexing="ij")
        grid = np.column_stack([gr.ravel(), gc.ravel()])
    else:
        coords = np.asarray(coords, float)
        if coords.shape[0] == mask.size:
            pts = coords[idx]
            grid = coords
        else:
            pts = coords[finite] if coords.shape[0] == finite.size else coords
            gr, gc = np.meshgrid(np.arange(shape[0], dtype=float),
                                 np.arange(shape[1], dtype=float), indexing="ij")
            grid = np.column_stack([gr.ravel(), gc.ravel()])

    if idx.size == 1:
        out = np.full(shape, v[0])
        return out

    if method == "nearest":
        out = NearestNDInterpolator(pts, v)(grid).reshape(shape)
    elif method == "idw":
        tree = cKDTree(pts)
        k = min(neighbours, pts.shape[0])
        dist, nbr = tree.query(grid, k=k)
        if k == 1:
            dist, nbr = dist[:, None], nbr[:, None]
        with np.errstate(divide="ignore"):
            w = 1.0 / np.power(dist, power)
        exact = ~np.isfinite(w)
        w = np.where(exact, 0.0, w)
        # a grid point sitting exactly on a PS takes that PS's value
        w[exact.any(axis=1)] = exact[exact.any(axis=1)].astype(float)
        out = (w * v[nbr]).sum(axis=1) / w.sum(axis=1)
        out = out.reshape(shape)
    elif method == "linear":
        out = LinearNDInterpolator(pts, v)(grid).reshape(shape)
        if fill == "nearest":
            hole = ~np.isfinite(out)
            if hole.any():
                near = NearestNDInterpolator(pts, v)(grid[hole.ravel()])
                out[hole] = near
    else:
        raise ValueError(f"unknown interpolation method {method!r}")

    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        good = np.isfinite(out)
        filled = np.where(good, out, 0.0)
        num = gaussian_filter(filled, smooth, mode="nearest")
        den = gaussian_filter(good.astype(float), smooth, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
    return out


# ------------------------------------------------------------------- workflow
class PSResult:
    """Output of :func:`unwrap_with_ps`, with the diagnostics that matter."""

    def __init__(self, unwrapped, interpolated, residual, mask, ps_unwrapped,
                 n_ps, n_unresolved):
        #: full-grid unwrapped phase, radians
        self.unwrapped = unwrapped
        #: the smooth PS-interpolated field that was subtracted
        self.interpolated = interpolated
        #: what was left after subtracting it, wrapped; should be well under pi
        self.residual = residual
        #: the PS mask used
        self.mask = mask
        #: unwrapped phase at the PS alone
        self.ps_unwrapped = ps_unwrapped
        self.n_ps = n_ps
        #: PS the spanning tree could not tie to the reference
        self.n_unresolved = n_unresolved

    @property
    def suspect(self):
        """Pixels whose residual came back near ``pi``.

        Where the interpolated field was off by more than half a fringe, the
        wrap in the subtract-wrap-add step aliased and the answer at that pixel
        is wrong.  This is the honest failure map, not a quality score.
        """
        return np.abs(self.residual) > 0.8 * np.pi

    @property
    def suspect_fraction(self):
        s = self.suspect
        return float(np.count_nonzero(s)) / float(s.size) if s.size else 0.0

    def __repr__(self):
        return (f"PSResult({self.n_ps} PS, "
                f"{self.suspect_fraction * 100:.1f}% suspect, "
                f"{self.n_unresolved} unresolved)")


def unwrap_with_ps(phase, mask=None, amplitudes=None, coherence=None,
                   coords=None, reference=None, method="linear",
                   max_dispersion=0.25, min_coherence=None, max_count=20000,
                   max_edge=None, smooth=0.0, weights=None):
    """PS-interpolation unwrapping, end to end (Chen, Zebker & Knight 2015).

    >>> res = unwrap_with_ps(wrapped, coherence=cc, min_coherence=0.7)
    >>> res.unwrapped.shape, res.suspect_fraction
    ((396, 22101), 0.004)

    Parameters
    ----------
    phase : array
        Wrapped interferometric phase, or a complex interferogram (its angle is
        taken).
    mask : bool array, optional
        PS mask.  If not given it is built from ``amplitudes`` / ``coherence``
        via :func:`select_ps`.
    coords : (n_pixels, 2) array, optional
        Physical coordinates for every grid pixel, ``(azimuth_m, range_m)``.
        Strongly recommended for GPRI, where pixel spacing is wildly anisotropic
        — see :func:`gpri_tools.gamma.ground_range`.
    weights : array, optional
        Per-pixel quality (coherence is the obvious choice) steering both the
        spanning tree and the choice of reference.

    Returns
    -------
    :class:`PSResult`
    """
    p = np.asarray(phase)
    wrapped = wrap(np.angle(p) if np.iscomplexobj(p) else p.astype(float))

    if mask is None:
        mask = select_ps(amplitudes=amplitudes, coherence=coherence,
                         max_dispersion=max_dispersion,
                         min_coherence=min_coherence, max_count=max_count)
    mask = np.asarray(mask, bool)
    if mask.shape != wrapped.shape:
        raise ValueError(f"mask {mask.shape} does not match phase {wrapped.shape}")
    if not mask.any():
        raise ValueError("no persistent scatterers selected; relax the thresholds")

    if weights is None and coherence is not None:
        weights = coherence

    ps_coords = None
    if coords is not None:
        coords = np.asarray(coords, float)
        ps_coords = coords.reshape(-1, coords.shape[-1])[np.flatnonzero(mask.ravel())]

    ps_unwrapped = unwrap_sparse(wrapped, mask, coords=ps_coords,
                                 reference=reference, weights=weights,
                                 max_edge=max_edge)
    n_unresolved = int(np.count_nonzero(~np.isfinite(ps_unwrapped)))

    interpolated = interpolate_ps(ps_unwrapped, mask, shape=wrapped.shape,
                                  coords=coords, method=method, smooth=smooth)

    # the whole trick: a good interpolation leaves a sub-fringe residual, and
    # wrapping something already inside (-pi, pi] does nothing to it
    residual = wrap(wrapped - interpolated)
    unwrapped = interpolated + residual

    return PSResult(unwrapped, interpolated, residual, mask, ps_unwrapped,
                    int(mask.sum()), n_unresolved)
