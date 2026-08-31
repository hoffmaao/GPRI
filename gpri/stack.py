"""Patch-wise access to a whole directory of GAMMA interferograms.

A BakerBend1 ``diff0`` directory holds 723 interferograms of 22101 x 396
FCOMPLEX — 70 MB each, **50 GB** for the stack, with the matching ``.cc``
coherence rasters adding another 25 GB.  Nothing here loads that; every raster
is memory-mapped and read in tiles sized to a budget you set.

The tiling is not incidental.  Phase linking needs the N x N coherence matrix
at every output pixel, which is ``723^2 * 16 = 8.4 MB`` *per pixel*, so the
only way through the full stack is a small spatial window at a time — and
:meth:`DiffStack.patches` hands you exactly that window across all epochs at
once, which is the shape :mod:`gpri.covariance` and :mod:`gpri.phaselink` want.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .gamma import ParFile, map_image, read_image
from .network import Network, parse_epoch

__all__ = ["DiffStack", "find_pairs", "SCENE_ID_RE"]

#: A GPRI scene id: date, time, and the antenna letter (``u`` upper, ``l`` lower).
SCENE_ID_RE = re.compile(r"(\d{8}_\d{6}[ul]?)")


def find_pairs(diff_dir, suffix=".diff", exclude_self=True):
    """Discover interferogram files and the scene pair each one spans.

    GAMMA names differential interferograms ``<ref>_<sec><suffix>``, e.g.
    ``20170803_222136u_20170803_222556u.diff``.  Returns
    ``[(ref_id, sec_id, path), ...]`` sorted by acquisition time.

    ``exclude_self`` drops the ``<x>_<x>`` self-pair that GAMMA's stacking
    scripts emit as the first row of an ``itab``; it carries no phase and would
    make any design matrix rank deficient.
    """
    diff_dir = Path(diff_dir)
    out = []
    for path in sorted(diff_dir.iterdir()):
        if not path.name.endswith(suffix):
            continue
        # ".adf.diff" must not be picked up by a ".diff" query
        stem = path.name[: -len(suffix)]
        if "." in stem:
            continue
        ids = SCENE_ID_RE.findall(stem)
        if len(ids) != 2:
            continue
        ref, sec = ids
        if exclude_self and ref == sec:
            continue
        out.append((ref, sec, path))
    out.sort(key=lambda t: (parse_epoch(t[0]), parse_epoch(t[1])))
    return out


class DiffStack:
    """A stack of GAMMA interferograms, read lazily in tiles.

    >>> stack = DiffStack.from_directory("20170803/diff0", slc_tab="20170803/SLCu_tab")
    >>> stack
    DiffStack(722 pairs, 723 epochs, 396x22101)
    >>> for rows, cols, ifg, cc in stack.patches(max_gib=2.0):
    ...     ...          # ifg is (722, nrows, ncols) complex64
    """

    def __init__(self, paths, par, network=None, cc_paths=None,
                 image_format="FCOMPLEX", cc_format="FLOAT"):
        self.paths = [Path(p) for p in paths]
        self.cc_paths = None if cc_paths is None else [
            None if p is None else Path(p) for p in cc_paths]
        self.par = par if isinstance(par, ParFile) else ParFile.load(par)
        self.network = network
        self.image_format = image_format
        self.cc_format = cc_format
        self._maps = {}
        self._cc_maps = {}

    # ------------------------------------------------------------ construction
    @classmethod
    def from_directory(cls, diff_dir, slc_tab=None, par=None, suffix=".diff",
                       cc_suffix=".cc", network=None, epochs=None):
        """Build a stack from a GAMMA ``diff0`` directory.

        Parameters
        ----------
        diff_dir : path
        slc_tab : path, optional
            ``SLC_tab``/``MLI_tab`` defining the epoch ordering.  Without it the
            epochs are taken to be the scenes that actually appear in the
            filenames, in time order.
        par : path or :class:`gpri.gamma.ParFile`, optional
            Geometry for the interferograms.  Defaults to the first ``.off``
            beside them, then to the first SLC parameter file in ``slc_tab``.
        suffix : str
            ``".diff"`` for the raw interferograms, ``".adf.diff"`` for the
            adaptive-filtered ones (only 296 of the 723 BakerBend1 pairs have
            those).
        """
        diff_dir = Path(diff_dir)
        found = find_pairs(diff_dir, suffix=suffix)
        if not found:
            raise FileNotFoundError(f"no *{suffix} interferograms in {diff_dir}")

        if slc_tab is not None:
            from .network import read_slc_tab
            images, _ = read_slc_tab(slc_tab)
            order = [SCENE_ID_RE.search(Path(p).name).group(1) for p in images]
        else:
            order = sorted({i for r, s, _ in found for i in (r, s)},
                           key=parse_epoch)
        index = {sid: k for k, sid in enumerate(order)}

        pairs, paths, ccs = [], [], []
        for ref, sec, path in found:
            if ref not in index or sec not in index:
                continue
            if epochs is not None and (index[ref] not in epochs or index[sec] not in epochs):
                continue
            pairs.append((index[ref], index[sec]))
            paths.append(path)
            cc = path.with_name(path.name[: -len(suffix)] + cc_suffix)
            ccs.append(cc if cc.exists() else None)

        if network is None:
            network = Network([parse_epoch(s) for s in order], pairs,
                              paths=[str(p) for p in paths])

        if par is None:
            # Prefer the SLC/MLI parameter file: a ".off" carries the raster
            # dimensions but not radar_frequency, near_range_slc, or the
            # GPRI azimuth-sweep keys, all of which the rest of the package
            # needs.  Fall back to the ".off" only if there is no tab.
            if slc_tab is not None:
                from .network import read_slc_tab
                _, par_paths = read_slc_tab(slc_tab)
                par = Path(slc_tab).parent / par_paths[0]
            else:
                offs = sorted(diff_dir.glob("*.off"))
                if not offs:
                    raise ValueError("cannot find a parameter file; pass par=")
                par = offs[0]
        return cls(paths, par, network=network, cc_paths=ccs)

    # -------------------------------------------------------------- properties
    @property
    def shape(self):
        """``(azimuth_lines, range_samples)`` of every raster in the stack."""
        return self.par.shape

    @property
    def n_pairs(self):
        return len(self.paths)

    @property
    def n_epochs(self):
        return self.network.n_epochs if self.network is not None else 0

    @property
    def wavelength(self):
        return self.par.wavelength

    def slant_range(self):
        return self.par.slant_range()

    def azimuth_angles(self):
        from .gamma import azimuth_angles
        return azimuth_angles(self.par)

    # --------------------------------------------------------------- reading
    def _map(self, p):
        if p not in self._maps:
            self._maps[p] = map_image(self.paths[p], shape=self.shape,
                                      image_format=self.image_format)
        return self._maps[p]

    def _cc_map(self, p):
        if self.cc_paths is None or self.cc_paths[p] is None:
            return None
        if p not in self._cc_maps:
            self._cc_maps[p] = map_image(self.cc_paths[p], shape=self.shape,
                                         image_format=self.cc_format)
        return self._cc_maps[p]

    def read_pair(self, p, rows=None, cols=None):
        """One interferogram, or a tile of it, as ``complex64``."""
        rows = slice(None) if rows is None else rows
        cols = slice(None) if cols is None else cols
        return np.asarray(self._map(p)[rows, cols], dtype=np.complex64)

    def read_coherence(self, p, rows=None, cols=None):
        """The matching ``.cc`` tile, or ``None`` if that pair has no ``.cc``."""
        m = self._cc_map(p)
        if m is None:
            return None
        rows = slice(None) if rows is None else rows
        cols = slice(None) if cols is None else cols
        return np.asarray(m[rows, cols], dtype=np.float32)

    def read_patch(self, rows, cols, coherence=True):
        """All pairs over one tile: ``(P, nrows, ncols)`` complex, plus coherence."""
        ifg = np.empty((self.n_pairs,
                        len(range(*rows.indices(self.shape[0]))),
                        len(range(*cols.indices(self.shape[1])))), np.complex64)
        cc = np.empty(ifg.shape, np.float32) if coherence else None
        for p in range(self.n_pairs):
            ifg[p] = self.read_pair(p, rows, cols)
            if cc is not None:
                c = self.read_coherence(p, rows, cols)
                cc[p] = np.abs(ifg[p]) if c is None else c
        return ifg, cc

    def patch_shape(self, max_gib=2.0, full_width=True):
        """Tile size that keeps one patch of the whole stack under ``max_gib``.

        Counts the interferograms (8 bytes/pixel) and the coherence
        (4 bytes/pixel) together.
        """
        na, nr = self.shape
        per_pixel = self.n_pairs * 12
        budget = max(1, int(max_gib * 2 ** 30 // per_pixel))
        if full_width:
            rows = max(1, min(na, budget // nr))
            return rows, nr
        side = max(1, int(np.sqrt(budget)))
        return min(na, side), min(nr, side)

    def patches(self, rows=None, cols=None, max_gib=2.0, coherence=True):
        """Iterate tiles of the stack.

        Yields ``(row_slice, col_slice, ifg, cc)`` where ``ifg`` is
        ``(n_pairs, nrows, ncols)`` complex64 and ``cc`` is the same shape in
        float32 (falling back to interferogram magnitude where no ``.cc``
        exists).
        """
        na, nr = self.shape
        if rows is None or cols is None:
            r, c = self.patch_shape(max_gib=max_gib)
            rows = rows or r
            cols = cols or c
        for i in range(0, na, rows):
            for j in range(0, nr, cols):
                rs = slice(i, min(i + rows, na))
                cs = slice(j, min(j + cols, nr))
                ifg, cc = self.read_patch(rs, cs, coherence=coherence)
                yield rs, cs, ifg, cc

    def close(self):
        self._maps.clear()
        self._cc_maps.clear()

    def __len__(self):
        return self.n_pairs

    def __repr__(self):
        na, nr = self.shape
        return (f"DiffStack({self.n_pairs} pairs, {self.n_epochs} epochs, "
                f"{na}x{nr})")
