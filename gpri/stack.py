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

__all__ = ["DiffStack", "SlcPairStack", "find_pairs", "SCENE_ID_RE",
           "coherence_window"]

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


class _PairStack:
    """Tile iteration shared by every stack that can ``read_pair`` a tile."""

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

    def __len__(self):
        return self.n_pairs


class DiffStack(_PairStack):
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

    def close(self):
        self._maps.clear()
        self._cc_maps.clear()

    def __repr__(self):
        na, nr = self.shape
        return (f"DiffStack({self.n_pairs} pairs, {self.n_epochs} epochs, "
                f"{na}x{nr})")


# ------------------------------------------------------------ SLC-formed pairs
def coherence_window(size=(5, 5), weighting="triangular"):
    """Separable coherence-estimation weights ``(w_azimuth, w_range)``.

    GAMMA's ``cc_wave`` with ``bx = by = 5`` and a triangular weighting is what
    produced the BakerBend1 ``.cc`` rasters: a 5 x 5 triangle reproduces them
    at correlation 0.998 and 0.014 rms, where a plain 5 x 5 boxcar manages
    0.95 and 0.09.  Matching the estimator matters because every mask and
    weight downstream is a threshold on this number, and the two antennas
    have to be judged on the same scale.
    """
    def one(n):
        n = int(n)
        if weighting == "triangular":
            w = 1.0 - np.abs(np.arange(n) - (n - 1) / 2) / ((n + 1) / 2)
        elif weighting == "gaussian":
            x = np.arange(n) - (n - 1) / 2
            w = np.exp(-0.5 * (x / max(n / 4, 1e-9)) ** 2)
        elif weighting == "boxcar":
            w = np.ones(n)
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
        return w / w.sum()
    return one(size[0]), one(size[1])


def _smooth(a, wa, wr):
    """Separable weighted mean, edge-replicated; complex arrays pass through."""
    from scipy.ndimage import convolve1d
    if np.iscomplexobj(a):
        return (_smooth(a.real, wa, wr) + 1j * _smooth(a.imag, wa, wr)).astype(np.complex64)
    out = convolve1d(np.asarray(a, np.float32), wa, axis=0, mode="nearest")
    return convolve1d(out, wr, axis=1, mode="nearest")


def _multilook(a, looks):
    """Boxcar-average then subsample; drops the ragged edge like GAMMA does."""
    la, lr = looks
    if la == 1 and lr == 1:
        return a
    na, nr = a.shape[0] // la, a.shape[1] // lr
    return a[: na * la, : nr * lr].reshape(na, la, nr, lr).mean(axis=(1, 3))


class SlcPairStack(_PairStack):
    """Interferograms formed on the fly from a set of coregistered SLCs.

    Same interface as :class:`DiffStack`, so every consumer that walks a
    ``diff0`` walks this unchanged — but the pairs come from
    ``s_ref * conj(s_sec)`` over an SLC list, with the network defined by
    ``lags``: ``(1,)`` is the daisy chain GAMMA shipped, ``(1, 2, 3)`` adds
    the i->i+2 and i->i+3 pairs that close triangles.

    Parameters
    ----------
    images : sequence of paths
        SLCs in time order.  GPRI is tripod-mounted, so a deployment's scenes
        are coregistered already.
    par : path or :class:`gpri.gamma.ParFile`
        Geometry of the SLCs (any one of them; they share it).
    lags : sequence of int
        Epoch offsets to pair, ``(i, i + lag)`` for every ``i``.
    looks : (int, int)
        Multilook factors (azimuth, range) applied to the complex product
        before the phase is taken.  ``(1, 1)`` gives GAMMA's 1-look ``.diff``
        exactly.  Anything larger *creates* closure-phase bias (1-look
        closure is identically zero) — which is the point when the aim is to
        measure it.
    coherence : (int, int)
        Window of the coherence estimate on the full-resolution product, before
        multilooking (``cc_wave`` semantics; see :func:`coherence_window`).
    weighting : str
        ``"triangular"`` (GAMMA's, the default), ``"gaussian"`` or ``"boxcar"``.
    cache : int, optional
        SLCs held in memory.  Pairs are ordered so that walking them in index
        order needs only ``max(lags) + 1`` scenes at a time; that is the
        default.
    """

    def __init__(self, images, par, lags=(1,), looks=(1, 1), coherence=(5, 5),
                 weighting="triangular", cache=None, network=None,
                 image_format=None):
        self.images = [Path(p) for p in images]
        slc_par = par if isinstance(par, ParFile) else ParFile.load(par)
        self.slc_par = slc_par
        self.image_format = image_format or slc_par.image_format
        self.lags = tuple(int(k) for k in lags)
        if not self.lags or min(self.lags) < 1:
            raise ValueError("lags must be positive epoch offsets")
        self.looks = (int(looks[0]), int(looks[1]))
        self.par = self._looked_par(slc_par, self.looks)
        self._wa, self._wr = coherence_window(coherence, weighting)
        n = len(self.images)
        # ordered by reference epoch, then lag: (0,1) (0,2) (0,3) (1,2) ...
        pairs = [(i, i + k) for i in range(n) for k in self.lags if i + k < n]
        self._pairs = pairs
        if network is None:
            ids = [SCENE_ID_RE.search(p.name) for p in self.images]
            if any(m is None for m in ids):
                raise ValueError("cannot parse acquisition times from SLC names")
            network = Network([parse_epoch(m.group(1)) for m in ids], pairs)
        self.network = network
        self._cache_size = (max(self.lags) + 1) if cache is None else int(cache)
        self._slcs = {}          # epoch -> complex64 array, LRU by insertion
        self._power = {}         # epoch -> smoothed intensity, same policy
        self._last = None        # (p, ifg, cc) of the last pair formed

    # ------------------------------------------------------------ construction
    @classmethod
    def from_tab(cls, slc_tab, **kw):
        """From a GAMMA ``SLC_tab`` (image and parameter paths per line)."""
        from .network import read_slc_tab
        slc_tab = Path(slc_tab)
        images, pars = read_slc_tab(slc_tab)
        root = slc_tab.parent
        return cls([root / p for p in images], root / pars[0], **kw)

    @classmethod
    def from_directory(cls, slc_dir, antenna="u", suffix=".slc", **kw):
        """Every ``*<antenna><suffix>`` in a directory, in time order.

        GAMMA writes an ``SLC_tab`` only for the antenna it processed; the
        other antenna's SLCs sit beside them with nothing pointing at them.
        """
        slc_dir = Path(slc_dir)
        images = sorted(p for p in slc_dir.glob(f"*{antenna}{suffix}")
                        if SCENE_ID_RE.search(p.name))
        if not images:
            raise FileNotFoundError(f"no *{antenna}{suffix} in {slc_dir}")
        images.sort(key=lambda p: parse_epoch(SCENE_ID_RE.search(p.name).group(1)))
        par = Path(str(images[0]) + ".par")
        return cls(images, par, **kw)

    @staticmethod
    def _looked_par(par, looks):
        la, lr = looks
        if la == 1 and lr == 1:
            return par
        e = {k: list(v) for k, v in par.entries.items()}
        e["azimuth_lines"] = [str(par.azimuth_lines // la)]
        e["range_samples"] = [str(par.range_samples // lr)]
        e["range_pixel_spacing"] = [str(par.range_pixel_spacing * lr), "m"]
        e["range_looks"] = [str(par.int("range_looks", 1) * lr)]
        e["azimuth_looks"] = [str(par.int("azimuth_looks", 1) * la)]
        step = par.float("GPRI_az_angle_step", 0.0)
        if step:
            start = par.float("GPRI_az_start_angle", 0.0)
            e["GPRI_az_angle_step"] = [f"{step * la:.6e}", "degrees"]
            e["GPRI_az_start_angle"] = [f"{start + step * (la - 1) / 2:.6f}", "degrees"]
        return ParFile(e, par.header)

    # -------------------------------------------------------------- properties
    @property
    def shape(self):
        return self.par.shape

    @property
    def n_pairs(self):
        return len(self._pairs)

    @property
    def n_epochs(self):
        return len(self.images)

    @property
    def wavelength(self):
        return self.par.wavelength

    def slant_range(self):
        return self.par.slant_range()

    def azimuth_angles(self):
        from .gamma import azimuth_angles
        return azimuth_angles(self.par)

    # --------------------------------------------------------------- reading
    def _slc(self, e):
        if e not in self._slcs:
            while len(self._slcs) >= self._cache_size:
                oldest = next(iter(self._slcs))
                self._slcs.pop(oldest)
                self._power.pop(oldest, None)
            self._slcs[e] = read_image(self.images[e], shape=self.slc_par.shape,
                                       image_format=self.image_format
                                       ).astype(np.complex64)
        return self._slcs[e]

    def _smoothed_power(self, e):
        """Windowed intensity of one epoch: shared by every pair it is in."""
        if e not in self._power:
            self._power[e] = _smooth(np.abs(self._slc(e)) ** 2, self._wa, self._wr)
        return self._power[e]

    def _form(self, p):
        """Full-frame interferogram and coherence for pair ``p`` (cached)."""
        if self._last is not None and self._last[0] == p:
            return self._last[1], self._last[2]
        i, j = self._pairs[p]
        a, b = self._slc(i), self._slc(j)
        prod = a * np.conj(b)
        num = _smooth(prod, self._wa, self._wr)
        den = np.sqrt(self._smoothed_power(i) * self._smoothed_power(j))
        with np.errstate(invalid="ignore", divide="ignore"):
            cc = np.where(den > 0, np.abs(num) / den, 0.0).astype(np.float32)
        ifg = _multilook(prod, self.looks).astype(np.complex64)
        cc = _multilook(cc, self.looks).astype(np.float32)
        self._last = (p, ifg, cc)
        return ifg, cc

    def read_pair(self, p, rows=None, cols=None):
        """One interferogram, or a tile of it, as ``complex64``."""
        rows = slice(None) if rows is None else rows
        cols = slice(None) if cols is None else cols
        return self._form(p)[0][rows, cols]

    def read_coherence(self, p, rows=None, cols=None):
        """``cc_wave``-style coherence estimate for the pair."""
        rows = slice(None) if rows is None else rows
        cols = slice(None) if cols is None else cols
        return self._form(p)[1][rows, cols]

    def close(self):
        self._slcs.clear()
        self._power.clear()
        self._last = None

    def __repr__(self):
        na, nr = self.shape
        return (f"SlcPairStack({self.n_pairs} pairs, {self.n_epochs} epochs, "
                f"lags {self.lags}, looks {self.looks}, {na}x{nr})")
