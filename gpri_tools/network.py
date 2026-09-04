"""Interferogram networks: GAMMA table files, epochs, and the SBAS design matrix.

GAMMA describes a stack with a handful of plain-text tables:

``SLC_tab`` / ``MLI_tab``
    One row per epoch: ``<image path>  <parameter path>``.
``itab``
    One row per pair, 1-based indices into the ``SLC_tab``:
    ``<ref> <sec> <pair number> <flag>``.  ``flag`` of 0 disables the pair.
``DIFF_tab``
    One column of differential interferogram paths, ordered like the ``itab``.

Both network topologies in the BakerBend1 data are handled: single-reference
(``1 2``, ``1 3``, ``1 4`` …) and sequential daisy-chain (``1 2``, ``2 3`` …).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

#: GPRI scene identifiers look like ``20170803_222136u`` (date_time + antenna).
SCENE_RE = re.compile(r"(\d{8})[_T](\d{6})")


def parse_epoch(name) -> datetime:
    """Pull an acquisition time out of a GPRI file or scene name.

    >>> parse_epoch("slc/20170803_222136u.slc")
    datetime.datetime(2017, 8, 3, 22, 21, 36)
    """
    m = SCENE_RE.search(str(name))
    if not m:
        raise ValueError(f"no YYYYMMDD_HHMMSS timestamp in {name!r}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def scene_id(name) -> str:
    """The ``20170803_222136u`` part of a path, without directory or suffix."""
    stem = Path(str(name)).name
    for suffix in (".slc", ".mli", ".diff", ".cc", ".rslc"):
        stem = stem.split(suffix)[0]
    return stem


# ------------------------------------------------------------------ table IO
def read_tab(path) -> list[list[str]]:
    """Read any GAMMA ``*_tab`` file as a list of whitespace-split rows."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line.split())
    return rows


def read_slc_tab(path) -> tuple[list[str], list[str]]:
    """Return ``(image_paths, par_paths)`` from an ``SLC_tab`` / ``MLI_tab``."""
    rows = read_tab(path)
    images = [r[0] for r in rows]
    pars = [r[1] if len(r) > 1 else r[0] + ".par" for r in rows]
    return images, pars


def read_diff_tab(path) -> list[str]:
    """Return the differential interferogram paths from a ``DIFF_tab``."""
    return [r[0] for r in read_tab(path)]


def read_itab(path, drop_self=True, honour_flag=True) -> np.ndarray:
    """Read an ``itab`` into a 0-based ``(n_pairs, 2)`` integer array.

    GAMMA itabs are 1-based and routinely include a ``1 1`` self-pair, which
    carries no phase and would make the design matrix rank-deficient; it is
    dropped by default.
    """
    pairs = []
    for row in read_tab(path):
        if len(row) < 2:
            continue
        ref, sec = int(row[0]) - 1, int(row[1]) - 1
        if honour_flag and len(row) >= 4 and int(row[3]) == 0:
            continue
        if drop_self and ref == sec:
            continue
        pairs.append((ref, sec))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def write_itab(path, pairs, flags=None) -> None:
    """Write a 0-based ``(n_pairs, 2)`` array back out as a 1-based itab."""
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    flags = np.ones(len(pairs), int) if flags is None else np.asarray(flags, int)
    with open(path, "w") as fh:
        for k, ((i, j), f) in enumerate(zip(pairs, flags), start=1):
            fh.write(f"{i + 1:5d}{j + 1:5d}{k:5d}{f:5d}\n")


# ------------------------------------------------------------------- network
class Network:
    """An interferogram network: epochs plus the pairs connecting them.

    >>> net = Network.from_gamma("SLCu_tab", "itab_mr")
    >>> net.n_epochs, net.n_pairs
    (723, 722)
    """

    def __init__(self, epochs, pairs, paths=None):
        self.epochs = list(epochs)
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self.paths = list(paths) if paths is not None else None
        if len(self.pairs) and self.pairs.max() >= len(self.epochs):
            raise ValueError("pair index beyond the number of epochs")

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_gamma(cls, slc_tab, itab, diff_tab=None, root=None) -> "Network":
        root = Path(root) if root else Path(slc_tab).parent
        images, _ = read_slc_tab(slc_tab)
        epochs = [parse_epoch(p) for p in images]
        pairs = read_itab(itab)
        paths = None
        if diff_tab is not None:
            diffs = read_diff_tab(diff_tab)
            paths = [str(root / p) for p in diffs]
        return cls(epochs, pairs, paths)

    # ------------------------------------------------------------- properties
    @property
    def n_epochs(self) -> int:
        return len(self.epochs)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def times(self) -> np.ndarray:
        """Epoch times in days relative to the first acquisition."""
        t0 = self.epochs[0]
        return np.array([(e - t0).total_seconds() / 86400.0 for e in self.epochs])

    def temporal_baselines(self) -> np.ndarray:
        """Pair time spans in days."""
        t = self.times
        return t[self.pairs[:, 1]] - t[self.pairs[:, 0]]

    # ---------------------------------------------------------------- topology
    def is_connected(self) -> bool:
        return len(self.components()) <= 1

    def components(self) -> list[list[int]]:
        """Connected components of the network, as sorted lists of epoch indices."""
        adj: dict[int, set[int]] = {i: set() for i in range(self.n_epochs)}
        for i, j in self.pairs:
            adj[i].add(j)
            adj[j].add(i)
        seen, out = set(), []
        for start in range(self.n_epochs):
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            out.append(sorted(comp))
        return out

    # ----------------------------------------------------------- design matrix
    def design_matrix(self, reference=0) -> np.ndarray:
        """SBAS design matrix ``G`` for displacement *relative to a reference epoch*.

        Solves ``G @ d = phi`` where ``d`` holds the ``n_epochs - 1`` unknown
        epoch displacements (the reference epoch is fixed at zero) and ``phi``
        holds one unwrapped value per pair.  Row for pair ``(i, j)`` is
        ``d_j - d_i``.
        """
        cols = [e for e in range(self.n_epochs) if e != reference]
        index = {e: k for k, e in enumerate(cols)}
        G = np.zeros((self.n_pairs, len(cols)))
        for r, (i, j) in enumerate(self.pairs):
            if i != reference:
                G[r, index[i]] = -1.0
            if j != reference:
                G[r, index[j]] = +1.0
        return G

    def incremental_design_matrix(self) -> np.ndarray:
        """Design matrix in terms of *increments* between consecutive epochs.

        Unknowns are the ``n_epochs - 1`` steps ``d_{k+1} - d_k``.  This is the
        classic SBAS parameterisation and is better conditioned than the
        reference-epoch form when the network is a daisy chain.
        """
        G = np.zeros((self.n_pairs, self.n_epochs - 1))
        for r, (i, j) in enumerate(self.pairs):
            lo, hi = (i, j) if i < j else (j, i)
            sign = 1.0 if i < j else -1.0
            G[r, lo:hi] = sign
        return G

    def __repr__(self) -> str:
        span = self.times[-1] if self.n_epochs else 0.0
        return (f"Network({self.n_epochs} epochs, {self.n_pairs} pairs, "
                f"{span:.3f} d, {'connected' if self.is_connected() else 'DISCONNECTED'})")
