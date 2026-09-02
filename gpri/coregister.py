"""Azimuth co-registration: did the tripod hold its heading all campaign?

:class:`gpri.stack.SlcPairStack` assumes what a tripod promises — that every
scan of a campaign looks the same way, so line ``l`` of one SLC and line
``l`` of the next see the same ground.  Usually it does: 20170713_full,
20170827 and 20170913 hold their heading to 0.03 deg over a day or two.
The 2018 campaign does not.  Its mount turned 5.2 deg anticlockwise (26
azimuth lines, about 1 deg/h) over the first 4.8 h and then held for the
remaining 2.1 h.

A 0.15-line slide between consecutive 2-min acquisitions costs a lag-1
interferogram nothing measurable (coherence p90 0.510 unaligned, 0.512
aligned); anything longer does — a 20-min pair drops from 0.51 to 0.40 and
comes all the way back once the shift is taken out.  What suffers without
correction is the *grid*: a pixel's cumulative series over the drifting
hours is a walk across 1.6 decimated cells, and the geocoded mask is
rotated by up to 5 deg from the ground it claims to describe.

The remedy is to co-register in azimuth on read.  The offset of every SLC
against a reference is measured from the image texture alone — the
high-passed dB intensity, cross-correlated along azimuth with a parabolic
refinement of the peak (:func:`azimuth_offset`) — and each SLC is shifted
by that many lines with a Fourier phase ramp (:func:`shift_azimuth`), the
exact interpolator for a band-limited signal.  It preserves the
interferometric phase: a scatterer's range does not depend on which line
the antenna caught it on.  ``gpri coregister <scene> --write`` records the
offsets as ``azimuth_offsets.json`` beside the scene's other sidecars and
:meth:`gpri.stack.SlcPairStack.apply_azimuth_offsets` takes them up.  The
scan heading is then measured once, on the reference (``gpri heading``
shifts what it averages), and holds for every epoch.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .gamma import ParFile, read_slc
from .network import scene_id

__all__ = [
    "AZIMUTH_FILE", "texture", "azimuth_offset", "shift_azimuth",
    "AzimuthOffsets", "campaign_offsets", "acquisition_id",
    "write_azimuth_offsets", "scene_azimuth_offsets", "shifts_for",
]

#: Sidecar written beside a scene's caches by ``gpri coregister --write``.
AZIMUTH_FILE = "azimuth_offsets.json"


def acquisition_id(path) -> str:
    """``20180710_133506`` for either antenna's file of that acquisition."""
    return scene_id(path).rstrip("ul")


# ------------------------------------------------------------------ measuring
def texture(slc, range_looks=20, highpass=5.0) -> np.ndarray:
    """High-passed dB intensity: the pattern of the terrain, not its brightness.

    Range is averaged over ``range_looks`` samples first (speckle, and the
    array shrinks twenty-fold); ``highpass`` is the sigma in coarse pixels of
    the Gaussian mean taken out.  What is left is the ridge-and-shadow
    pattern that fixes an image in azimuth.
    """
    p = np.abs(slc).astype(np.float32) ** 2
    nr = (p.shape[1] // range_looks) * range_looks
    p = p[:, :nr].reshape(p.shape[0], -1, range_looks).mean(axis=2)
    db = 10.0 * np.log10(p + 1e-9)
    return (db - gaussian_filter(db, highpass)).astype(np.float32)


def azimuth_offset(tex, ref, search=40, columns=slice(None)):
    """Lines by which ``ref``'s features lie *later* in ``tex``.

    Returns ``(offset, corr)`` with ``ref[l] ~ tex[l + offset]``, so
    :func:`shift_azimuth` by ``offset`` aligns ``tex``'s image to the
    reference.  Integer shifts up to ``search`` are tried on the columns
    selected, the peak of the normalised correlation is refined with a
    parabola through its neighbours, and the peak's correlation is returned
    with it (0.7 or so for two GPRI images of the same terrain; well under
    0.3 means the two do not overlap and the offset is noise).
    """
    n = min(tex.shape[0], ref.shape[0])
    a, b = tex[:n, columns], ref[:n, columns]
    shifts = np.arange(-int(search), int(search) + 1)
    corr = np.full(shifts.size, -1.0)
    for k, s in enumerate(shifts):
        x = a[max(0, s):n + min(0, s)]
        y = b[max(0, -s):n + min(0, -s)]
        if x.shape[0] < 8:
            continue
        corr[k] = np.corrcoef(x.ravel(), y.ravel())[0, 1]
    k = int(np.argmax(corr))
    d = float(shifts[k])
    if 0 < k < shifts.size - 1:
        y0, y1, y2 = corr[k - 1:k + 2]
        curv = y0 - 2.0 * y1 + y2
        if curv < 0:
            d += 0.5 * (y0 - y2) / curv
    return d, float(corr[k])


def shift_azimuth(image, lines):
    """``out[l] = image[l + lines]`` for a fractional number of ``lines``.

    A Fourier phase ramp along axis 0 — exact for the band-limited azimuth
    response of a scanning antenna sampled at half its beamwidth, and phase
    preserving, so interferograms can be formed from shifted SLCs.  The
    lines that would come around from the other end are zeroed instead, so
    they drop out of any coherence-weighted estimate — those more than a
    quarter wrapped, that is; a held tripod's 0.03-line jitter costs
    nothing.
    """
    lines = float(lines)
    if lines == 0.0:
        return image
    n = image.shape[0]
    ramp = np.exp(2j * np.pi * np.fft.fftfreq(n) * lines)
    ramp = ramp.reshape((n,) + (1,) * (image.ndim - 1))
    out = np.fft.ifft(np.fft.fft(image, axis=0) * ramp, axis=0)
    if not np.iscomplexobj(image):
        out = out.real
    m = int(np.ceil(abs(lines) - 0.25))
    if m > 0 and lines > 0:
        out[n - m:] = 0
    elif m > 0:
        out[:m] = 0
    return out.astype(image.dtype, copy=False)


# ------------------------------------------------------------------ a campaign
@dataclass
class AzimuthOffsets:
    """Every SLC's offset against the reference, ``ref[l] ~ slc[l + offset]``."""
    ids: list                   # acquisition ids, in the order given
    offsets: np.ndarray         # lines
    corr: np.ndarray            # peak correlation of each measurement
    reference: str              # acquisition id the others are aligned to
    step: float = float("nan")  # GPRI_az_angle_step, deg/line, if known

    @property
    def span(self) -> float:
        """Peak-to-peak offset in lines: 0 for a tripod that held."""
        return float(np.ptp(self.offsets))

    @property
    def span_deg(self) -> float:
        return self.span * self.step

    def as_dict(self) -> dict:
        return {
            "reference": self.reference,
            "step_deg": self.step,
            "span_lines": round(self.span, 3),
            "offsets": {i: round(float(o), 3) for i, o in zip(self.ids, self.offsets)},
            "corr": {i: round(float(c), 3) for i, c in zip(self.ids, self.corr)},
        }


def campaign_offsets(images, reference=-1, search=40, range_looks=20,
                     ranges=(1500.0, 12000.0), progress=None) -> AzimuthOffsets:
    """Measure every SLC of a campaign against one of them.

    ``reference`` is an index into ``images`` (the last by default: a mount
    that creeps settles, so the end of a campaign is its steadiest part) or
    an acquisition id.  ``ranges`` bounds the slant ranges (m) correlated —
    inside the near-field clutter and beyond the far edge of the terrain
    there is no texture to match.
    """
    images = [Path(p) for p in images]
    ids = [acquisition_id(p) for p in images]
    if isinstance(reference, str):
        reference = ids.index(acquisition_id(reference))
    reference = range(len(images))[reference]          # -1 -> a real index
    par = ParFile.load(str(images[reference]) + ".par")
    r0, dr = par.near_range, par.range_pixel_spacing * range_looks
    c0 = max(0, int((ranges[0] - r0) / dr))
    c1 = int((ranges[1] - r0) / dr)
    cols = slice(c0, c1)
    ref = texture(read_slc(images[reference]), range_looks)
    offsets = np.zeros(len(images))
    corr = np.zeros(len(images))
    for k, im in enumerate(images):
        if k == reference:
            corr[k] = 1.0
            continue
        offsets[k], corr[k] = azimuth_offset(texture(read_slc(im), range_looks),
                                             ref, search=search, columns=cols)
        if progress is not None:
            progress(k, len(images), ids[k], offsets[k], corr[k])
    return AzimuthOffsets(ids, offsets, corr, ids[reference],
                          par.float("GPRI_az_angle_step", float("nan")))


# ------------------------------------------------------------------ sidecar
def _work_dir(scene):
    return Path(os.environ.get("GPRI_WORK_ROOT", "work")) / Path(scene).name


def write_azimuth_offsets(scene, result: AzimuthOffsets, extra=None) -> Path:
    """Record the offsets as ``azimuth_offsets.json`` under the work dir."""
    d = _work_dir(scene)
    d.mkdir(parents=True, exist_ok=True)
    rec = {**result.as_dict(), "method": "texture", **(extra or {})}
    (d / AZIMUTH_FILE).write_text(json.dumps(rec, indent=1) + "\n")
    return d / AZIMUTH_FILE


def scene_azimuth_offsets(scene):
    """``{acquisition id: lines}`` recorded for a scene, or ``None``.

    ``None`` is the common case — a tripod that held needs no sidecar — and
    every consumer should treat it as "no shift".
    """
    f = _work_dir(scene) / AZIMUTH_FILE
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    return {k: float(v) for k, v in rec["offsets"].items()}


def shifts_for(images, offsets) -> np.ndarray:
    """Per-image shifts from an ``{id: lines}`` table, in the images' order.

    An image the table does not know gets no shift and a warning: the
    sidecar was measured on a different SLC list and should be remade.
    """
    out = np.zeros(len(images))
    missing = []
    for k, im in enumerate(images):
        i = acquisition_id(im)
        if i in offsets:
            out[k] = offsets[i]
        else:
            missing.append(i)
    if missing:
        warnings.warn(f"{len(missing)} SLC(s) without a recorded azimuth offset "
                      f"(first {missing[0]}): left unshifted -- rerun "
                      f"`gpri coregister --write`", stacklevel=2)
    return out
