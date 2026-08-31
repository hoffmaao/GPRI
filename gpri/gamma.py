"""Readers and writers for GAMMA binary products and parameter files.

GAMMA writes its raster products as flat binary in **big-endian** byte order
with no header, and describes them in a companion ``.par`` text file.  Byte
order was confirmed empirically against BakerBend1 SLCs: interpreting them
little-endian yields amplitudes of order 1e38, big-endian yields order 1.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0

#: GAMMA ``image_format`` -> numpy dtype.  All big-endian.
DTYPES = {
    "FCOMPLEX": np.dtype(">c8"),    # pairs of float32
    "SCOMPLEX": np.dtype(">i2"),    # pairs of int16, handled specially
    "FLOAT": np.dtype(">f4"),
    "REAL*4": np.dtype(">f4"),
    "DOUBLE": np.dtype(">f8"),
    "REAL*8": np.dtype(">f8"),
    "SHORT": np.dtype(">i2"),
    "INTEGER*2": np.dtype(">i2"),
    "INT": np.dtype(">i4"),
    "INTEGER*4": np.dtype(">i4"),
    "BYTE": np.dtype("u1"),
}

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


class ParFile:
    """A parsed GAMMA parameter file.

    Values are kept as the raw whitespace-separated token list so that unit
    suffixes (``s``, ``m``, ``degrees``, ``Hz``) and multi-value entries such as
    ``GPRI_tx_coord`` survive intact.  Use the typed accessors to pull numbers.

    >>> par = ParFile.load("20170803_222136u.slc.par")
    >>> par.range_samples, par.azimuth_lines
    (22101, 396)
    >>> round(par.wavelength, 6)
    0.017430
    """

    def __init__(self, entries: dict[str, list[str]], header: str = ""):
        self.entries = entries
        self.header = header

    # ------------------------------------------------------------ construction
    @classmethod
    def loads(cls, text: str) -> "ParFile":
        entries: dict[str, list[str]] = {}
        header = ""
        for i, line in enumerate(text.splitlines()):
            line = line.rstrip()
            if not line.strip():
                continue
            if ":" not in line:
                # the first non-empty line is the format banner
                if i == 0 or not header:
                    header = line.strip()
                continue
            key, _, rest = line.partition(":")   # split on the FIRST colon only,
            key = key.strip()                    # `title` values contain colons
            if key:
                entries[key] = rest.split()
        return cls(entries, header)

    @classmethod
    def load(cls, path) -> "ParFile":
        return cls.loads(Path(path).read_text(errors="replace"))

    # --------------------------------------------------------------- accessors
    def __contains__(self, key) -> bool:
        return key in self.entries

    def __getitem__(self, key) -> list[str]:
        return self.entries[key]

    def tokens(self, key, default=None):
        return self.entries.get(key, default)

    def str(self, key, default=None):
        v = self.entries.get(key)
        return " ".join(v) if v else default

    def float(self, key, default=None, index=0) -> float:
        v = self.entries.get(key)
        if not v or index >= len(v):
            if default is None:
                raise KeyError(f"{key!r} not in parameter file")
            return default
        # GAMMA occasionally emits Fortran 'D' exponents
        return float(v[index].replace("D", "E").replace("d", "e"))

    def int(self, key, default=None, index=0) -> int:
        v = self.entries.get(key)
        if not v or index >= len(v):
            if default is None:
                raise KeyError(f"{key!r} not in parameter file")
            return default
        return int(float(v[index]))

    def floats(self, key) -> np.ndarray:
        """All numeric tokens for a key, discarding trailing unit words."""
        return np.array(
            [float(t.replace("D", "E")) for t in self.entries.get(key, []) if _NUM.match(t)]
        )

    # -------------------------------------------------------- common shorthand
    @property
    def range_samples(self) -> int:
        for k in ("range_samples", "interferogram_width", "width", "range_samp_1"):
            if k in self.entries:
                return self.int(k)
        raise KeyError("no width key in parameter file")

    @property
    def azimuth_lines(self) -> int:
        for k in ("azimuth_lines", "interferogram_azimuth_lines", "nlines", "az_samp_1"):
            if k in self.entries:
                return self.int(k)
        raise KeyError("no height key in parameter file")

    @property
    def shape(self) -> tuple[int, int]:
        """``(azimuth_lines, range_samples)`` — numpy row-major order."""
        return (self.azimuth_lines, self.range_samples)

    @property
    def image_format(self) -> str:
        return (self.str("image_format") or "FCOMPLEX").upper()

    @property
    def radar_frequency(self) -> float:
        return self.float("radar_frequency")

    @property
    def wavelength(self) -> float:
        """Radar wavelength in metres.  GPRI-II is Ku band, lambda ~ 1.74 cm."""
        return SPEED_OF_LIGHT / self.radar_frequency

    @property
    def date(self) -> str:
        """Acquisition date as ``YYYYMMDD``."""
        y, m, d = (self.int("date", index=i) for i in range(3))
        return f"{y:04d}{m:02d}{d:02d}"

    @property
    def near_range(self) -> float:
        return self.float("near_range_slc")

    @property
    def range_pixel_spacing(self) -> float:
        return self.float("range_pixel_spacing")

    def slant_range(self) -> np.ndarray:
        """Slant range (m) at each range sample."""
        return self.near_range + self.range_pixel_spacing * np.arange(self.range_samples)

    def __repr__(self) -> str:
        return (f"ParFile({self.str('sensor')!r}, {self.range_samples}x"
                f"{self.azimuth_lines}, {self.image_format})")


# ------------------------------------------------------------------- raster IO
def dtype_for(image_format: str) -> np.dtype:
    fmt = image_format.upper()
    if fmt not in DTYPES:
        raise ValueError(f"unsupported GAMMA image_format {image_format!r}")
    return DTYPES[fmt]


def read_image(path, par=None, shape=None, image_format=None) -> np.ndarray:
    """Read a GAMMA binary raster into a ``(azimuth_lines, range_samples)`` array.

    Supply either ``par`` (a :class:`ParFile` or path to one) or both ``shape``
    and ``image_format``.  If neither is given, ``<path>.par`` is tried.
    """
    path = Path(path)
    if par is None and shape is None:
        cand = Path(str(path) + ".par")
        if cand.exists():
            par = cand
    if par is not None:
        if not isinstance(par, ParFile):
            par = ParFile.load(par)
        shape = shape or par.shape
        image_format = image_format or par.image_format
    if shape is None or image_format is None:
        raise ValueError("need a parameter file, or explicit shape and image_format")

    fmt = image_format.upper()
    if fmt == "SCOMPLEX":
        raw = np.fromfile(path, dtype=DTYPES["SCOMPLEX"])
        data = raw.astype(np.float32).view(np.complex64)  # interleaved re/im
    else:
        data = np.fromfile(path, dtype=dtype_for(fmt))

    expected = int(shape[0]) * int(shape[1])
    if data.size != expected:
        raise ValueError(
            f"{path.name}: got {data.size} samples, parameter file implies "
            f"{shape[0]}x{shape[1]}={expected}"
        )
    return data.reshape(shape)


def write_image(path, array, image_format=None) -> None:
    """Write a 2-D array as a GAMMA big-endian binary raster."""
    array = np.asarray(array)
    if image_format is None:
        image_format = "FCOMPLEX" if np.iscomplexobj(array) else "FLOAT"
    array.astype(dtype_for(image_format)).tofile(str(path))


def read_slc(path, par=None) -> np.ndarray:
    """Read an SLC, always returning complex64."""
    return read_image(path, par=par).astype(np.complex64)


def azimuth_angles(par) -> np.ndarray:
    """Azimuth look angle (degrees) at each azimuth line.

    The GPRI-II sweeps its antenna in azimuth rather than flying, so the
    ``azimuth_lines`` axis is an *angle* axis, described by
    ``GPRI_az_start_angle`` and ``GPRI_az_angle_step`` rather than by a pixel
    spacing (``azimuth_pixel_spacing`` is 0 in these files).  BakerBend1 sweeps
    -27.955 deg to +51.246 deg in 396 steps of 0.2 deg.
    """
    if not isinstance(par, ParFile):
        par = ParFile.load(par)
    n = par.azimuth_lines
    start = par.float("GPRI_az_start_angle", 0.0)
    step = par.float("GPRI_az_angle_step", 0.0)
    if step == 0.0:
        # not a GPRI file - fall back to a plain line index
        return np.arange(n, dtype=float)
    return start + step * np.arange(n, dtype=float)


def ground_range(par) -> np.ndarray:
    """Horizontal ground range (m) per range sample, for a fixed look elevation.

    The GPRI antenna elevation is a single number per acquisition
    (``GPRI_ant_elev_angle``), so ground range is just the slant range
    foreshortened by its cosine.  Good enough for plotting and for the
    range-dependent atmospheric term; not a substitute for real geocoding.
    """
    if not isinstance(par, ParFile):
        par = ParFile.load(par)
    elev = np.deg2rad(par.float("GPRI_ant_elev_angle", 0.0))
    return par.slant_range() * np.cos(elev)


def map_image(path, par=None, shape=None, image_format=None, mode="r"):
    """Memory-map a GAMMA raster instead of reading it.

    A single BakerBend1 interferogram is 70 MB and there are 723 of them; the
    whole stack is 50 GB, well past what fits in memory.  Mapping lets
    :class:`gpri.stack.DiffStack` walk the stack patch by patch and only ever
    touch the rows it needs.

    ``SCOMPLEX`` cannot be mapped as complex (it is interleaved int16 and needs
    a conversion), so it is rejected here — use :func:`read_image` for those.
    """
    path = Path(path)
    if par is None and shape is None:
        cand = Path(str(path) + ".par")
        if cand.exists():
            par = cand
    if par is not None:
        if not isinstance(par, ParFile):
            par = ParFile.load(par)
        shape = shape or par.shape
        image_format = image_format or par.image_format
    if shape is None or image_format is None:
        raise ValueError("need a parameter file, or explicit shape and image_format")

    fmt = image_format.upper()
    if fmt == "SCOMPLEX":
        raise ValueError("SCOMPLEX needs conversion on read; use read_image()")
    dtype = dtype_for(fmt)
    expected = int(shape[0]) * int(shape[1]) * dtype.itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path.name}: file is {actual} bytes, parameter file implies "
            f"{shape[0]}x{shape[1]}x{dtype.itemsize}={expected}"
        )
    return np.memmap(str(path), dtype=dtype, mode=mode, shape=tuple(int(s) for s in shape))


def read_par(path) -> ParFile:
    """Alias for :meth:`ParFile.load`, for symmetry with :func:`read_image`."""
    return ParFile.load(path)
