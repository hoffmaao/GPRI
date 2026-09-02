"""Focus GPRI-II raw FMCW sweeps into single-look complex images.

This is a port of GAMMA's ``gpri2_proc.py`` (v2.7, 29-May-2014, shipped in
the ``GPRI2-2`` tree of the GAMMA distribution), the program the radar's
acquisition computer runs after every scan to turn the recorded de-ramped
sweeps into the ``<scene>l.slc`` / ``<scene>u.slc`` pair that everything else
in this package consumes.  The processing chain is kept step for step:

1.  echoes are read in blocks of ``dec`` consecutive sweeps and averaged
    (azimuth decimation, which is also the azimuth presum);
2.  the first and last ``zero`` samples of every sweep are tapered with a
    Hann window to suppress the transient at the sawtooth turnaround;
3.  each frequency bin is shifted in azimuth by the antenna's frequency
    dependent squint (a slotted-waveguide array steers with frequency), by
    linear interpolation;
4.  a Kaiser-windowed real FFT along the sweep turns beat frequency into
    slant range; the spectrum is conjugated, the alternating-sign carrier
    removed and an ``r**1.5`` range weighting applied;
5.  lines acquired during the scanner's acceleration and deceleration ramps
    are dropped.

Channel 1 is the **lower** receive antenna and channel 2 the **upper** one,
which is the naming GAMMA's own parameter files carry (``CH1 lower``).  The
SLC parameter files are written with the same keywords, formats and values
as GAMMA writes, so the products are interchangeable with a GAMMA-processed
campaign: on the 2017-08-03 BakerBend campaign, for which both the raw
sweeps and GAMMA's SLCs are archived, the port reproduces GAMMA's images to
single-precision rounding.

Nothing here needs the GAMMA binaries; it is numpy only.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gamma import ParFile

C = 299792458.0            # speed of light, m/s
RANGE_OFFSET = -3.0        # effective two-way range offset of the cable and filter delays, m
RA = 6378137.0000          # WGS-84 semi-major axis
RB = 6356752.3141          # WGS-84 semi-minor axis
KU_WIDTH = 15.798e-3       # WG-62 Ku-band waveguide width, m
KU_DZ = 10.682e-3          # Ku-band waveguide slot spacing, m


# ------------------------------------------------------------------ raw_par
@dataclass
class RawPar:
    """The ``.raw_par`` written beside every GPRI-II raw acquisition."""
    time_start: str                 # "YYYY-MM-DD HH:MM:SS.ffffff+00:00"
    lat: float
    lon: float
    alt: float
    geoid: float
    RF_center_freq: float
    RF_freq_min: float
    RF_freq_max: float
    RF_chirp_rate: float
    ns: int                         # samples per sweep (CHP_num_samp)
    TX_mode: str
    atten_dB: int
    capture_time: float
    sample_rate: float
    antenna_start: float
    antenna_end: float
    rotation_speed: float
    gear_ratio: int
    antenna_elev: float
    TSC_version: str = "None"
    TSC_acc_ramp_angle: float = 0.0
    TSC_acc_ramp_time: float = 0.0
    TSC_rotation_speed: float = 0.0
    TSC_acc_ramp_step: float = 0.0

    @classmethod
    def load(cls, path) -> "RawPar":
        p = ParFile.load(path)
        t = p.tokens("time_start")
        geo = p.floats("geographic_coordinates")
        tsc = p.tokens("TSC_version")
        kw = {}
        if "TSC_acc_ramp_angle" in p:
            kw = dict(TSC_acc_ramp_angle=p.float("TSC_acc_ramp_angle"),
                      TSC_acc_ramp_time=p.float("TSC_acc_ramp_time"),
                      TSC_rotation_speed=p.float("TSC_rotation_speed"),
                      TSC_acc_ramp_step=p.float("TSC_acc_ramp_step"))
        return cls(
            time_start=" ".join(t[:2]),
            lat=geo[0], lon=geo[1], alt=geo[2],
            geoid=geo[3] if len(geo) > 3 else 0.0,
            RF_center_freq=p.float("RF_center_freq"),
            RF_freq_min=p.float("RF_freq_min"),
            RF_freq_max=p.float("RF_freq_max"),
            RF_chirp_rate=p.float("RF_chirp_rate"),
            ns=p.int("CHP_num_samp"),
            TX_mode=p.str("TX_mode", "None"),
            atten_dB=p.int("IMA_atten_dB"),
            capture_time=p.float("ADC_capture_time"),
            sample_rate=p.float("ADC_sample_rate"),
            antenna_start=p.float("STP_antenna_start"),
            antenna_end=p.float("STP_antenna_end"),
            rotation_speed=p.float("STP_rotation_speed"),
            gear_ratio=p.int("STP_gear_ratio"),
            antenna_elev=p.float("antenna_elevation"),
            TSC_version=tsc[1] if tsc and len(tsc) > 1 else "None",
            **kw)


@dataclass
class FocusOptions:
    """The ``gpri2_proc.py`` command-line options.

    The BakerBend campaigns were processed with
    ``-d 5 -z 300 -r 300 -R 0 -k 3.84 -h 0``, which :func:`baker_options`
    returns.
    """
    dec: int = 1            # azimuth decimation (presum) factor
    zero: int = 300         # samples tapered at the start and end of each sweep
    rmin: float = 50.0      # minimum slant range, m
    rmax: float = 0.0       # maximum slant range, m; 0 = 0.9 of the aliasing range
    kbeta: float = 3.0      # Kaiser window beta
    heading: float = 0.0    # boresight heading at scan centre, degrees clockwise from north
    ati: bool = False       # along-track interferometry: no squint interpolation
    tx_antenna: str = "V"
    datatype: str = "int16"


def baker_options(**overrides) -> FocusOptions:
    """The options GAMMA's capture script used for the BakerBend campaigns."""
    o = dict(dec=5, zero=300, rmin=300.0, rmax=0.0, kbeta=3.84, heading=0.0)
    o.update(overrides)
    return FocusOptions(**o)


# ------------------------------------------------------------------ geometry
@dataclass
class FocusGeometry:
    """Everything ``gpri2_proc.py`` derives before touching the samples."""
    raw: RawPar
    opts: FocusOptions
    nl_tot: int                              # sweeps in the file
    nsamp: int = field(init=False)
    block_length: int = field(init=False)
    bytes_per_record: int = field(init=False)
    tcycle: float = field(init=False)
    ang_per_tcycle: float = field(init=False)
    t_acc: float = field(init=False)
    ang_acc: float = field(init=False)
    rate_max: float = field(init=False)
    nl_acc: int = field(init=False)
    nl_tot_dec: int = field(init=False)
    nl_image: int = field(init=False)
    image_time: float = field(init=False)
    rps: float = field(init=False)
    ns_min: int = field(init=False)
    ns_max: int = field(init=False)
    ns_out: int = field(init=False)
    rmin: float = field(init=False)
    rmax: float = field(init=False)
    sqfc: float = field(init=False)
    sq_lin: np.ndarray = field(init=False, repr=False)
    win: np.ndarray = field(init=False, repr=False)
    win2: np.ndarray = field(init=False, repr=False)
    scale: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        g, o = self.raw, self.opts
        self.nsamp = g.ns
        self.block_length = self.nsamp + 1      # one extra sample: the jump back to the start frequency
        itemsize = np.dtype(o.datatype).itemsize
        self.bytes_per_record = 2 * itemsize * self.block_length
        self.win = np.kaiser(self.nsamp, o.kbeta)
        self.win2 = np.hanning(2 * o.zero)
        pn1 = np.arange(self.nsamp // 2 + 1)
        self.rps = (g.sample_rate / self.nsamp * C / 2.0) / g.RF_chirp_rate
        slr = (pn1 * g.sample_rate / self.nsamp * C / 2.0) / g.RF_chirp_rate + RANGE_OFFSET
        self.scale = (np.abs(slr) / slr[self.nsamp // 8]) ** 1.5
        self.ns_min = int(round(o.rmin / self.rps))
        self.rmin = self.ns_min * self.rps
        self.ns_max = int(round(0.90 * self.nsamp / 2))
        self.rmax = self.ns_max * self.rps
        if o.rmax != 0.0:
            if int(round(o.rmax / self.rps)) <= self.ns_max:
                self.ns_max = int(round(o.rmax / self.rps))
                self.rmax = self.ns_max * self.rps
            else:
                raise ValueError(f"rmax {o.rmax} m exceeds the {self.rmax:.3f} m "
                                 "this chirp allows")
        self.ns_out = self.ns_max - self.ns_min + 1

        if g.TX_mode == "HV":
            self.tcycle = 2 * self.block_length / g.sample_rate
        else:
            self.tcycle = self.block_length / g.sample_rate

        if g.antenna_end != g.antenna_start:
            self.ang_acc = g.TSC_acc_ramp_angle
            self.rate_max = g.TSC_rotation_speed
            self.t_acc = g.TSC_acc_ramp_time
            if g.antenna_end < g.antenna_start:
                self.rate_max = -self.rate_max
            self.ang_per_tcycle = self.tcycle * self.rate_max
        else:
            self.t_acc = self.ang_acc = self.rate_max = self.ang_per_tcycle = 0.0

        capture = g.capture_time
        if capture == 0.0:
            angc = abs(g.antenna_end - g.antenna_start) - 2 * self.ang_acc
            capture = 2 * self.t_acc + abs(angc / self.rate_max)
        self.nl_acc = int(self.t_acc / (self.tcycle * o.dec))
        self.nl_tot_dec = int(capture / (self.tcycle * o.dec))
        self.nl_image = self.nl_tot_dec - 2 * self.nl_acc
        self.image_time = (self.nl_image - 1) * (self.tcycle * o.dec)

        # frequency-dependent beam squint of the Ku-band slotted waveguide
        freq = g.RF_freq_min + np.arange(self.nsamp, dtype=float) * g.RF_chirp_rate / g.sample_rate
        self.sqfc = 0.0
        if g.RF_freq_min > 17.0e9:
            lamg = (C / freq) / np.sqrt(1.0 - (C / (2 * KU_WIDTH * freq)) ** 2)
            dphi = math.pi * (2.0 * KU_DZ / lamg - 1.0)
            sq_ang = 180.0 / math.pi * np.arcsin(C / freq * dphi / (2.0 * math.pi * KU_DZ))
            if self.ang_per_tcycle != 0.0:
                sq_lin = sq_ang / (self.ang_per_tcycle * o.dec)
            else:
                sq_lin = np.zeros(freq.shape)
            mid = (self.nsamp - 1) // 2
            self.sqfc = sq_lin[mid] * (self.ang_per_tcycle * o.dec)
            self.sq_lin = sq_lin - sq_lin[mid]
        else:
            self.sq_lin = np.zeros(freq.shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nl_image, self.ns_out

    # ------------------------------------------------------------- SLC par
    def slc_par(self, channel: int) -> str:
        """The ISP parameter file for channel 1 (lower) or 2 (upper)."""
        g, o = self.raw, self.opts
        ant_elev = math.radians(g.antenna_elev)
        xoff = 0.112           # X offset to the antenna holder rotation axis
        ant_radius = 0.1115    # rotation radius of the antenna holder
        rx2_dz = 0.250         # rx2 is 25 cm below rx1
        tx_dz = -0.350         # the transmit antenna is 35 cm above rx1
        x = xoff + ant_radius * math.cos(ant_elev)
        z = -ant_radius * math.sin(ant_elev)
        rx1 = (x, 0.0, z)
        rx2 = (x, 0.0, rx2_dz + z)
        tx = (x, 0.0, tx_dz + z)

        ts = g.time_start
        ymd = ts.split()[0].split("-")
        clock = ts.split()[1]
        for sep in "+-":
            if sep in clock:
                clock = clock.split(sep)[0]
        hms = clock.split(":")
        sod = int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])
        st0 = sod + self.nl_acc * self.tcycle * o.dec + (o.dec / 2.0) * self.tcycle
        az_step = self.ang_per_tcycle * o.dec
        prf = abs(1.0 / (self.tcycle * o.dec))
        if g.antenna_end > g.antenna_start:
            az_start = g.antenna_start + self.sqfc + self.ang_acc
        else:
            az_start = g.antenna_start + self.sqfc - self.ang_acc
        fadc = C / (2.0 * self.rps)
        cbw = g.RF_freq_max - g.RF_freq_min
        label = "CH1 lower" if channel == 1 else "CH2 upper"

        p = [
            "Gamma Interferometric SAR Processor (ISP) - Image Parameter File",
            f"title: {ts} {label}",
            "sensor: GPRI 2.0",
            f"date:  {ymd[0]} {ymd[1]} {ymd[2]}",
            "start_time:     %12.6f  s" % st0,
            "center_time:    %12.6f  s" % (st0 + self.image_time / 2.0),
            "end_time:       %12.6f  s" % (st0 + self.image_time),
            "azimuth_line_time: %e   s" % (1.0 / prf),
            "line_header_size:      0",
            "range_samples:    %d" % self.ns_out,
            "azimuth_lines:    %d" % self.nl_image,
            "range_looks:           1",
            "azimuth_looks:         1",
            "image_format:          FCOMPLEX",
            "image_geometry:        SLANT_RANGE",
            "range_scale_factor:    1.0",
            "azimuth_scale_factor:  1.0",
            "center_latitude:       %.8f  degrees" % 0.0,
            "center_longitude:      %.8f  degrees" % 0.0,
            "heading:               %f  degrees" % 0.0,
            "range_pixel_spacing:   %f  m" % self.rps,
            "azimuth_pixel_spacing: %f  m" % 0.0,
            "near_range_slc:        %f  m" % self.rmin,
            "center_range_slc:  %f  m" % ((self.rmin + self.rmax) / 2.0),
            "far_range_slc:     %f  m" % self.rmax,
            "first_slant_range_polynomial:  0.0 0.0 0.0 0.0 0.0 0.0",
            "center_slant_range_polynomial: 0.0 0.0 0.0 0.0 0.0 0.0",
            "last_slant_range_polynomial:   0.0 0.0 0.0 0.0 0.0 0.0",
            "incidence_angle:       0.0  degrees",
            "azimuth_deskew:        OFF",
            "azimuth_angle:         0.0 degrees",
            "radar_frequency:       %e  Hz" % g.RF_center_freq,
            "adc_sampling_rate:     %e  Hz" % fadc,
            "chirp_bandwidth:       %e  Hz" % cbw,
            "prf:                   %f  Hz" % prf,
            "azimuth_proc_bandwidth: 0.0  Hz",
            "doppler_polynomial:     0.0 0.0 0.0 0.0",
            "doppler_poly_dot:       0.0 0.0 0.0 0.0",
            "doppler_poly_ddot:      0.0 0.0 0.0 0.0",
            "receiver_gain:        %8.3f  dB" % (60 - g.atten_dB),
            "calibration_gain:     %8.3f  dB" % 0.0,
            "sar_to_earth_center:       %12.4f  m" % 0.0,
            "earth_radius_below_sensor: %12.4f  m" % 0.0,
            "earth_semi_major_axis:     %12.4f  m" % RA,
            "earth_semi_minor_axis:     %12.4f  m" % RB,
            "number_of_state_vectors:   0",
            f"GPRI_TX_mode: {g.TX_mode}",
            f"GPRI_TX_antenna: {o.tx_antenna}",
            "GPRI_az_start_angle:  %12.6f  degrees" % az_start,
            "GPRI_az_angle_step:   %e  degrees" % az_step,
            "GPRI_ant_elev_angle:  %12.6f  degrees" % g.antenna_elev,
            "GPRI_ref_north:   %14.8f" % g.lat,
            "GPRI_ref_east:    %14.8f" % g.lon,
            "GPRI_ref_alt:     %14.5f  m" % g.alt,
            "GPRI_geoid:       %14.5f  m" % g.geoid,
            "GPRI_scan_heading: %12.5f  degrees" % o.heading,
            "GPRI_tx_coord:  %10.5f %10.5f %10.5f  m m m" % tx,
            "GPRI_rx1_coord: %10.5f %10.5f %10.5f  m m m" % rx1,
            "GPRI_rx2_coord: %10.5f %10.5f %10.5f  m m m" % rx2,
            "GPRI_tower_roll:   %.5f  degrees" % 0.0,
            "GPRI_tower_pitch:  %.5f  degrees" % 0.0,
            "GPRI_phase_offset: %.5f  radians\n" % 0.0,
        ]
        return "\n".join(p) + "\n"


def geometry(raw_par, raw_data=None, opts: FocusOptions | None = None,
             nl_tot: int | None = None) -> FocusGeometry:
    """Processing geometry for one acquisition.

    ``nl_tot`` (the number of sweeps in the file) is taken from the size of
    ``raw_data`` when not given.
    """
    rp = raw_par if isinstance(raw_par, RawPar) else RawPar.load(raw_par)
    opts = opts or FocusOptions()
    if nl_tot is None:
        itemsize = np.dtype(opts.datatype).itemsize
        nl_tot = os.path.getsize(raw_data) // (2 * itemsize * (rp.ns + 1))
    return FocusGeometry(rp, opts, nl_tot)


# -------------------------------------------------------------- processing
def read_decimated(geom: FocusGeometry, raw_data) -> tuple[np.ndarray, np.ndarray]:
    """Read the sweeps, taper the turnaround, presum by ``dec``.

    Returns the two receive channels as ``(nl_tot_dec, nsamp)`` float32
    arrays: channel 1 (lower antenna) and channel 2 (upper antenna).
    """
    g, o = geom.raw, geom.opts
    dt = np.dtype(o.datatype)
    n_rec = geom.nl_tot_dec * o.dec
    if g.TX_mode == "HV":
        raise NotImplementedError("alternating-transmit (HV) raw data")
    if n_rec > geom.nl_tot:
        raise ValueError(f"{raw_data}: {geom.nl_tot} sweeps in the file, "
                         f"{n_rec} expected from the capture time")
    with open(raw_data, "rb") as f:
        din = np.fromfile(f, dtype=dt, count=n_rec * geom.block_length * 2)
    din = din.reshape(geom.nl_tot_dec, o.dec, geom.block_length, 2)

    out = []
    for ch in range(2):
        x = din[..., ch].astype(np.float32)               # (nl_tot_dec, dec, block_length)
        if o.zero > 0:
            z = o.zero
            x[..., :z] = (x[..., :z] * geom.win2[:z]).astype(np.float32)
            x[..., -z:] = (x[..., -z:] * geom.win2[z:]).astype(np.float32)
        x = x[..., 1:]                                    # drop the flyback sample
        if dt == np.int16:
            x = x / np.float32(32768.0)
        out.append((x.sum(axis=1, dtype=np.float64) / o.dec).astype(np.float32))
    return out[0], out[1]


def squint_correct(geom: FocusGeometry, ch: np.ndarray) -> None:
    """Shift every frequency bin in azimuth by the antenna squint, in place."""
    if geom.opts.ati or geom.raw.RF_freq_min <= 17.0e9:
        return
    azp0 = np.arange(geom.nl_tot_dec, dtype=float)
    for i in range(geom.nsamp):
        ch[:, i] = np.interp(azp0 + geom.sq_lin[i], azp0, ch[:, i],
                             left=0.0, right=0.0)


def range_compress(geom: FocusGeometry, ch: np.ndarray) -> np.ndarray:
    """Kaiser-windowed FFT along the sweep -> ``(nl_image, ns_out)`` complex64."""
    fshift = np.ones(geom.nsamp // 2 + 1)
    fshift[1::2] = -1
    rows = ch[geom.nl_acc:geom.nl_acc + geom.nl_image]
    spec = np.fft.rfft(rows * geom.win, axis=1)
    sl = slice(geom.ns_min, geom.ns_max + 1)
    return (spec[:, sl].conj() * (fshift * geom.scale)[sl]).astype(np.complex64)


def focus_channels(geom: FocusGeometry, raw_data) -> tuple[np.ndarray, np.ndarray]:
    """Focus one raw file: ``(lower, upper)`` complex64 images."""
    ch1, ch2 = read_decimated(geom, raw_data)
    squint_correct(geom, ch1)
    squint_correct(geom, ch2)
    return range_compress(geom, ch1), range_compress(geom, ch2)


def focus(raw_data, raw_par, slc_lower, slc_upper,
          opts: FocusOptions | None = None) -> FocusGeometry:
    """Process one acquisition to its two SLCs, GAMMA big-endian, with pars.

    Mirrors ``gpri2_proc.py raw raw_par slc1 slc2 [options]``; ``slc1`` is
    the lower antenna (channel 1) and ``slc2`` the upper.
    """
    geom = geometry(raw_par, raw_data, opts)
    lower, upper = focus_channels(geom, raw_data)
    for path, img, ch in ((slc_lower, lower, 1), (slc_upper, upper, 2)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.astype(">c8").tofile(path)
        Path(str(path) + ".par").write_text(geom.slc_par(ch))
    return geom


# ---------------------------------------------------------------- campaigns
def find_raw(campaign_dir, raw_list=None) -> list[tuple[Path, Path]]:
    """Every ``(raw, raw_par)`` pair under a campaign directory, in time order.

    Every ``*.raw`` in the directory and its ``raw*/`` subdirectories is
    taken.  A GAMMA ``RAW_list`` (paths relative to the campaign directory,
    ``raw raw_par`` per line) restricts that to the acquisitions it names —
    the ones in the archive list only the subset somebody once set up to
    process, so it is not honoured unless asked for.
    """
    root = Path(campaign_dir)
    if raw_list is not None:
        pairs = []
        for line in Path(raw_list).read_text().splitlines():
            tok = line.split()
            if len(tok) >= 2:
                pairs.append((root / tok[0], root / tok[1]))
        return pairs
    raws = list(root.glob("*.raw"))
    for sub in sorted(root.glob("raw*")):
        if sub.is_dir():
            raws += sub.glob("*.raw")
    raws = sorted(set(raws), key=lambda p: p.name)
    return [(r, r.with_name(r.name + "_par")) for r in raws]


def scene_id(raw: Path) -> str:
    """``20170827_234940`` from ``.../20170827_234940.raw``."""
    return Path(raw).name.split(".")[0]


def write_slc_tabs(scene_dir, ids, slc_dir="slc") -> None:
    """``SLCu_tab`` and ``SLCl_tab`` listing the focused images, GAMMA style."""
    scene_dir = Path(scene_dir)
    for letter, tab in (("u", "SLCu_tab"), ("l", "SLCl_tab")):
        lines = [f"{slc_dir}/{i}{letter}.slc  {slc_dir}/{i}{letter}.slc.par"
                 for i in ids]
        (scene_dir / tab).write_text("\n".join(lines) + "\n")


def _focus_one(job):
    raw, raw_par, scene_dir, opts, overwrite = job
    sid = scene_id(raw)
    out_l = Path(scene_dir) / "slc" / f"{sid}l.slc"
    out_u = Path(scene_dir) / "slc" / f"{sid}u.slc"
    if not overwrite and out_l.exists() and out_u.exists() \
            and Path(str(out_u) + ".par").exists():
        return sid, "skipped"
    try:
        focus(raw, raw_par, out_l, out_u, opts)
    except Exception as e:                      # keep the campaign going
        for p in (out_l, out_u):
            if p.exists():
                p.unlink()
        return sid, f"failed: {e}"
    return sid, "done"


def focus_campaign(campaign_dir, scene_dir, opts: FocusOptions | None = None,
                   workers: int = 1, overwrite: bool = False, limit: int = 0,
                   raw_list=None, log=print) -> list[str]:
    """Focus every acquisition of a campaign into ``<scene_dir>/slc``.

    Writes ``SLCu_tab`` / ``SLCl_tab`` in ``scene_dir`` when done, so the
    result is a scene directory the rest of the package can open.  Returns
    the scene ids that have both SLCs.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pairs = find_raw(campaign_dir, raw_list)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise FileNotFoundError(f"no raw acquisitions under {campaign_dir}")
    scene_dir = Path(scene_dir)
    (scene_dir / "slc").mkdir(parents=True, exist_ok=True)
    jobs = [(r, p, scene_dir, opts, overwrite) for r, p in pairs]
    log(f"{len(jobs)} acquisitions -> {scene_dir}/slc  ({workers} workers)")

    t0 = time.time()
    done, n = [], 0
    if workers <= 1:
        results = map(_focus_one, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = (f.result() for f in as_completed(
            [pool.submit(_focus_one, j) for j in jobs]))
    for sid, status in results:
        n += 1
        if status.startswith("failed"):
            log(f"  {sid}  {status}")
        else:
            done.append(sid)
        if n % 25 == 0 or n == len(jobs):
            log(f"  {n}/{len(jobs)}  ({time.time() - t0:.0f} s)")
    if workers > 1:
        pool.shutdown()

    done.sort()
    write_slc_tabs(scene_dir, done)
    log(f"{len(done)} scenes focused, {len(jobs) - len(done)} failed; "
        f"wrote {scene_dir}/SLCu_tab and SLCl_tab")
    return done
