"""Phase-linked LOS displacement time series from GPRI interferograms.

The package consumes GAMMA products directly — the ``.diff`` / ``.cc`` rasters
and ``SLC_tab`` / ``itab`` tables that GAMMA's ISP and DIFF modules write — and
carries them through to a line-of-sight displacement time series, in a map
projection, without needing the GAMMA binaries themselves.  Raw campaigns are
covered too: :mod:`gpri.focus` turns the instrument's FMCW sweeps into SLCs
the way GAMMA's ``gpri2_proc.py`` does.

Pipeline
--------
0.  :mod:`gpri.focus`         raw FMCW sweeps to SLCs (GAMMA's gpri2_proc.py)
1.  :mod:`gpri.gamma`         read GAMMA parameter files and binary rasters
2.  :mod:`gpri.network`       epochs, pairs, design matrices, closure triplets
3.  :mod:`gpri.stack`         patch-wise access to a whole ``diff0`` directory,
                              or pairs formed on demand from the SLCs
4.  :mod:`gpri.covariance`    sample coherence matrices
5.  :mod:`gpri.phaselink`     EVD / eigenSAR / EMI / ML phase linking
6.  :mod:`gpri.atmosphere`    range-dependent refractivity screen removal
6b. :mod:`gpri.aps`           network-consistent epoch screens, drift and turbulence
6c. :mod:`gpri.glaciers`      RGI outlines: where the ice actually is
7.  :mod:`gpri.refractivity`  the same screens from meteorology, and per-epoch N
8.  :mod:`gpri.closure`       closure-phase bias estimation and correction
9.  :mod:`gpri.psinterp`      PS-interpolation unwrapping over decorrelated ground
10. :mod:`gpri.timeseries`    network inversion, stacking, LOS displacement
10b. :mod:`gpri.pairlsq`      single-step pair-domain WLS with uncertainties
11. :mod:`gpri.diurnal`       harmonic analysis, and telling ice from atmosphere
12. :mod:`gpri.geocode`       polar radar geometry to a local stereographic map
13. :mod:`gpri.plot`          figures, in radar and map geometry

Sign convention
---------------
Every phase in this package follows GAMMA's ``SLC_intf`` convention, in which
the interferogram for pair ``(i, j)`` is ``z_i * conj(z_j)`` and therefore
carries phase ``theta_i - theta_j``.  Displacement is reported **positive
toward the radar** (a decrease in slant range).  See
:func:`gpri.timeseries.los_displacement` for the derivation.
"""
from __future__ import annotations

__version__ = "0.4.0"

from . import (aps, atmosphere, closure, covariance, diurnal, focus, gamma,
               geocode, glaciers,
               network, pairlsq, phaselink, psinterp, refractivity, stack,
               timeseries)
from .aps import epoch_screen_correction, invert_screens, turbulence_screen
from .closure import correct_bias, estimate_bias
from .diurnal import diurnal_amplitude, fit_harmonics, range_dependence
from .focus import FocusOptions, focus as focus_raw, focus_campaign
from .gamma import ParFile, read_image, read_slc, write_image
from .geocode import RadarGeometry, geocode as geocode_image, local_stereographic
from .network import Network, read_itab, read_slc_tab
from .pairlsq import fit_pairs
from .phaselink import phase_link, temporal_coherence
from .psinterp import select_ps, unwrap_with_ps
from .refractivity import invert_refractivity, refractivity as refractivity_of
from .stack import DiffStack, SlcPairStack
from .timeseries import invert_network, los_displacement, stack_velocity

__all__ = [
    "__version__",
    "aps", "atmosphere", "closure", "covariance", "diurnal", "focus", "gamma",
    "geocode",
    "network", "pairlsq", "phaselink", "psinterp", "refractivity", "stack",
    "timeseries",
    "ParFile", "read_image", "read_slc", "write_image",
    "Network", "read_itab", "read_slc_tab",
    "phase_link", "temporal_coherence",
    "DiffStack", "SlcPairStack",
    "los_displacement", "invert_network", "stack_velocity",
    "estimate_bias", "correct_bias",
    "invert_screens", "epoch_screen_correction", "turbulence_screen",
    "glaciers",
    "fit_pairs",
    "FocusOptions", "focus_raw", "focus_campaign",
    "fit_harmonics", "diurnal_amplitude", "range_dependence",
    "select_ps", "unwrap_with_ps",
    "invert_refractivity", "refractivity_of",
    "RadarGeometry", "geocode_image", "local_stereographic",
]


def __getattr__(name):
    # matplotlib is an optional dependency, so gpri.plot is imported on first
    # use rather than at import time.  It has to go through importlib:
    # `from . import plot` looks the name up on this package first, which lands
    # straight back in here and recurses until the stack runs out.
    if name == "plot":
        import importlib

        module = importlib.import_module(".plot", __name__)
        globals()["plot"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
