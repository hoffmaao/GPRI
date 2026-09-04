"""Atmospheric refractivity from meteorology, and per-epoch refractivity series.

:mod:`gpri_tools.atmosphere` estimates the atmospheric phase *empirically*, from the
interferograms themselves.  This module comes at the same quantity from the
other side — from pressure, temperature and humidity — which is what lets you
check whether an estimated screen is physically plausible, correct with met
data where you have it, and convert an estimated screen back into the weather
it implies.

The physics
-----------
Radio refractivity ``N = (n - 1) * 1e6`` for moist air, Smith & Weintraub (1953):

    N = 77.6 * P / T  +  3.73e5 * e / T^2

with ``P`` total pressure (hPa), ``T`` temperature (K), ``e`` partial pressure
of water vapour (hPa).  The first term is the dry (hydrostatic) part, the
second the wet part.  At 5 degC and 70 % relative humidity the wet term is only
about 8 % of the total — but it is the term that *changes*, and it changes fast.

For a horizontally-looking tripod radar the path is entirely inside the
boundary layer, so a change ``dN`` along the path puts a phase ramp that is
linear in slant range:

    phi(r) = (4 pi / lambda) * (dN * 1e-6) * r          (two-way)

Why this matters so much here
-----------------------------
At GPRI-II's ``lambda = 1.743 cm`` the scale factor is ``4 pi / lambda = 721``
radians per metre of path change.  Over the 16.6 km BakerBend1 swath, at
880 hPa, 5 degC and 70 % RH (:func:`sensitivity` computes these, so they stay
true if you change the reference conditions):

=====================================  ======  ==========================
change                                 dN      phase across swath
=====================================  ======  ==========================
1 degC at constant relative humidity   +1.01   +12.1 rad  (1.9 fringes)
1 degC at constant vapour pressure     -1.09   -13.0 rad  (2.1 fringes)
1 hPa pressure                         +0.28   +3.3 rad   (0.5 fringes)
1 % relative humidity                  +0.42   +5.0 rad   (0.8 fringes)
10 % relative humidity                 +4.21   +50.3 rad  (8.0 fringes)
=====================================  ======  ==========================

Note the first two rows disagree in **sign**.  Warming the air thins it, which
lowers the dry term; but at fixed relative humidity, warming also raises the
saturation vapour pressure by about 7 % per degree, and that raises the wet
term by more than the dry term falls.  So whether a temperature rise lengthens
or shortens the optical path depends entirely on what the humidity did at the
same time — which is why correcting from a thermometer alone is worse than not
correcting at all.  :func:`sensitivity` returns both partial derivatives for
exactly this reason.

Humidity dominates regardless, and a 10 % relative-humidity swing between two
acquisitions four minutes apart on a glacier is unremarkable — eight fringes.
This is the single largest error source in ground-based radar interferometry,
larger than the signal by an order of magnitude, which is why
:mod:`gpri_tools.atmosphere` exists at all.

Per-epoch refractivity
----------------------
:func:`invert_refractivity` treats the per-pair range ramps as a network
observation and solves for one refractivity value per epoch — the same SBAS
inversion :mod:`gpri_tools.timeseries` runs on displacement, applied to the
atmosphere instead.  The result is a refractivity time series you can plot
against a weather station and sanity-check directly.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "saturation_vapour_pressure", "vapour_pressure", "refractivity",
    "dry_refractivity", "wet_refractivity", "refractivity_phase",
    "delta_n_from_met", "sensitivity", "invert_refractivity",
    "ramp_from_delta_n", "MetRecord",
]

#: Smith & Weintraub (1953) coefficients.
K1 = 77.6      # K / hPa,  dry term
K2 = 3.73e5    # K^2 / hPa, wet term
ZERO_C = 273.15


# --------------------------------------------------------------- moist air
def saturation_vapour_pressure(temperature_c, over="water"):
    """Saturation vapour pressure (hPa) from temperature in degrees Celsius.

    Buck (1981), which is accurate to better than 0.1 % over the range a
    glacier boundary layer ever sees.  ``over='ice'`` uses the ice curve, which
    matters below freezing: at -10 degC the ice value is 9 % lower than the
    water value, and getting that wrong is a real 0.2 N-unit error.
    """
    t = np.asarray(temperature_c, float)
    if over == "ice":
        return 6.1115 * np.exp((23.036 - t / 333.7) * t / (279.82 + t))
    return 6.1121 * np.exp((18.678 - t / 234.5) * t / (257.14 + t))


def vapour_pressure(temperature_c, relative_humidity, over="water"):
    """Partial pressure of water vapour (hPa).

    ``relative_humidity`` may be a fraction or a percentage; anything above
    1.5 is taken as a percentage.
    """
    rh = np.asarray(relative_humidity, float)
    rh = np.where(rh > 1.5, rh / 100.0, rh)
    return rh * saturation_vapour_pressure(temperature_c, over=over)


def dry_refractivity(pressure_hpa, temperature_c):
    """Hydrostatic term ``K1 * P / T``, in N-units."""
    return K1 * np.asarray(pressure_hpa, float) / (np.asarray(temperature_c, float) + ZERO_C)


def wet_refractivity(temperature_c, relative_humidity=None, vapour_hpa=None,
                     over="water"):
    """Wet term ``K2 * e / T^2``, in N-units."""
    t = np.asarray(temperature_c, float) + ZERO_C
    if vapour_hpa is None:
        if relative_humidity is None:
            raise ValueError("need relative_humidity or vapour_hpa")
        vapour_hpa = vapour_pressure(temperature_c, relative_humidity, over=over)
    return K2 * np.asarray(vapour_hpa, float) / t ** 2


def refractivity(pressure_hpa, temperature_c, relative_humidity=None,
                 vapour_hpa=None, over="water"):
    """Total radio refractivity ``N = (n - 1) * 1e6`` for moist air.

    >>> round(float(refractivity(880.0, 5.0, 0.70)), 2)
    272.72
    """
    return (dry_refractivity(pressure_hpa, temperature_c)
            + wet_refractivity(temperature_c, relative_humidity, vapour_hpa, over))


class MetRecord:
    """One set of surface met observations, and the refractivity it implies."""

    def __init__(self, pressure_hpa, temperature_c, relative_humidity=None,
                 vapour_hpa=None, time=None, over="water"):
        self.pressure_hpa = float(pressure_hpa)
        self.temperature_c = float(temperature_c)
        self.relative_humidity = relative_humidity
        self.vapour_hpa = vapour_hpa
        self.time = time
        self.over = over

    @property
    def N(self):
        return float(refractivity(self.pressure_hpa, self.temperature_c,
                                  self.relative_humidity, self.vapour_hpa,
                                  self.over))

    @property
    def N_dry(self):
        return float(dry_refractivity(self.pressure_hpa, self.temperature_c))

    @property
    def N_wet(self):
        return self.N - self.N_dry

    def __repr__(self):
        return (f"MetRecord(P={self.pressure_hpa:.1f} hPa, "
                f"T={self.temperature_c:.1f} C, N={self.N:.2f} "
                f"[dry {self.N_dry:.2f} + wet {self.N_wet:.2f}])")


# ----------------------------------------------------- refractivity -> phase
def ramp_from_delta_n(delta_N, wavelength):
    """Range-phase slope (rad/m) produced by a refractivity change in N-units.

    The inverse of :func:`gpri_tools.atmosphere.delta_refractivity`, in the units met
    people actually use.  ``ramp = 4 pi * dN * 1e-6 / lambda``.
    """
    return 4.0 * np.pi * np.asarray(delta_N, float) * 1e-6 / np.asarray(wavelength, float)


def refractivity_phase(delta_N, slant_range, wavelength, reference_range=None):
    """Two-way phase screen (radians) from a path-averaged refractivity change.

    ``reference_range`` sets where the screen is zero — near range by default,
    which keeps the numbers readable and matches how
    :class:`gpri_tools.atmosphere.PhaseScreen` references its own ramp.
    """
    r = np.asarray(slant_range, float)
    r0 = r.min() if reference_range is None else float(reference_range)
    ramp = np.asarray(ramp_from_delta_n(delta_N, wavelength), float)
    if ramp.ndim:                    # one screen per epoch/pair -> (..., n_range)
        ramp = ramp[..., None]
    return ramp * (r - r0)


def delta_n_from_met(met_a, met_b):
    """Refractivity change between two met records, ``N(b) - N(a)``, in N-units.

    Sign matches :func:`refractivity_phase`: a positive result means the air
    got denser (or wetter) between the two acquisitions, the path got
    optically longer, and the target appears to have moved away.
    """
    a = met_a if isinstance(met_a, MetRecord) else MetRecord(*met_a)
    b = met_b if isinstance(met_b, MetRecord) else MetRecord(*met_b)
    return b.N - a.N


def sensitivity(pressure_hpa=880.0, temperature_c=5.0, relative_humidity=0.70,
                wavelength=0.01743, swath=16583.0):
    """Partial derivatives of ``N`` — and of the phase across a swath — w.r.t. met.

    Returns a dict with, per key, the change in N-units and a matching
    ``phase_<key>`` in radians across ``swath`` metres at ``wavelength``:

    ``dN_dT``
        per degC **at constant relative humidity** — what you get if a
        thermometer and a hygrometer both read the air.  Positive: the wet
        term's growth beats the dry term's decline.
    ``dN_dT_dry``
        per degC **at constant vapour pressure** — the dry term alone.
        Negative.  The two differ in sign, so which one applies depends on what
        the humidity did, and quoting either alone is misleading.
    ``dN_dP``
        per hPa.
    ``dN_dRH1``, ``dN_dRH10``
        per 1 % and per 10 % relative humidity.  These dominate everything else.

    Defaults are BakerBend1: 880 hPa at 1250 m, 5 degC, 70 % RH, Ku band, and a
    swath of ``far_range - near_range``.
    """
    P, T, RH = pressure_hpa, temperature_c, relative_humidity
    N0 = refractivity(P, T, RH)
    e0 = vapour_pressure(T, RH)
    d = {
        "dN_dT": float(refractivity(P, T + 1.0, RH) - N0),
        "dN_dT_dry": float(refractivity(P, T + 1.0, vapour_hpa=e0) - N0),
        "dN_dP": float(refractivity(P + 1.0, T, RH) - N0),
        "dN_dRH1": float(refractivity(P, T, min(RH + 0.01, 1.0)) - N0),
        "dN_dRH10": float(refractivity(P, T, min(RH + 0.10, 1.0)) - N0),
    }
    for k in list(d):
        d["phase_" + k] = float(ramp_from_delta_n(d[k], wavelength) * swath)
    d["N"] = float(N0)
    d["N_dry"] = float(dry_refractivity(P, T))
    return d


# ------------------------------------------------- per-epoch refractivity series
def invert_refractivity(pair_delta_n, network, weights=None, reference=0,
                        method="lstsq", **kwargs):
    """Per-epoch refractivity from per-pair refractivity changes.

    Each pair ``(i, j)`` gives one observation of ``N_j - N_i`` — take it from
    :attr:`gpri_tools.atmosphere.PhaseScreen.delta_n` (times ``1e6`` for N-units) on
    every interferogram.  This runs the same network inversion
    :mod:`gpri_tools.timeseries` uses for displacement, so the result is a per-epoch
    refractivity relative to the reference epoch.

    Two reasons to bother.  It is a real, physical time series you can plot
    against a weather station, which is the only independent check available on
    an empirically estimated screen.  And a per-epoch atmosphere is a stronger
    correction than a per-pair one: the pairs share epochs, so solving for
    ``n_epochs`` numbers instead of ``n_pairs`` averages the estimate down.

    Parameters
    ----------
    pair_delta_n : array (n_pairs, ...)
        Observed ``N_j - N_i`` per pair, N-units.
    network : :class:`gpri_tools.network.Network`

    Returns
    -------
    N : array (n_epochs, ...)
        Refractivity relative to the reference epoch, N-units.
    """
    from .timeseries import invert_network

    obs = np.asarray(pair_delta_n, float)
    ts = invert_network(obs, network, weights=weights, method=method,
                        reference=reference, **kwargs)
    return ts.displacement


def screens_to_delta_n(screens, wavelength=None):
    """Pull ``dN`` in N-units out of a list of fitted
    :class:`gpri_tools.atmosphere.PhaseScreen` objects."""
    out = []
    for s in screens:
        if s is None:
            out.append(np.nan)
            continue
        wl = wavelength if s.wavelength is None else s.wavelength
        if wl is None:
            raise ValueError("screen has no wavelength; pass wavelength=")
        out.append(s.ramp * wl / (4.0 * np.pi) * 1e6)
    return np.asarray(out, float)


__all__.append("screens_to_delta_n")
