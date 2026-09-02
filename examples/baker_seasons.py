#!/usr/bin/env python3
"""Every processed day on one clock: does the glacier's day repeat across a season?

    python examples/baker_seasons.py --scenes 20170713_full 20170803 20170827

``baker_population.py`` caches, per campaign and antenna, the median departure
of the RGI ice from its secular trend — the same-hour rate where the record
is longer than a day, each pixel's linear fit where it is not — and the same
series over the held-out bedrock that no correction ever saw.  This script overlays them
against the UTC hour of day — one trace per calendar day, so a two-day
campaign contributes two — and asks the only question a handful of days can
answer: does the *timing* repeat?  Amplitudes are free to differ (melt
differs from day to day and week to week); a diurnal cycle driven by the sun
keeps its hours, an atmospheric residual keeps the bedrock's.

Printed per day: the hourly-binned ice median (a 24-column clock), the hour
and depth of the trough, the hour the ice climbs back above its trend, and
the bedrock RMS at the same hours.  Then the correlation between every pair
of days with at least ``--overlap`` hours in common, for the ice and for the
rock: the ice matrix is the repeat, the rock matrix is what chance and shared
atmosphere give the same statistic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baker_aps import SCENES                                        # noqa: E402
from baker_population import population_path                        # noqa: E402


def load_days(scene: Path, antenna: str, dec: int, detrend: str = "auto"):
    """One record per UTC calendar day: (date, hour-of-day, ice, rock)."""
    npz = population_path(scene, antenna, dec)
    if not npz.exists():
        sys.exit(f"{npz} is missing: run baker_population.py --scene "
                 f"{scene.name} --antenna {antenna} --decimate {dec} --rgi first")
    z = np.load(npz)
    t = z["epoch0"].astype("datetime64[s]") + (z["hours"] * 3600).astype("timedelta64[s]")
    date = t.astype("datetime64[D]")
    hod = (t - date).astype(float) / 3600.0
    ice, rock = (z["ice"], z["rock"]) if detrend == "auto" else \
        (z["ice_linear"], z["rock_linear"])
    days = []
    for d in np.unique(date):
        m = date == d
        days.append((str(d), hod[m], ice[m], rock[m]))
    how = str(z["detrend"]) if detrend == "auto" else "linear"
    rates = ("ice_rate", "rock_rate") if how != "linear" else \
        ("ice_rate_linear", "rock_rate_linear")
    return days, float(z[rates[0]]), float(z[rates[1]]), how


def hourly(hod, y):
    """Median of ``y`` inside each UTC hour; NaN where the hour was not observed."""
    out = np.full(24, np.nan)
    for h in range(24):
        m = (hod >= h) & (hod < h + 1)
        if m.sum() >= 3:
            out[h] = np.nanmedian(y[m])
    return out


def trough_and_recovery(hod, y):
    """Hour of the minimum and the first later hour the series is back above zero."""
    i = int(np.nanargmin(y))
    after = np.where((hod > hod[i]) & (y > 0))[0]
    return hod[i], y[i], (hod[after[0]] if after.size else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+",
                    default=["20170713_full", "20170803", "20170827"])
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--overlap", type=float, default=12.0,
                    help="hours two days must share before they are correlated")
    ap.add_argument("--detrend", default="auto", choices=("auto", "linear"),
                    help="'auto' takes each campaign's best line (same-hour "
                         "rate where the record allows it); 'linear' the "
                         "per-pixel least-squares line everywhere")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    days = []              # (label, hod, ice, rock)
    colours = {}
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, name in enumerate(args.scenes):
        scene = Path(SCENES.get(name, name))
        recs, ice_rate, rock_rate, how = load_days(scene, args.antenna,
                                                   args.decimate, args.detrend)
        colours[name] = palette[i % len(palette)]
        print(f"{name}: {len(recs)} UTC day(s), {how} detrend, median LOS rate "
              f"ice {ice_rate:+.2f} m/yr, held-out bedrock {rock_rate:+.2f} m/yr")
        for date, hod, ice, rock in recs:
            if hod[-1] - hod[0] < 2:
                continue                       # a few epochs past midnight
            days.append((name, date, hod, ice, rock))

    # ---- the clock table ---------------------------------------------------
    print(f"\nhourly ice median, departure from trend (mm), by UTC hour")
    print(f"{'day':22s}" + "".join(f"{h:>4d}" for h in range(24)))
    H = {}
    for name, date, hod, ice, rock in days:
        H[date] = (hourly(hod, ice), hourly(hod, rock))
        row = "".join("   ." if not np.isfinite(v) else f"{v:4.0f}" for v in H[date][0])
        print(f"{date:22s}{row}")
    print(f"\n{'day':12s} {'span (UTC)':>13s} {'trough':>9s} {'depth':>8s} "
          f"{'above trend':>12s} {'rock RMS':>9s}")
    for name, date, hod, ice, rock in days:
        th, depth, rec = trough_and_recovery(hod, ice)
        rec_s = f"{rec:5.1f} h" if np.isfinite(rec) else "    - "
        print(f"{date:12s} {hod[0]:5.1f}-{hod[-1]:5.1f} h {th:6.1f} h {depth:6.1f} mm "
              f"{rec_s:>12s} {np.nanstd(rock):6.2f} mm")

    # ---- does the timing repeat?  correlation across days -------------------
    dates = [d[1] for d in days]
    for what, k in (("ice", 0), ("held-out bedrock", 1)):
        print(f"\ncorrelation between days, hourly {what} median "
              f"(>= {args.overlap:g} h in common)")
        print(f"{'':12s}" + "".join(f"{d[5:]:>8s}" for d in dates))
        for a in dates:
            cells = []
            for b in dates:
                x, y = H[a][k], H[b][k]
                ok = np.isfinite(x) & np.isfinite(y)
                if a == b or ok.sum() < args.overlap:
                    cells.append(f"{'-':>8s}")
                else:
                    cells.append(f"{np.corrcoef(x[ok], y[ok])[0, 1]:8.2f}")
            print(f"{a:12s}" + "".join(cells))

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    styles = ("-", "--", ":")
    nth = {}
    for name, date, hod, ice, rock in days:
        c, i = colours[name], nth.get(name, 0)
        nth[name] = i + 1
        label = f"{name} ({date})" if i == 0 else date
        axes[0].plot(hod, ice, color=c, lw=1.2, ls=styles[i % 3], label=label)
        axes[1].plot(hod, rock, color=c, lw=1.0, ls=styles[i % 3])
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.grid(axis="x", color="0.85", lw=0.5)
        ax.axvspan(3.5, 13.0, color="0.93", zorder=0)   # sunset to sunrise, PDT
    axes[0].set_ylabel("Ice anomaly (mm)")
    axes[0].set_title(f"RGI ice: median departure from its secular trend, every "
                      f"processed day on one UTC clock ({args.antenna} antenna; "
                      f"shading: local night)", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8, ncol=3)
    axes[1].set_ylabel("Rock anomaly (mm)")
    axes[1].set_title("held-out bedrock, the same statistic", fontsize=10)
    axes[1].set_xlabel("UTC (hr)")
    plt.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = ("" if args.antenna == "upper" else f"_{args.antenna}") + \
        ("" if args.detrend == "auto" else "_linear")
    out = args.outdir / f"20_seasons{tag}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
