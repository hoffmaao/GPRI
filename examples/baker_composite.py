#!/usr/bin/env python3
"""The part of the detrended displacement that repeats, campaign by campaign.

    python examples/baker_composite.py --scenes 20170827 20180808 20190719

``baker_population.py`` removes each campaign's secular rate and keeps what is
left; ``baker_seasons.py`` lays every UTC day of that on one clock.  The
question after those two is how much of the detrended displacement comes back
at the same hour on the next day.  For every campaign with two or more UTC days
this stacks them into an hour-of-day composite
(:func:`gpri.diurnal.hour_composite`) — the shape-agnostic diurnal, no sinusoid
assumed — and draws it over the days it came from, with the day-to-day spread
as a band.

The residual ``day - composite`` is the part no 24 h-periodic signal explains.
Printed beside the composite RMS it is the honest scale of any "diurnal
amplitude" read off a single day: where the two are the same size, one day
measures nothing.  Held-out bedrock gets the identical treatment in the lower
panel — it does not move, so its composite is the noise floor a diurnal on ice
has to beat.
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
from baker_population import population_path                       # noqa: E402
from baker_seasons import hourly, load_days                         # noqa: E402

from gpri.diurnal import hour_composite                             # noqa: E402


def composite_of(days, which: int):
    """``(composite, spread, residual RMS)`` over the UTC days of one campaign.

    ``which`` picks the population out of a ``load_days`` record: 2 is ice, 3
    held-out bedrock.  The composite is the hour-of-day mean over the whole
    record; the spread is the standard deviation *between* days of their hourly
    medians, NaN in an hour only one day saw, and so is only drawn where two
    days can disagree.
    """
    hod = np.concatenate([d[1] for d in days])
    y = np.concatenate([d[which] for d in days])
    comp, _ = hour_composite(hod, y)
    per_day = np.array([hourly(d[1], d[which]) for d in days])
    with np.errstate(invalid="ignore"):
        seen = np.sum(np.isfinite(per_day), axis=0)
        spread = np.where(seen >= 2, np.nanstd(per_day, axis=0), np.nan)
    resid = y - comp[np.minimum((hod % 24).astype(int), 23)]
    return comp, spread, float(np.sqrt(np.nanmean(resid ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="scene keys; default is every scene with a population "
                         "file and enough days")
    ap.add_argument("--antenna", default="upper", choices=("upper", "lower"))
    ap.add_argument("--decimate", type=int, default=16)
    ap.add_argument("--detrend", default="auto", choices=("auto", "linear"),
                    help="'auto' uses the same-hour secular rate the population "
                         "step removed; 'linear' the one-cycle line")
    ap.add_argument("--min-days", type=int, default=2,
                    help="UTC days a campaign needs before a composite means "
                         "anything (default 2)")
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    names = args.scenes or sorted(SCENES)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    campaigns = []
    for name in names:
        scene = Path(SCENES.get(name, name))
        if not population_path(scene, args.antenna, args.decimate).exists():
            if args.scenes:
                print(f"{name}: no population file for the {args.antenna} "
                      f"antenna, skipped")
            continue
        days, ice_rate, rock_rate, how = load_days(scene, args.antenna,
                                                   args.decimate, args.detrend)
        days = [d for d in days if d[1][-1] - d[1][0] >= 2]   # not a few epochs
        if len(days) < args.min_days:
            print(f"{name}: {len(days)} UTC day(s) of more than 2 h, fewer than "
                  f"{args.min_days} -- nothing repeats within one day, skipped")
            continue
        campaigns.append((name, days, ice_rate, rock_rate, how))

    if not campaigns:
        sys.exit("no campaign has enough days; run baker_population.py first")

    print(f"\nhour-of-day composite, {args.antenna} antenna, "
          f"{campaigns[0][4]} detrend")
    print(f"{'campaign':14s} {'days':>5s} {'secular':>10s} {'composite':>10s} "
          f"{'not repeated':>13s} {'trough UTC':>11s} {'depth':>8s} {'rock':>9s}")
    rows = {}
    for name, days, ice_rate, rock_rate, how in campaigns:
        ci, si, ri = composite_of(days, 2)
        cr, sr, rr = composite_of(days, 3)
        rows[name] = (days, ci, si, cr, sr)
        h = int(np.nanargmin(ci))
        print(f"{name:14s} {len(days):5d} {ice_rate:+8.2f} m/yr "
              f"{np.sqrt(np.nanmean(ci ** 2)):7.2f} mm {ri:10.2f} mm "
              f"{h:8d} h {ci[h]:6.1f} mm {np.sqrt(np.nanmean(cr ** 2)):6.2f} mm")
    print("\n  'composite' is the RMS of the repeating waveform, 'not repeated' "
          "the RMS of\n  what is left of each day once it is removed; 'rock' is "
          "the same composite on\n  held-out bedrock, which does not move.")

    # ---- figure ------------------------------------------------------------
    hours = np.arange(24) + 0.5
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for i, (name, days, ice_rate, rock_rate, how) in enumerate(campaigns):
        c = palette[i % len(palette)]
        _, ci, si, cr, sr = rows[name]
        for ax, comp, spread, which, lw in ((axes[0], ci, si, 2, 2.0),
                                            (axes[1], cr, sr, 3, 1.4)):
            for d in days:                       # the days behind the composite
                ax.plot(d[1], d[which], color=c, lw=0.7, alpha=0.35)
            ax.fill_between(hours, comp - spread, comp + spread, color=c,
                            alpha=0.15, lw=0)
            label = (f"{name}  {len(days)} days, "
                     f"{ice_rate:+.1f} m/yr removed"
                     if which == 2 else None)
            ax.plot(hours, comp, color=c, lw=lw, drawstyle="steps-mid",
                    label=label)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.grid(axis="x", color="0.85", lw=0.5)
        ax.axvspan(3.5, 13.0, color="0.93", zorder=0)   # sunset to sunrise, PDT
    axes[0].set_ylabel("Ice anomaly (mm)")
    axes[0].set_title("RGI ice: the hour-of-day composite (bold) over the days "
                      "it averages (thin), band ±1 sigma between days  "
                      "(+ toward radar; shading: local night)", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8, ncol=2)
    axes[1].set_ylabel("Rock anomaly (mm)")
    axes[1].set_title("held-out bedrock, the same statistic: the noise floor",
                      fontsize=10)
    axes[1].set_xlabel("UTC (hr)")
    plt.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = ("" if args.antenna == "upper" else f"_{args.antenna}") + \
        ("" if args.detrend == "auto" else "_linear")
    out = args.outdir / f"21_composite{tag}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
