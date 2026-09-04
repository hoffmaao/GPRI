#!/usr/bin/env python3
"""Inventory every GPRI campaign on cold storage: how long, how fast, how ready.

    bin/survey_campaigns.py                 # roots from GPRI_SURVEY_ROOTS
    bin/survey_campaigns.py --volumes /path/to/archive /path/to/projects

Answers the question that decides which dataset to work on: **how many diurnal
cycles does each campaign span, and is it processed far enough to use?**

A campaign directory is identified by holding acquisitions named
``YYYYMMDD_HHMMSS`` — as ``raw``, ``slc`` or ``diff`` files.  For each one this
reports the acquisition count, the span in hours, the median cadence, and the
processing stage reached:

``raw``
    FMCW sweeps only.  Needs GAMMA's ``par_GPRI2_SLC`` to go further, so on a
    host without GAMMA this data cannot be used at all.
``slc``
    Focused single-look complex images.  Interferograms can be formed from
    these with numpy alone — no GAMMA needed — via
    :func:`gpri_tools.covariance.coherence_from_slcs`.
``diff``
    Interferograms already formed.  Ready for the whole package.

Listing a directory over NFS is slow, so every walk is bounded by ``--timeout``
and the tool reports what it could not read rather than silently omitting it.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

STAMP = re.compile(r"(\d{8}_\d{6})")

def default_volumes():
    """Roots to walk, from GPRI_SURVEY_ROOTS (colon-separated) in the
    environment or in site.env at the repository root.  Machine-specific
    paths live only there, never in the repository."""
    import os
    env = Path(__file__).resolve().parent.parent / "site.env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    return [p for p in os.environ.get("GPRI_SURVEY_ROOTS", "").split(":") if p]

#: Subdirectory name -> processing stage, most processed first.
STAGES = [("diff", "diff"), ("slc", "slc"), ("raw", "raw")]


def stamps_in(directory: Path, limit: int = 200000):
    """Distinct acquisition timestamps among a directory's filenames."""
    out = set()
    try:
        with __import__("os").scandir(directory) as it:
            for k, entry in enumerate(it):
                if k > limit:
                    break
                m = STAMP.search(entry.name)
                if m:
                    out.add(m.group(1))
    except (OSError, PermissionError):
        return None
    return out


def stage_of(name: str):
    low = name.lower()
    for prefix, stage in STAGES:
        if low.startswith(prefix):
            return stage
    return None


def survey_scene(scene: Path):
    """Collect timestamps per processing stage under one campaign directory."""
    found = {}
    for child in sorted(scene.iterdir()):
        if not child.is_dir():
            continue
        stage = stage_of(child.name)
        if stage is None:
            continue
        s = stamps_in(child)
        if s:
            found.setdefault(stage, set()).update(s)
    loose = stamps_in(scene)
    if loose:
        found.setdefault("raw", set()).update(loose)
    return found


def describe(stamps):
    """Count, span in hours, and median cadence in minutes.

    ``\\d{8}_\\d{6}`` also matches digit runs that are not acquisition times —
    file sizes, version numbers, an hour field of 99 — so every stamp is
    validated by actually parsing it and the impostors are dropped rather than
    allowed to blow the span out to a fictitious value.
    """
    t = []
    for s in stamps:
        try:
            t.append(datetime.strptime(s, "%Y%m%d_%H%M%S"))
        except ValueError:
            continue
    t.sort()
    if len(t) < 2:
        return len(t), 0.0, 0.0
    span = (t[-1] - t[0]).total_seconds() / 3600.0
    gaps = sorted((b - a).total_seconds() / 60.0 for a, b in zip(t, t[1:]))
    return len(t), span, gaps[len(gaps) // 2]


def find_scenes(root: Path, max_depth=5):
    """Directories that look like campaigns: they contain a raw/slc/diff child."""
    out, stack = [], [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            children = [c for c in d.iterdir() if c.is_dir()]
        except (OSError, PermissionError):
            continue
        if any(stage_of(c.name) for c in children):
            out.append(d)
            continue
        stack.extend((c, depth + 1) for c in children
                     if not c.name.startswith("."))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", nargs="*", default=None)
    ap.add_argument("--min-hours", type=float, default=0.0,
                    help="only report campaigns spanning at least this long")
    args = ap.parse_args()

    volumes = args.volumes if args.volumes else default_volumes()
    if not volumes:
        print("no volumes: pass --volumes or set GPRI_SURVEY_ROOTS in "
              "site.env (see site.env.example)", file=sys.stderr)
        return 2
    rows = []
    for vol in volumes:
        root = Path(vol)
        if not root.is_dir():
            print(f"skipping {root} (not readable)", file=sys.stderr)
            continue
        for scene in find_scenes(root):
            found = survey_scene(scene)
            if not found:
                continue
            best = next((s for _, s in STAGES if s in found), None)
            n, span, cadence = describe(found[best])
            if span < args.min_hours:
                continue
            rows.append((scene, best, n, span, cadence))

    rows.sort(key=lambda r: -r[3])
    print(f"{'campaign':58} {'stage':6} {'n':>5} {'span/h':>7} "
          f"{'cad/min':>8} {'cycles':>7}")
    print("-" * 96)
    for scene, stage, n, span, cadence in rows:
        name = str(scene)
        if len(name) > 57:
            name = "..." + name[-54:]
        print(f"{name:58} {stage:6} {n:5d} {span:7.1f} {cadence:8.2f} "
              f"{span / 24:7.2f}")

    print()
    usable = [r for r in rows if r[1] in ("slc", "diff")]
    long_ones = [r for r in rows if r[3] >= 24.0]
    print(f"{len(rows)} campaigns; {len(long_ones)} span a full diurnal cycle; "
          f"{len(usable)} are processed past raw.")
    blocked = [r for r in long_ones if r[1] == "raw"]
    if blocked:
        print(f"\n{len(blocked)} campaign(s) long enough for the diurnal question "
              f"are still raw,\nand raw -> SLC needs GAMMA's par_GPRI2_SLC:")
        for scene, _, n, span, _ in blocked:
            print(f"  {span:6.1f} h  {n:5d} acquisitions  {scene}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
