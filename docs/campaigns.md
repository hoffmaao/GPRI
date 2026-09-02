# GPRI campaign inventory

Surveyed 2026-08-30. Regenerate with `bin/survey_campaigns.py` (data roots
come from `GPRI_SURVEY_ROOTS` in `site.env` — see `site.env.example`; nothing
machine-specific lives in this repository).

The question this answers: **which campaign should the diurnal analysis use?**
The target signal is sub-daily velocity and uplift variation driven by the
subglacial drainage system, so what matters is how many diurnal cycles a
campaign spans and whether it is processed far enough to use.

## The campaigns that matter

| campaign | copy | stage | acquisitions | span | cadence | cycles |
|---|---|---|---:|---:|---:|---:|
| `20170827` | archive | **raw** | 1335 | **44.9 h** | 2.0 min | **1.87** |
| `20170827` | working | **slc** | 1335 | 44.9 h | 2.0 min | 1.87 |
| `20170803` | archive | slc | 723 | 24.2 h | 2.0 min | 1.01 |
| `20170803` | working (×3) | **diff** | 722 | 24.1 h | 2.0 min | 1.01 |
| `20170713` | archive | raw | 271 | 23.9 h | 5.0 min | 0.996 |
| `20170713_full` | working | slc | 271 | 23.9 h | 5.0 min | 0.996 |
| `20170713` | working | diff | 246 | 21.8 h | 5.0 min | 0.91 |
| `20180709` | archive | raw | 213 | 17.9 h | 2.0 min | 0.75 |
| `20170913` | archive | raw | 437 | 14.5 h | 2.0 min | 0.61 |
| `20160826` | archive | raw | 57 | 5.2 h | 5.0 min | 0.22 |
| `20160826` | working (×2) | diff | 26+27 | 3.7 h | 5.0 min | 0.15 |

25 campaign directories in total across the surveyed roots; the rest are lab,
balcony and short field tests under an hour, listed by the survey tool but not
useful for this question.

Stage means: **raw** — FMCW sweeps only, which `gpri focus` turns into SLCs
for both antennas (a port of GAMMA's `gpri2_proc.py`, validated to float32
rounding against the 20170803 archive); **slc** — focused complex images, which `gpri.stack.SlcPairStack` turns into
interferograms and coherence on demand (either antenna, any lag set, any
multilook — validated against GAMMA's own `.diff`/`.cc`); **diff** —
interferograms already formed by GAMMA.  For this package **slc** and
**diff** are equally usable; a GAMMA `diff0/` only covers the upper antenna
and lag 1, so even on `20170803` the lower antenna and the closure network
come from the SLCs.

## What this means

**The best dataset for the diurnal question is `20170827`.** It spans 44.9
hours — 1.87 diurnal cycles, 1335 acquisitions at 2-minute cadence — which is
worth far more than one cycle for three reasons:

1. A single cycle cannot cleanly separate the diurnal amplitude from the
   secular flow rate: in an epoch-domain fit the two are correlated at 0.78
   over one period and at 0.37 over 1.87. Two cycles break the degeneracy.
2. Two cycles let you check whether the diurnal **repeats**. A signal that
   recurs at the same phase on consecutive days is hard to explain as anything
   but a forced response; a one-off is not.
3. The phase (hour of peak) is the diagnostic quantity for drainage-system
   behaviour, and its uncertainty falls sharply with a second cycle.

It was left as **582 GB of raw data** with `SLCu_tab`/`SLCl_tab`/`itab_mr`
already written, pointing at an SLC directory that does not exist — set up
for processing and never processed, or the SLCs were deleted. Its `itab_mr`
is an *i*→*i*+3 network, not a daisy chain, so unlike 20170803 it has closed
triangles: `gpri closure` works on it, and the pair-domain least squares in
`gpri.pairlsq` gains real sensitivity from the longer combinations.

`gpri focus <campaign> <scene> --workers 6` writes the scene (2670 SLCs,
~90 minutes, limited by how fast the raw can be read). What the raw actually
holds, which the archive's own `RAW_list` (44 entries) does not say:

- **1335 acquisitions** in three subdirectories: `raw/` (197), `raw2/` (558)
  and `raw3/` (580).
- **Two scan geometries.** The first 197 sweep −30..50° (17.0 s capture, 396
  image lines, as on 20170803); from 06:42 UTC on the 28th the scan is
  −30..60° (19.0 s, 446 lines). Both start at the same azimuth, so
  `SlcPairStack` crops each pair to the common 396-line block and every
  script runs unchanged across the change.
- **Two gaps**: 19.3 minutes at the geometry change (Aug 28 06:22 → 06:42)
  and 7.7 minutes on the 29th (01:16 → 01:23). The network stays connected;
  those two pairs just have longer baselines.
- 219 of the `raw_par` files carry no GPS fix; the radar position for
  geocoding comes from the first acquisition, which does.

`20170803` is the scene GAMMA shipped processed: 24.1 h at 2-minute cadence,
interferograms formed, and the default scene for this repository. One cycle
exactly — enough to fit a diurnal harmonic but not enough to check that it
repeats, and marginal for separating amplitude from secular rate. Its SLCs cover **both receive antennas**, which
gives the one replicate the campaign has: the lower antenna, formed by
`SlcPairStack`, run through the identical chain (`examples/baker_antennas.py`
— the noise floor and the replication test in the README). The same SLCs
supply the *i*→*i*+2, *i*→*i*+3 (and longer) pairs the shipped daisy chain
lacks, so closure phase is measurable on this day too.

`20170713` as shipped spans 21.8 h — **0.91 of a cycle**.
`gpri.diurnal.harmonic_design` refuses it, correctly: over less than one
period the amplitude and the secular rate are not separable, and a number
returned there would be meaningless. The raw archive holds 271 acquisitions
(the survey's 279 counted `.raw.log` files) over 23.9 h — 0.996 of a cycle,
five minutes short of the last sample. Refocused with `gpri focus` into the
`20170713_full` scene it is accepted: the fits tolerate `MIN_CYCLES = 0.98`
of a period because the rate/harmonic correlation has no cliff at exactly one
cycle (0.78 at 0.996 cycles against 0.78 at 1.0 in an epoch-domain fit, and
close to zero either way in the pair domain), so a record a few minutes short
of a day is the same fit as one exactly a day long.

`20160826` is short (3.7 h processed) but was processed into **both** a
single-reference and a chain network, so the merged set has 25 closed
triangles from GAMMA's own products — the closure data that needs nothing
formed here (`examples/baker_closure.py`). With `SlcPairStack` the same
script measures closure on `20170803` from thousands of triangles.

## Recommended order

1. Use `20170803` now, with the caveats above.
2. `20170827`, focused with `gpri focus` (`bin/run_scene.sh 20170827` runs
   the whole chain). It is the dataset the experiment deserves: 1.87 cycles,
   so the diurnal signal can be checked for repeating from one day to the
   next (`examples/baker_repeat.py`).
3. `20170713_full`, the same archive refocused to its full 23.9 h — a second
   independent day at a different time of season, also run end to end
   (`bin/run_scene.sh 20170713_full`, half an hour). It is the quiet
   campaign: no per-pixel diurnal detection, no net line-of-sight rate over
   the coherent ice, and a night-time trough of a quarter the August depth
   at the same hours (`examples/baker_seasons.py`, README "Three campaigns
   on one clock").
4. `20180709` (17.9 h) and `20170913` (14.5 h) fall short of a cycle even fully
   processed. They are useful for secular velocity, not for diurnal phase.

## Practical notes

- Some project trees on the shared storage are not group-readable; if a
  campaign is missing from the survey output, check permissions with whoever
  administers the storage.
- Listing large acquisition directories over a network mount is slow; the
  survey tool bounds every walk and reports what it could not read rather
  than silently omitting it.
