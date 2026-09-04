# GPRI campaign inventory

Surveyed 2026-08-30; two campaigns added 2026-09-02 from a backup of the
field computer that is not under the survey roots (see "Two campaigns off a
backup" below). Regenerate with `bin/survey_campaigns.py` (data roots come
from `GPRI_SURVEY_ROOTS` in `site.env` — see `site.env.example`; nothing
machine-specific lives in this repository).

The question this answers: **which campaign should the diurnal analysis use?**
The target signal is sub-daily velocity and uplift variation driven by the
subglacial drainage system, so what matters is how many diurnal cycles a
campaign spans and whether it is processed far enough to use.

## The campaigns that matter

| campaign | copy | stage | acquisitions | span | cadence | cycles |
|---|---|---|---:|---:|---:|---:|
| `20190719` | backup | **raw** | 1140 | **45.7 h** | 2.0 min | **1.90** |
| `20190719` | working | **slc** | 1137 | 45.7 h | 2.0 min | 1.90 |
| `20170827` | archive | **raw** | 1335 | **44.9 h** | 2.0 min | **1.87** |
| `20170827` | working | **slc** | 1335 | 44.9 h | 2.0 min | 1.87 |
| `20180808` | backup | **raw** | 1229 | **41.4 h** | 2.0 min | **1.73** |
| `20180808` | backup | slc | 1228 | 41.4 h | 2.0 min | 1.73 |
| `20180808` | working | **slc** | 1227 | 41.4 h | 2.0 min | 1.73 |
| `20170803` | archive | slc | 723 | 24.2 h | 2.0 min | 1.01 |
| `20170803_full` | working | slc | 723 | 24.2 h | 2.0 min | 1.01 |
| `20170803` | working (×3) | **diff** | 722 | 24.1 h | 2.0 min | 1.01 |
| `20170713` | archive | raw | 271 | 23.9 h | 5.0 min | 0.996 |
| `20170713_full` | working | slc | 271 | 23.9 h | 5.0 min | 0.996 |
| `20170713` | working | diff | 246 | 21.8 h | 5.0 min | 0.91 |
| `20180709` | archive | raw | 213 | 17.9 h | 2.0 min | 0.75 |
| `20180709` | working | slc | 197 (+6 set-up) | **6.9 h** | 2.0 min | 0.29 |
| `20170913` | archive | raw | 437 | 14.5 h | 2.0 min | 0.61 |
| `20170913` | working | slc | 437 | 14.5 h | 2.0 min | 0.61 |
| `20160826` | archive | raw | 57 | 5.2 h | 5.0 min | 0.22 |
| `20160826_full` | working | slc | 44 | 3.7 h | 5.0 min | 0.15 |
| `20160826` | working (×2) | diff | 26+27 | 3.7 h | 5.0 min | 0.15 |

25 campaign directories in total across the surveyed roots; the rest are lab,
balcony and short field tests under an hour, listed by the survey tool but not
useful for this question.

Stage means: **raw** — FMCW sweeps only, which `gpri focus` turns into SLCs
for both antennas (a port of GAMMA's `gpri2_proc.py`, validated to float32
rounding against the 20170803 archive); **slc** — focused complex images, which `gpri_tools.stack.SlcPairStack` turns into
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
   The same holds for a waveform that is not a sinusoid. The one estimate
   of the secular rate that no 24 h-periodic shape can bias is the
   difference between epochs exactly a day apart
   (`gpri_tools.diurnal.secular_slope`); on a 24.2 h record that is five pairs of
   epochs at the two ends, on 44.9 h it is every epoch of the first 20.9 h
   against its partner a day later.
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
`gpri_tools.pairlsq` gains real sensitivity from the longer combinations.

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
— the noise floor and the replication test in
[`baker.md`](baker.md#two-antennas-one-day-the-replicate)). The same SLCs
supply the *i*→*i*+2, *i*→*i*+3 (and longer) pairs the shipped daisy chain
lacks, so closure phase is measurable on this day too.

`20170713` as shipped spans 21.8 h — **0.91 of a cycle**.
`gpri_tools.diurnal.harmonic_design` refuses it, correctly: over less than one
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

## Two campaigns off a backup

Two of the three campaigns that span more than one diurnal cycle were not on
the analysis volumes at all. They are in a backup of the GPRI field computer
made in July 2019 — a directory tree that had never been group-readable, and
whose permissions had to be opened by its owner before anything could be read:

- **`20180808`** — 1,229 acquisitions, 2018-08-09 00:03 to 08-10 17:25 UTC,
  2-minute cadence, scan −55° → +30°. GAMMA had already focused it (1,228
  SLC pairs to 12.5 km range, with `mli`, `diff0`–`diff3` and a
  `processing_flow_ndh.sh`); it is refocused here across the full swath so
  that it matches the rest. 226 of its raw files are gzipped, which
  `gpri focus` now reads directly.
- **`20190719`** — 1,140 acquisitions, 2019-07-19 17:47 to 07-21 15:28 UTC,
  2-minute cadence, scan −45° → +40°, split across three day directories and
  never processed at all. It was recorded while the backup was being made.
  Three acquisitions do not focus: two have a zero-byte `.raw_par` and one
  file is truncated.

The same tree settles a negative worth recording: **there is no GPRI data
from late August or September 2019**, though the field plan for that summer
anticipated a second trip. Nothing later than 2019-07-21 exists in any copy.

Two archive quirks turned up while focusing these and `20160826`, and
`gpri_tools.focus.find_raw` now handles both: the 2016 archive was written from a
Mac and carries `._`-prefixed AppleDouble stubs beside the real files (a 4 KB
"parameter file" with no `time_start`), and seven of its `.raw`/`.raw_par`
pairs are zero bytes. The stubs are skipped — they carry the same timestamps
as the files they shadow, so they change no count — and the empty files are
reported as failures; with one further file truncated, the campaign keeps its
other 44 acquisitions.

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
   at the same hours (`examples/baker_seasons.py`, [`baker.md`](baker.md) "Eight campaigns
   on one clock").
4. `20170913` (14.5 h) and `20180709` (6.9 h) fall short of a cycle even
   fully processed. They are useful for rates, not for diurnal phase, and
   the scripts fit rates on them (`MIN_CYCLES` in `gpri_tools.diurnal` refuses the
   harmonic). Both are focused (`gpri focus`) and run through the chain.
   What focusing them showed:
   - `20170913` was acquired on **2017-09-15** (05:57–20:29 UTC), not the
     13th, on one geometry (446 lines from −27.96°).
   - `20180709` holds 203 raw acquisitions on 2018-07-10, all of which
     focus, but only 197 are a campaign: two 02:36 UTC scans on the 2017
     geometry and four 12:08 scans sweeping −88° to +1° are set-up, and the
     campaign proper runs 13:35–20:30 UTC from −42.96° — 6.9 h, not the
     17.9 h between first and last file. `gpri focus` writes all 203;
     the set-up scans are moved out of `slc/` (`slc_setup/`) and the tabs
     regenerated before the chain runs. During its first 4.8 h the mount
     turned 5.1° (`gpri coregister`, [`baker.md`](baker.md) "Did the tripod hold?"); the
     recorded offsets put every epoch on the stable block's grid.
5. `20180808` and `20190719`, the two campaigns off the backup: both are
   two-cycle records and both go through the chain unchanged
   (`bin/run_scene.sh 20180808`, `bin/run_scene.sh 20190719`). With
   `20170827` they make three campaigns in three years that can be asked
   whether the diurnal repeats (`examples/baker_composite.py`, [`baker.md`](baker.md) "Does
   it repeat between years?").
6. `20170803_full`: the August 2017 day refocused from its raw archive. The
   GAMMA scene ships a `diff0`, and `bin/run_scene.sh` skips co-registration
   and heading for such a scene, so only the refocused copy has an
   azimuth-offset sidecar and a heading measured from eight SLCs. The two
   agree on the day's rate to 0.3 m/yr.
7. `20160826_full`: the oldest campaign that survives at all, refocused from
   its raw archive. The directory holds 57 distinct acquisition timestamps —
   what the survey tool counts, and what the table above reports — but only
   52 of them have a `.raw` file at all; the other five
   (`20160826_184803`, `20160826_213325`, `20160826_215518`,
   `20160826_233239`, `20160827_000004`) left only a `.raw.log`. Of the 52,
   seven are zero bytes and one is truncated, so 44 focus, giving 3.7 h at
   5-minute cadence. GAMMA processed the same day into **both** a
   single-reference and a chain network, and the closure triangles those two
   make are still its main value: 3.7 h is far short of a cycle, so the
   refocused scene carries rates and a noise floor, not diurnal phase.

## Scan headings

`GPRI_scan_heading` is 0.0 in every parameter file. `gpri heading` measures
it from the Copernicus DEM (`GPRI_DEM` in `site.env`; the tile
`Copernicus_DSM_COG_10_N48_00_W122_00_DEM` from the public
`copernicus-dem-30m` bucket covers the swath) and `bin/run_scene.sh` runs it
before any map is drawn:

| scene | heading (° true) | offset span (lines) | what the mount did |
|---|---:|---:|---|
| `20170713_full` | 111.38 | 0.44 (0.09°) | 0.08° clockwise through the night, back within an hour of sunrise |
| `20170803` | 107.42 | — | GAMMA's diff0; not co-registered |
| `20170827` | 100.13 | 0.26 (0.05°) | two 0.02° steps the first night, then a 0.03° daytime swing on both days |
| `20170913` | 108.38 | 0.10 (0.02°) | a 0.015° step at sunrise |
| `20180709` | 122.80 | 25.6 (5.1°) | 1°/h anticlockwise for 4.8 h, then held |
| `20170803_full` | 107.46 | 0.27 (0.05°) | steady; the refocused copy of the scene above |
| `20180808` | 124.86 | 11.7 (2.33°) | 2.2° in the first six hours, then held to 0.02° for 35 h |
| `20190719` | 109.70 | 22.4 (4.47°) | 4.4° in the first six hours, then held to 0.06° for 40 h |
| `20160826_full` | 114.55 | 0.30 (0.03°) | steady over its 3.7 h |

Three campaigns now show the same shape: a mount that settles by degrees
over the first hours after set-up and then holds to hundredths of a degree
for the rest of the record. On `20180709` that settling took 4.8 h of a 6.9 h
campaign, which is why it mattered; on `20180808` and `20190719` it is over
within the first six hours of a two-day record.

The offsets are measured against the last SLC of the campaign
(`gpri coregister --write` → `azimuth_offsets.json`, applied on read), so
every campaign sits on the grid of its final acquisition. The sub-line
drifts of the 2017 campaigns are thermal — they keep the sun's hours — and
are applied all the same; only the 2018 and 2019 mounts needed them.

## Practical notes

- Some project trees on the shared storage are not group-readable; if a
  campaign is missing from the survey output, check permissions with whoever
  administers the storage.
- Listing large acquisition directories over a network mount is slow; the
  survey tool bounds every walk and reports what it could not read rather
  than silently omitting it.
