# GPRI

Scripts and figures for processing and understanding GAMMA terrestrial radar
interferometric data.

This repository turns GAMMA's interferogram products into **line-of-sight
displacement time series on a map**, for GPRI-II ground-based radar. It reads
GAMMA's `.diff` / `.cc` rasters and `SLC_tab` / `itab` tables directly and does
not need the GAMMA binaries. It also focuses the instrument's raw FMCW sweeps
into SLCs itself (`gpri focus`, a port of GAMMA's `gpri2_proc.py` that
reproduces its output to float32 rounding), so campaigns that were never
processed are usable too.

![LOS displacement, north side of Mount Baker](docs/figures/04_displacement.png)

*LOS displacement over 6.7 hours on the north side of Mount Baker, from 200
consecutive BakerBend1 interferograms, projected to a local stereographic
frame. Backdrop is mean backscatter; areas below coherence 0.5 are masked —
beyond about 8 km the beam is in shadow behind the mountain.*

## What it does

```
focus/         raw FMCW sweeps -> SLCs, as GAMMA's gpri2_proc.py does it
gamma/         read GAMMA parameter files and binary rasters
network/       epochs, pairs, SBAS design matrices, closure triplets
stack/         patch-wise access to a whole diff0 directory (50 GB, memory-mapped),
               or the same interface formed on demand from SLCs (either antenna,
               any lag set, any multilook)
covariance/    sample coherence matrices
phaselink/     EVD, eigenSAR, EMI and exact ML phase linking
atmosphere/    range-dependent refractivity screens, estimated on wrapped phase
refractivity/  the same screens from meteorology, and per-epoch N
closure/       closure-phase bias estimation and correction
psinterp/      PS-interpolation unwrapping over decorrelated ground
timeseries/    network inversion, stacking, LOS displacement
diurnal/       harmonic analysis, and telling ice from atmosphere
geocode/       polar radar geometry to a local stereographic map frame
plot/          figures, in radar and map geometry
```

### Phase linking

`gpri.phaselink` fits one phase per epoch to the whole *N* × *N* coherence
matrix rather than reading each pair independently:

- **`evd`** — principal eigenvector (CAESAR). Cheap; optimal only when every
  pair is equally coherent.
- **`eigensar`** — EVD hardened for low PS/DS density: coherence floor,
  shrinkage toward the identity, inverse-iteration refinement, and an
  eigen-gap test that returns NaN rather than confident nonsense where the
  rank-one model is not supported.
- **`emi`** — Ansari et al.'s closed-form ML relaxation. Much better than EVD
  when coherence varies across pairs.
- **`mle`** — the exact ML / phase-triangulation solution by coordinate
  descent from an EMI start. Monotone by construction.

### Diurnal signals — the point of the experiment

The BakerBend1 campaigns were run to catch **sub-daily velocity and uplift
variation driven by water pressure in the subglacial drainage system**.
`20170803` is 723 acquisitions on a 2-minute cadence spanning 24.18 hours: one
complete diurnal cycle, sampled 723 times.

`gpri.diurnal` fits secular rate plus harmonics per pixel and reports amplitude
and the **hour of peak**, which is the diagnostic quantity — it says how long
the bed takes to respond to surface melt. It refuses records shorter than one
cycle outright, because over less than a period the amplitude and the secular
rate are not separable and a number returned there would be meaningless.

The hard part is that **the atmosphere is diurnal too**. Temperature and
humidity on a mountain flank cycle at exactly the period being looked for, and
roughly in phase with it — melt and warming peak together. Before correction
the atmospheric diurnal is one to two orders of magnitude larger than the
glaciological one. So the module ships three tests, and a claim should survive
all three:

1. **`range_dependence`** — the sharp one. Residual refractivity is *linear in
   slant range*; ice motion has no reason to correlate with distance from a
   tripod. A diurnal amplitude that grows with range is atmosphere.
2. **`atmospheric_coherence`** — regress each pixel against the independently
   estimated per-epoch refractivity series.
3. **`stable_ground_null`** — run the same fit on bedrock, which is not moving.
   That amplitude is the error floor; a diurnal on ice below it is not a
   detection.

And before any of that, **reference the series to stable ground**
(`gpri.timeseries.reference_to_stable`). Running the full 722-pair day without
it produced a clean 27.8 mm diurnal on ice — and a 33.0 mm one on bedrock, at
the same phase. An interferogram fixes phase only up to an additive constant,
so integrating the network accumulates 722 arbitrary offsets into a scene-wide
drift that is smooth, coherent, and diurnal. The range-dependence test does not
catch it (r = −0.015) because it is not range-dependent, it is *constant*. Only
the bedrock null caught it. Hold reference pixels out of that null, or the test
is circular.

![The artefact: bedrock and ice share one diurnal curve](docs/figures/08a_diurnal_unreferenced.png)

*What an unreferenced series looks like. Bottom right is the tell: the
bedrock null (red) traces the same curve as the ice (blue), offset by a
constant. Bedrock is not moving. Bottom left shows why the range test misses
it — the artefact is flat in range, not sloped.*

And a geometry limit worth stating before anyone reads uplift off a LOS series:
at a beam elevation of 10°, LOS sensitivity to vertical motion is `sin(10°) =
0.17` against `cos(10°) = 0.98` for horizontal. **A tripod radar is a
horizontal-motion instrument, nearly blind to uplift, and one line of sight
cannot separate the two at all.** `vertical_sensitivity` and `decompose_los`
make that explicit.

### What the 20170803 day actually shows

Running the full 722 pairs, referenced to bedrock (`examples/baker_diurnal.py`):

| | unreferenced | referenced |
|---|---:|---:|
| ice diurnal amplitude (median) | 27.8 mm | **17.9 mm** |
| held-out bedrock null (median) | 33.0 mm | **11.3 mm** |
| bedrock phase concentration | 0.919 | **0.141** |
| variance explained by refractivity | 70.3 % | **41.2 %** |
| amplitude vs slant range | r = −0.015 | r = −0.164 |
| **ice / bedrock ratio** | **0.84** | **1.59** |

Referencing removed a 99.4 mm peak-to-peak common mode and did what it should:
the bedrock phase concentration collapsed from 0.919 (every rock pixel peaking
at the same hour — a systematic error) to 0.141 (incoherent, as unmoving ground
should be), and the peak-hour map went from one uniform colour to real spatial
structure.

**The honest verdict is still negative.** Ice diurnal amplitude is 1.59× the
bedrock error floor — suggestive, but under the 2× bar the script applies, and
41 % of the remaining variance is still explained by the refractivity series
alone. This is not a detection of subglacial hydrology, and the pipeline says
so rather than reporting the 17.9 mm on its own.

That is close enough to the floor to be worth pursuing with better data, which
is the case for the next section.

### The reference audit: coherence is not stationarity

The stable-ground reference was originally chosen by coherence alone — and
slowly moving ice stays coherent at a 2-minute pair spacing. Auditing the
mask against the **Randolph Glacier Inventory** (`gpri.glaciers`,
`examples/baker_rgi.py`) found that **62.8 % of the coherence-chosen
"bedrock" was on RGI glacier** — Coleman, Roosevelt and Mazama surfaces were
in the reference, so every earlier bedrock-referenced product was tied to
moving ground: real motion subtracted from the maps, an artificial signal
pushed onto rock, and a "bedrock null" that was mostly ice.

![Reference audit against RGI](docs/figures/16_rgi_reference_20170803.png)

Correcting the masks (reference = coherent ∧ outside buffered RGI outlines;
ice = RGI-defined) changes the diurnal verdict materially — see the
single-step least-squares numbers below. The overlay also doubles as an
independent check on the unsurveyed scan heading: at 105° the inventory
outlines land on the right backscatter features.

### Does the pipeline actually recover ice motion?

The sharpest check available, on the corrected 20170803 day — cumulative LOS
displacement after 24.2 h, RGI-defined ice against **held-out** rock (rock the
corrections never saw):

| | pixels | median | mean | p16–p84 |
|---|---:|---:|---:|---:|
| RGI ice | 26,401 | **+62.2 mm** | +80.3 mm | −19 to +189 |
| held-out rock | 4,090 | **−2.5 mm** | −0.1 mm | −29 to +27 |

Rock sits at zero — as unmoving ground must, and it was never used to fit the
corrections — while the ice moves 62 mm toward the radar over the day. The
correlation between ice displacement and slant range is **−0.07**, so this is
not the epoch screens extrapolating a ramp over the ice; it is spatially
organised motion where the inventory says there is a glacier. A day of
~60 mm LOS is the right order for Coleman and Roosevelt flow projected onto a
near-horizontal look direction.

This is the secular signal, and it is the part the old ice-contaminated
reference was actively destroying. The diurnal remains the harder question
below.

### Single-step least squares, after Ohenhen et al.

`gpri.pairlsq` fits the temporal model — secular rate + diurnal harmonics (+
optional covariates such as the refractivity series) — **directly to the pair
observations** by weighted least squares, in the style of Ohenhen et al.'s
subsidence mapping, with formal per-pixel uncertainties. Three things the
integrate-then-fit pipeline cannot do:

- the pair errors are independent, so this is the correctly *whitened*
  problem (integration turns them into a random walk, and OLS on a random
  walk both loses efficiency and reports optimistic error bars);
- per-pair coherence weights enter naturally — the measured win is ~2× lower
  amplitude error under uneven pair quality;
- every amplitude map comes with a σ map, so "diurnal detection" can mean
  `amplitude > 3σ` per pixel, and held-out bedrock gives the real false-alarm
  rate of the whole chain for free.

The constant cancels in the differencing, so no reference epoch is needed and
disconnected networks still constrain rate and harmonics. And the error bars
say something blunt worth hearing: a short-pair chain sees only `A·ω·Δt` of a
smooth harmonic per pair, so most of the diurnal sensitivity lives in the
longer combinations a daisy chain does not have — the same conclusion the
campaign inventory reached about 20170827 from the other side.
`examples/baker_pairlsq.py` runs the comparison on real data
(`docs/figures/15_pairlsq_20170803.png`). With coherence-only masks the
result was null (ice/bedrock ratio 2.1, 3.6 % of ice above 3σ vs a 1.2 %
false-alarm rate). **With the RGI-corrected masks** (`--rgi`) the picture
sharpens:

| | coherence-only | RGI-corrected |
|---|---:|---:|
| ice median amplitude | 16.1 mm | 16.6 mm |
| held-out bedrock amplitude | 7.7 mm | 6.9 mm |
| ice/bedrock ratio | 2.1 | **2.42** |
| ice above 3σ | 3.6 % | **7.7 %** |
| bedrock false-alarm rate | 1.2 % | **0.8 %** |

The bedrock false-alarm rate falls to ~the value the error bars predict —
the uncertainty model is close to calibrated once the null is on actual rock
— and the ice contrast clears the 2× bar for the first time, at ~10× the
bedrock detection rate. Projecting the refractivity series out inside the
fit barely moves the ice amplitude (16.6 → 16.5 mm): what remains on
RGI-defined ice is not refractivity-shaped. Still a population-level
contrast rather than a per-pixel detection (median SNR 1.54), and one cycle
cannot show the signal repeats — that remains 20170827's job.

### Which campaign to use

[`docs/campaigns.md`](docs/campaigns.md) inventories all 25 GPRI campaigns on
cold storage (`bin/survey_campaigns.py` regenerates it). The short version:

| campaign | stage | span | cycles |
|---|---|---:|---:|
| `20170827` | raw → **slc** (`gpri focus`) | **44.9 h** | **1.87** |
| `20170803` | diff | 24.1 h | 1.01 |
| `20170713_full` | raw → slc (`gpri focus`) | 23.9 h | 0.996 |
| `20170713` | diff | 21.8 h | 0.91 |

**`20170827` is the dataset the experiment deserves** — 44.9 hours, 1335
acquisitions at 2-minute cadence, nearly two full cycles, and an *i*→*i*+3
network that actually has closure. It was left as 582 GB of raw sweeps;
`gpri focus` turns those into SLCs for both antennas (about 90 minutes,
I/O-bound). Two things about it to know before using it: the scan was widened
from −30..50° to −30..60° after the first 197 acquisitions, so the SLCs come
in two lengths (396 and 446 lines — `SlcPairStack` crops every pair to the
common leading block, which starts at the same azimuth), and there are two
gaps in the cadence, 19 minutes at that geometry change and 8 minutes a day
later. `20170803` — one cycle, processed by GAMMA — remains the default scene.

`20170713` as GAMMA shipped it stops at 21.8 h, and the harmonic fits refuse
it — over 0.91 of a cycle amplitude and rate are not separable. Its raw
archive runs to 23.9 h (271 acquisitions at 5-minute cadence), five minutes
short of a day, and `gpri focus` writes that as the `20170713_full` scene.
The fits accept it: `MIN_CYCLES = 0.98` in `gpri.diurnal`, because the
rate/harmonic correlation has no cliff at exactly one period (0.78 at 1.00
cycles, 0.80 at 0.98, against 0.87 at 0.75 and 0.99 at half a cycle for an
epoch-domain fit; near zero either side of one cycle in the pair domain).

### Two days: does the diurnal repeat?

`20170827` is now focused and run through the whole chain (`bin/run_scene.sh
20170827`: aps ladder, RGI audit, pair-domain fit, repeat test, four movies,
two-antenna replicate and closure, both antennas, about two hours after the
88-minute focus). The RGI audit drops 77 % of the coherence-only reference as
glacier (17,853 of 23,094 px); the campaign's coherence is lower than
20170803's (median 0.28 at 5×5 looks), so the true-rock reference is 5,241
px against 8,180. On that reference the atmospheric ladder repeats its
20170803 shape a third and fourth time — per-pair screens hurt (B 122 % of
A), turbulence recovers most of it (D 107 %), and plain referencing at 36.9
mm over 44.9 h is what 20170803's 29.1 mm over 24.2 h becomes under √t
growth (39.6 mm predicted) — see
[`docs/atmosphere.md`](docs/atmosphere.md).

The pair-domain fit over both days (`15_pairlsq_20170827.png`) is weaker
than 20170803's: ice median amplitude 11.2 mm against held-out rock 6.2 mm
(ratio 1.8), 2.3 % of ice above SNR 3 against a 1.0 % false-alarm rate. The
two antennas replicate each other (11.2 / 11.1 mm; 102 ice pixels pass SNR 3
in both, 6× chance, peak times within 2 h for 83 % of them; 0.08 % of rock
survives the same test — `17_antennas_20170827.png`) and the measured noise
floor is 21.5 mm single-antenna against 23.0 mm common-mode. Per pixel,
then, a two-day daisy chain at single look does not detect the diurnal any
better than one day did. The question the second day was bought for is
answered at the population level instead.

**`examples/baker_repeat.py`** fits the same corrected observations three
times — pairs inside the first 24 h, inside the last 24 h, and all of them —
and compares the mean of the per-pixel phasors `a + ib` over all RGI ice
with the same mean over held-out bedrock
([`18_repeat_20170827.png`](docs/figures/18_repeat_20170827.png)):

| fit | ice mean phasor | peak (UTC) | held-out bedrock | ice / rock |
|---|---:|---:|---:|---:|
| day 1 | 4.5 mm | 02:00 | 0.11 mm | 40 |
| day 2 | 11.0 mm | 20:24 | 0.91 mm | 12 |
| both | 6.3 mm | 21:48 | 0.34 mm | 18 |

The lower antenna gives 4.5 / 10.8 / 6.1 mm at the same hours. The glacier
population has a diurnal term on both days that the bedrock population does
not; but it is 2.5× larger on the second day and peaks 5.6 h earlier, and
the two days' phasor maps correlate at only 0.23 across the ice — no more
than the 0.25 that the bedrock's residuals manage. Read as a harmonic, the
signal does not repeat.

**`examples/baker_population.py`** shows why the harmonic is the wrong
basis. It plots the median of every pixel's departure from its own linear
trend, over the ice and over held-out rock, against a UTC clock
([`19_population_20170827.png`](docs/figures/19_population_20170827.png)).
The ice median is a **night-time trough with a sharp morning recovery** on
both days: behind trend from about 05 UTC (22:00 PDT), lowest at 08–13 UTC,
back above trend by 15 UTC (08:00 PDT), highest at 02–03 UTC (19:00–20:00
PDT). The trough is −9 mm on the first night and −20 mm on the second, and
the second morning's rise is a step of some 25 mm in two hours — which a
24 h sinusoid can only render as a larger amplitude at an earlier phase,
exactly what the table above reports. Over the same 45 hours the held-out
bedrock median stays within ±1.5 mm (RMS 0.6 mm against the ice's 7.5 mm,
correlation −0.23); the median ice rate is 1.2 mm/h LOS, the rock's 0.02.

So the timing repeats — the trough and the morning recovery fall at the
same hours on consecutive days — while the amplitude does not, and the
waveform is nothing like a sinusoid. That is what a melt-forced response
looks like (input stops at dusk, the system drains overnight, the morning
speed-up comes with the sun), and it is not what a residual atmosphere on
the control looks like. The caveat is the one attached to every ice result
here: the control is rock that sits at the ranges and heights where rock
is, and the corrections are extrapolated from there onto ice that is higher
and farther, so a stratified atmospheric term the rock cannot see is not
excluded by the rock being flat. The two antennas cannot help with that —
they share the atmosphere — and the next real control is meteorology.

### Three campaigns on one clock

`20170713_full` — the July archive refocused to its full 23.9 h — goes
through the same chain (`bin/run_scene.sh 20170713_full`, 271 epochs at
5-minute cadence, both antennas, half an hour end to end). Its numbers sit
beside the two August campaigns':

| | `20170713_full` | `20170803` | `20170827` |
|---|---:|---:|---:|
| span, pairs, cadence | 23.9 h, 270, 5 min | 24.2 h, 722, 2 min | 44.9 h, 1334, 2 min |
| coherence-only reference on glacier (RGI) | 48 % | 62 % | 77 % |
| held-out rock (px) | 3,437 | 4,090 | 2,618 |
| ladder A → D, held-out rock | 24.3 → **21.5 mm** (88 %) | 29.1 → 28.2 mm (97 %) | 36.9 → 39.5 mm (107 %) |
| pair-domain diurnal, ice / rock | 10.2 / 6.8 mm (1.5) | 16.6 / 6.9 mm (2.4) | 11.2 / 6.2 mm (1.8) |
| ice above SNR 3 / rock false alarms | 1.6 % / 1.0 % | 7.7 % / 0.8 % | 2.3 % / 1.0 % |
| SNR 3 in both antennas, ice / rock | 0.3 % / 0.35 % | 1.9 % / 0.07 % | 0.3 % / 0.08 % |
| single-antenna noise / common-mode, rock | 12.8 / 14.1 mm | 16.2 / 17.0 mm | 21.5 / 23.0 mm |
| median LOS rate, ice / rock | −0.5 / +0.04 mm/h | +1.5 / −0.03 mm/h | +1.2 / −0.02 mm/h |
| trend-anomaly RMS, ice / rock | 2.7 / 0.3 mm | 8.2 / 0.7 mm | 7.5 / 0.6 mm |

July is the quiet one on every line. The ladder is the only one of the three
that gains on true rock (turbulence takes 12 % off, the per-pair screens are
neutral); the per-pixel diurnal is a null — the ice/rock ratio is 1.5, and
the replication test that on 20170827 keeps 0.08 % of rock keeps 0.35 % in
July, the same rate as the ice's 0.3 %; and the coherent ice population has
no net line-of-sight motion (median −0.5 mm/h, 10th–90th percentiles −2.9 to
+2.0 mm/h, against −2.5 to +8.6 on 20170803, with the rock's own spread ±0.9
on both days). Whether that is a glacier that moves less in July or a July
in which the coherent "ice" pixels are a different, more marginal set — the
coherence-only reference is 48 % glacier in July and 77 % in late August, so
the ice that holds coherence changes with the season — the single-look data
cannot say.

**`examples/baker_seasons.py`** puts what the population series *do* share
on one figure: every processed UTC day, ice median departure from trend
against the hour, with the held-out bedrock underneath
([`20_seasons.png`](docs/figures/20_seasons.png)):

| UTC day | span | trough | depth | back above trend | rock RMS |
|---|---:|---:|---:|---:|---:|
| 2017-07-14 | 00:18–19:42 | 06:48 | −5.6 mm | 09:12 | 0.29 mm |
| 2017-08-04 | 00:00–22:30 | 11:36 | −16.8 mm | 13:06 | 0.68 mm |
| 2017-08-28 | 00:00–24:00 | 14:06 | −10.5 mm | 15:12 | 0.67 mm |
| 2017-08-29 | 00:00–20:42 | 08:12 | −21.3 mm | 13:00 | 0.46 mm |

Hourly-binned, the ice medians of 07-14, 08-04 and 08-29 correlate at
**0.72, 0.70 and 0.78** with each other — three days six weeks apart, in
two campaigns focused from raw, with the same night-time trough (falling
behind trend from 04–05 UTC, 21:00–22:00 PDT, deepest 06–12 UTC, back above
trend by 09–13 UTC) — and 08-28 correlates with none of them (−0.08, 0.04,
0.02): its night is flat and its trough comes at 12–15 UTC, after sunrise.
The bedrock's hourly medians correlate between the same days at −0.61 to
+0.33 with no pattern, and the lower antenna reproduces every entry (0.69 /
0.67 / 0.80; −0.15 / 0.05 / 0.02). So the timing repeats on three days of
four, the amplitude does not repeat at all (−5.6 to −21 mm), and the one
day that breaks the pattern is the first of the two-day campaign, whose
anomaly is measured against a 45 h trend that the second day's larger
swing helps set.

Two things in the rock panel deserve stating. On 08-04 the bedrock median is
a mirror image of the ice at one-fifteenth the scale (correlation −0.90
over the day; +1 mm while the ice is at −15, −2 mm in the last hour while
the ice climbs 20 mm) — the signature of a correction whose residual has
opposite sign on rock and on the higher, farther ice, and a reason to
distrust that day's amplitude more than its hours. On the other days the
correlation is −0.23, +0.09 and +0.12, and the rock stays within ±1.2 mm
throughout. The night-time trough is the most repeatable thing this data
set has produced; what it is made of — ice, or an atmosphere stratified in
a way rock at rock heights cannot register — is the question the next
campaign has to be designed to answer, with meteorology on the glacier.

### Movies of the deformation field

`examples/baker_movie.py` renders the corrected LOS field as an MP4 in the
map frame — backscatter backdrop, real UTC clock, every processed campaign:

- [`14_los_movie_20170803.mp4`](docs/figures/14_los_movie_20170803.mp4) —
  cumulative displacement through the 24.2 h day (723 frames, 30 s)
- [`14_los_movie_rate2h_20170803.mp4`](docs/figures/14_los_movie_rate2h_20170803.mp4)
  — motion over a trailing 2 h window, the right view for a diurnal signal:
  unlike the cumulative view its noise is bounded instead of growing as √t
- the same pair for `20170713`
- [`14_los_movie_anommean_20170803.mp4`](docs/figures/14_los_movie_anommean_20170803.mp4)
  and [`14_los_movie_anomtrend_20170803.mp4`](docs/figures/14_los_movie_anomtrend_20170803.mp4)
  — each frame as an **anomaly**: the pixel's departure from its day mean
  (`--anomaly mean`), or from its linear trend (`--anomaly trend`), which is
  what a diurnal response looks like once steady flow is taken out. These
  carry a second panel with the **reference displacement rate** the anomaly
  is read against — the per-pixel linear LOS rate over the day, in mm/h —
  and a time strip with the median anomaly over the moving pixels, its
  interquartile band, and a cursor at the current frame. Same two views for
  `20170713`.
- the same four for `20170827`
  ([cumulative](docs/figures/14_los_movie_20170827.mp4),
  [2 h rate](docs/figures/14_los_movie_rate2h_20170827.mp4),
  [mean anomaly](docs/figures/14_los_movie_anommean_20170827.mp4),
  [trend anomaly](docs/figures/14_los_movie_anomtrend_20170827.mp4)) —
  1335 frames over 44.9 h; the anomaly views' time strip is where the
  night-time trough of the section above is easiest to see.
- and for `20170713_full`
  ([cumulative](docs/figures/14_los_movie_20170713_full.mp4),
  [2 h rate](docs/figures/14_los_movie_rate2h_20170713_full.mp4),
  [mean anomaly](docs/figures/14_los_movie_anommean_20170713_full.mp4),
  [trend anomaly](docs/figures/14_los_movie_anomtrend_20170713_full.mp4)),
  271 frames at 5-minute cadence.

Corrections are the validated recipe (reference + drift removal + turbulence,
no per-pair screens), referenced to **true rock** — coherent pixels outside
the RGI outlines (`--rgi`). That matters visually as much as statistically:
with the old coherence-only reference the corrections were partly subtracting
glacier motion, and the field looked patchy and two-signed. Tied to rock, a
coherent toward-radar lobe appears over Coleman and Roosevelt, reaching
~25 mm per 2 h in the rate view.

Two caveats stay attached. The true-rock reference is smaller (8,180 px
against 22,046 at this decimation), so the epoch screens extrapolate further
over the ice than before. And display smoothing — a rolling temporal mean and
a light spatial Gaussian — is printed on every frame rather than hidden;
without it a per-pixel movie of single-look data is snow.

### Two antennas, one day: the replicate

The GPRI-II receives on two antennas 25 cm apart on the same mast, sampled in
the same sweep, and GAMMA only ever processed the upper one. `SlcPairStack`
forms the lower antenna's daisy chain from its SLCs, and
`examples/baker_antennas.py` runs the identical chain — RGI reference, held-out
split, corrections, pair-domain diurnal fit — on both
([`17_antennas_20170803.png`](docs/figures/17_antennas_20170803.png)):

| | upper | lower |
|---|---:|---:|
| ladder A / D, held-out rock | 29.1 / 28.2 mm | 30.5 / 29.3 mm |
| ice median diurnal amplitude | 16.6 mm | 16.5 mm |
| ice above SNR 3 | 7.7 % | 7.0 % |
| held-out rock above SNR 3 (false alarms) | 0.8 % | 0.9 % |

Every number replicates, which is the first time this pipeline has had a
replicate at all. Two things follow that a single antenna could never give:

- **A measured noise floor.** `upper − lower` cancels deformation,
  atmosphere and reference error alike. On held-out rock
  RMS(u − l)/√2 = **16.2 mm** over the day, against 23.5 mm total, so
  17.0 mm of the rock residual is *common-mode* — shared error the two
  channels cannot see, not measurement noise. (Surface decorrelation is
  common to both antennas too, so 16 mm is a lower bound on single-antenna
  noise and 17 mm an upper bound on atmosphere plus reference.)
- **A replication test for the diurnal detections.** 494 ice pixels
  (1.9 %) pass SNR 3 in *both* antennas — 3.5× the 0.5 % that two
  independent chance detections would give — and their peak times agree:
  median difference −0.1 h, interquartile range 2.0 h, 81 % within 2 h,
  where independent noise would spread them uniformly over 24 h. On rock,
  0.07 % survive the same test.

Averaging the two channels raises the ice median SNR from 1.54 to 1.73 — not
the √2 = 2.18 of independent noise, because the rest is common-mode. Which
is also the limit of the replicate: the antennas share the atmosphere, so
agreement between them is evidence against phase noise, not against
atmosphere. The held-out-bedrock false-alarm rate remains the atmosphere
control.

### Atmospheric correction, validated on held-out bedrock

`gpri.aps` adds three corrections on top of the per-pair screens, and
[`docs/atmosphere.md`](docs/atmosphere.md) scores the whole ladder on bedrock
that no correction ever saw. The measured result, on both processed days:

| stage | 20170713 (21.8 h) | 20170803 (24.2 h) |
|---|---:|---:|
| A reference only | 25.9 mm | 47.1 mm |
| B + per-pair screens | 26.3 mm | 49.2 mm |
| C + drift removal (`epoch_screen_correction`) | 26.0 mm | 46.8 mm |
| D + turbulence (`turbulence_screen`) | **20.9 mm** | **30.1 mm** |

**Caveat, post-RGI audit:** these tables were scored against a reference
later shown to be 63 % glacier; with a true-rock reference (see below) plain
referencing already achieves what the full ladder appeared to, and the
turbulence gain shrinks to ~3 % — `docs/atmosphere.md` carries the corrected
table. The methodological findings survive:

Three findings worth stating plainly. **Per-pair screens do not improve the
integrated series** — their fit noise integrates into a random walk roughly as
large as the atmosphere they remove (about half the "drift" on 20170713 was
manufactured by the correction itself). **The non-parametric turbulence screen
is the workhorse**: 19–36 % RMS reduction, from nothing but a normalised
convolution of each epoch's residual over stable ground. And **what remains
grows as √t and is spatially uncorrelated** — single-look phase noise, not
atmosphere, so the next lever is multilooking/phase linking, not more screens.

There is deliberately no stratified (height-dependent) term: with one beam
elevation and no DEM, height is exactly linear in slant range and
unidentifiable from the mixing ramp.

Closure phase is also now measured on real data: 20160826's merged
single-reference + chain networks give 25 triangles
(`examples/baker_closure.py`). On 1-look pixels closure is identically zero —
multilooking creates the bias — and after a 3×15 boxcar the fitted `b(dt)`
grows to ~0.08 rad (~0.1 mm) at 3 h with the classic fading-signal shape.

On the day the analysis actually uses, the answer is different and cleaner.
20170803's shipped network is a daisy chain with no triangles, so the pairs
are formed from the SLCs instead: *i*→*i*+1..3 gives 2161 triangles, and
adding the 1, 2, 3, 6 and 12 h baselines (`--lags 1 2 3 30 60 90 180 360`)
gives 4996. In both cases the closure rms is ~1 rad (0.955 and 1.102 rad
before correction) and the fitted `b(dt)` removes **none of it** — 0.953 and
1.097 rad after, a 0 % reduction. That closure is dominated by decorrelation
noise, not by a systematic short-baseline bias — and since the fitted bias
is velocity-blind by construction, there is nothing here for a closure
correction to change in the displacement chain, which applies none. The
two campaigns focused from raw say the same: *i*→*i*+1..3 on `20170827`
gives 3997 triangles at 0.997 rad closure rms, 0.996 after the fit;
`20170713_full` gives 805 at 0.983 → 0.969 rad (a 1 % reduction). Four
days, ~1 rad everywhere, nothing for `b(dt)` to take out.

### Methods after Ann Chen

- **PS interpolation** (`gpri.psinterp`), after Chen, Zebker & Knight (2015).
  Unwrap only at the persistent scatterers, interpolate that sparse reliable
  field across the scene, subtract it, and what is left is sub-fringe and needs
  no unwrapping at all. Recovers deformation over ground that decorrelated.
  The sparse unwrapper integrates along a Delaunay-based minimum spanning tree
  — *not* a k-nearest-neighbour graph, which fragments under GPRI's wildly
  anisotropic sampling (0.75 m in range against tens of metres in azimuth).
- **Closure-phase bias** (`gpri.closure`). Fits the systematic
  short-baseline bias `b(dt)` to the observed closure phases. It states its own
  limit: a bias linear in temporal baseline closes perfectly and is invisible
  to closure phase — and that is exactly a constant velocity, so **a closure
  correction can never validate a rate**. `BiasModel.velocity_blind` says so in
  the object.
- **Refractivity** (`gpri.refractivity`). Smith–Weintraub moist-air
  refractivity from pressure, temperature and humidity, and a per-epoch
  refractivity series inverted from the per-pair range ramps — the only
  independent check on an empirically estimated screen.

## Install

```bash
pip install -e '.[all]'      # numpy, scipy + pyproj, rasterio, matplotlib
pytest                       # 324 tests
```

Only `numpy` and `scipy` are required. `pyproj` and `rasterio` are needed for
geocoding and GeoTIFF output, `matplotlib` for figures; all three are imported
at point of use, so the core works without them.

## Use

```bash
S=$GPRI_SCENE_20170803          # set in site.env -- see site.env.example

gpri info       $S                              # what is in it, how coherent
gpri screens    $S                              # per-pair refractivity screens
gpri velocity   $S -o vel.npz --geotiff --heading 105
gpri timeseries $S -o ts.npz  --method wls
gpri phaselink  $S -o pl.npz  --method eigensar
gpri closure    $S                              # bias against temporal baseline
gpri unwrap     $S --pair 0 --min-coherence 0.6 -o unw.npz
gpri geocode    $S vel.npz --field velocity --heading 105

# a raw campaign -> a scene directory (slc/, SLCu_tab, SLCl_tab), both antennas
gpri focus      $GPRI_RAW_20170827 $GPRI_SCENE_20170827 --workers 6

bin/survey_campaigns.py                         # what data exists, and how long
```

`gpri focus` defaults to the BakerBend recipe (`-d 5 -z 300 -r 300 -k 3.84`
in `gpri2_proc.py` terms: presum 5 sweeps, 300-sample Hann taper, 300 m
minimum range, Kaiser β 3.84). Point it at a campaign directory and it finds
every `.raw` in it and its `raw*/` subdirectories; `--raw-list` restricts it
to the campaign's own `RAW_list`. Output is byte-compatible with GAMMA's: the
`.slc.par` files are identical, the samples agree to float32 rounding
(max 2e-9 relative), and GAMMA's `multi_look` on our SLC reproduces its own
MLI to 4e-7.

```bash
```

```python
from gpri import DiffStack, RadarGeometry, geocode_image, stack_velocity
from gpri.timeseries import los_displacement

stack = DiffStack.from_directory(f"{S}/diff0", slc_tab=f"{S}/SLCu_tab")
rows, cols, ifg, cc = next(stack.patches(max_gib=2.0))
v = stack_velocity(los_displacement(np.angle(ifg), stack.wavelength),
                   stack.network, weights=cc)

geom = RadarGeometry(stack.par, heading=105.0)      # see the caveat below
v_map, transform = geocode_image(v, geom, spacing=25.0)

# the same interface, formed from SLCs: lower antenna, i->i+1..3, 3x15 looks
from gpri import SlcPairStack
lower = SlcPairStack.from_directory(f"{S}/slc", antenna="l",
                                    lags=(1, 2, 3), looks=(3, 15))
```

Reproduce the figures in `docs/figures/` (the scripts cache the decimated
day under `GPRI_WORK_ROOT`, so only the first one pays for the read):

```bash
python examples/baker_north_side.py --pairs 200 --decimate 8 --spacing 25
python examples/baker_diurnal.py --decimate 16        # full day + the three tests
python examples/baker_aps.py --scene 20170803 --decimate 16 --sigma 5 25 --rgi --screens-on-bedrock
python examples/baker_rgi.py --scene 20170803 --decimate 16
python examples/baker_pairlsq.py --scene 20170803 --decimate 16 --rgi
python examples/baker_movie.py --scene 20170803 --rgi                  # cumulative
python examples/baker_movie.py --scene 20170803 --rgi --rate-hours 2
python examples/baker_movie.py --scene 20170803 --rgi --anomaly mean   # + reference rate panel
python examples/baker_movie.py --scene 20170803 --rgi --anomaly trend
# the lower antenna: every script takes --antenna lower
python examples/baker_antennas.py --scene 20170803 --decimate 16 --rgi
# closure on the day the analysis uses: pairs formed from the SLCs
python examples/baker_closure.py --scene 20170803 --lags 1 2 3 30 60 90 180 360 --looks 3 15
# the two-day campaign, once `gpri focus` has written it: same scripts, --scene 20170827,
# plus the test only two cycles can make -- does the diurnal repeat?
python examples/baker_repeat.py --scene 20170827 --decimate 16 --rgi
python examples/baker_population.py --scene 20170827 --decimate 16 --rgi
# every processed day on one UTC clock (needs baker_population.py run per scene)
python examples/baker_seasons.py --scenes 20170713_full 20170803 20170827
```

`bin/run_scene.sh <scene> [upper|lower|both]` runs that whole chain for one
scene, both antennas side by side, logging each step under
`$GPRI_WORK_ROOT/<scene>/logs/`.

## The scan heading is not in the data

`gpri.geocode` maps the polar fan onto a local stereographic projection centred
on the radar. Everything it needs is in the parameter file except one number:

**`GPRI_scan_heading` is `0.0` in every BakerBend1 parameter file.** It was
never surveyed, and a heading of exactly zero would point the fan due north, at
nothing. `RadarGeometry` warns rather than accepting it quietly.

Two ways to fix it:

- `gpri.geocode.heading_from_tiepoint(par, lat, lon, row)` — one identifiable
  feature with known coordinates solves it exactly; several let you check it.
- Pass `heading=` from field notes.

`BAKERBEND1_HEADING = 105.0` is a **starting guess**, not a survey, derived
from the bearings to the north-side glaciers: the radar at 48.82132 N,
121.92018 W, 1252 m sees Baker's summit at bearing 122.5° and 9.2 km, Coleman
Glacier at 120.6°, Mazama at 107.4°, Colfax Peak at 129.9°. The 79° fan
(−27.96° to +51.05°) needs a heading near 105° to cover them, and
[`docs/figures/01_coverage.png`](docs/figures/01_coverage.png) confirms it
does. Half a degree of heading error is 80 m of position error at 9 km — it
does not touch the phase, but tie it to a real feature before publishing a map.

## Sign convention

Every phase follows GAMMA's `SLC_intf` convention: the interferogram for pair
`(i, j)` is `z_i * conj(z_j)` and carries phase `theta_i - theta_j`.
Displacement is reported **positive toward the radar**. See
`gpri.timeseries.los_displacement` for the derivation — it is the easiest thing
in InSAR to get backwards and the hardest to notice.

## The data

Machine-specific locations — data roots, scene directories, scratch — live in
`site.env` at the repository root, which is gitignored; copy
`site.env.example` and fill it in. Nothing in the repository names a host, a
mount point, or a storage layout.

The default scene, `20170803`, holds 723 SLCs for **each of the two receive
antennas** (95 GB) plus a `diff0/` with 722 upper-antenna interferograms and
matching coherence rasters — 396 × 22101 FCOMPLEX, 70 MB each, 50 GB for the
stack. Nothing here loads that: every raster is memory-mapped and read in
tiles.

Three things worth knowing about it:

- The `itab` is a **daisy chain** (1–2, 2–3, 3–4, …), so the network contains
  no closed triangles and `gpri closure` correctly refuses. `SlcPairStack`
  forms the *i*→*i*+2, *i*→*i*+3 interferograms from the SLCs on demand
  (`--lags 1 2 3`), which is how the closure figure for this day was made.
- GAMMA only processed the **upper** antenna. The lower antenna's SLCs are
  there, 25 cm below, sampled in the same sweeps; `SlcPairStack` (or any
  script's `--antenna lower`) runs the identical chain on them. Its products
  are exchangeable with GAMMA's: the phase of `s_i * conj(s_j)` matches the
  `.diff` to 2e-7 rad, and a 5 × 5 triangular-window coherence reproduces the
  `.cc` at correlation 0.998.
- The `.diff` files are **magnitude-normalised** — `abs(ifg)` is 0 dB
  everywhere. Backscatter for figure backdrops comes from the MLIs
  (`baker_mli_upper.ave`), on the identical grid.

## GAMMA

Nothing in this package calls GAMMA, but a GAMMA installation is useful for
cross-checks and is what `gpri focus` was validated against. The 2017-07-11
Linux distribution installs by unpacking under `/usr/local`; `config.sh` and
`bin/check_env.sh` discover `/usr/local/GAMMA_SOFTWARE-*` (or `$GAMMA_HOME`)
and put the `ISP`, `DIFF`, `LAT` and `DISP` binaries on the path. That
distribution has no `par_GPRI2_SLC`: GPRI raw processing is the Python 2
script `GPRI2-2/trunk/python/gpri2_proc.py`, which is what `gpri/focus.py`
ports (the geometry, squint correction, Kaiser window, range scaling and
`.slc.par` writer are the same, line for line, in Python 3 and numpy).

With GAMMA on the path, `bin/smoke_test.sh` runs its ISP chain
(`create_offset → SLC_intf → multi_look → cc_wave → rasmph_pwr`) on the first
pair of a scene. On SLCs focused by `gpri focus` the products agree with
`SlcPairStack` exactly as GAMMA's own archive did: interferogram phase to
2e-7 rad, 5 × 5 coherence at correlation 0.998. One thing that test taught:
`SLC_intf`'s azimuth common-band filter must be **off** for GPRI — with it on,
the phase of a rotating-antenna pair is scrambled to noise (rms 1.6 rad).

## License

MIT — see [`LICENSE`](LICENSE).
