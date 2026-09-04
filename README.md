# GPRI_tools

Tools for turning GAMMA / GPRI-II terrestrial radar interferometry into
**line-of-sight displacement time series on a map**.

The package reads GAMMA's `.diff` / `.cc` rasters and `SLC_tab` / `itab`
tables directly and does not need the GAMMA binaries. It also focuses the
instrument's raw FMCW sweeps into SLCs itself (`gpri focus`, a port of GAMMA's
`gpri2_proc.py` that reproduces its output to float32 rounding), so campaigns
that were never processed are usable too. Everything after that — atmospheric
screens, phase linking, network inversion, closure, geocoding, harmonic
analysis — lives in `gpri_tools/`, with a `gpri` command line over it.

**The worked example is Mount Baker**: eight GPRI-II campaigns between 2016
and 2019, processed end to end, written up in
[`docs/baker.md`](docs/baker.md). It is the only example here, and nothing in
`gpri_tools/` is specific to it.

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

`gpri_tools.phaselink` fits one phase per epoch to the whole *N* × *N* coherence
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

### Methods after Ann Chen

- **PS interpolation** (`gpri_tools.psinterp`), after Chen, Zebker & Knight (2015).
  Unwrap only at the persistent scatterers, interpolate that sparse reliable
  field across the scene, subtract it, and what is left is sub-fringe and needs
  no unwrapping at all. Recovers deformation over ground that decorrelated.
  The sparse unwrapper integrates along a Delaunay-based minimum spanning tree
  — *not* a k-nearest-neighbour graph, which fragments under GPRI's wildly
  anisotropic sampling (0.75 m in range against tens of metres in azimuth).
- **Closure-phase bias** (`gpri_tools.closure`). Fits the systematic
  short-baseline bias `b(dt)` to the observed closure phases. It states its own
  limit: a bias linear in temporal baseline closes perfectly and is invisible
  to closure phase — and that is exactly a constant velocity, so **a closure
  correction can never validate a rate**. `BiasModel.velocity_blind` says so in
  the object.
- **Refractivity** (`gpri_tools.refractivity`). Smith–Weintraub moist-air
  refractivity from pressure, temperature and humidity, and a per-epoch
  refractivity series inverted from the per-pair range ramps — the only
  independent check on an empirically estimated screen.

## The worked example: Mount Baker

Everything under `examples/` is one analysis: eight GPRI-II campaigns on the
north side of Mount Baker, 2016 to 2019, asking whether those glaciers carry a
diurnal signal from the subglacial drainage system. It is what every
capability above was built for, and it is the only example in this repository.

- [`docs/baker.md`](docs/baker.md) — the analysis and what it found: the
  atmospheric ladder scored on held-out bedrock, the RGI reference audit, the
  per-pixel diurnal null, the population night-time trough, three two-cycle
  campaigns and whether the diurnal repeats from one year to the next.
- [`docs/campaigns.md`](docs/campaigns.md) — the campaign inventory, the
  measured scan headings and the per-campaign processing notes.
- [`docs/atmosphere.md`](docs/atmosphere.md) — the correction ladder in full.

`bin/run_scene.sh <scene> [upper|lower|both]` runs the whole chain for one
campaign, logging every step.

## Install

```bash
pip install -e '.[all]'      # numpy, scipy + pyproj, rasterio, matplotlib
pytest                       # 344 tests
```

Only `numpy` and `scipy` are required. `pyproj` and `rasterio` are needed for
geocoding and GeoTIFF output, `matplotlib` for figures; all three are imported
at point of use, so the core works without them.

## Use

```bash
S=$GPRI_SCENE_20170803          # set in site.env -- see site.env.example

gpri info       $S                              # what is in it, how coherent
gpri screens    $S                              # per-pair refractivity screens
gpri coregister $S --write                      # did the tripod turn?  offsets per SLC
gpri heading    $S --dem $GPRI_DEM --write      # scan heading from the terrain's shadows
gpri velocity   $S -o vel.npz --geotiff         # maps use the heading measured above
gpri timeseries $S -o ts.npz  --method wls
gpri phaselink  $S -o pl.npz  --method eigensar
gpri closure    $S                              # bias against temporal baseline
gpri unwrap     $S --pair 0 --min-coherence 0.6 -o unw.npz
gpri geocode    $S vel.npz --field velocity

# a raw campaign -> a scene directory (slc/, SLCu_tab, SLCl_tab), both antennas
gpri focus      $GPRI_RAW_20170827 $GPRI_SCENE_20170827 --workers 6

bin/survey_campaigns.py                         # what data exists, and how long
```

`gpri focus` defaults to the BakerBend recipe (`-d 5 -z 300 -r 300 -k 3.84`
in `gpri2_proc.py` terms: presum 5 sweeps, 300-sample Hann taper, 300 m
minimum range, Kaiser β 3.84). Point it at a campaign directory and it takes
every `.raw` or `.raw.gz` in it and its `raw*/` subdirectories, decompressing
gzipped sweeps as it reads them, skipping macOS `._` sidecars, and focusing
one acquisition once even if several copies of it exist; `--raw-list`
restricts it to the campaign's own `RAW_list`. Output is byte-compatible
with GAMMA's: the
`.slc.par` files are identical, the samples agree to float32 rounding
(max 2e-9 relative), and GAMMA's `multi_look` on our SLC reproduces its own
MLI to 4e-7.

```bash
```

```python
from gpri_tools import DiffStack, RadarGeometry, geocode_image, stack_velocity
from gpri_tools.timeseries import los_displacement

stack = DiffStack.from_directory(f"{S}/diff0", slc_tab=f"{S}/SLCu_tab")
rows, cols, ifg, cc = next(stack.patches(max_gib=2.0))
v = stack_velocity(los_displacement(np.angle(ifg), stack.wavelength),
                   stack.network, weights=cc)

from gpri_tools.heading import scene_heading
geom = RadarGeometry(stack.par, heading=scene_heading(S))   # see below
v_map, transform = geocode_image(v, geom, spacing=25.0)

# the same interface, formed from SLCs: lower antenna, i->i+1..3, 3x15 looks
from gpri_tools import SlcPairStack
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
python examples/baker_movie.py --scene 20170803 --rgi --anomaly periodic  # tilt-free trend
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
python examples/baker_seasons.py --detrend linear   # the same on per-pixel linear trends
# what repeats hour to hour, for the campaigns that ran more than one day
python examples/baker_composite.py --scenes 20170827 20180808 20190719
```

`bin/run_scene.sh <scene> [upper|lower|both]` runs that whole chain for one
scene, both antennas side by side, logging each step under
`$GPRI_WORK_ROOT/<scene>/logs/`.

## The scan heading is not in the data

`gpri_tools.geocode` maps the polar fan onto a local stereographic projection centred
on the radar. Everything it needs is in the parameter file except one number:

**`GPRI_scan_heading` is `0.0` in every BakerBend1 parameter file.** It was
never surveyed, and a heading of exactly zero would point the fan due north, at
nothing. `RadarGeometry` warns rather than accepting it quietly.

`BAKERBEND1_HEADING = 105.0` was the working answer through v0.5.0 — a
**starting guess** derived from the bearings to the north-side glaciers (the
radar at 48.82132 N, 121.92018 W, 1252 m sees Baker's summit at bearing 122.5°
and 9.2 km, Coleman Glacier at 120.6°, Mazama at 107.4°, Colfax Peak at
129.9°, and the 79° fan needs a heading near 105° to cover them). It was
wrong for every campaign, by 2.4° to 19.9°, and a tripod set up by hand on a
different day points a different way, so one number was never going to do.

### Measuring it from the terrain

`gpri heading` (`gpri_tools.heading`) needs a DEM, the radar's position and the mean
backscatter — no survey, no tie point. What a ground-based radar sees of
terrain is dominated by two things that are pure geometry: **shadow**
(everything behind the first ridge along a bearing is black) and **facing**
(slopes leaning toward the radar are bright). The DEM is resampled onto a
polar grid around the radar, a running-max elevation-angle test along each
bearing finds the shadow, the cosine of the local incidence gives the
brightness, and the result is binned into a (bearing, slant range) image.
For each trial heading the antenna angles pick their bearings, and the
high-passed simulation is correlated with the high-passed measurement. The
correlation is ~0.02 everywhere and 0.32–0.35 within 0.6° of the answer,
sharp because a shadow edge moves a full cross-range cell per degree at 5 km.
The first and last eight SLCs of a campaign agree to 0.03° where the mount
held; where it turned, the heading is fitted on the co-registered reference
block (below).

![Heading from the DEM, 20170827](docs/figures/02_heading_20170827.png)

Every campaign's measured heading, and what its mount did afterwards, is
tabulated in [`docs/campaigns.md`](docs/campaigns.md#scan-headings): nine
scenes spanning 100.13° (`20170827`) to 124.86° (`20180808`), none of them
105°.

Half a degree of heading is 80 m at 9 km, so the 2.4–6.4° errors of the 2017
masks moved glacier margins by 200–500 m — enough to put ice in the
reference on one side and rock in the ice on the other. `gpri heading
--write` leaves the answer as `heading.json` under the scene's work
directory, `scene_heading()` reads it, and every example and `gpri
velocity --geotiff` default to it (warning and falling back to 105° when it
is missing). The DEM is the Copernicus 30 m tile N48 W122; `site.env` names
it as `GPRI_DEM` and `bin/run_scene.sh` measures the heading before anything
that draws a map. The other two routes remain:
`gpri_tools.geocode.heading_from_tiepoint(par, lat, lon, row)` from one identified
feature, or `heading=` from field notes.

## Sign convention

Every phase follows GAMMA's `SLC_intf` convention: the interferogram for pair
`(i, j)` is `z_i * conj(z_j)` and carries phase `theta_i - theta_j`.
Displacement is reported **positive toward the radar**. See
`gpri_tools.timeseries.los_displacement` for the derivation — it is the easiest thing
in InSAR to get backwards and the hardest to notice.

## Where the data lives

Machine-specific locations — data roots, scene directories, scratch — live in
`site.env` at the repository root, which is gitignored; copy
`site.env.example` and fill it in. Nothing in the repository names a host, a
mount point, or a storage layout.

## GAMMA

Nothing in this package calls GAMMA, but a GAMMA installation is useful for
cross-checks and is what `gpri focus` was validated against. The 2017-07-11
Linux distribution installs by unpacking under `/usr/local`; `config.sh` and
`bin/check_env.sh` discover `/usr/local/GAMMA_SOFTWARE-*` (or `$GAMMA_HOME`)
and put the `ISP`, `DIFF`, `LAT` and `DISP` binaries on the path. That
distribution has no `par_GPRI2_SLC`: GPRI raw processing is the Python 2
script `GPRI2-2/trunk/python/gpri2_proc.py`, which is what `gpri_tools/focus.py`
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

