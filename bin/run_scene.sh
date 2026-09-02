#!/bin/bash
# The whole figure chain for one scene, as the README's "reproduce" block
# runs it, with every step logged under $GPRI_WORK_ROOT/<scene>/logs/.
#
#   bin/run_scene.sh 20170827            # both antennas, movies, closure
#   bin/run_scene.sh 20170803 upper      # one antenna, no cross-antenna steps
#
# Two geometry steps come first.  `gpri coregister` measures every SLC's
# azimuth offset against the last and, when the tripod turned during the
# campaign (20180709), leaves the sidecar that reads the stack on one grid;
# `gpri heading` measures the scan heading from the DEM at $GPRI_DEM (set it
# in site.env) so the RGI masks and maps are rotated correctly.  Both are
# skipped for a scene that has no slc/ of its own, e.g. GAMMA's diff0 for
# 20170803, whose heading was measured separately.
#
# The first script for each antenna pays for reading the day (the decimated
# stack is cached, so the rest take seconds to a few minutes); the two
# antennas are independent and run side by side.  Closure comes last: the
# i->i+3 network is three times the pairs of the daisy chain.
set -uo pipefail
. "$(cd "$(dirname "$0")/.." && pwd)/config.sh"
cd "$GPRI_REPO"

scene="${1:?usage: run_scene.sh <scene> [upper|lower|both]}"
which="${2:-both}"
case "$which" in
  upper) antennas="upper" ;;
  lower) antennas="lower" ;;
  both)  antennas="upper lower" ;;
  *) gpri_die "antenna must be upper, lower or both" ;;
esac
var="GPRI_SCENE_$scene"
[ -d "${!var:-}" ] || gpri_die "$var is not a scene directory (set it in site.env)"

logs="$GPRI_WORK_ROOT/$scene/logs"; mkdir -p "$logs"
export PYTHONPATH="$GPRI_REPO${PYTHONPATH:+:$PYTHONPATH}"
py() { python3 -u "$@"; }
step() {   # step <log-name> <command...>
  local log="$logs/$1.log"; shift
  printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"
  if "$@" > "$log" 2>&1; then printf '%s    ok  (%s)\n' "$(date '+%H:%M:%S')" "$log"
  else printf '%s    FAILED (%s)\n' "$(date '+%H:%M:%S')" "$log"; fi
}
chain() {  # chain <antenna>
  local a="$1"
  step "aps_$a"     py examples/baker_aps.py     --scene "$scene" --antenna "$a" --decimate 16 --sigma 5 25 --rgi --screens-on-bedrock
  step "rgi_$a"     py examples/baker_rgi.py     --scene "$scene" --antenna "$a" --decimate 16
  step "pairlsq_$a" py examples/baker_pairlsq.py --scene "$scene" --antenna "$a" --decimate 16 --rgi
  step "repeat_$a"  py examples/baker_repeat.py  --scene "$scene" --antenna "$a" --decimate 16 --rgi
  step "population_$a" py examples/baker_population.py --scene "$scene" --antenna "$a" --decimate 16 --rgi
  if [ "$a" = upper ]; then
    step movie          py examples/baker_movie.py --scene "$scene" --rgi
    step movie_rate2h   py examples/baker_movie.py --scene "$scene" --rgi --rate-hours 2
    step movie_anommean py examples/baker_movie.py --scene "$scene" --rgi --anomaly mean
    step movie_anomtrend py examples/baker_movie.py --scene "$scene" --rgi --anomaly trend
    step movie_anomperiodic py examples/baker_movie.py --scene "$scene" --rgi --anomaly periodic
  fi
}

# the two sidecars are measured once (a thousand SLCs over a network mount
# is a quarter of an hour); GPRI_REDO_GEOMETRY=1 measures them again
work="$GPRI_WORK_ROOT/$scene"
if [ -d "${!var}/slc" ] && ! [ -d "${!var}/diff0" ]; then
  if [ -f "$work/azimuth_offsets.json" ] && [ -z "${GPRI_REDO_GEOMETRY:-}" ]; then
    echo "coregister: $work/azimuth_offsets.json exists, kept"
  else
    step coregister bin/gpri coregister "${!var}" --write \
         --figure "docs/figures/03_coregister_$scene.png"
  fi
  if [ -f "$work/heading.json" ] && [ -z "${GPRI_REDO_GEOMETRY:-}" ]; then
    echo "heading: $work/heading.json exists, kept"
  elif [ -f "${GPRI_DEM:-}" ]; then
    step heading bin/gpri heading "${!var}" --dem "$GPRI_DEM" --write \
         --figure "docs/figures/02_heading_$scene.png"
  else
    echo "GPRI_DEM is not a file: heading not measured, maps use heading.json if present"
  fi
fi
step info bin/gpri info "${!var}"
pids=()
for a in $antennas; do chain "$a" & pids+=($!); done
wait "${pids[@]}"
if [ "$which" = both ]; then
  step antennas py examples/baker_antennas.py --scene "$scene" --decimate 16 --rgi
fi
step closure_lag123 py examples/baker_closure.py --scene "$scene" --lags 1 2 3 --looks 3 15
echo "done: $scene"
