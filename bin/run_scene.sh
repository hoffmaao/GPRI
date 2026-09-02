#!/bin/bash
# The whole figure chain for one scene, as the README's "reproduce" block
# runs it, with every step logged under $GPRI_WORK_ROOT/<scene>/logs/.
#
#   bin/run_scene.sh 20170827            # both antennas, movies, closure
#   bin/run_scene.sh 20170803 upper      # one antenna, no cross-antenna steps
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
  fi
}

step info bin/gpri info "${!var}"
pids=()
for a in $antennas; do chain "$a" & pids+=($!); done
wait "${pids[@]}"
if [ "$which" = both ]; then
  step antennas py examples/baker_antennas.py --scene "$scene" --decimate 16 --rgi
fi
step closure_lag123 py examples/baker_closure.py --scene "$scene" --lags 1 2 3 --looks 3 15
echo "done: $scene"
