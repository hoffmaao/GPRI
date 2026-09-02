#!/bin/bash
# Shared configuration for GPRI/GAMMA processing.
# Source this from any script:  . "$(dirname "$0")/../config.sh"
#
# Machine-specific locations (data roots, scene directories, scratch space)
# live in site.env at the repository root -- gitignored, so nothing about a
# particular host or storage layout ever enters the repository.  Copy
# site.env.example to site.env and fill it in.

# ---------------------------------------------------------------- repo layout
GPRI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPRI_REPO

# ------------------------------------------------------------------- site env
if [ -f "$GPRI_REPO/site.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$GPRI_REPO/site.env"
  set +a
fi

# ------------------------------------------------------------------- storage
# Root of the GPRI data holdings.  The mount may be on-demand (automounted);
# scripts that touch it allow for a slow first access.
# Falls back to the first of GPRI_SURVEY_ROOTS.
export GPRI_DATA_ROOT="${GPRI_DATA_ROOT:-${GPRI_SURVEY_ROOTS%%:*}}"

# Local scratch.  GAMMA is I/O heavy; staging to fast local disk beats
# working across a network mount that the tools will re-read many times.
export GPRI_WORK_ROOT="${GPRI_WORK_ROOT:-$GPRI_REPO/work}"

# Default test scene: a processed GPRI acquisition with SLCs on disk.
export GPRI_TEST_SCENE="${GPRI_TEST_SCENE:-${GPRI_SCENE_20170803:-}}"
export GPRI_TEST_SLC_TAB="${GPRI_TEST_SLC_TAB:-SLCu_tab}"   # upper antenna

# --------------------------------------------------------------------- GAMMA
# GAMMA ships as per-module trees (ISP, DIFF, LAT, MSP, IPTA) under GAMMA_HOME.
# Set GAMMA_HOME in your environment, or drop the install somewhere below.
if [ -z "${GAMMA_HOME:-}" ]; then
  for _c in /usr/local/GAMMA_SOFTWARE-* /opt/GAMMA_SOFTWARE-* "$HOME"/GAMMA_SOFTWARE-*; do
    [ -d "$_c/ISP/bin" ] && { export GAMMA_HOME="$_c"; break; }
  done
  unset _c
fi

if [ -n "${GAMMA_HOME:-}" ] && [ -d "$GAMMA_HOME" ]; then
  export ISP_HOME="$GAMMA_HOME/ISP"
  export DIFF_HOME="$GAMMA_HOME/DIFF"
  export LAT_HOME="$GAMMA_HOME/LAT"
  export DISP_HOME="$GAMMA_HOME/DISP"
  export MSP_HOME="$GAMMA_HOME/MSP"
  for _m in ISP DIFF LAT DISP MSP IPTA; do
    [ -d "$GAMMA_HOME/$_m/bin" ]    && PATH="$GAMMA_HOME/$_m/bin:$PATH"
    [ -d "$GAMMA_HOME/$_m/scripts" ] && PATH="$GAMMA_HOME/$_m/scripts:$PATH"
  done
  unset _m
  export PATH
fi

# ---------------------------------------------------------- processing knobs
export GPRI_RLKS="${GPRI_RLKS:-1}"    # range looks
export GPRI_AZLKS="${GPRI_AZLKS:-1}"  # azimuth looks

# ------------------------------------------------------------------- helpers
gpri_have() { command -v "$1" >/dev/null 2>&1; }
gpri_die()  { echo "ERROR: $*" >&2; exit 1; }
