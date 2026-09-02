#!/bin/bash
# Minimal end-to-end GAMMA ISP run on a single GPRI interferometric pair.
# This is the actual answer to "can we run GAMMA here?" -- it exercises the
# binaries and I/O against real BakerBend1 data.
#
#   bin/stage_data.sh 2 && bin/smoke_test.sh
#
# Chain: create_offset -> SLC_intf -> multi_look -> cc_wave -> rasmph_pwr
set -euo pipefail
. "$(cd "$(dirname "$0")/.." && pwd)/config.sh"

scene="$GPRI_WORK_ROOT/$(basename "$GPRI_TEST_SCENE")"
tab="$scene/$GPRI_TEST_SLC_TAB"
out="$scene/smoke"

gpri_have SLC_intf || gpri_die "GAMMA not on PATH -- run bin/check_env.sh first"
[ -r "$tab" ]      || gpri_die "no staged scene at $scene -- run bin/stage_data.sh first"
[ "$(wc -l < "$tab")" -ge 2 ] || gpri_die "need >=2 staged SLCs, have $(wc -l < "$tab")"

mkdir -p "$out"; cd "$scene"

slc1=$(awk 'NR==1{print $1}' "$tab"); par1=$(awk 'NR==1{print $2}' "$tab")
slc2=$(awk 'NR==2{print $1}' "$tab"); par2=$(awk 'NR==2{print $2}' "$tab")
id1=$(basename "$slc1" .slc); id2=$(basename "$slc2" .slc)
pair="${id1}_${id2}"

echo "pair: $pair"
echo "  ref: $slc1"
echo "  sec: $slc2"
echo "  looks: ${GPRI_RLKS} range x ${GPRI_AZLKS} azimuth"
echo

# 1. offset parameter file for the pair.  GPRI is a fixed tripod-mounted
#    instrument, so the pair is already co-registered -- no offset estimation.
echo "[1/5] create_offset"
create_offset "$par1" "$par2" "$out/$pair.off" 1 "$GPRI_RLKS" "$GPRI_AZLKS" 0

# 2. form the interferogram.  Both spectral filters off: the range shift
#    filter needs state vectors GPRI has none of (SLC_intf skips it anyway),
#    and the azimuth common-band filter is meaningless for a rotating
#    antenna -- with it on, the phase of a GPRI pair is scrambled to noise.
echo "[2/5] SLC_intf"
SLC_intf "$slc1" "$slc2" "$par1" "$par2" "$out/$pair.off" \
         "$out/$pair.int" "$GPRI_RLKS" "$GPRI_AZLKS" - - 0 0

# 3. multi-looked intensity for the reference scene
echo "[3/5] multi_look"
multi_look "$slc1" "$par1" "$out/$id1.mli" "$out/$id1.mli.par" \
           "$GPRI_RLKS" "$GPRI_AZLKS"
multi_look "$slc2" "$par2" "$out/$id2.mli" "$out/$id2.mli.par" \
           "$GPRI_RLKS" "$GPRI_AZLKS"

width=$(awk '/^interferogram_width:/{print $2}' "$out/$pair.off")
[ -n "$width" ] || gpri_die "could not read interferogram_width from $out/$pair.off"
echo "  interferogram width: $width"

# 4. coherence
echo "[4/5] cc_wave"
cc_wave "$out/$pair.int" "$out/$id1.mli" "$out/$id2.mli" "$out/$pair.cc" "$width" 5 5 1

# 5. quicklook raster
echo "[5/5] rasmph_pwr"
rasmph_pwr "$out/$pair.int" "$out/$id1.mli" "$width" 1 1 0 1 1 1.0 0.35 1 \
           "$out/$pair.int.bmp"

echo
echo "SMOKE TEST PASSED"
echo "products in $out:"
ls -lh "$out"
