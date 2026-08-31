#!/bin/bash
# Copy a subset of a GPRI scene from cold storage onto local scratch.
#
#   bin/stage_data.sh [N]        stage the first N SLCs (default 4) + their .par
#
# Rationale: the network mount is gigabit-limited, local scratch writes
# at ~200 MB/s.  Staging once and processing locally avoids paying NFS latency
# on every one of GAMMA's many passes over the same files.
set -euo pipefail
. "$(cd "$(dirname "$0")/.." && pwd)/config.sh"

N="${1:-4}"
src="$GPRI_TEST_SCENE"
tab="$src/$GPRI_TEST_SLC_TAB"
dst="$GPRI_WORK_ROOT/$(basename "$src")"

[ -d "$src" ] || gpri_die "scene not found: $src (is the storage mount up?)"
[ -r "$tab" ] || gpri_die "SLC table not readable: $tab"

mkdir -p "$dst/slc"
echo "staging $N SLCs"
echo "  from: $src"
echo "  to:   $dst"

: > "$dst/$GPRI_TEST_SLC_TAB"
head -n "$N" "$tab" | while read -r slc par _rest; do
  for f in "$slc" "$par"; do
    [ -n "$f" ] || continue
    if [ -f "$src/$f" ]; then
      mkdir -p "$dst/$(dirname "$f")"
      rsync -a "$src/$f" "$dst/$f" && echo "  + $f ($(du -h "$dst/$f" | cut -f1))"
    else
      echo "  WARNING: missing on source: $f" >&2
    fi
  done
  # rewrite the table with paths relative to the staged copy
  echo "$slc  $par" >> "$dst/$GPRI_TEST_SLC_TAB"
done

echo
echo "staged $(wc -l < "$dst/$GPRI_TEST_SLC_TAB") entries, $(du -sh "$dst" | cut -f1) total"
echo "scene now at: $dst"
