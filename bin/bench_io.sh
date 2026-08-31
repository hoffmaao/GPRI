#!/bin/bash
# Measure read throughput from cold storage vs. local scratch.
# Informs whether GAMMA should read in place over NFS or work on staged copies.
set -euo pipefail
. "$(cd "$(dirname "$0")/.." && pwd)/config.sh"

SIZE_MB="${1:-2048}"

pick_probe() {
  find "$GPRI_TEST_SCENE" -maxdepth 1 -type f -size +2G -print -quit 2>/dev/null
}

echo "I/O benchmark (${SIZE_MB} MB per test) -- $(date)"

echo
echo "== read: cold storage over the network mount =="
probe=$(pick_probe || true)
if [ -n "${probe:-}" ]; then
  echo "  probe file: $probe"
  dd if="$probe" of=/dev/null bs=1M count="$SIZE_MB" 2>&1 | tail -1
else
  echo "  SKIP: no file >2G found in $GPRI_TEST_SCENE"
fi

echo
echo "== write: local scratch $GPRI_WORK_ROOT =="
mkdir -p "$GPRI_WORK_ROOT"
tmp="$GPRI_WORK_ROOT/.bench.$$"
dd if=/dev/zero of="$tmp" bs=1M count="$SIZE_MB" conv=fdatasync 2>&1 | tail -1
echo
echo "== read: local scratch (cache dropped not possible unprivileged; indicative only) =="
dd if="$tmp" of=/dev/null bs=1M count="$SIZE_MB" 2>&1 | tail -1
rm -f "$tmp"

echo
echo "Reference measurements, 2026-08-30:"
echo "  network read            : ~118 MB/s  (gigabit link, effectively saturated)"
echo "  local scratch write     : ~200 MB/s"
echo "  one GPRI day (~95 GB)   : ~13 min to stage"
