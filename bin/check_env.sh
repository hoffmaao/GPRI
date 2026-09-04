#!/bin/bash
# Feasibility probe: can this host run GAMMA InSAR processing against the data?
# Every check prints PASS / FAIL / WARN and the script exits non-zero on any FAIL.

. "$(cd "$(dirname "$0")/.." && pwd)/config.sh"

fails=0; warns=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fails=$((fails+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; warns=$((warns+1)); }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

echo "GPRI/GAMMA environment check -- $(hostname) -- $(date)"

head_ "1. Host resources"
cores=$(nproc)
memgb=$(free -g | awk '/^Mem:/{print $2}')
[ "$cores" -ge 4 ]  && pass "$cores CPU cores"      || warn "only $cores CPU cores"
[ "$memgb" -ge 16 ] && pass "${memgb} GB RAM"       || warn "only ${memgb} GB RAM"
scratch=$(df -BG --output=avail "$GPRI_WORK_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$scratch" ] && [ "$scratch" -ge 100 ]; then
  pass "${scratch} GB free on scratch ($GPRI_WORK_ROOT)"
else
  warn "${scratch:-?} GB free on scratch ($GPRI_WORK_ROOT) -- a single GPRI day is ~95 GB"
fi

head_ "2. Cold storage (network mount)"
if timeout 60 ls "$GPRI_DATA_ROOT" >/dev/null 2>&1; then
  pass "automount triggered: $GPRI_DATA_ROOT is readable"
  mnt=$(df -P "$GPRI_DATA_ROOT" 2>/dev/null | tail -1 | awk '{print $1}')
  avail=$(df -PBG "$GPRI_DATA_ROOT" 2>/dev/null | tail -1 | awk '{print $4}')
  pass "backing export: $mnt ($avail free)"
  case "$(stat -f -c %T "$GPRI_DATA_ROOT" 2>/dev/null)" in
    nfs*) pass "filesystem type is NFS (as expected)" ;;
    *)    warn "filesystem type is $(stat -f -c %T "$GPRI_DATA_ROOT" 2>/dev/null), expected nfs" ;;
  esac
  probe="$GPRI_DATA_ROOT/.gpri_write_probe.$$"
  if timeout 30 touch "$probe" 2>/dev/null; then
    pass "write access to $GPRI_DATA_ROOT"; rm -f "$probe"
  else
    warn "no write access to $GPRI_DATA_ROOT (read-only workflow only)"
  fi
else
  fail "cannot read $GPRI_DATA_ROOT -- storage mount is down or GPRI_DATA_ROOT unset"
fi

head_ "3. Test scene"
if [ -d "$GPRI_TEST_SCENE" ]; then
  pass "scene present: $GPRI_TEST_SCENE"
  tab="$GPRI_TEST_SCENE/$GPRI_TEST_SLC_TAB"
  if [ -r "$tab" ]; then
    n=$(wc -l < "$tab")
    pass "$GPRI_TEST_SLC_TAB lists $n SLCs"
    first=$(awk 'NR==1{print $1}' "$tab")
    if [ -r "$GPRI_TEST_SCENE/$first" ]; then
      pass "first SLC readable: $first"
    else
      fail "first SLC listed in $GPRI_TEST_SLC_TAB is not readable: $first"
    fi
  else
    fail "cannot read $tab"
  fi
else
  fail "test scene missing: $GPRI_TEST_SCENE"
fi

head_ "4. GAMMA installation"
if [ -n "${GAMMA_HOME:-}" ] && [ -d "$GAMMA_HOME" ]; then
  pass "GAMMA_HOME=$GAMMA_HOME"
  for m in ISP DIFF LAT DISP; do
    [ -d "$GAMMA_HOME/$m/bin" ] && pass "module $m present" || warn "module $m missing"
  done
else
  fail "GAMMA not installed (GAMMA_HOME unset and no install found)"
fi
for b in SLC_intf multi_look cc_wave rasmph_pwr create_offset; do
  gpri_have "$b" && pass "binary on PATH: $b" || fail "binary missing: $b"
done
# GPRI raw -> SLC is a python2 script in this distribution, not a binary;
# gpri_tools/focus.py ports it, so this is informational
if [ -n "${GAMMA_HOME:-}" ] && [ -f "$GAMMA_HOME/GPRI2-2/trunk/python/gpri2_proc.py" ]; then
  pass "gpri2_proc.py present (gpri focus is the python3 port)"
else
  warn "gpri2_proc.py not found (not needed: gpri focus focuses raw)"
fi
if gpri_have SLC_intf; then
  if SLC_intf 2>&1 | head -6 | grep -qi 'usage'; then
    pass "SLC_intf executes"
  else
    fail "SLC_intf present but did not run -- check the linker (ldd \$(which SLC_intf))"
  fi
fi

head_ "5. Supporting toolchain"
for t in gcc make python3; do
  gpri_have "$t" && pass "$t: $($t --version 2>&1 | head -1)" || warn "$t missing"
done
ls /usr/lib64/libfftw3f.so* >/dev/null 2>&1 && pass "libfftw3f present" || warn "libfftw3f not found in /usr/lib64"

head_ "Summary"
echo "  $fails FAIL, $warns WARN"
[ "$fails" -eq 0 ] && echo "  -> environment is ready to process." \
                   || echo "  -> blocked; resolve the FAILs above."
exit $(( fails > 0 ))
