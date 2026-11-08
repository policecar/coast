#!/usr/bin/env bash
# h5-browser G-suite (G1-G7) benchmark runner.
#
# Usage: scripts/bench_h5browse.sh DATA_DIR [BUILD_DIR]
#   DATA_DIR  directory with the retained verification assets
#             (buf_4g.h5, buf_4g_plain.h5, obs_out/, probe.h5, corpus/)
#   BUILD_DIR CMake build dir containing libcoast_lib.a (default: build)
#
# Results go to stdout; promote them to benchmarks/perf_h5browse.md with
# this script's git commit hash (repo reproducibility convention).

set -euo pipefail

DATA=${1:?usage: bench_h5browse.sh DATA_DIR [BUILD_DIR]}
BUILD=${2:-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DRIVER_SRC="$ROOT/scripts/bench_h5browse_driver.cpp"
DRIVER=$(mktemp -d)/bench_h5browse

echo "== building driver =="
g++ -O3 -march=native -std=c++23 \
    -I "$ROOT/include" -I "$ROOT/include/hdf5" \
    "$DRIVER_SRC" "$ROOT/$BUILD/libcoast_lib.a" \
    -lhdf5 -ltbb -o "$DRIVER"

OBS="$DATA/obs_out"
OBS_FILE=$(ls "$OBS"/*.h5 2>/dev/null | head -1 || true)

run() { echo "-- $*"; "$DRIVER" "$@"; echo; }

echo "== G1: open -> Tier-1 overview (cold) =="
run open "$DATA/buf_4g.h5"
run open "$DATA/buf_4g_plain.h5"
[ -n "$OBS_FILE" ] && run open "$OBS_FILE"
run open "$DATA/probe.h5"

echo "== G2: stats scan vs raw sequential read (same session) =="
run rawread "$DATA/buf_4g_plain.h5"
run scan "$DATA/buf_4g_plain.h5"
run rawread "$DATA/buf_4g.h5"
run scan "$DATA/buf_4g.h5"
[ -n "$OBS_FILE" ] && run rawread "$OBS_FILE"
[ -n "$OBS_FILE" ] && run scan "$OBS_FILE"

echo "== G3: histogram pass + rebin-without-rescan =="
run hist "$DATA/buf_small.h5" 2

echo "== G4: pairwise sampled k=2000 =="
run pairwise "$DATA/buf_4g_plain.h5" 2

echo "== G5: structural search =="
[ -n "$OBS_FILE" ] && run search "$OBS_FILE"

echo "== G6: corpus zero-crash =="
run corpus "$DATA/corpus"

echo "== G7: GUI-side stall proxy during full scan =="
run stall "$DATA/buf_4g_plain.h5"

echo "== done =="
