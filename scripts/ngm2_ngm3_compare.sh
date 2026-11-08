#!/usr/bin/env bash
#
# ngm2_ngm3_compare.sh
#
# Drive a side-by-side mnist run of ngm2 (configs/mnist.json) and
# ngm3 (configs/mnist_ngm3.json) for N steps with seed 0, capturing
# everything needed for analyze_ngm3_comparison.py:
#   - periodic state checkpoints (for rep-count timeline + per-step
#     permanence snapshots)
#   - final state HDF5 (for end-of-run permanence histograms /
#     representation samples)
#   - full IO-buffer log (for activity histograms + per-class
#     specialization heatmap, since the mnist labels buffer rides
#     along)
#
# Usage:
#   scripts/ngm2_ngm3_compare.sh                  # default N=500000
#   scripts/ngm2_ngm3_compare.sh 50000            # custom N
#   scripts/ngm2_ngm3_compare.sh 500000 5000      # custom N + checkpoint stride
#
# Output goes to benchmarks/ngm3_comparison_<git-sha>/{ngm2,ngm3}/.
#

set -euo pipefail

N="${1:-500000}"
CKPT_STRIDE="${2:-50000}"
SHA="$(git rev-parse --short HEAD)"
ROOT="benchmarks/ngm3_comparison_${SHA}"

# Resolve repo root so we can run from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x ./build/coast ]]; then
    echo "error: ./build/coast not found. Build first with:" >&2
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
    exit 1
fi

mkdir -p "$ROOT"
echo "[compare] output: $ROOT  N=$N  checkpoint stride=$CKPT_STRIDE"

run_one() {
    local label="$1"
    local src_cfg="$2"
    local dir="$ROOT/$label"
    local abs_snap
    abs_snap="$REPO_ROOT/$ROOT/$label/snapshots"
    mkdir -p "$abs_snap"
    # Materialise a per-run config with snapshot_dir set to an ABSOLUTE
    # path so the snapshot location is fixed at exactly $abs_snap: an absolute
    # leaf wins outright and bypasses the app config's config_base (a relative
    # leaf would instead be placed under <config_dir>/<config>/).
    local cfg="$dir/config.json"
    jq --arg sd "$abs_snap" '.snapshot_dir = $sd' "$src_cfg" > "$cfg"
    echo
    echo "[compare] === $label ==="
    echo "[compare] config:    $cfg"
    echo "[compare] snapshots: $dir/snapshots"
    echo "[compare] buffers:   $dir/buffers.h5"
    echo "[compare] state:     $dir/state_final.h5"
    ./build/coast -n "$N" \
        --checkpoint "$CKPT_STRIDE" \
        --save-state "$dir/state_final.h5" \
        --log-buffers "$dir/buffers.h5" \
        --seed-offset 0 \
        "$cfg"
}

run_one ngm2 configs/mnist.json
run_one ngm3 configs/mnist_ngm3.json

echo
echo "[compare] done. Analyze with:"
echo "  python3 scripts/analyze_ngm3_comparison.py $ROOT"
