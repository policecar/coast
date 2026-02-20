#!/usr/bin/env bash
#
# bench_prng_compare.sh — Paired PRNG benchmark for COAST
#
# Builds two binaries (one per PRNG variant), runs N paired trials with
# different seed offsets, and reports per-trial SPS and summary statistics.
#
# Usage:
#   ./scripts/bench_prng_compare.sh [OPTIONS]
#
# Options:
#   -a PRNG_A    fast_rng_t backend A (default: mt19937_simd)
#   -b PRNG_B    fast_rng_t backend B (default: xor_table_prng)
#   -n STEPS     Simulation steps per trial (default: 500000)
#   -t TRIALS    Number of paired trials (default: 5)
#   -s STATE     Checkpoint file for --load-state (default: /tmp/coast_seed0.h5)
#   -c CONFIG    Simulation config (default: configs/mnist.json)
#   -j JOBS      Parallel build jobs (default: nproc)
#
# Output:
#   Per-trial SPS for both variants, paired delta, and summary statistics.
#   A one-tailed paired t-test p-value is reported (H0: B <= A).
#
# Examples:
#   # Compare mt19937_simd vs xor_table_prng (default)
#   ./scripts/bench_prng_compare.sh
#
#   # More trials for tighter confidence
#   ./scripts/bench_prng_compare.sh -t 10
#
#   # Compare two specific PRNGs
#   ./scripts/bench_prng_compare.sh -a mt19937_simd -b mt11213b_simd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
FAST_RNG_H="${PROJECT_ROOT}/include/core/rng/fast_rng.h"

# ── Defaults ────────────────────────────────────────────────────────────────
PRNG_A="mt19937_simd"
PRNG_B="xor_table_prng"
STEPS=500000
TRIALS=5
STATE="/tmp/coast_seed0.h5"
CONFIG="${PROJECT_ROOT}/configs/mnist.json"
JOBS=$(nproc)

# ── Argument parsing ─────────────────────────────────────────────────────────
while getopts "a:b:n:t:s:c:j:h" opt; do
    case $opt in
        a) PRNG_A="$OPTARG" ;;
        b) PRNG_B="$OPTARG" ;;
        n) STEPS="$OPTARG" ;;
        t) TRIALS="$OPTARG" ;;
        s) STATE="$OPTARG" ;;
        c) CONFIG="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        h) sed -n '3,30p' "$0"; exit 0 ;;
        *) echo "Unknown option -$OPTARG" >&2; exit 1 ;;
    esac
done

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log() { echo -e "${GREEN}[bench]${NC} $*"; }
err() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── Validate ─────────────────────────────────────────────────────────────────
[[ -f "$FAST_RNG_H" ]] || err "fast_rng.h not found at $FAST_RNG_H"
[[ -f "$CONFIG"     ]] || err "Config not found: $CONFIG"
[[ -f "$STATE"      ]] || err "State checkpoint not found: $STATE"
command -v perf >/dev/null || err "'perf' not found in PATH"
command -v python3 >/dev/null || err "'python3' not found in PATH"

BIN_A="${BUILD_DIR}/coast_bench_a"
BIN_B="${BUILD_DIR}/coast_bench_b"

# ── Helper: patch fast_rng.h, build, restore ─────────────────────────────────
build_with_prng() {
    local prng="$1"
    local outbin="$2"
    local header_file="$3"    # name to include (e.g. xor_table_prng.h)

    log "Building with $prng ..."

    # Save original
    cp "$FAST_RNG_H" "${FAST_RNG_H}.bak"

    # Patch: replace include and type alias
    # We use a temp file to avoid issues with sed -i on different platforms
    python3 - "$FAST_RNG_H" "$header_file" "$prng" << 'PYEOF'
import sys, re

path, header, prng = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    text = f.read()

# Replace #include for the rng implementation
text = re.sub(r'#include\s+"core/rng/\S+"', f'#include "core/rng/{header}"', text)
# Replace using fast_rng_t = rng_t<...>
text = re.sub(r'using fast_rng_t\s*=\s*rng_t<\S+>;',
              f'using fast_rng_t = rng_t<{prng}>;', text)
with open(path, 'w') as f:
    f.write(text)
PYEOF

    cmake --build "$BUILD_DIR" -j "$JOBS" --target coast > /dev/null 2>&1 || {
        cp "${FAST_RNG_H}.bak" "$FAST_RNG_H"
        err "Build failed for $prng"
    }
    cp "${BUILD_DIR}/coast" "$outbin"

    # Restore original
    cp "${FAST_RNG_H}.bak" "$FAST_RNG_H"
    rm "${FAST_RNG_H}.bak"
}

# Map PRNG name → header filename
prng_header() {
    case "$1" in
        mt19937_simd)        echo "mt19937_simd.h" ;;
        mt11213b_simd)       echo "mt11213b_simd.h" ;;
        xor_table_prng)      echo "xor_table_prng.h" ;;
        philox_4x32_avx512)  echo "philox_4x32_avx512.h" ;;
        *)  err "Unknown PRNG '$1'. Add it to prng_header() in this script." ;;
    esac
}

# ── Build both variants ───────────────────────────────────────────────────────
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -S "$PROJECT_ROOT" > /dev/null 2>&1

build_with_prng "$PRNG_A" "$BIN_A" "$(prng_header "$PRNG_A")"
build_with_prng "$PRNG_B" "$BIN_B" "$(prng_header "$PRNG_B")"

log "Binaries: $BIN_A  $BIN_B"
log "Trials: $TRIALS  Steps: $STEPS  Config: $(basename $CONFIG)"
echo ""

# ── Run paired trials ─────────────────────────────────────────────────────────
COMMON_ARGS=("-n" "$STEPS" "--load-state" "$STATE")

run_trial() {
    local bin="$1" seed="$2"
    local elapsed
    elapsed=$(perf stat "$bin" "${COMMON_ARGS[@]}" \
              --seed-offset "$seed" "$CONFIG" 2>&1 \
              | grep "seconds time elapsed" \
              | awk '{print $1}' \
              | tr -d '.')
    # perf formats as "X.XXX.XXX,XXX" with locale separators — normalise
    elapsed=$(perf stat "$bin" "${COMMON_ARGS[@]}" \
              --seed-offset "$seed" "$CONFIG" 2>&1 \
              | grep "seconds time elapsed")
    # extract the number (last field before "seconds")
    echo "$elapsed" | awk '{print $1}'
}

# Use Python for the actual timing (perf stat always reports wall time at end)
printf "%-6s  %-12s  %-12s  %s\n" "seed" "$PRNG_A" "$PRNG_B" "delta"
printf "%-6s  %-12s  %-12s  %s\n" "------" "------------" "------------" "------"

SPS_A_ALL=()
SPS_B_ALL=()

for (( i=0; i<TRIALS; i++ )); do
    seed=$i

    # Alternate order to cancel slow load drift
    if (( seed % 2 == 0 )); then
        T_A=$(perf stat "${BIN_A}" "${COMMON_ARGS[@]}" --seed-offset $seed "$CONFIG" 2>&1 | grep "seconds time elapsed" | awk '{print $1}')
        T_B=$(perf stat "${BIN_B}" "${COMMON_ARGS[@]}" --seed-offset $seed "$CONFIG" 2>&1 | grep "seconds time elapsed" | awk '{print $1}')
    else
        T_B=$(perf stat "${BIN_B}" "${COMMON_ARGS[@]}" --seed-offset $seed "$CONFIG" 2>&1 | grep "seconds time elapsed" | awk '{print $1}')
        T_A=$(perf stat "${BIN_A}" "${COMMON_ARGS[@]}" --seed-offset $seed "$CONFIG" 2>&1 | grep "seconds time elapsed" | awk '{print $1}')
    fi

    python3 - "$seed" "$T_A" "$T_B" "$STEPS" << 'PYEOF'
import sys
seed, ta, tb, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
# Normalise locale number format (e.g. "125,321187821" → "125.321187821")
ta = float(ta.replace('.', '').replace(',', '.'))
tb = float(tb.replace('.', '').replace(',', '.'))
sps_a = steps / ta
sps_b = steps / tb
delta = (sps_b - sps_a) / sps_a * 100
print(f"PAIR {seed} {sps_a:.1f} {sps_b:.1f} {delta:+.2f}")
PYEOF

done | tee /tmp/bench_prng_pairs.txt | while IFS=' ' read -r tag seed sps_a sps_b delta; do
    printf "%-6s  %-12s  %-12s  %s\n" "$seed" "$sps_a" "$sps_b" "${delta}%"
done

echo ""

# ── Summary statistics (paired t-test) ────────────────────────────────────────
python3 - /tmp/bench_prng_pairs.txt "$PRNG_A" "$PRNG_B" << 'PYEOF'
import sys, math

path, name_a, name_b = sys.argv[1], sys.argv[2], sys.argv[3]

sps_a_all, sps_b_all, deltas = [], [], []
with open(path) as f:
    for line in f:
        parts = line.split()
        if parts[0] == 'PAIR':
            sps_a_all.append(float(parts[2]))
            sps_b_all.append(float(parts[3]))
            deltas.append(float(parts[4]))

n = len(deltas)
mean_a = sum(sps_a_all) / n
mean_b = sum(sps_b_all) / n
mean_d = sum(deltas) / n
std_d  = math.sqrt(sum((d - mean_d)**2 for d in deltas) / (n - 1)) if n > 1 else 0
sem_d  = std_d / math.sqrt(n) if n > 1 else float('inf')
t_stat = mean_d / sem_d if sem_d > 0 else float('inf')

print(f"{name_a:25s}  mean={mean_a:8.1f} SPS")
print(f"{name_b:25s}  mean={mean_b:8.1f} SPS")
print()
print(f"Paired delta  mean={mean_d:+.2f}%  std={std_d:.2f}pp  t={t_stat:.2f}  (df={n-1})")
print()

if   abs(t_stat) > 4.604: sig = "p < 0.005"
elif abs(t_stat) > 2.776: sig = "p < 0.025"
elif abs(t_stat) > 2.132: sig = "p < 0.05"
else:                      sig = "not significant at p=0.05"

direction = "B > A" if mean_d > 0 else "B < A"
all_same  = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
print(f"Result: {direction},  {sig}")
print(f"All trials same sign: {all_same}  {[f'{d:+.1f}%' for d in deltas]}")
PYEOF
