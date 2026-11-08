#!/usr/bin/env bash
#
# Interleaved A/B SPS benchmark for the error-handling redesign.
#
# Measures the per-step overhead of the value-based error channel by alternating
# a PRE binary and a POST binary within one time window (so CPU thermal/boost
# drift hits both conditions equally — a non-interleaved A/B is confounded by it).
# Each run emits its steady-state SPS (the last per-10k "SPS:" report); order
# flips each round to cancel within-round order bias. Pair the two values from
# each round for a paired t-test.
#
# Usage:
#   scripts/bench_error_handling_sps.sh <pre_binary> <post_binary> <config.json> [N] [ROUNDS]
#
# Example (build a PRE binary at the parent commit into ./coast_pre, the POST at
# HEAD into ./coast_post, both Release):
#   scripts/bench_error_handling_sps.sh ./coast_pre ./coast_post configs/mnist_ngm3.json 40000 10
#
# Output: lines "pre <sps>" / "post <sps>" (two per round). Aggregate with a
# paired t-test (smallest meaningful effect ~3% on this machine; see benchmarks/).
set -euo pipefail
PRE="$1"; POST="$2"; CONFIG="$3"; N="${4:-40000}"; ROUNDS="${5:-10}"
last_sps() { "$1" -n "$N" "$CONFIG" 2>/dev/null | grep -oE 'SPS: [0-9.]+' | tail -1 | grep -oE '[0-9.]+'; }
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 0 )); then
    echo "pre  $(last_sps "$PRE")";  echo "post $(last_sps "$POST")"
  else
    echo "post $(last_sps "$POST")"; echo "pre  $(last_sps "$PRE")"
  fi
done
