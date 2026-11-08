#!/usr/bin/env bash
# The real-coast acceptance gate (Discover §5), driven against a REAL coast
# binary and the real mnist dataset: a job run through coastd == a direct coast
# run (bit-identical).
#
# (A second gate — a child KILLED mid-run and auto-resumed == an uninterrupted
# run — was removed 2026-11: the kill had to land between the first checkpoint and
# the end of a real coast run, a window held only by timing; CLAUDE.md "Testing".)
#
# Usage:  tests/unit/coastd/run_real_gate.sh <repo_root> <coast_binary> <test_binary>
#
# Exit codes:  0 = gate passed · 1 = gate FAILED · 77 = prerequisites missing
# (ctest maps 77 to "Skipped", so a missing dataset or h5diff shows up as SKIPPED
# in the summary — never as a pass. The cases themselves carry doctest::skip(),
# so a plain ctest run of the unit binary neither runs nor reports them.)
set -euo pipefail

REPO="${1:?repo root}"
COAST="${2:?coast binary}"
TESTBIN="${3:?test binary}"

skip() { echo "coastd real-coast gate SKIPPED: $*" >&2; exit 77; }

[ -x "$COAST" ]   || skip "no coast binary at $COAST"
[ -x "$TESTBIN" ] || skip "no test binary at $TESTBIN"
command -v h5diff  >/dev/null 2>&1 || skip "h5diff not on PATH (hdf5-tools)"
command -v python3 >/dev/null 2>&1 || skip "python3 not on PATH"
[ -f "$REPO/configs/mnist.json" ]  || skip "configs/mnist.json missing"
[ -d "$REPO/data/mnist" ]          || skip "data/mnist missing (scripts/download_datasets.sh)"

# Refuse to run under TSan. This gate drives a REAL mnist simulation, and the sim
# core is TBB-parallel: under TSan the child spends its time emitting TBB
# work-stealing race reports rather than stepping. MEASURED on this repo — a
# TSan-built `coast -n 4000 configs/mnist.json` produced 52k lines of report and had
# still not left `phase=start` after 10 minutes, so no wait budget makes this gate
# pass; raising one only converts a fast wrong answer into a slow one.
#
# The reports themselves are the known TBB partitioner noise (SANITIZERS.md), NOT a
# coastd finding: `tsan.supp` scopes its suppression by `called_from_lib:libtbb.so`,
# which cannot match TBB's HEADER-inlined partitioner code (oneapi/tbb/partitioner.h)
# in a first-party TU. coastd's own threading is covered by the rest of the suite,
# which does run clean under TSan.
#
# Detected from the binary rather than plumbed through CMake so the gate stays
# honest when run by hand from any build directory.
#
# `grep -c`, NOT `grep -q`. This script runs under `set -o pipefail`, and `-q`
# exits at the FIRST match — which SIGPIPEs `nm` mid-stream, so the pipeline's
# status is nonzero precisely when the pattern IS present. The test would be
# silently inverted: it would skip on every build EXCEPT the TSan one. `-c` reads
# its input to the end, so nm exits 0 and the status means what it looks like.
if command -v nm >/dev/null 2>&1; then
    tsan_syms="$(nm -D "$COAST" 2>/dev/null | grep -c __tsan_init || true)"
    [ "$tsan_syms" = "0" ] ||
        skip "coast is TSan-instrumented; a TBB-parallel sim cannot make progress under TSan"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Absolute dataset paths: coastd runs the child with cwd = run_dir, so the
# config's relative ./data would not resolve there.
python3 - "$REPO" "$WORK" <<'PY'
import json, os, sys
repo, work = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{repo}/configs/mnist.json"))
for ds in cfg.get("datasets", {}).values():
    if isinstance(ds, dict):
        for v in ds.values():
            if isinstance(v, dict):
                for pk in ("image_file", "label_file"):
                    if pk in v and v[pk].startswith("./"):
                        v[pk] = os.path.join(repo, v[pk][2:])
json.dump(cfg, open(f"{work}/gate_cfg.json", "w"), indent=2)
PY

export COASTD_REAL_COAST="$COAST"
export COASTD_REAL_CONFIG="$WORK/gate_cfg.json"

# --no-skip is what un-hides the skip()-marked case; the filter keeps the
# binary's ordinary cases from running a second time here.
#
# ASSERT THE MATCH COUNT FIRST. doctest exits 0 with "test cases: 0 ... Status:
# SUCCESS!" when a --test-case filter matches nothing (verified against doctest
# 2.4.11), so renaming or rewording the TEST_CASE would turn this gate into a
# silent green tick that ran nothing. Same class as the repo's recorded
# comma-in-TEST_CASE-name trap. The count is exact, so a case going missing (or a
# second one appearing unannounced) fails loudly.
EXPECT_CASES=1
matched="$("$TESTBIN" --no-skip=true --list-test-cases --test-case="*real coast; opt-in*" \
           | sed -n 's/.*unskipped test cases passing the current filters: \([0-9]*\)/\1/p')"
if [ "${matched:-0}" != "$EXPECT_CASES" ]; then
    echo "coastd real-coast gate FAILED: filter matched ${matched:-0} case(s), expected $EXPECT_CASES." >&2
    echo "  The gate cases were renamed or removed — this gate would otherwise have" >&2
    echo "  reported SUCCESS while running nothing. Fix the filter or the case names." >&2
    exit 1
fi

"$TESTBIN" --no-skip=true --test-case="*real coast; opt-in*"
