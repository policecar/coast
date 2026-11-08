# h5-browser G-suite results (G1–G7)

**Date**: 2026-07-03
**Script**: `scripts/bench_h5browse.sh` + `scripts/bench_h5browse_driver.cpp`
(commit hash: pending — the h5-browser feature branch is not yet committed;
fill in when the scripts land, per the repo reproducibility convention)
**Machine**: AMD Ryzen AI MAX+ 395 (16 Zen5 cores, SMT), 128 GB LPDDR5x,
local NVMe Gen4 ext4; g++ 15 `-O3 -march=native -std=c++23`; libhdf5 2.1.1.
**Assets**: the retained design-verification files
(`.dew/design-verification/data/`, local workflow notes, not in the repository): `buf_4g_plain.h5` (5.03 GB logical,
uncompressed buffer log), `buf_4g.h5` (same data, shuffle+gzip-4),
`obs_out/dendrite.observe.h5` (1000 sites, writer-default 1024-element
chunks), `probe.h5`, `buf_small.h5` (2000×784), `corpus/` (46 seeded
malformed cases + manifest). Cold cache via `posix_fadvise(DONTNEED)`;
raw-read denominators measured in the same session (DVD caveat: cold raw
reads vary 4.9–6.5 GB/s across sessions).

## Results vs targets (targets from `.dew/docs/01-discover.md` §G, local workflow notes, not in the repository)

| Target | Criterion (good) | Measured | Verdict |
|---|---|---|---|
| G1 open → Tier-1 | < 0.5 s | 0.003 s (probe, 32 ds) … 0.051 s (observe, 1000 ds); 4 GB buffer logs 0.015–0.026 s | **PASS** |
| G2 first stats visible | < 1 s | 0.023–0.047 s | **PASS** |
| G2 sustained, uncompressed | ≥ ~85% of raw seq. read | 4.26 GB/s vs 5.84 GB/s raw = **73%** | **PASS per gate ruling** (see below) |
| G2 sustained, compressed | ≥ 85% of single-thread inflate bound (amended criterion, T1/H1d) | 0.56 GB/s vs 0.604–0.613 bound = **92%** | **PASS** |
| G3 interactive rebin | < 100 ms, no rescan | **0.001 ms** per rebin (re-aggregation of 512 base bins) | **PASS** |
| G4 pairwise ~2k × 784d | < 1 s | **0.431 s** (k=2000 cosine on 1.5M×784 buffer data, 1,999,000 pairs) | **PASS** |
| G5 structural search | < 100 ms per keystroke | **0.005 ms** (1000-dataset tree) | **PASS** |
| G6 malformed corpus | 0 crashes; actionable message per rejection | 47 files: 14 opened (damage contained per-dataset), 33 rejected with triage messages, **0 crashes** | **PASS** |
| G7 GUI-thread stall | < 50 ms | worst GUI-side call 0.01 ms across 1123 frames during a full 5 GB scan (headless proxy); interactive session user-validated | **PASS** |

## Notes

- **G2 uncompressed at 73%**: this is the documented single-thread
  accumulation ceiling (~4.4 GB/s; DVD H1a measured 67–86% on this
  workstation depending on drive state). The stage-gate decision (2026-07-03)
  accepted the ceiling: the design target medium (NAS, ~250 MB/s) passes at
  ~100%, and TBB-parallel accumulation remains a known extension (merges are
  associative by design). Any medium ≤ ~3.7 GB/s meets the 85% floor.
- **Observe-family scans** run at ~2.0 GB/s **file** throughput (matches the
  DVD's 2.09–2.13 GB/s small-chunk cap) but only 0.67 GB/s **logical**,
  because the writer's 1024-element chunks overallocate heavily on this
  1000-site file (1.82 GB file for 0.60 GB of data). Both effects are the
  writer-side chunk-size issue measured in T1 (65536-element chunks give
  +103%); the writer change was explicitly deferred out of this feature's
  scope (gate decision).
- **G6 detail**: metadata bit-flips can demote a file to "valid HDF5, but not
  a recognized COAST log file" (family markers destroyed) — a clean schema
  rejection, counted as such. Truncations are refused at open with
  eof/stored_eof triage text; flips in compressed chunk data are contained
  per-dataset (siblings scan to done). Silent value corruption in
  *uncompressed* chunk data remains structurally undetectable (no checksum
  on raw chunks) — known blind spot, documented in the DVD.
- **Kernel regression check**: `bench_h5b_kernel_throughput` (manual
  benchmark; until 2026-11 the ctest `unit.h5browse.kernel_perf`, which
  asserted ≥ 2.0 GB/s — removed from ctest because a throughput floor makes
  a test's verdict depend on the machine's load) measures single-thread
  accumulate; ~4.6 GB/s at the time of writing (DVD production rule 4 guard).
