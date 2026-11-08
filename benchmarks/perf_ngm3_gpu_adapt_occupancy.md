# ngm3_gpu adapt kernel — occupancy restructure (thread-per-dendrite → cell-parallel)

**Date:** 2026-06-05
**Branch:** feature/ngm3
**Pre-change baseline commit:** df0308d (S12 + ckpt-γ)
**Change:** `src/hd_ngm3/hd_ngm3_gpu.wgsl` — `adapt_main` entry point only (WGSL).
No `.cpp`/`.h`/binding/dispatch changes.

> **Config note (added 2026-06-05):** all numbers below were measured at the
> *then* config `neuron_cnt=64, proximal_cnt=4` (d_cnt=4). The canonical ngm3
> config was subsequently changed to **`neuron_cnt=50, proximal_cnt=5`** (d_cnt=5).
> The occupancy win is unchanged in kind — d_cnt=5 ≪ WG_N=64, so the old
> thread-per-dendrite map still left ~59/64 threads idle — but the exact SPS
> figures here are specific to the 64/4 measurement and were not re-run at 50/5.

## Problem

`adapt_main` is dispatched one workgroup per neuron, `WG_N = 64` threads. The
original cell map was **thread-per-dendrite**:

```wgsl
for (var d = t; d < d_cnt; d = d + WG_N) { ... sweep rep_cnt*input_size cells ... }
```

For `configs/mnist_ngm3_gpu_full.json` (neuron_cnt=64, apical_cnt=0,
proximal_cnt=4 → **d_cnt=4**, branch_depth=3 → rep_cnt=8, input_size=784):

- only **4 of 64 threads** do any work per neuron-workgroup (60 idle);
- each active thread sweeps `dendrite_block = rep_cnt*input_size = 6272` cells
  **serially** → critical path ≈ 6272 iterations.

This is the cost the S7 adapt step added: full-config SPS fell from ~5400 (S6,
pre-adapt) to ~260 (S7).

## Change

Restructured `adapt_main` to **outer-dendrite-serial, inner-cell-parallel**:

```
for d in [0, d_cnt):                                  // serial, uniform bound
    if (t == 0): build segment-tree fold -> workgroup-shared as_segw/as_sega
    workgroupBarrier
    for syn = t; syn < dendrite_block; syn += WG_N:    // all 64 threads
        per-cell perm / adapt_history / mismatch update
    workgroupBarrier
```

- The fold has a sequential dependency (the leaf "first to reach max_activity"
  rule + the tree max-fold), so thread 0 builds it (~rep_cnt≈8 iterations,
  negligible). It moved from **private** arrays (32+32 f32 × 64 threads ≈ 16 KB
  aggregate private pressure) to **workgroup-shared** (`as_segw`/`as_sega`,
  256 B total).
- The (rep,syn) sweep is now 64-way parallel. New critical path ≈
  `d_cnt × (dendrite_block / WG_N)` ≈ 4 × 98 ≈ 392 iterations + 4 folds +
  8 barriers — ~16× shorter.

**Correctness — bit-identical.** `adapt_main` performs no PRNG draws, and each
cell `(perm_base+syn)` is written by exactly one thread from read-only inputs
plus the deterministic per-dendrite fold. Reordering which thread does which
cell cannot change any result. Barriers sit in uniform control flow (the `d`
loop bound and all early-outs are per-neuron uniform).

## Measurements

**SPS** — `./build/coast -n 20000 configs/mnist_ngm3_gpu_full.json` (Release,
COAST_WITH_WGPU=ON, AMD Radeon 8060S / RADV STRIX_HALO / Vulkan), steady-state
SPS line at ~step 10k, run unsandboxed from repo root:

| Build | SPS (run 1) | SPS (run 2) |
|-------|-------------|-------------|
| baseline (thread-per-dendrite) | 258.9 | 261.8 |
| restructured (cell-parallel)   | 1994.5 | 1991.2 |

**≈ 7.6× speedup** (260 → 1992 SPS). Recovers ~37% of the S6 pre-adapt ceiling
(~5400 SPS); the remaining gap is the other per-step passes (response_combine,
winner, inhibition, branch) and the 2-submit/wait_idle per-step overhead — out
of scope for this node (adapt occupancy only).

**Correctness** — `./build-tests/coast_ngm3_gpu_probe`, all checks PASS with
numbers **identical** to baseline:

| Check | Baseline | Restructured |
|-------|----------|--------------|
| learning moved | 1605104/1605632 | 1605104/1605632 |
| learning mean\|Δ\| | 0.00840995 | 0.00840995 |
| learning max\|Δ\| | 0.679818 | 0.679818 |
| aggregate mean/max/nse | 0.12099 / 0.44765 / 0.96944 | 0.12099 / 0.44765 / 0.96944 |
| branch seg cells reassigned | 3608 | 3608 |

A-tier suite: `ctest -R 'integration\.mnist_ngm3'` → **4/4 pass** (philox,
cpu_to_gpu, gpu_to_cpu, heterogeneous).

## Notes / margins

- The fold is thread-0-serial; for very deep trees (branch_depth=4 → rep_cnt=16
  leaves) this is still ~16 iterations — negligible. If `dendrite_block <
  WG_N`, some threads idle in the sweep (harmless; all still reach the barrier).
- The win scales with `d_cnt × rep_cnt × input_size / WG_N` vs the old
  `dendrite_block`; i.e. it grows with rep_cnt·input_size and shrinks only if
  d_cnt ≥ WG_N (then the old map already had full occupancy). For the studied
  config d_cnt=4 ≪ 64, so the gain is large.
- Bank-generic: when an apical bank is present `adapt_main` is dispatched twice;
  the restructure is unchanged (reads `bd.d_cnt`/`bd.input_size`).
