# COAST Performance Baseline — mnist.json

**Date:** 2026-02-20
**Binary:** `./build/coast` (`-DCMAKE_BUILD_TYPE=Release`, `-O3`)
**Config:** `configs/mnist.json`
**Steps:** 500,000
**Model size:** 50 neurons × 5 dendrites × 784 synapses = 196,000 synapses

## Measurement Command

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses \
./build/coast -n 500000 configs/mnist.json
```

Run from project root. Hardware PMU events accessed with `:u` suffix semantics
(kernel.perf_event_paranoid = 2; user-space hardware events remain available).
`LLC-loads` / `LLC-load-misses` were `<not supported>` on this hardware;
`cache-references` / `cache-misses` are used instead (they map to LLC on x86).

## Raw Counter Values

| Counter | Value |
|---------|-------|
| Cycles | 11,845,925,341,315 |
| Instructions | 11,666,088,291,273 |
| Cache-references (≈ LLC) | 151,755,035,497 |
| Cache-misses (≈ LLC misses) | 21,302,068,987 |
| L1-dcache-loads | 6,775,596,913,354 |
| L1-dcache-load-misses | 77,979,495,822 |
| Branches | 1,557,058,817,938 |
| Branch-misses | 43,764,924,912 |
| Wall time | 85.416 s |
| User time (all threads) | 2512.808 s |

## Derived Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| IPC | **0.985** | instructions / cycles; theoretical max ≈ 4+ |
| L1 miss rate | 1.15% | dcache-load-misses / dcache-loads; acceptable |
| LLC miss rate | **14.04%** | cache-misses / cache-references; very high |
| Branch miss rate | 2.81% | branch-misses / branches; moderate |
| Implicit thread count | ~29.4 | user-time / wall-time |
| Steps/s | ~5,852 | 500k / 85.4 s |
| Wall time/step | ~170.8 µs | |
| CPU time/step (all threads) | ~5,026 µs | |
| LLC misses/step | ~42,604 | 21.3B / 500k |
| Instructions/synapse visit | ~59.5 | 11.67T / (500k × 2 × 196k) |
| Estimated memory stall/step | **~145 µs/thread** | LLC-misses/thread × 100 ns |
| Memory-stall fraction | **~85%** | stall / wall-time |
| DRAM read rate | ~15.96 GB/s | LLC-misses/s × 64 B/line |

## Analysis

### The fundamental bottleneck: LLC miss rate

With a 14% LLC miss rate and an IPC of only 0.985 (roughly **half** of what a
modern out-of-order core delivers on well-structured compute), the workload is
**severely memory-latency bound**. The back-of-the-envelope says ~85% of wall
time is spent waiting on DRAM rather than executing instructions.

This has an immediate and important consequence for the optimization strategy:

> **Compute-side improvements — faster exp/pow approximations, more efficient
> arithmetic, SIMD vectorization of the inner loops — cannot meaningfully
> reduce wall-clock time while DRAM latency dominates.** The CPU's OOO engine
> already hides latency by overlapping work wherever possible; the 0.985 IPC
> tells us it has run out of independent work to overlap.

This is exactly why the Schraudolph sigmoid benchmark (which showed 17× speedup
in isolation) delivered no improvement in production — sigmoid was not on the
critical path; DRAM latency was.

### Why 14% LLC miss rate?

The primary inner loops visit 784 synapses per dendrite. Each synapse stores:
- `weight` (float)
- `permanence` (float)
- `segment_idx` (uint16)
- `input_inc` (uint16 — pre-indexed step into input buffer)

That is 10 bytes per synapse, so 784 synapses ≈ 7.84 KB per dendrite.
A typical L2 cache is 256–512 KB; L3 is a few MB shared across cores.

With TBB parallelism across 50 neurons, each thread works on a different
neuron's dendrites. The working set per thread at any moment is small (one
dendrite = ~8 KB), but the *aggregate* working set for all threads is
50 × 5 × 7.84 KB ≈ 1.96 MB for weights/permanences alone — already
approaching or exceeding a shared L3 slice per NUMA node on many configurations.

Additionally, the `input_inc[]` field causes the inner loop to load from
**random positions** in the input buffer (`input_ptr[input_inc[k]]`). On
a 784-element input vector (float = 3.14 KB) this fits in L1, but the
indirect access pattern defeats hardware prefetching.

The `segment_activity[]` array (indexed by `segment_idx`) is also written
with a scatter pattern, introducing further pressure.

### Branch mispredictions (secondary concern)

43.8B branch mispredictions over 500k steps = ~87,500 mispredictions per
step. Each misprediction costs ~15–20 cycles. That accounts for roughly:

    87,500 × 17.5 cycles × (1/3.6 GHz) / 29.4 threads ≈ 14.4 µs/step

This is non-trivial (~8% of wall time) but secondary to the memory stalls.
The likely source is the data-dependent branch in the synapse loop:
`if (permanence > threshold)` — when the distribution of permanences is
close to the threshold, misprediction rates are high.

### Per-synapse instruction count

At ~59.5 instructions per synapse visit (counting two passes: `get_response`
+ `adapt_synapses`), the ISA-level efficiency appears reasonable. The issue
is not instruction count per se — it is that half of those instructions are
blocked waiting for data.

## Optimization Priority Revised

Given the profiling data, the optimization priority changes from the
compute-oriented proposals in `hot_path_analysis.md` to data-locality work:

1. **Reduce LLC working set** — Profile per-thread L3 occupancy; consider
   whether TBB's default task-stealing causes cache thrashing between cores.
   Using `tbb::static_partitioner` pins neurons to threads, potentially
   improving L3 reuse across steps.

2. **Software prefetching** — In the synapse inner loop, `input_ptr[input_inc[k]]`
   is an indirect load with 16–20 cycle latency. A `__builtin_prefetch` 8–12
   iterations ahead could hide that latency. Must be verified with perf — the
   gain depends on whether L1 or L2 already absorbs it.

3. **Branch prediction improvement** — Sort synapses by permanence bucket or
   use branchless arithmetic (`permanence > threshold` → multiplier 0.0 or 1.0)
   to eliminate the misprediction source.

4. **Input buffer layout** — If input data were interleaved with synaptic weights
   (i.e., store `weight` and the referenced `input_value` together), the indirect
   load latency would disappear. This requires restructuring the IO buffer contract.

5. **Compute-side** (lower priority until memory bottleneck is addressed) —
   Precomputed sigmoid params (`sigmoid_params_t`), branchless synapse math.
   These will not move the aggregate number until the memory stall fraction drops
   well below 50%.

## Reproducing This Measurement

```bash
# From project root
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

perf stat -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses \
./build/coast -n 500000 configs/mnist.json
```

Key numbers to watch in future comparisons:
- **IPC** (target: > 2.0)
- **LLC miss rate** (target: < 5%)
- **Steps/s** (currently: ~5,852)
