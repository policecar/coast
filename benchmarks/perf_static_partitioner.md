# COAST Performance — Static Partitioner (vs Baseline)

**Date:** 2026-02-20
**Binary:** `./build/coast` (`-DCMAKE_BUILD_TYPE=Release`, `-O3`)
**Config:** `configs/mnist.json`
**Steps:** 500,000
**Model size:** 50 neurons × 5 dendrites × 784 synapses = 196,000 synapses
**Change:** `std::for_each(std::execution::par, ...)` → `tbb::parallel_for(..., tbb::static_partitioner{})`

## Measurement Command

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses \
./build/coast -n 500000 configs/mnist.json
```

## Raw Counter Values

| Counter | Baseline | Static partitioner | Change |
|---------|----------|--------------------|--------|
| Cycles | 11,845,925,341,315 | 10,361,388,894,399 | −12.5% |
| Instructions | 11,666,088,291,273 | 11,535,759,882,337 | −1.1% |
| Cache-references (≈ LLC) | 151,755,035,497 | 90,992,829,567 | −40.1% |
| Cache-misses (≈ LLC misses) | 21,302,068,987 | **2,477,845,419** | **−88.4%** |
| L1-dcache-loads | 6,775,596,913,354 | 6,438,141,968,928 | −5.0% |
| L1-dcache-load-misses | 77,979,495,822 | 69,727,054,783 | −10.6% |
| Branches | 1,557,058,817,938 | 1,540,747,537,996 | −1.0% |
| Branch-misses | 43,764,924,912 | 38,818,325,087 | −11.3% |
| Wall time | 85.416 s | **76.377 s** | **−10.6%** |
| User time (all threads) | 2512.808 s | 2220.267 s | −11.6% |
| Sys time | (not recorded) | 158.892 s | — |

## Derived Metrics

| Metric | Baseline | Static partitioner | Change |
|--------|----------|--------------------|--------|
| IPC | 0.985 | **1.113** | **+13.0%** |
| L1 miss rate | 1.15% | 1.08% | −6.1% |
| LLC miss rate | 14.04% | **2.72%** | **−80.6%** |
| Branch miss rate | 2.81% | 2.52% | −10.3% |
| Implicit thread count | ~29.4 | ~29.1 | — |
| Steps/s | ~5,852 | **~6,547** | **+11.9%** |
| Wall time/step | ~170.8 µs | ~152.8 µs | −10.6% |
| LLC misses/step | ~42,604 | **~4,956** | **−88.4%** |
| Est. memory stall/step | ~145 µs/thread | **~17 µs/thread** | **−88.3%** |
| Memory-stall fraction | ~85% | **~11%** | **−74 pp** |
| DRAM read rate | ~15.96 GB/s | **~2.08 GB/s** | **−87.0%** |
| Instructions/synapse visit | ~59.5 | ~58.9 | −1.0% |

## Analysis

### The LLC miss result

The LLC miss count dropped from **21.3 billion to 2.48 billion** — an 88% reduction — for
an identical workload. This is the largest single-change improvement recorded for this codebase.

The mechanism is straightforward. With `tbb::static_partitioner{}` the neuron range
`[0, 50)` is divided upfront into equal fixed chunks, one per worker thread, and those
assignments never change. Thread 0 always processes neurons 0–1, thread 1 always
processes neurons 2–3, and so on, for all 500,000 steps.

The default TBB partitioner (`auto_partitioner`) uses work-stealing: when a thread
finishes its chunk it steals tasks from the back of another thread's deque. This is
excellent for heterogeneous workloads where tasks vary in duration, but here all neurons
take essentially the same time (same dendrite/synapse count). Work-stealing adds no
load-balance benefit but **invalidates L3 cache affinity** — neuron N may be processed
by a different thread on every step, so its synapse data is never reused from cache.
Each step then requires loading all synaptic weights, permanences, and `input_inc`
arrays from DRAM.

With the static partitioner each thread "owns" its neurons. After a brief warmup
period, all synapse data for those neurons lives in that thread's L3 slice and is
reused every step from cache rather than fetched from DRAM.

### The warmup ramp

The per-step SPS output reveals this vividly:

```
Step 10k:   4,616 SPS   ← cold cache, loading from DRAM
Step 50k:   5,367 SPS
Step 100k:  5,944 SPS
Step 200k:  6,871 SPS   ← L3 fully warm
Step 300k:  6,738 SPS
Step 400k:  7,231 SPS   ← peak sustained
```

The baseline showed far less ramp-up because work-stealing never builds persistent
cache affinity. After warmup, the static-partitioner binary sustains 6,500–7,800 SPS.

### What the new numbers say about remaining bottlenecks

With the memory-stall fraction now at ~11%, the bottleneck profile has shifted:

| Bottleneck | Baseline fraction | Now |
|------------|-------------------|-----|
| DRAM stalls | ~85% | ~11% |
| Compute + L1/L2 + other | ~15% | ~89% |

The IPC increased from 0.985 to 1.113 — better, but still well below the 2–4+
that this core architecture can sustain on compute-bound work. The residual
is likely L2 miss latency (data that fits in per-thread L3 but not L1/L2), branch
mispredictions, and the serialized inner loops in the dendrite code.

The 11.3% reduction in branch mispredictions (44B → 39B) comes for free from
thread affinity: each thread works on "its" neurons across steps, so branch
predictors build up history for the same code paths and data patterns.

### Sys time observation

Sys time of 158.9 seconds appeared in this run (not recorded in the baseline).
This is ~7.2% of total CPU time. Possible sources:
- TBB's internal synchronization barriers between `parallel_for` calls now
  spin-wait or block differently with the static partitioner.
- Memory-mapping activity as TBB initializes fixed thread assignments.

This warrants monitoring in future measurements; it is currently smaller than
the gain but is non-trivial.

### Next most impactful bottleneck

Given the current IPC of 1.113 and the residual ~11% memory stall fraction, the
next measurable bottleneck is likely **L2 cache pressure within each thread's owned
neurons**. Each thread now holds ~2 neurons × 5 dendrites × 784 synapses × 10 B
≈ ~78 KB in its working set. A typical L2 is 256 KB — enough for 3 threads' full
working sets, but with the indirect `input_ptr[input_inc[k]]` access pattern, L2
prefetch effectiveness is limited.

Candidate: **software prefetch** of `input_ptr[input_inc[k+N]]` in the synapse
inner loop (benchmarks/perf_baseline.md priority #2). But profile first — the ~11%
remaining memory stall may already be hiding behind branch-miss latency.

## Reproducing This Measurement

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

perf stat -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses \
./build/coast -n 500000 configs/mnist.json
```

The change is in `src/hd_ngm2/hd_ngm2_neuron_group.cpp`: two `std::for_each`
calls with `std::execution::par` replaced by `tbb::parallel_for` with
`tbb::static_partitioner{}`.
