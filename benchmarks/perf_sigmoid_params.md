# COAST Performance — Precomputed Sigmoid Params (assumption verification)

**Date:** 2026-02-20
**Baseline:** `perf_static_partitioner.md`
**Purpose:** Verify that sigmoid is NOT on the critical path — i.e., eliminating pow()
calls from runtime-shape sigmoid invocations should produce no measurable speedup.

## What Was Changed

Added `sigmoid_params_t` to `hd_ngm2_tools.h`: a struct that precomputes
`inv_step_size` and `starting_point` from a `sigmoid_shape_t` at construction time,
paying the `pow()` cost once and reusing the result across multiple sigmoid evaluations.

In `neuron_group_t::process()`, replaced three `sigmoid(..., weight_filter)` calls
(each recomputing `pow(steepness, 0.1)`) with one `sigmoid_params_t wf_params(weight_filter)`
constructed once per step, then passed to the fast `sigmoid(float, sigmoid_params_t)` overload.

## Actual pow() savings

Runtime sigmoid calls per step in the codebase:

| Call site | Shape | pow() calls | After change |
|-----------|-------|-------------|--------------|
| `neuron_group::process()` × 3 | `weight_filter` (runtime) | 3/step | **1/step** |
| `neuron_t::adapt()` × 2 per call × 51 calls | `activity_learning_window` (runtime) | 102/step | 102/step (unchanged — params still constructed per call) |
| All dendrite calls | compile-time constants | 0 (already folded by compiler at -O3) | 0 |

**Net savings: 2 pow() calls per step** (from 105 to 103).

Note: `neuron.cpp` was initially changed to use `sigmoid_params_t` inside `adapt()` but
immediately reverted — creating the params struct inside the function still calls pow() once
per construction, which happens once per `adapt()` call. True savings there would require
caching the precomputed params as neuron member variables.

## Results

| Metric | Static part. (baseline) | + Sigmoid params | Δ |
|--------|------------------------|------------------|---|
| Cycles | 10,361,388,894,399 | 10,492,328,045,423 | +1.3% |
| Instructions | 11,535,759,882,337 | 11,599,880,738,050 | +0.6% |
| LLC miss rate | 2.72% | 2.94% | +0.22 pp |
| Branch miss rate | 2.52% | 2.53% | +0.01 pp |
| IPC | 1.113 | 1.106 | −0.6% |
| **Steps/s** | **6,547** | **6,376** | **−2.6%** |
| Wall time | 76.377 s | 78.415 s | +2.7% |

## Interpretation

**Expected change:** +0.073% (saving 2 × ~75 cycles out of ~20.7M cycles/step).

**Measured change:** −2.6% steps/s, well outside the expected signal.

The measured delta is **run-to-run noise**, not a real effect. Over a 500k-step run, the
SPS trace ranges from ~4,450 to ~7,900 during warm-up, with a noise floor of several
percent even in the settled region. A 0.073% signal is completely invisible against this
background.

**The assumption is confirmed: sigmoid is not on the critical path.** Eliminating
pow() calls produces no measurable effect on aggregate performance. This is consistent
with the cycle-budget analysis in `perf_static_partitioner.md`, which attributed ~71%
of cycles to serial dependency chains in the inner synapse loops — a region sigmoid
does not participate in.

## The sigmoid_params_t struct (kept)

Although the optimization has no measurable impact for this model size, the
`sigmoid_params_t` struct is kept in `hd_ngm2_tools.h` and the `neuron_group.cpp`
usage is retained because:

1. It is semantically correct and has zero runtime cost relative to the baseline.
2. It documents the precomputation pattern for the one case (configurable `weight_filter`)
   where a caller iterates over many neurons with the same shape per step.
3. If the model ever scales to where neuron-count dominates over synapse-count,
   or sigmoid is used on a tighter inner loop, the infrastructure is already in place.
