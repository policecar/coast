# Sigmoid Benchmark — Findings

**File:** `sigmoid_bench.cpp`
**Compile:** `g++ -std=c++23 -O3 -march=native -I ../include -o sigmoid_bench sigmoid_bench.cpp`
**Date:** 2026-02-19

## What we were looking at

`ngm2::sigmoid` in `include/hd_ngm2/hd_ngm2_tools.h`:

```cpp
struct sigmoid_shape_t {
    double steepness        = 0.5;   // double — required for clean JSON serialization
    double transition_point = 0.5;
};

inline float sigmoid(float x, sigmoid_shape_t shape = {0.5, 0.5}) {
    const float inv_step_size  = 1.0f / (1.0f - std::pow(shape.steepness, 0.1f));
    const float starting_point = -shape.transition_point * inv_step_size;
    return 1.0f / (1.0f + std::exp(-(x * inv_step_size + starting_point)));
}
```

Called from neuron group learning (per-winner loop), dendrite attenuation, and
neuron activity gating — with both fixed and varying shapes.

## Machine-operation analysis

With `double` shape members, `std::pow(shape.steepness, 0.1f)` promotes `0.1f`
to `double` and returns `double`. The entire `inv_step_size` / `starting_point`
computation therefore runs in **double precision** before being truncated to
`float`. This is unintentional but harmless in terms of correctness.

Estimated cost per call (variable shape, no constant-folding):

| Operation | Cycles |
|-----------|--------|
| `pow(double, double)` — internal `exp(0.1·log(x))` | ~50–100 |
| double `sub` + `div` for `inv_step_size` | ~20–25 |
| double `mul` + narrowing for `starting_point` | ~5 |
| float FMA + negation | ~5 |
| `expf` | ~20–40 |
| float `add` + `div` | ~15–20 |
| **Total** | **~120–190** |

Dominant costs: `pow` and `expf`.

## Implementations benchmarked

| ID | Description |
|----|-------------|
| v0 | Original (double shape members, `std::exp`) |
| v1 | Float-clean: change shape members to `float`, use `powf`/`expf` |
| v2 | Precomputed params: cache `inv_step_size` and `starting_point` in a struct, eliminating `pow` from the hot path |
| v3 | Precomputed params + fast polynomial `exp` (~1e-6 relative error) |
| v4 | Precomputed params + Schraudolph (1999) bit-hack `exp` (~5% max error, 3 integer instructions) |

## Microbenchmark results (N = 100 000 000 calls)

### A. Fixed default shape — compiler constant-folds `pow(0.5, 0.1)` at -O3

| Variant | ns/call | vs v0 |
|---------|---------|-------|
| v0 original | 2.42 | 1.0× |
| v1 float-clean | 2.40 | 1.0× |
| v2 precomputed + expf | 2.41 | 1.0× |
| v3 precomputed + fast poly exp | **0.49** | **4.9×** |
| v4 precomputed + Schraudolph | **0.41** | **5.9×** |

GCC at -O3 constant-folds `pow(0.5, 0.1)` when the shape is a compile-time
constant, so v0, v1, and v2 are identical. The speedup for v3/v4 comes
entirely from the compiler auto-vectorizing the loop body (AVX SIMD) once
`expf` is replaced with pure arithmetic.

### B. 4 rotating shapes — forces `pow()` recomputation on every call

| Variant | ns/call | vs v0 |
|---------|---------|-------|
| v0 original | 9.75 | 1.0× |
| v1 float-clean | 6.50 | 1.5× |
| v2 precomputed + expf | 2.42 | 4.0× |
| v3 precomputed + fast poly exp | 2.38 | 4.1× |
| v4 precomputed + Schraudolph | **0.57** | **17.1×** |

The ~7.3 ns overhead of v0 over v2 is entirely `pow(double, double)`.
Changing shape members to `float` (v1) gives a free 1.5× by switching to
`powf`. v3 does **not** outperform v2 here because the `params4[i & 3]`
indirect access prevents the compiler from hoisting shape constants into
registers, blocking SIMD vectorization of the polynomial. v4 stays fast in
both scenarios because its integer-only arithmetic vectorizes trivially even
with indirect shape access.

## Accuracy

Measured over x ∈ [-5, 5] against v0 as reference:

| Variant | Max relative error |
|---------|-------------------|
| v1 float-clean | 0 |
| v2 precomputed + expf | 0 |
| v3 fast poly exp | 2.89e-6 |
| v4 Schraudolph | 3.97e-2 (~4%) |

## What happened when v4 was applied to the production binary

The Schraudolph fast exp (v4) was implemented as an `#ifdef NGM2_FAST_SIGMOID`
branch inside the existing `sigmoid` function and tested in the full build.

**Result: ~1.5–2% slower than the original.**

### Why the microbenchmark didn't predict this

1. **sigmoid is not on the critical path.** The surrounding loops load neuron
   activations and weights from memory. Memory latency dominates; the CPU's
   OOO engine already hides `expf` behind cache-miss stalls. Making exp cheaper
   saves nothing — the bottleneck is elsewhere.

2. **Integer ↔ float domain crossing.** The Schraudolph union trick delivers
   its result in the integer execution domain. Crossing to the float domain via
   `MOVD` incurs a 2–3 cycle bypass penalty on Skylake-class cores. In the
   isolated benchmark this is hidden by independent iterations overlapping it.
   In production the sigmoid result feeds directly into the next float
   operation, making the penalty visible on the critical dependency chain.

3. **The benchmark's speedup was from auto-vectorization, not exp speed.**
   In the tight isolated loop the compiler could SIMD-vectorize the entire
   body. In production, sigmoid sits inside complex control flow — no
   auto-vectorization occurs. Without SIMD the per-call advantage of v4
   disappears, leaving only the domain-crossing overhead.

## Lessons

- Profile first. The right question before any of this would have been: *what
  fraction of total cycles does sigmoid account for?* If it is a small
  fraction, no exp-level optimization can move the aggregate number.

- Microbenchmarks measure peak throughput in ideal conditions. They tell you
  the upper bound of what an optimization can do, not what it will do in
  context.

- The compiler's constant-folding of `pow` for fixed shapes means the original
  code is already near-optimal for the most common call sites.

- If a future hot path does make sigmoid the measured bottleneck, the right
  lever is `sigmoid_params_t` (precomputed shape constants) combined with
  a fast exp, and only after verifying with profiler data that the gain is
  real in context.
