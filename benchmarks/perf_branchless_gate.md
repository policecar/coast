# COAST Performance — Branchless Synapse Gate (negative result)

**Date:** 2026-02-20
**Baseline:** `perf_static_partitioner.md` (tbb::static_partitioner active)
**Change tested:** `get_response()` outer synapse gate `if (perm > threshold)` converted to
float-multiply form + `if (< 0.0f)` clamp replaced with `std::max`
**Result: reverted — change is counterproductive**

## The Code Change

**Original:**
```cpp
if (synapses.permanence[i] > params.permanence_threshold) {
    segment_activity[synapses.segment_idx[i]] += *cur_inp_it;          // 1.1
    if (dis1(rgen) > *cur_inp_it) {                                    // 1.2
        const float inp_contrib   = *cur_inp_it * inv_sum;
        const float perm_strength = (synapses.permanence[i] - params.permanence_threshold) * inv_perm_range_;
        segment_activity[synapses.segment_idx[i]] -= perm_strength * (1.0f - inp_contrib);
        if (segment_activity[synapses.segment_idx[i]] < 0.0f)
            segment_activity[synapses.segment_idx[i]] = 0.0f;
    }
}
```

**Attempted branchless gate:**
```cpp
const bool active = synapses.permanence[i] > params.permanence_threshold;
segment_activity[synapses.segment_idx[i]] += static_cast<float>(active) * *cur_inp_it;
if (active) {
    if (dis1(rgen) > *cur_inp_it) {
        const float inp_contrib   = *cur_inp_it * inv_sum;
        const float perm_strength = (synapses.permanence[i] - params.permanence_threshold) * inv_perm_range_;
        segment_activity[synapses.segment_idx[i]] -= perm_strength * (1.0f - inp_contrib);
        segment_activity[synapses.segment_idx[i]] = std::max(segment_activity[synapses.segment_idx[i]], 0.0f);
    }
}
```

## Results

| Metric | Static part. (baseline) | + Branchless gate | Δ |
|--------|------------------------|-------------------|---|
| Cycles | 10,361,388,894,399 | 10,898,999,118,585 | **+5.2%** |
| Instructions | 11,535,759,882,337 | 12,377,513,649,079 | **+7.3%** |
| LLC miss rate | 2.72% | 2.60% | −4.4% |
| Branch miss rate | 2.52% | 2.55% | +1.2% |
| IPC | 1.113 | 1.136 | +2.1% |
| **Steps/s** | **6,547** | **6,069** | **−7.3%** |
| Wall time | 76.377 s | 82.385 s | **+7.9%** |
| Instructions/synapse | 58.9 | 63.2 | +7.3% |

## Why It Is Slower

### The branch was not eliminated

The transformation `if (perm > threshold) { seg += inp; ... }` →
`float gate = ...; seg += gate * inp; if (active) { ... }` does NOT
remove the branch from the machine code. `if (active)` is the same
comparison as before; the branch predictor sees the identical instruction
with the identical prediction difficulty.

What it does is split one branch into:
1. A branchless float-multiply for the `seg += inp` part
2. The same mispredicting branch (`if (active)`) for the penalty part

The branch misprediction count is therefore unchanged (+1.2% noise).

### Added instructions without benefit

The float-multiply path requires:
- `SETCC` or `VCMPSS` — convert comparison to 0/1
- `VCVTSI2SS` or `VMOVMSK` — convert int/mask to float
- `VMULSS` — multiply input by 0.0f or 1.0f
- `VADDSS` — accumulate

In the original, a correctly-predicted `if` (which is most iterations)
requires:
- `VCOMISS` / `JBE` — one compare + one taken branch
- `VADDSS` — accumulate (inside the taken path)

The original path is actually shorter when correctly predicted. The OOO
engine handles speculative execution efficiently and the cost of an
occasional misprediction (pipeline flush, ~17 cycles) is less than the
permanent overhead of 3 extra instructions per synapse × 98B synapse
visits.

### Instruction count increase confirms this

Instructions went from 11.54T to 12.38T (+7.3%), almost exactly matching
the step-rate decrease (−7.3%). The CPU was kept busier executing extra
float conversions and multiplies without doing any more useful work.

IPC improved slightly (1.113 → 1.136) because the extra instructions don't
stall — they just execute — confirming we are now instruction-throughput
bound in this path, not latency bound.

## Lesson

**The existing branch pattern is already near-optimal for the OOO
pipeline in this context.** The pattern has the form:

```
if (condition that is true ~50% of the time) {
    cheap_op();         // ~2 instructions
    if (another_cond) { // ~50% of outer-true cases
        moderate_work();
    }
}
```

On a modern superscalar core with branch prediction, this executes in
~2–3 cycles per iteration on average. The float-multiply alternative costs
~3–4 cycles unconditionally (due to the conversion + multiply chain) plus
the same misprediction penalty as before. It is strictly worse.

**To actually reduce branch mispredictions from this inner loop, one
would need to eliminate the branch entirely** — e.g., by sorting synapses
into active and inactive halves (separate sweeps, 100% branch prediction
accuracy each) or by replacing the stochastic penalty with a deterministic
smooth function. Both are larger algorithmic changes, not minor rewrites,
and would need profiling evidence that branch mispredictions are the
dominant remaining bottleneck before justifying the complexity.

## Status

Change reverted. Binary restored to static-partitioner-only state
(see `perf_static_partitioner.md`).
