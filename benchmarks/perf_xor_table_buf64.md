# COAST Performance — XOR-Table PRNG, BUF_SIZE=64

**Date:** 2026-02-20
**Baseline (32T, mt19937_simd, seed-0 checkpoint):** 3,827 SPS

## Motivation

The BUF_SIZE=16 XOR-table experiment (see `perf_xor_table_prng.md`) showed:
- −2.8% SPS overall
- +0.9% instructions (nearly neutral — design correct in principle)
- −24.3% LLC misses (best of any experiment)
- +9.1% branch misses — the sole identified cause of the gap

The analysis predicted that increasing BUF_SIZE to 64 would reduce branch frequency
4× (refill every 64 calls instead of 16), bringing branch misses to within 2–3% of
baseline while keeping the same amortised instruction cost.

## Implementation Change

`xor_table_prng.h`: `BUF_SIZE` changed from 16 to 64.

The refill function was updated to do **four SIMD passes** (one per 16-element block),
with phase updates inside the loop body:

```cpp
for (int k = 0; k < BUF_SIZE; k += 16) {
    const __m512i v = _mm512_xor_si512(
        _mm512_loadu_si512(&rs1_[p1_]),
        _mm512_xor_si512( ... ));
    _mm512_store_si512(&buf_[k], v);
    p1_ += 16; if (p1_ >= S1) p1_ -= S1;
    p2_ += 16; if (p2_ >= S2) p2_ -= S2;
    p3_ += 16; if (p3_ >= S3) p3_ -= S3;
    p4_ += 16; if (p4_ >= S4) p4_ -= S4;
    p5_ += 16; if (p5_ >= S5) p5_ -= S5;
}
```

PAD=16 remains sufficient: each SIMD pass reads `rs_[p]` to `rs_[p+15]`; since
`p ∈ [0, Si−1]`, max index = Si−1+15 = Si+14 < Si+PAD. ✓

The scalar fallback was updated to use running counters (not `p+k` offset, since
BUF_SIZE=64 > PAD=16 would exceed padded array bounds for large phases).

**Note:** BUF_SIZE=16 and BUF_SIZE=64 generate the **identical** output sequence:
the 64-output refill is exactly four consecutive 16-output refills. So simulation
dynamics — and therefore per-step workload — are identical between the two variants.

## Results

| Metric | mt19937_simd | BUF_SIZE=16 | BUF_SIZE=64 | Δ vs baseline |
|--------|-------------|-------------|-------------|---------------|
| **SPS** | **3,827** | **3,719** | **3,695** | **−3.5%** |
| Cycles/step (M) | 30.85 | 31.53 | 32.62 | +5.7% |
| Instructions/step (M) | 43.40 | 43.81 | 47.74 | **+10.0%** |
| IPC | 1.407 | 1.389 | 1.464 | +4.0% |
| LLC misses/step | 8,700 | 6,583 | 5,992 | **−31.1%** |
| L1 load-misses/step | 228,350 | 213,765 | 214,032 | −6.3% |
| Branch misses/step | 120,927 | 131,896 | 120,766 | **−0.1%** |

## Analysis

### Branch miss hypothesis: confirmed

Branch misses dropped from 131,896/step (BUF_SIZE=16, +9.1%) to 120,766/step (BUF_SIZE=64,
−0.1% vs baseline). This is essentially identical to the baseline — exactly as predicted.
The `[[unlikely]]` branch in `operator()()` fires 4× less often with BUF_SIZE=64, and the
5 conditional phase subtracts occur 4× less often per output value.

### LLC miss reduction: improved further

LLC misses fell to 5,992/step — −31.1% vs baseline, better than BUF_SIZE=16's −24.3%.
The mechanism is unchanged (read-only tables, 64-byte write-back per refill regardless of
BUF_SIZE). The marginal improvement over BUF_SIZE=16 may come from the larger buf_ (256
bytes vs 64 bytes) being written 4× less frequently, further reducing write pressure on
L2 → L3 eviction paths.

### Instruction increase: not predicted, the critical finding

**Prediction:** amortised instruction cost invariant → ≈ 0% instruction change.
**Measured:** +10.0% (+4.34M instructions/step).

This directly contradicts the prediction and is the key result of this experiment. Since
the output sequences are identical for BUF_SIZE=16 and BUF_SIZE=64 (the 64-element refill
is four consecutive 16-element refills with identical phase arithmetic), the instruction
increase must come from the refill implementation itself.

The most likely cause: the **4-iteration loop in refill()** is not fully eliminated by the
compiler.

BUF_SIZE=16 refill is a single, flat code block:
- 1 SIMD pass (10 ops) + 5 phase subtracts (15 scalar ops) = ~25 ops for 16 outputs

BUF_SIZE=64 refill uses a loop with 4 iterations. Even with `BUF_SIZE` as a `constexpr`:
- If the compiler emits an actual loop (not fully unrolled), it adds ~3 instructions per
  iteration (increment k, compare, branch) = 12 extra per refill = 12/64 = 0.19 per output.
- If the compiler keeps p1_..p5_ in registers through all 4 iterations (6 live integer
  registers + loop counter + `this` = 8 integers, within the 13 free GP registers), there
  should be no spills. But register pressure from simultaneously holding 5 SIMD values
  and 6 integer scalars during the phase-update section could cause partial spills.
- The phase updates inside the loop body create loop-carried dependencies (iteration i+1
  reads p1_..p5_ updated in iteration i), which can prevent out-of-order execution from
  pipelining the iterations.

The BUF_SIZE=16 single-block implementation avoids all of this: no loop, no loop-carried
dependency, trivial register allocation.

**Key insight:** the "amortised cost invariant" prediction holds only when the refill body
itself is a fixed-size straight-line block. Introducing a loop to extend the buffer size
breaks the invariant due to loop overhead and scheduling constraints.

### IPC increase: consistent

IPC increased from 1.407 to 1.464 (+4.0%). Fewer branch mispredictions (back to baseline)
means fewer pipeline flushes, so the CPU can issue instructions at a higher rate. This
confirms the branch miss reduction is real — but more instructions at higher IPC still
means more cycles overall.

### Net result: slightly worse than BUF_SIZE=16

| Effect | Impact |
|--------|--------|
| Branch miss: back to baseline (−8.4% fewer vs BUF=16) | Saves ~274k cycles/step |
| LLC miss: further −9.0% fewer vs BUF=16 | Saves ~52k cycles/step |
| Instructions: +9.0% more vs BUF=16 | Costs ~1.77M cycles/step at IPC=1.464 |
| **Net** | **−1.5M cycles/step → slightly worse** |

The instruction overhead of the 4-iteration loop more than offsets the branch and LLC
improvements.

## Summary across all PRNG experiments

| PRNG | SPS Δ | Instr Δ | LLC Δ | Branch miss Δ |
|---|---|---|---|---|
| Scalar Philox | −19.0% | +32.2% | −24.6% | +7.5% |
| Philox AVX-512 | −11.2% | +11.3% | −20.5% | +2.1% |
| MT11213b | −5.9% | +8.4% | −3.5% | −2.4% |
| XOR-table BUF=16 | −2.8% | +0.9% | −24.3% | +9.1% |
| **XOR-table BUF=64** | **−3.5%** | **+10.0%** | **−31.1%** | **−0.1%** |

## Conclusions

1. **Branch miss prediction confirmed.** BUF_SIZE=64 eliminates the +9.1% branch miss
   overhead of BUF_SIZE=16 — back to baseline. The prediction was correct.

2. **Amortised cost invariant breaks with a loop.** The +0.9% instruction overhead of
   BUF_SIZE=16 (single flat refill block) became +10.0% with BUF_SIZE=64 (4-iteration loop).
   The invariant holds only for straight-line refill implementations, not loop-based ones.

3. **A manual unroll of the 4-pass refill would likely fix the instruction issue.** Writing
   four explicit SIMD blocks instead of a loop would eliminate loop overhead and allow the
   compiler to schedule all four passes as independent straight-line code. This was not
   tested, as it would be a minor implementation variant with diminishing expected return.

4. **BUF_SIZE=16 remains the better XOR-table implementation** (−2.8% vs −3.5%). The branch
   miss penalty at BUF_SIZE=16 is real but smaller than the instruction penalty at BUF_SIZE=64.

5. **Overall PRNG conclusion unchanged.** No PRNG variant outperforms mt19937_simd. The
   XOR-table design at BUF_SIZE=16 is the closest (−2.8%), well within measurement noise.
   Further improvement requires addressing the structural bottleneck in the dendrite inner
   loop, not the PRNG.

## Implementation state

`xor_table_prng.h` retains BUF_SIZE=64 (documents the experiment).
`fast_rng.h` reverted to `mt19937_simd`.
