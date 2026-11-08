# Perf: value-based hot-path error channel (error-handling redesign)

**Question.** Does replacing silently-compiled-out hot-path `assert()`s with the
always-compiled-in `COAST_REQUIRE` / `std::expected` channel (and reducing faults
across the TBB fan-out) cost measurable steps/sec?

**Script.** `scripts/bench_error_handling_sps.sh` — interleaved A/B: alternates a
PRE and a POST binary within one time window (so CPU thermal/boost drift on the
Strix Halo part hits both equally), each run reports its steady-state SPS, order
flips per round, paired by round. Commit the script before citing; this run used
it at the Phase-5 commit of branch `feat/error-handling`.

**Machine.** AMD Ryzen AI MAX+ 395 (Strix Halo), 16 Zen5 / 32 logical. Noise
floor ~±3% SPS; effects below ~5% need many trials to resolve.

## CPU (`configs/mnist_ngm3.json`, N=40k, 10 rounds, paired)

PRE = pre-Phase-1 (no error channel); POST = allocation-free reduction.

| | mean SPS | std |
|---|---|---|
| PRE  | 1419.0 | 35.0 (2.5%) |
| POST | 1400.7 | 28.4 (2.0%) |

- paired Δ = **−1.29%**, paired **t = −1.79** (df=9) → **NOT significant** (p≈0.10).
- Below the ±3% noise floor.

### Methodology note (why interleaving matters here)

A first, **non-interleaved** pass (POST measured, then PRE ~5 min later) showed
−1.78% with t=−2.99 — apparently significant. It was an artifact: the two
conditions sat in different thermal/boost windows, and that systematic drift was
confounded with the change. Interleaving (above) collapses it to non-significance.
Lesson: on a boost-clocking CPU, **always interleave A/B perf conditions**.

An intermediate POST that built a `tbb::combinable` per step measured −1.48%
(t=−1.67). Rewriting the reduction allocation-free (a single relaxed atomic flag +
a cold-path mutex; no per-step container construction) moved the point estimate
only to −1.29% — so the per-step `combinable` construction was **not** the cost.
The residual is the irreducible `expected`-return plumbing (process() returns a
value per entity per step) and/or binary-layout/I-cache noise from added code —
not per-step work proportional to the fault machinery.

## GPU (`configs/mnist_ngm3_gpu.json`, N=20k, 4 rounds, paired) — outside the sandbox

PRE = pre-Phase-1; POST = Phase-2 (GPU `check()` harmonized onto the channel).

| | mean SPS | std |
|---|---|---|
| PRE  | 1165.3 | 33.9 (2.9%) |
| POST | 1151.9 | 13.1 (1.1%) |

- Δ = **−1.15%**, within noise (the PRE std alone, 2.9%, exceeds the delta; per-round
  diffs swing both ways). The GPU ngm3 path is compute-bound (~500µs/dispatch GPU
  exec), so the host-side `check()`/`swap_io` change is structurally negligible.

## Conclusion

The value-based error channel is **within the measurement noise floor** on both
CPU and GPU. The point estimate is consistently a hair negative (~−1.2%) but never
statistically significant under a proper interleaved protocol. The granularity
argument holds: the per-step cost is fixed at K-entity granularity (single digits),
not synapse granularity, so it does not scale with the real work and shrinks as a
fraction on denser/heavier configs.
