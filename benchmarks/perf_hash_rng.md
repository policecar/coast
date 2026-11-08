# Hash-RNG micro-benchmark — ngm2.5 Demonstrate step (2026-07-09)

**Script:** `benchmarks/rng_hash_bench.cpp` (committed alongside this file)
**Machine:** AMD Ryzen AI MAX+ 395 (Strix Halo), single thread, `taskset -c 4`
**Build:** g++ 16.1.1, `-O3 -march=native -std=c++23`
**Method:** ngm2-response-shaped loop (transcription map R2 step 3: threshold
skip, segment accumulate, conditional penalty with per-synapse clamp) over
343,000 synapses (matched-config maturity), 200 steps/round, 7 measured
rounds, variants interleaved round-robin within each round, round 0 warmup
discarded. Working set ~2.4 MB (cache-resident): this isolates the
latency/ILP story, not DRAM bandwidth.

## Question

`design/ngm2_5_flat_sparse.md` §2 hypothesized that replacing the sequential
mt19937 penalty draw (D1) with a stateless per-synapse hash would relieve the
serial dependency chain behind ngm2's measured ~52.9 cyc/synapse
latency-bound profile, buying a single-threaded win.

## Result: hypothesis REFUTED

| variant | median ns/syn | paired ratio vs stream_mt |
|---|---|---|
| none (RNG-cost floor) | 4.113 | 1.103× |
| **stream_mt (production: fast_rng_t + uniform_real_distribution)** | **4.535** | 1× |
| hash_pcg (production GPU hash, structural key) | 4.793 | 0.946× |
| hash_sm32 (splitmix32 finalizer, same key) | 4.801 | 0.945× |

Per-round paired ratios are tight (hash_pcg 0.930–0.954× across 7 rounds);
penalty-branch fraction is 0.3743 for all RNG variants (work parity
confirmed).

Two facts, both contrary to the hypothesis:

1. **The RNG is only ~10% of the loop's cost** (`none` vs `stream_mt`). The
   latency-bound profile of this loop shape is *not* the RNG dependency chain
   — `mt19937_simd` amortizes generation into SIMD-filled buffers, so the
   per-draw cost is a buffer load, not a serial state update. The latency
   presumably lives in the `seg_activity` load-modify-store chain and the
   ~37%-taken data-dependent penalty branch. 10% is the hard ceiling on what
   *any* RNG change can buy here.
2. **The scalar hash is ~5.5% slower than the buffered SIMD stream.** Key
   composition (4 mul-xor) + 6-op finalizer per draw loses to an amortized
   buffered load. hash_sm32 ≡ hash_pcg within noise — no reason to deviate
   from the production hash.

## Hash quality (2^24 draws, production-shaped keys)

| hash | mean (exp 0.5) | var (exp 0.08333) | χ² df=255 (exp 255 ± 22.6) |
|---|---|---|---|
| pcg_hash (shared_prng.wgsl:12) | 0.499989 | 0.083319 | 244.9 |
| sm32_hash | 0.499874 | 0.083324 | 264.5 |

Both pass. **Selected: pcg_hash** — production-identical with the GPU path
(ADR-6), no measurable quality or speed difference from the alternative.

## Consequences for ngm2.5 (design doc updated accordingly)

- The hash RNG's justification is now **purely structural**: order-free draws
  make any parallel granularity safe, keys survive arena re-packs, and
  checkpoint stream-position state disappears. Its measured price is ~5% on
  the response sweep — negligible against the 4.6× traffic advantage that
  motivates ngm2.5, but it is a price, not a win.
- The `design/ngm2_5_flat_sparse.md` §12 "hash may relieve the latency chain,
  upside beyond ngm2" bullet is struck (measured, refuted). Expected
  single-thread response-sweep cost: ngm2-parity minus ~5%.
- If the 5% ever matters: hash draws are mutually independent, so a batched
  SIMD hash path (hash 8–16 keys per vector, buffer like mt19937_simd does)
  should recover it. Not worth building unless profiling of the real
  implementation says so.

## Limitations

- Cache-resident and single-threaded by design; says nothing about
  multi-threaded bandwidth behavior.
- Static input vector across steps (production input changes per step); the
  threshold-skip branch is therefore perfectly predictable here, as it is in
  production between adapt updates. The penalty branch stays data-dependent
  (draw varies per step).
- Loop shape is response-only; the adapt sweep has no per-synapse draw and is
  unaffected.
