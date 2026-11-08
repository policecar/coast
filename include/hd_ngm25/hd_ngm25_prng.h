//
// Created 09.07.26 — ngm2.5 scaffold: stateless PRNG + penalty-policy seam.
// Measured basis: benchmarks/perf_hash_rng.md (2026-07-09). pcg_hash selected
// (production-identical with src/core/gpu/shared_prng.wgsl:12; χ² nominal).
// Known cost: ~5% on the response sweep vs the buffered mt19937 stream — a
// structural price (order-free draws, re-pack-stable keys, no checkpoint
// stream state), NOT a performance win. Do not "optimize" it back to a
// sequential stream; that reintroduces every problem it exists to solve.
//

#ifndef HD_NGM25_PRNG_H
#define HD_NGM25_PRNG_H

#include <algorithm>
#include <cstdint>

namespace ngm25 {

// Production hash, bit-identical to the GPU path (shared_prng.wgsl:12-16).
[[nodiscard]] inline uint32_t pcg_hash(uint32_t input)
{
    const uint32_t state = input * 747796405u + 2891336453u;
    const uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// [0,1) with the GPU's 2^-32 scaling (pcg_uniform).
[[nodiscard]] inline float to_unit(uint32_t h)
{
    return static_cast<float>(h) * (1.0f / 4294967296.0f);
}

// PITFALL P2 (the most important correctness detail in the module): keys are
// STRUCTURAL IDENTITY — (seed, step, dendrite, seg, input_idx) — never arena
// slot indices. Keying by slot makes every branch-event re-pack silently
// reshuffle all subsequent draws; no test attributes that correctly.
// Composition constants echo the GPU noise key (hd_ngm3_gpu.wgsl:299).
[[nodiscard]] inline uint32_t syn_key(uint32_t seed, uint32_t step,
                                      uint32_t dend, uint32_t seg,
                                      uint32_t input_idx)
{
    return seed
         ^ (step      * 3266489917u)
         ^ (dend      * 2246822519u)
         ^ (seg       * 668265263u)
         ^ (input_idx * 374761393u);
}

// CONSCIOUS NON-GOAL: cross-draw-site domain separation. The six draw sites
// (D1 response penalty, D2 combine noise seg=0, D3 winner threshold
// dend=0xFFFFFFFF seg=0, D4 split wiggle keyed on branch_epoch in the `step`
// slot, D5 Poisson init, D6 winner choice dend=0xFFFFFFFF seg=1) are
// distinguished for INTENT by sentinel arguments, but syn_key is an XOR of
// products with no keyed domain tag, so key SPACES are not disjoint: e.g. D4's
// branch_epoch can numerically equal a real `step` at which the same
// (dend,seg,input_idx) existed, aliasing that split draw onto a historical D1
// draw at birthday odds. This is accepted, not fixed — deliberately, per an
// explicit project decision: this is a neural model, not cryptography. An
// aliased pair is two UNRELATED quantities (a penalty threshold vs a wiggle vs
// an init permanence) with NO feedback path between them, so a collided draw is
// statistically indistinguishable from an independent one for learning
// dynamics; it can never produce a correlated, self-reinforcing error. The
// uniqueness that DOES matter — distinct draws within a single D1 sweep — is
// structurally guaranteed by the arena antichain invariant (checker inv4), not
// by key hygiene. Do NOT add per-site domain constants "to be safe": it changes
// every draw (forcing a full golden/GPU re-baseline) for zero model benefit.



// ---------------------------------------------------------------------------
// Penalty-policy seam (design §8). The response kernel is templated on one of
// these; all three are explicitly instantiated in the .cpp and dispatched
// ONCE per group per step (switch on params.penalty_policy). Hot loops carry
// zero runtime branches on the mode.
//
// Interface: penalty_weight(...) returns w ∈ [0,1]; the kernel applies
//   if (w > 0) { acc -= w * perm_strength * (1 - inp*inv_sum); clamp acc >= 0; }
// so stochastic_penalty (w ∈ {0,1}) reproduces ngm2's D1 semantics exactly
// (map R2 step 3), and expected_penalty replaces the sample by its mean.
// ---------------------------------------------------------------------------

// ngm2 semantics: draw ~ U(0, half_max); penalty iff draw > inp.
struct stochastic_penalty {
    [[nodiscard]] static float penalty_weight(uint32_t seed, uint32_t step,
                                              uint32_t dend, uint32_t seg,
                                              uint32_t input_idx,
                                              float inp, float half_max)
    {
        const float draw = to_unit(pcg_hash(syn_key(seed, step, dend, seg, input_idx)))
                         * half_max;
        return draw > inp ? 1.0f : 0.0f;
    }
};

// Deterministic expectation: P(U(0,half_max) > inp) = clamp(1 - inp/half_max).
// RESEARCH VARIANT (§8 matrix) — the fractional penalty interacts differently
// with the per-synapse clamp (P1) than a sampled all-or-nothing penalty;
// whether that matters is exactly what the learning-quality comparison
// measures. Validate before treating as equivalent.
struct expected_penalty {
    [[nodiscard]] static float penalty_weight(uint32_t, uint32_t,
                                              uint32_t, uint32_t, uint32_t,
                                              float inp, float half_max)
    {
        return half_max > 0.0f ? std::clamp(1.0f - inp / half_max, 0.0f, 1.0f)
                               : 0.0f;
    }
};

// Ablation: no penalty. Removing stochasticity outright measured NEGATIVE on
// learning quality in earlier experiments — research control only.
struct no_penalty {
    [[nodiscard]] static float penalty_weight(uint32_t, uint32_t,
                                              uint32_t, uint32_t, uint32_t,
                                              float, float)
    { return 0.0f; }
};

} // namespace ngm25

#endif // HD_NGM25_PRNG_H
