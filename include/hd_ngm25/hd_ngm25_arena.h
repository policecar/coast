//
// Created 09.07.26 — ngm2.5 scaffold: the flat-sparse synapse arena.
// Layout contract: design/ngm2_5_flat_sparse.md §3. One arena per group
// (decision §13.1); traffic ∝ realized structure by construction.
//

#ifndef HD_NGM25_ARENA_H
#define HD_NGM25_ARENA_H

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "hd_ngm25/hd_ngm25_cfg.h"

namespace ngm25 {

// All synapses of all dendrites of one group, SoA, dendrites contiguous.
// Within a dendrite: INPUT-MAJOR order (ascending input_idx, clones of the
// same input adjacent) — matches ngm2's sweep order, gives monotonic input
// access, and makes re-packing a stable partition (map R2/R8).
//
// PITFALL P7: capacity growth at branch events invalidates pointers/spans.
// Accessors below hand out INDEX RANGES, not spans; nothing outside a single
// step may cache addresses into these vectors.
struct synapse_arena_t {

    // --- per-synapse SoA (size == syn_count()) ---
    std::vector<float>    perm;         // permanence [0,1]
    std::vector<uint16_t> input_idx;    // explicit input position — THE ngm2.5
                                        // change that kills the input_inc
                                        // prefix-sum coupling (map R2)
    std::vector<uint8_t>  seg;          // segment id, heap-indexed, root = 1;
                                        // valid range [1, 2^(depth+1) - 1]
    std::vector<float>    adapt_hist;   // accumulated theta (reset on split)
    std::vector<float>    mismatch;     // IIR mismatch (reset on split)

    // --- dendrite directory (dend_cnt = neuron_cnt * (apical+proximal)) ---
    std::vector<uint32_t> dend_offset;  // size dend_cnt + 1; CSR offsets,
                                        // dendrite d owns [off[d], off[d+1])
    std::vector<bank_t>   dend_type;    // apical | proximal per dendrite
    std::vector<uint16_t> dend_neuron;  // owning neuron per dendrite

    [[nodiscard]] std::size_t syn_count()  const { return perm.size(); }
    [[nodiscard]] std::size_t dend_count() const
    { return dend_offset.empty() ? 0 : dend_offset.size() - 1; }

    // Index range of dendrite d — deliberately not a span (P7).
    struct range_t { uint32_t begin, end; };
    [[nodiscard]] range_t dendrite(std::size_t d) const
    { return { dend_offset[d], dend_offset[d + 1] }; }
};

// Group shape derived from params + resolved channel routing; needed by the
// consistency checker and the kernels.
struct group_shape_t {
    std::size_t neuron_cnt   = 0;
    std::size_t apical_cnt   = 0;      // per neuron
    std::size_t proximal_cnt = 0;      // per neuron
    uint8_t     max_depth    = 0;      // decision §13.2: <= 6
    uint8_t     max_seg_id   = 0;      // 2^(depth+1) - 1
    // Per-type routed input widths. input_idx values of a dendrite of type t
    // must fall inside type t's positions of the group-wide concatenated
    // input buffer (§13.1). Branch formulas use the TYPE's width, not the
    // group total (map R8 / §13.1).
    std::size_t input_size[2]   = {0, 0}; // indexed by bank_t
    // Position ranges of each type's channels in the concatenated buffer.
    // Simplest v1: route apical channels first, then proximal, so each type
    // owns one contiguous [lo, hi) range. Revisit only if a config needs
    // interleaved routing.
    std::size_t input_lo[2] = {0, 0};
    std::size_t input_hi[2] = {0, 0};
};

// Heap-indexed segment-tree geometry (invariant 4 support). A segment id is a
// node in the binary rep-tree (root = 1, children of s are 2s / 2s+1). Each
// node covers a dyadic sub-interval of the rep axis; we represent it as an
// INTEGER interval [lo, hi) over [0, 2^max_depth) so the antichain-cut check
// is exact (no float rounding). Exposed for direct unit testing — invariant 4
// is the subtle one (plan I1).
namespace detail {

// Depth of a heap node (root = depth 0). seg must be >= 1.
[[nodiscard]] inline int seg_depth(uint8_t seg)
{ return std::bit_width(static_cast<uint32_t>(seg)) - 1; }

struct seg_span_t { int lo, hi; };

// Integer dyadic interval of `seg` over the leaf axis [0, 2^max_depth).
// Precondition: 1 <= seg <= 2^(max_depth+1) - 1 (i.e. seg_depth(seg) <=
// max_depth), so the shift never goes negative — the caller (invariant 5)
// checks the seg range before this is used.
[[nodiscard]] inline seg_span_t seg_interval(uint8_t seg, uint8_t max_depth)
{
    const int d    = seg_depth(seg);
    const int span = 1 << (static_cast<int>(max_depth) - d); // leaves under seg
    const int pos  = static_cast<int>(seg) - (1 << d);       // index within depth d
    const int lo   = pos * span;
    return { lo, lo + span };
}

} // namespace detail

// ---------------------------------------------------------------------------
// Consistency checker — WRITE AND WIRE THIS FIRST (design §9). Re-packing
// bugs corrupt silently and surface as "learning got worse", the most
// expensive failure mode to localize. Run after init and after every branch
// event in debug/test runs; introspection accessors build ON TOP of the
// invariants it guarantees, never in parallel with it.
//
// Invariants to enforce (return false + fill *err with the first violation):
//   1. dend_offset monotone non-decreasing; dend_offset.back() == syn_count();
//      all SoA vectors equal length.
//   2. Directory shape matches group_shape_t (dend_count, per-neuron dendrite
//      counts and types in fixed order).
//   3. Per dendrite: input_idx ascending (input-major), clones adjacent;
//      every input_idx within the dendrite's type range [input_lo, input_hi).
//   4. Per (dendrite, input): the seg-id set is a valid ANTICHAIN CUT of the
//      binary tree — no seg is an ancestor of another, and the union of their
//      rep-axis intervals covers the root exactly (each input's split tree is
//      complete: a split always produced BOTH children, map R8).
//   5. All seg ids in [1, max_seg_id]; permanences in [0,1]; adapt_hist and
//      mismatch finite and >= 0.
// ---------------------------------------------------------------------------
[[nodiscard]] bool check_arena_consistency(const synapse_arena_t &arena,
                                           const group_shape_t   &shape,
                                           std::string           *err);

// Branch-event re-pack (map R8, design §7). Executed OUTSIDE the hot path,
// fires ~1/branch_interval steps per neuron.
//
// Contract:
//  - `clone_flags` marks qualifying synapses (R8 decide predicate, evaluated
//    by the caller); repack produces a NEW arena where each flagged synapse
//    becomes two children (seg*2, seg*2+1) with adapt_hist = mismatch = 0 and
//    permanences = parent ± wiggle (D4 hash draws keyed on
//    (dend, NEW seg, input_idx, branch_epoch) — two independent draws).
//  - Preserves input-major order and clone adjacency (invariant 3).
//  - Capacity-doubling growth; amortized O(1) per synapse.
//  - Must leave check_arena_consistency() true — asserted in tests.
void repack_with_splits(synapse_arena_t          &arena,
                        const std::vector<uint8_t> &clone_flags, // size syn_count()
                        const group_shape_t      &shape,
                        uint32_t                  rng_seed,
                        uint32_t                  branch_epoch);

// Initial arena construction (RNG site D5, plan I1). Builds a group's synapse
// arena from its resolved shape. Postcondition:
// check_arena_consistency(result, shape, ...) == true.
//
//  - Dendrite directory order — THE fixed order that invariant 2 enforces:
//    neuron-major; within each neuron, apical dendrites (shape.apical_cnt)
//    first, then proximal (shape.proximal_cnt). dend_type / dend_neuron record
//    it so nothing downstream depends on the order implicitly.
//  - One synapse per (dendrite, routed-input position): a dendrite of type t
//    gets shape.input_size[t] synapses, input_idx running input_lo[t] ..
//    input_hi[t]-1 ascending (input-major), all seg = 1 (root), adapt_hist =
//    mismatch = 0.
//  - Permanences: Poisson(100*perm_threshold)/100, clamped to [0,1] — ngm2's
//    D5 formula (ngm::cold_permanence_t, ngm/cold_init.h). DRAW ORDER (documented D5
//    contract): ONE per-group std::mt19937 seeded with `seed`, drawn in
//    arena-slot order (dendrite 0 first, input positions ascending within
//    each). This deliberately differs from ngm2's per-dendrite fast_rng
//    engines — no bit-parity with ngm2 is sought (design §5); the choice only
//    needs to be deterministic and documented, which a single ordered engine
//    gives.
[[nodiscard]] synapse_arena_t build_arena(const group_shape_t &shape,
                                          float                 perm_threshold,
                                          uint32_t              seed);

} // namespace ngm25

#endif // HD_NGM25_ARENA_H
