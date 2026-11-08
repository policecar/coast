//
// Created 10.07.26 — ngm2.5 increment I5: branch integration.
// Map R8 — the per-neuron gate cadence, the decide predicate (variance-as-std
// verbatim), and the wiring to repack_with_splits (I2). Semantic reference:
//   decide: dendrite_t::adapt_branches, hd_ngm2_dendrite.cpp:481-506
//   gate:   neuron_t::adapt cadence,    hd_ngm2_neuron.cpp:177-183
// The map is law. Only the decide predicate + split effect transcribe; the
// execution (deferred to map §0 step 8, one group re-pack) is a permitted change.
//

#ifndef HD_NGM25_BRANCH_H
#define HD_NGM25_BRANCH_H

#include <cstddef>
#include <cstdint>
#include <span>

#include "hd_ngm25/hd_ngm25_arena.h"

namespace ngm25 {

// R8 decide-predicate parameters (subset of neuron_group_params_t).
struct branch_params_t {
    float       min_mismatch_deviation  = 1.0f;
    float       accumulated_theta_thres = 2.0f;
    float       min_mismatch_percentage = 0.002f;
    // DESIGN INTENT (author). branch_interval is always LARGE — thousands, by
    // design; it is not expected to shrink into the low tens. input_count is not a
    // meaningful quantity in itself: it exists ONLY to time branching — "has enough
    // adaptation happened that we should re-check for branching?". Every neuron
    // adapts once per learning step (SPEC.md M.4.14), so each counter advances by
    // exactly 1 and the crossing test below reduces to ngm2's landing test
    // (`count % interval == 0`). It is kept as a crossing test (see
    // crossed_branch_interval) so a boundary can never be skipped should a step
    // ever advance a counter by more than 1 (it once did: the winner's +2).
    std::size_t branch_interval         = 5000;
};

namespace detail {

// Did a neuron's per-step adapt-call counter cross a branch_interval boundary
// this step? Single source of truth for the branch cadence, shared by the CPU
// group phase (maybe_branch_group) and the GPU host gate (grow_if_branched) so
// the two paths can never disagree.
//
// It detects the CROSSING (integer-division change), NOT an exact landing
// (`after % interval == 0`). This is load-bearing: the winner's counter advances
// by +2 in one step, so it can jump OVER a multiple without ever equalling it
// (e.g. interval-1 -> interval+1). A modulo-on-final-count test would silently
// SKIP that branch; the division-change test catches it. `interval >= 1` (ctor-
// enforced) so the division is safe; before <= after always (counter is monotone).
[[nodiscard]] constexpr bool crossed_branch_interval(std::uint64_t before,
                                                     std::uint64_t after,
                                                     std::uint64_t interval) noexcept
{
    return (before / interval) != (after / interval);
}

// R8 decide for ONE dendrite (ngm2 :481-506). mm_avg/mm_std over the dendrite's
// CURRENT synapses (mm_std is a VARIANCE — mean of squared deviations, no sqrt;
// replicate the misnomer). mm_thres = mm_avg + variance·dev + 1/input_size,
// where input_size is the dendrite's routed WIDTH (§13.1), not its synapse
// count. A synapse qualifies iff adapt_history >= accumulated_theta_thres,
// mismatch >= mm_thres, and seg·2+1 <= max_seg_id (still splittable). Proceeds
// only if the qualifying count >= input_size·min_mismatch_percentage; then sets
// clone_flags[s]=1 for each qualifier. Returns the number flagged (0 = no
// branch for this dendrite). The predicate is evaluated exactly as ngm2 does
// (count, then flag the same predicate).
std::size_t branch_decide_dendrite(const synapse_arena_t &arena, std::size_t d,
                                   const group_shape_t   &shape,
                                   const branch_params_t &bp,
                                   std::span<uint8_t>      clone_flags);

// R8 group branch phase (map §0 step 8). A neuron branches this step iff its
// per-neuron adapt-call counter crossed a branch_interval boundary during the
// step: input_count_before/interval != input_count_after/interval. (The winner
// advances twice, so its cadence drifts ahead — map R8; the crossing test
// captures either the +1 or +2 crossing.) For each such neuron, decide all its
// dendrites; if anything is flagged group-wide, repack_with_splits ONCE and
// ++branch_epoch. Returns true iff a re-pack happened. Deferred to post-adapt,
// so branching sees the final (post-both-adapt) state — a design-sanctioned
// change from ngm2's inline timing (branch could fire between the winner's two
// adapts in ngm2; here it never does).
bool maybe_branch_group(synapse_arena_t &arena, const group_shape_t &shape,
                        const branch_params_t &bp,
                        std::span<const std::uint64_t> input_count_before,
                        std::span<const std::uint64_t> input_count_after,
                        uint32_t rng_seed, uint32_t &branch_epoch);

} // namespace detail

} // namespace ngm25

#endif // HD_NGM25_BRANCH_H
