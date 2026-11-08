//
// Created 10.07.26 — ngm2.5 increment I5: branch decide + group re-pack wiring.
// Map R8. Decide predicate transcribed from hd_ngm2_dendrite.cpp:481-506
// (variance-as-std verbatim); execution delegates to repack_with_splits (I2).
//

#include "hd_ngm25/hd_ngm25_branch.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ngm25::detail {

namespace {

// The R8 qualify predicate for one synapse (ngm2 :497-499). Kept in one place so
// the count pass and the flag pass cannot disagree.
[[nodiscard]] bool qualifies(const synapse_arena_t &arena, uint32_t s,
                             float mm_thres, const branch_params_t &bp,
                             uint8_t max_seg_id)
{
    return arena.adapt_hist[s] >= bp.accumulated_theta_thres
        && arena.mismatch[s]   >= mm_thres
        && 2u * static_cast<uint32_t>(arena.seg[s]) + 1u <= max_seg_id;
}

} // namespace

std::size_t branch_decide_dendrite(const synapse_arena_t &arena, std::size_t d,
                                   const group_shape_t   &shape,
                                   const branch_params_t &bp,
                                   std::span<uint8_t>      clone_flags)
{
    const auto        rng = arena.dendrite(d);
    const std::size_t cnt = rng.end - rng.begin;
    if (cnt == 0) return 0;
    const float syn_cnt_f = static_cast<float>(cnt);

    // mm_avg over the dendrite's current synapses (ngm2 :482).
    float sum = 0.0f;
    for (uint32_t s = rng.begin; s < rng.end; ++s) sum += arena.mismatch[s];
    const float mm_avg = sum / syn_cnt_f;

    // mm_std is a VARIANCE (mean of squared deviations, no sqrt) — replicate the
    // misnomer (ngm2 :483-487).
    float vsum = 0.0f;
    for (uint32_t s = rng.begin; s < rng.end; ++s) {
        const float dv = mm_avg - arena.mismatch[s];
        vsum += dv * dv;
    }
    const float mm_std = vsum / syn_cnt_f;

    // input_size = the dendrite's routed WIDTH (§13.1), not its synapse count.
    const auto  ti         = static_cast<std::size_t>(arena.dend_type[d]);
    const float input_size = static_cast<float>(shape.input_size[ti]);
    const float mm_thres   = mm_avg + mm_std * bp.min_mismatch_deviation
                           + 1.0f / input_size;

    // Count qualifiers (ngm2 :496-502).
    std::size_t mm_cnt = 0;
    for (uint32_t s = rng.begin; s < rng.end; ++s)
        if (qualifies(arena, s, mm_thres, bp, shape.max_seg_id)) ++mm_cnt;

    // Early exit if too few (ngm2 :505-506).
    if (static_cast<float>(mm_cnt) < input_size * bp.min_mismatch_percentage)
        return 0;

    // Flag the same predicate.
    for (uint32_t s = rng.begin; s < rng.end; ++s)
        if (qualifies(arena, s, mm_thres, bp, shape.max_seg_id)) clone_flags[s] = 1;

    return mm_cnt;
}

bool maybe_branch_group(synapse_arena_t &arena, const group_shape_t &shape,
                        const branch_params_t &bp,
                        std::span<const std::uint64_t> input_count_before,
                        std::span<const std::uint64_t> input_count_after,
                        uint32_t rng_seed, uint32_t &branch_epoch)
{
    assert(bp.branch_interval > 0 && "branch_interval must be > 0");
    assert(input_count_before.size() == shape.neuron_cnt &&
           input_count_after.size()  == shape.neuron_cnt);

    std::vector<uint8_t> clone_flags(arena.syn_count(), 0u);
    bool any = false;

    const std::size_t per_neuron = shape.apical_cnt + shape.proximal_cnt;
    for (std::size_t n = 0; n < shape.neuron_cnt; ++n) {
        // Crossed a branch_interval boundary this step? (shared predicate — see
        // crossed_branch_interval: detects the crossing, so the winner's +2 can
        // jump OVER a multiple without the branch being skipped.)
        if (!crossed_branch_interval(input_count_before[n], input_count_after[n],
                                     bp.branch_interval))
            continue;

        const std::size_t d_begin = n * per_neuron;
        const std::size_t d_end   = d_begin + per_neuron;
        for (std::size_t d = d_begin; d < d_end; ++d)
            if (branch_decide_dendrite(arena, d, shape, bp, clone_flags) > 0) any = true;
    }

    if (!any) return false;

    repack_with_splits(arena, clone_flags, shape, rng_seed, branch_epoch);
    ++branch_epoch;               // next branch event draws a fresh D4 wiggle stream
    return true;
}

} // namespace ngm25::detail
