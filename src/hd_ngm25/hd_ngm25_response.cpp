//
// Created 10.07.26 — ngm2.5 increment I3: response/combine kernels.
// Map R1 (build_step_ctx), R2 (response_dendrite), R3 (combine_neuron) +
// the R2/R3 driver response_all. Semantic reference:
//   R1: ngm2::compute_input_stats (verbatim) + group process() :153-156
//   R2: dendrite_t::get_response,  hd_ngm2_dendrite.cpp:149-328
//   R3: neuron_t::get_response,    hd_ngm2_neuron.cpp:82-140
// The map is law; formula constants come from its constant sheet. No perf work
// (plan I9) — the only structural choice is dendrite-parallel (P1: inner
// synapse loop stays serial).
//

#include "hd_ngm25/hd_ngm25_response.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include "hd_ngm25/hd_ngm25_prng.h"          // pcg_hash / syn_key / to_unit + penalty policies

namespace ngm25::detail {

using ngm2::noise_gate_min_t;
using ngm2::response_noise_window;

void build_step_ctx(const group_shape_t              &shape,
                    std::span<const channel_stats_t>  channels,
                    step_ctx_t                       &out)
{
    // Width of the concatenated input buffer = the largest routed position.
    uint32_t total_width = 0;
    for (const auto &c : channels) total_width = std::max(total_width, c.pos_hi);

    out.inv_sum_pos.assign(total_width, 0.0f);
    out.half_max_pos.assign(total_width, 0.0f);
    out.inp_sum        = {0.0f, 0.0f};
    out.inv_inp_sum    = {0.0f, 0.0f};
    out.response_atten = {1.0f, 1.0f};

    // adapt-side (R7)
    out.high_thres_pos.assign(total_width, 0.0f);
    out.inv_high_range_pos.assign(total_width, 0.0f);
    out.inv_high_thres_pos.assign(total_width, 0.0f);
    out.adapt_atten_pos.assign(total_width, 1.0f);
    out.inv_last_max_inp = {0.0f, 0.0f};
    std::array<float, 2> last_max_inp = {0.0f, 0.0f};   // max over type channels' max_val

    // Per-type min-combined noise (1 - Hoyer) and avg over the channels that
    // carry shape evidence (R1 / §13.1, noise_gate_min_t); a type with no channels keeps
    // its default atten (1.0).
    std::array<noise_gate_min_t, 2> gate_min {};
    std::array<bool, 2>             seen = {false, false};

    for (const auto &c : channels) {
        const auto ti = static_cast<std::size_t>(c.type);
        seen[ti]        = true;
        out.inp_sum[ti] += c.stats.sum;
        gate_min[ti].add(c.stats);

        // Per-position expansion (value-identical lookup, map R1 permitted) of
        // the per-channel constants, one definition with ngm3
        // (ngm2::channel_constants): inv_sum is the CURRENT channel's reciprocal
        // sum (ngm2 dendrite :250); half_max is the channel's max_val/2 = the D1
        // draw range (ngm2 :253); high_thres = avg/2 + eps (ngm2 dendrite :414);
        // adapt attenuation uses the learning window (NOT R2's response window),
        // per channel.
        const ngm2::channel_constants_t k = ngm2::channel_constants(c.stats);
        last_max_inp[ti] = std::max(last_max_inp[ti], c.stats.max_val);

        for (uint32_t p = c.pos_lo; p < c.pos_hi; ++p) {
            out.inv_sum_pos[p]        = k.inv_sum;
            out.half_max_pos[p]       = k.half_max;
            out.high_thres_pos[p]     = k.high_thres;
            out.inv_high_range_pos[p] = k.inv_high_range;
            out.inv_high_thres_pos[p] = k.inv_high_thres;
            out.adapt_atten_pos[p]    = k.adapt_atten;
        }
    }

    for (std::size_t ti = 0; ti < 2; ++ti) {
        if (!seen[ti]) continue;
        // R2 attenuation (constant sheet): min-combined noise/avg through the
        // response window (ngm2 dendrite get_response).
        out.response_atten[ti] = gate_min[ti].attenuation(response_noise_window);
        out.inv_inp_sum[ti] = out.inp_sum[ti] != 0.0f ? 1.0f / out.inp_sum[ti] : 0.0f;
        out.inv_last_max_inp[ti] = last_max_inp[ti] != 0.0f ? 1.0f / last_max_inp[ti] : 0.0f;
    }

    (void)shape;   // shape currently only documents the routing the caller built
}

template <class Penalty>
float response_dendrite(const synapse_arena_t &arena, std::size_t d,
                        std::span<const float>  input_concat,
                        const step_ctx_t       &ctx,
                        bank_t                  type,
                        uint8_t                 max_seg_id,
                        float                   perm_threshold,
                        float                   inv_perm_range,
                        uint32_t                seed,
                        uint32_t                step,
                        std::span<float>        seg_scratch)
{
    const auto        ti  = static_cast<std::size_t>(type);
    const std::size_t seg_cnt = static_cast<std::size_t>(max_seg_id) + 1;

    // clear segment activities (ngm2 :188)
    std::fill_n(seg_scratch.begin(), seg_cnt, 0.0f);

    // Early exit: the type's Σ channel sums must be normal (ngm2 :209). This is
    // per-TYPE in ngm2.5 (all dendrites of a type read the same channel set).
    if (!std::isnormal(ctx.inp_sum[ti]))
        return 0.0f;

    // 1) sweep synapses in arena (input-major) order. Skip perm <= thres
    //    (ngm2 :273 uses strict '>', i.e. include when perm > thres); accumulate
    //    seg_activity[seg] += inp; draw D1; on penalty subtract
    //    perm_strength·(1 − inp·inv_sum) and clamp the running value >= 0
    //    IMMEDIATELY (ngm2 :275-281). The clamp makes this order-dependent (P1).
    const auto rng = arena.dendrite(d);
    for (uint32_t s = rng.begin; s < rng.end; ++s) {
        const float perm = arena.perm[s];
        if (perm <= perm_threshold) continue;

        const uint16_t pos = arena.input_idx[s];
        const uint8_t  seg = arena.seg[s];
        const float    inp = input_concat[pos];

        seg_scratch[seg] += inp;

        const float w = Penalty::penalty_weight(seed, step, static_cast<uint32_t>(d),
                                                static_cast<uint32_t>(seg), pos,
                                                inp, ctx.half_max_pos[pos]);
        if (w > 0.0f) {
            const float perm_strength = (perm - perm_threshold) * inv_perm_range;
            seg_scratch[seg] -= w * perm_strength * (1.0f - inp * ctx.inv_sum_pos[pos]);
            if (seg_scratch[seg] < 0.0f) seg_scratch[seg] = 0.0f;
        }
    }

    // 2) push activities root->leaf (ngm2 :289-293)
    const std::size_t leaf_begin = (static_cast<std::size_t>(max_seg_id) + 1) / 2;
    for (std::size_t si = 1; si < leaf_begin; ++si) {
        seg_scratch[si * 2 + 0] += seg_scratch[si];
        seg_scratch[si * 2 + 1] += seg_scratch[si];
    }

    // 3) attenuate + normalize leaves; track max (ngm2 :304-311). Attenuation is
    //    the per-type min-combined value, applied post-sweep to leaves only.
    //    Inner entries stay raw (R7 reads them). inv_inp_sum is per type.
    const float atten       = ctx.response_atten[ti];
    const float inv_inp_sum = ctx.inv_inp_sum[ti];
    float       max_activity = 0.0f;
    for (std::size_t si = leaf_begin; si <= max_seg_id; ++si) {
        seg_scratch[si] = std::clamp(seg_scratch[si] * atten * inv_inp_sum, 0.0f, 1.0f);
        max_activity    = std::max(max_activity, seg_scratch[si]);
    }
    return max_activity;
}

type_max_t neuron_type_max(const synapse_arena_t &arena, std::size_t n,
                           const group_shape_t   &shape,
                           std::span<const float> dendrite_act)
{
    // Type-max over this neuron's dendrites. Apical starts at the -1.0 absent-
    // sentinel, proximal at 0.0 (ngm2 :93-95). Neurons own a contiguous
    // dendrite range (neuron-major directory, arena invariant 2).
    const std::size_t per_neuron = shape.apical_cnt + shape.proximal_cnt;
    const std::size_t d_begin    = n * per_neuron;
    const std::size_t d_end      = d_begin + per_neuron;

    float apical   = -1.0f;
    float proximal =  0.0f;
    for (std::size_t d = d_begin; d < d_end; ++d) {
        if (arena.dend_type[d] == bank_t::apical)
            apical = std::max(apical, dendrite_act[d]);
        else
            proximal = std::max(proximal, dendrite_act[d]);
    }

    // Sentinel: no apical dendrite responded/exists -> 1.0 (ngm2 :104-105).
    if (apical < 0.0f) apical = 1.0f;
    return { std::clamp(apical, 0.0f, 1.0f), std::clamp(proximal, 0.0f, 1.0f) };
}

float combine_neuron(const synapse_arena_t &arena, std::size_t n,
                     const group_shape_t   &shape,
                     std::span<const float> dendrite_act,
                     std::pair<float, float> noise_range,
                     uint32_t seed, uint32_t step)
{
    const type_max_t tm = neuron_type_max(arena, n, shape, dendrite_act);

    // D2 noise: hash(seed, neuron, step) scaled to the noise range. Keyed via
    // syn_key with seg=0,input=0 (never a real synapse key — seg>=1 always) so
    // (seed, neuron, step) gets its own stream. noise_range=[0,0] => 0 (the
    // deterministic-fixture freeze), matching ngm2's frozen-eval convention.
    const float noise = noise_range.first
                      + to_unit(pcg_hash(syn_key(seed, step, static_cast<uint32_t>(n), 0, 0)))
                        * (noise_range.second - noise_range.first);

    return std::clamp(tm.apical * tm.proximal + noise, 0.0f, 1.0f);
}

template <class Penalty>
void response_all(const synapse_arena_t &arena, const group_shape_t &shape,
                  std::span<const float>  input_concat,
                  const step_ctx_t       &ctx,
                  float                   perm_threshold,
                  std::pair<float, float> noise_range,
                  uint32_t                seed,
                  uint32_t                step,
                  std::span<float>        seg_scratch,
                  std::span<float>        dendrite_act_out,
                  std::span<float>        neuron_act_out)
{
    const std::size_t n_dend  = arena.dend_count();
    const std::size_t seg_cnt = static_cast<std::size_t>(shape.max_seg_id) + 1;
    const float       inv_perm_range = 1.0f / (1.0f - perm_threshold);

    // R2: parallel over dendrites. Each writes its own dendrite_act slot and its
    // own seg_scratch slice -> bitwise identical for any thread count (no shared
    // writes, and hash draws are order-free). Inner synapse loop is serial (P1).
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n_dend),
        [&](const tbb::blocked_range<std::size_t> &r) {
            for (std::size_t d = r.begin(); d != r.end(); ++d) {
                dendrite_act_out[d] = response_dendrite<Penalty>(
                    arena, d, input_concat, ctx, arena.dend_type[d], shape.max_seg_id,
                    perm_threshold, inv_perm_range, seed, step,
                    seg_scratch.subspan(d * seg_cnt, seg_cnt));
            }
        });

    // R3: combine per neuron (reads dendrite_act only — race-free).
    for (std::size_t n = 0; n < shape.neuron_cnt; ++n)
        neuron_act_out[n] = combine_neuron(arena, n, shape, dendrite_act_out,
                                           noise_range, seed, step);
}

// Explicit instantiation for the three penalty policies (map §8 / prng.h). All
// three exist in the binary; the group dispatches once per step (I6).
template float response_dendrite<stochastic_penalty>(
    const synapse_arena_t &, std::size_t, std::span<const float>, const step_ctx_t &,
    bank_t, uint8_t, float, float, uint32_t, uint32_t, std::span<float>);
template float response_dendrite<expected_penalty>(
    const synapse_arena_t &, std::size_t, std::span<const float>, const step_ctx_t &,
    bank_t, uint8_t, float, float, uint32_t, uint32_t, std::span<float>);
template float response_dendrite<no_penalty>(
    const synapse_arena_t &, std::size_t, std::span<const float>, const step_ctx_t &,
    bank_t, uint8_t, float, float, uint32_t, uint32_t, std::span<float>);

template void response_all<stochastic_penalty>(
    const synapse_arena_t &, const group_shape_t &, std::span<const float>,
    const step_ctx_t &, float, std::pair<float, float>, uint32_t, uint32_t,
    std::span<float>, std::span<float>, std::span<float>);
template void response_all<expected_penalty>(
    const synapse_arena_t &, const group_shape_t &, std::span<const float>,
    const step_ctx_t &, float, std::pair<float, float>, uint32_t, uint32_t,
    std::span<float>, std::span<float>, std::span<float>);
template void response_all<no_penalty>(
    const synapse_arena_t &, const group_shape_t &, std::span<const float>,
    const step_ctx_t &, float, std::pair<float, float>, uint32_t, uint32_t,
    std::span<float>, std::span<float>, std::span<float>);

} // namespace ngm25::detail
