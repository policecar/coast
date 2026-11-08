//
// Created 10.07.26 — ngm2.5 increment I4: winner + adapt kernels.
// Map R5 (winner + two adapt passes), R6 (adapt envelope), R7 (dendrite adapt
// sweep). Semantic reference lines in the header. The map is law; formula
// constants come from its constant sheet. Statement order in R7 is load-bearing.
//

#include "hd_ngm25/hd_ngm25_adapt.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include "hd_ngm2/hd_ngm2_tools.h"          // ngm2::sigmoid, sigmoid_params_t
#include "hd_ngm25/hd_ngm25_prng.h"          // pcg_hash / syn_key / to_unit

namespace ngm25::detail {

using ngm2::sigmoid;
using ngm2::sigmoid_params_t;

void adapt_dendrite(synapse_arena_t &arena, std::size_t d,
                    std::span<const float>  input_concat,
                    const step_ctx_t       &ctx,
                    bank_t                  type,
                    uint8_t                 max_seg_id,
                    float                   type_max,
                    float                   synapse_weight,
                    const adapt_params_t   &ap,
                    std::span<float>        seg_activity,
                    std::span<float>        seg_weights)
{
    // Early return if the type max is broken or zero (ngm2 :345). A dendrite
    // whose type-max lives in a sibling still enters — it just gets no primary
    // leaf below (that case is intentional, R7 note).
    if (!std::isnormal(type_max)) return;

    const auto        ti         = static_cast<std::size_t>(type);
    const std::size_t seg_cnt    = static_cast<std::size_t>(max_seg_id) + 1;
    const std::size_t leaf_begin = (static_cast<std::size_t>(max_seg_id) + 1) / 2;
    const float       inv_max    = 1.0f / type_max;
    constexpr float   eps        = std::numeric_limits<float>::epsilon();

    // 1) leaf weights (ngm2 :365-377). The FIRST leaf reaching the type-max gets
    //    weight·primary_lr and blocks any later primary; others get
    //    act·weight·secondary_lr/max. A dendrite may have no primary leaf.
    std::fill_n(seg_weights.begin(), seg_cnt, 0.0f);
    bool primary_seen = false;
    for (std::size_t si = leaf_begin; si <= max_seg_id; ++si) {
        if (!primary_seen && seg_activity[si] + eps >= type_max) {
            primary_seen    = true;
            seg_weights[si] = synapse_weight * ap.primary_learning_rate;
        } else {
            seg_weights[si] = seg_activity[si] * synapse_weight
                            * ap.secondary_learning_rate * inv_max;
        }
    }

    // 2) fold leaves->root, parent = max(children), for BOTH weights and
    //    activities (ngm2 :381-385). Idempotent (leaves untouched), so the
    //    winner's second adapt re-folding is safe.
    for (std::size_t level_start = leaf_begin; level_start > 1; level_start /= 2)
        for (std::size_t si = level_start; si < level_start * 2; si += 2) {
            seg_weights[si / 2]  = std::max(seg_weights[si],  seg_weights[si + 1]);
            seg_activity[si / 2] = std::max(seg_activity[si], seg_activity[si + 1]);
        }

    // 3) sweep synapses (ngm2 :422-467). ORDER is load-bearing: theta -> perm
    //    IIR -> adapt_hist -> mismatch (which reads the JUST-UPDATED perm).
    const auto rng = arena.dendrite(d);
    for (uint32_t s = rng.begin; s < rng.end; ++s) {
        const uint16_t pos = arena.input_idx[s];
        const uint8_t  seg = arena.seg[s];
        const float    inp = input_concat[pos];

        const float high_thres = ctx.high_thres_pos[pos];
        const float ramp = inp > high_thres
            ? (inp - high_thres) * ctx.inv_high_range_pos[pos]
            : (high_thres - inp) * ctx.inv_high_thres_pos[pos];
        const float theta = std::clamp(seg_weights[seg] * ramp * ctx.adapt_atten_pos[pos],
                                       0.0f, 1.0f);

        arena.perm[s] = std::clamp(arena.perm[s] * (1.0f - theta)
                                   + (inp > high_thres ? theta : 0.0f), 0.0f, 1.0f);
        arena.adapt_hist[s] += theta;

        const float act_ratio = seg_activity[seg] * inv_max;
        if (act_ratio >= ap.mismatch_act_thres) {
            const float inp_ratio = inp * ctx.inv_last_max_inp[ti];
            float mm = arena.perm[s] > ap.perm_threshold ? 1.0f - inp_ratio : inp_ratio;
            mm *= act_ratio;
            arena.mismatch[s] = arena.mismatch[s] * (1.0f - ap.mismatch_smoothing)
                              + mm * ap.mismatch_smoothing;
        }
    }
}

void adapt_neuron(synapse_arena_t &arena, std::size_t n,
                  const group_shape_t    &shape,
                  std::span<const float>  input_concat,
                  const step_ctx_t       &ctx,
                  const type_max_t       &tmax,
                  float                   neuron_act,
                  float                   neuron_mx_act,
                  float                   weight,
                  const learning_window_t &window,
                  const adapt_params_t   &ap,
                  std::span<float>        seg_activity,
                  std::span<float>        seg_weights,
                  std::uint64_t          &input_count)
{
    // R6 envelope (ngm2 :163-166). Three sigmoid parameterizations in one
    // expression: sig(mx_act) DEFAULT shape, the two window sigmoids use the
    // tracked learning window's first/second shape.
    const float synapse_weight = weight * std::min(
        std::max(1.0f - sigmoid(neuron_mx_act),
                 sigmoid(neuron_act, window.first)),
        1.0f - sigmoid(neuron_act, window.second));

    const std::size_t per_neuron = shape.apical_cnt + shape.proximal_cnt;
    const std::size_t d_begin    = n * per_neuron;
    const std::size_t d_end      = d_begin + per_neuron;
    const std::size_t seg_cnt    = static_cast<std::size_t>(shape.max_seg_id) + 1;

    for (std::size_t d = d_begin; d < d_end; ++d) {
        const bank_t type     = arena.dend_type[d];
        const float  type_max = (type == bank_t::apical) ? tmax.apical : tmax.proximal;
        adapt_dendrite(arena, d, input_concat, ctx, type, shape.max_seg_id,
                       type_max, synapse_weight, ap,
                       seg_activity.subspan(d * seg_cnt, seg_cnt),
                       seg_weights.subspan(d * seg_cnt, seg_cnt));
    }

    ++input_count;   // per adapt CALL (R8 cadence); the branch check is I5.
}

std::size_t adapt_all(synapse_arena_t &arena, const group_shape_t &shape,
               std::span<const float>  input_concat,
               const step_ctx_t       &ctx,
               std::span<const float>  dendrite_act,
               std::span<const float>  neuron_act,
               std::span<const float>  neuron_mx_act,
               std::span<const float>  post_inhib_out,
               const adapt_params_t   &ap,
               const learning_window_t &window_winner,
               const learning_window_t &window_other,
               std::span<float>        seg_activity,
               std::span<float>        seg_weights,
               std::span<std::uint64_t> input_count,
               uint32_t seed, uint32_t step)
{
    const std::size_t ncnt = shape.neuron_cnt;

    // Per-neuron type maxes (needed by both passes; R6 uses them per type).
    std::vector<type_max_t> tmax(ncnt);
    for (std::size_t n = 0; n < ncnt; ++n)
        tmax[n] = neuron_type_max(arena, n, shape, dendrite_act);

    // R5.1 group max activity over the POST-inhibition output (ngm2 :204).
    float mx_act = 0.0f;
    for (std::size_t i = 0; i < ncnt; ++i) mx_act = std::max(mx_act, post_inhib_out[i]);

    // R5.2 D3 winner threshold, draw ~ U(mx_act·win_thres, mx_act) (ngm2 :208).
    // hash(seed, step) via syn_key with a sentinel dend (0xFFFFFFFF) that no
    // real dendrite/neuron index takes -> collision-free vs D1/D2 (map D3).
    const float lo      = mx_act * ap.stochastic_win_thres;
    const float win_act = lo + to_unit(pcg_hash(syn_key(seed, step, 0xFFFFFFFFu, 0, 0)))
                             * (mx_act - lo);
    // D6: which of the neurons clearing win_act wins, uniform among them
    // (SPEC.md M.4.4) — the D3 key with seg = 1. Same key and f32 arithmetic in
    // the GPU winner kernel.
    const float win_pick = to_unit(pcg_hash(syn_key(seed, step, 0xFFFFFFFFu, 1, 0)));
    const sigmoid_params_t wf(ap.weight_filter);


    // R5.3 loop 3: the winner — uniform among the neurons reaching the winner
    //     threshold (M.4.4) — adapts PRIMARY (weight = sig(1−out), no common_lr).
    const std::size_t winner = ngm2::pick_winner(post_inhib_out.first(ncnt), win_act, win_pick);
    if (winner < ncnt) {   // ncnt = no winner this step
        const float w = sigmoid(1.0f - post_inhib_out[winner], wf);
        adapt_neuron(arena, winner, shape, input_concat, ctx, tmax[winner],
                     neuron_act[winner], neuron_mx_act[winner], w, window_winner, ap,
                     seg_activity, seg_weights, input_count[winner]);
    }

    // R5.4 loop 4: every OTHER neuron adapts SECONDARY, weight =
    //     sig(1 − out/act_sum)·common_lr. The winner is skipped: every neuron
    //     adapts exactly once per step, the winner with its primary weight
    //     instead of the common one (SPEC.md M.4.14). Parallel over neurons; each
    //     touches its own dendrites -> race-free and bitwise deterministic across
    //     thread counts.
    float act_sum = 0.0f;
    for (std::size_t i = 0; i < ncnt; ++i) act_sum += post_inhib_out[i];

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, ncnt),
        [&](const tbb::blocked_range<std::size_t> &r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (i == winner) continue;   // adapted once, in loop 3
                // Guard the sole division by a summed quantity. ngm2 divides by
                // act_sum unguarded (hd_ngm2_neuron_group.cpp:226): if every
                // post-inhibition output is 0 (e.g. neuron_noise_range=[0,0] with
                // all-subthreshold input) act_sum is exactly 0 and 0/0 -> NaN, which
                // the flat arena would then persist as a NaN permanence (inv5) for
                // the whole group. Conscious deviation from ngm2 (documented): a
                // zero act_sum has no meaningful secondary weight, so use 0 (a no-op
                // adapt). We test `!= 0`, not `> 0`, so a negative act_sum still
                // matches ngm2 exactly — only the NaN-producing case is caught.
                // input_count still advances inside adapt_neuron, preserving the
                // R8 branch cadence.
                const float sec = act_sum != 0.0f
                                ? sigmoid(1.0f - post_inhib_out[i] / act_sum, wf)
                                : 0.0f;
                adapt_neuron(arena, i, shape, input_concat, ctx, tmax[i],
                             neuron_act[i], neuron_mx_act[i],
                             sec * ap.common_learning_rate, window_other, ap,
                             seg_activity, seg_weights, input_count[i]);
            }
        });
    return winner;
}

} // namespace ngm25::detail
