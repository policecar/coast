//
// Created 10.07.26 — ngm2.5 increment I4: winner selection + adapt path.
// Map regions R5 (winner + the two adapt passes), R6 (per-neuron adapt
// envelope), R7 (dendrite adapt sweep). Semantic reference:
//   R5: group process(), hd_ngm2_neuron_group.cpp:199-236
//   R6: neuron_t::adapt,  hd_ngm2_neuron.cpp:156-184
//   R7: dendrite_t::adapt_synapses, hd_ngm2_dendrite.cpp:342-468
// The map is law. These are detail free kernels (I6's neuron_group_t is a thin
// caller); unit tests hit them directly. Adapt has NO stochastic draw except
// R5's single D3 winner draw — so nothing here is penalty-templated.
//

#ifndef HD_NGM25_ADAPT_H
#define HD_NGM25_ADAPT_H

#include <cstdint>
#include <span>

#include "hd_ngm25/hd_ngm25_arena.h"
#include "hd_ngm25/hd_ngm25_cfg.h"          // sigmoid_shape_t, learning_window_t, bank_t
#include "hd_ngm25/hd_ngm25_response.h"     // step_ctx_t, type_max_t, neuron_type_max

namespace ngm25 {

// Learning parameters the adapt path needs (a subset of neuron_group_params_t,
// passed as a POD so the kernels stay free functions).
struct adapt_params_t {
    float             perm_threshold          = 0.3f;
    float             primary_learning_rate   = 0.01f;
    float             secondary_learning_rate = 0.0001f;
    float             common_learning_rate    = 0.0001f;
    float             stochastic_win_thres    = 0.8f;
    sigmoid_shape_t   weight_filter           = { 0.5, 0.33 };
    float             mismatch_act_thres      = 0.8f;
    float             mismatch_smoothing      = 0.001f;
};

namespace detail {

// R7. Adapt one dendrite (mutates arena perm/adapt_hist/mismatch for dendrite
// d). `type_max` is the neuron's per-type max for this dendrite's type (R6).
// `seg_activity` is this dendrite's R2 output (leaves clamped, inner raw) and is
// FOLDED IN PLACE (idempotent). `seg_weights` is per-dendrite scratch. Both are
// sized max_seg_id+1. Statement order is load-bearing (mismatch reads the
// just-updated permanence) — do not reorder.
void adapt_dendrite(synapse_arena_t &arena, std::size_t d,
                    std::span<const float>  input_concat,
                    const step_ctx_t       &ctx,
                    bank_t                  type,
                    uint8_t                 max_seg_id,
                    float                   type_max,
                    float                   synapse_weight,
                    const adapt_params_t   &ap,
                    std::span<float>        seg_activity,
                    std::span<float>        seg_weights);

// R6. Per-neuron adapt envelope + per-dendrite adapt. `neuron_act` is this
// step's combined activity (R3); `neuron_mx_act` the decaying running max (R6's
// sig(mx_act) uses the DEFAULT shape). `weight` is R5's primary or secondary
// weight; `window` the learning window that gates it (the group's tracked
// window for this neuron). Advances `input_count` by one (R8 cadence; the
// branch check itself is I5). seg_activity / seg_weights span the WHOLE group; this neuron's dendrites
// index their own slices.
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
                  std::uint64_t          &input_count);

// R5. Winner selection + the two adapt passes. Loop-3 primary (the single
// winner, serial) and loop-4 secondary (every other neuron, parallel): each
// neuron adapts exactly once per step (SPEC.md M.4.14), so its input_count
// advances by 1. D3 winner draw = hash(seed, step) (map D3 permits hash).
// Mutates arena + input_count[]. Sizes: dendrite_act/neuron_act/neuron_mx_act/
// post_inhib_out/input_count = neuron_cnt; seg_activity/seg_weights =
// dend_cnt*(max_seg_id+1); seg_activity carries R2's per-dendrite output.
// The winner adapts gated by `window_winner`, every other neuron by
// `window_other` (the group's tracked windows, ngm/window_tracking.h).
// Returns the winner's index, neuron_cnt if none.
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
               uint32_t seed, uint32_t step);

} // namespace detail

} // namespace ngm25

#endif // HD_NGM25_ADAPT_H
