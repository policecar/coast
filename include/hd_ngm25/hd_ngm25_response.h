//
// Created 10.07.26 — ngm2.5 increment I3: per-step context + response/combine.
// Map regions R1 (per-channel stats -> per-position + per-type expansion),
// R2 (dendrite response sweep), R3 (neuron combine). The map is law; every
// kernel names its region and the ngm2 source lines it transcribes.
//
// These are the file-local-style free kernels (ngm25::detail) that the future
// neuron_group_t (I6) will call as a thin wrapper. They are declared here, and
// exercised DIRECTLY by unit tests, so I3 is validated without the io_entity
// machinery. Templated on the penalty policy (hd_ngm25_prng.h); the three
// policy instantiations live in the .cpp.
//

#ifndef HD_NGM25_RESPONSE_H
#define HD_NGM25_RESPONSE_H

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "hd_ngm2/hd_ngm2_tools.h"          // ngm2::input_stats_t, ngm2::sigmoid
#include "hd_ngm25/hd_ngm25_arena.h"
#include "hd_ngm25/hd_ngm25_cfg.h"           // bank_t

namespace ngm25 {

// ---------------------------------------------------------------------------
// R1 output. Per-step derived statistics, expanded from per-channel
// ngm2::compute_input_stats so the hot loops index by input_idx with zero
// channel logic. FORBIDDEN (map R1 / design §6): pooling stats over the
// concatenated input — that is ngm3's undocumented delta. Stats are per-CHANNEL
// (per-position expansion) and min-combined per dendrite TYPE.
//
// I3 fills the response-side fields. The adapt-side high_thres family (R7)
// is added to this struct in I4 when its consumer exists — kept out here so
// nothing untested ships.
struct step_ctx_t {
    // Per concatenated-buffer position (size == total routed width).
    std::vector<float> inv_sum_pos;    // 1 / (owning channel's sum) — penalty scaling
    std::vector<float> half_max_pos;   // owning channel's max_val/2 — D1 draw range

    // Per dendrite TYPE (bank_t index). §13.1: one scalar set per routed channel
    // set (per type when apical/proximal read different channels).
    std::array<float, 2> inp_sum        = {0.0f, 0.0f};  // Σ type-channel sums (early-exit)
    std::array<float, 2> inv_inp_sum    = {0.0f, 0.0f};  // 1 / inp_sum, R2 normalization
    std::array<float, 2> response_atten = {1.0f, 1.0f};  // R2 step 5, min-combined stats

    // --- adapt-side (R7), added in I4. Per concatenated-buffer position. ---
    // TRAP (map R2 vs R7): adapt attenuation uses the DEFAULT sigmoid shape per
    // CHANNEL in-sweep — NOT R2's {0.25,0.5} min-combined-per-type post-sweep.
    // Three sigmoid parameterizations coexist; do not unify.
    std::vector<float> high_thres_pos;      // avg_ch/2 + FLT_EPSILON
    std::vector<float> inv_high_range_pos;  // 1 / (1 - high_thres)
    std::vector<float> inv_high_thres_pos;  // 1 / high_thres
    std::vector<float> adapt_atten_pos;     // ngm2::noise_attenuation(noise_hoyer, avg, learning window)
    std::array<float, 2> inv_last_max_inp = {0.0f, 0.0f};  // 1 / max(type channels' max_val)
};

// One routed channel: its bank/type, its position range in the concatenated
// input buffer, and its per-channel statistics. build_step_ctx consumes a list
// of these (the neuron_group builds it from its input buffers at I6).
struct channel_stats_t {
    bank_t              type   = bank_t::proximal;
    uint32_t            pos_lo = 0;      // [pos_lo, pos_hi) in the concat buffer
    uint32_t            pos_hi = 0;
    ngm2::input_stats_t stats;           // from ngm2::compute_input_stats (verbatim)
};

namespace detail {

// R1. Expand per-channel stats to the per-position arrays + per-type scalars.
// Serial, small (a few hundred floats). total_width = max channel pos_hi.
void build_step_ctx(const group_shape_t                 &shape,
                    std::span<const channel_stats_t>     channels,
                    step_ctx_t                          &out);

// R2. One dendrite's response sweep. Writes this dendrite's segment activities
// into seg_scratch (size max_seg_id+1 — leaves hold the clamped/attenuated
// activity, inner nodes stay raw for R7) and returns the max leaf activity.
// PITFALL P1: the per-synapse clamp makes the accumulate order-dependent — the
// synapse loop is SERIAL. seg_scratch is caller-owned per-dendrite scratch so
// dendrites parallelize without sharing.
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
                        std::span<float>        seg_scratch);

// R3 type reduction, shared by combine (R3) and the adapt envelope (R6/R7 need
// the per-type max). Apical -1.0 absent-sentinel -> 1.0 if no apical responded;
// both results clamped to [0,1] (ngm2 neuron :93-109).
struct type_max_t { float apical; float proximal; };
type_max_t neuron_type_max(const synapse_arena_t &arena, std::size_t n,
                           const group_shape_t   &shape,
                           std::span<const float> dendrite_act);

// R3. Combine: act = clamp(apical·proximal + D2_noise, 0, 1). D2 keys
// (seed, neuron, step).
float combine_neuron(const synapse_arena_t &arena, std::size_t n,
                     const group_shape_t   &shape,
                     std::span<const float> dendrite_act,
                     std::pair<float, float> noise_range,
                     uint32_t seed, uint32_t step);

// R2+R3 driver. Parallel over dendrites (v1 granularity), then combine per
// neuron. Bitwise deterministic across TBB thread counts: each dendrite owns
// its seg_scratch slice and dendrite_act slot; combine reads only. Sizes:
// seg_scratch = dend_cnt*(max_seg_id+1), dendrite_act = dend_cnt,
// neuron_act = neuron_cnt.
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
                  std::span<float>        neuron_act_out);

} // namespace detail

} // namespace ngm25

#endif // HD_NGM25_RESPONSE_H
