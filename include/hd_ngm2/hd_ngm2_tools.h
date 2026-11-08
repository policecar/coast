//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_TOOLS_H
#define HD_NGM2_TOOLS_H

#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <algorithm>
#include <random>
#include <span>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/seed.h"

#include "sim_core/io_entity.h"

namespace ngm2 {

// A channel as it reaches a neuron or dendrite: its id and width, and `source`,
// the name it has in the group's input map. channel_mappings may rename it on
// the way, so the channel record (M.7.5) stores `source`, not the local name.
struct channel_source_t {
    sim::output_channel_t channel;
    std::string           source;
};
using channel_source_map_t = std::unordered_map<std::string, channel_source_t>;

void softmax(auto &vec, float beta = 1.0f)
{
    using vec_elem_t = typename std::remove_reference<decltype(vec[0])>::type;
    if (vec.empty())
        return;
    const vec_elem_t zero {};
    auto max_val = std::reduce(vec.begin(),vec.end(),zero,[](const auto a, const auto b){ return std::max(a,b); });
    for (auto &val : vec) {
        val -= max_val;
    }
    vec_elem_t sum {};
    for (auto &val : vec) {
        val = std::exp(val * beta);
        sum += val;
    }
    if (std::isnormal(sum)) {
        for (auto &val : vec) {
            val /= sum;
        }
    } else {
        std::ranges::fill(vec,zero);
    }
}

struct sigmoid_shape_t {
    double steepness        = 0.5;
    double transition_point = 0.5;
};


// Precomputed sigmoid parameters — eliminates pow() from repeated calls with the same shape.
struct sigmoid_params_t {
    float inv_step_size;
    float starting_point;
    explicit sigmoid_params_t(sigmoid_shape_t shape)
        : inv_step_size  ( 1.0f / (1.0f - std::pow((float)shape.steepness, 0.1f)) )
        , starting_point ( -(float)shape.transition_point * inv_step_size           )
    {}
};

// Sigmoid with configurable steepness and transition point.
inline float sigmoid(
    float x,
    sigmoid_shape_t shape = {0.5, 0.5}
){
    const float inv_step_size  = 1.0f / (1.0f - std::pow((float)shape.steepness, 0.1f));
    const float starting_point = -shape.transition_point * inv_step_size;
    const float z = -(x * inv_step_size + starting_point);
    return 1.0f / (1.0f + std::exp(z));
}

// Fast overload: uses precomputed params, no pow() call.
inline float sigmoid(float x, sigmoid_params_t p) {
    return 1.0f / (1.0f + std::exp(-(x * p.inv_step_size + p.starting_point)));
}

void normalize(auto &vec)
{
    using vec_elem_t = typename std::remove_reference<decltype(vec[0])>::type;
    if (vec.empty())
        return;
    const vec_elem_t zero {};
    auto max_val = std::reduce(vec.begin(),vec.end(),zero,[](const auto a, const auto b){ return std::max(a,b); });
    auto min_val = std::reduce(vec.begin(),vec.end(),max_val,[](const auto a, const auto b){ return std::min(a,b); });
    if (max_val - min_val <= std::numeric_limits<vec_elem_t>::epsilon()) {
        for (auto &val : vec)
            val = 0.0f;
        return;
    }
    for (auto &val : vec) {
        val = (val - min_val) / (max_val - min_val);
    }
}

void root_vec(auto &vec, float rt = 2.0)
{
    using vec_elem_t = typename std::remove_reference<decltype(vec[0])>::type;
    if (vec.empty())
        return;
    const vec_elem_t rt_val = static_cast<vec_elem_t>(1.0f / rt);
    for (auto &val : vec) {
        val = std::pow(val,rt_val);
    }
}

// Seed for one neuron's or dendrite's random stream, derived from its configured
// seed, its index within its parent, and the parent's stream seed. The parts are
// hashed (splitmix64 finaliser), not added: with a plain sum, seed + 1 reproduced
// the neighbouring neuron's or dendrite's stream, so a different seed or seed
// offset only shifted the same synapses by one position (SPEC.md M.0.6 b, M.2.3).
// The seeds enter the hash zero-extended (seed_t is unsigned, see core/seed.h).
inline seed_t stream_seed(seed_t config_seed, std::int64_t index, seed_t parent_seed)
{
    auto mix = [](std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    };
    std::uint64_t h = mix(std::uint64_t{ std::to_underlying(config_seed) });
    h = mix(h ^ static_cast<std::uint64_t>(index));
    h = mix(h ^ std::uint64_t{ std::to_underlying(parent_seed) });
    return seed_t{ static_cast<std::uint32_t>(h >> 32) };
}

// 1 - Hoyer sparsity (Hoyer 2004): Hoyer = (sqrt(n) - L1/L2) / (sqrt(n) - 1).
// (L1/L2)^2 is the participation ratio, the effective number of entries that
// carry the mass. 0 for a one-hot vector, 1 for a flat one. Needs only the sum
// and the sum of squares. Returns 0 for an empty or all-zero (or non-finite)
// vector, and 0 for n == 1: a single entry carries no shape evidence. At 0 the
// noise gate passes a channel whatever its brightness (attenuation >= 0.99), and
// the response gate's min leaves such a channel out (noise_gate_min_t).
inline float hoyer_noise(std::span<const float> vec)
{
    if (vec.empty())
        return 0.0f;
    const float l1 = std::reduce(vec.begin(), vec.end());
    if (!std::isnormal(l1))
        return 0.0f;
    if (vec.size() == 1)
        return 0.0f;
    const float l2 = std::sqrt(std::transform_reduce(vec.begin(), vec.end(), 0.0f, std::plus<>(),
                                                     [](float v) { return v * v; }));
    const float sqrt_n = std::sqrt(static_cast<float>(vec.size()));
    const float hoyer  = (sqrt_n - l1 / l2) / (sqrt_n - 1.0f);
    return std::clamp(1.0f - hoyer, 0.0f, 1.0f);
}


/*
 * Dendrite noise gate, shared by ngm2, ngm2.5 and ngm3 (CPU; the ngm3 GPU
 * kernels get the same windows as generated WGSL constants). An input counts as
 * noise when it is flat and dim:
 *
 *   attenuation = 1 - sigmoid((noise - start) / width, shape) * (1 - sigmoid(avg))
 *
 * noise is 1 - Hoyer sparsity of the input (hoyer_noise: 0 for one active
 * entry, 1 for a flat input). The (1 - sigmoid(avg)) factor lets bright inputs
 * through regardless of their shape.
 *
 * The windows were fitted on simulated neuron-group outputs of 50 to 400
 * entries (explore/nse_alt/hoyer_fit2.py) against these targets for the
 * attenuation factor:
 *                                       response      learning
 *   few strong winners (signal)         >= 0.9        >= 0.9
 *   many moderate winners (neither)     0.35 - 0.65   0.15 - 0.35
 *   flat and weak (noise)               <= 0.1        <= 0.1
 * They are targets, not guarantees: on the fit data the medians meet them
 * except learning on signal at 200-400 entries (0.86-0.88) and on "neither" at
 * 50 (0.40), and 44-77 % of the signal samples fall inside their band. The
 * gate is graded on purpose: "neither" inputs are damped, not cut off, and
 * learning on them is damped more than the response.
 * 1 - Hoyer at a fixed active fraction moves little with the input size
 * between 50 and 400 entries. Smaller inputs read sparser (a 30 % active
 * input of 8 entries passes the response gate at ~0.9), so the windows do not
 * carry over to inputs of only a few entries.
 */
struct noise_window_t {
    float           start;
    float           width;
    sigmoid_shape_t shape;
};

inline constexpr noise_window_t response_noise_window { 0.125f, 0.84f,  {0.25, 0.5} };
inline constexpr noise_window_t learning_noise_window { -0.25f, 1.405f, {0.5,  0.5} };

static_assert(response_noise_window.width > 0.0f && learning_noise_window.width > 0.0f);

inline float noise_attenuation(const float noise, const float avg, const noise_window_t &w)
{
    return 1.0f - sigmoid((noise - w.start) / w.width, w.shape) * (1.0f - sigmoid(avg, {0.5f,0.5f}));
}



// Per-buffer input statistics. Single shared definition used by every neuron
// group that consumes an input buffer. Previously these were computed inside
// io_buffer (io_buffer::stats); they are a consumer concern, so the group
// computes them once per step and passes them by reference to its dendrites.
struct input_stats_t {
    float sum     = 0.0f;
    float avg     = 0.0f;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float noise_hoyer = 0.0f;   // 1 - Hoyer sparsity: 0 for a one-hot vector, 1 for a flat one
    // Number of entries. compute_input_stats fills it; a record built by hand
    // must set it too, or noise_gate_min_t leaves the channel out (count < 2).
    std::size_t count = 0;
};

// Compute input statistics over a buffer. Semantics match the historical
// io_buffer::update_stats: max_val reduced from 0.0f (true data max for the
// non-negative inputs used here), min_val reduced from FLT_MAX. min_val is
// retained for completeness even though current consumers read only
// sum/avg/max_val/noise_hoyer — a min is a reasonable companion to a max for any
// multi-datapoint statistic.
inline input_stats_t compute_input_stats(std::span<const float> buf)
{
    input_stats_t s {};
    if (buf.empty())
        return s;
    s.sum     = std::reduce(buf.begin(), buf.end());
    s.avg     = s.sum / static_cast<float>(buf.size());
    s.min_val = std::reduce(
        buf.begin(), buf.end(),
        std::numeric_limits<float>::max(),
        [](const float a, const float b) { return std::min(a, b); });
    s.max_val = std::reduce(
        buf.begin(), buf.end(),
        0.0f,
        [](const float a, const float b) { return std::max(a, b); });
    s.noise_hoyer = hoyer_noise(buf);
    s.count       = buf.size();
    return s;
}

// The response gate over several input channels takes the least noisy and the
// darkest of them. A channel that carries no shape evidence is left out: one
// without input (zero or non-finite sum), whose noise 0 would read as a perfect
// signal and unmask a noisy sibling and whose avg 0 would cancel a bright
// sibling's bypass, and one of a single entry (noise 0 by definition, the same
// unmasking). With no channel left the gate is open (attenuation 1).
struct noise_gate_min_t {
    float noise = std::numeric_limits<float>::infinity();
    float avg   = std::numeric_limits<float>::infinity();
    bool  any   = false;

    void add(const input_stats_t &s)
    {
        if (!std::isnormal(s.sum) || s.count < 2)
            return;
        noise = std::min(s.noise_hoyer, noise);
        avg   = std::min(s.avg, avg);
        any   = true;
    }

    [[nodiscard]] float attenuation(const noise_window_t &w) const
    {
        return any ? noise_attenuation(noise, avg, w) : 1.0f;
    }
};

// The per-channel constants the flat models (ngm3, ngm2.5) expand to every
// input position of a channel, each step: one definition, with the guards
// both need. Values transcribe ngm2's dendrite (get_response / adapt_synapses).
struct channel_constants_t {
    float inv_sum;          // 1/sum (the response leaf normaliser); 0 for a zero sum
    float half_max;         // max/2: the penalty draw range
    float high_thres;       // avg/2 + FLT_EPSILON: adapt's high/low input split
    float inv_high_range;   // 1/(1 - high_thres); 0 once high_thres >= 1
    float inv_high_thres;   // 1/high_thres; 0 for 0
    float adapt_atten;      // the learning window's noise attenuation
};

inline channel_constants_t channel_constants(const input_stats_t &st)
{
    channel_constants_t c;
    c.inv_sum    = st.sum != 0.0f ? 1.0f / st.sum : 0.0f;
    c.half_max   = st.max_val * 0.5f;
    c.high_thres = st.avg * 0.5f + std::numeric_limits<float>::epsilon();
    // For normalized [0,1] input high_thres is in (0, 0.5], so this equals ngm2's
    // 1/(1 - high_thres). Only un-normalized input can push high_thres to 1
    // (inv_high_range -> inf) or past it (negative); either poisons a persisted
    // permanence via inf/NaN in the adapt ramp, so the ramp goes inert (0).
    c.inv_high_range = c.high_thres < 1.0f ? 1.0f / (1.0f - c.high_thres) : 0.0f;
    c.inv_high_thres = c.high_thres != 0.0f ? 1.0f / c.high_thres : 0.0f;
    c.adapt_atten    = noise_attenuation(st.noise_hoyer, st.avg, learning_noise_window);
    return c;
}

// Map from a source output id to the input statistics of that buffer. A neuron
// group computes this once per step (one entry per input buffer it consumes)
// and passes it by const reference down to its neurons / dendrites, so each
// buffer's statistics are computed exactly once per step. Key type is
// std::size_t (== sim::output_id_t).
using input_stats_map_t = std::unordered_map<std::size_t, input_stats_t>;


// The learning winner among the neurons that clear the winner threshold
// (SPEC.md M.4.4): of the k neurons with out + FLT_EPSILON >= win_act, the
// floor(u·k)-th in index order, so for u ~ U[0,1) each is chosen with
// probability 1/k and none is favoured by its index. u may round up to 1 (a
// float [0,1) draw can); it then picks the last. Returns out.size() if no neuron
// qualifies. The GPU winner kernels compute the same thing in f32.
[[nodiscard]] inline std::size_t pick_winner(std::span<const float> out, float win_act, float u)
{
    constexpr float eps = std::numeric_limits<float>::epsilon();
    std::size_t k = 0;
    for (const float o : out) k += (o + eps >= win_act) ? 1u : 0u;
    if (k == 0) return out.size();
    std::size_t j = std::min(static_cast<std::size_t>(u * static_cast<float>(k)), k - 1);
    for (std::size_t i = 0; i < out.size(); ++i)
        if (out[i] + eps >= win_act && j-- == 0) return i;
    return out.size();   // unreachable: k qualifying neurons were counted above
}

}

#endif //HD_NGM2_TOOLS_H
