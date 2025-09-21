//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_TOOLS_H
#define HD_NGM2_TOOLS_H

#include <cmath>
#include <numeric>
#include <type_traits>
#include <algorithm>
#include <random>
#include <span>
#include <print>

namespace ngm2 {

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
        //std::fill(vec.begin(),vec.end(),zero);
    }
}

struct sigmoid_shape_t {
    float steepness        = 0.5f;
    float transition_point = 0.5f;
};

inline float sigmoid(
    float x,
    sigmoid_shape_t shape = {0.5f, 0.5f}
){
    const float step_size      = 1.0f - std::pow(shape.steepness, 0.1f);
    const float starting_point = -shape.transition_point / step_size;
    return 1.0f / ( 1.0f + std::exp( -(x / step_size + starting_point) ) );
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

inline float normalized_shannon_entropy(std::span<const float> vec)
{
    if (vec.empty())
        return 0.0f;

    const float sum = std::reduce(vec.begin(),vec.end());
    if (!std::isnormal(sum))
        return 0.0f;

    const float entropy = -1.0f * std::transform_reduce(
        vec.begin(), vec.end(),
        0.0f, std::plus<>(),
        [sum](float val)
        {
            const float p = val / sum;
            return p >= std::numeric_limits<float>::epsilon() ? p * std::log2(p) : 0.0f;
        }
    );

    return std::clamp(entropy / std::log2(static_cast<float>(vec.size())), 0.0f, 1.0f);
}


inline void local_inhibition(std::span<float> vec, float strength = 1.0f, float rnd_activity = 0.0f)
{
    // desired behavior: strong signals suppress weak signals...
    // semi-strong signals should stand up better to the suppression than weak signals
    static std::mt19937 rgen {};
    std::uniform_real_distribution<float> dis(0.0,rnd_activity);
    if (vec.empty())
        return;
    float max_val = std::reduce(vec.begin(),vec.end(),0.0f,[](float a, float b){ return std::max(a,b); });
    if (!std::isnormal(max_val))
        return;
    for (auto &val : vec) {
        auto max_ratio = val / max_val;
        val *= std::pow(max_ratio,strength);
        val += dis(rgen);
        val = std::clamp(val,0.0f,1.0f);
    }
}

inline void local_inhibition2(std::span<float> vec, [[maybe_unused]] float strength = 1.0f)
{
    // desired behavior: strong signals suppress weak signals...
    // semi-strong signals should stand up better to the suppression than weak signals
    if (vec.empty())
        return;
    float max_val = std::reduce(vec.begin(),vec.end(),0.0f,[](float a, float b){ return std::max(a,b); });
    if (!std::isnormal(max_val))
        return;
    const float act_thres = max_val * max_val;
    for (auto &val : vec) {
        if (val < act_thres)
            val = 0.0f;
    }
}

inline void local_inhibition3(std::span<float> vec, float strength = 1.0f)
{
    // desired behavior: strong signals suppress weak signals...
    // semi-strong signals should stand up better to the suppression than weak signals
    constexpr std::size_t hist_size = 100;
    std::array<float,hist_size> histogram {};
    if (vec.empty())
        return;
    // determine max
    float max_val = std::reduce(vec.begin(),vec.end(),0.0f,[](float a, float b){ return std::max(a,b); });
    if (!std::isnormal(max_val))
        return;
    // create cumulative histogram
    std::ranges::fill(histogram,0.0f);
    const float vec_size = static_cast<float>(vec.size());
    for (auto &val : vec) {
        histogram[std::clamp(static_cast<std::size_t>(std::round(val * static_cast<float>(hist_size))),
                             0ul,
                             hist_size - 1)] += 1.0f / vec_size;
    }
    for (std::size_t hi = 1; hi < hist_size; ++hi) {
        histogram[hi] += histogram[hi-1];
    }
    // rescale vector
    for (auto &val : vec) {
        /*
        val = std::pow(histogram[std::clamp(static_cast<std::size_t>(std::round(val * static_cast<float>(hist_size))),
                                            0ul,
                                            hist_size - 1)],
                       1.0f / (1.0f - 1.0f/strength)) * max_val;
                       */
        val = histogram[std::clamp(static_cast<std::size_t>(std::round(val * static_cast<float>(hist_size))),
                                   0ul,
                                   hist_size - 1)] * max_val;
        auto max_ratio = val / max_val;
        val *= std::pow(max_ratio,strength);
    }
}

inline void local_inhibition4(std::span<float> vec, float strength = 1.0f, float rnd_activity = 0.0f)
{
    // desired behavior: strong signals suppress weak signals...
    // semi-strong signals should stand up better to the suppression than weak signals
    static std::mt19937 rgen {};
    std::uniform_real_distribution<float> dis(0.0,rnd_activity);
    if (vec.empty())
        return;
    const float max_val = std::reduce(vec.begin(),vec.end(),0.0f,[](float a, float b){ return std::max(a,b); });
    if (!std::isnormal(max_val))
        return;
    const float nse_fact = 1.0f - sigmoid(normalized_shannon_entropy(vec) - 0.8f / 0.2f);
    for (auto &val : vec) {
        auto max_ratio = val / max_val;
        val *= std::pow(max_ratio, 1.0f + (strength-1.0f) * nse_fact );
        val += dis(rgen);
        val = std::clamp(val,0.0f,1.0f);
    }
}

inline void local_inhibition5(std::span<float> vec, float strength = 1.0f, float rnd_activity = 0.0f)
{
    // desired behavior: strong signals suppress weak signals...
    // semi-strong signals should stand up better to the suppression than weak signals
    static std::mt19937 rgen {};
    if (vec.empty())
        return;

    softmax(vec,strength);

    normalize(vec);

    /*
    const float nse = normalized_shannon_entropy(vec);
    std::uniform_real_distribution<float> dis(0.0,(1.0 - nse) * rnd_activity);

    for (auto &val : vec) {
        val = val * (1.0f - rnd_activity) + dis(rgen);
        val = std::clamp(val,0.0f,1.0f);
    }
    */

}

}

#endif //HD_NGM2_TOOLS_H
