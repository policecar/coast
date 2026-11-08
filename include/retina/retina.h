//
// Retina node math: centre-surround filter and ON/OFF split.
//
// Kept free of the io_entity plumbing so it can be unit-tested in isolation.
//

#ifndef COAST_RETINA_H
#define COAST_RETINA_H

#include <cstddef>
#include <span>
#include <vector>

#include "retina/retina_cfg.h"

namespace retina {

// Accepted sigma range, pixels. At 0.1 the kernel is already a delta in float
// (its tails are exp(-50)); far below it sigma * sigma underflows to 0 and the
// kernel turns NaN. 100 exceeds every supported image and keeps the kernel
// (2 * ceil(3 sigma) + 1 taps) and the blur cost bounded.
inline constexpr double min_sigma = 0.1;
inline constexpr double max_sigma = 100.0;
// surround_sigma >= min_surround_ratio * center_sigma: keeps the surround clearly
// wider than the centre, so the filter neither inverts nor cancels.
inline constexpr double min_surround_ratio = 1.2;
// Upper end of noise_range: the floor is a null signal, not a second input.
inline constexpr double max_noise = 0.1;
// Lower bound of norm_sigma when normalize is on. With gain <= FLT_MAX it keeps
// the scale gain / (norm_sigma + RMS) finite in double (at most ~3.4e50), so a
// blank plane comes out 0 instead of inf * 0 = NaN.
inline constexpr double min_norm_sigma = 1e-12;

// Throws std::invalid_argument unless params are usable: positive sizes with
// width * height * channels <= INT_MAX, sigmas in [min_sigma, max_sigma] with
// surround_sigma >= min_surround_ratio * center_sigma, gain in (0, FLT_MAX],
// surround_weight in [0, FLT_MAX], norm_sigma finite and >= min_norm_sigma when
// normalize is on (ignored when off), luminance only with 3 channels,
// 0 <= noise_range.first <= noise_range.second <= max_noise and
// noise_range.second > 0
// (the noise keeps every output's sum above zero; see docs, retina module).
void validate(const retina_param_t& params);

[[nodiscard]] std::size_t input_size(const retina_param_t& params);
[[nodiscard]] std::size_t output_size(const retina_param_t& params);

// Signed centre-surround response, optionally divisively normalised per plane
// (see retina_cfg.h for both formulas). Gaussian
// blurs are separable, truncated at 3 sigma and renormalised, with nearest-pixel
// borders, so a uniform plane blurs to itself and k = 1 maps it to ~0.
class contrast_filter_t {
    retina_param_t     params_;
    std::vector<float> center_kernel_;
    std::vector<float> surround_kernel_;
    std::vector<float> plane_, center_, surround_, tmp_;   // scratch, one plane each

public:
    // Validates params (throws std::invalid_argument).
    explicit contrast_filter_t(const retina_param_t& params);

    // in: input_size(params) values (HWC); d: output_size(params) values (HWC).
    void apply(std::span<const float> in, std::span<float> d);
};

// ON: out = min(max(d, 0), 1); OFF: out = min(max(-d, 0), 1). Sizes must match.
void polarize(std::span<const float> d, polarity_t polarity, std::span<float> out);

}

#endif // COAST_RETINA_H
