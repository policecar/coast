//
// Retina node math
//

#include "retina/retina.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace retina {

namespace {

void require(bool ok, const std::string& what)
{
    if (!ok)
        throw std::invalid_argument("retina: " + what);
}

std::vector<float> gaussian_kernel(double sigma)
{
    const int radius = static_cast<int>(std::ceil(3.0 * sigma));
    std::vector<float> k(static_cast<std::size_t>(2 * radius + 1));
    for (int i = -radius; i <= radius; ++i)
        k[static_cast<std::size_t>(i + radius)] = static_cast<float>(std::exp(-0.5 * (i * i) / (sigma * sigma)));
    const float sum = std::reduce(k.begin(), k.end());
    for (auto& v : k)
        v /= sum;
    return k;
}

// Separable blur of one w x h plane with clamp-to-edge borders.
void blur(std::span<const float> src, std::span<float> dst, std::span<float> tmp,
          int w, int h, std::span<const float> kernel)
{
    const int r = static_cast<int>(kernel.size() / 2);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int i = -r; i <= r; ++i)
                acc += kernel[static_cast<std::size_t>(i + r)] * src[static_cast<std::size_t>(y * w + std::clamp(x + i, 0, w - 1))];
            tmp[static_cast<std::size_t>(y * w + x)] = acc;
        }
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int i = -r; i <= r; ++i)
                acc += kernel[static_cast<std::size_t>(i + r)] * tmp[static_cast<std::size_t>(std::clamp(y + i, 0, h - 1) * w + x)];
            dst[static_cast<std::size_t>(y * w + x)] = acc;
        }
}

}

void validate(const retina_param_t& p)
{
    require(p.width > 0 && p.height > 0 && p.channels > 0, "width, height and channels must be > 0");
    // blur() and apply() index planes with int
    require(std::int64_t{p.width} * p.height <= std::numeric_limits<int>::max() / p.channels,
            "width * height * channels must fit in an int");
    require(!p.luminance || p.channels == 3, "luminance needs channels == 3 (RGB), got " + std::to_string(p.channels));
    require(p.polarity == polarity_t::on || p.polarity == polarity_t::off, "polarity must be on or off");
    const auto sigma_ok = [](double s) { return std::isfinite(s) && min_sigma <= s && s <= max_sigma; };
    require(sigma_ok(p.center_sigma), "center_sigma must be in [0.1, 100] pixels");
    require(sigma_ok(p.surround_sigma), "surround_sigma must be in [0.1, 100] pixels");
    // A centre at least as wide as its surround inverts the polarity (a bright
    // spot drives OFF); as the sigmas approach each other the response at k = 1
    // cancels towards the noise floor.
    require(p.surround_sigma >= min_surround_ratio * p.center_sigma,
            "surround_sigma must be >= 1.2 * center_sigma");
    // apply() works in float: a larger value would turn into inf there, and
    // inf * 0 in a uniform region into NaN
    constexpr double float_max = std::numeric_limits<float>::max();
    require(std::isfinite(p.surround_weight) && 0.0 <= p.surround_weight && p.surround_weight <= float_max,
            "surround_weight must be in [0, FLT_MAX]");
    require(std::isfinite(p.gain) && 0.0 < p.gain && p.gain <= float_max, "gain must be in (0, FLT_MAX]");
    // A blank plane (RMS 0) needs the floor twice over: norm_sigma > 0 avoids
    // x / RMS = 0 / 0, and min_norm_sigma keeps gain / norm_sigma finite with a
    // wide margin: a much smaller value could overflow double at extreme gain
    // (below gain / DBL_MAX, ~1.9e-270 at FLT_MAX), and then scale * 0 is NaN
    require(!p.normalize || (std::isfinite(p.norm_sigma) && min_norm_sigma <= p.norm_sigma),
            "norm_sigma must be finite and >= 1e-12 when normalize is on");
    require(std::isfinite(p.noise_range.first) && std::isfinite(p.noise_range.second) &&
            0.0 <= p.noise_range.first && p.noise_range.first <= p.noise_range.second &&
            0.0 < p.noise_range.second && p.noise_range.second <= max_noise,
            "noise_range must satisfy 0 <= lo <= hi <= 0.1 and hi > 0");
}

std::size_t input_size(const retina_param_t& p)
{
    return static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.channels);
}

std::size_t output_size(const retina_param_t& p)
{
    const int out_channels = p.luminance ? 1 : p.channels;
    return static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height) * static_cast<std::size_t>(out_channels);
}

contrast_filter_t::contrast_filter_t(const retina_param_t& params) : params_(params)
{
    validate(params_);
    center_kernel_   = gaussian_kernel(params_.center_sigma);
    surround_kernel_ = gaussian_kernel(params_.surround_sigma);
    const auto pixels = static_cast<std::size_t>(params_.width) * static_cast<std::size_t>(params_.height);
    plane_.resize(pixels);
    center_.resize(pixels);
    surround_.resize(pixels);
    tmp_.resize(pixels);
}

void contrast_filter_t::apply(std::span<const float> in, std::span<float> d)
{
    assert(in.size() == input_size(params_));
    assert(d.size() == output_size(params_));
    const int w = params_.width, h = params_.height, c = params_.channels;
    const int out_c = params_.luminance ? 1 : c;
    const std::size_t pixels = plane_.size();
    const float k = static_cast<float>(params_.surround_weight);
    // Normalising, the loops below write the gain-free x and the pass after them
    // applies gain; otherwise they apply gain themselves, exactly as before the
    // option existed.
    const float g = params_.normalize ? 1.0f : static_cast<float>(params_.gain);

    for (int oc = 0; oc < out_c; ++oc) {
        const auto at = [&](std::size_t p) -> float& {
            return d[p * static_cast<std::size_t>(out_c) + static_cast<std::size_t>(oc)];
        };
        // gather one plane from the interleaved input
        for (std::size_t p = 0; p < pixels; ++p) {
            const float* px = &in[p * static_cast<std::size_t>(c)];
            plane_[p] = params_.luminance ? 0.299f * px[0] + 0.587f * px[1] + 0.114f * px[2] : px[oc];
        }
        const float mean = std::reduce(plane_.begin(), plane_.end()) / static_cast<float>(pixels);
        blur(plane_, center_, tmp_, w, h, center_kernel_);
        if (k == 0.0f) {   // the surround term vanishes; skip its blur
            for (std::size_t p = 0; p < pixels; ++p)
                at(p) = g * (center_[p] - mean);
        } else {
            blur(plane_, surround_, tmp_, w, h, surround_kernel_);
            for (std::size_t p = 0; p < pixels; ++p)
                at(p) = g * (center_[p] - k * surround_[p] - (1.0f - k) * mean);
        }
        if (params_.normalize) {
            // d = gain * x / (norm_sigma + RMS(x)), RMS over this whole plane. In
            // double: scale can reach FLT_MAX / min_norm_sigma (~3.4e50), beyond
            // float's range, and |x| <= sqrt(pixels) * RMS(x) keeps scale * x bounded.
            double ss = 0.0;
            for (std::size_t p = 0; p < pixels; ++p)
                ss += static_cast<double>(at(p)) * at(p);
            const double scale = params_.gain / (params_.norm_sigma + std::sqrt(ss / static_cast<double>(pixels)));
            for (std::size_t p = 0; p < pixels; ++p)
                at(p) = static_cast<float>(scale * at(p));
        }
    }
}

void polarize(std::span<const float> d, polarity_t polarity, std::span<float> out)
{
    assert(d.size() == out.size());
    const float sign = polarity == polarity_t::on ? 1.0f : -1.0f;
    for (std::size_t i = 0; i < d.size(); ++i)
        out[i] = std::clamp(sign * d[i], 0.0f, 1.0f);
}

}
