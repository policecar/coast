//
// distortion.h — pure, GUI-free signal distortions for the Metric Explorer's
// "distortion mode" (Layer 4): given a sample, produce a corrupted copy of it so
// a metric can be probed against controlled degradation.
//
// Header-only `inline` free functions so both the GUI tool and the unit test just
// `#include "tools/distortion.h"`. Cold-path utilities (one call per matrix column,
// off the sim hot loop), so precondition violations are reported by *throwing* per
// the project's cold-path convention.
//
// Convention: datasets and synthetic codes are normalized to [0,1], so every
// distorted value is clamped back into [0,1]. magnitude == 0 is the identity.
//

#ifndef COAST_DISTORTION_H
#define COAST_DISTORTION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace coast {

enum class distortion_kind : uint8_t {
    gaussian_noise,
    dropout,
};

struct distortion_params {
    float         magnitude = 0.1f;  // gaussian: sigma; dropout: drop probability p
    std::uint32_t seed      = 0;
};

/// Add zero-mean Gaussian noise (std-dev `sigma`) to each element, then clamp to
/// [0,1]. sigma == 0 is the identity (exact copy). pre: sigma >= 0 (throws otherwise).
/// `out` is resized to in.size().
inline void apply_gaussian_noise(std::span<const float> in, std::vector<float>& out,
                                 float sigma, std::mt19937& rng)
{
    if (!(sigma >= 0.0f))
        throw std::invalid_argument("apply_gaussian_noise: sigma must be >= 0");

    out.resize(in.size());
    if (sigma == 0.0f) {                 // identity: leave normalized values untouched
        std::copy(in.begin(), in.end(), out.begin());
        return;
    }
    std::normal_distribution<float> nd(0.0f, sigma);
    for (std::size_t i = 0; i < in.size(); ++i)
        out[i] = std::clamp(in[i] + nd(rng), 0.0f, 1.0f);
}

/// Zero exactly round(p * n) distinct elements chosen uniformly at random; all
/// other elements are copied through unchanged. p == 0 is the identity; p == 1
/// zeroes everything. pre: 0 <= p <= 1 (throws otherwise). `out` is resized.
inline void apply_dropout(std::span<const float> in, std::vector<float>& out,
                          float p, std::mt19937& rng)
{
    if (!(p >= 0.0f && p <= 1.0f))
        throw std::invalid_argument("apply_dropout: p must be in [0,1]");

    out.assign(in.begin(), in.end());
    const std::size_t n = in.size();
    if (n == 0) return;

    const std::size_t k = static_cast<std::size_t>(std::lround(static_cast<double>(p) * n));
    if (k == 0) return;

    // Partial Fisher-Yates: pick k distinct indices to zero (bounded, deterministic).
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    for (std::size_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<std::size_t> d(i, n - 1);
        std::swap(idx[i], idx[d(rng)]);
        out[idx[i]] = 0.0f;
    }
}

/// Dispatch on distortion_kind, seeding a fresh rng from params.seed. `magnitude`
/// is the sigma (gaussian) or drop probability (dropout). `out` is resized to
/// in.size(); values are in [0,1].
inline void apply_distortion(distortion_kind kind, const distortion_params& params,
                             std::span<const float> in, std::vector<float>& out)
{
    std::mt19937 rng(params.seed);
    switch (kind) {
        case distortion_kind::gaussian_noise:
            apply_gaussian_noise(in, out, params.magnitude, rng);
            return;
        case distortion_kind::dropout:
            apply_dropout(in, out, params.magnitude, rng);
            return;
    }
}

/// Display name for UI combo boxes.
inline const char* distortion_kind_name(distortion_kind kind)
{
    switch (kind) {
        case distortion_kind::gaussian_noise: return "Gaussian noise";
        case distortion_kind::dropout:        return "Dropout";
    }
    return "?";
}

} // namespace coast

#endif // COAST_DISTORTION_H
