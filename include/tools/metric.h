//
// metric.h — pure, GUI-free similarity/distance metrics for the Metric Explorer.
//
// Header-only `inline` free functions so both the GUI tool and the unit test just
// `#include "tools/metric.h"` with no extra translation unit. These are cold-path
// utilities (one call per matrix cell, off the sim hot loop), so preconditions are
// reported by *throwing* per the project's cold-path convention — NOT via
// COAST_REQUIRE, which is the sim-thread std::expected value channel.
//

#ifndef COAST_METRIC_H
#define COAST_METRIC_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

namespace coast {

enum class metric_kind : uint8_t {
    cosine_similarity,
    minkowski_distance,
};

struct metric_params {
    float p = 2.0f;  // Minkowski order; ignored by cosine.
};

/// Closed value range used as a histogram's x-domain.
struct value_range {
    float lo = 0.0f;
    float hi = 1.0f;
};

/// Theoretical value range of a metric, for a fixed x-axis when binning values.
///   - cosine    -> bounded, fixed [-1, 1] (independent of the data).
///   - minkowski -> unbounded below max; use [0, observed_max], falling back to
///                  hi = 1 when observed_max <= 0 so the range is never degenerate.
inline value_range metric_domain(metric_kind kind, float observed_max)
{
    switch (kind) {
        case metric_kind::cosine_similarity:  return { -1.0f, 1.0f };
        case metric_kind::minkowski_distance: return { 0.0f, observed_max > 0.0f ? observed_max : 1.0f };
    }
    return { 0.0f, 1.0f };  // unreachable; silences -Wreturn-type on some compilers.
}

/// 64-bit FNV-1a hash over the raw bytes of a float span. Order-sensitive and
/// deterministic across runs on a given platform; use for cheap fingerprinting /
/// change-detection of vectors, NOT as a cryptographic digest. Accepts
/// `std::span<const float>`, so non-const `std::span<float>` binds implicitly.
///
/// Caveat: this hashes bit patterns, so +0.0f and -0.0f hash differently and any
/// two distinct NaN encodings collide only if their bits match — value equality
/// and hash equality are NOT the same here.
inline uint64_t hash_span(std::span<const float> v)
{
    uint64_t h = 1469598103934665603ull;  // FNV offset basis
    for (const float f : v) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        for (int b = 0; b < 4; ++b) {
            h ^= static_cast<uint8_t>(bits >> (b * 8));
            h *= 1099511628211ull;         // FNV prime
        }
    }
    return h;
}

/// Cosine similarity in [-1, 1]. Returns 0 when either vector has zero norm
/// (the angle is undefined; 0 is the neutral "no alignment" choice).
/// Accumulates in double to keep D=784-element sums well-conditioned.
/// Templated on the element type so integer sources can be gathered as
/// double (no f32 pre-truncation of values > 2^24) while f32 callers are
/// unaffected.
template <class T>
inline float cosine_similarity(std::span<const T> a, std::span<const T> b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("cosine_similarity: a.size() != b.size()");

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double ai = a[i];
        const double bi = b[i];
        dot += ai * bi;
        na  += ai * ai;
        nb  += bi * bi;
    }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

/// Minkowski-p distance (>= 0). p=1 is Manhattan, p=2 is Euclidean. Requires p > 0.
template <class T>
inline float minkowski_distance(std::span<const T> a, std::span<const T> b, float p)
{
    if (a.size() != b.size())
        throw std::invalid_argument("minkowski_distance: a.size() != b.size()");
    if (!(p > 0.0f))
        throw std::invalid_argument("minkowski_distance: p must be > 0");

    const double pp = p;
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        acc += std::pow(d, pp);
    }
    return static_cast<float>(std::pow(acc, 1.0 / pp));
}

/// Dispatch on metric_kind. pre: a.size() == b.size() (enforced by callees).
template <class T>
inline float compute_metric(metric_kind kind, const metric_params& params,
                            std::span<const T> a, std::span<const T> b)
{
    switch (kind) {
        case metric_kind::cosine_similarity:  return cosine_similarity(a, b);
        case metric_kind::minkowski_distance: return minkowski_distance(a, b, params.p);
    }
    return 0.0f;  // unreachable; silences -Wreturn-type on some compilers.
}

/// Display name for UI combo boxes.
inline const char* metric_kind_name(metric_kind kind)
{
    switch (kind) {
        case metric_kind::cosine_similarity:  return "Cosine similarity";
        case metric_kind::minkowski_distance: return "Minkowski distance";
    }
    return "?";
}

} // namespace coast

#endif // COAST_METRIC_H
