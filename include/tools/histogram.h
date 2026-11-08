//
// histogram.h — pure, GUI-free binning + ordering helpers for the Metric Explorer.
//
// Header-only `inline` free functions (same pattern as tools/metric.h and
// tools/distortion.h): both the GUI tool and the unit test just
// `#include "tools/histogram.h"` with no extra translation unit. These are
// cold-path utilities (run once per matrix recompute / re-bin, off the sim hot
// loop), so preconditions are reported by *throwing* per the project's cold-path
// convention — NOT via COAST_REQUIRE.
//

#ifndef COAST_HISTOGRAM_H
#define COAST_HISTOGRAM_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace coast {

/// Bin `vals` into `nbins` equal-width buckets over the closed domain [lo, hi].
/// Returns a length-`nbins` count vector that sums to `vals.size()`.
///
/// bin(v) = clamp(floor((v - lo) / (hi - lo) * nbins), 0, nbins - 1)
///   - v == lo  -> bin 0;  v == hi -> bin (nbins - 1).
///   - values outside [lo, hi] clamp into the first / last bin (never dropped).
///   - hi == lo (degenerate domain): every value lands in bin 0.
///
/// pre: nbins >= 1 and hi >= lo (a NaN endpoint also fails hi >= lo) — else throw.
inline std::vector<int> histogram_bins(std::span<const float> vals, int nbins,
                                       float lo, float hi)
{
    if (nbins < 1)
        throw std::invalid_argument("histogram_bins: nbins must be >= 1");
    if (!(hi >= lo))
        throw std::invalid_argument("histogram_bins: hi must be >= lo (and finite)");

    std::vector<int> counts(static_cast<std::size_t>(nbins), 0);

    const float span = hi - lo;
    if (span == 0.0f) {              // degenerate domain: all in bin 0
        counts[0] = static_cast<int>(vals.size());
        return counts;
    }

    const float inv = static_cast<float>(nbins) / span;
    for (const float v : vals) {
        int b = static_cast<int>(std::floor((v - lo) * inv));
        b = std::clamp(b, 0, nbins - 1);
        ++counts[static_cast<std::size_t>(b)];
    }
    return counts;
}

/// Stable display order of indices [0..n) sorted by `keys`.
///   - descending == false -> ascending key order (smallest first).
///   - descending == true  -> largest first.
///   - ties keep their original index order (std::stable_sort).
///   - empty input -> empty result.
/// The result maps display position -> underlying index (display->underlying).
inline std::vector<int> order_by_key(std::span<const float> keys, bool descending)
{
    std::vector<int> idx(keys.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return descending ? (keys[static_cast<std::size_t>(a)] > keys[static_cast<std::size_t>(b)])
                          : (keys[static_cast<std::size_t>(a)] < keys[static_cast<std::size_t>(b)]);
    });
    return idx;
}

} // namespace coast

#endif // COAST_HISTOGRAM_H
