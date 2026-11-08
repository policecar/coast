#ifndef COAST_SCAN_STATS_H
#define COAST_SCAN_STATS_H

// Tier-2 per-dataset sanity statistics (IDD §5). Double accumulators;
// merge() is associative and commutative, so results are independent of
// chunk order and of how work was split across batches — the property the
// 16-lane kernel in stats_kernel.cpp relies on.
//
// NaN/Inf are findings, not errors: they are counted and excluded from
// min/max/mean/m2.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace h5b {

struct scan_stats {
    std::uint64_t count = 0, nan_count = 0, inf_count = 0;
    double        min  = std::numeric_limits<double>::infinity();
    double        max  = -std::numeric_limits<double>::infinity();
    double        mean = 0.0;
    double        m2   = 0.0;

    void update(double x)
    {
        if (std::isnan(x)) { ++nan_count; return; }
        if (std::isinf(x)) { ++inf_count; return; }
        ++count;
        min = std::min(min, x);
        max = std::max(max, x);
        const double delta = x - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (x - mean);
    }

    // Chan's parallel merge.
    void merge(const scan_stats& o)
    {
        if (o.count == 0) {
            nan_count += o.nan_count;
            inf_count += o.inf_count;
            return;
        }
        if (count == 0) {
            const std::uint64_t nans = nan_count + o.nan_count;
            const std::uint64_t infs = inf_count + o.inf_count;
            *this     = o;
            nan_count = nans;
            inf_count = infs;
            return;
        }
        const std::uint64_t n     = count + o.count;
        const double        delta = o.mean - mean;
        mean += delta * static_cast<double>(o.count) / static_cast<double>(n);
        m2 += o.m2 + delta * delta * static_cast<double>(count) *
                         static_cast<double>(o.count) / static_cast<double>(n);
        count = n;
        min   = std::min(min, o.min);
        max   = std::max(max, o.max);
        nan_count += o.nan_count;
        inf_count += o.inf_count;
    }

    [[nodiscard]] double variance() const
    {
        return count > 1 ? m2 / static_cast<double>(count - 1) : 0.0;
    }

    [[nodiscard]] double stddev() const { return std::sqrt(variance()); }
};

} // namespace h5b

#endif // COAST_SCAN_STATS_H
