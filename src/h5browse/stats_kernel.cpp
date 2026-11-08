// Hot kernels — own TU per DVD production rule 4 (codegen pinning).
// accumulate_stats is the T1-validated 16-lane shifted-sums kernel
// (.dew/design-verification/t1_scan_bench.cpp, accumulate_sums): one
// vectorizable pass computing count/nan/inf/min/max plus sum and sumsq of
// (x - K), K = first finite value of the block (kills catastrophic
// cancellation), Chan-merged into scan_stats. No per-element division.

#include "h5browse/stats_kernel.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

namespace h5b {

template <class T>
void accumulate_stats(scan_stats& out, std::span<const T> values)
{
    const T*          v = values.data();
    const std::size_t n = values.size();
    if (n == 0) return;

    constexpr int L = 16;

    // Shift constant K: first finite value. Leading non-finite values are
    // routed through the scalar path (floating-point inputs only).
    double      K     = 0.0;
    std::size_t start = 0;
    if constexpr (std::is_floating_point_v<T>) {
        while (start < n && !std::isfinite(static_cast<double>(v[start]))) {
            out.update(static_cast<double>(v[start]));
            ++start;
        }
        if (start == n) return;
        K = static_cast<double>(v[start]);
    } else {
        K = static_cast<double>(v[0]);
    }

    // SoA lane state so the inner loop autovectorizes (independent chains).
    double        sum[L] = {}, sumsq[L] = {}, mn[L], mx[L];
    std::uint64_t cnt[L] = {}, nans[L] = {}, infs[L] = {};
    for (int l = 0; l < L; ++l) {
        mn[l] = std::numeric_limits<double>::infinity();
        mx[l] = -std::numeric_limits<double>::infinity();
    }

    const std::size_t m      = n - start;
    const T*          p0     = v + start;
    const std::size_t blocks = m / L;
    for (std::size_t b = 0; b < blocks; ++b) {
        const T* p = p0 + b * L;
        for (int l = 0; l < L; ++l) {
            const double x = static_cast<double>(p[l]);
            if constexpr (std::is_floating_point_v<T>) {
                if (x != x) { ++nans[l]; continue; }
                if (std::isinf(x)) { ++infs[l]; continue; }
            }
            ++cnt[l];
            mn[l]          = std::min(mn[l], x);
            mx[l]          = std::max(mx[l], x);
            const double d = x - K;
            sum[l] += d;
            sumsq[l] += d * d;
        }
    }

    for (int l = 0; l < L; ++l) {
        if (cnt[l] == 0 && nans[l] == 0 && infs[l] == 0) continue;
        scan_stats s;
        s.count     = cnt[l];
        s.nan_count = nans[l];
        s.inf_count = infs[l];
        if (cnt[l] > 0) {
            const double mean_d = sum[l] / static_cast<double>(cnt[l]);
            s.min               = mn[l];
            s.max               = mx[l];
            s.mean              = K + mean_d;
            s.m2 = sumsq[l] - static_cast<double>(cnt[l]) * mean_d * mean_d;
            if (s.m2 < 0.0) s.m2 = 0.0; // fp guard
        }
        out.merge(s);
    }
    for (std::size_t i = start + blocks * L; i < n; ++i) {
        out.update(static_cast<double>(v[i]));
    }
}

template <class T>
std::uint64_t insert_histograms(std::span<base_histogram> per_column,
                                std::span<const T> rows, std::size_t row_width,
                                const row_filter& filter,
                                std::uint64_t first_row)
{
    assert(row_width > 0);
    assert(rows.size() % row_width == 0);
    assert(per_column.size() == row_width);
    const std::size_t n_rows  = rows.size() / row_width;
    std::uint64_t     matched = 0;
    for (std::size_t r = 0; r < n_rows; ++r) {
        if (!filter.row_in_range(first_row + r)) continue;
        const std::span<const T> row = rows.subspan(r * row_width, row_width);
        if (!row_matches(filter, row)) continue;
        ++matched;
        for (std::size_t c = 0; c < row_width; ++c) {
            per_column[c].insert(static_cast<double>(row[c]));
        }
    }
    return matched;
}

template <class T>
void collect_matching_rows(std::vector<std::uint64_t>& out,
                           std::span<const T> rows, std::size_t row_width,
                           const row_filter& filter, std::uint64_t first_row)
{
    assert(row_width > 0);
    assert(rows.size() % row_width == 0);
    const std::size_t n_rows = rows.size() / row_width;
    for (std::size_t r = 0; r < n_rows; ++r) {
        if (!filter.row_in_range(first_row + r)) continue;
        if (!row_matches(filter, rows.subspan(r * row_width, row_width))) continue;
        out.push_back(first_row + r);
    }
}

#define COAST_H5B_KERNEL_INSTANTIATE(T)                                       \
    template void accumulate_stats<T>(scan_stats&, std::span<const T>);       \
    template std::uint64_t insert_histograms<T>(                              \
        std::span<base_histogram>, std::span<const T>, std::size_t,           \
        const row_filter&, std::uint64_t);                                    \
    template void collect_matching_rows<T>(                                   \
        std::vector<std::uint64_t>&, std::span<const T>, std::size_t,         \
        const row_filter&, std::uint64_t);

COAST_H5B_KERNEL_INSTANTIATE(float)
COAST_H5B_KERNEL_INSTANTIATE(double)
COAST_H5B_KERNEL_INSTANTIATE(std::int8_t)
COAST_H5B_KERNEL_INSTANTIATE(std::int16_t)
COAST_H5B_KERNEL_INSTANTIATE(std::int32_t)
COAST_H5B_KERNEL_INSTANTIATE(std::int64_t)
COAST_H5B_KERNEL_INSTANTIATE(std::uint8_t)
COAST_H5B_KERNEL_INSTANTIATE(std::uint16_t)
COAST_H5B_KERNEL_INSTANTIATE(std::uint32_t)
COAST_H5B_KERNEL_INSTANTIATE(std::uint64_t)

#undef COAST_H5B_KERNEL_INSTANTIATE

} // namespace h5b
