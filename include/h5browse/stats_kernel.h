#ifndef COAST_STATS_KERNEL_H
#define COAST_STATS_KERNEL_H

// Hot per-chunk loops, deliberately isolated in their own TU
// (src/h5browse/stats_kernel.cpp) — DVD production rule 4: the identical
// kernel source measured 5.01 vs 3.46 GB/s in two different TU
// compositions, so hot-kernel throughput is pinned by a perf test and the
// TU boundary keeps unrelated edits from perturbing its codegen.
//
// All kernels take native-dtype spans (D10) and widen per element to
// double internally. extern template below — the 10 instantiations live
// in the .cpp.

#include "h5browse/base_histogram.h"
#include "h5browse/row_filter.h"
#include "h5browse/scan_stats.h"

#include <cstdint>
#include <span>
#include <vector>

namespace h5b {

// 16-lane shifted-sums accumulation (DVD rule 2: 8.1x over scalar
// end-to-end; numerically equivalent to Welford at ~1e-11 relative).
template <class T>
void accumulate_stats(scan_stats& s, std::span<const T> values);

// One pass filling all per-column histograms (D4). rows is n_rows *
// row_width elements, row-major; first_row is the absolute index of the
// first row (for the filter's row range). per_column.size() == row_width.
// Returns the number of rows that passed the filter.
template <class T>
std::uint64_t insert_histograms(std::span<base_histogram> per_column,
                                std::span<const T> rows, std::size_t row_width,
                                const row_filter& filter,
                                std::uint64_t first_row);

// Appends the absolute indices of filter-matching rows (pairwise sampling
// and table-filter materialization; <= 8 B per matching row).
template <class T>
void collect_matching_rows(std::vector<std::uint64_t>& out,
                           std::span<const T> rows, std::size_t row_width,
                           const row_filter& filter, std::uint64_t first_row);

#define COAST_H5B_KERNEL_EXTERN(T)                                            \
    extern template void accumulate_stats<T>(scan_stats&, std::span<const T>);\
    extern template std::uint64_t insert_histograms<T>(                       \
        std::span<base_histogram>, std::span<const T>, std::size_t,           \
        const row_filter&, std::uint64_t);                                    \
    extern template void collect_matching_rows<T>(                            \
        std::vector<std::uint64_t>&, std::span<const T>, std::size_t,         \
        const row_filter&, std::uint64_t);

COAST_H5B_KERNEL_EXTERN(float)
COAST_H5B_KERNEL_EXTERN(double)
COAST_H5B_KERNEL_EXTERN(std::int8_t)
COAST_H5B_KERNEL_EXTERN(std::int16_t)
COAST_H5B_KERNEL_EXTERN(std::int32_t)
COAST_H5B_KERNEL_EXTERN(std::int64_t)
COAST_H5B_KERNEL_EXTERN(std::uint8_t)
COAST_H5B_KERNEL_EXTERN(std::uint16_t)
COAST_H5B_KERNEL_EXTERN(std::uint32_t)
COAST_H5B_KERNEL_EXTERN(std::uint64_t)

#undef COAST_H5B_KERNEL_EXTERN

} // namespace h5b

#endif // COAST_STATS_KERNEL_H
