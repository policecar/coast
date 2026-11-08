// Pairwise computation kernels (IDD D6). TBB parallelism lives here and
// only here (IDD §8): parallel_for over one tile's rows with thread-local
// histograms merged at the end. Everything is pure CPU work on gathered
// f64 matrices — no HDF5, no executor.

#include "h5browse/pairwise.h"

#include "h5browse/base_histogram.h"

#include <tbb/combinable.h>
#include <tbb/parallel_for.h>

#include <cassert>

namespace h5b {

base_histogram make_metric_histogram(coast::metric_kind kind)
{
    for (const coast::metric_def& def : coast::metric_registry()) {
        if (def.kind == kind && def.fixed_domain) {
            return {pairwise_base_bins, def.fixed_domain->lo,
                    def.fixed_domain->hi};
        }
    }
    return base_histogram{pairwise_base_bins};
}

std::uint64_t pairwise_tile(std::span<const double> a, std::uint64_t a_rows,
                            std::span<const double> b, std::uint64_t b_rows,
                            std::size_t width, bool same_tile,
                            coast::metric_kind          kind,
                            const coast::metric_params& params,
                            base_histogram&             out)
{
    assert(width > 0);
    assert(a.size() == a_rows * width);
    assert(b.size() == b_rows * width);
    if (a_rows == 0 || b_rows == 0) return 0;

    tbb::combinable<base_histogram> locals(
        [&] { return make_metric_histogram(kind); });

    tbb::parallel_for(
        tbb::blocked_range<std::uint64_t>(0, a_rows),
        [&](const tbb::blocked_range<std::uint64_t>& range) {
            base_histogram& h = locals.local();
            for (std::uint64_t i = range.begin(); i != range.end(); ++i) {
                const std::span<const double> ri = a.subspan(i * width, width);
                const std::uint64_t j0 = same_tile ? i + 1 : 0;
                for (std::uint64_t j = j0; j < b_rows; ++j) {
                    const float v = coast::compute_metric(
                        kind, params, ri, b.subspan(j * width, width));
                    h.insert(static_cast<double>(v)); // NaN/Inf excluded
                }
            }
        });

    locals.combine_each([&](const base_histogram& h) { out.merge(h); });

    // Pairs computed (progress denominator). a_rows/b_rows are bounded by
    // pairwise_tile_rows (4096) at every call site, so this product cannot
    // overflow; the histogram may hold fewer (non-finite metrics dropped).
    assert(a_rows <= pairwise_tile_rows && b_rows <= pairwise_tile_rows);
    return same_tile ? a_rows * (a_rows - 1) / 2 : a_rows * b_rows;
}

} // namespace h5b
