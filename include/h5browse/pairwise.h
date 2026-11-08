#ifndef COAST_PAIRWISE_H
#define COAST_PAIRWISE_H

// Pairwise distance/similarity histogram types (IDD D6/D6a).
//
// Sampled mode: k rows drawn uniformly without replacement from the
// filtered row set (k GUI-configurable, default 2000, seed 42), gathered
// through the chunk cache into a dense k x width f64 matrix — the ONLY
// dtype conversion point in the browser (D10); f64 (not f32) so wide
// integer sources keep full precision (values > 2^24 survive) — then all
// pairs via TBB with thread-local histograms merged at completion.
// Full mode: tiled block-pair iteration through the same code path
// (sampled = one tile), progressive and cancellable per tile pair.

#include "tools/metric_registry.h"

#include "h5browse/base_histogram.h"
#include "h5browse/row_filter.h"

#include <cstdint>
#include <span>

namespace h5b {

inline constexpr std::uint32_t pairwise_base_bins = 4096;

struct pairwise_request {
    coast::metric_kind   kind = coast::metric_kind::cosine_similarity;
    coast::metric_params params{};

    std::uint32_t k        = 2000; // sample size (D6a: GUI-configurable)
    std::uint64_t seed     = 42;   // bumped by the GUI "resample" action
    bool          full_run = false;

    row_filter filter;
};

inline constexpr std::uint64_t pairwise_tile_rows = 4096; // IDD-pinned

// Builds an empty histogram of the right form for a metric: fixed domain
// when the registry declares one (cosine [-1,1]), adaptive otherwise
// (distances are >= 0, so adaptive tiles anchor at 0 and merge exactly).
[[nodiscard]] base_histogram make_metric_histogram(coast::metric_kind kind);

// All pairs between row groups A and B (same_tile: A==B, j>i triangular;
// else the full A x B rectangle). Rows are dense f64, row-major, `width`
// elements each. TBB-parallel over A's rows with thread-local histograms
// merged into `out`. Returns the number of pairs *computed* — this is the
// progress denominator, NOT the histogram population: non-finite metric
// values (e.g. cosine of a zero-norm row = 0/0 = NaN) are dropped by
// base_histogram::insert, so out.total() <= the returned count.
std::uint64_t pairwise_tile(std::span<const double> a, std::uint64_t a_rows,
                            std::span<const double> b, std::uint64_t b_rows,
                            std::size_t width, bool same_tile,
                            coast::metric_kind          kind,
                            const coast::metric_params& params,
                            base_histogram&             out);

} // namespace h5b

#endif // COAST_PAIRWISE_H
