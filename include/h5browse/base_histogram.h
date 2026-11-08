#ifndef COAST_BASE_HISTOGRAM_H
#define COAST_BASE_HISTOGRAM_H

// Fixed-bin-count histogram with power-of-two range doubling (IDD §5).
//
// Adaptive form: the range is always [0, 2^k) ("nonneg", while only
// non-negative values have been seen) or [-2^k, 2^k) ("symmetric").
// A value outside the range converts/doubles the range, merging adjacent
// bin pairs — every transition maps old bins exactly onto new bins, so
// counts are preserved and any two adaptive histograms are mergeable
// exactly (the nesting invariant is structural, not call-site dependent).
// This is G3's rebin-without-rescan by construction.
//
// Fixed-domain form (e.g. cosine [-1, 1]): the range never adapts;
// out-of-range values (fp jitter like 1+eps) clamp into the edge bins.
//
// NaN/Inf are excluded silently — scan_stats counts them.
//
// Resolution limit (documented, accepted): power-of-two ranges cannot
// resolve offsets tiny relative to magnitude (data clustered at
// 1000.0001 lands in one bin). min/max/stddev cover that case.

#include <cstdint>
#include <span>
#include <vector>

namespace h5b {

class base_histogram {
public:
    // Adaptive range. n_bins must be a multiple of 4 (range transitions
    // merge bin pairs around the n/2 and n/4 boundaries).
    explicit base_histogram(std::uint32_t n_bins);

    // Fixed domain [lo, hi]; requires lo < hi and n_bins % 4 == 0.
    base_histogram(std::uint32_t n_bins, double lo, double hi);

    void insert(double x); // NaN/Inf ignored

    // Exact merge. Both histograms must have the same bin count and the
    // same form (both adaptive, or both fixed with identical domains) —
    // internal invariant, asserted.
    void merge(const base_histogram& o);

    // Re-aggregated view with m bins; m must divide n_bins.
    [[nodiscard]] std::vector<std::uint64_t> rebin(std::uint32_t m) const;

    [[nodiscard]] std::span<const std::uint64_t> bins() const { return bins_; }
    [[nodiscard]] std::uint32_t n_bins() const { return static_cast<std::uint32_t>(bins_.size()); }
    [[nodiscard]] double lo() const { return lo_; }
    [[nodiscard]] double hi() const { return hi_; }
    [[nodiscard]] std::uint64_t total() const { return count_; }
    [[nodiscard]] bool empty() const { return count_ == 0; }
    [[nodiscard]] bool fixed_domain() const { return fixed_; }

private:
    void init_range(double x);
    void grow_to_fit(double x);
    void to_symmetric();  // [0, 2^k) -> [-2^k, 2^k), old bins -> upper half
    void double_range();  // k -> k+1, exact bin-pair merge
    [[nodiscard]] std::size_t bin_index(double x) const;

    std::vector<std::uint64_t> bins_;
    double        lo_ = 0.0, hi_ = 0.0;
    std::uint64_t count_     = 0;
    int           k_         = 0;     // adaptive range exponent
    bool          fixed_     = false;
    bool          symmetric_ = false;
    bool          has_range_ = false;
};

} // namespace h5b

#endif // COAST_BASE_HISTOGRAM_H
