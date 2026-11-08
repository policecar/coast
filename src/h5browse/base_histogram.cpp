#include "h5browse/base_histogram.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace h5b {

namespace {

// Largest supported range exponent. At k_max the symmetric form spans
// [-2^1020, 2^1020), i.e. width 2^1021 — comfortably below DBL_MAX, so
// hi - lo stays finite; values beyond that clamp into the edge bins.
constexpr int k_max = 1020;

// Lower bound on the range exponent (one binade above DBL_MIN's 2^-1022,
// a deliberate safety margin); values below 2^k_min land in bin 0.
constexpr int k_min = -1021;

// Smallest k in [k_min, k_max] with |x| < 2^k.
int exponent_for(double x)
{
    const double a = std::fabs(x);
    if (a == 0.0) return 0;
    int e = std::ilogb(a) + 1;                       // a < 2^e
    if (a >= std::ldexp(1.0, e)) ++e;                // ilogb edge safety
    return std::clamp(e, k_min, k_max);
}

} // namespace

base_histogram::base_histogram(std::uint32_t n_bins) : bins_(n_bins, 0)
{
    assert(n_bins >= 4 && n_bins % 4 == 0);
}

base_histogram::base_histogram(std::uint32_t n_bins, double lo, double hi)
    : bins_(n_bins, 0), lo_(lo), hi_(hi), fixed_(true), has_range_(true)
{
    assert(n_bins >= 4 && n_bins % 4 == 0);
    assert(std::isfinite(lo) && std::isfinite(hi) && lo < hi);
}

void base_histogram::init_range(double x)
{
    k_         = exponent_for(x);
    symmetric_ = x < 0.0;
    hi_        = std::ldexp(1.0, k_);
    lo_        = symmetric_ ? -hi_ : 0.0;
    has_range_ = true;
}

void base_histogram::to_symmetric()
{
    // [0, 2^k) -> [-2^k, 2^k): old bin pair (2j, 2j+1) -> new bin n/2 + j.
    const std::size_t          n = bins_.size();
    std::vector<std::uint64_t> nb(n, 0);
    for (std::size_t j = 0; j < n / 2; ++j) {
        nb[n / 2 + j] = bins_[2 * j] + bins_[2 * j + 1];
    }
    bins_      = std::move(nb);
    symmetric_ = true;
    lo_        = -hi_;
}

void base_histogram::double_range()
{
    assert(k_ < k_max);
    const std::size_t          n = bins_.size();
    std::vector<std::uint64_t> nb(n, 0);
    if (symmetric_) {
        // [-2^k, 2^k) -> [-2^(k+1), 2^(k+1)): old pair (2j, 2j+1) -> n/4 + j.
        for (std::size_t j = 0; j < n / 2; ++j) {
            nb[n / 4 + j] = bins_[2 * j] + bins_[2 * j + 1];
        }
    } else {
        // [0, 2^k) -> [0, 2^(k+1)): old pair (2j, 2j+1) -> j.
        for (std::size_t j = 0; j < n / 2; ++j) {
            nb[j] = bins_[2 * j] + bins_[2 * j + 1];
        }
    }
    bins_ = std::move(nb);
    ++k_;
    hi_ = std::ldexp(1.0, k_);
    lo_ = symmetric_ ? -hi_ : 0.0;
}

void base_histogram::grow_to_fit(double x)
{
    if (x < 0.0 && !symmetric_) to_symmetric();
    while ((x < lo_ || x >= hi_) && k_ < k_max) double_range();
    // If x still does not fit (|x| >= 2^k_max), bin_index clamps it into
    // an edge bin — bounded behavior on pathological magnitudes.
}

std::size_t base_histogram::bin_index(double x) const
{
    const std::size_t n = bins_.size();
    const double      w = hi_ - lo_;
    double            i = std::floor((x - lo_) / w * static_cast<double>(n));
    if (i < 0.0) i = 0.0;
    const auto idx = static_cast<std::size_t>(i);
    return idx >= n ? n - 1 : idx;
}

void base_histogram::insert(double x)
{
    if (!std::isfinite(x)) return; // excluded; scan_stats counts NaN/Inf
    if (!fixed_) {
        if (!has_range_) init_range(x);
        if (x < lo_ || x >= hi_) grow_to_fit(x);
    }
    ++bins_[bin_index(x)];
    ++count_;
}

void base_histogram::merge(const base_histogram& o)
{
    assert(bins_.size() == o.bins_.size());
    assert(fixed_ == o.fixed_);
    if (o.count_ == 0) return;

    if (fixed_) {
        assert(lo_ == o.lo_ && hi_ == o.hi_);
        for (std::size_t i = 0; i < bins_.size(); ++i) bins_[i] += o.bins_[i];
        count_ += o.count_;
        return;
    }

    if (count_ == 0) {
        *this = o;
        return;
    }

    // Align the two adaptive grids exactly: same form, then same exponent.
    base_histogram tmp = o;
    if (symmetric_ && !tmp.symmetric_) tmp.to_symmetric();
    if (!symmetric_ && tmp.symmetric_) to_symmetric();
    while (k_ < tmp.k_) double_range();
    while (tmp.k_ < k_) tmp.double_range();

    for (std::size_t i = 0; i < bins_.size(); ++i) bins_[i] += tmp.bins_[i];
    count_ += tmp.count_;
}

std::vector<std::uint64_t> base_histogram::rebin(std::uint32_t m) const
{
    assert(m > 0 && bins_.size() % m == 0);
    const std::size_t          group = bins_.size() / m;
    std::vector<std::uint64_t> out(m, 0);
    for (std::size_t i = 0; i < bins_.size(); ++i) out[i / group] += bins_[i];
    return out;
}

} // namespace h5b
