// h5browse scan_stats + accumulate_stats kernel.
//
// Contracts from the Design Verification Document (T1/H1b):
//  - the 16-lane shifted-sums kernel is numerically equivalent to scalar
//    Welford (~1e-11 relative on mean, m2)
//  - merge() is associative -> results independent of batch splits
//  - NaN/Inf are counted, excluded from min/max/mean/m2

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/scan_stats.h"
#include "h5browse/stats_kernel.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

using h5b::accumulate_stats;
using h5b::scan_stats;

namespace {

// Naive two-pass reference: exact mean, then exact m2.
scan_stats reference_stats(const std::vector<double>& v)
{
    scan_stats s;
    double     sum = 0.0;
    for (double x : v) {
        if (std::isnan(x)) { ++s.nan_count; continue; }
        if (std::isinf(x)) { ++s.inf_count; continue; }
        ++s.count;
        s.min = std::min(s.min, x);
        s.max = std::max(s.max, x);
        sum += x;
    }
    if (s.count == 0) return s;
    s.mean = sum / static_cast<double>(s.count);
    for (double x : v) {
        if (!std::isfinite(x)) continue;
        s.m2 += (x - s.mean) * (x - s.mean);
    }
    return s;
}

std::vector<double> random_values(std::size_t n, std::uint32_t seed)
{
    std::mt19937                     rng(seed);
    std::normal_distribution<double> dist(5.0, 3.0);
    std::vector<double>              v(n);
    for (double& x : v) x = dist(rng);
    return v;
}

void check_close(const scan_stats& a, const scan_stats& b, double rel = 1e-10)
{
    CHECK(a.count == b.count);
    CHECK(a.nan_count == b.nan_count);
    CHECK(a.inf_count == b.inf_count);
    if (a.count == 0) return;
    CHECK(a.min == b.min);
    CHECK(a.max == b.max);
    CHECK(a.mean == doctest::Approx(b.mean).epsilon(rel));
    CHECK(a.m2 == doctest::Approx(b.m2).epsilon(1e-8));
}

} // namespace

TEST_CASE("scan_stats: update matches two-pass reference")
{
    const auto v = random_values(10'000, 1);
    scan_stats s;
    for (double x : v) s.update(x);
    check_close(s, reference_stats(v));
}

TEST_CASE("scan_stats: merge is associative and split-independent")
{
    const auto v = random_values(9'999, 2);

    scan_stats whole;
    for (double x : v) whole.update(x);

    // Three uneven splits, merged in two different association orders.
    auto part = [&](std::size_t lo, std::size_t hi) {
        scan_stats s;
        for (std::size_t i = lo; i < hi; ++i) s.update(v[i]);
        return s;
    };
    const scan_stats a = part(0, 1234), b = part(1234, 7000), c = part(7000, v.size());

    scan_stats ab = a;
    ab.merge(b);
    scan_stats ab_c = ab;
    ab_c.merge(c);

    scan_stats bc = b;
    bc.merge(c);
    scan_stats a_bc = a;
    a_bc.merge(bc);

    check_close(ab_c, whole, 1e-12);
    check_close(a_bc, whole, 1e-12);
    check_close(ab_c, a_bc, 1e-12);
}

TEST_CASE("accumulate_stats: shifted-sums kernel equivalent to Welford reference")
{
    // Sizes chosen to exercise the lane remainder path (not multiples of 16).
    for (std::size_t n : {std::size_t{1}, std::size_t{15}, std::size_t{16},
                          std::size_t{17}, std::size_t{100'003}}) {
        const auto          vd = random_values(n, static_cast<std::uint32_t>(n));
        std::vector<float>  vf(vd.begin(), vd.end());
        std::vector<double> as_double(vf.begin(), vf.end());

        scan_stats kernel;
        accumulate_stats<float>(kernel, std::span<const float>(vf));
        check_close(kernel, reference_stats(as_double));
    }
}

TEST_CASE("accumulate_stats: NaN/Inf counted and excluded, incl. non-finite prefix")
{
    constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
    constexpr double inf  = std::numeric_limits<double>::infinity();

    // Leading non-finite values exercise the scalar prefix path (K search).
    std::vector<double> v{qnan, inf, -inf, 1.0, 2.0, qnan, 3.0, inf, 4.0};
    scan_stats          s;
    accumulate_stats<double>(s, std::span<const double>(v));
    CHECK(s.count == 4);
    CHECK(s.nan_count == 2);
    CHECK(s.inf_count == 3);
    CHECK(s.min == 1.0);
    CHECK(s.max == 4.0);
    CHECK(s.mean == doctest::Approx(2.5));

    // All-non-finite input: counts only, no finite stats.
    std::vector<double> bad{qnan, inf, qnan};
    scan_stats          sb;
    accumulate_stats<double>(sb, std::span<const double>(bad));
    CHECK(sb.count == 0);
    CHECK(sb.nan_count == 2);
    CHECK(sb.inf_count == 1);
}

TEST_CASE("accumulate_stats: shift constant kills catastrophic cancellation")
{
    // Large offset, tiny variance — naive sum-of-squares would lose it.
    std::vector<double> v(50'000);
    std::mt19937        rng(7);
    std::normal_distribution<double> dist(1.0e9, 0.5);
    for (double& x : v) x = dist(rng);

    scan_stats s;
    accumulate_stats<double>(s, std::span<const double>(v));
    const scan_stats ref = reference_stats(v);
    CHECK(s.stddev() == doctest::Approx(ref.stddev()).epsilon(1e-6));
    CHECK(s.stddev() > 0.4);
    CHECK(s.stddev() < 0.6);
}

TEST_CASE("accumulate_stats: all 10 dtypes produce exact integer stats")
{
    // 0..99 in every integral type: mean 49.5, min 0, max 99.
    auto check_int = [](auto tag) {
        using T = typename decltype(tag)::type;
        std::vector<T> v(100);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<T>(i);
        scan_stats s;
        accumulate_stats<T>(s, std::span<const T>(v));
        CHECK(s.count == 100);
        CHECK(s.min == 0.0);
        CHECK(s.max == 99.0);
        CHECK(s.mean == doctest::Approx(49.5).epsilon(1e-12));
    };
    check_int(std::type_identity<std::int8_t>{});
    check_int(std::type_identity<std::int16_t>{});
    check_int(std::type_identity<std::int32_t>{});
    check_int(std::type_identity<std::int64_t>{});
    check_int(std::type_identity<std::uint8_t>{});
    check_int(std::type_identity<std::uint16_t>{});
    check_int(std::type_identity<std::uint32_t>{});
    check_int(std::type_identity<std::uint64_t>{});
    check_int(std::type_identity<float>{});
    check_int(std::type_identity<double>{});
}
