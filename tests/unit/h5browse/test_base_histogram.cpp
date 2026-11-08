// h5browse base_histogram — range-doubling correctness.
//
// Contracts: count preservation across every range transition, exact
// rebin re-aggregation (G3), NaN/Inf exclusion, fixed-domain clamping,
// exact merge between any two adaptive histograms (nesting invariant),
// bounded degradation on extreme outliers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/base_histogram.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using h5b::base_histogram;

namespace {

std::uint64_t bin_sum(const base_histogram& h)
{
    std::uint64_t s = 0;
    for (std::uint64_t b : h.bins()) s += b;
    return s;
}

} // namespace

TEST_CASE("base_histogram: counts preserved across range doubling")
{
    base_histogram h(512);
    // Values spanning 12 octaves force repeated doubling.
    std::uint64_t inserted = 0;
    for (int e = 0; e < 12; ++e) {
        for (int i = 0; i < 100; ++i) {
            h.insert(std::ldexp(1.0, e) * (0.5 + 0.001 * i));
            ++inserted;
        }
    }
    CHECK(h.total() == inserted);
    CHECK(bin_sum(h) == inserted);
    CHECK(h.lo() == 0.0);
    CHECK(h.hi() >= 2048.0); // covers 2^11 values
}

TEST_CASE("base_histogram: negative value converts to symmetric form exactly")
{
    base_histogram h(512);
    for (int i = 0; i < 1000; ++i) h.insert(0.1 + 0.0005 * i); // nonneg [0,1)
    CHECK(h.lo() == 0.0);
    h.insert(-0.25); // triggers [0,2^k) -> [-2^k,2^k)
    CHECK(h.lo() < 0.0);
    CHECK(h.total() == 1001);
    CHECK(bin_sum(h) == 1001);
}

TEST_CASE("base_histogram: NaN/Inf silently excluded")
{
    base_histogram h(512);
    h.insert(std::numeric_limits<double>::quiet_NaN());
    h.insert(std::numeric_limits<double>::infinity());
    h.insert(-std::numeric_limits<double>::infinity());
    CHECK(h.total() == 0);
    CHECK(h.empty());
    h.insert(1.0);
    CHECK(h.total() == 1);
}

TEST_CASE("base_histogram: rebin preserves totals and aggregates exactly")
{
    base_histogram h(512);
    std::mt19937   rng(3);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    for (int i = 0; i < 10'000; ++i) h.insert(dist(rng));

    for (std::uint32_t m : {512u, 256u, 128u, 64u, 32u, 16u, 8u}) {
        const auto    r   = h.rebin(m);
        std::uint64_t sum = std::accumulate(r.begin(), r.end(), std::uint64_t{0});
        CHECK(r.size() == m);
        CHECK(sum == h.total());
    }
    // rebin(m) groups are exact sums of base bins.
    const auto r256 = h.rebin(256);
    for (std::size_t i = 0; i < 256; ++i) {
        CHECK(r256[i] == h.bins()[2 * i] + h.bins()[2 * i + 1]);
    }
}

TEST_CASE("base_histogram: fixed domain clamps out-of-range into edge bins")
{
    base_histogram h(512, -1.0, 1.0); // cosine domain
    h.insert(1.0 + 1e-9);             // fp jitter above hi
    h.insert(-1.0 - 1e-9);
    h.insert(0.0);
    CHECK(h.total() == 3);
    CHECK(h.bins().front() == 1);
    CHECK(h.bins().back() == 1);
    CHECK(h.lo() == -1.0); // fixed: range never adapts
    CHECK(h.hi() == 1.0);
}

TEST_CASE("base_histogram: merge equals single-histogram insertion (any forms)")
{
    // Cover the form matrix: nonneg×nonneg (different k), nonneg×symmetric,
    // symmetric×nonneg, and merge-into-empty.
    struct split {
        std::vector<double> a, b;
    };
    const std::vector<split> cases{
        {{0.1, 0.2, 0.3}, {10.0, 20.0, 700.0}},          // nonneg, k differs
        {{0.5, 1.5}, {-3.0, 4.0}},                        // nonneg × symmetric
        {{-1.0, 2.0}, {0.25, 0.75}},                      // symmetric × nonneg
        {{}, {1.0, -2.0, 3.0}},                           // empty × mixed
    };
    for (const auto& c : cases) {
        base_histogram ha(512), hb(512), hall(512);
        for (double x : c.a) { ha.insert(x); hall.insert(x); }
        for (double x : c.b) { hb.insert(x); hall.insert(x); }
        ha.merge(hb);
        CHECK(ha.total() == hall.total());
        CHECK(ha.lo() == hall.lo());
        CHECK(ha.hi() == hall.hi());
        const auto ba = ha.bins(), bl = hall.bins();
        for (std::size_t i = 0; i < ba.size(); ++i) CHECK(ba[i] == bl[i]);
    }
}

TEST_CASE("base_histogram: merge is exact under random thread-style splits")
{
    // Simulates pairwise thread-local histograms: nonneg distance data,
    // random split across 8 "threads", merged vs. sequential.
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> dist(0.0, 50.0);
    std::vector<base_histogram> parts(8, base_histogram(4096));
    base_histogram              whole(4096);
    for (int i = 0; i < 100'000; ++i) {
        const double x = dist(rng);
        parts[static_cast<std::size_t>(rng() % 8)].insert(x);
        whole.insert(x);
    }
    base_histogram merged(4096);
    for (const auto& p : parts) merged.merge(p);
    CHECK(merged.total() == whole.total());
    const auto bm = merged.bins(), bw = whole.bins();
    for (std::size_t i = 0; i < bm.size(); ++i) CHECK(bm[i] == bw[i]);
}

TEST_CASE("base_histogram: single extreme outlier degrades but never loses counts")
{
    base_histogram h(512);
    std::mt19937   rng(5);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 10'000; ++i) h.insert(dist(rng));
    h.insert(1.0e30); // ~100 doublings
    CHECK(h.total() == 10'001);
    CHECK(bin_sum(h) == 10'001);
    // Original data is now compressed into bin 0 — degraded, not lost.
    CHECK(h.bins()[0] == 10'000);
}

TEST_CASE("base_histogram: pathological magnitudes clamp, never crash")
{
    base_histogram h(512);
    h.insert(std::numeric_limits<double>::max());
    h.insert(-std::numeric_limits<double>::max());
    h.insert(std::numeric_limits<double>::denorm_min());
    h.insert(0.0);
    CHECK(h.total() == 4);
    CHECK(bin_sum(h) == 4);
    CHECK(std::isfinite(h.lo()));
    CHECK(std::isfinite(h.hi()));
}

TEST_CASE("base_histogram: zero-only and first-value edge cases")
{
    base_histogram h(512);
    h.insert(0.0);
    CHECK(h.total() == 1);
    CHECK(h.lo() == 0.0);
    CHECK(h.bins()[0] == 1);

    base_histogram hn(512);
    hn.insert(-0.5); // negative first value -> symmetric immediately
    CHECK(hn.lo() < 0.0);
    CHECK(hn.total() == 1);
}
