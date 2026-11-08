#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "tools/histogram.h"
#include "tools/metric.h"

using coast::histogram_bins;
using coast::metric_domain;
using coast::metric_kind;
using coast::order_by_key;

namespace {
std::span<const float> sp(const std::vector<float>& v) {
    return std::span<const float>(v.data(), v.size());
}
int total(const std::vector<int>& v) { return std::accumulate(v.begin(), v.end(), 0); }
} // namespace

// ---------------------------------------------------------------------------
// histogram_bins
// ---------------------------------------------------------------------------

TEST_CASE("histogram_bins: counts sum to N") {
    std::vector<float> v{0.0f, 0.1f, 0.5f, 0.9f, 1.0f, 0.3f, 0.7f};
    auto h = histogram_bins(sp(v), 10, 0.0f, 1.0f);
    CHECK(h.size() == 10);
    CHECK(total(h) == static_cast<int>(v.size()));
}

TEST_CASE("histogram_bins: value at lo -> bin 0, value at hi -> last bin") {
    std::vector<float> v{0.0f, 1.0f};
    auto h = histogram_bins(sp(v), 8, 0.0f, 1.0f);
    CHECK(h.front() == 1);
    CHECK(h.back() == 1);
    CHECK(total(h) == 2);
}

TEST_CASE("histogram_bins: uniform input -> ~equal bins") {
    std::vector<float> v;
    for (int i = 0; i < 1000; ++i) v.push_back(static_cast<float>(i) / 1000.0f);  // [0, 1)
    auto h = histogram_bins(sp(v), 10, 0.0f, 1.0f);
    for (int c : h) CHECK(c == 100);
}

TEST_CASE("histogram_bins: values outside [lo,hi] clamp into the end bins") {
    std::vector<float> v{-5.0f, -0.001f, 1.001f, 99.0f};
    auto h = histogram_bins(sp(v), 4, 0.0f, 1.0f);
    CHECK(h.front() == 2);  // both below-lo values
    CHECK(h.back() == 2);   // both above-hi values
    CHECK(total(h) == 4);
}

TEST_CASE("histogram_bins: hi == lo puts everything in bin 0") {
    std::vector<float> v{3.0f, 3.0f, 3.0f};
    auto h = histogram_bins(sp(v), 5, 3.0f, 3.0f);
    CHECK(h.front() == 3);
    CHECK(total(h) == 3);
}

TEST_CASE("histogram_bins: empty input -> all-zero bins of requested size") {
    std::vector<float> v;
    auto h = histogram_bins(sp(v), 6, 0.0f, 1.0f);
    CHECK(h.size() == 6);
    CHECK(total(h) == 0);
}

TEST_CASE("histogram_bins: invalid args throw") {
    std::vector<float> v{0.5f};
    CHECK_THROWS_AS(histogram_bins(sp(v), 0, 0.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS_AS(histogram_bins(sp(v), -3, 0.0f, 1.0f), std::invalid_argument);
    CHECK_THROWS_AS(histogram_bins(sp(v), 4, 1.0f, 0.0f), std::invalid_argument);  // hi < lo
}

// ---------------------------------------------------------------------------
// order_by_key
// ---------------------------------------------------------------------------

TEST_CASE("order_by_key: ascending (smallest first)") {
    std::vector<float> k{0.3f, 0.1f, 0.2f};
    CHECK(order_by_key(sp(k), false) == std::vector<int>{1, 2, 0});
}

TEST_CASE("order_by_key: descending (largest first)") {
    std::vector<float> k{0.3f, 0.1f, 0.2f};
    CHECK(order_by_key(sp(k), true) == std::vector<int>{0, 2, 1});
}

TEST_CASE("order_by_key: stable on ties (keeps input order within equal keys)") {
    std::vector<float> k{0.5f, 0.5f, 0.5f, 0.2f};
    CHECK(order_by_key(sp(k), false) == std::vector<int>{3, 0, 1, 2});  // 0.2 first; ties stable
    CHECK(order_by_key(sp(k), true)  == std::vector<int>{0, 1, 2, 3});  // ties stable; 0.2 last
}

TEST_CASE("order_by_key: empty -> empty") {
    std::vector<float> k;
    CHECK(order_by_key(sp(k), false).empty());
    CHECK(order_by_key(sp(k), true).empty());
}

// ---------------------------------------------------------------------------
// metric_domain
// ---------------------------------------------------------------------------

TEST_CASE("metric_domain: cosine -> [-1, 1] regardless of observed max") {
    auto r = metric_domain(metric_kind::cosine_similarity, 42.0f);
    CHECK(r.lo == doctest::Approx(-1.0f));
    CHECK(r.hi == doctest::Approx(1.0f));
}

TEST_CASE("metric_domain: minkowski(max = 7) -> [0, 7]") {
    auto r = metric_domain(metric_kind::minkowski_distance, 7.0f);
    CHECK(r.lo == doctest::Approx(0.0f));
    CHECK(r.hi == doctest::Approx(7.0f));
}

TEST_CASE("metric_domain: minkowski(max <= 0) -> [0, 1] fallback") {
    auto r0 = metric_domain(metric_kind::minkowski_distance, 0.0f);
    CHECK(r0.lo == doctest::Approx(0.0f));
    CHECK(r0.hi == doctest::Approx(1.0f));
    auto rneg = metric_domain(metric_kind::minkowski_distance, -3.0f);
    CHECK(rneg.hi == doctest::Approx(1.0f));
}
