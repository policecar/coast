#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "tools/metric.h"

using coast::compute_metric;
using coast::cosine_similarity;
using coast::metric_kind;
using coast::metric_params;
using coast::minkowski_distance;

namespace {
constexpr float kTol = 1e-5f;

std::span<const float> sp(const std::vector<float>& v) {
    return std::span<const float>(v.data(), v.size());
}
} // namespace

TEST_CASE("cosine: identical vectors -> 1") {
    std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(cosine_similarity(sp(a), sp(a)) == doctest::Approx(1.0f).epsilon(kTol));
}

TEST_CASE("cosine: orthogonal vectors -> 0") {
    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{0.0f, 1.0f};
    CHECK(cosine_similarity(sp(a), sp(b)) == doctest::Approx(0.0f).epsilon(kTol));
}

TEST_CASE("cosine: anti-parallel vectors -> -1") {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{-1.0f, -2.0f, -3.0f};
    CHECK(cosine_similarity(sp(a), sp(b)) == doctest::Approx(-1.0f).epsilon(kTol));
}

TEST_CASE("cosine: known value (45 degrees) -> 1/sqrt(2)") {
    // a=(1,0), b=(1,1): cos = 1 / (1 * sqrt(2)).
    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{1.0f, 1.0f};
    CHECK(cosine_similarity(sp(a), sp(b))
          == doctest::Approx(1.0f / std::sqrt(2.0f)).epsilon(kTol));
}

TEST_CASE("cosine: zero-vector guard -> 0") {
    std::vector<float> a{0.0f, 0.0f, 0.0f};
    std::vector<float> b{1.0f, 2.0f, 3.0f};
    CHECK(cosine_similarity(sp(a), sp(b)) == doctest::Approx(0.0f));
    CHECK(cosine_similarity(sp(a), sp(a)) == doctest::Approx(0.0f));
}

TEST_CASE("minkowski p=1: Manhattan known value") {
    std::vector<float> a{0.0f, 0.0f, 0.0f};
    std::vector<float> b{1.0f, 2.0f, 3.0f};
    // |1|+|2|+|3| = 6
    CHECK(minkowski_distance(sp(a), sp(b), 1.0f) == doctest::Approx(6.0f).epsilon(kTol));
}

TEST_CASE("minkowski p=2: Euclidean known value (3-4-5)") {
    std::vector<float> a{0.0f, 0.0f};
    std::vector<float> b{3.0f, 4.0f};
    CHECK(minkowski_distance(sp(a), sp(b), 2.0f) == doctest::Approx(5.0f).epsilon(kTol));
}

TEST_CASE("minkowski p=3: known value") {
    std::vector<float> a{0.0f, 0.0f};
    std::vector<float> b{1.0f, 1.0f};
    // (1^3 + 1^3)^(1/3) = 2^(1/3)
    CHECK(minkowski_distance(sp(a), sp(b), 3.0f)
          == doctest::Approx(std::cbrt(2.0f)).epsilon(kTol));
}

TEST_CASE("minkowski: identical -> 0") {
    std::vector<float> a{1.5f, -2.0f, 7.0f, 0.0f};
    CHECK(minkowski_distance(sp(a), sp(a), 1.0f) == doctest::Approx(0.0f));
    CHECK(minkowski_distance(sp(a), sp(a), 2.0f) == doctest::Approx(0.0f));
    CHECK(minkowski_distance(sp(a), sp(a), 4.0f) == doctest::Approx(0.0f));
}

TEST_CASE("minkowski: monotonic with separation") {
    std::vector<float> a{0.0f, 0.0f};
    std::vector<float> near{1.0f, 0.0f};
    std::vector<float> far{5.0f, 0.0f};
    CHECK(minkowski_distance(sp(a), sp(near), 2.0f)
          < minkowski_distance(sp(a), sp(far), 2.0f));
}

TEST_CASE("minkowski: p<=0 throws") {
    std::vector<float> a{1.0f, 2.0f};
    std::vector<float> b{3.0f, 4.0f};
    CHECK_THROWS_AS(minkowski_distance(sp(a), sp(b), 0.0f), std::invalid_argument);
    CHECK_THROWS_AS(minkowski_distance(sp(a), sp(b), -1.0f), std::invalid_argument);
}

TEST_CASE("dimension-mismatch precondition fires") {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{1.0f, 2.0f};
    CHECK_THROWS_AS(cosine_similarity(sp(a), sp(b)), std::invalid_argument);
    CHECK_THROWS_AS(minkowski_distance(sp(a), sp(b), 2.0f), std::invalid_argument);
    CHECK_THROWS_AS(compute_metric(metric_kind::cosine_similarity, {}, sp(a), sp(b)),
                    std::invalid_argument);
}

TEST_CASE("compute_metric dispatch matches direct calls") {
    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{4.0f, 5.0f, 6.0f};
    metric_params mp{2.5f};
    CHECK(compute_metric(metric_kind::cosine_similarity, mp, sp(a), sp(b))
          == doctest::Approx(cosine_similarity(sp(a), sp(b))));
    CHECK(compute_metric(metric_kind::minkowski_distance, mp, sp(a), sp(b))
          == doctest::Approx(minkowski_distance(sp(a), sp(b), mp.p)));
}
