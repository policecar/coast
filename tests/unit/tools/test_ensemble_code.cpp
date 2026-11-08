#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "tools/ensemble_code.h"
#include "tools/metric.h"

using coast::cosine_similarity;
using coast::ensemble_code_params;
using coast::ensemble_codes;
using coast::generate_ensemble_codes;

namespace {
// View onto code `c` (the full padded stride = side*side).
std::span<const float> code_span(const ensemble_codes& e, int c) {
    const std::size_t stride = static_cast<std::size_t>(e.side) * e.side;
    return std::span<const float>(e.data.data() + static_cast<std::size_t>(c) * stride, stride);
}
} // namespace

TEST_CASE("geometry: length = N * S^2 with S = ceil(sqrt(D))") {
    ensemble_code_params p; p.D = 256; p.N = 64;
    auto e = generate_ensemble_codes(p);
    CHECK(e.side == 16);                 // ceil(sqrt(256))
    CHECK(e.count == 64);
    CHECK(e.data.size() == static_cast<std::size_t>(64) * 16 * 16);
}

TEST_CASE("all values within [0,1]") {
    ensemble_code_params p; p.seed = 3;
    auto e = generate_ensemble_codes(p);
    for (float v : e.data) CHECK((v >= 0.0f && v <= 1.0f));
}

TEST_CASE("padding past D is zero (non-square D)") {
    ensemble_code_params p; p.D = 200; p.N = 8;   // S = ceil(sqrt(200)) = 15, stride 225
    auto e = generate_ensemble_codes(p);
    REQUIRE(e.side == 15);
    const std::size_t stride = 225;
    for (int c = 0; c < p.N; ++c)
        for (std::size_t i = static_cast<std::size_t>(p.D); i < stride; ++i)
            CHECK(e.data[static_cast<std::size_t>(c) * stride + i] == 0.0f);
}

TEST_CASE("each code has exactly k winners above threshold (defaults, recognition=1)") {
    ensemble_code_params p;             // D=256,k=8,recognition=1,amp_var=0.15 -> winners in [0.85,1]
    auto e = generate_ensemble_codes(p);
    for (int c = 0; c < p.N; ++c) {
        auto s = code_span(e, c);
        int above = 0;
        for (float v : s) if (v > 0.5f) ++above;
        CHECK(above == p.k);
    }
}

TEST_CASE("cluster structure: same-prototype cosine > cross-prototype cosine") {
    ensemble_code_params p;             // M=4, N=64
    auto e = generate_ensemble_codes(p);

    double same_sum = 0.0, cross_sum = 0.0;
    int    same_n = 0, cross_n = 0;
    for (int i = 0; i < p.N; ++i)
        for (int j = i + 1; j < p.N; ++j) {
            const float cs = cosine_similarity(code_span(e, i), code_span(e, j));
            if (i % p.M == j % p.M) { same_sum += cs; ++same_n; }
            else                    { cross_sum += cs; ++cross_n; }
        }
    REQUIRE(same_n > 0);
    REQUIRE(cross_n > 0);
    const double mean_same  = same_sum  / same_n;
    const double mean_cross = cross_sum / cross_n;
    INFO("mean_same=", mean_same, " mean_cross=", mean_cross);
    CHECK(mean_same > mean_cross);
    CHECK(mean_same > mean_cross + 0.1);   // clearly, not marginally, separable
}

TEST_CASE("recognition=0 -> bounded low-amplitude noise (no winners)") {
    ensemble_code_params p; p.recognition = 0.0f; p.noise = 0.05f;
    auto e = generate_ensemble_codes(p);
    float mx = 0.0f;
    for (float v : e.data) mx = std::max(mx, v);
    CHECK(mx <= 3.0f * p.noise + 1e-6f);   // background is capped at 3*noise by construction
    CHECK(mx < 0.5f);                      // and far below winner amplitude
}

TEST_CASE("deterministic: same params + seed -> identical buffer") {
    ensemble_code_params p; p.seed = 12345;
    auto a = generate_ensemble_codes(p);
    auto b = generate_ensemble_codes(p);
    CHECK(a.data == b.data);
}

TEST_CASE("precondition violations throw") {
    ensemble_code_params p;
    p = {}; p.D = 10; p.k = 11;   CHECK_THROWS_AS(generate_ensemble_codes(p), std::invalid_argument);
    p = {}; p.D = 0;              CHECK_THROWS_AS(generate_ensemble_codes(p), std::invalid_argument);
    p = {}; p.M = 0;              CHECK_THROWS_AS(generate_ensemble_codes(p), std::invalid_argument);
}
