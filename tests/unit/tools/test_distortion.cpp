#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "tools/distortion.h"

using coast::apply_distortion;
using coast::apply_dropout;
using coast::apply_gaussian_noise;
using coast::distortion_kind;
using coast::distortion_params;

namespace {
std::span<const float> sp(const std::vector<float>& v) {
    return std::span<const float>(v.data(), v.size());
}
std::size_t count_zeros(const std::vector<float>& v) {
    return static_cast<std::size_t>(std::count(v.begin(), v.end(), 0.0f));
}
} // namespace

// --- Gaussian noise -------------------------------------------------------

TEST_CASE("gaussian sigma=0 -> identity") {
    std::vector<float> in{0.1f, 0.5f, 0.9f, 0.0f, 1.0f};
    std::vector<float> out;
    std::mt19937 rng(123);
    apply_gaussian_noise(sp(in), out, 0.0f, rng);
    CHECK(out == in);
}

TEST_CASE("gaussian: same seed identical, different seed differs") {
    std::vector<float> in(64, 0.5f);
    std::vector<float> a, b, c;
    { std::mt19937 r(7);  apply_gaussian_noise(sp(in), a, 0.2f, r); }
    { std::mt19937 r(7);  apply_gaussian_noise(sp(in), b, 0.2f, r); }
    { std::mt19937 r(99); apply_gaussian_noise(sp(in), c, 0.2f, r); }
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("gaussian: output stays in [0,1]") {
    std::vector<float> in(256, 0.5f);
    std::vector<float> out;
    std::mt19937 rng(1);
    apply_gaussian_noise(sp(in), out, 0.5f, rng);   // large sigma -> clamping exercised
    for (float v : out) CHECK((v >= 0.0f && v <= 1.0f));
}

TEST_CASE("gaussian: sigma<0 throws") {
    std::vector<float> in{0.5f};
    std::vector<float> out;
    std::mt19937 rng(1);
    CHECK_THROWS_AS(apply_gaussian_noise(sp(in), out, -0.1f, rng), std::invalid_argument);
}

// --- Dropout --------------------------------------------------------------

TEST_CASE("dropout p=0 -> identity") {
    std::vector<float> in{0.1f, 0.5f, 0.9f, 0.3f};
    std::vector<float> out;
    std::mt19937 rng(5);
    apply_dropout(sp(in), out, 0.0f, rng);
    CHECK(out == in);
}

TEST_CASE("dropout p=1 -> all zero") {
    std::vector<float> in(50, 0.7f);
    std::vector<float> out;
    std::mt19937 rng(5);
    apply_dropout(sp(in), out, 1.0f, rng);
    CHECK(count_zeros(out) == 50u);
}

TEST_CASE("dropout 0<p<1 -> exactly round(p*n) zeros, survivors unchanged") {
    std::vector<float> in(100, 0.5f);   // all non-zero so dropped == zero count
    std::vector<float> out;
    std::mt19937 rng(42);
    apply_dropout(sp(in), out, 0.3f, rng);
    CHECK(count_zeros(out) == 30u);     // round(0.3 * 100)
    for (std::size_t i = 0; i < out.size(); ++i)
        if (out[i] != 0.0f) CHECK(out[i] == doctest::Approx(in[i]));
}

TEST_CASE("dropout: p out of range throws") {
    std::vector<float> in{0.5f, 0.5f};
    std::vector<float> out;
    std::mt19937 rng(1);
    CHECK_THROWS_AS(apply_dropout(sp(in), out, -0.1f, rng), std::invalid_argument);
    CHECK_THROWS_AS(apply_dropout(sp(in), out,  1.1f, rng), std::invalid_argument);
}

// --- Dispatch -------------------------------------------------------------

TEST_CASE("apply_distortion matches the direct calls (same seed)") {
    std::vector<float> in(128, 0.5f);
    distortion_params dp; dp.seed = 2025;

    SUBCASE("gaussian") {
        dp.magnitude = 0.2f;
        std::vector<float> via_dispatch, direct;
        apply_distortion(distortion_kind::gaussian_noise, dp, sp(in), via_dispatch);
        std::mt19937 rng(dp.seed);
        apply_gaussian_noise(sp(in), direct, dp.magnitude, rng);
        CHECK(via_dispatch == direct);
    }
    SUBCASE("dropout") {
        dp.magnitude = 0.4f;
        std::vector<float> via_dispatch, direct;
        apply_distortion(distortion_kind::dropout, dp, sp(in), via_dispatch);
        std::mt19937 rng(dp.seed);
        apply_dropout(sp(in), direct, dp.magnitude, rng);
        CHECK(via_dispatch == direct);
    }
}

TEST_CASE("apply_distortion: invalid magnitude throws") {
    std::vector<float> in{0.5f};
    std::vector<float> out;
    distortion_params dp; dp.magnitude = -1.0f;
    CHECK_THROWS_AS(apply_distortion(distortion_kind::gaussian_noise, dp, sp(in), out),
                    std::invalid_argument);
    dp.magnitude = 2.0f;
    CHECK_THROWS_AS(apply_distortion(distortion_kind::dropout, dp, sp(in), out),
                    std::invalid_argument);
}
