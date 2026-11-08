#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "inhibition/inhibition.h"
#include "inhibition/max_softmax.h"

using inhibition::max_softmax_apply;

// The inhibitor built from a max_softmax entry with strength (beta) `beta`.
static inhibition::inhibition_params_t ms_params(double beta)
{
    return { inhibition::inhibition_type_t::max_softmax, beta };
}

TEST_CASE("max_softmax_apply: empty span is a no-op") {
    std::vector<float> v;
    max_softmax_apply(v, v, 10.0f);
    CHECK(v.empty());
}

TEST_CASE("max_softmax_apply: all-zero passes through (max not normal -> early return)") {
    std::vector<float> v{0.f, 0.f, 0.f};
    max_softmax_apply(v, v, 10.0f);
    CHECK(v == std::vector<float>{0.f, 0.f, 0.f});
}

TEST_CASE("max_softmax_apply: beta 0 lifts every entry to the max (uniform softmax, rescaled)") {
    std::vector<float> v{0.1f, 0.7f, 0.3f};
    max_softmax_apply(v, v, 0.0f);
    CHECK(v == std::vector<float>{0.7f, 0.7f, 0.7f});
}

TEST_CASE("max_softmax_apply: a weak entry is raised when exp(-beta*gap) > v/max") {
    // documented property, not a bug: the rescaled softmax is not damping-only
    std::vector<float> v{0.05f, 0.08f};
    max_softmax_apply(v, v, 10.0f);
    CHECK(v[0] == doctest::Approx(0.08f * std::exp(-0.3f)));
    CHECK(v[0] > 0.05f);
}

TEST_CASE("max_softmax_apply: equals max-rescaled softmax") {
    const std::vector<float> raw{0.05f, 0.2f, 0.68f, 0.6f, 0.0f};
    const float beta = 10.0f;
    // reference: softmax, then rescale so the top entry equals max(raw)
    std::vector<double> e(raw.size());
    double z = 0.0;
    for (std::size_t i = 0; i < raw.size(); ++i) { e[i] = std::exp(beta * raw[i]); z += e[i]; }
    const double smax = *std::ranges::max_element(e) / z;
    std::vector<float> out(raw.size());
    max_softmax_apply(raw, out, beta);
    for (std::size_t i = 0; i < raw.size(); ++i)
        CHECK(out[i] == doctest::Approx(0.68 * (e[i] / z) / smax).epsilon(1e-5));
}

TEST_CASE("max_softmax_apply: leader keeps its activity, others are damped by exp(-beta*gap)") {
    std::vector<float> v{0.4f, 0.6f};
    max_softmax_apply(v, v, 10.0f);
    CHECK(v[1] == 0.6f);
    CHECK(v[0] == doctest::Approx(0.6f * std::exp(-2.0f)));
}

TEST_CASE("max_softmax_apply: a flat, low response stays flat and low") {
    // the property softmax lacks: no blow-up of a group that does not respond
    std::vector<float> v{0.05f, 0.06f, 0.05f, 0.04f};
    max_softmax_apply(v, v, 10.0f);
    CHECK(*std::ranges::max_element(v) == 0.06f);
    CHECK(*std::ranges::min_element(v) > 0.06f * std::exp(-0.2f) - 1e-6f);
}

TEST_CASE("max_softmax_apply: the same relative pattern is damped less at a lower level") {
    // gap-based, not ratio-based: scaling the vector down shrinks the gaps
    std::vector<float> hi{0.3f, 0.6f}, lo{0.03f, 0.06f};
    max_softmax_apply(hi, hi, 10.0f);
    max_softmax_apply(lo, lo, 10.0f);
    CHECK(hi[0] / hi[1] < lo[0] / lo[1]);
}

TEST_CASE("max_softmax_apply: distinct raw/out copies before transform; raw untouched") {
    const std::vector<float> raw{0.4f, 0.6f};
    std::vector<float> out(2, -1.f);
    max_softmax_apply(raw, out, 10.0f);
    CHECK(raw == std::vector<float>{0.4f, 0.6f});
    CHECK(out[1] == 0.6f);
    CHECK(out[0] < 0.4f);
}

TEST_CASE("max_softmax_t: rejects negative or non-finite beta") {
    CHECK_THROWS_AS(inhibition::max_softmax_t(ms_params(-1.0f)), std::invalid_argument);
    CHECK_THROWS_AS(inhibition::max_softmax_t(ms_params(std::numeric_limits<float>::infinity())), std::invalid_argument);
    CHECK_THROWS_AS(inhibition::max_softmax_t(ms_params(std::numeric_limits<float>::quiet_NaN())), std::invalid_argument);
    CHECK_NOTHROW(inhibition::max_softmax_t(ms_params(0.0f)));
}

TEST_CASE("make_inhibitor: builds max_softmax from params") {
    inhibition::inhibition_params_t p;
    p.type = inhibition::inhibition_type_t::max_softmax;
    p.strength = 10.0;
    auto inh = inhibition::make_inhibitor(p);
    std::vector<float> v{0.4f, 0.6f};
    inh->inhibit(v, v);
    CHECK(v[1] == 0.6f);
    CHECK(v[0] == doctest::Approx(0.6f * std::exp(-2.0f)));
}
