#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/gpu/gpu_device.h"
#include "inhibition/gpu_inhibition.h"
#include "inhibition/gpu_max_softmax.h"
#include "inhibition/max_softmax.h"

// CPU <-> GPU parity for max_softmax. The GPU kernel is a hand-port of
// max_softmax_apply; any drift (reduction start, pass-through rule) silently
// splits CPU and GPU runs of the same config. Covers both regimes of beta, a
// group larger than the 256-wide workgroup (strided reduction), ties at the max
// and the pass-through cases. Needs a real adapter: labeled "gpu", run outside
// the Claude Code sandbox.

namespace {

namespace g = coast::gpu;

// The inhibitor built from a max_softmax entry with strength (beta) `beta`.
inhibition::inhibition_params_t ms_params(double beta)
{
    return { inhibition::inhibition_type_t::max_softmax, beta };
}

std::vector<float> run_gpu(g::gpu_device& dev, const std::vector<float>& act, float beta)
{
    const std::size_t bytes = act.size() * sizeof(float);
    const auto act_in = dev.register_buffer(g::topology::gpu_resident, bytes, /*ping_pong=*/false);
    const auto out    = dev.register_buffer(g::topology::gpu_resident, bytes, /*ping_pong=*/false);
    dev.upload(act_in, std::as_bytes(std::span<const float>(act)));

    inhibition::gpu_max_softmax_t inh(ms_params(beta));
    g::step_plan plan;
    inh.emit(dev, { act_in, out, static_cast<std::uint32_t>(act.size()), {1u, 1u, 1u} }, plan);
    const auto pid = dev.register_plan();
    dev.build_plan(pid, plan);

    dev.begin_step();
    dev.enqueue(pid);
    dev.submit_step();
    dev.rendezvous_step();

    std::vector<float> res(act.size());
    REQUIRE(dev.readback(out, std::as_writable_bytes(std::span<float>(res))));
    return res;
}

void check_parity(g::gpu_device& dev, const std::vector<float>& act, float beta)
{
    std::vector<float> cpu(act.size());
    inhibition::max_softmax_apply(act, cpu, beta);
    const auto gpu = run_gpu(dev, act, beta);
    for (std::size_t i = 0; i < act.size(); ++i) {
        CAPTURE(i);
        // scale(1e-30): doctest's allowance is epsilon*(scale + |x|), so the
        // default scale 1 makes this an absolute 1e-4 check — which every
        // non-leader output (all < 1e-4 at beta 20) passes even when zeroed.
        // Truly relative now; keep outputs out of the subnormal range, where a
        // GPU flush-to-zero would fail it spuriously.
        CHECK(gpu[i] == doctest::Approx(cpu[i]).epsilon(1e-4).scale(1e-30));
    }
}

std::vector<float> uniform_vec(std::size_t n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = u(rng);
    return v;
}

} // namespace

TEST_CASE("gpu max_softmax matches CPU max_softmax_apply") {
    auto dev = g::make_gpu_device();
    REQUIRE(dev);

    SUBCASE("50 neurons, beta 10") { check_parity(*dev, uniform_vec(50, 1), 10.0f); }
    SUBCASE("50 neurons, beta 0 lifts every entry to the max") {
        const auto v = uniform_vec(50, 2);
        check_parity(*dev, v, 0.0f);
        const auto gpu = run_gpu(*dev, v, 0.0f);
        const float mx = *std::max_element(v.begin(), v.end());
        for (const float x : gpu) CHECK(x == doctest::Approx(mx));
    }
    SUBCASE("1000 neurons (> workgroup), max in the last stride") {
        auto v = uniform_vec(1000, 3);
        v[999] = 1.5f;
        check_parity(*dev, v, 20.0f);
    }
    SUBCASE("ties at the max keep their value") {
        std::vector<float> v(64, 0.2f);
        v[3] = v[40] = 0.8f;
        check_parity(*dev, v, 10.0f);
        const auto gpu = run_gpu(*dev, v, 10.0f);
        CHECK(gpu[3] == doctest::Approx(0.8f));
        CHECK(gpu[40] == doctest::Approx(0.8f));
    }
    SUBCASE("single neuron") { check_parity(*dev, {0.7f}, 10.0f); }
    SUBCASE("all-zero input passes through") {
        const std::vector<float> v(64, 0.0f);
        check_parity(*dev, v, 10.0f);
        for (const float x : run_gpu(*dev, v, 10.0f)) CHECK(x == 0.0f);
    }
    SUBCASE("negative-only input passes through (max reduced from 0)") {
        const std::vector<float> v(32, -0.5f);
        for (const float x : run_gpu(*dev, v, 10.0f)) CHECK(x == -0.5f);
        check_parity(*dev, v, 10.0f);
    }
}

TEST_CASE("make_gpu_inhibitor builds max_softmax and validates beta") {
    using inhibition::inhibition_params_t;
    using inhibition::inhibition_type_t;
    CHECK(inhibition::make_gpu_inhibitor(inhibition_params_t{ inhibition_type_t::max_softmax, 10.0 }) != nullptr);
    CHECK_THROWS_AS(inhibition::gpu_max_softmax_t(ms_params(-1.0f)), std::invalid_argument);
    CHECK_THROWS_AS(inhibition::gpu_max_softmax_t(ms_params(std::numeric_limits<float>::infinity())), std::invalid_argument);
    CHECK_THROWS_AS(inhibition::gpu_max_softmax_t(ms_params(std::numeric_limits<float>::quiet_NaN())), std::invalid_argument);
}
