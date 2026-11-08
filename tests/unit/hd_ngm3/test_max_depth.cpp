// tests/unit/hd_ngm3/test_max_depth.cpp
//
// ngm3 at the deepest tree it accepts, max_branch_level = 4 (the u8
// segment-id walks in adapt are bounded by max_seg_id = 31, MAX_SEG_CNT = 32).
// Companion of tests/unit/hd_ngm2/test_max_depth.cpp (#237, where a u16 walk
// wrapped at ngm2's deepest tree): growth is forced with zero branch
// thresholds until some segment sits on the deepest level, and every step
// must return and keep segment ids in [1, 31]. Depth 5 stays rejected.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

struct test_group_t : ngm3::neuron_group_t {
    using ngm3::neuron_group_t::neuron_group_t;
    using ngm3::neuron_group_t::init_storage;
    using ngm3::neuron_group_t::compute_step;
};

constexpr std::size_t  k_input_size = 8;
constexpr std::uint8_t k_depth      = 4;

ngm3::cfg_store make_store(std::uint8_t depth)
{
    ngm3::cfg_store cs;
    auto& g = cs.neuron_groups["g"];
    g.neuron_cnt              = 4;
    g.apical_cnt              = 0;
    g.proximal_cnt            = 2;
    g.max_branch_level        = depth;
    g.branch_interval         = 1;
    g.accumulated_theta_thres = 0.0f;
    g.min_mismatch_deviation  = 0.0f;
    g.min_mismatch_percentage = 0.0f;
    g.mismatch_smoothing      = 0.5f;
    g.mismatch_act_thres      = 0.0f;
    g.random_seed             = seed_t{ 7 };
    return cs;
}

unsigned max_seg_id(const test_group_t& g)
{
    unsigned m = 0;
    for (std::size_t n = 0; n < 4; ++n)
        for (std::size_t d = 0; d < 2; ++d)
            for (std::size_t r = 0; r < g.get_rep_count(); ++r)
                for (const auto s : g.get_proximal_segment_ids(n, d, r)) {
                    REQUIRE(s >= 1);
                    m = std::max<unsigned>(m, s);
                }
    return m;
}

} // namespace

TEST_CASE("depth 4 is accepted, depth 5 is rejected")
{
    CHECK_NOTHROW(test_group_t(make_store(4), "g", 1));
    CHECK_THROWS_AS(test_group_t(make_store(5), "g", 1), std::runtime_error);
}

TEST_CASE("ngm3 grows to depth 4 and processes there")
{
    const auto cs = make_store(k_depth);
    test_group_t g(cs, "g", /*output_id=*/1);
    g.init_storage(/*apical_in_size=*/0, /*proximal_in_size=*/k_input_size);
    REQUIRE(g.get_max_segment_id() == 31);

    // Varied binary inputs keep the mismatch up at every depth, so growth does
    // not stall once two alternating patterns are told apart.
    std::vector<float> in(k_input_size);
    std::uint32_t lcg = 12345;
    auto step = [&] {
        for (auto& v : in) { lcg = lcg * 1664525u + 1013904223u; v = (lcg >> 31) ? 0.95f : 0.05f; }
        REQUIRE(g.compute_step({}, std::span<const float>(in)).has_value());
    };
    constexpr std::size_t k_cap = 2000;
    std::size_t steps = 0;
    unsigned deepest = 0;
    for (; steps < k_cap; ++steps) {
        step();
        deepest = max_seg_id(g);
        REQUIRE(deepest <= 31);
        if (std::bit_width(deepest) - 1 == k_depth) break;
    }
    MESSAGE("deepest level reached after " << steps + 1 << " steps");
    REQUIRE(std::bit_width(deepest) - 1 == k_depth);
    for (std::size_t s = 0; s < 20; ++s) step();        // keep running at full depth
    CHECK(max_seg_id(g) <= 31);
}
