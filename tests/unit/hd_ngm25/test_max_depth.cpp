// tests/unit/hd_ngm25/test_max_depth.cpp
//
// ngm25 (CPU) at the deepest tree it accepts, max_branch_level = 6: u8 segment
// ids up to max_seg_id = 127. Companion of tests/unit/hd_ngm2/test_max_depth.cpp
// (#237, where a u16 walk wrapped at ngm2's deepest tree): growth is forced
// with zero branch thresholds until some segment sits on the deepest level, and
// every step must return and keep segment ids in [1, 127]. Depth 7 stays
// rejected.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "sim_core/io_buffer.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace ngm25 {
// The seam declared `friend` by neuron_group_t (as in test_process_guard.cpp).
struct neuron_group_test_access {
    static const synapse_arena_t& arena(const neuron_group_t& g) { return g.arena_; }
};
} // namespace ngm25

using namespace ngm25;

namespace {

constexpr std::uint8_t k_depth = 6;

cfg_store make_store(std::uint8_t depth)
{
    neuron_group_params_t p;
    p.neuron_cnt              = 4;
    p.apical_cnt              = 0;
    p.proximal_cnt            = 2;
    p.max_branch_level        = depth;
    p.branch_interval         = 1;
    p.accumulated_theta_thres = 0.0f;
    p.min_mismatch_deviation  = 0.0f;
    p.min_mismatch_percentage = 0.0f;
    p.mismatch_smoothing      = 0.5f;
    p.mismatch_act_thres      = 0.0f;
    p.random_seed             = seed_t{ 7 };
    p.inputs                  = { { "in", bank_t::proximal } };
    cfg_store cs;
    cs.neuron_groups["g"] = p;
    return cs;
}

unsigned max_seg_id(const neuron_group_t& g)
{
    const auto& seg = neuron_group_test_access::arena(g).seg;
    REQUIRE(std::ranges::find(seg, std::uint8_t{ 0 }) == seg.end());
    return seg.empty() ? 0u : *std::ranges::max_element(seg);
}

} // namespace

TEST_CASE("depth 6 is accepted, depth 7 is rejected")
{
    CHECK_NOTHROW(neuron_group_t(make_store(6), "g", 1, seed_offset_t{ 0 }));
    CHECK_THROWS_AS(neuron_group_t(make_store(7), "g", 1, seed_offset_t{ 0 }), std::runtime_error);
}

TEST_CASE("ngm25 grows to depth 6 and processes there")
{
    constexpr std::size_t W = 8;
    const auto cs = make_store(k_depth);
    neuron_group_t g(cs, "g", /*output_id=*/1, seed_offset_t{ 0 });
    g.init_io({ { "in", sim::output_channel_t{ 0, W } } });
    REQUIRE(g.get_max_segment_id() == 127);
    sim::io_buffer_cpu in(W), out(g.get_outp_size());
    g.set_input_buffer(0, in);
    g.set_output_buffer(out);

    // Varied binary inputs keep the mismatch up at every depth, so growth does
    // not stall once two alternating patterns are told apart.
    std::uint32_t lcg = 12345;
    auto step = [&](std::size_t) {
        for (auto& v : in.cur_write_buffer()) { lcg = lcg * 1664525u + 1013904223u; v = (lcg >> 31) ? 0.95f : 0.05f; }
        in.swap_buffer();
        REQUIRE(g.process().has_value());
    };
    constexpr std::size_t k_cap = 2000;
    std::size_t steps = 0;
    unsigned deepest = 0;
    for (; steps < k_cap; ++steps) {
        step(steps);
        deepest = max_seg_id(g);
        REQUIRE(deepest <= 127);
        if (std::bit_width(deepest) - 1 == k_depth) break;
    }
    MESSAGE("deepest level reached after " << steps + 1 << " steps");
    REQUIRE(std::bit_width(deepest) - 1 == k_depth);
    for (std::size_t s = 0; s < 20; ++s) step(s);       // keep running at full depth
    CHECK(max_seg_id(g) <= 127);
    CHECK(g.get_total_representation_count() > 0);
}
