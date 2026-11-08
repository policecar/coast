// tests/unit/hd_ngm2/test_max_depth.cpp
//
// ngm2 at the deepest tree it accepts, max_branch_level = 4 (issue #237).
// ngm2 used to accept depth 15, where get_leaf_mask() walked its last level
// with a u16 counter that wrapped from 65535 to 0, so process() never returned
// (it evaluates the representation-count probe on every response). Depth is now
// capped at 4; these cases check that every accepted depth processes, counts and
// branches correctly at its deepest level. ctest's TIMEOUT turns a hang into a
// failure rather than a blocked run.
//
// The deep trees are injected through the in-memory checkpoint (as in
// test_state_shape.cpp), except in the last case, which grows them by learning.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "logging/mem_state_io.h"
#include "sim_core/io_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using logging::mem_group;
using logging::mem_state_reader;
using logging::mem_state_writer;

namespace {

constexpr sim::output_id_t k_id    = 0;
constexpr sim::output_id_t k_in_id = 1;
constexpr std::uint8_t     k_depth = 4;       // the deepest ngm2 accepts
constexpr std::uint32_t    k_max_seg = 31;    // 2^(k_depth+1) - 1
constexpr std::uint32_t    k_leaf_begin = 16; // first segment id of the deepest level

// Two neurons of two proximal dendrites each, branching checked on every step.
// `eager_branching` drops the branch gate's thresholds to 0, so any synapse
// whose mismatch clears mean + 1/input_size qualifies; `fast_mismatch` also
// lets every synapse's mismatch follow its input at once (smoothing 0.5).
ngm2::cfg_store make_store(std::uint8_t depth, bool eager_branching = false, bool fast_mismatch = false)
{
    ngm2::cfg_store cs;
    ngm2::dendrite_params_t dp;
    dp.input_channels   = { "da" };
    dp.max_branch_level = depth;
    if (eager_branching) {
        dp.default_accumulated_theta_thres = 0.0;
        dp.default_min_mismatch_deviation  = 0.0;
        dp.default_min_mismatch_percentage = 0.0;
    }
    if (fast_mismatch) {
        dp.default_mismatch_smoothing = 0.5;
        dp.default_mismatch_act_thres = 0.0;
    }
    cs.dendrites["prox"] = dp;

    ngm2::neuron_params_t np;
    np.input_channels          = { "na" };
    np.dendrites               = { { "p", "prox", 2 } };
    np.channel_mappings        = { { "na", "p", "da" } };
    np.default_branch_interval = 1;
    cs.neurons["n"] = np;

    ngm2::neuron_group_params_t gp;
    gp.input_channels   = { "ga" };
    gp.neurons          = { { "pyr", "n", 2 } };
    gp.channel_mappings = { { "ga", "pyr", "na" } };
    cs.neuron_groups["g"] = gp;
    return cs;
}

struct rig {
    std::unique_ptr<ngm2::neuron_group_t> g;
    sim::io_buffer_cpu in;
    sim::io_buffer_cpu out;

    rig(std::size_t width, bool eager_branching = false, bool fast_mismatch = false)
        : g(std::make_unique<ngm2::neuron_group_t>(make_store(k_depth, eager_branching, fast_mismatch), "g", k_id)),
          in(width), out(2)
    {
        g->init_io({ { "ga", sim::output_channel_t{ k_in_id, width } } });
        g->set_input_buffer(k_in_id, in);
        g->set_output_buffer(out);
    }

    // `steps` process() calls on a varying, non-zero input.
    void run(std::size_t steps)
    {
        for (std::size_t s = 0; s < steps; ++s) {
            auto w = in.cur_write_buffer();
            for (std::size_t i = 0; i < w.size(); ++i)
                w[i] = ((i + s) % 3 == 0) ? 1.0f : 0.25f;
            in.swap_buffer();
            REQUIRE(g->process().has_value());
        }
    }

    // One process() call on a pseudo-random binary input.
    void run_random()
    {
        for (auto& v : in.cur_write_buffer()) {
            lcg = lcg * 1664525u + 1013904223u;
            v = (lcg >> 31) ? 0.95f : 0.05f;
        }
        in.swap_buffer();
        REQUIRE(g->process().has_value());
    }
    std::uint32_t lcg = 12345;

    const ngm2::dendrite_t& dendrite(std::size_t n, std::size_t d) const
    { return g->get_neuron(n).get_dendrite(d); }
};

// One synapse per entry of `segs` on each of the dendrite's `width` inputs, in
// that order (input_inc 1 on the last of each run); mismatch per entry `mm`.
struct synapse_layout {
    std::vector<std::uint16_t> segs;
    std::vector<float>         mm;
};

void inject(mem_group& tree, std::size_t neuron, std::size_t dendrite,
            std::size_t width, const synapse_layout& l)
{
    auto& f = tree.groups.at("neurons").groups.at(std::to_string(neuron))
                  .groups.at("dendrites").groups.at(std::to_string(dendrite)).fields;
    std::vector<float> perm, mism, hist;
    std::vector<std::uint16_t> seg;
    std::vector<std::uint8_t> inc;
    for (std::size_t i = 0; i < width; ++i)
        for (std::size_t k = 0; k < l.segs.size(); ++k) {
            perm.push_back(0.9f);
            mism.push_back(l.mm.empty() ? 0.0f : l.mm[k]);
            hist.push_back(1.0f);
            seg.push_back(l.segs[k]);
            inc.push_back(k + 1 == l.segs.size() ? 1 : 0);
        }
    f.at("permanence")    = perm;
    f.at("mismatch")      = mism;
    f.at("adapt_history") = hist;
    f.at("segment_idx")   = seg;
    f.at("input_inc")     = inc;
}

mem_group saved(const ngm2::neuron_group_t& g)
{
    mem_state_writer w;
    g.save_state(w);
    return w.take_root();
}

void load(ngm2::neuron_group_t& g, const mem_group& tree)
{
    mem_state_reader r(tree, logging::k_format_version);
    g.load_state(r);
}

std::size_t count_seg(const ngm2::dendrite_t& d, std::uint32_t s)
{
    const auto& v = d.get_synapses().segment_idx;
    return static_cast<std::size_t>(std::ranges::count(v, s));
}

} // namespace

TEST_CASE("depth 4 is accepted, depth 5 is rejected")
{
    CHECK_NOTHROW(ngm2::neuron_group_t(make_store(4), "g", k_id));
    CHECK_THROWS_AS(ngm2::neuron_group_t(make_store(5), "g", k_id), std::runtime_error);
    CHECK_THROWS_AS(ngm2::neuron_group_t(make_store(15), "g", k_id), std::runtime_error);
    ngm2::cfg_store cs = make_store(k_depth);
    CHECK(ngm2::dendrite_t(cs, "prox").get_max_segment_idx() == k_max_seg);
}

TEST_CASE("process() returns at the maximum depth")
{
    // Fresh dendrites: every synapse on the root, one representation each.
    rig r(4);
    r.run(8);
    for (std::size_t n = 0; n < 2; ++n) {
        CHECK(r.g->get_neuron(n).get_representation_count() == 2);
        for (std::size_t d = 0; d < 2; ++d)
            CHECK(r.dendrite(n, d).get_representation_count() == 1);
    }
}

TEST_CASE("leaf mask and representation count at the deepest segments")
{
    rig r(4);
    auto tree = saved(*r.g);
    // root, the first and the last deepest leaf, and a depth-3 segment that is
    // an ancestor of neither
    inject(tree, 0, 0, 4, { { 1, 9, k_leaf_begin, k_max_seg }, {} });
    load(*r.g, tree);

    const auto mask = r.dendrite(0, 0).get_leaf_mask();
    REQUIRE(mask.size() == k_max_seg + 1);
    std::vector<std::size_t> leaves;
    for (std::size_t s = 0; s < mask.size(); ++s)
        if (mask[s]) leaves.push_back(s);
    CHECK(leaves == std::vector<std::size_t>{ 9, k_leaf_begin, k_max_seg });
    CHECK(r.dendrite(0, 0).get_representation_count() == 3);
    CHECK(r.dendrite(0, 0).get_representation(2).size() == 4 * 2);  // root + 31 per input

    r.run(4);
    CHECK(r.dendrite(0, 0).get_representation_count() == 3);
}

TEST_CASE("branching grows depth 3 into depth 4 and never past it")
{
    // Per input: a root synapse (mismatch 0), one on depth-3 segment 8 and one
    // on the deepest leaf 31 (mismatch 1). With the thresholds at 0 both
    // high-mismatch synapses clear the gate; only 8 has children to move to.
    rig r(4, /*eager_branching=*/true);
    auto tree = saved(*r.g);
    inject(tree, 0, 0, 4, { { 1, 8, k_max_seg }, { 0.0f, 1.0f, 1.0f } });
    load(*r.g, tree);

    r.run(1);
    const auto& d = r.dendrite(0, 0);
    CHECK(count_seg(d, 8) == 0);
    CHECK(count_seg(d, 16) == 4);
    CHECK(count_seg(d, 17) == 4);
    CHECK(count_seg(d, k_max_seg) == 4);
    CHECK(std::ranges::all_of(d.get_synapses().segment_idx,
                              [](auto s) { return s >= 1 && s <= k_max_seg; }));
    CHECK(d.get_synapse_count() == 4 * 4);
    CHECK(d.get_representation_count() == 3);
}

TEST_CASE("learning grows a dendrite to the maximum depth and keeps processing there")
{
    // No injection: branching itself carries synapses down to the deepest level.
    rig r(8, /*eager_branching=*/true, /*fast_mismatch=*/true);
    auto deepest = [&] {
        std::uint32_t m = 0;
        for (std::size_t n = 0; n < 2; ++n)
            for (std::size_t d = 0; d < 2; ++d)
                for (const auto s : r.dendrite(n, d).get_synapses().segment_idx)
                    m = std::max<std::uint32_t>(m, s);
        return m;
    };
    std::size_t steps = 0;
    for (; steps < 200 && deepest() < k_leaf_begin; ++steps)
        r.run_random();
    MESSAGE("deepest level reached after " << steps << " steps");
    REQUIRE(deepest() >= k_leaf_begin);
    REQUIRE(deepest() <= k_max_seg);

    for (std::size_t s = 0; s < 5; ++s) r.run_random();
    for (std::size_t n = 0; n < 2; ++n) {
        std::size_t sum = 0;
        for (std::size_t d = 0; d < 2; ++d) {
            const auto c = r.dendrite(n, d).get_representation_count();
            CHECK(c >= 1);
            CHECK(c <= k_leaf_begin);   // at most 2^L leaves
            sum += c;
        }
        CHECK(r.g->get_neuron(n).get_representation_count() == sum);
    }
}
