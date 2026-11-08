// tests/unit/hd_ngm2/test_window_tracking.cpp
//
// Activity window tracking (ngm/window_tracking.h), shared by ngm2, ngm25
// and ngm3: the update rule on hand-built activities, the generated WGSL
// prelude, and — through an ngm2 group — the starting windows and the edges'
// checkpoint round-trip. Whether the three implementations apply it alike is
// covered by integration.impl_equivalence. No dataset or IO buffer is needed:
// save/load run through the in-memory state IO and do not touch the input.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "ngm/window_tracking.h"
#include "logging/mem_state_io.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace wt = ngm::window_tracking;
using logging::mem_group;
using logging::mem_state_reader;
using logging::mem_state_writer;

namespace {

constexpr sim::output_id_t k_id    = 0;
constexpr sim::output_id_t k_in_id = 1;

std::unique_ptr<ngm2::neuron_group_t> make_group()
{
    ngm2::cfg_store cs;
    ngm2::dendrite_params_t dp;
    dp.input_channels = { "da" };
    cs.dendrites["prox"] = dp;

    ngm2::neuron_params_t np;
    np.input_channels   = { "na" };
    np.dendrites        = { { "p", "prox", 2 } };
    np.channel_mappings = { { "na", "p", "da" } };
    cs.neurons["n"] = np;

    ngm2::neuron_group_params_t gp;
    gp.input_channels   = { "ga" };
    gp.neurons          = { { "pyr", "n", 3 } };
    gp.channel_mappings = { { "ga", "pyr", "na" } };
    cs.neuron_groups["g"] = gp;

    auto g = std::make_unique<ngm2::neuron_group_t>(cs, "g", k_id);
    g->init_io({ { "ga", sim::output_channel_t{ k_in_id, 8 } } });
    return g;
}

mem_group saved(const ngm2::neuron_group_t& g)
{
    mem_state_writer w;
    g.save_state(w);
    return w.take_root();
}

std::string load_error(ngm2::neuron_group_t& g, const mem_group& tree,
                       std::uint32_t version = logging::k_format_version)
{
    try {
        mem_state_reader r(tree, version);
        g.load_state(r);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

bool contains(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }

} // namespace

TEST_CASE("an edge step moves by rate * (tau - share below), clamped to [0, 1]")
{
    CHECK(wt::step(0.5f, 0.25f, 1.0f) == 0.5f + wt::rate * (0.25f - 1.0f));
    CHECK(wt::step(0.5f, 0.25f, 0.0f) == 0.5f + wt::rate * 0.25f);
    CHECK(wt::step(0.0f, 0.25f, 1.0f) == 0.0f);
    CHECK(wt::step(1.0f, 0.75f, 0.0f) == 1.0f);
}

TEST_CASE("the winner's edges see the winner only, the others' edges everyone else")
{
    // neuron 1 wins with 0.9: above both winner edges -> they rise by rate * tau.
    // Others: 0.1 (below both other edges), 0.5 (between), 0.7 (above both).
    const std::vector<float> act = { 0.1f, 0.9f, 0.5f, 0.7f };
    wt::edges_t e;
    wt::update(e, act, 1);
    CHECK(e.winner_lower == wt::step(wt::initial_lower, wt::quantile_lower, 0.0f));
    CHECK(e.winner_upper == wt::step(wt::initial_upper, wt::quantile_upper, 0.0f));
    CHECK(e.other_lower  == wt::step(wt::initial_lower, wt::quantile_lower, 1.0f / 3.0f));
    CHECK(e.other_upper  == wt::step(wt::initial_upper, wt::quantile_upper, 2.0f / 3.0f));
}

TEST_CASE("without a winner only the others' edges move, over all neurons")
{
    const std::vector<float> act = { 0.1f, 0.9f };
    wt::edges_t e;
    wt::update(e, act, act.size());
    CHECK(e.winner_lower == wt::initial_lower);
    CHECK(e.winner_upper == wt::initial_upper);
    CHECK(e.other_lower  == wt::step(wt::initial_lower, wt::quantile_lower, 0.5f));
    CHECK(e.other_upper  == wt::step(wt::initial_upper, wt::quantile_upper, 0.5f));
}

TEST_CASE("the edges settle at the configured quantiles of a stationary population")
{
    // Uniform activities on [0.2, 0.8]: the tau-quantile is 0.2 + 0.6 * tau. A
    // fixed LCG (not std:: distributions, which differ between standard
    // libraries) and the mean over the last 10k steps keep the tolerance tight
    // enough that an edge which never moves fails: winner_lower must travel
    // 0.016 from its start, the others further.
    std::uint32_t x = 12345u;
    auto uniform = [&x] { x = x * 1664525u + 1013904223u; return 0.2f + 0.6f * static_cast<float>(x >> 8) / 16777216.0f; };
    std::vector<float> act(50);
    wt::edges_t e;
    double wl = 0, wu = 0, ol = 0, ou = 0;
    constexpr int steps = 60000, tail = 10000;
    for (int s = 0; s < steps; ++s) {
        for (auto& a : act) a = uniform();
        wt::update(e, act, static_cast<std::size_t>(s % 50));
        if (s >= steps - tail) { wl += e.winner_lower; wu += e.winner_upper; ol += e.other_lower; ou += e.other_upper; }
    }
    const double lo = 0.2 + 0.6 * wt::quantile_lower, hi = 0.2 + 0.6 * wt::quantile_upper;
    CHECK(std::abs(wl / tail - lo) < 0.005);
    CHECK(std::abs(wu / tail - hi) < 0.005);
    CHECK(std::abs(ol / tail - lo) < 0.002);
    CHECK(std::abs(ou / tail - hi) < 0.002);
}

TEST_CASE("the WGSL prelude carries the C++ constants and the edge step")
{
    const std::string s = wt::window_tracking_wgsl();
    CHECK(contains(s, "const WT_RATE: f32 = 0.00100000005;"));
    CHECK(contains(s, "const WT_Q_LOWER: f32 = 0.189999998;"));
    CHECK(contains(s, "const WT_Q_UPPER: f32 = 0.939999998;"));
    CHECK(contains(s, "const WT_STEEP_LOWER: f32 = 0.569999993;"));
    CHECK(contains(s, "const WT_STEEP_UPPER: f32 = 0.779999971;"));
    CHECK(contains(s, "fn wt_step(q: f32, tau: f32, share_below: f32) -> f32"));
}

TEST_CASE("both GPU shaders take the tracker constants from the prelude, not literals")
{
    // Runs from the source root (WORKING_DIRECTORY). A shader that hard-coded a
    // quantile or steepness would pass every test until the constants change.
    for (const char* path : { "src/hd_ngm3/hd_ngm3_gpu.wgsl", "src/hd_ngm25/hd_ngm25_gpu.wgsl" }) {
        CAPTURE(path);
        std::ifstream f(path);
        REQUIRE(f.is_open());
        const std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        CHECK(contains(src, "WT_Q_LOWER"));
        CHECK(contains(src, "WT_Q_UPPER"));
        CHECK(contains(src, "wt_step("));
        for (const char* literal : { "0.19", "0.94", "0.57", "0.78", "0.001" })
            CHECK_MESSAGE(!contains(src, literal), "hard-coded tracker constant " << literal);
    }
}

TEST_CASE("a group starts both windows at the initial edges with the fixed steepness")
{
    const auto g = make_group();
    const auto [winner, other] = g->get_tracked_windows();
    for (const auto& w : { winner, other }) {
        CHECK(w.first.transition_point  == wt::initial_lower);
        CHECK(w.second.transition_point == wt::initial_upper);
        CHECK(w.first.steepness         == wt::steep_lower);
        CHECK(w.second.steepness        == wt::steep_upper);
    }
}

TEST_CASE("window edges round-trip through a checkpoint")
{
    auto tree = saved(*make_group());
    auto& f = tree.groups.at("activity_window_tracking").fields;
    f.at("winner_lower") = 0.51f;
    f.at("winner_upper") = 0.77f;
    f.at("other_lower")  = 0.18f;
    f.at("other_upper")  = 0.62f;

    auto g = make_group();
    REQUIRE(load_error(*g, tree).empty());
    const auto [winner, other] = g->get_tracked_windows();
    CHECK(winner.first.transition_point  == 0.51f);
    CHECK(winner.second.transition_point == 0.77f);
    CHECK(other.first.transition_point   == 0.18f);
    CHECK(other.second.transition_point  == 0.62f);

    // ... and save() writes the edges the group holds, not the initial ones.
    const auto again = saved(*g);
    const auto& f2 = again.groups.at("activity_window_tracking").fields;
    CHECK(std::get<float>(f2.at("winner_lower")) == 0.51f);
    CHECK(std::get<float>(f2.at("winner_upper")) == 0.77f);
    CHECK(std::get<float>(f2.at("other_lower"))  == 0.18f);
    CHECK(std::get<float>(f2.at("other_upper"))  == 0.62f);
}

TEST_CASE("a window edge outside [0, 1] in a checkpoint is refused")
{
    auto tree = saved(*make_group());
    tree.groups.at("activity_window_tracking").fields.at("other_upper") = 1.5f;

    auto g = make_group();
    CHECK(contains(load_error(*g, tree), "activity_window_tracking.other_upper = 1.5 is outside [0, 1]"));
}

TEST_CASE("a format-2 checkpoint without window edges is corrupt")
{
    auto tree = saved(*make_group());
    tree.groups.erase("activity_window_tracking");

    auto g = make_group();
    CHECK(contains(load_error(*g, tree, 2), "corrupt checkpoint — activity_window_tracking missing"));
}

TEST_CASE("a window-edge field missing from a checkpoint is corrupt, not filled in")
{
    auto tree = saved(*make_group());
    tree.groups.at("activity_window_tracking").fields.erase("other_lower");

    auto g = make_group();
    CHECK(contains(load_error(*g, tree), "corrupt checkpoint — activity_window_tracking.other_lower missing"));
}

TEST_CASE("a format-1 checkpoint (before tracking) starts the edges over, whatever they held")
{
    auto g = make_group();
    {   // move the group's edges away from the initial ones first
        auto moved = saved(*g);
        auto& f = moved.groups.at("activity_window_tracking").fields;
        f.at("winner_lower") = 0.5f;
        f.at("other_upper")  = 0.9f;
        REQUIRE(load_error(*g, moved).empty());
        REQUIRE(g->get_tracked_windows().first.first.transition_point == 0.5f);
    }
    auto old = saved(*g);
    old.groups.erase("activity_window_tracking");
    REQUIRE(load_error(*g, old, 1).empty());
    const auto [winner, other] = g->get_tracked_windows();
    CHECK(winner.first.transition_point  == wt::initial_lower);
    CHECK(winner.second.transition_point == wt::initial_upper);
    CHECK(other.first.transition_point   == wt::initial_lower);
    CHECK(other.second.transition_point  == wt::initial_upper);
}
