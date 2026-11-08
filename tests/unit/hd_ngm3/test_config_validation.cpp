// tests/unit/hd_ngm3/test_config_validation.cpp
//
// Config-validation guard in neuron_group_t::init_io: a route with an empty
// channel name. An empty name would otherwise reach the channel record
// (SPEC.md M.7.5) unchecked and only fail there, on the first save after this
// step — and then on every later --checkpoint snapshot too
// (config_record.h's write_channel_record throws on an empty name, which is
// how the real-world case this guards against was found: a route left blank
// in a config made --save-state exit 1 and every subsequent --checkpoint
// snapshot fail the same way). Caught here instead, at configuration time.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <memory>
#include <string>
#include <stdexcept>
#include <string_view>

#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#ifdef COAST_WITH_GPU
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#endif
#include "sim_core/io_entity.h"

using namespace ngm3;

namespace {

void build_and_init(const neuron_group_params_t &p,
                    const std::unordered_map<std::string, sim::output_channel_t> &routes)
{
    cfg_store cs;
    cs.neuron_groups["g"] = p;
    auto grp = std::make_unique<neuron_group_t>(cs, "g", /*output_id=*/1);
    grp->init_io(routes);
}

[[nodiscard]] bool throws_with(
    const neuron_group_params_t &p,
    const std::unordered_map<std::string, sim::output_channel_t> &routes,
    std::string_view needle)
{
    try { build_and_init(p, routes); }
    catch (const std::runtime_error &e) {
        return std::string_view(e.what()).find(needle) != std::string_view::npos;
    }
    catch (...) { return false; }
    return false;
}

neuron_group_params_t base()
{
    neuron_group_params_t p;
    p.neuron_cnt            = 4;
    p.proximal_cnt          = 2;
    p.max_branch_level = 2;
    p.random_seed           = seed_t{ 1 };
    return p;
}

} // namespace

TEST_CASE("config: a proximal route with an empty channel name is rejected") {
    neuron_group_params_t p = base();
    p.inputs = { { "", bank_t::proximal } };
    CHECK(throws_with(p, {{ "", sim::output_channel_t{0, 16} }}, "empty channel name"));
}

TEST_CASE("config: an apical route with an empty channel name is rejected") {
    neuron_group_params_t p = base();
    p.apical_cnt = 1;
    p.inputs = { { "img", bank_t::proximal }, { "", bank_t::apical } };
    CHECK(throws_with(p, {{ "img", sim::output_channel_t{0, 16} }, { "", sim::output_channel_t{1, 8} }},
                      "empty channel name"));
}

TEST_CASE("config: a route with a non-empty channel name is accepted") {
    neuron_group_params_t p = base();
    p.inputs = { { "img", bank_t::proximal } };
    CHECK_NOTHROW(build_and_init(p, {{ "img", sim::output_channel_t{0, 16} }}));
}

// Issue #199: the CPU group feeds both values to std::uniform_real_distribution
// (a <= b precondition: a reversed noise range or a winning threshold above 1 was
// UB). The GPU group refuses the same configs so the two backends agree; its
// checks run in the constructor, before any device is touched, so no GPU is
// needed here.
TEST_CASE("config: a reversed noise range or a stochastic_win_thres outside [0, 1] is rejected, CPU and GPU") {
    const auto ctor_error = [](const neuron_group_params_t& p, bool gpu) {
        cfg_store cs;
        cs.neuron_groups["g"] = p;
        try {
            if (gpu) {
#ifdef COAST_WITH_GPU
                neuron_group_gpu_t g(cs, "g", /*output_id=*/1);
#endif
            } else {
                neuron_group_t g(cs, "g", /*output_id=*/1);
            }
        } catch (const std::runtime_error& e) { return std::string(e.what()); }
        return std::string();
    };
    bool backends[] = { false, true };
#ifndef COAST_WITH_GPU
    MESSAGE("GPU backend not built: only the CPU group is checked");
#endif
    for (const bool gpu : backends) {
#ifndef COAST_WITH_GPU
        if (gpu) continue;
#endif
        CAPTURE(gpu);
        neuron_group_params_t p = base();
        p.neuron_noise_range = { 0.05f, 0.01f };
        CHECK(ctor_error(p, gpu).find("neuron_noise_range") != std::string::npos);
        // both bounds finite, but the width overflows float (review #3)
        p.neuron_noise_range = { -3e38f, 3e38f };
        CHECK(ctor_error(p, gpu).find("neuron_noise_range") != std::string::npos);
        p.neuron_noise_range = { 0.0f, 0.0f };
        CHECK(ctor_error(p, gpu).empty());

        p = base();
        p.stochastic_win_thres = 1.2f;
        CHECK(ctor_error(p, gpu).find("stochastic_win_thres") != std::string::npos);
        p.stochastic_win_thres = -0.1f;
        CHECK(ctor_error(p, gpu).find("stochastic_win_thres") != std::string::npos);
        p.stochastic_win_thres = 1.0f;
        CHECK(ctor_error(p, gpu).empty());
    }
}
