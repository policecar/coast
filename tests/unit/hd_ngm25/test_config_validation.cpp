// tests/unit/hd_ngm25/test_config_validation.cpp
//
// Config-validation guards in neuron_group_t::init_io. The flat config schema
// sets the structural dendrite counts (apical_cnt/proximal_cnt) and the input
// routing (per-route bank tag) independently, so an internally-inconsistent
// config is easy to write. These guards turn a silently-degenerate group into an
// explicit throw. The apical case is the one both branch reviewers flagged: an
// apical_cnt > 0 group with no routed apical channel builds dead-empty apical
// dendrites and — because the "no apical bank" type-max sentinel only fires when
// there are ZERO apical dendrites — silently zeroes the whole group forever.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <memory>
#include <stdexcept>
#include <string_view>

#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "sim_core/io_entity.h"

using namespace ngm25;

namespace {

// Build a group from `p` and drive init_io with the given routing map. Returns
// nothing — the tests assert on whether construction/init_io throws.
void build_and_init(const neuron_group_params_t &p,
                    const std::unordered_map<std::string, sim::output_channel_t> &routes)
{
    cfg_store cs;
    cs.neuron_groups["g"] = p;
    auto grp = std::make_unique<neuron_group_t>(cs, "g", /*output_id=*/1, seed_offset_t{ 0 });
    grp->init_io(routes);
}

// True iff build_and_init throws a runtime_error whose message CONTAINS `needle`.
// Checking the message (not just the type) pins WHICH guard fired: every guard
// throws std::runtime_error, so a type-only check would pass even if an unrelated
// earlier guard tripped on the same mutated config.
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
    p.neuron_cnt       = 4;
    p.proximal_cnt     = 2;
    p.max_branch_level = 2;
    p.random_seed      = seed_t{ 1 };
    return p;
}

} // namespace

TEST_CASE("config: apical_cnt > 0 with no routed apical channel is rejected") {
    neuron_group_params_t p = base();
    p.apical_cnt = 2;
    // Only a proximal route — the apical bank gets zero width. Without the guard
    // this would build a silently-dead group; with it, init_io throws.
    p.inputs = { { "img", bank_t::proximal } };
    CHECK(throws_with(p, {{ "img", sim::output_channel_t{0, 16} }}, "apical"));
}

TEST_CASE("config: apical_cnt > 0 WITH a routed apical channel is accepted") {
    neuron_group_params_t p = base();
    p.apical_cnt = 2;
    p.inputs = { { "top", bank_t::apical }, { "img", bank_t::proximal } };
    CHECK_NOTHROW(build_and_init(p, {{ "top", sim::output_channel_t{0, 8} },
                                     { "img", sim::output_channel_t{1, 16} }}));
}

TEST_CASE("config: apical_cnt == 0 (no apical bank) is accepted — sentinel path") {
    neuron_group_params_t p = base();
    p.apical_cnt = 0;
    p.inputs = { { "img", bank_t::proximal } };
    CHECK_NOTHROW(build_and_init(p, {{ "img", sim::output_channel_t{0, 16} }}));
}

TEST_CASE("config: no routed proximal channel is rejected (existing guard)") {
    neuron_group_params_t p = base();
    p.apical_cnt = 1;
    // Everything routed to apical, nothing to proximal.
    p.inputs = { { "top", bank_t::apical } };
    CHECK(throws_with(p, {{ "top", sim::output_channel_t{0, 8} }}, "proximal"));
}

// permanence_threshold feeds inv_perm_range = 1/(1-thr) (response attenuation) and
// the Poisson cold-init mean = int(100*thr). thr>=1 makes the former +inf/negative
// (NaN group); thr<0.01 truncates the Poisson mean to <=0 (UB per [rand.dist.pois]).
// The ctor guard rejects both ends, and NaN/inf, before any degenerate arena builds.
TEST_CASE("config: permanence_threshold >= 1.0 is rejected (NaN attenuation)") {
    neuron_group_params_t p = base();
    p.inputs = { { "img", bank_t::proximal } };
    p.permanence_threshold = 1.0f;
    CHECK(throws_with(p, {{ "img", sim::output_channel_t{0, 16} }}, "permanence_threshold"));
}

TEST_CASE("config: permanence_threshold < 0.01 is rejected (zero Poisson mean)") {
    neuron_group_params_t p = base();
    p.inputs = { { "img", bank_t::proximal } };
    p.permanence_threshold = 0.0f;   // int(100*0.0) == 0 -> poisson mean 0 -> UB
    CHECK(throws_with(p, {{ "img", sim::output_channel_t{0, 16} }}, "permanence_threshold"));
}

TEST_CASE("config: neuron_cnt < 2 is rejected (a group needs two neurons to compete)") {
    for (std::size_t n : { std::size_t{0}, std::size_t{1} }) {
        CAPTURE(n);
        neuron_group_params_t p = base();
        p.neuron_cnt = n;
        p.inputs = { { "img", bank_t::proximal } };
        // The ctor guard throws before init_io — routes are unused but must be supplied.
        CHECK(throws_with(p, {{ "img", sim::output_channel_t{0, 16} }}, "neuron_cnt must be >= 2"));
    }
}

TEST_CASE("config: neuron_cnt past the u16 dend_neuron id space is rejected") {
    neuron_group_params_t p = base();
    p.neuron_cnt = 70000;            // > 65536 => a neuron id would not fit u16
    p.inputs = { { "img", bank_t::proximal } };
    // Guard is at the top of the ctor, so this throws before any large allocation.
    CHECK(throws_with(p, {{ "img", sim::output_channel_t{0, 16} }}, "u16 dend_neuron id space"));
}

TEST_CASE("config: a route with an empty channel name is rejected") {
    // An empty name would otherwise reach the channel record (SPEC.md M.7.5)
    // unchecked and only fail there, on the first save after this step -- and
    // then on every later --checkpoint snapshot too (config_record.h's
    // write_channel_record throws on an empty name). Caught here instead, at
    // configuration time, for both banks.
    neuron_group_params_t p = base();
    p.apical_cnt = 1;
    p.inputs = { { "", bank_t::proximal } };
    CHECK(throws_with(p, {{ "", sim::output_channel_t{0, 16} }}, "empty channel name"));

    p.inputs = { { "img", bank_t::proximal }, { "", bank_t::apical } };
    CHECK(throws_with(p, {{ "img", sim::output_channel_t{0, 16} }, { "", sim::output_channel_t{1, 8} }},
                      "empty channel name"));
}

TEST_CASE("config: an in-range permanence_threshold is accepted") {
    neuron_group_params_t p = base();
    p.inputs = { { "img", bank_t::proximal } };
    p.permanence_threshold = 0.3f;   // the production default
    CHECK_NOTHROW(build_and_init(p, {{ "img", sim::output_channel_t{0, 16} }}));
}
