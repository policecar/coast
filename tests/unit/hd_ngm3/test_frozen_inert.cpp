// tests/unit/hd_ngm3/test_frozen_inert.cpp
//
// Frozen (inference-only) contract at the model level: with params.frozen a
// neuron_group_t's persisted state must be bit-identical across compute steps,
// except for the RNG streams (neuron noise stays on — it is functionally
// load-bearing and not part of the learning state).
//
// The comparison is whole-state-tree equality minus the RNG groups, NOT a
// per-field whitelist: any future mechanism that mutates persisted state
// without a frozen gate fails this test the day it is added. The unfrozen
// negative control proves the equality check has teeth (mismatch smoothing,
// neuron_mx_act, and the adapt pass all move state under the same drive).
//
// Constructing a neuron_group_t needs only a cfg_store; init_storage() and
// compute_step() are exposed via a test subclass, so this stays a pure
// sub-second unit test with no IO wiring.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "logging/mem_state_io.h"

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace {

struct test_group_t : ngm3::neuron_group_t {
    using ngm3::neuron_group_t::neuron_group_t;
    using ngm3::neuron_group_t::init_storage;
    using ngm3::neuron_group_t::compute_step;
};

constexpr std::size_t k_input_size = 8;

ngm3::cfg_store make_store(bool frozen)
{
    ngm3::cfg_store cs;
    auto& g = cs.neuron_groups["g"];
    g.neuron_cnt            = 4;
    g.apical_cnt            = 0;
    g.proximal_cnt          = 1;
    g.max_branch_level = 2;
    g.branch_interval       = 10;   // give branching a chance within the run
    g.random_seed           = seed_t{ 7 };
    cs.frozen               = frozen;   // the run's, not the group's
    return cs;
}

test_group_t make_group(bool frozen)
{
    static const ngm3::cfg_store cs_frozen   = make_store(true);
    static const ngm3::cfg_store cs_learning = make_store(false);
    test_group_t g(frozen ? cs_frozen : cs_learning, "g", /*output_id=*/1);
    g.init_storage(/*apical_in_size=*/0, /*proximal_in_size=*/k_input_size);
    return g;
}

// Alternate two strong input patterns so the response/adapt machinery has
// real activity to chew on.
void drive(test_group_t& g, std::size_t steps)
{
    std::vector<float> a(k_input_size, 0.0f);
    std::vector<float> b(k_input_size, 0.0f);
    for (std::size_t i = 0; i < k_input_size; ++i) {
        a[i] = (i % 2 == 0) ? 0.95f : 0.1f;
        b[i] = (i % 2 == 0) ? 0.1f  : 0.95f;
    }
    for (std::size_t s = 0; s < steps; ++s) {
        // compute_step reports faults by value (nodiscard step_result);
        // tests bridge back to an exception per the io_entity.h contract.
        sim::throw_on_step_fault(
            g.compute_step({}, (s % 2 == 0) ? std::span<const float>(a)
                                            : std::span<const float>(b)),
            "ngm3 compute_step");
    }
}

logging::mem_group state_of(const test_group_t& g)
{
    logging::mem_state_writer w;
    g.save_state(w);
    return w.take_root();
}

// The RNG streams legitimately advance in frozen mode (neuron noise stays on);
// everything else in the persisted state must be bit-stable.
void scrub_rng(logging::mem_group& g)
{
    g.groups.erase("rng");
    g.groups.erase("neuron_rgens");
    for (auto& [name, child] : g.groups) {
        scrub_rng(child);
    }
}

} // namespace

TEST_CASE("frozen ngm3 group state is bit-inert across steps (rng excepted)")
{
    auto g = make_group(/*frozen=*/true);
    drive(g, 5);                         // warm up: real activity in flight

    auto before = state_of(g);
    drive(g, 100);                       // crosses branch_interval repeatedly
    auto after = state_of(g);

    // The forward pass genuinely ran (inertness is not vacuous).
    float mx = 0.0f;
    for (const float v : g.get_outp()) mx = std::max(mx, v);
    CHECK(mx > 0.0f);

    scrub_rng(before);
    scrub_rng(after);
    CHECK(before == after);
}

TEST_CASE("frozen is the run's: a group entry that sets it itself is refused")
{
    // A per-group value would be silently replaced by the run's (cfg_store::frozen); the
    // JSON key is refused by the loader, and an entry set in code by group_or_default.
    ngm3::cfg_store cs = make_store(false);
    cs.neuron_groups["g"].frozen = true;
    CHECK_THROWS_WITH_AS((void)cs.group_or_default("g"),
                         doctest::Contains("frozen is set per group"), std::logic_error);
    cs.neuron_groups["g"].frozen = false;
    cs.frozen = true;
    CHECK(cs.group_or_default("g").frozen);
    CHECK(cs.group_or_default("not configured").frozen);   // the defaults get it too
}

TEST_CASE("load_state keeps the configured learning rate regardless of frozen")
{
    // M.7.8: the configuration wins on load, frozen or not. The checkpoint below
    // claims a training-time rate of 0.5 twice over — in its configuration record
    // (as if saved under another configuration) and as the top-level field older
    // checkpoints carried — and the group keeps its configured rate either way.
    // Frozen isolation is the step-time early-return in compute_step, not a load
    // special case; this guards against re-introducing either a frozen load gate
    // or the restore of checkpointed tunables.
    auto make_and_load = [](bool frozen) {
        auto g = make_group(frozen);
        const float configured = g.get_common_learning_rate();
        drive(g, 3);
        logging::mem_state_writer w;
        g.save_state(w);
        auto tree = w.take_root();
        REQUIRE(tree.groups.contains("config"));
        tree.groups["config"].fields["common_learning_rate"] = 0.5f;
        tree.fields["common_learning_rate"] = 0.5f;
        logging::mem_state_reader r(tree, logging::k_format_version);
        g.load_state(r);
        return std::pair{ configured, g.get_common_learning_rate() };
    };

    for (const bool frozen : { true, false }) {
        const auto [configured, loaded] = make_and_load(frozen);
        CHECK(configured != 0.5f);
        CHECK_MESSAGE(loaded == configured, "frozen=" << frozen);
    }
}

TEST_CASE("negative control: unfrozen ngm3 group state moves under the same drive")
{
    auto g = make_group(/*frozen=*/false);
    drive(g, 5);

    auto before = state_of(g);
    drive(g, 100);
    auto after = state_of(g);

    scrub_rng(before);
    scrub_rng(after);
    CHECK_FALSE(before == after);        // mx_act/mismatch/adapt must move
}
