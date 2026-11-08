// tests/integration_ngm3/branch_timing_test.cpp
//
// Branch-*timing* test for ngm3 (plan item 1.7). Complements — does not
// duplicate — the synthetic smoke test, which already asserts the permanence /
// segment-id / output invariants and that branching *eventually* occurs by
// observing the end state at 20000 steps.
//
// What the smoke test CANNOT see, because it only inspects the final state, is
// the schedule's lower bound: a bug that fires a branch far too early (e.g. at
// step 1 due to an off-by-one in the interval counter) would still leave a
// grown tree at step 20000 and pass the smoke test. This test pins that lower
// bound: with branch_interval = B, the gate `input_cnt % B == 0` is
// mathematically unsatisfiable for input_cnt in [1, B-1], so NO branch can
// occur before step B — regardless of mismatch dynamics. We then confirm the
// upper region (branching has occurred after several intervals) so the test
// brackets the schedule from both sides in one run.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstddef>
#include <cstdint>

#include "app/build_env.h"
#include "cli/options.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "hdf5/executor.h"
#include "integration_helpers.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"

namespace {

constexpr sim::output_id_t k_ngm3_id = 1;

// Must match "branch_interval" in configs/test_ngm3_synthetic.json. The
// lower-bound assertion below is only meaningful while this holds.
constexpr std::uint64_t k_branch_interval = 5000;

// Total steps for the upper-region check. The smoke test establishes that this
// config branches within 20000 steps (4 intervals); we reuse that measured fact
// rather than guess a smaller bound.
constexpr std::uint64_t k_steps_total = 20000;

ngm3::neuron_group_t& require_group(coast::sim_instance& sim, sim::output_id_t id) {
    auto* ent = sim.get_sim_env().get_entity(id);
    REQUIRE(ent != nullptr);
    auto* ng = dynamic_cast<ngm3::neuron_group_t*>(ent);
    REQUIRE(ng != nullptr);
    return *ng;
}

} // namespace

TEST_CASE("ngm3 branch timing: no branch before branch_interval, branch after") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = seed_offset_t{ 0 };

    hdf5::executor h5;
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);

    auto& g = require_group(*sim, k_ngm3_id);

    // Baseline: each dendrite of both banks starts with a single seg_id (=1),
    // so the total representation count is exactly n*(apical + proximal).
    const std::size_t n = g.get_neuron_count();
    const std::size_t initial_total =
        n * (g.get_apical_count() + g.get_proximal_count());
    REQUIRE(g.get_total_representation_count() == initial_total);

    // --- lower bound: strictly inside the first interval ---------------------
    // Run to one step short of the first interval boundary. No branch gate can
    // have opened, so the tree must be untouched.
    coast::test::run_for(*sim, k_branch_interval - 1);
    CHECK(g.get_total_representation_count() == initial_total);

    // --- upper region: after several intervals -------------------------------
    coast::test::run_for(*sim, k_steps_total - (k_branch_interval - 1));
    CHECK(g.get_total_representation_count() > initial_total);
}
