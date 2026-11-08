// tests/integration/frozen_inert/test_frozen_inert.cpp
//
// Full-path frozen (inference-only) contract, CPU models: build a real sim via
// coast::app::build_env from a frozen-patched config, run it, and require the
// model entity's persisted state to be bit-identical across steps except for
// its RNG streams. This covers the whole propagation chain — JSON field ->
// cfg_store -> group params -> the group's early return — that the
// direct unit test (tests/unit/hd_ngm3/test_frozen_inert.cpp) bypasses.
//
// Self-comparing (state(t) == state(t+N)), so no goldens. The unfrozen
// negative controls prove the equality check has teeth. GPU variant:
// test_frozen_inert_gpu.cpp (needs a real adapter).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "frozen_inert_helpers.h"
#include "integration_helpers.h"

#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"

using frozen_inert::freeze_ngm3_group;
using frozen_inert::njson;
using frozen_inert::state_inert_over;
using frozen_inert::write_patched;

namespace {

void freeze_ngm2(njson& j)
{
    j["frozen"] = true;   // the run's (SPEC S.5.14)
    j["ngm2"]["neuron_groups"]["L4-mnist"]["default_common_learning_rate"] = 0.0;
    j["ngm2"]["dendrites"]["default"]["default_primary_learning_rate"] = 0.0;
    j["ngm2"]["dendrites"]["default"]["default_secondary_learning_rate"] = 0.0;
}

// Capture both population groups on every step of the run, so the observation
// pass reads the group's state each step while it is checked for inertness.
void enable_population_observe(njson& j, const std::filesystem::path& dir)
{
    const njson every_step = {{"ranges", {{0, 1000}}}, {"combine", "or"}};
    j["observe"] = {{"output_dir", dir.string()},
        {"groups", {{"population", every_step}, {"population_structure", every_step}}}};
}

} // namespace

TEST_CASE("frozen ngm2 model state is bit-inert through the full sim path")
{
    const auto cfg = write_patched("configs/mnist.json", "ngm2_frozen.json",
                                   freeze_ngm2);
    CHECK(state_inert_over<ngm2::neuron_group_t>(cfg, 300));
}

TEST_CASE("negative control: unfrozen ngm2 model state moves")
{
    const auto cfg = write_patched("configs/mnist.json", "ngm2_learning.json",
                                   [](njson&) {});
    CHECK_FALSE(state_inert_over<ngm2::neuron_group_t>(cfg, 300));
}

TEST_CASE("frozen ngm3 model state is bit-inert through the full sim path")
{
    const auto cfg = write_patched("configs/mnist_ngm3.json", "ngm3_frozen.json",
                                   freeze_ngm3_group);
    CHECK(state_inert_over<ngm3::neuron_group_t>(cfg, 300));
}

TEST_CASE("negative control: unfrozen ngm3 model state moves")
{
    const auto cfg = write_patched("configs/mnist_ngm3.json", "ngm3_learning.json",
                                   [](njson&) {});
    CHECK_FALSE(state_inert_over<ngm3::neuron_group_t>(cfg, 300));
}

TEST_CASE("frozen ngm3 model state is bit-inert with population observations enabled")
{
    const coast::test::scratch_dir_t observe_dir("coast_frozen_inert_ngm3_observe");
    const auto cfg = write_patched("configs/mnist_ngm3.json", "ngm3_frozen_observe.json",
        [&](njson& j) {
            freeze_ngm3_group(j);
            enable_population_observe(j, observe_dir.path());
        });
    CHECK(state_inert_over<ngm3::neuron_group_t>(cfg, 300));
    CHECK(std::filesystem::exists(observe_dir.path() / "population.observe.h5"));   // the pass ran
    CHECK(std::filesystem::exists(observe_dir.path() / "population_structure.observe.h5"));
}

TEST_CASE("frozen ngm25 model state is inert with population observations enabled")
{
    const coast::test::scratch_dir_t observe_dir("coast_frozen_inert_ngm25_observe");
    const auto cfg = write_patched("configs/mnist_ngm25.json", "ngm25_frozen.json",
        [&](njson& j) {
            j["frozen"] = true;
            enable_population_observe(j, observe_dir.path());
        });
    // CPU ngm25 also keys stateless noise on step; learning state must remain exact.
    CHECK(state_inert_over<ngm25::neuron_group_t>(cfg, 300, /*scrub_step=*/true));
    CHECK(std::filesystem::exists(observe_dir.path() / "population.observe.h5"));   // the pass ran
    CHECK(std::filesystem::exists(observe_dir.path() / "population_structure.observe.h5"));
}

TEST_CASE("negative control: unfrozen ngm25 model state moves")
{
    const auto cfg = write_patched("configs/mnist_ngm25.json", "ngm25_learning.json", [](njson&) {});
    CHECK_FALSE(state_inert_over<ngm25::neuron_group_t>(cfg, 300, /*scrub_step=*/true));
}
