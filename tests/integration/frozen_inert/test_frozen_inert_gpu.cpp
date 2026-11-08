// tests/integration/frozen_inert/test_frozen_inert_gpu.cpp
//
// Full-path frozen (inference-only) contract, ngm3 GPU group: with frozen=true
// the winner/adapt/branch dispatches are never emitted into the step plan and
// the respcmb kernel's mx_act update is gated by the frozen uniform, so the
// group's persisted state (readback in save_state) must be bit-identical
// across steps. The only exclusion is the host `step` counter, which keys the
// stateless per-step noise draws (the device analogue of the CPU RNG streams)
// and must keep advancing.
//
// Registered only when COAST_GPU_BACKEND=webgpu; the _gpu test-name suffix
// keeps it out of sandboxed runs (no adapter there).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "frozen_inert_helpers.h"

#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#include "probe/probe_instance.h"

#include <array>
#include <cstddef>

using frozen_inert::freeze_ngm3_group;
using frozen_inert::njson;
using frozen_inert::state_inert_over;
using frozen_inert::write_patched;

TEST_CASE("frozen ngm3 GPU model state is bit-inert through the full sim path")
{
    const auto cfg = write_patched("configs/mnist_ngm3_gpu_full.json",
                                   "ngm3_gpu_frozen.json", freeze_ngm3_group);
    CHECK(state_inert_over<ngm3::neuron_group_gpu_t>(cfg, 300, /*scrub_step=*/true));
}

TEST_CASE("negative control: unfrozen ngm3 GPU model state moves")
{
    const auto cfg = write_patched("configs/mnist_ngm3_gpu_full.json",
                                   "ngm3_gpu_learning.json", [](njson&) {});
    CHECK_FALSE(state_inert_over<ngm3::neuron_group_gpu_t>(cfg, 300, /*scrub_step=*/true));
}

// A frozen group skips its learning dispatches, but its activity window edges
// stay valid state: the edge probes must publish them (as checkpointed), not
// zeros.
template <typename Group>
void check_frozen_edge_probes(const frozen_inert::fs::path& cfg)
{
    coast::cli::options opts;
    opts.config_path = cfg.string();
    hdf5::executor h5;
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);
    auto vec = sim->get_sim_env().template get_entities<Group>();
    REQUIRE(vec.has_value());
    REQUIRE((*vec)->size() == 1);
    Group& g = (**vec)[0];

    constexpr std::array<const char*, 4> probes{ "window winner lower", "window winner upper",
                                                 "window other lower",  "window other upper" };
    constexpr std::array<const char*, 4> fields{ "winner_lower", "winner_upper",
                                                 "other_lower",  "other_upper" };
    std::array<probe::probe_instance*, 4> p{};
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = g.probes().activate(&g, probes[i]);

    coast::test::run_for(*sim, 20);

    const auto state = frozen_inert::state_of(g);
    const auto& edges = state.groups.at("activity_window_tracking").fields;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const float expected = std::get<float>(edges.at(fields[i]));
        INFO("probe: ", probes[i], ", checkpointed edge: ", expected);
        CHECK(expected > 0.0f);
        std::size_t n = 0;
        probe::probe_sample s{};
        while (p[i]->samples.try_pop(s)) {
            ++n;
            CHECK(s.value == static_cast<double>(expected));
        }
        CHECK(n == 20);
    }
}

TEST_CASE("frozen ngm3 GPU group publishes its window edges")
{
    const auto cfg = write_patched("configs/mnist_ngm3_gpu_full.json",
                                   "ngm3_gpu_frozen_probes.json", freeze_ngm3_group);
    check_frozen_edge_probes<ngm3::neuron_group_gpu_t>(cfg);
}

TEST_CASE("frozen ngm25 GPU group publishes its window edges")
{
    const auto cfg = write_patched("configs/mnist_ngm25_gpu.json",
                                   "ngm25_gpu_frozen_probes.json", [](njson& j) {
        j["frozen"] = true;
    });
    check_frozen_edge_probes<ngm25::neuron_group_gpu_t>(cfg);
}
