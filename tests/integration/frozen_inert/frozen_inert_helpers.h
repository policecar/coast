// tests/integration/frozen_inert/frozen_inert_helpers.h
//
// Shared machinery for the CPU and GPU frozen-inertness integration tests:
// patch a base config, build a real sim via coast::app::build_env, snapshot
// the single model group's persisted state through logging::mem_state_writer,
// and compare across steps. The comparison is whole-state-tree equality minus
// an explicit exclusion set (RNG streams; optionally the GPU step counter) --
// NOT a per-field whitelist, so future ungated mutations fail immediately.

#ifndef COAST_TEST_FROZEN_INERT_HELPERS_H
#define COAST_TEST_FROZEN_INERT_HELPERS_H

#include "doctest.h"

#include "integration_helpers.h"

#include "app/build_env.h"
#include "cli/options.h"
#include "hdf5/executor.h"
#include "logging/mem_state_io.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>

namespace frozen_inert {

namespace fs = std::filesystem;
using njson = nlohmann::json;

inline fs::path write_patched(const char* base_config, const char* name,
                              const std::function<void(njson&)>& patch)
{
    njson j;
    {
        std::ifstream in(base_config);
        REQUIRE(in.good());
        in >> j;
    }
    patch(j);

    // One directory per process: two builds running the suite at once must not
    // overwrite each other's config, which can differ (it may embed a scratch path).
    static const coast::test::scratch_dir_t dir("coast_frozen_inert");
    const fs::path path = dir.path() / name;
    std::ofstream out(path);
    REQUIRE(out.good());
    out << j.dump(2);
    return path;
}

// Both ngm3 mnist configs (CPU and GPU-full) name their group "L4-mnist-ngm3".
inline void freeze_ngm3_group(njson& j)
{
    j["frozen"] = true;   // the run's (SPEC S.5.14)
    auto& g = j["ngm3"]["neuron_groups"]["L4-mnist-ngm3"];
    g["common_learning_rate"] = 0.0;
    g["primary_learning_rate"] = 0.0;
    g["secondary_learning_rate"] = 0.0;
}

template <typename Entity>
logging::mem_group state_of(const Entity& e)
{
    logging::mem_state_writer w;
    e.save_state(w);
    return w.take_root();
}

// The RNG streams legitimately advance in frozen mode (neuron noise stays
// on); everything else in the persisted state must be bit-stable.
inline void scrub_rng(logging::mem_group& g)
{
    g.groups.erase("rng");
    g.groups.erase("neuron_rgens");
    for (auto& [name, child] : g.groups) {
        scrub_rng(child);
    }
}

// Build the sim, warm up, snapshot the single model group, run `steps`,
// snapshot again; return whether the scrubbed states compare equal.
// `scrub_step` additionally drops the "step" field: the GPU group's step
// counter keys its stateless noise draws (the device analogue of the CPU RNG
// streams) and must keep advancing in frozen mode.
template <typename Group>
bool state_inert_over(const fs::path& cfg, std::uint64_t steps,
                      bool scrub_step = false)
{
    coast::cli::options opts;
    opts.config_path = cfg.string();

    hdf5::executor h5;
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);

    auto vec = sim->get_sim_env().template get_entities<Group>();
    REQUIRE(vec.has_value());
    REQUIRE((*vec)->size() == 1);
    const Group& group = (**vec)[0];

    coast::test::run_for(*sim, 20);      // warm up: activity + running maxima
    auto before = state_of(group);
    coast::test::run_for(*sim, steps);
    auto after = state_of(group);

    scrub_rng(before);
    scrub_rng(after);
    if (scrub_step) {
        before.fields.erase("step");
        after.fields.erase("step");
    }
    return before == after;
}

} // namespace frozen_inert

#endif // COAST_TEST_FROZEN_INERT_HELPERS_H
