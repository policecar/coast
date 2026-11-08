// tests/integration_ngm3/state_resume_stepafter_test.cpp
//
// Deterministic-resume regression: stepping AFTER a load must reproduce a
// straight-through run bit-for-bit. This is the property the entity-state
// persistence work exists to guarantee — the per-step carried state that
// save_state/load_state must restore: the fast_rng_t PRNG streams, plus the
// decaying running maxes (ngm2 neuron max_activity, ngm3 neuron_mx_act), the
// branch-interval counter (ngm2 input_count), the dendrite mismatch thresholds,
// and the dataset cursors. If any of these is dropped, a resumed run diverges.
//
// Protocol:
//   straight : build, run N+M steps, extract state tree S.
//   checkpoint: build, run N steps, extract tree C, serialize to HDF5.
//   resume   : fresh env, parse+apply the checkpoint, run M more steps, extract R.
//   property : R.entities == S.entities and R.io_buffers == S.io_buffers.
//   vacuity  : S.entities != C.entities (the M post-load steps changed state)
//              and S has non-zero io_buffers (comparison is not all-zero).
//
// Unlike state_roundtrip_split (no stepping → RNG-immune), this steps after the
// load and so fails if any per-step carried state is not restored. Valid on CPU;
// the GPU config variant (registered in CMake, "gpu"-labeled) needs a real device
// and so must run outside the Claude Code sandbox — see CLAUDE.md "GPU tests and
// the sandbox". The GPU ngm3 path is bit-deterministic run-to-run, so the exact
// comparison holds there too.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>
#include <unistd.h>

#include "app/build_env.h"
#include "cli/options.h"
#include "hdf5/executor.h"
#include "integration_helpers.h"
#include "logging/state_logger.h"
#include "logging/mem_state_io.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t k_checkpoint_steps = 50;   // N
constexpr std::uint64_t k_resume_steps     = 50;   // M
constexpr seed_offset_t k_seed_offset      = {};

// Guard against a vacuous all-zero comparison.
bool any_buffer_nonzero(const logging::mem_group& tree) {
    auto it = tree.groups.find("io_buffers");
    if (it == tree.groups.end()) return false;
    for (const auto& [id, g] : it->second.groups) {
        (void)id;
        auto fit = g.fields.find("data");
        if (fit == g.fields.end()) continue;
        if (const auto* v = std::get_if<std::vector<float>>(&fit->second))
            if (std::any_of(v->begin(), v->end(), [](float x){ return x != 0.0f; }))
                return true;
    }
    return false;
}

} // namespace

TEST_CASE("stepping after a load reproduces a straight-through run bit-for-bit") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;
    // Every env below is the same run, rebuilt: its checkpoints continue it, datasets
    // included (one run_id, as coastd fixes one per job; SPEC.md S.5.14).
    std::string run_id;
    const auto snap = fs::temp_directory_path() /
        ("coast_resume_stepafter_" + std::to_string(::getpid()) + ".h5");
    fs::remove(snap);

    // --- straight-through: N+M steps -----------------------------------------
    logging::mem_group straight_tree;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        run_id = sim->get_sim_env().run_id();
        coast::test::run_for(*sim, k_checkpoint_steps + k_resume_steps);
        straight_tree = logging::state_logger::extract_state(
            sim->get_sim_env(), k_checkpoint_steps + k_resume_steps);
    }
    REQUIRE_MESSAGE(any_buffer_nonzero(straight_tree),
                    "straight-through io_buffers all-zero; comparison would be vacuous");

    // --- checkpoint at N, serialize to file ----------------------------------
    logging::mem_group ckpt_tree;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        sim->get_sim_env().set_run_id(run_id);
        coast::test::run_for(*sim, k_checkpoint_steps);
        ckpt_tree = logging::state_logger::extract_state(sim->get_sim_env(), k_checkpoint_steps);

        logging::state_save_config cfg;
        cfg.output_path = snap;
        REQUIRE(logging::state_logger::serialize_state(ckpt_tree, cfg, h5).get().success);
        REQUIRE(fs::exists(snap));
    }   // drop env (frees any GPU context) before building the resume env

    // --- resume: parse + apply the checkpoint, then M more steps -------------
    logging::mem_group resumed_tree;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        sim->get_sim_env().set_run_id(run_id);

        auto pr = logging::state_logger::parse_state(snap, h5);
        REQUIRE(pr.result.success);
        for (const auto& w : pr.result.warnings) { INFO("parse warning: ", w); }
        CHECK(pr.result.warnings.empty());

        auto applied = logging::state_logger::apply_state(sim->get_sim_env(), pr.tree);
        REQUIRE(applied.success);
        for (const auto& w : applied.warnings) { INFO("apply warning: ", w); }
        CHECK(applied.warnings.empty());

        coast::test::run_for(*sim, k_resume_steps);
        resumed_tree = logging::state_logger::extract_state(
            sim->get_sim_env(), k_checkpoint_steps + k_resume_steps);
    }
    fs::remove(snap);

    // --- the property: resume == straight, bit-for-bit (metadata excluded:
    //     serialize stamps a wall-clock timestamp) ----------------------------
    CHECK(resumed_tree.groups.at("entities")   == straight_tree.groups.at("entities"));
    CHECK(resumed_tree.groups.at("io_buffers") == straight_tree.groups.at("io_buffers"));

    // --- vacuity: the M post-load steps actually advanced entity state -------
    CHECK(straight_tree.groups.at("entities") != ckpt_tree.groups.at("entities"));
}
