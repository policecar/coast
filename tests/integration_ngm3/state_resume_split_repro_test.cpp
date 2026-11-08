// tests/integration_ngm3/state_resume_split_repro_test.cpp
//
// Step 3 gate: step-once reproducibility through the NEW load split
// (parse_state + apply_state), the runtime path Step 4 will wire onto
// run_quiesced. The companion state_resume_repro_test exercises the same
// property through the monolithic load_state; this one proves the split apply
// produces the same coherent read-page state a consumer steps on.
//
// Protocol: run an uninterrupted reference to step N+1 (saving at N). Separately,
// parse the step-N save into a tree, apply it into a fresh env, set_step_count to
// the restored step, and step once. The two envs' io_buffer read pages at step
// N+1 must match.
//
// GPU config only, for the same reason as state_resume_repro_test: a full
// step-once reproducibility property needs the whole sim to reproduce across
// save/load. The GPU entities use counter-based PRNGs keyed on the step count, so
// they do; the CPU entities (mnist_io, ngm3 neuron_group_t) carry stateful
// fast_rng_t RNGs that save_state does NOT persist (a separate, pre-existing
// entity-state gap). The CPU read-page restore through the split is covered
// instead by the deterministic, no-step state_roundtrip_split_test. See design
// §5.3 and design/entity-rng-state-restore-todo.md.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <unistd.h>

#include "app/build_env.h"
#include "cli/options.h"
#include "hdf5/executor.h"
#include "integration_helpers.h"
#include "logging/state_logger.h"
#include "sim_core/io_buffer.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t k_warmup_steps = 50;   // step N: enough for non-trivial state
constexpr seed_offset_t k_seed_offset  = {};

// Capture every io_buffer's READ page via snapshot() — the canonical read-page
// accessor (CPU page, CPU mirror, on-demand GPU readback handled uniformly).
std::map<std::size_t, std::vector<float>> capture_read_pages(coast::sim_instance& sim) {
    std::map<std::size_t, std::vector<float>> out;
    for (auto& [id, buf] : sim.get_sim_env().get_io_buffers()) {
        std::vector<float> v(buf->size());
        if (!v.empty()) {
            REQUIRE(buf->snapshot(v));
        }
        out.emplace(id, std::move(v));
    }
    return out;
}

} // namespace

TEST_CASE("split parse+apply reproduces the first resumed step (read-page restore)") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;
    // Every env below is the same run, rebuilt: its checkpoints continue it, datasets
    // included (one run_id, as coastd fixes one per job; SPEC.md S.5.14).
    std::string run_id;

    const auto snap = fs::temp_directory_path() /
        ("coast_resume_split_repro_" + std::to_string(::getpid()) + ".h5");
    fs::remove(snap);

    // --- reference: run to step N, save, then step once more (N+1) -----------
    std::map<std::size_t, std::vector<float>> ref;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        run_id = sim->get_sim_env().run_id();
        coast::test::run_for(*sim, k_warmup_steps);

        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);
        const std::uint64_t n = ctrl->get_step_count();
        REQUIRE(n == k_warmup_steps);

        // save_state internally uses extract_state + serialize_state.
        logging::state_logger      sl;
        logging::state_save_config cfg;
        cfg.output_path = snap;
        REQUIRE(sl.save_state(sim->get_sim_env(), n, cfg, h5).success);
        REQUIRE(fs::exists(snap));

        coast::test::run_for(*sim, 1);          // N -> N+1
        ref = capture_read_pages(*sim);
    }   // drop the reference env (frees any GPU context) before building the next

    bool any_nonzero = false;
    for (auto& [id, v] : ref) {
        (void)id;
        if (std::any_of(v.begin(), v.end(), [](float x) { return x != 0.0f; })) {
            any_nonzero = true;
            break;
        }
    }
    REQUIRE_MESSAGE(any_nonzero, "reference state is all-zero; test would be vacuous");

    // --- fresh env: parse + apply the step-N save, then step once (N+1) -------
    std::map<std::size_t, std::vector<float>> got;
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

        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);
        ctrl->set_step_count(applied.restored_step);   // mirror the Step-4 load path
        REQUIRE(ctrl->get_step_count() == k_warmup_steps);

        coast::test::run_for(*sim, 1);          // N -> N+1
        got = capture_read_pages(*sim);
    }

    fs::remove(snap);

    // --- the first resumed step must match the uninterrupted run -------------
    REQUIRE(got.size() == ref.size());
    constexpr float abs_tol = 1e-4f;   // tolerates minor GPU recompute noise; the
                                       // restore bug diverges by O(0.1-1).
    for (auto& [id, rv] : ref) {
        auto it = got.find(id);
        REQUIRE(it != got.end());
        const auto& gv = it->second;
        REQUIRE(gv.size() == rv.size());
        float max_diff = 0.0f;
        for (std::size_t i = 0; i < rv.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(gv[i] - rv[i]));
        }
        INFO("buffer ", id, " max|Δ| = ", max_diff);
        CHECK(max_diff <= abs_tol);
    }
}
