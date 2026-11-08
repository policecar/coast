// tests/integration_ngm3/state_resume_repro_test.cpp
//
// Step-once reproducibility regression for the io_buffer read-page restore fix
// (design §5.3). A state save captures each io_buffer's READ page; load must
// restore the READ page directly (no swap), so the first step after a load
// consumes exactly the data the uninterrupted run consumed.
//
// The prior bug restored the WRITE page and relied on a swap that production
// never performed: on resume, do_step() ran process() first — overwriting the
// restored write page and reading the stale (un-restored) read page — then
// swapped, so restored buffer state never reached a consumer and the first
// resumed step diverged. The only existing round-trip test
// (gpu_state_roundtrip_test) hid this by inserting a manual swap_buffer() with
// no process() in between. This test steps the sim AFTER a load, which is the
// sequence that exposes the bug.
//
// Protocol: run an uninterrupted reference to step N+1 (saving at N); separately,
// load the step-N save into a fresh env and step once. The two envs' io_buffer
// read pages at step N+1 must match.
//
// GPU config only. This is a *full step-once reproducibility* test, which needs
// the whole sim to reproduce across save/load. The GPU entities use counter-based
// PRNGs keyed on the step count, so they do. The CPU entities (mnist_io, ngm3
// neuron_group_t) carry stateful fast_rng_t RNGs that save_state does NOT persist
// — a separate, pre-existing entity-state gap that would confound this property
// on a CPU config. The CPU read-page restore is covered instead by the
// deterministic, RNG-immune state_restore_readpage_test (restored read page ==
// saved, no stepping). See design §5.3.

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

// Capture every io_buffer's READ page, keyed by buffer id, via snapshot() — the
// canonical read-page accessor that handles CPU pages, CPU mirrors, and
// on-demand GPU readback uniformly. Empty buffers map to an empty vector.
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

TEST_CASE("state save/load reproduces the first resumed step (read-page restore)") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;
    // Every env below is the same run, rebuilt: its checkpoints continue it, datasets
    // included (one run_id, as coastd fixes one per job; SPEC.md S.5.14).
    std::string run_id;

    const auto snap = fs::temp_directory_path() /
        ("coast_resume_repro_" + std::to_string(::getpid()) + ".h5");
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

        logging::state_logger      sl;
        logging::state_save_config cfg;
        cfg.output_path = snap;
        REQUIRE(sl.save_state(sim->get_sim_env(), n, cfg, h5).success);
        REQUIRE(fs::exists(snap));

        coast::test::run_for(*sim, 1);          // N -> N+1
        ref = capture_read_pages(*sim);
    }   // drop the reference env (frees any GPU context) before building the next

    // Sanity: the reference must carry signal, else the comparison is vacuous —
    // an all-zero state round-trips even with the restore bug present.
    bool any_nonzero = false;
    for (auto& [id, v] : ref) {
        (void)id;
        if (std::any_of(v.begin(), v.end(), [](float x) { return x != 0.0f; })) {
            any_nonzero = true;
            break;
        }
    }
    REQUIRE_MESSAGE(any_nonzero, "reference state is all-zero; test would be vacuous");

    // --- fresh env: load the step-N save, then step once (N+1) ---------------
    std::map<std::size_t, std::vector<float>> got;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        sim->get_sim_env().set_run_id(run_id);

        logging::state_logger sl;
        auto res = sl.load_state(sim->get_sim_env(), snap, h5);
        REQUIRE(res.success);
        for (const auto& w : res.warnings) { INFO("load warning: ", w); }
        CHECK(res.warnings.empty());

        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);
        ctrl->set_step_count(res.restored_step);   // mirror the production load path
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
