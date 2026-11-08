// tests/integration_ngm3/state_load_runtime_test.cpp
//
// End-to-end regression for the RUNTIME state-IO glue, CPU config. The component
// layers are already covered in isolation: run_quiesced mechanics by the
// thread-controller unit test, and the io_buffer read-page restore by
// state_restore_readpage_test (which drives state_logger directly). This test
// closes the gap between them — the production runtime path
// logging_manager::extract_state() / logging_manager::load_state(), which run the
// extract, the apply, AND the step-counter update INSIDE ctrl.run_quiesced() on
// the sim thread. A bug confined to that glue (wrong step variable, wrong tree,
// the step counter not set, a warning/metadata merge slip) would slip past every
// other test, since none of them call logging_manager's runtime entry points.
//
// Deterministic and RNG-immune, like state_restore_readpage_test: it asserts a
// fresh, never-stepped env's io_buffer read pages equal the saved values after a
// runtime load, and that the step counter was restored through the quiesced task
// and is visible to the caller once load_state returns. No stepping past the
// load, so the pre-existing CPU entity-RNG gap (save_state does not persist
// fast_rng_t state) cannot confound the property under test. See design §5.3 and
// design/entity-rng-state-restore-todo.md.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <unistd.h>

#include "app/build_env.h"
#include "cli/options.h"
#include "hdf5/executor.h"
#include "integration_helpers.h"
#include "logging/logging_manager.h"
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
// canonical read-page accessor. Empty buffers map to an empty vector.
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

TEST_CASE("logging_manager runtime save/load restores read pages and step (run_quiesced glue)") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;

    const auto snap = fs::temp_directory_path() /
        ("coast_load_runtime_" + std::to_string(::getpid()) + ".h5");
    fs::remove(snap);

    // --- run to step N, capture read pages, save via the runtime glue --------
    std::map<std::size_t, std::vector<float>> saved;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        coast::test::run_for(*sim, k_warmup_steps);

        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);
        REQUIRE(ctrl->get_step_count() == k_warmup_steps);

        saved = capture_read_pages(*sim);

        // extract_state runs on the sim thread via run_quiesced and reads the step
        // counter INSIDE the task — the returned step must match the live counter
        // (the property the headless checkpoint path depends on for consistency).
        logging::logging_manager lm(sim->get_sim_env(), h5);
        auto es = lm.extract_state(*ctrl);
        CHECK(es.step == k_warmup_steps);

        logging::state_save_config cfg;
        cfg.output_path = snap;
        REQUIRE(lm.serialize_state(std::move(es.tree), cfg).get().success);
        REQUIRE(fs::exists(snap));
    }

    // Sanity: there must be signal, else a zero-vs-zero match would pass even with
    // a restore bug present.
    bool any_nonzero = false;
    for (auto& [id, v] : saved) {
        (void)id;
        if (std::any_of(v.begin(), v.end(), [](float x) { return x != 0.0f; })) {
            any_nonzero = true;
            break;
        }
    }
    REQUIRE_MESSAGE(any_nonzero, "saved state is all-zero; test would be vacuous");

    // --- runtime load into a fresh, never-stepped env ------------------------
    std::map<std::size_t, std::vector<float>> got;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);

        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);
        REQUIRE(ctrl->get_step_count() == 0);

        // The runtime path: parse on the executor, then apply + set_step_count
        // INSIDE run_quiesced on the (parked, alive) sim thread.
        logging::logging_manager lm(sim->get_sim_env(), h5);
        auto res = lm.load_state(*ctrl, snap);
        REQUIRE(res.success);
        for (const auto& w : res.warnings) { INFO("load warning: ", w); }
        CHECK(res.warnings.empty());
        CHECK(res.restored_step == k_warmup_steps);

        // The step counter must be visible to this caller after load_state returns
        // — it is set on the sim thread inside the task and synchronised out via
        // the task's future (fut.get() in run_quiesced).
        CHECK(ctrl->get_step_count() == k_warmup_steps);

        // Deliberately NO stepping: the read page must already hold the restored
        // data after the quiesced apply.
        got = capture_read_pages(*sim);
    }

    fs::remove(snap);

    // --- every buffer's read page must equal the saved value, bit-for-bit ----
    REQUIRE(got.size() == saved.size());
    for (auto& [id, sv] : saved) {
        auto it = got.find(id);
        REQUIRE(it != got.end());
        INFO("buffer ", id);
        CHECK(it->second == sv);
    }
}

// A runtime load that an entity refuses returns a failed result instead of
// throwing: run_quiesced rethrows the entity's exception on the caller (the GUI
// thread), and the GUI's two load sites report !result but do not catch. The
// step counter is set only after every entity has loaded, so it stays unset.
TEST_CASE("logging_manager runtime load of a checkpoint that does not fit fails without throwing") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;

    const auto snap = fs::temp_directory_path() /
        ("coast_load_runtime_misfit_" + std::to_string(::getpid()) + ".h5");
    fs::remove(snap);

    // Save a checkpoint whose neuron group claims two more neurons than it has.
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        coast::test::run_for(*sim, 5);
        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);

        logging::logging_manager lm(sim->get_sim_env(), h5);
        auto es = lm.extract_state(*ctrl);
        int edited = 0;
        for (auto& [id, g] : es.tree.groups.at("entities").groups) {
            if (auto it = g.fields.find("neuron_cnt"); it != g.fields.end()) {
                auto& n = std::get<std::uint64_t>(it->second);
                n += 2;
                ++edited;
            }
        }
        REQUIRE(edited == 1);

        logging::state_save_config cfg;
        cfg.output_path = snap;
        REQUIRE(lm.serialize_state(std::move(es.tree), cfg).get().success);
    }

    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);
    auto* ctrl = sim->get_thread_controller();
    REQUIRE(ctrl);

    logging::logging_manager lm(sim->get_sim_env(), h5);
    logging::state_load_result res;
    REQUIRE_NOTHROW(res = lm.load_state(*ctrl, snap));
    fs::remove(snap);

    INFO("error: ", res.error_message);
    CHECK_FALSE(res.success);
    CHECK(res.error_message.find("checkpoint does not fit this group — neuron_cnt") != std::string::npos);
    CHECK(ctrl->get_step_count() == 0);
}
