// tests/integration_ngm3/state_restore_readpage_test.cpp
//
// Read-page restore regression (design §5.3), CPU config. Directly inverts the
// pre-existing bug: load_state used to restore each io_buffer into its WRITE
// page and rely on a swap that production never performed, so the restored data
// never reached the READ page a consumer reads. The fix has restore() write the
// READ page directly (no swap).
//
// This test asserts exactly that, deterministically and without stepping past
// the load: capture every io_buffer's read page at step N, save, load into a
// fresh env, and assert each buffer's read page equals the saved value
// bit-for-bit (CPU copy + HDF5 float32 are lossless). Under the old bug a fresh,
// never-stepped env would still show a zero read page (the data landed in the
// write page) → mismatch.
//
// Why no step-once here (unlike the GPU variant): the CPU entities (mnist_io,
// ngm3 neuron_group_t) carry stateful fast_rng_t RNGs that save_state does NOT
// persist, so a CPU sim is not bit-reproducible across save/load regardless of
// the buffer fix — a separate, pre-existing entity-state gap. Stepping after a
// load would conflate that gap with the read-page property under test. The GPU
// step-once test (counter-based PRNG, reproducible) covers "read page consumed
// after a step"; this test covers "restore writes the read page" on the CPU
// io_buffer_cpu::restore path.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
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

TEST_CASE("state load restores io_buffer read pages directly (no swap)") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;

    const auto snap = fs::temp_directory_path() /
        ("coast_restore_readpage_" + std::to_string(::getpid()) + ".h5");
    fs::remove(snap);

    // --- run to step N, capture read pages, save -----------------------------
    std::map<std::size_t, std::vector<float>> saved;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        coast::test::run_for(*sim, k_warmup_steps);

        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);
        const std::uint64_t n = ctrl->get_step_count();
        REQUIRE(n == k_warmup_steps);

        saved = capture_read_pages(*sim);

        logging::state_logger      sl;
        logging::state_save_config cfg;
        cfg.output_path = snap;
        REQUIRE(sl.save_state(sim->get_sim_env(), n, cfg, h5).success);
        REQUIRE(fs::exists(snap));
    }

    // Sanity: there must be signal, else a zero-vs-zero match would pass even
    // with the restore bug present.
    bool any_nonzero = false;
    for (auto& [id, v] : saved) {
        (void)id;
        if (std::any_of(v.begin(), v.end(), [](float x) { return x != 0.0f; })) {
            any_nonzero = true;
            break;
        }
    }
    REQUIRE_MESSAGE(any_nonzero, "saved state is all-zero; test would be vacuous");

    // --- load into a fresh, never-stepped env; read pages must be restored ---
    std::map<std::size_t, std::vector<float>> got;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);

        logging::state_logger sl;
        auto res = sl.load_state(sim->get_sim_env(), snap, h5);
        REQUIRE(res.success);
        for (const auto& w : res.warnings) { INFO("load warning: ", w); }
        CHECK(res.warnings.empty());

        // Deliberately NO stepping: the read page must already hold the restored
        // data. Under the old write-page+swap bug a fresh env's read page would
        // still be zero here.
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
