// tests/integration_ngm3/gpu_state_roundtrip_test.cpp
//
// B1 regression: state save/load must round-trip a GPU-only io_buffer
// (io_buffer_gpu — GPU writer, GPU reader, no standing CPU mirror).
//
// Before the fix, state_logger::save_state read every buffer through
// cur_read_buffer(), which returns an EMPTY span for io_buffer_gpu, so such a
// buffer was persisted as a zero-length dataset; load_state then skipped it
// with a silent "size mismatch" warning. Save/load reported success while
// dropping all GPU-resident buffer content — invisible reproducibility loss.
//
// This drives a real save_state -> load_state cycle through state_logger over a
// full-GPU ngm3 config and asserts the GPU-only buffer's bytes survive exactly.
// The expected and restored values come from the same saved float32 bytes
// (HDF5 is lossless, GPU upload/download is a memcpy), so the comparison is
// exact and independent of GPU compute determinism. GPU-labeled: needs a real
// device, and is excluded by `ctest -LE gpu`.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
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

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t k_steps       = 50;   // enough for non-zero GPU output
constexpr seed_offset_t k_seed_offset = {};

// Locate the first pure GPU-resident buffer in the env: size > 0 but an empty
// CPU read span. Only io_buffer_gpu satisfies this — the cpu_to_gpu / gpu_to_cpu
// variants keep a CPU mirror that cur_read_buffer() exposes. Returns true and
// sets `id` on success.
bool find_gpu_only_buffer(coast::sim_instance& sim, std::size_t& id) {
    for (auto& [bid, buf] : sim.get_sim_env().get_io_buffers()) {
        if (buf->size() > 0 && buf->cur_read_buffer().empty()) {
            id = bid;
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("ngm3 gpu state save/load round-trips a GPU-only io_buffer") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;   // full-GPU ngm3 config
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;

    // --- run a GPU sim and capture a GPU-only buffer's content -------------
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);
    REQUIRE(sim->get_sim_env().has_gpu());   // config must build a GPU pipeline

    coast::test::run_for(*sim, k_steps);

    std::size_t gpu_id = 0;
    REQUIRE_MESSAGE(find_gpu_only_buffer(*sim, gpu_id),
                    "config must contain a GPU-only io_buffer (io_buffer_gpu)");

    auto& src = *sim->get_sim_env().get_io_buffers().at(gpu_id);
    std::vector<float> expected(src.size());
    REQUIRE(src.read_back_sync(expected));   // on-demand readback of the GPU page

    // The run must have produced signal: an all-zero buffer would round-trip
    // even with the bug present (zero-length save reloads as zeros).
    REQUIRE(std::any_of(expected.begin(), expected.end(),
                        [](float v) { return v != 0.0f; }));

    // --- save state, then drop the first sim (frees its GPU context) -------
    const auto snap = fs::temp_directory_path() /
        ("coast_b1_gpu_roundtrip_" + std::to_string(::getpid()) + ".h5");
    fs::remove(snap);
    {
        logging::state_logger    sl;
        logging::state_save_config cfg;
        cfg.output_path = snap;
        REQUIRE(sl.save_state(sim->get_sim_env(), k_steps, cfg, h5).success);
    }
    REQUIRE(fs::exists(snap));
    sim.reset();

    // --- load into a fresh env and verify the GPU buffer was restored ------
    auto sim2 = coast::app::build_env(opts, h5);
    REQUIRE(sim2);
    {
        logging::state_logger sl;
        auto res = sl.load_state(sim2->get_sim_env(), snap, h5);
        REQUIRE(res.success);
        // No restore failure / size-mismatch warning for our GPU buffer.
        for (const auto& w : res.warnings) {
            INFO("load warning: ", w);
            CHECK(w.find(std::to_string(gpu_id)) == std::string::npos);
        }
    }

    auto& dst = *sim2->get_sim_env().get_io_buffers().at(gpu_id);
    REQUIRE(dst.size() == expected.size());
    // load now writes the read page directly (snapshot/restore are read-page
    // symmetric — no swap), so read_back_sync observes the restored data
    // immediately, with no manual swap_buffer() (the sequence production never
    // performed; its absence is what hid the prior buffer-restore bug).
    std::vector<float> restored(dst.size());
    REQUIRE(dst.read_back_sync(restored));

    // Same float32 bytes through HDF5 + GPU upload/download: bit-exact.
    CHECK(restored == expected);

    fs::remove(snap);
}
