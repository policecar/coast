// tests/unit/logging/test_buffer_logger_lifecycle.cpp
//
// buffer_logger start/stop against a LIVE sim thread, and what it records.
//
// Issue #197: stop() used to run on the caller's thread while the sim thread
// could still be inside the logger's post-swap hook — hook removal is only
// queued — so the hook's capture_frame()/write_frames() raced stop()'s final
// write and HDF5 teardown, and then the freed logger. stop_buffer_logging(ctrl)
// now detaches the hook inside run_quiesced. These cases are meant to be run
// under TSan/ASan (build-tsan / build-asan): without a sanitizer they assert
// only that every cycle succeeds, which the racy version also did most of the
// time.
//
// Issue #202: the log recorded frame counts (0, 1, 2, ...) as step_numbers and
// always start_step = 0, and its timestamps' origin was a process-lifetime
// static. The last case checks the recorded steps are simulation steps.
//
// Issue #214: stats() was a reference to counters write_frames() updates on
// the HDF5 executor, read unsynchronised by the GUI / headless main thread.
// The polling case is again a TSan target. It also checks that the counts
// survive stop_buffer_logging(): headless reports them on its terminal progress
// line, which is emitted only after the stop (PR #228).
//
// Uses an empty sim::env (no entities): the hook still runs every step and,
// with chunk_size = 1, calls write_frames() every step — the maximum overlap
// with stop().

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <cstdint>
#include <string>
#include <vector>
#include <unistd.h>

#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"
#include "logging/logging_manager.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"

#include "await_progress.h"
#include "sim_sync.h"

// No wall-clock time decides a verdict: "the hook runs" is counted in sim-loop
// iterations, waits are on progress counters, and the timestamp origin is
// checked against an injected clock (buffer_logger_config::clock).

namespace {

std::filesystem::path scratch_file(const std::string& stem)
{
    return std::filesystem::temp_directory_path() /
           (stem + "_" + std::to_string(::getpid()) + ".h5");
}

logging::buffer_logger_config small_chunk_config(const std::filesystem::path& p)
{
    logging::buffer_logger_config cfg;
    cfg.output_path = p;
    cfg.chunk_size  = 1;   // write_frames() from the hook on every step
    return cfg;
}

TEST_CASE("stop_buffer_logging while the sim free-runs (issue #197)") {
    hdf5::executor h5;
    sim::env env;
    sim::sim_thread_controller ctrl(env);
    logging::logging_manager lm(env, h5);
    const auto path = scratch_file("coast_buffer_logger_197");

    ctrl.set_mode(sim::sim_mode::unlimited);
    for (int cycle = 0; cycle < 25; ++cycle) {
        CAPTURE(cycle);
        REQUIRE(lm.start_buffer_logging(small_chunk_config(path)));
        coast_test::sim_iterations(ctrl, 4);   // the hook runs (and writes) on several steps
        const auto r = lm.stop_buffer_logging(ctrl);
        CHECK_MESSAGE(r.success, r.error_message);
        CHECK_FALSE(lm.is_buffer_logging());
    }
    ctrl.set_mode(sim::sim_mode::paused);
    std::filesystem::remove(path);
}

TEST_CASE("logging_manager destroyed after the controller stops an active log") {
    // The ~sim_instance order: join the sim thread, then destroy the manager,
    // whose stop_all() finishes the still-active logger without quiescing.
    hdf5::executor h5;
    sim::env env;
    const auto path = scratch_file("coast_buffer_logger_197_teardown");
    {
        auto ctrl = std::make_unique<sim::sim_thread_controller>(env);
        auto lm   = std::make_unique<logging::logging_manager>(env, h5);
        REQUIRE(lm->start_buffer_logging(small_chunk_config(path)));
        ctrl->set_mode(sim::sim_mode::unlimited);
        coast_test::sim_iterations(*ctrl, 4);   // logging is live when the controller stops
        ctrl.reset();
        lm.reset();
    }
    std::filesystem::remove(path);
}

TEST_CASE("combined_stats polled while the sim logs (issue #214)") {
    hdf5::executor h5;
    sim::env env;
    sim::sim_thread_controller ctrl(env);
    logging::logging_manager lm(env, h5);
    const auto path = scratch_file("coast_buffer_logger_214");

    REQUIRE(lm.start_buffer_logging(small_chunk_config(path)));
    ctrl.set_mode(sim::sim_mode::unlimited);
    // Poll while a fixed number of frames is written, not for a fixed time: the
    // amount of overlap (what TSan needs to see the pre-fix race) no longer
    // depends on the machine's speed. frames_written is its own heartbeat.
    constexpr std::uint64_t k_frames = 2000;
    std::uint64_t last = 0;
    bool monotonic = true;
    REQUIRE(coast_test::await_progress(
        [&] {
            const auto frames = lm.combined_stats().frames_written;
            monotonic = monotonic && frames >= last;
            last = frames;
            return frames >= k_frames;
        },
        [&] { return lm.combined_stats().frames_written; }));
    CHECK(monotonic);
    REQUIRE(lm.stop_buffer_logging(ctrl));
    // The logger is gone; its final counts (incl. the tail stop() flushed) stay.
    CHECK(lm.combined_stats().frames_written >= last);
    ctrl.set_mode(sim::sim_mode::paused);
    std::filesystem::remove(path);
}

struct logged_steps {
    std::optional<std::uint64_t> start_step;
    std::vector<std::uint64_t>   steps;
    std::vector<double>          timestamps;
};

logged_steps read_logged_steps(hdf5::executor& h5, const std::filesystem::path& p)
{
    return h5.submit_sync([&] {
        logged_steps r;
        auto f     = hdf5::file::open_read(p.string());
        auto meta  = hdf5::group::open(f.id(), "metadata");
        auto steps = hdf5::group::open(f.id(), "steps");
        r.start_step = hdf5::read_attr_uint64(meta.id(), "start_step");
        auto sn = hdf5::dataset::open(steps.id(), "step_numbers");
        r.steps = hdf5::read_array<std::uint64_t>(sn.id(), H5T_NATIVE_UINT64, 1).values;
        auto ts = hdf5::dataset::open(steps.id(), "timestamps");
        r.timestamps = hdf5::read_array<double>(ts.id(), H5T_NATIVE_DOUBLE, 1).values;
        return r;
    });
}

// Step the paused controller until its count reaches `target`, then park it.
void run_to(sim::sim_thread_controller& ctrl, std::uint64_t target)
{
    ctrl.set_step_limit(target);
    ctrl.set_mode(sim::sim_mode::unlimited);
    REQUIRE(coast_test::await_progress([&] { return ctrl.get_step_count() >= target; },
                                       [&] { return ctrl.get_step_count(); }));
    REQUIRE(ctrl.get_step_count() == target);
    // The limit pauses at the top of the loop; quiesce to be sure it is parked
    // there and no step is in flight.
    ctrl.run_quiesced([] {});
}

TEST_CASE("buffer log records simulation steps, not frame counts (issue #202)") {
    hdf5::executor h5;
    sim::env env;
    sim::sim_thread_controller ctrl(env);
    logging::logging_manager lm(env, h5);
    const auto p1 = scratch_file("coast_buffer_logger_202_a");
    const auto p2 = scratch_file("coast_buffer_logger_202_b");

    // As after --load-state of a step-100 checkpoint.
    ctrl.run_quiesced([&] { ctrl.set_step_count(100); });

    // A test clock that stands still except where the test moves it, so each
    // session's timestamp origin is known exactly.
    std::atomic<std::int64_t> now_s{1000};
    const auto clock = [&now_s] {
        return std::chrono::steady_clock::time_point(std::chrono::seconds(now_s.load()));
    };

    auto cfg = small_chunk_config(p1);
    cfg.clock      = clock;
    cfg.chunk_size = 2;   // 5 frames -> batches of 2, 2, then 1 at stop
    REQUIRE(lm.start_buffer_logging(cfg));
    run_to(ctrl, 105);
    REQUIRE(lm.stop_buffer_logging(ctrl));

    const auto a = read_logged_steps(h5, p1);
    CHECK(a.steps == std::vector<std::uint64_t>{100, 101, 102, 103, 104});
    REQUIRE(a.start_step.has_value());
    CHECK(*a.start_step == 100);
    REQUIRE(a.timestamps.size() == 5);
    for (std::size_t i = 1; i < a.timestamps.size(); ++i)
        CHECK(a.timestamps[i] >= a.timestamps[i - 1]);

    // A second session's timestamps start from ITS start(), not the first's: it
    // starts 4000 s later on the test clock.
    now_s += 4000;
    auto cfg2  = small_chunk_config(p2);
    cfg2.clock = clock;
    REQUIRE(lm.start_buffer_logging(cfg2));
    run_to(ctrl, 108);
    REQUIRE(lm.stop_buffer_logging(ctrl));

    const auto b = read_logged_steps(h5, p2);
    CHECK(b.steps == std::vector<std::uint64_t>{105, 106, 107});
    REQUIRE(b.start_step.has_value());
    CHECK(*b.start_step == 105);
    REQUIRE_FALSE(b.timestamps.empty());
    CHECK(b.timestamps.front() == 0.0);   // an origin at the first start() gives 4000

    std::filesystem::remove(p1);
    std::filesystem::remove(p2);
}

} // namespace
