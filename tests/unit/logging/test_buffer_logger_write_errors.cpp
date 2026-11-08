// tests/unit/logging/test_buffer_logger_write_errors.cpp
//
// Issue #203: buffer_logger dropped every HDF5 write result, so a full disk
// produced an unreadable log while stop() and the run reported success.
//
// The failure is injected with RLIMIT_FSIZE (SIGXFSZ ignored, so the write
// returns EFBIG instead of killing the process): the same errno path as a full
// disk, with no special filesystem needed. The limit is process-wide, which
// includes the HDF5 executor thread; a guard restores it after each case.
//
// The sim is driven by hand (process() + swap_io() on this thread), so the
// post-swap hook runs synchronously and every batch boundary is deterministic.
//
// Two places a failure can surface, one case each:
//   - in write_frames(): a chunk larger than HDF5's chunk cache bypasses it,
//     so H5Dwrite itself fails;
//   - only at stop: small chunks sit in the cache, every H5Dwrite succeeds,
//     and the error appears when stop()'s H5Fflush writes them out.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

#include "hdf5/executor.h"
#include "logging/buffer_logger.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"

namespace {

namespace fs = std::filesystem;

// An entity with an output buffer of `size` floats and nothing else.
struct sized_entity : sim::io_entity {
    sim::output_id_t oid;
    std::size_t      size;
    sized_entity(sim::output_id_t id, std::size_t n) : oid(id), size(n) {}
    sim::step_result process() override { return {}; }
    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return size; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// Lower the soft file-size limit until release() or the end of the scope.
// While it is active, doctest output must not go to a regular file (pipe it):
// the limit applies to every file the process writes.
struct fsize_limit {
    rlimit saved{};
    void (*saved_handler)(int) = SIG_DFL;
    bool active = false;   // not saved_handler: SIG_DFL is a null pointer
    explicit fsize_limit(rlim_t bytes) {
        REQUIRE(::getrlimit(RLIMIT_FSIZE, &saved) == 0);
        saved_handler = std::signal(SIGXFSZ, SIG_IGN);
        rlimit lim = saved;
        lim.rlim_cur = bytes;
        REQUIRE(::setrlimit(RLIMIT_FSIZE, &lim) == 0);
        active = true;
    }
    void release() {
        if (!active) return;
        active = false;
        REQUIRE(::setrlimit(RLIMIT_FSIZE, &saved) == 0);
        std::signal(SIGXFSZ, saved_handler);
    }
    ~fsize_limit() {
        if (!active) return;
        ::setrlimit(RLIMIT_FSIZE, &saved);
        std::signal(SIGXFSZ, saved_handler);
    }
    fsize_limit(const fsize_limit&) = delete;
    fsize_limit& operator=(const fsize_limit&) = delete;
};

fs::path scratch_file(const std::string& stem)
{
    return fs::temp_directory_path() / (stem + "_" + std::to_string(::getpid()) + ".h5");
}

logging::buffer_logger_config config_for(const fs::path& p, std::size_t chunk)
{
    logging::buffer_logger_config cfg;
    cfg.output_path = p;
    cfg.chunk_size  = chunk;
    cfg.compression.algorithm = logging::compression_config::type::none;  // bytes on disk = bytes logged
    return cfg;
}

void step(sim::env& env, int n)
{
    for (int i = 0; i < n; ++i) {
        REQUIRE(env.process().has_value());
        REQUIRE(env.swap_io().has_value());
    }
}

// Floats per frame such that one frame (chunk_size 1 = one chunk) exceeds the
// default raw-data chunk cache: such a chunk bypasses the cache, so H5Dwrite
// writes it to the file at once. Measured, not assumed: the default is 8 MiB in
// HDF5 2.2 (1 MiB in older releases).
std::size_t uncached_frame_floats(hdf5::executor& h5)
{
    const std::size_t cache_bytes = h5.submit_sync([] {
        const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        std::size_t nslots = 0, nbytes = 0;
        double w0 = 0;
        const herr_t rc = H5Pget_cache(fapl, nullptr, &nslots, &nbytes, &w0);
        H5Pclose(fapl);
        return rc < 0 ? std::size_t{0} : nbytes;
    });
    REQUIRE(cache_bytes > 0);
    return cache_bytes / sizeof(float) + 1024;
}

}  // namespace

TEST_CASE("control: no limit, the uncached-chunk log succeeds") {
    hdf5::executor h5;
    sim::env env;
    env.emplace_back<sized_entity>(sim::output_id_t{0}, uncached_frame_floats(h5));
    env.init_io_buffers();
    const auto path = scratch_file("coast_bl203_control");

    logging::buffer_logger bl(env, config_for(path, 1), h5);
    REQUIRE(bl.start());
    step(env, 3);   // the hook captures from the first step's swap on
    CHECK(bl.write_ok());
    CHECK(bl.stats().frames_written == 3);
    const auto r = bl.stop();
    CHECK_MESSAGE(r.success, r.error_message);
    fs::remove(path);
}

TEST_CASE("a write that fails in write_frames trips the sticky flag and fails stop()") {
    hdf5::executor h5;
    sim::env env;
    env.emplace_back<sized_entity>(sim::output_id_t{0}, uncached_frame_floats(h5));
    env.init_io_buffers();
    const auto path = scratch_file("coast_bl203_write");

    logging::buffer_logger bl(env, config_for(path, 1), h5);
    REQUIRE(bl.start());
    step(env, 1);
    REQUIRE(bl.write_ok());
    REQUIRE(bl.stats().frames_written == 1);

    {
        // Room for much less than one more frame.
        fsize_limit lim(static_cast<rlim_t>(fs::file_size(path)) + 4096);
        step(env, 1);
        CHECK_FALSE(bl.write_ok());
        CHECK(bl.stats().frames_written == 1);   // the failed batch is not counted

        // Sticky: nothing further is written, even once the disk has room again.
        lim.release();
        step(env, 1);
        CHECK_FALSE(bl.write_ok());
        CHECK(bl.stats().frames_written == 1);
    }

    const auto r = bl.stop();
    CHECK_FALSE(r.success);
    CHECK(r.error_message.find("incomplete") != std::string::npos);
    CHECK(r.error_message.find("steps/") != std::string::npos);   // names what failed
    fs::remove(path);
}

TEST_CASE("a failure that surfaces only at close fails stop()") {
    hdf5::executor h5;
    sim::env env;
    env.emplace_back<sized_entity>(sim::output_id_t{0}, 16);
    env.init_io_buffers();
    const auto path = scratch_file("coast_bl203_close");

    logging::buffer_logger bl(env, config_for(path, 1000), h5);
    REQUIRE(bl.start());
    step(env, 50);   // fewer than one chunk: all frames still buffered

    // The file may not grow at all: stop()'s write_frames() only fills the
    // chunk cache (succeeds); the flush that writes it out fails.
    fsize_limit lim(static_cast<rlim_t>(fs::file_size(path)));
    const auto r = bl.stop();
    lim.release();

    CHECK_FALSE(bl.write_ok());
    CHECK_FALSE(r.success);
    CHECK(r.error_message.find("H5Fflush") != std::string::npos);
    fs::remove(path);
}

TEST_CASE("start() after a failed session clears the flag") {
    hdf5::executor h5;
    sim::env env;
    env.emplace_back<sized_entity>(sim::output_id_t{0}, 16);
    env.init_io_buffers();
    const auto path = scratch_file("coast_bl203_restart");

    logging::buffer_logger bl(env, config_for(path, 1000), h5);
    REQUIRE(bl.start());
    step(env, 5);
    {
        fsize_limit lim(static_cast<rlim_t>(fs::file_size(path)));
        CHECK_FALSE(bl.stop().success);
    }
    REQUIRE(bl.start());
    CHECK(bl.write_ok());
    step(env, 5);
    const auto r = bl.stop();
    CHECK_MESSAGE(r.success, r.error_message);
    fs::remove(path);
}

// A file whose flush failed cannot be closed: HDF5 keeps it open (measured:
// H5Fget_obj_count reports it after the failed H5Fclose). With the limit lifted
// again, libhdf5's atexit H5close then aborts this process in H5SL_term_package
// (assertion, libhdf5 2.2.0 as packaged on the dev box) — after every case has
// passed. Not reproduced by the coast binary or a plain C probe; cause not
// determined. _Exit skips that teardown so ctest sees doctest's verdict.
int main(int argc, char** argv)
{
    doctest::Context ctx(argc, argv);
    const int rc = ctx.run();
    std::fflush(nullptr);
    std::_Exit(rc);
}
