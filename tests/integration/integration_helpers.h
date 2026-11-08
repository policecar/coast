#ifndef COAST_TEST_INTEGRATION_HELPERS_H
#define COAST_TEST_INTEGRATION_HELPERS_H

#include <cstdint>
#include <filesystem>
#include <string_view>

#include "blob_writer.h"
#include "compare.h"   // coast::compare::compare_mode

namespace coast { class sim_instance; }

namespace coast::test {

// Scratch directory unique per process and instance (pid + counter), so
// parallel ctest runs never share a path; removed recursively on destruction,
// including when a REQUIRE unwinds the test.
class scratch_dir_t {
public:
    explicit scratch_dir_t(std::string_view label);
    ~scratch_dir_t();
    scratch_dir_t(const scratch_dir_t&)            = delete;
    scratch_dir_t& operator=(const scratch_dir_t&) = delete;
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// Drive sim through `steps` additional steps from its current step count.
// Synchronous: returns when the step limit is reached. Mirrors the headless
// runner's step-limit + adaptive-polling pattern.
void run_for(sim_instance& sim, std::uint64_t steps);

// Open a blob at the given path. Parent dir is auto-created. The caller
// must call close() (or let the destructor handle it). Tests typically pass
// std::filesystem::path{ COAST_TEST_BLOB_PATH }; the macro is defined per
// test by coast_add_integration_test.
[[nodiscard]] blob_writer open_blob(const std::filesystem::path& path);

// Walk every io_buffer in the env (sorted by ID), writing:
//   buffer_<id>/size  (int64)
//   buffer_<id>/sum, mean, min, max  (double, computed from cur_read_buffer)
//   buffer_<id>/values — inline array if size <= 64, else binary sidecar
//                        named "buffer_<id>.bin"
// step_count is NOT written here — call site adds meta/step_count separately.
//
// A pure io_buffer_gpu (GPU writer, GPU/no reader) returns an empty CPU read
// span. When COAST_WITH_GPU is on and the env carries a gpu_context, such a
// buffer is read back on demand (staging CopyBufferToBuffer + map) at this
// post-run idle boundary so it fingerprints identically to a CPU-visible
// variant. This is test-side only — production io_buffer_gpu stays GPU-resident.
void fingerprint_io_buffers(blob_writer& w, sim_instance& sim);

// Compare blob against goldens_dir.
//   - If env COAST_TEST_CAPTURE_ONLY is "1": return 0 immediately (used by
//     scripts/update_golden.sh to capture without requiring a pre-existing
//     golden).
//   - Else: invoke coast::compare::compare_against_goldens, render report
//     to stdout, return exit_code_of(outcome): 0/1/2.
[[nodiscard]] int compare(const std::filesystem::path& blob,
                          const std::filesystem::path& goldens_dir,
                          const std::filesystem::path& tolerances_path = {},
                          coast::compare::compare_mode mode =
                              coast::compare::compare_mode::strict);

// Canonical "io-buffer fingerprint baseline" test body. Used by every
// A-tier baseline test (one per config × seed_offset). Each test's main.cpp
// is just a TEST_CASE that calls this with the COAST_TEST_* macros.
//
// Builds env, runs the sim for `steps`, fingerprints io_buffers + step
// count, compares against goldens_dir. Triggers doctest CHECK on the
// compare result.
void run_io_buffer_fingerprint_test(
    const char*                    config_path,
    std::uint32_t                  seed_offset,
    std::uint64_t                  steps,
    const std::filesystem::path&   blob_path,
    const std::filesystem::path&   goldens_dir,
    coast::compare::compare_mode   mode  = coast::compare::compare_mode::strict,
    const std::filesystem::path&   tolerances_path = {});

} // namespace coast::test

#endif // COAST_TEST_INTEGRATION_HELPERS_H
