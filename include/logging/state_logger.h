//
// State Logger - full simulation state persistence
//

#ifndef COAST_STATE_LOGGER_H
#define COAST_STATE_LOGGER_H

#include "logging/logging_common.h"
#include "logging/state_io.h"
#include "logging/mem_state_io.h"
#include "logging/hdf5_utils.h"

#include <filesystem>
#include <future>
#include <string>
#include <vector>
#include <cstdint>

namespace sim {
class env;
}

namespace hdf5 { class executor; }

namespace logging {

/// Configuration for state saving
struct state_save_config {
    std::filesystem::path output_path;
    std::string config_json;         // Simulation config for validation on load
};

/// Result of loading state
struct state_load_result {
    bool success = false;
    std::uint64_t restored_step = 0;
    std::string config_json;         // Original config from file
    std::vector<std::string> warnings;
    std::string error_message;

    explicit operator bool() const { return success; }
};

/// HDF5 implementation of state_writer
class hdf5_state_writer : public state_writer {
public:
    explicit hdf5_state_writer(hdf5::file& file);
    ~hdf5_state_writer() override;

    void begin_group(std::string_view name) override;
    void end_group() override;

    void write(std::string_view name, double value) override;
    void write(std::string_view name, float value) override;
    void write(std::string_view name, std::int32_t value) override;
    void write(std::string_view name, std::int64_t value) override;
    void write(std::string_view name, std::uint32_t value) override;
    void write(std::string_view name, std::uint64_t value) override;
    void write(std::string_view name, bool value) override;

    void write(std::string_view name, std::span<const float> values) override;
    void write(std::string_view name, std::span<const double> values) override;
    void write(std::string_view name, std::span<const std::int32_t> values) override;
    void write(std::string_view name, std::span<const std::uint32_t> values) override;
    void write(std::string_view name, std::span<const std::uint64_t> values) override;
    void write(std::string_view name, std::span<const std::uint16_t> values) override;
    void write(std::string_view name, std::span<const std::uint8_t> values) override;

    void write_2d(std::string_view name, std::span<const float> values,
                  std::size_t rows, std::size_t cols) override;

    void write_attr(std::string_view name, std::string_view value) override;

    /// True while every group/dataset create and every H5Dwrite has succeeded.
    /// A durable writer MUST consult this after replaying the tree and before
    /// committing the file: a failed write (e.g. ENOSPC) otherwise passes
    /// silently and yields a truncated-but-renamed file. Sticky once tripped.
    [[nodiscard]] bool ok() const { return ok_; }

private:
    hdf5::file& file_;
    std::vector<hdf5::group> group_stack_;
    bool ok_ = true;

    hid_t current_loc() const;
    void record(herr_t rc) { if (rc < 0) ok_ = false; }
};

/// HDF5 implementation of state_reader
class hdf5_state_reader : public state_reader {
public:
    /// `format_version`: the version of the checkpoint the file holds (see
    /// state_reader::format_version).
    hdf5_state_reader(hdf5::file& file, std::uint32_t format_version);
    ~hdf5_state_reader() override;

    bool enter_group(std::string_view name) override;
    void leave_group() override;
    bool has_field(std::string_view name) override;

    double read_double(std::string_view name, double default_value) override;
    float read_float(std::string_view name, float default_value) override;
    std::int32_t read_int32(std::string_view name, std::int32_t default_value) override;
    std::int64_t read_int64(std::string_view name, std::int64_t default_value) override;
    std::uint32_t read_uint32(std::string_view name, std::uint32_t default_value) override;
    std::uint64_t read_uint64(std::string_view name, std::uint64_t default_value) override;
    bool read_bool(std::string_view name, bool default_value) override;

    std::vector<float> read_float_vec(std::string_view name) override;
    std::vector<double> read_double_vec(std::string_view name) override;
    std::vector<std::int32_t> read_int32_vec(std::string_view name) override;
    std::vector<std::uint32_t> read_uint32_vec(std::string_view name) override;
    std::vector<std::uint64_t> read_uint64_vec(std::string_view name) override;
    std::vector<std::uint16_t> read_uint16_vec(std::string_view name) override;
    std::vector<std::uint8_t> read_uint8_vec(std::string_view name) override;

    std::vector<float> read_float_2d(std::string_view name, array_2d_info& info) override;

    std::string read_attr(std::string_view name) override;

    std::vector<std::pair<std::string, scalar_value>> read_scalars() override;

private:
    hdf5::file& file_;
    std::vector<hdf5::group> group_stack_;

    hid_t current_loc() const;
};

/// Manages saving and loading simulation state.
///
/// All HDF5 work is funnelled through an hdf5::executor so that the
/// non-threadsafe libhdf5 build is touched from exactly one thread,
/// process-wide. The executor is passed in per call rather than held as
/// a member because state_logger is owned by logging_manager, which
/// already holds an executor reference.
class state_logger {
public:
    state_logger() = default;

    /// Save complete simulation state to file (extract_state + serialize_state
    /// with the write awaited, both on the caller's thread). Convenience for
    /// tests and non-threaded contexts; runtime call sites instead split the two
    /// halves across the sim thread and the executor via run_quiesced (see
    /// logging_manager).
    log_result save_state(sim::env&                env,
                          std::uint64_t            step_count,
                          const state_save_config& config,
                          hdf5::executor&        h5_exec);

    /// Load simulation state from file (monolithic; retained for build_env's
    /// startup-load exemption — design §5.2). Runtime load sites use the
    /// parse_state + apply_state split below.
    state_load_result load_state(sim::env&                    env,
                                 const std::filesystem::path& path,
                                 hdf5::executor&            h5_exec);

    // --- Quiesced-state split (design §4, §5): extract / serialize / parse /
    //     apply. Static and stateless — state_logger holds no per-call state.

    /// Sim-thread-only. Walk every io_buffer (snapshot() reads the read page for
    /// each variant, incl. on-demand GPU readback) and every entity
    /// (entity.save_state) into an in-memory tree, plus the env-derived metadata
    /// (schema_version, type, step_count). No HDF5. The save-event metadata
    /// (config, timestamp) is added by serialize_state — the extractor cannot
    /// know it.
    static mem_group extract_state(sim::env& env, std::uint64_t step_count);

    /// Executor-bound, async. Adds the save-event metadata, then submits a job
    /// that opens the file and replays the tree through hdf5_state_writer. The
    /// tree is moved into the job (ownership transfer; design §5.1) — the caller
    /// must not touch the moved-from tree. Returns the write future so the caller
    /// chooses whether to await durability (headless does; GUI may not).
    static std::future<log_result> serialize_state(mem_group                tree,
                                                    const state_save_config& config,
                                                    hdf5::executor&          h5_exec);

    /// Parsed state plus the metadata the caller needs to drive set_step_count.
    struct parse_result {
        state_load_result result;   // success / restored_step / config_json / warnings
        mem_group         tree;     // full parsed state tree
    };

    /// Executor-bound, blocking. Parse the HDF5 file into a tree and read its
    /// metadata (format version, step_count, config). HDF5 only here. Fails on a
    /// missing, malformed or newer format version (k_format_version).
    static parse_result parse_state(const std::filesystem::path& path,
                                    hdf5::executor&              h5_exec);

    /// Sim-thread-only. Restore each io_buffer's read page directly (no swap;
    /// design §5.3) and each entity (entity.load_state) from the tree. No HDF5,
    /// no swap. Returns restored_step (the call site applies set_step_count).
    /// Entities read the tree's format version from the reader.
    static state_load_result apply_state(sim::env& env, const mem_group& tree);
};

} // namespace logging

#endif // COAST_STATE_LOGGER_H
