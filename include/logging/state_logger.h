//
// State Logger - full simulation state persistence
//

#ifndef COAST_STATE_LOGGER_H
#define COAST_STATE_LOGGER_H

#include "logging_common.h"
#include "state_io.h"
#include "hdf5_utils.h"

#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>

namespace sim {
class env;
}

namespace logging {

/// Configuration for state saving
struct state_save_config {
    std::filesystem::path output_path;
    compression_config compression;
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
    void write(std::string_view name, std::span<const std::uint16_t> values) override;
    void write(std::string_view name, std::span<const std::uint8_t> values) override;

    void write_2d(std::string_view name, std::span<const float> values,
                  std::size_t rows, std::size_t cols) override;

    void write_attr(std::string_view name, std::string_view value) override;

private:
    hdf5::file& file_;
    std::vector<hdf5::group> group_stack_;

    hid_t current_loc() const;
};

/// HDF5 implementation of state_reader
class hdf5_state_reader : public state_reader {
public:
    explicit hdf5_state_reader(hdf5::file& file);
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
    std::vector<std::uint16_t> read_uint16_vec(std::string_view name) override;
    std::vector<std::uint8_t> read_uint8_vec(std::string_view name) override;

    std::vector<float> read_float_2d(std::string_view name, array_2d_info& info) override;

    std::string read_attr(std::string_view name) override;

private:
    hdf5::file& file_;
    std::vector<hdf5::group> group_stack_;

    hid_t current_loc() const;
};

/// Manages saving and loading simulation state
class state_logger {
public:
    state_logger() = default;

    /// Save complete simulation state to file
    log_result save_state(sim::env& env, std::uint64_t step_count,
                          const state_save_config& config);

    /// Load simulation state from file
    state_load_result load_state(sim::env& env, const std::filesystem::path& path);

private:
    static constexpr std::string_view SCHEMA_VERSION = "1.0";
};

} // namespace logging

#endif // COAST_STATE_LOGGER_H
