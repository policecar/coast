//
// State I/O interfaces for simulation state persistence
//

#ifndef COAST_STATE_IO_H
#define COAST_STATE_IO_H

#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <cstdint>

namespace logging {

/// Interface for writing hierarchical state data
class state_writer {
public:
    virtual ~state_writer() = default;

    /// Begin a named group (can be nested)
    virtual void begin_group(std::string_view name) = 0;

    /// End the current group
    virtual void end_group() = 0;

    /// Write scalar values
    virtual void write(std::string_view name, double value) = 0;
    virtual void write(std::string_view name, float value) = 0;
    virtual void write(std::string_view name, std::int32_t value) = 0;
    virtual void write(std::string_view name, std::int64_t value) = 0;
    virtual void write(std::string_view name, std::uint32_t value) = 0;
    virtual void write(std::string_view name, std::uint64_t value) = 0;
    virtual void write(std::string_view name, bool value) = 0;

    /// Write array values
    virtual void write(std::string_view name, std::span<const float> values) = 0;
    virtual void write(std::string_view name, std::span<const double> values) = 0;
    virtual void write(std::string_view name, std::span<const std::int32_t> values) = 0;
    virtual void write(std::string_view name, std::span<const std::uint32_t> values) = 0;
    virtual void write(std::string_view name, std::span<const std::uint16_t> values) = 0;
    virtual void write(std::string_view name, std::span<const std::uint8_t> values) = 0;

    /// Write 2D array values
    virtual void write_2d(std::string_view name, std::span<const float> values,
                          std::size_t rows, std::size_t cols) = 0;

    /// Write string attribute (for metadata)
    virtual void write_attr(std::string_view name, std::string_view value) = 0;
};

/// Interface for reading hierarchical state data
class state_reader {
public:
    virtual ~state_reader() = default;

    /// Enter a named group (returns false if group doesn't exist)
    virtual bool enter_group(std::string_view name) = 0;

    /// Leave the current group
    virtual void leave_group() = 0;

    /// Check if a field exists in current group
    virtual bool has_field(std::string_view name) = 0;

    /// Read scalar values (returns default if not found)
    virtual double read_double(std::string_view name, double default_value = 0.0) = 0;
    virtual float read_float(std::string_view name, float default_value = 0.0f) = 0;
    virtual std::int32_t read_int32(std::string_view name, std::int32_t default_value = 0) = 0;
    virtual std::int64_t read_int64(std::string_view name, std::int64_t default_value = 0) = 0;
    virtual std::uint32_t read_uint32(std::string_view name, std::uint32_t default_value = 0) = 0;
    virtual std::uint64_t read_uint64(std::string_view name, std::uint64_t default_value = 0) = 0;
    virtual bool read_bool(std::string_view name, bool default_value = false) = 0;

    /// Read array values
    virtual std::vector<float> read_float_vec(std::string_view name) = 0;
    virtual std::vector<double> read_double_vec(std::string_view name) = 0;
    virtual std::vector<std::int32_t> read_int32_vec(std::string_view name) = 0;
    virtual std::vector<std::uint32_t> read_uint32_vec(std::string_view name) = 0;
    virtual std::vector<std::uint16_t> read_uint16_vec(std::string_view name) = 0;
    virtual std::vector<std::uint8_t> read_uint8_vec(std::string_view name) = 0;

    /// Read 2D array values
    struct array_2d_info {
        std::size_t rows = 0;
        std::size_t cols = 0;
    };
    virtual std::vector<float> read_float_2d(std::string_view name, array_2d_info& info) = 0;

    /// Read string attribute (returns empty string if not found)
    virtual std::string read_attr(std::string_view name) = 0;
};

/// Interface for entities that can save/restore state
class state_serializable {
public:
    virtual ~state_serializable() = default;

    /// Save entity state to writer
    virtual void save_state(state_writer& writer) const = 0;

    /// Load entity state from reader
    virtual void load_state(state_reader& reader) = 0;
};

} // namespace logging

#endif // COAST_STATE_IO_H
