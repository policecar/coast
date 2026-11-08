//
// State I/O interfaces for simulation state persistence
//

#ifndef COAST_STATE_IO_H
#define COAST_STATE_IO_H

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>
#include <span>
#include <cstdint>

namespace logging {

/// The checkpoint format version this build writes, recorded in every
/// checkpoint's metadata (`metadata/schema_version`). Bump it with every change
/// to what a checkpoint holds that a load must tell apart, and gate the check
/// that needs the newer format on state_reader::format_version(), so older
/// checkpoints still load: e.g. `if (r.format_version() >= 2) check the field`.
/// A load refuses a checkpoint whose version is newer than this one.
///
///   1  every checkpoint up to and including this constant's introduction
///      (written as "1.0" before the version was an integer)
///   2  every model group (ngm2, ngm25, ngm3; CPU and GPU) carries its activity
///      window edges, `activity_window_tracking/` (ngm/window_tracking.h)
///   3  ngm2 lays a dendrite's input channels out in declared order (SPEC.md
///      M.7.5); it refuses older checkpoints of multi-channel dendrites,
///      whose channel order was not recorded.
///   4  each group records its channels' names and widths in order (channel
///      record, config_record.h); a load refuses a record that differs. ngm25
///      also records its apical/proximal dendrite split (apical_cnt,
///      proximal_cnt), checked alongside its other structural fields.
///   5  ngm2 records each dendrite's type and tree depth (`apical`,
///      `max_branch_level`), checked before anything is assigned (SPEC.md
///      M.7.18); an older checkpoint's depths are checked against its
///      configuration record, its types not at all (one notice). ngm2 also
///      writes the group's `effective_seed`, which a load restores with the rng
///      (read by presence, not version: an older checkpoint falls back to its
///      configuration record).
///   6  metadata/run_id names the run that wrote the checkpoint (core/run_id.h);
///      a dataset restores its sequence only when it equals the loading run's
///      (io/dataset_run.h, SPEC.md S.5.14), and records its resolved params,
///      checked when it restores. An older checkpoint names no run: its
///      sequences start fresh.
inline constexpr std::uint32_t k_format_version = 6;

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
    virtual void write(std::string_view name, std::span<const std::uint64_t> values) = 0;
    virtual void write(std::string_view name, std::span<const std::uint16_t> values) = 0;
    virtual void write(std::string_view name, std::span<const std::uint8_t> values) = 0;

    /// Write 2D array values
    virtual void write_2d(std::string_view name, std::span<const float> values,
                          std::size_t rows, std::size_t cols) = 0;

    /// Write string attribute (for metadata)
    virtual void write_attr(std::string_view name, std::string_view value) = 0;
};

/// A scalar field or string attribute, as the type it was written with.
using scalar_value = std::variant<double, float, std::int32_t, std::int64_t,
                                  std::uint32_t, std::uint64_t, bool, std::string>;

/// Interface for reading hierarchical state data
class state_reader {
public:
    virtual ~state_reader() = default;

    /// The format version of the checkpoint being read (see k_format_version).
    /// Every reader is built with it, so no load path can leave it unset: a load
    /// reads it from the checkpoint's metadata first (state_logger), and a reader
    /// of state this build just wrote passes k_format_version.
    [[nodiscard]] std::uint32_t format_version() const { return format_version_; }

    /// Whether the checkpoint was written by the run loading it (equal run_id,
    /// metadata/run_id): then its datasets continue their sequences (SPEC.md S.5.14).
    /// A load sets it before any entity loads (state_logger::apply_state); false for
    /// a checkpoint older than format version 6, and for a reader built by hand that
    /// nobody set it on.
    [[nodiscard]] bool continues_run() const { return continues_run_; }
    void set_continues_run(bool continues) { continues_run_ = continues; }

    /// Every read below may throw std::runtime_error naming the field: "not
    /// found" (the default, empty, or "") means absent, not malformed. Both
    /// implementations throw when a field exists with another rank than the
    /// read's (rank 0 scalars, rank 1 arrays, rank 2 for read_float_2d) or is a
    /// group, and the HDF5 reader also when a value does not convert. A file
    /// holding anything a state writer never produces is refused before any
    /// read (parse_hdf5_to_tree, mem_state_io.h). A load reports a throw as its
    /// error.

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

    /// Read array values (empty if not found)
    virtual std::vector<float> read_float_vec(std::string_view name) = 0;
    virtual std::vector<double> read_double_vec(std::string_view name) = 0;
    virtual std::vector<std::int32_t> read_int32_vec(std::string_view name) = 0;
    virtual std::vector<std::uint32_t> read_uint32_vec(std::string_view name) = 0;
    virtual std::vector<std::uint64_t> read_uint64_vec(std::string_view name) = 0;
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

    /// Every scalar field and string attribute of the current group, by name
    /// (fields first, then attributes), each as the type it was written with.
    /// Arrays and subgroups are left out; a malformed entry throws, as above. For a reader that must see entries it
    /// cannot name in advance, e.g. a configuration record's.
    virtual std::vector<std::pair<std::string, scalar_value>> read_scalars() = 0;

    /// The groups read so far that had no configuration record (config_record.h),
    /// in load order. check_config_record notes each one here instead of logging
    /// it, so the load that owns this reader reports them all in one notice
    /// (report_unrecorded_configs). Taking them empties the list.
    void note_unrecorded_config(std::string_view who) { unrecorded_configs_.emplace_back(who); }
    [[nodiscard]] std::vector<std::string> take_unrecorded_configs()
    {
        return std::exchange(unrecorded_configs_, {});
    }

    /// Likewise the groups whose checkpoint predates the channel record
    /// (config_record.h), so their channel order was not checked; reported in
    /// one notice by report_unrecorded_channels.
    void note_unrecorded_channels(std::string_view who) { unrecorded_channels_.emplace_back(who); }
    [[nodiscard]] std::vector<std::string> take_unrecorded_channels()
    {
        return std::exchange(unrecorded_channels_, {});
    }

    /// Likewise the ngm2 groups whose checkpoint predates format version 5, which
    /// records each dendrite's type (apical/proximal) and tree depth, so their
    /// types were not checked; reported in one notice by
    /// report_unchecked_dendrite_types (state_checks.h). `what` names the group,
    /// and any dendrite cfg whose depth could not be checked either.
    void note_unchecked_dendrite_types(std::string_view what) { unchecked_dendrite_types_.emplace_back(what); }
    [[nodiscard]] std::vector<std::string> take_unchecked_dendrite_types()
    {
        return std::exchange(unchecked_dendrite_types_, {});
    }

protected:
    explicit state_reader(std::uint32_t format_version) : format_version_(format_version) {}

private:
    const std::uint32_t      format_version_;
    bool                     continues_run_ = false;
    std::vector<std::string> unrecorded_configs_;
    std::vector<std::string> unrecorded_channels_;
    std::vector<std::string> unchecked_dendrite_types_;
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
