//
// In-memory state tree + adapters on the abstract state_io interfaces.
//
// The simulation's full-state save/load threads through the abstract
// state_writer / state_reader interfaces (state_io.h). Today the only
// implementations write/read libhdf5 directly (hdf5_state_writer /
// hdf5_state_reader). The quiesced-state-access redesign needs to split the
// *extraction* of sim state (which must happen on the sim thread, touching the
// wgpu queue) from its *serialization* to disk (heavy libhdf5, kept on the
// hdf5::executor worker). The bridge between the two halves is an in-memory
// value tree:
//
//   - mem_state_writer : state_writer  -- entities/buffers write into a tree
//                                          (no HDF5; fast; sim-thread-safe)
//   - mem_state_reader : state_reader  -- entities/buffers read from a tree
//   - replay(tree, state_writer&)      -- drive *any* writer from a tree;
//                                          tree -> hdf5_state_writer reproduces
//                                          the exact on-disk structure by
//                                          construction (same call sequence)
//   - parse_hdf5_to_tree(file)         -- generic HDF5 -> tree (the load half)
//
// The tree carries the full state_io type set (f32/f64/i32/i64/u32/u64/bool
// scalars; f32/f64/i32/u32/u64/u16/u8 vectors; 2D-f32; string attrs) so a
// round-trip through it is lossless and type-preserving.
//
// Threading: the tree types and replay() touch no libhdf5 and are safe on any
// thread. parse_hdf5_to_tree() and replay-into-hdf5_state_writer DO call
// libhdf5 and must run on the hdf5::executor worker (see hdf5_utils.h).
//

#ifndef COAST_MEM_STATE_IO_H
#define COAST_MEM_STATE_IO_H

#include "logging/state_io.h"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace hdf5 { class file; }

namespace logging {

/// A 2D float field (the only 2D shape in the state_io interface).
struct mem_array2d {
    std::vector<float> data;
    std::size_t        rows = 0;
    std::size_t        cols = 0;

    friend bool operator==(const mem_array2d&, const mem_array2d&) = default;
};

/// One typed field value. The active alternative encodes both the element type
/// and the shape (scalar vs. vector vs. 2D), mirroring the state_io type set.
/// std::variant's type-sensitive operator== gives us exact structural equality
/// (different alternative => not equal), which is what the equivalence gate
/// needs.
using mem_value = std::variant<
    double, float, std::int32_t, std::int64_t, std::uint32_t, std::uint64_t, bool,
    std::vector<float>, std::vector<double>, std::vector<std::int32_t>,
    std::vector<std::uint32_t>, std::vector<std::uint64_t>,
    std::vector<std::uint16_t>, std::vector<std::uint8_t>,
    mem_array2d>;

/// A node in the state tree: named datasets (fields), named string attributes,
/// and named child groups. Keyed by name with a transparent comparator (so
/// lookups take a string_view without allocating). std::map is node-based, so
/// element addresses are stable across insertion — mem_state_writer relies on
/// this to keep a pointer stack into the tree it is building.
///
/// Maps are sorted by name. HDF5's name index (H5_INDEX_NAME) is also
/// lexicographic, so a parsed tree and a freshly-built tree compare equal
/// regardless of original write order.
struct mem_group {
    std::map<std::string, mem_value,  std::less<>> fields;
    std::map<std::string, std::string, std::less<>> attrs;
    std::map<std::string, mem_group,  std::less<>> groups;

    friend bool operator==(const mem_group&, const mem_group&) = default;
};

/// state_writer that builds an in-memory mem_group tree instead of writing HDF5.
class mem_state_writer final : public state_writer {
public:
    mem_state_writer() { stack_.push_back(&root_); }

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

    /// The built tree (valid for the writer's lifetime).
    [[nodiscard]] const mem_group& root() const { return root_; }

    /// Move the built tree out (the writer is empty afterwards).
    [[nodiscard]] mem_group take_root() { return std::move(root_); }

private:
    mem_group               root_;
    std::vector<mem_group*> stack_;   // current group path; back() == cursor

    mem_group& cur() { return *stack_.back(); }
};

/// state_reader that navigates an in-memory mem_group tree. Borrows the tree;
/// the tree must outlive the reader. Like hdf5_state_reader, a read of a field
/// that exists with another rank (a vector read as a scalar, ...) or that names
/// a group throws std::runtime_error naming the field.
class mem_state_reader final : public state_reader {
public:
    /// `format_version`: the version of the checkpoint the tree holds (see
    /// state_reader::format_version).
    mem_state_reader(const mem_group& root, std::uint32_t format_version)
        : state_reader(format_version) { stack_.push_back({&root, ""}); }

    bool enter_group(std::string_view name) override;
    void leave_group() override;
    bool has_field(std::string_view name) override;

    double       read_double(std::string_view name, double default_value) override;
    float        read_float(std::string_view name, float default_value) override;
    std::int32_t read_int32(std::string_view name, std::int32_t default_value) override;
    std::int64_t read_int64(std::string_view name, std::int64_t default_value) override;
    std::uint32_t read_uint32(std::string_view name, std::uint32_t default_value) override;
    std::uint64_t read_uint64(std::string_view name, std::uint64_t default_value) override;
    bool         read_bool(std::string_view name, bool default_value) override;

    std::vector<float>         read_float_vec(std::string_view name) override;
    std::vector<double>        read_double_vec(std::string_view name) override;
    std::vector<std::int32_t>  read_int32_vec(std::string_view name) override;
    std::vector<std::uint32_t> read_uint32_vec(std::string_view name) override;
    std::vector<std::uint64_t> read_uint64_vec(std::string_view name) override;
    std::vector<std::uint16_t> read_uint16_vec(std::string_view name) override;
    std::vector<std::uint8_t>  read_uint8_vec(std::string_view name) override;

    std::vector<float> read_float_2d(std::string_view name, array_2d_info& info) override;

    std::string read_attr(std::string_view name) override;

    std::vector<std::pair<std::string, scalar_value>> read_scalars() override;

private:
    struct level {
        const mem_group* group;
        std::string      path;   // "/a/b" for messages; "" at the root
    };
    std::vector<level> stack_;   // current group path; back() == cursor

    [[nodiscard]] const mem_group&  cur() const { return *stack_.back().group; }
    [[nodiscard]] const mem_value*  find_field(std::string_view name) const;
    [[noreturn]] void wrong_rank(std::string_view name, int expected, const mem_value& v) const;
    template <class T> T              scalar(std::string_view name, T default_value) const;
    template <class T> std::vector<T> vec(std::string_view name) const;
};

/// Replay a tree into any state_writer, reproducing the same sequence of
/// write/begin_group calls. Driving an hdf5_state_writer with this yields the
/// same on-disk structure the entities would have written inline (structural /
/// coast_compare identity, not byte identity). Pure replay — no HDF5 here; the
/// caller chooses the writer (and thus the threading).
void replay(const mem_group& tree, state_writer& writer);

/// Parse an open HDF5 file generically into a tree, preserving element type and
/// shape. Throws std::runtime_error, naming the object, if the file holds
/// anything a state writer never produces — read by a load or not. Calls
/// libhdf5 directly — invoke only on the hdf5::executor worker.
[[nodiscard]] mem_group parse_hdf5_to_tree(hdf5::file& file);

} // namespace logging

#endif // COAST_MEM_STATE_IO_H
