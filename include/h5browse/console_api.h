#ifndef COAST_CONSOLE_API_H
#define COAST_CONSOLE_API_H

// Programmatic, range-style access to an open COAST HDF5 log file — the surface
// an embedded C++ interpreter reaches through `open()` (design/console_api_design.md).
//
// This header is the PARSE BOUNDARY: an interpreter parses it, so it is kept
// deliberately lean. All coast internals (executor, file_model, chunk_cache,
// dataset_desc, row_block) are forward-declared and held by pointer; the only
// coast include is the tiny, pure scalar_type.h. Implementations live in
// console_api.cpp, compiled by the host and reached via the JIT symbol table.
//
// Error model: failures (unknown dataset, read failure, out-of-range column,
// dtype mismatch) throw std::runtime_error. Standard exception types are used
// deliberately — their typeinfo already lives in libstdc++, so a compiled throw
// is catchable across the interpreter boundary without extra symbol export.
//
// Access modes mirror the two read paths: scan() streams (read_pipeline, 4 MB
// prefetched batches); row(i) is random (read_block + chunk_cache). scan() rows
// are single-pass and valid only until the iterator advances; row(i) rows pin
// their cache block and stay valid while held.
//
// Both modes hand out row_views that own nothing: row_view::as<T>() is a span
// into the source block and never outlives its row_view. When a row's values
// must outlive the row_view, use row_view::as_copy<T>(), which returns a vector.

#include "h5browse/scalar_type.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace hdf5 {
class executor;
}

namespace h5b {

struct file_model;
struct dataset_desc;
class chunk_cache;

namespace console {

struct range_state; // pimpl for row_range iteration (defined in the .cpp)

// Maps a C++ scalar type to its scalar_type tag (for row_view::as<T>()).
template <class T>
constexpr scalar_type scalar_type_of()
{
    if constexpr (std::is_same_v<T, float>) return scalar_type::f32;
    else if constexpr (std::is_same_v<T, double>) return scalar_type::f64;
    else if constexpr (std::is_same_v<T, std::int8_t>) return scalar_type::i8;
    else if constexpr (std::is_same_v<T, std::int16_t>) return scalar_type::i16;
    else if constexpr (std::is_same_v<T, std::int32_t>) return scalar_type::i32;
    else if constexpr (std::is_same_v<T, std::int64_t>) return scalar_type::i64;
    else if constexpr (std::is_same_v<T, std::uint8_t>) return scalar_type::u8;
    else if constexpr (std::is_same_v<T, std::uint16_t>) return scalar_type::u16;
    else if constexpr (std::is_same_v<T, std::uint32_t>) return scalar_type::u32;
    else if constexpr (std::is_same_v<T, std::uint64_t>) return scalar_type::u64;
    else static_assert(sizeof(T) == 0, "as<T>(): unsupported scalar type");
}

// A single row's native-dtype bytes, row-major. Non-owning for streaming
// (valid until the range advances); `keep_` pins the source block for random
// access so the view survives cache eviction while held.
class row_view {
public:
    row_view() = default;

    [[nodiscard]] std::uint64_t index() const { return index_; }
    [[nodiscard]] std::uint64_t width() const { return width_; }
    [[nodiscard]] scalar_type   dtype() const { return dtype_; }
    [[nodiscard]] std::span<const std::uint64_t> shape() const
    {
        return shape_ ? std::span<const std::uint64_t>(*shape_)
                      : std::span<const std::uint64_t>{};
    }

    // Column value widened to double — always valid for any stored dtype.
    [[nodiscard]] double at(std::uint64_t col) const;

    // Zero-copy typed view of the row; throws if T does not match the stored
    // dtype.
    //
    // LIFETIME: the span points into this row_view's source block and is valid
    // only while *this* row_view is alive. It does NOT extend it. The trap:
    //
    //     std::span<const float> ref = ds.row(i).as<float>();  // WRONG: dangles
    //
    // The temporary row_view dies at the semicolon, dropping the keep_ that pins
    // the block; the span then survives only until the LRU evicts it, after which
    // it silently reads whatever reuses the memory. Safe only when the row_view
    // outlives the span (a named row_view, or a by-const& parameter). If the
    // values must outlive the row_view at all, use as_copy<T>().
    template <class T>
    [[nodiscard]] std::span<const T> as() const
    {
        if (scalar_type_of<T>() != dtype_) {
            throw std::runtime_error("row_view::as<T>(): dtype mismatch");
        }
        return std::span<const T>(reinterpret_cast<const T*>(data_), width_);
    }

    // Owning typed copy of the row; throws if T does not match the stored dtype.
    // Free of as<T>()'s lifetime coupling — `auto ref = ds.row(i).as_copy<float>();`
    // is correct — at the cost of one row-sized copy. Prefer it for any value held
    // across further reads (e.g. a fixed reference vector compared against a scan);
    // prefer as<T>() for transient use inside a single expression.
    template <class T>
    [[nodiscard]] std::vector<T> as_copy() const
    {
        const std::span<const T> r = as<T>();
        return {r.begin(), r.end()};
    }

    [[nodiscard]] std::span<const std::byte> bytes() const
    {
        return {data_, width_ * scalar_size(dtype_)};
    }

    [[nodiscard]] bool valid() const { return data_ != nullptr; }

private:
    friend class dataset_handle;
    friend struct range_state;

    row_view(const std::byte* data, std::uint64_t index, std::uint64_t width,
             scalar_type dt, const std::vector<std::uint64_t>* shape,
             std::shared_ptr<const void> keep)
        : data_(data), index_(index), width_(width), dtype_(dt), shape_(shape),
          keep_(std::move(keep))
    {
    }

    const std::byte*                   data_  = nullptr;
    std::uint64_t                      index_ = 0;
    std::uint64_t                      width_ = 1;
    scalar_type                        dtype_ = scalar_type::f64;
    const std::vector<std::uint64_t>*  shape_ = nullptr; // into dataset_desc
    std::shared_ptr<const void>        keep_;             // random-access keepalive
};

class file_console; // fwd

// Single-pass input range over a dataset's rows, backed by read_pipeline.
class row_range {
public:
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type        = row_view;
        using difference_type    = std::ptrdiff_t;

        row_view   operator*() const;
        iterator&  operator++();
        void       operator++(int) { ++*this; }
        bool operator==(std::default_sentinel_t) const;
        bool operator!=(std::default_sentinel_t s) const { return !(*this == s); }

    private:
        friend class row_range;
        std::shared_ptr<range_state> st_;
    };

    [[nodiscard]] iterator                begin() const;
    [[nodiscard]] std::default_sentinel_t end() const { return {}; }

private:
    friend class dataset_handle;
    const file_console* fc_       = nullptr;
    std::uint32_t       dataset_  = 0;
    std::uint64_t       first_    = 0;
    std::uint64_t       n_        = 0;
};

// Lightweight handle to one dataset in the open file. Valid while its
// file_console lives.
class dataset_handle {
public:
    [[nodiscard]] std::string_view name()  const;
    [[nodiscard]] std::uint64_t     rows()  const;
    [[nodiscard]] std::uint64_t     width() const;
    [[nodiscard]] scalar_type       dtype() const;

    [[nodiscard]] row_range scan() const;                                   // whole dataset
    [[nodiscard]] row_range scan(std::uint64_t first, std::uint64_t n) const;
    [[nodiscard]] row_view  row(std::uint64_t i) const;                     // random access

    [[nodiscard]] std::vector<double>        column(std::uint64_t col) const;
    [[nodiscard]] std::vector<std::uint64_t> steps() const; // empty if none

private:
    friend class file_console;
    const file_console* fc_      = nullptr;
    std::uint32_t       dataset_ = 0;
};

// The object `open()` hands back (as a shared_ptr) to the interpreter. Owns a
// private read-only HDF5 handle and an LRU chunk cache; shares the process-wide
// hdf5::executor. Non-copyable and non-movable (owns an HDF5 handle + a
// mutex-bearing cache).
class file_console {
public:
    file_console(hdf5::executor& exec, const std::filesystem::path& path); // throws on open failure
    ~file_console();

    file_console(const file_console&)            = delete;
    file_console& operator=(const file_console&) = delete;

    [[nodiscard]] std::vector<std::string> list()   const; // dataset display names
    [[nodiscard]] std::size_t              count()  const;
    [[nodiscard]] std::string_view         family() const;
    [[nodiscard]] const std::filesystem::path& path() const;

    [[nodiscard]] dataset_handle dataset(std::string_view name)  const; // throws if unknown
    [[nodiscard]] dataset_handle dataset(std::uint32_t index)    const; // throws if OOR

private:
    friend class dataset_handle;
    friend class row_range;
    friend struct range_state;

    [[nodiscard]] const dataset_desc& desc(std::uint32_t index) const;

    hdf5::executor*                   exec_;
    std::shared_ptr<const file_model> model_;
    std::int64_t                      file_ = -1;
    std::unique_ptr<chunk_cache>      cache_;
};

// ------------------------------------------------------------------ open() façade
//
// The console opens files through the free open() below rather than binding one
// current file: it returns a shared handle, so a session can hold several files
// at once, each in its own variable. open() needs the process-wide HDF5
// executor (see the file_console ctor); the host registers it once at startup
// via set_executor() — mirroring coast::plot::set_executor. Both live in
// coast_lib and are reached from the interpreter via the process symbol table.

// Register the single process-wide HDF5 executor open() constructs file_consoles
// on. Must be called once, before the first open(), by the host that owns the
// executor. The executor must outlive every file_console open() hands out.
void set_executor(hdf5::executor& exec);

// Un-register `exec` when its owner tears down, so a later open() fails cleanly
// ("no executor registered") instead of dereferencing a dangling pointer. No-op
// if a different executor is currently registered (compare-and-clear), so a
// second registrant is never wiped by the first's teardown. Pair with
// set_executor() in the owner's ctor/dtor.
void clear_executor(hdf5::executor& exec);

// Open an HDF5 log file for console access and return a shared handle. Multiple
// files (including the same path) may be open at once — each handle owns a
// private read-only HDF5 handle over the shared executor. Throws
// std::runtime_error on open failure, or if set_executor() was never called.
[[nodiscard]] std::shared_ptr<file_console> open(const std::filesystem::path& path);

// A default C++ identifier for the variable a file is opened into, derived from
// the path stem (e.g. "run/mnist_seed0.h5" -> "f_mnist_seed0"). Deterministic
// and dependency-free so the console panel and the HDF5 browser agree on it;
// the caller disambiguates collisions (two files with the same stem).
[[nodiscard]] std::string var_name_for(const std::filesystem::path& path);

} // namespace console
} // namespace h5b

#endif // COAST_CONSOLE_API_H
