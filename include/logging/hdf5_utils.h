//
// RAII wrappers and utilities for HDF5 operations.
//
// ============================================================================
// THREADING CONTRACT — READ BEFORE USING ANY SYMBOL IN THIS HEADER
// ============================================================================
//
// libhdf5 is built without --enable-threadsafe: exactly one thread may be
// inside the library at any time, process-wide. The project funnels all
// libhdf5 access through the single worker thread owned by `hdf5::executor`
// (see include/hdf5/executor.h).
//
// EVERY constructor, method, and free function declared in this header
// calls libhdf5 directly. None of them check, lock, or delegate. They are
// therefore ONLY safe to invoke from inside an `hdf5::executor::submit(...)`
// or `submit_sync(...)` lambda. Calling them from any other thread races
// with the executor worker and corrupts libhdf5's global ID table / error
// stack / init path. Symptoms range from silent data corruption ("weird
// looking" log output) to SIGSEGV.
//
// The RAII wrappers here let a caller batch many libhdf5 calls into a
// single submit_sync lambda — e.g. the entire body of
// state_logger::save_state or buffer_logger::write_frames — amortizing
// the queueing cost.
//
// Two RAII layers, one rule. Both close on the executor thread; they differ
// in who gets it there:
//   - these wrappers close directly, so their destructor and move-assignment
//     must themselves run inside a job. A wrapper kept as a member of a
//     longer-lived object (buffer_logger, buffer_log_reader) is reset inside
//     a submit_sync by its owner, and the owner's move-assignment is deleted
//     or does the same — a defaulted one would close on the caller's thread.
//   - hdf5/handle.h POSTS the close to the executor, so its handle may die on
//     any thread. Use it for a handle that outlives one job and is not owned
//     by an object that tears down in a job (observe's per-site datasets).
// They are kept apart on purpose: posting every close would make each
// batched job's teardown asynchronous, and the writers here must check the
// H5Fclose status (file::close) synchronously.
//
// Pattern:
//
//     h5_exec.submit_sync([&] {
//         auto file = hdf5::file::create(path);     // direct H5Fcreate
//         auto grp  = hdf5::group::create(...);     // direct H5Gcreate2
//         auto ds   = hdf5::dataset::create(...);   // direct H5Dcreate2
//         ds.write(...);                            // direct H5Dwrite
//         // ... many operations, one queue round-trip ...
//     });
//
// If you find yourself wanting to call any of these from outside a
// submit lambda — don't. Wrap your block in submit_sync.
// ============================================================================

#ifndef COAST_HDF5_UTILS_H
#define COAST_HDF5_UTILS_H

#include <hdf5.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>

namespace hdf5 {

/// RAII wrapper for HDF5 file handles
class file {
    hid_t id_ = -1;

public:
    file() = default;

    /// Open existing file for reading
    static file open_read(const std::string& path) {
        file f;
        f.id_ = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        return f;
    }

    /// Open existing file for read/write
    static file open_readwrite(const std::string& path) {
        file f;
        f.id_ = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        return f;
    }

    /// Create new file (truncates if exists). Ensures the parent directory
    /// exists first: H5Fcreate opens the file via the OS and does NOT create
    /// intermediate directories, so a path into a not-yet-created output dir
    /// (e.g. a config-relative <base>/logs/ or <base>/snapshots/ leaf that no
    /// prior run materialized) would otherwise fail. Best-effort: if the mkdir
    /// fails, H5Fcreate is still attempted and reports the error as before.
    static file create(const std::string& path) {
        if (const auto parent = std::filesystem::path(path).parent_path();
            !parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
        }
        file f;
        f.id_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        return f;
    }

    ~file() { if (id_ >= 0) H5Fclose(id_); }

    /// Explicitly close the file, returning the H5Fclose status. On some
    /// filesystems H5Fclose surfaces a deferred write error (e.g. ENOSPC/EIO
    /// buffered from an earlier H5Dwrite) that the individual writes did not
    /// see, so a durable writer must close explicitly and check this rather
    /// than leaving it to the destructor (which discards the status). Idempotent:
    /// after this the destructor is a no-op.
    [[nodiscard]] herr_t close() {
        herr_t rc = 0;
        if (id_ >= 0) { rc = H5Fclose(id_); id_ = -1; }
        return rc;
    }

    file(const file&) = delete;
    file& operator=(const file&) = delete;
    file(file&& other) noexcept : id_(other.id_) { other.id_ = -1; }
    file& operator=(file&& other) noexcept {
        if (this != &other) {
            if (id_ >= 0) H5Fclose(id_);
            id_ = other.id_;
            other.id_ = -1;
        }
        return *this;
    }

    [[nodiscard]] hid_t id() const { return id_; }
    [[nodiscard]] explicit operator bool() const { return id_ >= 0; }

    [[nodiscard]] herr_t flush() { return id_ >= 0 ? H5Fflush(id_, H5F_SCOPE_GLOBAL) : 0; }
};

/// RAII wrapper for HDF5 group handles
class group {
    hid_t id_ = -1;

public:
    group() = default;

    /// Create a new group
    static group create(hid_t parent, const char* name) {
        group g;
        g.id_ = H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        return g;
    }

    /// Open existing group
    static group open(hid_t parent, const char* name) {
        group g;
        g.id_ = H5Gopen2(parent, name, H5P_DEFAULT);
        return g;
    }

    /// Create or open group (creates path recursively)
    static group create_or_open(hid_t parent, const char* name);

    ~group() { if (id_ >= 0) H5Gclose(id_); }

    group(const group&) = delete;
    group& operator=(const group&) = delete;
    group(group&& other) noexcept : id_(other.id_) { other.id_ = -1; }
    group& operator=(group&& other) noexcept {
        if (this != &other) {
            if (id_ >= 0) H5Gclose(id_);
            id_ = other.id_;
            other.id_ = -1;
        }
        return *this;
    }

    [[nodiscard]] hid_t id() const { return id_; }
    [[nodiscard]] explicit operator bool() const { return id_ >= 0; }
};

/// RAII wrapper for HDF5 dataspace handles
class dataspace {
    hid_t id_ = -1;

public:
    dataspace() = default;

    /// Create scalar dataspace
    static dataspace scalar() {
        dataspace ds;
        ds.id_ = H5Screate(H5S_SCALAR);
        return ds;
    }

    /// Create simple dataspace with fixed dimensions
    static dataspace simple(std::span<const hsize_t> dims) {
        dataspace ds;
        ds.id_ = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
        return ds;
    }

    /// Create extensible dataspace (unlimited in first dimension)
    static dataspace extensible(std::span<const hsize_t> initial_dims) {
        dataspace ds;
        std::vector<hsize_t> max_dims(initial_dims.size(), H5S_UNLIMITED);
        ds.id_ = H5Screate_simple(static_cast<int>(initial_dims.size()),
                                   initial_dims.data(), max_dims.data());
        return ds;
    }

    /// Get dataspace from dataset
    static dataspace from_dataset(hid_t dataset) {
        dataspace ds;
        ds.id_ = H5Dget_space(dataset);
        return ds;
    }

    ~dataspace() { if (id_ >= 0) H5Sclose(id_); }

    dataspace(const dataspace&) = delete;
    dataspace& operator=(const dataspace&) = delete;
    dataspace(dataspace&& other) noexcept : id_(other.id_) { other.id_ = -1; }
    dataspace& operator=(dataspace&& other) noexcept {
        if (this != &other) {
            if (id_ >= 0) H5Sclose(id_);
            id_ = other.id_;
            other.id_ = -1;
        }
        return *this;
    }

    [[nodiscard]] hid_t id() const { return id_; }
    [[nodiscard]] explicit operator bool() const { return id_ >= 0; }

    /// Get current dimensions
    [[nodiscard]] std::vector<hsize_t> dims() const {
        if (id_ < 0) return {};
        int ndims = H5Sget_simple_extent_ndims(id_);
        if (ndims <= 0) return {};
        std::vector<hsize_t> d(static_cast<std::size_t>(ndims));
        H5Sget_simple_extent_dims(id_, d.data(), nullptr);
        return d;
    }

    /// Select hyperslab for partial I/O. HDF5 reads one offset and one count per
    /// dimension of the dataspace, so spans of any other length are refused
    /// (negative return) rather than read past.
    [[nodiscard]] herr_t select_hyperslab(std::span<const hsize_t> offset, std::span<const hsize_t> count) {
        const int ndims = H5Sget_simple_extent_ndims(id_);
        if (ndims < 0 || offset.size() != static_cast<std::size_t>(ndims) ||
            count.size() != static_cast<std::size_t>(ndims)) {
            return -1;
        }
        return H5Sselect_hyperslab(id_, H5S_SELECT_SET, offset.data(), nullptr, count.data(), nullptr);
    }
};

/// RAII wrapper for HDF5 property list handles
class plist {
    hid_t id_ = -1;

public:
    plist() = default;

    /// Create dataset creation property list
    static plist dataset_create() {
        plist p;
        p.id_ = H5Pcreate(H5P_DATASET_CREATE);
        return p;
    }

    /// Create dataset access property list
    static plist dataset_access() {
        plist p;
        p.id_ = H5Pcreate(H5P_DATASET_ACCESS);
        return p;
    }

    ~plist() { if (id_ >= 0) H5Pclose(id_); }

    plist(const plist&) = delete;
    plist& operator=(const plist&) = delete;
    plist(plist&& other) noexcept : id_(other.id_) { other.id_ = -1; }
    plist& operator=(plist&& other) noexcept {
        if (this != &other) {
            if (id_ >= 0) H5Pclose(id_);
            id_ = other.id_;
            other.id_ = -1;
        }
        return *this;
    }

    [[nodiscard]] hid_t id() const { return id_; }
    [[nodiscard]] explicit operator bool() const { return id_ >= 0; }

    /// Set chunking
    void set_chunk(std::span<const hsize_t> chunk_dims) {
        H5Pset_chunk(id_, static_cast<int>(chunk_dims.size()), chunk_dims.data());
    }

    /// Set GZIP compression (requires chunking)
    void set_deflate(int level) {
        H5Pset_deflate(id_, static_cast<unsigned>(level));
    }

    /// Set shuffle filter (improves compression for numerical data)
    void set_shuffle() {
        H5Pset_shuffle(id_);
    }
};

/// RAII wrapper for HDF5 dataset handles
class dataset {
    hid_t id_ = -1;

public:
    dataset() = default;

    /// Create new dataset
    static dataset create(hid_t parent, const char* name, hid_t dtype,
                          hid_t dataspace, hid_t dcpl = H5P_DEFAULT) {
        dataset ds;
        ds.id_ = H5Dcreate2(parent, name, dtype, dataspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        return ds;
    }

    /// Open existing dataset
    static dataset open(hid_t parent, const char* name) {
        dataset ds;
        ds.id_ = H5Dopen2(parent, name, H5P_DEFAULT);
        return ds;
    }

    ~dataset() { if (id_ >= 0) H5Dclose(id_); }

    dataset(const dataset&) = delete;
    dataset& operator=(const dataset&) = delete;
    dataset(dataset&& other) noexcept : id_(other.id_) { other.id_ = -1; }
    dataset& operator=(dataset&& other) noexcept {
        if (this != &other) {
            if (id_ >= 0) H5Dclose(id_);
            id_ = other.id_;
            other.id_ = -1;
        }
        return *this;
    }

    [[nodiscard]] hid_t id() const { return id_; }
    [[nodiscard]] explicit operator bool() const { return id_ >= 0; }

    /// Extend dataset dimensions (for extensible datasets). Negative on failure
    /// (e.g. the dataset is not extensible, or the file cannot grow).
    [[nodiscard]] herr_t extend(std::span<const hsize_t> new_dims) {
        return H5Dset_extent(id_, new_dims.data());
    }

    /// Write data. Negative on failure. A successful return does not mean the
    /// bytes reached the disk: HDF5 caches chunks, so ENOSPC/EFBIG can surface
    /// only at a later flush() or close() — a durable writer checks those too.
    [[nodiscard]] herr_t write(hid_t mem_type, hid_t mem_space, hid_t file_space, const void* data) {
        return H5Dwrite(id_, mem_type, mem_space, file_space, H5P_DEFAULT, data);
    }

    /// Read data
    herr_t read(hid_t mem_type, hid_t mem_space, hid_t file_space, void* data) {
        return H5Dread(id_, mem_type, mem_space, file_space, H5P_DEFAULT, data);
    }
};

/// Write a scalar attribute to a location. Returns a negative herr_t if the
/// attribute could not be created OR its write failed, 0 on success — so a caller
/// that must not publish a checkpoint with missing metadata (hdf5_state_writer)
/// can detect the failure instead of silently dropping it.
[[nodiscard]] inline herr_t write_attr(hid_t loc, const char* name, std::int64_t value) {
    auto space = dataspace::scalar();
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT64, space.id(), H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) return -1;
    const herr_t w = H5Awrite(attr, H5T_NATIVE_INT64, &value);
    H5Aclose(attr);
    return w;
}

[[nodiscard]] inline herr_t write_attr(hid_t loc, const char* name, std::uint64_t value) {
    auto space = dataspace::scalar();
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_UINT64, space.id(), H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) return -1;
    const herr_t w = H5Awrite(attr, H5T_NATIVE_UINT64, &value);
    H5Aclose(attr);
    return w;
}

[[nodiscard]] inline herr_t write_attr(hid_t loc, const char* name, double value) {
    auto space = dataspace::scalar();
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space.id(), H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) return -1;
    const herr_t w = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    return w;
}

[[nodiscard]] inline herr_t write_attr(hid_t loc, const char* name, std::string_view value) {
    // Copy into NUL-terminated storage FIRST. H5T_STR_NULLTERM with size ==
    // value.size()+1 makes H5Awrite read value.size()+1 bytes; a std::string_view
    // carries no trailing NUL, so passing value.data() directly is a 1-byte
    // out-of-bounds read for any non-terminated view. std::string guarantees a
    // readable NUL at buf[size()], so c_str() is safe to hand to H5Awrite.
    const std::string buf(value);
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, buf.size() + 1);
    H5Tset_strpad(str_type, H5T_STR_NULLTERM);

    auto space = dataspace::scalar();
    hid_t attr = H5Acreate2(loc, name, str_type, space.id(), H5P_DEFAULT, H5P_DEFAULT);
    herr_t rc = -1;
    if (attr >= 0) {
        rc = H5Awrite(attr, str_type, buf.c_str());
        H5Aclose(attr);
    }
    H5Tclose(str_type);
    return rc;
}

/// Appends `n_rows` rows to `ds`, a dataset extensible along dimension 0: grows
/// it to `first_row + n_rows` rows and writes `data` there — `n_rows` rows of
/// shape `row_shape` (the dimensions after the first; empty for a 1-D dataset),
/// C order, in memory type `mem_type`. A `row_shape` that is not the dataset's
/// own (rank or any dimension after the first) is refused before the dataset
/// is touched. Returns < 0 if any step failed; after a refused shape nothing
/// has changed, after a failed grow or write the extent may already have
/// grown, so the caller must not count the rows (a retry at the same
/// `first_row` overwrites them). Executor job only.
[[nodiscard]] herr_t append_rows(hid_t ds, hsize_t first_row, hsize_t n_rows,
                                 std::span<const hsize_t> row_shape,
                                 hid_t mem_type, const void* data);

// ---------------------------------------------------------------------------
// Reading files COAST did not necessarily write.
//
// H5Dread with H5S_ALL, and every H5Aread, transfers as many elements as the
// FILE declares, whatever the caller allocated. A reader that sizes its buffer
// from part of the shape (dims[0] of a dataset that turns out to be 2-D, one
// value for an attribute that holds three) overflows that buffer with a length
// and content taken from the file. The helpers below check the shape before
// reading, read through a memory dataspace sized to the buffer, and check every
// libhdf5 return code. Anything that does not match throws std::runtime_error
// naming the object; a load reports that as its error.
//
// "Absent" stays distinct from "malformed": the attribute readers return
// nullopt only when the attribute does not exist.

/// Path of an open object (dataset, group, attribute's object) for messages.
inline std::string object_path(hid_t id) {
    const ssize_t n = H5Iget_name(id, nullptr, 0);
    if (n <= 0) return "<unnamed>";
    std::string s(static_cast<std::size_t>(n), '\0');
    H5Iget_name(id, s.data(), static_cast<std::size_t>(n) + 1);
    return s;
}

/// Closes a raw HDF5 id (attribute, datatype, ...) on scope exit.
class id_guard {
    hid_t id_;
    herr_t (*close_)(hid_t);

public:
    id_guard(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
    ~id_guard() { if (id_ >= 0) close_(id_); }
    id_guard(const id_guard&) = delete;
    id_guard& operator=(const id_guard&) = delete;
    [[nodiscard]] hid_t id() const { return id_; }
};

namespace detail {

[[noreturn]] inline void fail(const std::string& where, std::string_view what) {
    throw std::runtime_error("hdf5: " + where + ": " + std::string(what));
}

inline std::string attr_where(hid_t attr) {
    const ssize_t n = H5Aget_name(attr, 0, nullptr);
    std::string name(n > 0 ? static_cast<std::size_t>(n) : 0, '\0');
    if (n > 0) H5Aget_name(attr, static_cast<std::size_t>(n) + 1, name.data());
    return object_path(attr) + " attribute '" + name + "'";
}

/// The open attribute must hold exactly one element.
inline void require_single_element(hid_t attr) {
    const id_guard space(H5Aget_space(attr), H5Sclose);
    if (space.id() < 0) fail(attr_where(attr), "cannot get its dataspace");
    const hssize_t n = H5Sget_simple_extent_npoints(space.id());
    if (n != 1) {
        fail(attr_where(attr), "holds " + std::to_string(n) + " elements, expected 1");
    }
}

/// Opens attribute `name` on `loc`: an invalid id when it does not exist,
/// a throw when it exists but cannot be opened.
inline hid_t open_attr(hid_t loc, const char* name) {
    const htri_t exists = H5Aexists(loc, name);
    if (exists == 0) return H5I_INVALID_HID;
    const hid_t attr = exists > 0 ? H5Aopen(loc, name, H5P_DEFAULT) : H5I_INVALID_HID;
    if (attr < 0) fail(object_path(loc) + " attribute '" + name + "'", "cannot be opened");
    return attr;
}

template <class T>
std::optional<T> read_attr_scalar(hid_t loc, const char* name, hid_t mem_type) {
    const id_guard attr(open_attr(loc, name), H5Aclose);
    if (attr.id() < 0) return std::nullopt;
    require_single_element(attr.id());
    T value{};
    if (H5Aread(attr.id(), mem_type, &value) < 0) {
        fail(attr_where(attr.id()), "read failed (type does not convert?)");
    }
    return value;
}

} // namespace detail

/// Dimensions of dataset `ds`, which must have exactly `rank` of them. Rank 0
/// is a scalar dataspace. H5S_NULL also reports rank 0, but holds no element
/// and a read from it "succeeds" without writing anything, so it is refused.
[[nodiscard]] inline std::vector<hsize_t> require_rank(hid_t ds, int rank) {
    const dataspace space = dataspace::from_dataset(ds);
    if (!space) detail::fail(object_path(ds), "cannot get its dataspace");
    if (H5Sget_simple_extent_type(space.id()) == H5S_NULL) {
        detail::fail(object_path(ds), "holds no data (null dataspace)");
    }
    const int ndims = H5Sget_simple_extent_ndims(space.id());
    if (ndims != rank) {
        detail::fail(object_path(ds), "expected rank " + std::to_string(rank) +
                                       ", the file has rank " + std::to_string(ndims));
    }
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
    if (rank > 0 && H5Sget_simple_extent_dims(space.id(), dims.data(), nullptr) != rank) {
        detail::fail(object_path(ds), "cannot get its dimensions");
    }
    return dims;
}

/// Number of elements of a dataset with dimensions `dims`, refusing a product
/// that does not fit std::size_t. `ds` names the dataset in the message.
[[nodiscard]] inline std::size_t element_count(hid_t ds, std::span<const hsize_t> dims) {
    std::size_t n = 1;
    for (const hsize_t d : dims) {
        if (d != 0 && n > std::numeric_limits<std::size_t>::max() / d) {
            detail::fail(object_path(ds), "element count overflows");
        }
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

/// Refuses a dataset of `n` elements that its file does not store. Dims are
/// metadata: a small file can declare a [2^34] dataset with no storage behind
/// it, and a buffer sized from that exhausts memory (#159). The bound is what
/// the file has allocated for the dataset, at the file type's element size:
///   - contiguous or compact: the bytes stored;
///   - chunked: the allocated chunks at their uncompressed size, so a fully
///     written compressed dataset passes however well it compresses.
/// So a dataset with unwritten regions (which read as the fill value) beyond its
/// allocated storage is refused, though valid HDF5; no COAST writer leaves one.
/// External and virtual storage (data outside the file) are refused. Not
/// stopped: data that is really there but compresses well (deflate reaches
/// ~1000:1) can still ask for far more memory than the file's size.
inline void require_stored(hid_t ds, std::size_t n) {
    if (n == 0) return;
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    const auto mul = [](std::size_t a, std::size_t b) {   // saturates at max
        return a != 0 && b > max / a ? max : a * b;
    };

    const id_guard    type(H5Dget_type(ds), H5Tclose);
    const std::size_t elem = type.id() < 0 ? 0 : H5Tget_size(type.id());
    if (elem == 0) detail::fail(object_path(ds), "cannot get its element size");
    if (n > max / elem) detail::fail(object_path(ds), "byte size overflows");
    const std::size_t declared = n * elem;

    const id_guard dcpl(H5Dget_create_plist(ds), H5Pclose);
    if (dcpl.id() < 0) detail::fail(object_path(ds), "cannot get its creation properties");
    std::size_t stored = 0;
    switch (H5Pget_layout(dcpl.id())) {
    case H5D_COMPACT:
    case H5D_CONTIGUOUS:
        if (H5Pget_external_count(dcpl.id()) != 0) {
            detail::fail(object_path(ds), "stores its data in external files");
        }
        stored = H5Dget_storage_size(ds);
        break;
    case H5D_CHUNKED: {
        hsize_t   chunk[H5S_MAX_RANK];
        const int rank = H5Pget_chunk(dcpl.id(), H5S_MAX_RANK, chunk);
        hsize_t   chunks = 0;
        if (rank <= 0 || H5Dget_num_chunks(ds, H5S_ALL, &chunks) < 0) {
            detail::fail(object_path(ds), "cannot get its chunks");
        }
        std::size_t chunk_bytes = elem;
        for (int i = 0; i < rank; ++i) chunk_bytes = mul(chunk_bytes, chunk[i]);
        stored = mul(chunk_bytes, chunks);
        break;
    }
    default:
        detail::fail(object_path(ds), "has virtual or unknown storage");
    }
    if (declared > stored) {
        detail::fail(object_path(ds), "declares " + std::to_string(declared) +
                                      " bytes, the file stores at most " + std::to_string(stored));
    }
}

/// Reads the whole of dataset `ds` into `n` elements of `mem_type` at `dst`.
/// The memory dataspace is sized to `n`, so HDF5 itself refuses a dataset of
/// any other element count; a failed read (or type conversion) throws.
inline void read_all(hid_t ds, hid_t mem_type, void* dst, std::size_t n) {
    if (n == 0) return;
    const hsize_t mem_dims[1] = {n};
    const dataspace mem = dataspace::simple(mem_dims);
    if (!mem || H5Dread(ds, mem_type, mem.id(), H5S_ALL, H5P_DEFAULT, dst) < 0) {
        detail::fail(object_path(ds), "read of " + std::to_string(n) +
                                      " elements failed (count or type does not match)");
    }
}

/// A dataset read whole, with its dimensions.
template <class T>
struct array_read {
    std::vector<hsize_t> dims;
    std::vector<T>       values;
};

/// Reads the whole of dataset `ds`, which must have exactly `rank` dimensions,
/// as `mem_type` elements. The one place a buffer is sized from a dataset's
/// shape: require_rank, element_count and require_stored, then read_all.
template <class T>
[[nodiscard]] array_read<T> read_array(hid_t ds, hid_t mem_type, int rank) {
    array_read<T> r{require_rank(ds, rank), {}};
    const std::size_t n = element_count(ds, r.dims);
    require_stored(ds, n);
    r.values.resize(n);
    read_all(ds, mem_type, r.values.data(), r.values.size());
    return r;
}

/// Reads a 1-D dataset of fixed-length strings. A variable-length or
/// non-string type throws. A string that fills its slot has no NUL, so each
/// one is bounded by its slot.
[[nodiscard]] inline std::vector<std::string> read_fixed_strings(hid_t ds) {
    const auto     dims = require_rank(ds, 1);
    const id_guard type(H5Dget_type(ds), H5Tclose);
    const std::size_t size = type.id() < 0 ? 0 : H5Tget_size(type.id());
    if (type.id() < 0 || H5Tget_class(type.id()) != H5T_STRING ||
        H5Tis_variable_str(type.id()) != 0 || size == 0) {
        detail::fail(object_path(ds), "expected fixed-length strings");
    }
    require_stored(ds, element_count(ds, dims));
    const hsize_t     bytes[2] = {dims[0], size};
    std::vector<char> buf(element_count(ds, bytes));
    const id_guard    mem_type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(mem_type.id(), size);
    read_all(ds, mem_type.id(), buf.data(), dims[0]);

    std::vector<std::string> out;
    out.reserve(dims[0]);
    for (std::size_t i = 0; i < dims[0]; ++i) {
        const char* s = buf.data() + i * size;
        out.emplace_back(s, ::strnlen(s, size));
    }
    return out;
}

/// Value of the open string attribute `attr`. COAST writes every string
/// attribute as one fixed-length string. A variable-length string is refused
/// (H5Aread would store a char* into the buffer), as are a non-string type and
/// more than one element. The value ends at the first NUL.
inline std::string read_string_attr(hid_t attr) {
    detail::require_single_element(attr);
    const id_guard type(H5Aget_type(attr), H5Tclose);
    if (type.id() < 0) detail::fail(detail::attr_where(attr), "cannot get its type");
    if (H5Tget_class(type.id()) != H5T_STRING) {
        detail::fail(detail::attr_where(attr), "is not a string");
    }
    if (H5Tis_variable_str(type.id()) != 0) {
        detail::fail(detail::attr_where(attr), "is a variable-length string");
    }
    const std::size_t size = H5Tget_size(type.id());
    if (size == 0) detail::fail(detail::attr_where(attr), "has string size 0");
    std::string value(size, '\0');
    if (H5Aread(attr, type.id(), value.data()) < 0) {
        detail::fail(detail::attr_where(attr), "read failed");
    }
    if (const auto nul = value.find('\0'); nul != std::string::npos) value.resize(nul);
    return value;
}

/// Read a scalar attribute: nullopt if it does not exist; a throw if it exists
/// but does not hold exactly one value convertible to the requested type.
inline std::optional<std::int64_t> read_attr_int64(hid_t loc, const char* name) {
    return detail::read_attr_scalar<std::int64_t>(loc, name, H5T_NATIVE_INT64);
}

inline std::optional<std::uint64_t> read_attr_uint64(hid_t loc, const char* name) {
    return detail::read_attr_scalar<std::uint64_t>(loc, name, H5T_NATIVE_UINT64);
}

inline std::optional<double> read_attr_double(hid_t loc, const char* name) {
    return detail::read_attr_scalar<double>(loc, name, H5T_NATIVE_DOUBLE);
}

/// Read a string attribute: nullopt if it does not exist; otherwise see
/// read_string_attr for what is refused.
inline std::optional<std::string> read_attr_string(hid_t loc, const char* name) {
    const id_guard attr(detail::open_attr(loc, name), H5Aclose);
    if (attr.id() < 0) return std::nullopt;
    return read_string_attr(attr.id());
}

/// Check if a link (group/dataset) exists
inline bool link_exists(hid_t loc, const char* name) {
    return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}

/// Check if an attribute exists
inline bool attr_exists(hid_t loc, const char* name) {
    return H5Aexists(loc, name) > 0;
}

} // namespace hdf5

#endif // COAST_HDF5_UTILS_H
