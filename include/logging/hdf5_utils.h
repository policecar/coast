//
// RAII wrappers and utilities for HDF5 operations
//

#ifndef COAST_HDF5_UTILS_H
#define COAST_HDF5_UTILS_H

#include <hdf5.h>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>

namespace logging::hdf5 {

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

    /// Create new file (truncates if exists)
    static file create(const std::string& path) {
        file f;
        f.id_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        return f;
    }

    ~file() { if (id_ >= 0) H5Fclose(id_); }

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

    void flush() { if (id_ >= 0) H5Fflush(id_, H5F_SCOPE_GLOBAL); }
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

    /// Select hyperslab for partial I/O
    void select_hyperslab(std::span<const hsize_t> offset, std::span<const hsize_t> count) {
        H5Sselect_hyperslab(id_, H5S_SELECT_SET, offset.data(), nullptr, count.data(), nullptr);
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

    /// Extend dataset dimensions (for extensible datasets)
    void extend(std::span<const hsize_t> new_dims) {
        H5Dset_extent(id_, new_dims.data());
    }

    /// Write data
    herr_t write(hid_t mem_type, hid_t mem_space, hid_t file_space, const void* data) {
        return H5Dwrite(id_, mem_type, mem_space, file_space, H5P_DEFAULT, data);
    }

    /// Read data
    herr_t read(hid_t mem_type, hid_t mem_space, hid_t file_space, void* data) {
        return H5Dread(id_, mem_type, mem_space, file_space, H5P_DEFAULT, data);
    }
};

/// Write a scalar attribute to a location
inline void write_attr(hid_t loc, const char* name, std::int64_t value) {
    auto space = dataspace::scalar();
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT64, space.id(), H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0) {
        H5Awrite(attr, H5T_NATIVE_INT64, &value);
        H5Aclose(attr);
    }
}

inline void write_attr(hid_t loc, const char* name, std::uint64_t value) {
    auto space = dataspace::scalar();
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_UINT64, space.id(), H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0) {
        H5Awrite(attr, H5T_NATIVE_UINT64, &value);
        H5Aclose(attr);
    }
}

inline void write_attr(hid_t loc, const char* name, double value) {
    auto space = dataspace::scalar();
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space.id(), H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0) {
        H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attr);
    }
}

inline void write_attr(hid_t loc, const char* name, std::string_view value) {
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, value.size() + 1);
    H5Tset_strpad(str_type, H5T_STR_NULLTERM);

    auto space = dataspace::scalar();
    hid_t attr = H5Acreate2(loc, name, str_type, space.id(), H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0) {
        H5Awrite(attr, str_type, value.data());
        H5Aclose(attr);
    }
    H5Tclose(str_type);
}

/// Read a scalar attribute
inline std::optional<std::int64_t> read_attr_int64(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return std::nullopt;
    std::int64_t value;
    H5Aread(attr, H5T_NATIVE_INT64, &value);
    H5Aclose(attr);
    return value;
}

inline std::optional<std::uint64_t> read_attr_uint64(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return std::nullopt;
    std::uint64_t value;
    H5Aread(attr, H5T_NATIVE_UINT64, &value);
    H5Aclose(attr);
    return value;
}

inline std::optional<double> read_attr_double(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return std::nullopt;
    double value;
    H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    return value;
}

inline std::optional<std::string> read_attr_string(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return std::nullopt;

    hid_t atype = H5Aget_type(attr);
    std::size_t size = H5Tget_size(atype);

    std::string value(size, '\0');
    H5Aread(attr, atype, value.data());

    H5Tclose(atype);
    H5Aclose(attr);

    // Remove null terminator if present
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

/// Check if a link (group/dataset) exists
inline bool link_exists(hid_t loc, const char* name) {
    return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}

/// Check if an attribute exists
inline bool attr_exists(hid_t loc, const char* name) {
    return H5Aexists(loc, name) > 0;
}

} // namespace logging::hdf5

#endif // COAST_HDF5_UTILS_H
