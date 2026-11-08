//
// HDF5 reader for buffer_log files
//

#include "gui_replay/buffer_log_reader.h"
#include "diag/diag.h"
#include "hdf5/executor.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <vector>

// Every libhdf5-touching method in this file wraps its body in
// h5_exec_->submit_sync(...). The legacy hdf5::* RAII wrappers from
// include/logging/hdf5_utils.h remain in use; they are safe here because
// every call to them — including destruction of the cached `file_`
// handle — runs on the executor's worker thread. See the header comment
// in hdf5_utils.h for the rule.

namespace coast::replay {

namespace {

// The file's shapes are checked against what buffer_logger writes before any
// buffer is sized from them (#150): hdf5::read_array throws on any other rank,
// count or type. open() turns a throw into a refused file.

// Rows of an [n, cols] dataset, flattened. Any other rank or width throws.
template <class T>
std::vector<T> read_rows(hid_t ds, hid_t mem_type, hsize_t cols)
{
    auto a = hdf5::read_array<T>(ds, mem_type, 2);
    if (a.dims[1] != cols) {
        throw std::runtime_error("hdf5: " + hdf5::object_path(ds) + ": expected " +
                                 std::to_string(cols) + " columns, the file has " +
                                 std::to_string(a.dims[1]));
    }
    return std::move(a.values);
}

std::string frame_data_path(std::size_t buffer_id)
{
    return "steps/buffers/" + std::to_string(buffer_id) + "/data";
}

// Throws unless steps/buffers/<id>/data, where present, is [frames, size] and
// stored in the file: a frame read passes two-entry offset/count arrays and a
// row of `size` floats. Checked once, at open().
void check_frame_data(const hdf5::file& file, std::size_t buffer_id, std::size_t size)
{
    const std::string path = frame_data_path(buffer_id);
    if (!hdf5::link_exists(file.id(), path.c_str())) return;
    const auto ds = hdf5::dataset::open(file.id(), path.c_str());
    if (!ds) throw std::runtime_error("hdf5: " + path + ": cannot be opened");
    const auto dims = hdf5::require_rank(ds.id(), 2);
    if (dims[1] != size) {
        throw std::runtime_error("hdf5: " + hdf5::object_path(ds.id()) + ": expected [frames, " +
                                 std::to_string(size) + "], the file has [" +
                                 std::to_string(dims[0]) + ", " + std::to_string(dims[1]) + "]");
    }
    // A frame read sizes its buffer from `size` (buffer_sizes); the file must
    // store the frames it declares (#159).
    hdf5::require_stored(ds.id(), hdf5::element_count(ds.id(), dims));
}

} // namespace

buffer_log_reader::buffer_log_reader(hdf5::executor& h5_exec)
    : h5_exec_(&h5_exec)
{
}

buffer_log_reader::~buffer_log_reader() {
    close();
}

buffer_log_reader::buffer_log_reader(buffer_log_reader&&) noexcept = default;

bool buffer_log_reader::open(const std::filesystem::path& path) {
    close();

    path_ = path;

    bool opened = h5_exec_->submit_sync([this, &path]() -> bool {
        file_ = hdf5::file::open_read(path.string());
        return static_cast<bool>(file_);
    });
    if (!opened) {
        path_.clear();
        return false;
    }

    // Read metadata and topology. A malformed file throws from the checked
    // readers and is refused like an unreadable one.
    bool ok = false;
    try {
        ok = read_metadata() && read_topology();
    } catch (const std::exception& e) {
        COAST_ERROR("replay", "cannot open {}: {}", path.string(), e.what());
    }
    if (!ok) {
        close();
        return false;
    }

    return true;
}

void buffer_log_reader::close() {
    // The hdf5::file destructor (via reset to {}) calls H5Fclose, so it
    // must run on the worker thread.
    if (h5_exec_ && file_) {
        h5_exec_->submit_sync([this] {
            file_ = {};
        });
    } else {
        file_ = {};
    }
    path_.clear();
    metadata_ = {};
    topology_ = {};
    step_numbers_.clear();
    timestamps_.clear();
    steps_loaded_ = false;
}

bool buffer_log_reader::read_metadata() {
    return h5_exec_->submit_sync([this]() -> bool {
    auto metadata_group = hdf5::group::open(file_.id(), "metadata");
    if (!metadata_group) {
        return false;
    }

    // Read attributes (absent ones read as nullopt)
    if (auto val = hdf5::read_attr_string(metadata_group.id(), "schema_version")) {
        metadata_.schema_version = *val;
    }
    if (auto val = hdf5::read_attr_uint64(metadata_group.id(), "start_step")) {
        metadata_.start_step = *val;
    }
    if (auto val = hdf5::read_attr_string(metadata_group.id(), "config")) {
        metadata_.config_json = *val;
    }
    if (auto val = hdf5::read_attr_string(metadata_group.id(), "created")) {
        metadata_.created = *val;
    }

    // Get total frame count from step_numbers dataset
    auto steps_group = hdf5::group::open(file_.id(), "steps");
    if (steps_group) {
        auto step_numbers_ds = hdf5::dataset::open(steps_group.id(), "step_numbers");
        if (step_numbers_ds) {
            metadata_.total_frames = hdf5::require_rank(step_numbers_ds.id(), 1)[0];
        }
        // Checked here, at open, so the lazy load_step_data() finds it 1-D.
        if (auto ts_ds = hdf5::dataset::open(steps_group.id(), "timestamps")) {
            (void)hdf5::require_rank(ts_ds.id(), 1);
        }
    }

    return true;
    });
}

bool buffer_log_reader::read_topology() {
    return h5_exec_->submit_sync([this]() -> bool {
    auto metadata_group = hdf5::group::open(file_.id(), "metadata");
    if (!metadata_group) {
        return false;
    }

    auto topo_group = hdf5::group::open(metadata_group.id(), "topology");
    if (!topo_group) {
        return false;
    }

    // Read buffer IDs
    std::vector<std::uint64_t> buffer_ids;
    auto ids_ds = hdf5::dataset::open(topo_group.id(), "buffer_ids");
    if (ids_ds) {
        buffer_ids = hdf5::read_array<std::uint64_t>(ids_ds.id(), H5T_NATIVE_UINT64, 1).values;
    }

    // Read buffer sizes
    std::vector<std::uint64_t> buffer_sizes;
    auto sizes_ds = hdf5::dataset::open(topo_group.id(), "buffer_sizes");
    if (sizes_ds) {
        buffer_sizes = hdf5::read_array<std::uint64_t>(sizes_ds.id(), H5T_NATIVE_UINT64, 1).values;
    }

    // Read buffer names
    std::vector<std::string> buffer_names;
    auto names_ds = hdf5::dataset::open(topo_group.id(), "buffer_names");
    if (names_ds) {
        buffer_names = hdf5::read_fixed_strings(names_ds.id());
    }

    // Read connectivity
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    auto conn_ds = hdf5::dataset::open(topo_group.id(), "connectivity");
    if (conn_ds) {
        const auto edge_data = read_rows<std::uint64_t>(conn_ds.id(), H5T_NATIVE_UINT64, 2);
        for (std::size_t i = 0; i + 1 < edge_data.size(); i += 2) {
            edges.emplace_back(edge_data[i], edge_data[i + 1]);
        }
    }

    // Read visualization hints (optional, may not exist in older files). They
    // only steer the display, so hints of another shape or a non-numeric type
    // are skipped, not refused: the buffers then get the computed defaults below.
    std::vector<buffer_vis_hint> vis_hints;
    auto hints_ds = hdf5::dataset::open(topo_group.id(), "vis_hints");
    if (hints_ds) {
        const auto           dims = hdf5::dataspace::from_dataset(hints_ds.id()).dims();
        const hdf5::id_guard type(H5Dget_type(hints_ds.id()), H5Tclose);
        const H5T_class_t    cls  = H5Tget_class(type.id());
        if (dims.size() != 2 || dims[1] != 3 || (cls != H5T_INTEGER && cls != H5T_FLOAT)) {
            COAST_WARN("replay", "{}: vis_hints is not a numeric [n, 3], using default hints",
                       path_.string());
            hints_ds = {};
        }
    }
    if (hints_ds) {
        const auto hints_data = read_rows<std::int32_t>(hints_ds.id(), H5T_NATIVE_INT32, 3);
        vis_hints.reserve(hints_data.size() / 3);
        for (std::size_t i = 0; i + 2 < hints_data.size(); i += 3) {
            buffer_vis_hint hint;
            hint.width = hints_data[i];
            hint.height = hints_data[i + 1];
            hint.num_channels = hints_data[i + 2];
            vis_hints.push_back(hint);
        }
    }

    // Build buffer_info structures
    topology_.buffers.reserve(buffer_ids.size());
    for (std::size_t i = 0; i < buffer_ids.size(); ++i) {
        buffer_info info;
        info.id = buffer_ids[i];
        info.size = i < buffer_sizes.size() ? buffer_sizes[i] : 0;
        info.name = i < buffer_names.size() ? buffer_names[i] : "buffer_" + std::to_string(info.id);

        // Use vis_hints if available, otherwise compute default
        if (i < vis_hints.size()) {
            info.vis_hint = vis_hints[i];
        } else {
            // Compute default from size
            auto sqrt_size = static_cast<int>(std::sqrt(static_cast<double>(info.size)));
            if (sqrt_size * sqrt_size == static_cast<int>(info.size)) {
                info.vis_hint.width = sqrt_size;
                info.vis_hint.height = sqrt_size;
            } else {
                info.vis_hint.width = static_cast<int>(info.size);
                info.vis_hint.height = 1;
            }
            info.vis_hint.num_channels = 1;
        }

        check_frame_data(file_, info.id, info.size);

        // Find input IDs from connectivity
        for (const auto& [from, to] : edges) {
            if (to == info.id) {
                info.input_ids.push_back(from);
            }
        }

        topology_.buffers.push_back(std::move(info));
    }

    return true;
    });
}

void buffer_log_reader::load_step_data() const {
    if (steps_loaded_ || !file_) {
        return;
    }

    h5_exec_->submit_sync([this] {
    auto steps_group = hdf5::group::open(file_.id(), "steps");
    if (!steps_group) {
        return;
    }

    // open() already checked both are 1-D; a read can still fail (type, I/O).
    // That is logged once and leaves the steps unloaded, so get_step() and
    // get_timestamp() report 0 rather than throwing into the GUI frame.
    steps_loaded_ = true;
    try {
        if (auto step_ds = hdf5::dataset::open(steps_group.id(), "step_numbers")) {
            step_numbers_ = hdf5::read_array<std::uint64_t>(step_ds.id(), H5T_NATIVE_UINT64, 1).values;
        }
        if (auto ts_ds = hdf5::dataset::open(steps_group.id(), "timestamps")) {
            timestamps_ = hdf5::read_array<double>(ts_ds.id(), H5T_NATIVE_DOUBLE, 1).values;
        }
    } catch (const std::exception& e) {
        step_numbers_.clear();
        timestamps_.clear();
        COAST_ERROR("replay", "cannot read step index of {}: {}", path_.string(), e.what());
    }
    });
}

std::uint64_t buffer_log_reader::get_step(std::size_t frame_idx) const {
    load_step_data();
    if (frame_idx < step_numbers_.size()) {
        return step_numbers_[frame_idx];
    }
    return 0;
}

double buffer_log_reader::get_timestamp(std::size_t frame_idx) const {
    load_step_data();
    if (frame_idx < timestamps_.size()) {
        return timestamps_[frame_idx];
    }
    return 0.0;
}

bool buffer_log_reader::read_frame_data(std::size_t buffer_id, std::size_t frame_idx,
                                         std::vector<float>& out_data) {
    if (!file_ || frame_idx >= metadata_.total_frames) {
        return false;
    }

    const buffer_info* info = get_buffer(buffer_id);
    if (!info) {
        return false;
    }

    return h5_exec_->submit_sync([this, buffer_id, frame_idx, info, &out_data]() -> bool {
        // open() checked its shape (check_frame_data).
        auto data_ds = hdf5::dataset::open(file_.id(), frame_data_path(buffer_id).c_str());
        if (!data_ds) {
            return false;
        }

        // Read single row. open() bounded the size by what the file stores,
        // but data that compresses well can still exceed memory; a throw here
        // would leave through submit_sync into a GUI caller that catches nothing.
        try {
            out_data.resize(info->size);
        } catch (const std::bad_alloc&) {
            COAST_ERROR("replay", "cannot allocate a frame of buffer {} ({} floats)", buffer_id, info->size);
            return false;
        }

        hsize_t offset[2] = {frame_idx, 0};
        hsize_t count[2] = {1, info->size};

        auto file_space = hdf5::dataspace::from_dataset(data_ds.id());
        hsize_t mem_dims[1] = {info->size};
        auto mem_space = hdf5::dataspace::simple(std::span(mem_dims));

        return file_space.select_hyperslab(std::span(offset), std::span(count)) >= 0 &&
               data_ds.read(H5T_NATIVE_FLOAT, mem_space.id(), file_space.id(), out_data.data()) >= 0;
    });
}

bool buffer_log_reader::read_frame_range(std::size_t buffer_id, std::size_t start_frame,
                                          std::size_t count, std::vector<float>& out_data) {
    if (!file_ || start_frame >= metadata_.total_frames) {
        return false;
    }

    const buffer_info* info = get_buffer(buffer_id);
    if (!info) {
        return false;
    }

    // Clamp count to available frames
    count = std::min(count, metadata_.total_frames - start_frame);
    if (count == 0) {
        out_data.clear();
        return true;
    }

    return h5_exec_->submit_sync([this, buffer_id, start_frame, count, info, &out_data]() -> bool {
        // open() checked its shape (check_frame_data).
        auto data_ds = hdf5::dataset::open(file_.id(), frame_data_path(buffer_id).c_str());
        if (!data_ds) {
            return false;
        }

        // Read multiple rows. buffer_sizes comes from the file, so the product
        // is checked before it sizes the buffer.
        if (info->size != 0 && count > std::numeric_limits<std::size_t>::max() / info->size) {
            return false;
        }
        try {
            out_data.resize(count * info->size);   // see read_frame_data
        } catch (const std::bad_alloc&) {
            COAST_ERROR("replay", "cannot allocate {} frames of buffer {} ({} floats each)", count,
                        buffer_id, info->size);
            return false;
        }

        hsize_t offset[2] = {start_frame, 0};
        hsize_t h_count[2] = {count, info->size};

        auto file_space = hdf5::dataspace::from_dataset(data_ds.id());
        auto mem_space = hdf5::dataspace::simple(std::span(h_count));

        return file_space.select_hyperslab(std::span(offset), std::span(h_count)) >= 0 &&
               data_ds.read(H5T_NATIVE_FLOAT, mem_space.id(), file_space.id(), out_data.data()) >= 0;
    });
}

const buffer_info* buffer_log_reader::get_buffer(std::size_t id) const {
    for (const auto& buf : topology_.buffers) {
        if (buf.id == id) {
            return &buf;
        }
    }
    return nullptr;
}

std::optional<std::size_t> buffer_log_reader::get_buffer_index(std::size_t id) const {
    for (std::size_t i = 0; i < topology_.buffers.size(); ++i) {
        if (topology_.buffers[i].id == id) {
            return i;
        }
    }
    return std::nullopt;
}

} // namespace coast::replay
