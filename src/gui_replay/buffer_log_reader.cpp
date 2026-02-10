//
// HDF5 reader for buffer_log files
//

#include "gui_replay/buffer_log_reader.h"

#include <algorithm>
#include <cmath>

namespace coast::replay {

buffer_log_reader::~buffer_log_reader() {
    close();
}

buffer_log_reader::buffer_log_reader(buffer_log_reader&&) noexcept = default;
buffer_log_reader& buffer_log_reader::operator=(buffer_log_reader&&) noexcept = default;

bool buffer_log_reader::open(const std::filesystem::path& path) {
    close();

    file_ = logging::hdf5::file::open_read(path.string());
    if (!file_) {
        return false;
    }

    path_ = path;

    // Read metadata and topology
    if (!read_metadata() || !read_topology()) {
        close();
        return false;
    }

    return true;
}

void buffer_log_reader::close() {
    file_ = {};
    path_.clear();
    metadata_ = {};
    topology_ = {};
    step_numbers_.clear();
    timestamps_.clear();
    steps_loaded_ = false;
}

bool buffer_log_reader::read_metadata() {
    auto metadata_group = logging::hdf5::group::open(file_.id(), "metadata");
    if (!metadata_group) {
        return false;
    }

    // Read attributes (check existence first to avoid HDF5 error messages)
    if (logging::hdf5::attr_exists(metadata_group.id(), "schema_version")) {
        if (auto val = logging::hdf5::read_attr_string(metadata_group.id(), "schema_version")) {
            metadata_.schema_version = *val;
        }
    }

    if (logging::hdf5::attr_exists(metadata_group.id(), "start_step")) {
        if (auto val = logging::hdf5::read_attr_uint64(metadata_group.id(), "start_step")) {
            metadata_.start_step = *val;
        }
    }

    if (logging::hdf5::attr_exists(metadata_group.id(), "config")) {
        if (auto val = logging::hdf5::read_attr_string(metadata_group.id(), "config")) {
            metadata_.config_json = *val;
        }
    }

    if (logging::hdf5::attr_exists(metadata_group.id(), "created")) {
        if (auto val = logging::hdf5::read_attr_string(metadata_group.id(), "created")) {
            metadata_.created = *val;
        }
    }

    // Get total frame count from step_numbers dataset
    auto steps_group = logging::hdf5::group::open(file_.id(), "steps");
    if (steps_group) {
        auto step_numbers_ds = logging::hdf5::dataset::open(steps_group.id(), "step_numbers");
        if (step_numbers_ds) {
            auto space = logging::hdf5::dataspace::from_dataset(step_numbers_ds.id());
            auto dims = space.dims();
            if (!dims.empty()) {
                metadata_.total_frames = dims[0];
            }
        }
    }

    return true;
}

bool buffer_log_reader::read_topology() {
    auto metadata_group = logging::hdf5::group::open(file_.id(), "metadata");
    if (!metadata_group) {
        return false;
    }

    auto topo_group = logging::hdf5::group::open(metadata_group.id(), "topology");
    if (!topo_group) {
        return false;
    }

    // Read buffer IDs
    std::vector<std::uint64_t> buffer_ids;
    auto ids_ds = logging::hdf5::dataset::open(topo_group.id(), "buffer_ids");
    if (ids_ds) {
        auto space = logging::hdf5::dataspace::from_dataset(ids_ds.id());
        auto dims = space.dims();
        if (!dims.empty()) {
            buffer_ids.resize(dims[0]);
            ids_ds.read(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, buffer_ids.data());
        }
    }

    // Read buffer sizes
    std::vector<std::uint64_t> buffer_sizes;
    auto sizes_ds = logging::hdf5::dataset::open(topo_group.id(), "buffer_sizes");
    if (sizes_ds) {
        auto space = logging::hdf5::dataspace::from_dataset(sizes_ds.id());
        auto dims = space.dims();
        if (!dims.empty()) {
            buffer_sizes.resize(dims[0]);
            sizes_ds.read(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, buffer_sizes.data());
        }
    }

    // Read buffer names
    std::vector<std::string> buffer_names;
    auto names_ds = logging::hdf5::dataset::open(topo_group.id(), "buffer_names");
    if (names_ds) {
        auto space = logging::hdf5::dataspace::from_dataset(names_ds.id());
        auto dims = space.dims();
        if (!dims.empty()) {
            // Get the string type size
            hid_t dtype = H5Dget_type(names_ds.id());
            std::size_t str_size = H5Tget_size(dtype);
            H5Tclose(dtype);

            // Read as fixed-size strings
            std::vector<char> names_buf(dims[0] * str_size);
            hid_t memtype = H5Tcopy(H5T_C_S1);
            H5Tset_size(memtype, str_size);
            names_ds.read(memtype, H5S_ALL, H5S_ALL, names_buf.data());
            H5Tclose(memtype);

            // Extract individual strings
            buffer_names.reserve(dims[0]);
            for (std::size_t i = 0; i < dims[0]; ++i) {
                const char* str_start = names_buf.data() + i * str_size;
                std::string name(str_start);
                buffer_names.push_back(name);
            }
        }
    }

    // Read connectivity
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    auto conn_ds = logging::hdf5::dataset::open(topo_group.id(), "connectivity");
    if (conn_ds) {
        auto space = logging::hdf5::dataspace::from_dataset(conn_ds.id());
        auto dims = space.dims();
        if (dims.size() >= 2 && dims[0] > 0) {
            std::vector<std::uint64_t> edge_data(dims[0] * 2);
            conn_ds.read(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, edge_data.data());

            for (std::size_t i = 0; i < dims[0]; ++i) {
                edges.emplace_back(edge_data[i * 2], edge_data[i * 2 + 1]);
            }
        }
    }

    // Read visualization hints (optional, may not exist in older files)
    std::vector<buffer_vis_hint> vis_hints;
    auto hints_ds = logging::hdf5::dataset::open(topo_group.id(), "vis_hints");
    if (hints_ds) {
        auto space = logging::hdf5::dataspace::from_dataset(hints_ds.id());
        auto dims = space.dims();
        if (dims.size() >= 2 && dims[0] > 0 && dims[1] == 3) {
            std::vector<std::int32_t> hints_data(dims[0] * 3);
            hints_ds.read(H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, hints_data.data());

            vis_hints.reserve(dims[0]);
            for (std::size_t i = 0; i < dims[0]; ++i) {
                buffer_vis_hint hint;
                hint.width = hints_data[i * 3];
                hint.height = hints_data[i * 3 + 1];
                hint.num_channels = hints_data[i * 3 + 2];
                vis_hints.push_back(hint);
            }
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

        // Find input IDs from connectivity
        for (const auto& [from, to] : edges) {
            if (to == info.id) {
                info.input_ids.push_back(from);
            }
        }

        topology_.buffers.push_back(std::move(info));
    }

    return true;
}

void buffer_log_reader::load_step_data() const {
    if (steps_loaded_ || !file_) {
        return;
    }

    auto steps_group = logging::hdf5::group::open(file_.id(), "steps");
    if (!steps_group) {
        return;
    }

    // Load step numbers
    auto step_ds = logging::hdf5::dataset::open(steps_group.id(), "step_numbers");
    if (step_ds) {
        auto space = logging::hdf5::dataspace::from_dataset(step_ds.id());
        auto dims = space.dims();
        if (!dims.empty()) {
            step_numbers_.resize(dims[0]);
            step_ds.read(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, step_numbers_.data());
        }
    }

    // Load timestamps
    auto ts_ds = logging::hdf5::dataset::open(steps_group.id(), "timestamps");
    if (ts_ds) {
        auto space = logging::hdf5::dataspace::from_dataset(ts_ds.id());
        auto dims = space.dims();
        if (!dims.empty()) {
            timestamps_.resize(dims[0]);
            ts_ds.read(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, timestamps_.data());
        }
    }

    steps_loaded_ = true;
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

    // Open the buffer's data dataset
    std::string path = "steps/buffers/" + std::to_string(buffer_id) + "/data";
    auto data_ds = logging::hdf5::dataset::open(file_.id(), path.c_str());
    if (!data_ds) {
        return false;
    }

    // Read single row
    out_data.resize(info->size);

    hsize_t offset[2] = {frame_idx, 0};
    hsize_t count[2] = {1, info->size};

    auto file_space = logging::hdf5::dataspace::from_dataset(data_ds.id());
    file_space.select_hyperslab(std::span(offset), std::span(count));

    hsize_t mem_dims[1] = {info->size};
    auto mem_space = logging::hdf5::dataspace::simple(std::span(mem_dims));

    data_ds.read(H5T_NATIVE_FLOAT, mem_space.id(), file_space.id(), out_data.data());

    return true;
}

std::optional<frame_stats> buffer_log_reader::read_frame_stats(std::size_t buffer_id,
                                                                std::size_t frame_idx) {
    if (!file_ || frame_idx >= metadata_.total_frames) {
        return std::nullopt;
    }

    // Open the buffer's stats dataset
    std::string path = "steps/buffers/" + std::to_string(buffer_id) + "/stats";
    auto stats_ds = logging::hdf5::dataset::open(file_.id(), path.c_str());
    if (!stats_ds) {
        return std::nullopt;
    }

    // Read single row
    std::array<float, 5> stats_data;

    hsize_t offset[2] = {frame_idx, 0};
    hsize_t count[2] = {1, 5};

    auto file_space = logging::hdf5::dataspace::from_dataset(stats_ds.id());
    file_space.select_hyperslab(std::span(offset), std::span(count));

    hsize_t mem_dims[1] = {5};
    auto mem_space = logging::hdf5::dataspace::simple(std::span(mem_dims));

    stats_ds.read(H5T_NATIVE_FLOAT, mem_space.id(), file_space.id(), stats_data.data());

    frame_stats stats;
    stats.sum = stats_data[0];
    stats.avg = stats_data[1];
    stats.min_val = stats_data[2];
    stats.max_val = stats_data[3];
    stats.nse = stats_data[4];

    return stats;
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

    // Open the buffer's data dataset
    std::string path = "steps/buffers/" + std::to_string(buffer_id) + "/data";
    auto data_ds = logging::hdf5::dataset::open(file_.id(), path.c_str());
    if (!data_ds) {
        return false;
    }

    // Read multiple rows
    out_data.resize(count * info->size);

    hsize_t offset[2] = {start_frame, 0};
    hsize_t h_count[2] = {count, info->size};

    auto file_space = logging::hdf5::dataspace::from_dataset(data_ds.id());
    file_space.select_hyperslab(std::span(offset), std::span(h_count));

    auto mem_space = logging::hdf5::dataspace::simple(std::span(h_count));

    data_ds.read(H5T_NATIVE_FLOAT, mem_space.id(), file_space.id(), out_data.data());

    return true;
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
