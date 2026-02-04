//
// IO Buffer Logger implementation
//

#include "logging/buffer_logger.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_buffer.h"
#include "sim_core/io_entity.h"

#include <chrono>
#include <algorithm>
#include <cmath>

namespace logging {

buffer_logger::buffer_logger(sim::env& env, buffer_logger_config config)
    : env_(env)
    , config_(std::move(config))
{
}

buffer_logger::~buffer_logger()
{
    if (logging_active_) {
        stop();
    }
}

log_result buffer_logger::start()
{
    if (logging_active_) {
        return log_result::error("Buffer logger already active");
    }

    // Reset stats
    stats_ = {};
    step_count_ = 0;
    frame_buffer_.clear();

    // Create the HDF5 file
    auto result = create_file();
    if (!result) {
        return result;
    }

    // Write metadata
    result = write_metadata();
    if (!result) {
        file_ = {};  // Close file on error
        return result;
    }

    // Create datasets
    result = create_datasets();
    if (!result) {
        file_ = {};
        return result;
    }

    // Register hook to capture frames after swap_io
    hook_id_ = env_.set_post_swap_hook([this]() {
        capture_frame(step_count_);
        ++step_count_;

        // Flush if buffer is full
        if (frame_buffer_.size() >= config_.chunk_size) {
            write_frames();
        }
    });

    logging_active_ = true;
    return log_result::ok();
}

log_result buffer_logger::stop()
{
    if (!logging_active_) {
        return log_result::error("Buffer logger not active");
    }

    // Unregister hook
    env_.remove_post_swap_hook(hook_id_);
    hook_id_ = 0;

    // Write remaining frames
    if (!frame_buffer_.empty()) {
        write_frames();
    }

    // Flush and close HDF5 resources
    file_.flush();

    // Close datasets explicitly
    buffer_datasets_.clear();
    step_numbers_ = {};
    timestamps_ = {};
    buffers_group_ = {};
    steps_group_ = {};
    metadata_group_ = {};
    file_ = {};

    logging_active_ = false;
    return log_result::ok();
}

void buffer_logger::flush()
{
    if (logging_active_ && !frame_buffer_.empty()) {
        write_frames();
        file_.flush();
    }
}

log_result buffer_logger::create_file()
{
    file_ = hdf5::file::create(config_.output_path.string());
    if (!file_) {
        return log_result::error("Failed to create HDF5 file: " + config_.output_path.string());
    }
    return log_result::ok();
}

log_result buffer_logger::write_metadata()
{
    metadata_group_ = hdf5::group::create(file_.id(), "metadata");
    if (!metadata_group_) {
        return log_result::error("Failed to create metadata group");
    }

    // Write basic attributes
    hdf5::write_attr(metadata_group_.id(), "schema_version", std::string_view("1.0"));
    hdf5::write_attr(metadata_group_.id(), "start_step", static_cast<std::uint64_t>(start_step_));

    // Write config JSON if provided
    if (!config_.config_json.empty()) {
        hdf5::write_attr(metadata_group_.id(), "config", std::string_view(config_.config_json));
    }

    // Write timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&time_t);
    if (!timestamp.empty() && timestamp.back() == '\n') {
        timestamp.pop_back();
    }
    hdf5::write_attr(metadata_group_.id(), "created", std::string_view(timestamp));

    // Extract and write topology
    auto topology = extract_topology();

    auto topo_group = hdf5::group::create(metadata_group_.id(), "topology");
    if (!topo_group) {
        return log_result::error("Failed to create topology group");
    }

    // Write buffer IDs
    std::vector<std::uint64_t> buffer_ids;
    std::vector<std::uint64_t> buffer_sizes;
    buffer_ids.reserve(topology.buffers.size());
    buffer_sizes.reserve(topology.buffers.size());
    for (const auto& buf : topology.buffers) {
        buffer_ids.push_back(buf.id);
        buffer_sizes.push_back(buf.size);
    }

    if (!buffer_ids.empty()) {
        hsize_t dims[1] = {buffer_ids.size()};
        auto space = hdf5::dataspace::simple(std::span(dims));
        auto ds = hdf5::dataset::create(topo_group.id(), "buffer_ids", H5T_NATIVE_UINT64, space.id());
        ds.write(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, buffer_ids.data());

        auto ds_sizes = hdf5::dataset::create(topo_group.id(), "buffer_sizes", H5T_NATIVE_UINT64, space.id());
        ds_sizes.write(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, buffer_sizes.data());
    }

    // Write buffer names as a compound dataset with variable-length strings
    // For simplicity, write as fixed-size padded strings
    if (!topology.buffers.empty()) {
        std::size_t max_name_len = 1;
        for (const auto& buf : topology.buffers) {
            max_name_len = std::max(max_name_len, buf.entity_name.size() + 1);
        }

        hid_t str_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type, max_name_len);
        H5Tset_strpad(str_type, H5T_STR_NULLTERM);

        hsize_t dims[1] = {topology.buffers.size()};
        auto space = hdf5::dataspace::simple(std::span(dims));
        auto ds = hdf5::dataset::create(topo_group.id(), "buffer_names", str_type, space.id());

        // Create padded string buffer
        std::vector<char> names_buf(topology.buffers.size() * max_name_len, '\0');
        for (std::size_t i = 0; i < topology.buffers.size(); ++i) {
            std::copy(topology.buffers[i].entity_name.begin(),
                      topology.buffers[i].entity_name.end(),
                      names_buf.begin() + static_cast<std::ptrdiff_t>(i * max_name_len));
        }
        ds.write(str_type, H5S_ALL, H5S_ALL, names_buf.data());
        H5Tclose(str_type);
    }

    // Write connectivity as edge list (from_id, to_id)
    std::vector<std::uint64_t> edges;
    for (const auto& buf : topology.buffers) {
        for (auto inp_id : buf.input_ids) {
            edges.push_back(inp_id);
            edges.push_back(buf.id);
        }
    }

    if (!edges.empty()) {
        hsize_t dims[2] = {edges.size() / 2, 2};
        auto space = hdf5::dataspace::simple(std::span(dims));
        auto ds = hdf5::dataset::create(topo_group.id(), "connectivity", H5T_NATIVE_UINT64, space.id());
        ds.write(H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, edges.data());
    }

    // Write visualization hints (width, height, num_channels per buffer)
    if (!topology.buffers.empty()) {
        std::vector<std::int32_t> vis_hints_data;
        vis_hints_data.reserve(topology.buffers.size() * 3);
        for (const auto& buf : topology.buffers) {
            vis_hints_data.push_back(buf.vis_hint.width);
            vis_hints_data.push_back(buf.vis_hint.height);
            vis_hints_data.push_back(buf.vis_hint.num_channels);
        }

        hsize_t dims[2] = {topology.buffers.size(), 3};
        auto space = hdf5::dataspace::simple(std::span(dims));
        auto ds = hdf5::dataset::create(topo_group.id(), "vis_hints", H5T_NATIVE_INT32, space.id());
        ds.write(H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, vis_hints_data.data());
    }

    return log_result::ok();
}

log_result buffer_logger::create_datasets()
{
    steps_group_ = hdf5::group::create(file_.id(), "steps");
    if (!steps_group_) {
        return log_result::error("Failed to create steps group");
    }

    buffers_group_ = hdf5::group::create(steps_group_.id(), "buffers");
    if (!buffers_group_) {
        return log_result::error("Failed to create buffers group");
    }

    // Create chunked, extensible datasets for step tracking
    hsize_t initial_dims[1] = {0};
    hsize_t chunk_dims[1] = {config_.chunk_size};

    auto dcpl = hdf5::plist::dataset_create();
    dcpl.set_chunk(std::span(chunk_dims));
    if (config_.compression.algorithm == compression_config::type::gzip) {
        dcpl.set_shuffle();
        dcpl.set_deflate(config_.compression.level);
    }

    auto space = hdf5::dataspace::extensible(std::span(initial_dims));
    step_numbers_ = hdf5::dataset::create(steps_group_.id(), "step_numbers",
                                           H5T_NATIVE_UINT64, space.id(), dcpl.id());
    if (!step_numbers_) {
        return log_result::error("Failed to create step_numbers dataset");
    }

    space = hdf5::dataspace::extensible(std::span(initial_dims));
    timestamps_ = hdf5::dataset::create(steps_group_.id(), "timestamps",
                                         H5T_NATIVE_DOUBLE, space.id(), dcpl.id());
    if (!timestamps_) {
        return log_result::error("Failed to create timestamps dataset");
    }

    // Create per-buffer datasets
    auto& io_buffers = env_.get_io_buffers();
    buffer_datasets_.reserve(io_buffers.size());

    for (auto& [id, buffer] : io_buffers) {
        std::string name = std::to_string(id);
        auto buf_group = hdf5::group::create(buffers_group_.id(), name.c_str());
        if (!buf_group) {
            return log_result::error("Failed to create buffer group: " + name);
        }

        std::size_t buf_size = buffer.size();

        // Data dataset: [time, buffer_size]
        hsize_t data_dims[2] = {0, buf_size};
        hsize_t data_chunk[2] = {config_.chunk_size, buf_size};
        auto data_dcpl = hdf5::plist::dataset_create();
        data_dcpl.set_chunk(std::span(data_chunk));
        if (config_.compression.algorithm == compression_config::type::gzip) {
            data_dcpl.set_shuffle();
            data_dcpl.set_deflate(config_.compression.level);
        }

        auto data_space = hdf5::dataspace::extensible(std::span(data_dims));
        auto data_ds = hdf5::dataset::create(buf_group.id(), "data",
                                              H5T_NATIVE_FLOAT, data_space.id(), data_dcpl.id());
        if (!data_ds) {
            return log_result::error("Failed to create data dataset for buffer: " + name);
        }

        // Stats dataset: [time, 5] (sum, avg, min, max, nse)
        hdf5::dataset stats_ds;
        if (config_.include_stats) {
            hsize_t stats_dims[2] = {0, 5};
            hsize_t stats_chunk[2] = {config_.chunk_size, 5};
            auto stats_dcpl = hdf5::plist::dataset_create();
            stats_dcpl.set_chunk(std::span(stats_chunk));
            if (config_.compression.algorithm == compression_config::type::gzip) {
                stats_dcpl.set_shuffle();
                stats_dcpl.set_deflate(config_.compression.level);
            }

            auto stats_space = hdf5::dataspace::extensible(std::span(stats_dims));
            stats_ds = hdf5::dataset::create(buf_group.id(), "stats",
                                              H5T_NATIVE_FLOAT, stats_space.id(), stats_dcpl.id());
            if (!stats_ds) {
                return log_result::error("Failed to create stats dataset for buffer: " + name);
            }
        }

        buffer_datasets_.push_back({
            .buffer_id = id,
            .buffer_size = buf_size,
            .data = std::move(data_ds),
            .stats = std::move(stats_ds),
            .current_rows = 0
        });
    }

    return log_result::ok();
}

network_topology buffer_logger::extract_topology()
{
    network_topology topology;

    // Helper to look up vis hints from config by entity name
    auto lookup_vis_hint = [this](const std::string& name) -> std::optional<buffer_vis_hint> {
        if (!config_.vis_hints) {
            return std::nullopt;
        }
        const auto& vh = *config_.vis_hints;

        // Try all dataset types
        if (auto it = vh.datasets.mnist.find(name); it != vh.datasets.mnist.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }
        if (auto it = vh.datasets.dsprites.find(name); it != vh.datasets.dsprites.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }
        if (auto it = vh.datasets.emnist.find(name); it != vh.datasets.emnist.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }
        if (auto it = vh.datasets.kmnist.find(name); it != vh.datasets.kmnist.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }
        if (auto it = vh.datasets.cifar10.find(name); it != vh.datasets.cifar10.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }
        if (auto it = vh.datasets.cifar100.find(name); it != vh.datasets.cifar100.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }
        if (auto it = vh.datasets.chairs3d.find(name); it != vh.datasets.chairs3d.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }
        if (auto it = vh.datasets.color_mnist.find(name); it != vh.datasets.color_mnist.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }

        // Try neuron groups
        if (auto it = vh.ngm2.neuron_groups.find(name); it != vh.ngm2.neuron_groups.end()) {
            const auto& h = it->second.ngm_flat_vis;
            return buffer_vis_hint{h.width, h.height, h.num_channels};
        }

        return std::nullopt;
    };

    for (auto& entity : env_.iterate_entities()) {
        buffer_info info;
        info.id = entity.get_outp_id();
        info.size = entity.get_outp_size();

        // Get entity name from structure registry root
        auto* root = entity.structure().root();
        if (root) {
            info.entity_name = root->cached_path;
            if (info.entity_name.empty()) {
                info.entity_name = root->type.name();
            }
        } else {
            info.entity_name = "entity_" + std::to_string(info.id);
        }

        // Get input IDs
        auto inp_ids = entity.get_inp_ids();
        info.input_ids.assign(inp_ids.begin(), inp_ids.end());

        // Try to get vis_hint from simulation config first
        if (auto hint = lookup_vis_hint(info.entity_name)) {
            info.vis_hint = *hint;
        } else {
            // Fall back to auto-detection based on buffer size
            auto size = info.size;
            bool detected = false;

            // First, try to detect RGB images (size divisible by 3, and size/3 is a perfect square)
            if (size % 3 == 0) {
                auto rgb_pixels = size / 3;
                auto sqrt_rgb = static_cast<std::size_t>(std::sqrt(static_cast<double>(rgb_pixels)));
                if (sqrt_rgb * sqrt_rgb == rgb_pixels && sqrt_rgb >= 16) {
                    // Likely an RGB image (e.g., 32x32x3 = 3072)
                    info.vis_hint.width = static_cast<int>(sqrt_rgb);
                    info.vis_hint.height = static_cast<int>(sqrt_rgb);
                    info.vis_hint.num_channels = 3;
                    detected = true;
                } else {
                    // Try common RGB image sizes
                    for (int w : {32, 64, 96, 128, 256}) {
                        if (rgb_pixels % w == 0) {
                            int h = static_cast<int>(rgb_pixels / w);
                            if (h >= 16 && h <= 256) {
                                info.vis_hint.width = w;
                                info.vis_hint.height = h;
                                info.vis_hint.num_channels = 3;
                                detected = true;
                                break;
                            }
                        }
                    }
                }
            }

            // If not detected as RGB, try grayscale detection
            if (!detected) {
                auto sqrt_size = static_cast<std::size_t>(std::sqrt(static_cast<double>(size)));
                if (sqrt_size * sqrt_size == size) {
                    // Perfect square - grayscale
                    info.vis_hint.width = static_cast<int>(sqrt_size);
                    info.vis_hint.height = static_cast<int>(sqrt_size);
                    info.vis_hint.num_channels = 1;
                } else {
                    // Try common widths for grayscale
                    info.vis_hint.width = 0;
                    info.vis_hint.height = 0;
                    for (int w : {28, 32, 64, 96, 128, 256}) {
                        if (size % w == 0) {
                            info.vis_hint.width = w;
                            info.vis_hint.height = static_cast<int>(size / w);
                            break;
                        }
                    }
                    // Fallback: single row
                    if (info.vis_hint.width == 0) {
                        info.vis_hint.width = static_cast<int>(size);
                        info.vis_hint.height = 1;
                    }
                    info.vis_hint.num_channels = 1;
                }
            }
        }

        topology.buffers.push_back(std::move(info));
    }

    return topology;
}

void buffer_logger::capture_frame(std::uint64_t step)
{
    frame_data frame;
    frame.step = step;

    // Get timestamp
    auto now = std::chrono::steady_clock::now();
    static auto start_time = now;
    frame.timestamp = std::chrono::duration<double>(now - start_time).count();

    auto& io_buffers = env_.get_io_buffers();

    // Reserve space matching our dataset order
    frame.buffer_data.reserve(buffer_datasets_.size());
    if (config_.include_stats) {
        frame.buffer_stats.reserve(buffer_datasets_.size());
    }

    // Capture in same order as buffer_datasets_
    for (const auto& bds : buffer_datasets_) {
        auto it = io_buffers.find(bds.buffer_id);
        if (it == io_buffers.end()) {
            // Buffer not found, fill with zeros
            frame.buffer_data.emplace_back(bds.buffer_size, 0.0f);
            if (config_.include_stats) {
                frame.buffer_stats.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
            }
            continue;
        }

        auto& buffer = it->second;
        auto [data, stats] = buffer.inp_buffer_func()();

        frame.buffer_data.emplace_back(data.begin(), data.end());
        if (config_.include_stats) {
            frame.buffer_stats.push_back({
                stats.sum, stats.avg, stats.min_val, stats.max_val, stats.nse
            });
        }
    }

    frame_buffer_.push_back(std::move(frame));
}

void buffer_logger::write_frames()
{
    if (frame_buffer_.empty()) return;

    std::size_t num_frames = frame_buffer_.size();
    std::size_t current_rows = stats_.frames_written;
    std::size_t new_rows = current_rows + num_frames;

    // Extend and write step_numbers
    {
        hsize_t new_dims[1] = {new_rows};
        step_numbers_.extend(std::span(new_dims));

        std::vector<std::uint64_t> steps;
        steps.reserve(num_frames);
        for (const auto& f : frame_buffer_) {
            steps.push_back(f.step);
        }

        hsize_t offset[1] = {current_rows};
        hsize_t count[1] = {num_frames};
        auto file_space = hdf5::dataspace::from_dataset(step_numbers_.id());
        file_space.select_hyperslab(std::span(offset), std::span(count));

        auto mem_space = hdf5::dataspace::simple(std::span(count));
        step_numbers_.write(H5T_NATIVE_UINT64, mem_space.id(), file_space.id(), steps.data());
    }

    // Extend and write timestamps
    {
        hsize_t new_dims[1] = {new_rows};
        timestamps_.extend(std::span(new_dims));

        std::vector<double> times;
        times.reserve(num_frames);
        for (const auto& f : frame_buffer_) {
            times.push_back(f.timestamp);
        }

        hsize_t offset[1] = {current_rows};
        hsize_t count[1] = {num_frames};
        auto file_space = hdf5::dataspace::from_dataset(timestamps_.id());
        file_space.select_hyperslab(std::span(offset), std::span(count));

        auto mem_space = hdf5::dataspace::simple(std::span(count));
        timestamps_.write(H5T_NATIVE_DOUBLE, mem_space.id(), file_space.id(), times.data());
    }

    // Extend and write per-buffer data
    for (std::size_t bi = 0; bi < buffer_datasets_.size(); ++bi) {
        auto& bds = buffer_datasets_[bi];
        std::size_t buf_size = bds.buffer_size;

        // Write data
        {
            hsize_t new_dims[2] = {new_rows, buf_size};
            bds.data.extend(std::span(new_dims));

            // Flatten all frames' data for this buffer
            std::vector<float> data_flat;
            data_flat.reserve(num_frames * buf_size);
            for (const auto& f : frame_buffer_) {
                const auto& bd = f.buffer_data[bi];
                data_flat.insert(data_flat.end(), bd.begin(), bd.end());
            }

            hsize_t offset[2] = {current_rows, 0};
            hsize_t count[2] = {num_frames, buf_size};
            auto file_space = hdf5::dataspace::from_dataset(bds.data.id());
            file_space.select_hyperslab(std::span(offset), std::span(count));

            auto mem_space = hdf5::dataspace::simple(std::span(count));
            bds.data.write(H5T_NATIVE_FLOAT, mem_space.id(), file_space.id(), data_flat.data());

            stats_.bytes_written += data_flat.size() * sizeof(float);
        }

        // Write stats
        if (config_.include_stats && bds.stats) {
            hsize_t new_dims[2] = {new_rows, 5};
            bds.stats.extend(std::span(new_dims));

            std::vector<float> stats_flat;
            stats_flat.reserve(num_frames * 5);
            for (const auto& f : frame_buffer_) {
                const auto& s = f.buffer_stats[bi];
                stats_flat.insert(stats_flat.end(), s.begin(), s.end());
            }

            hsize_t offset[2] = {current_rows, 0};
            hsize_t count[2] = {num_frames, 5};
            auto file_space = hdf5::dataspace::from_dataset(bds.stats.id());
            file_space.select_hyperslab(std::span(offset), std::span(count));

            auto mem_space = hdf5::dataspace::simple(std::span(count));
            bds.stats.write(H5T_NATIVE_FLOAT, mem_space.id(), file_space.id(), stats_flat.data());

            stats_.bytes_written += stats_flat.size() * sizeof(float);
        }

        bds.current_rows = new_rows;
    }

    stats_.frames_written = new_rows;
    frame_buffer_.clear();
}

} // namespace logging
