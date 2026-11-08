//
// IO Buffer Logger implementation
//

#include "logging/buffer_logger.h"
#include "core/local_time.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_buffer.h"
#include "sim_core/io_entity.h"

#include "diag/diag.h"
#include "hdf5/executor.h"
#include "probe/probe_instance.h"

#include <chrono>
#include <algorithm>
#include <cmath>

namespace logging {

// All HDF5-touching member functions wrap their body in
// h5_exec_.submit_sync. The existing hdf5::* RAII members stay
// — they remain correct because every operation on them (including
// their destructors via explicit reset in stop()) runs on the executor's
// worker thread. See design/hdf5-executor.md.

buffer_logger::buffer_logger(sim::env& env, buffer_logger_config config, hdf5::executor& h5_exec)
    : env_(env)
    , h5_exec_(h5_exec)
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
    published_frames_.store(0, std::memory_order_relaxed);
    published_bytes_.store(0, std::memory_order_relaxed);
    write_failed_.store(false, std::memory_order_relaxed);
    write_error_.clear();
    frame_buffer_.clear();
    start_time_ = config_.clock ? config_.clock() : std::chrono::steady_clock::now();

    // Create the HDF5 file
    auto result = create_file();
    if (!result) {
        return result;
    }

    // Write metadata
    result = write_metadata();
    if (!result) {
        close_hdf5_resources(false);
        return result;
    }

    // Create datasets
    result = create_datasets();
    if (!result) {
        close_hdf5_resources(false);
        return result;
    }

    // Register hook to capture frames after swap_io
    hook_id_ = env_.set_post_swap_hook([this]() {
        // The SIMULATION step this frame belongs to — the counter the controller
        // publishes at the start of each step, the same one probe samples and
        // OBSERVE output carry — not a count of frames since logging began, so
        // a log started mid-run or after --load-state aligns with them (#202).
        capture_frame(probe::get_current_step());

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

    // Precondition (see header): the hook can no longer run concurrently —
    // detach_hook() already ran at a quiesced boundary, or the sim thread is not
    // stepping. Without that, the hook would race this function's write_frames()
    // and the HDF5 teardown below (issue #197).
    detach_hook();

    // Write remaining frames (each write_frames call already wraps itself
    // in submit_sync).
    if (!frame_buffer_.empty()) {
        write_frames();
    }

    close_hdf5_resources(true);

    logging_active_ = false;
    if (!write_ok()) {
        return log_result::error("Buffer log " + config_.output_path.string() +
                                 " is incomplete and may be unreadable: " + write_error_ +
                                 " (after " + std::to_string(stats_.frames_written) +
                                 " frames were accepted)");
    }
    return log_result::ok();
}

void buffer_logger::detach_hook()
{
    if (!hook_id_) return;
    // Synchronous from any thread but the sim thread inside a hook (#235): once
    // this returns, the hook is not running and never runs again.
    env_.remove_post_swap_hook(*hook_id_);
    hook_id_.reset();
}

void buffer_logger::flush()
{
    if (logging_active_ && !frame_buffer_.empty()) {
        write_frames();
        h5_exec_.submit_sync([this] {
            if (file_ && file_.flush() < 0) fail("H5Fflush failed");
        });
    }
}

log_result buffer_logger::create_file()
{
    return h5_exec_.submit_sync([this]() -> log_result {
        file_ = hdf5::file::create(config_.output_path.string());
        if (!file_) {
            return log_result::error("Failed to create HDF5 file: " + config_.output_path.string());
        }
        return log_result::ok();
    });
}

log_result buffer_logger::write_metadata()
{
    return h5_exec_.submit_sync([this]() -> log_result {
    metadata_group_ = hdf5::group::create(file_.id(), "metadata");
    if (!metadata_group_) {
        return log_result::error("Failed to create metadata group");
    }

    // Write basic attributes (fold the write results; warn once if any failed).
    herr_t arc = 0;
    arc |= hdf5::write_attr(metadata_group_.id(), "type", std::string_view("buffer_log"));
    arc |= hdf5::write_attr(metadata_group_.id(), "schema_version", std::string_view("1.0"));

    // Write config JSON if provided
    if (!config_.config_json.empty()) {
        arc |= hdf5::write_attr(metadata_group_.id(), "config", std::string_view(config_.config_json));
    }

    // Write timestamp
    const std::string timestamp =
        core::format_local_time(std::chrono::system_clock::now(), core::k_ctime_format);
    arc |= hdf5::write_attr(metadata_group_.id(), "created", std::string_view(timestamp));
    // WARN, not abort — deliberately unlike state_logger, which fails the commit on
    // the same condition. A CHECKPOINT with missing metadata is unresumable, so
    // publishing one is worse than not writing it; a LOG with missing metadata
    // still carries all its sample data and is worth keeping. The divergence is a
    // severity judgement, not an oversight.
    if (arc < 0) COAST_WARN("logging", "buffer_log: one or more metadata attributes failed to write");

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

    // Topology datasets fail start(), unlike the attributes above: the replay
    // reader needs them to interpret /steps, and a failure this early (nothing
    // logged yet) costs the user nothing to retry (#203).
    auto write_dataset = [&](const char* name, hid_t dtype,
                             std::span<const hsize_t> dims, const void* data) -> bool {
        auto space = hdf5::dataspace::simple(dims);
        auto ds = hdf5::dataset::create(topo_group.id(), name, dtype, space.id());
        return ds && ds.write(dtype, H5S_ALL, H5S_ALL, data) >= 0;
    };

    if (!buffer_ids.empty()) {
        const hsize_t dims[1] = {buffer_ids.size()};
        if (!write_dataset("buffer_ids", H5T_NATIVE_UINT64, dims, buffer_ids.data()) ||
            !write_dataset("buffer_sizes", H5T_NATIVE_UINT64, dims, buffer_sizes.data())) {
            return log_result::error("Failed to write topology buffer_ids/buffer_sizes");
        }
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

        const hsize_t dims[1] = {topology.buffers.size()};

        // Create padded string buffer
        std::vector<char> names_buf(topology.buffers.size() * max_name_len, '\0');
        for (std::size_t i = 0; i < topology.buffers.size(); ++i) {
            std::copy(topology.buffers[i].entity_name.begin(),
                      topology.buffers[i].entity_name.end(),
                      names_buf.begin() + static_cast<std::ptrdiff_t>(i * max_name_len));
        }
        const bool ok = write_dataset("buffer_names", str_type, dims, names_buf.data());
        H5Tclose(str_type);
        if (!ok) return log_result::error("Failed to write topology buffer_names");
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
        const hsize_t dims[2] = {edges.size() / 2, 2};
        if (!write_dataset("connectivity", H5T_NATIVE_UINT64, dims, edges.data()))
            return log_result::error("Failed to write topology connectivity");
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

        const hsize_t dims[2] = {topology.buffers.size(), 3};
        if (!write_dataset("vis_hints", H5T_NATIVE_INT32, dims, vis_hints_data.data()))
            return log_result::error("Failed to write topology vis_hints");
    }

        return log_result::ok();
    });
}

log_result buffer_logger::create_datasets()
{
    return h5_exec_.submit_sync([this]() -> log_result {
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

        std::size_t buf_size = buffer->size();

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

        buffer_datasets_.push_back({
            .buffer_id = id,
            .buffer_size = buf_size,
            .data = std::move(data_ds),
            .current_rows = 0
        });
    }

        return log_result::ok();
    });
}

void buffer_logger::close_hdf5_resources(bool flush_file)
{
    // hdf5_utils handles close directly in their destructors, so all reset
    // operations must run on the HDF5 executor worker. This is used both for
    // normal stop() and for start() failures before logging_active_ is true.
    h5_exec_.submit_sync([this, flush_file] {
        // A full disk usually surfaces only here: H5Dwrite fills the chunk cache
        // and succeeds; the write-out fails. Flush explicitly first, because the
        // dataset handles' H5Dclose would also write out their chunks, report
        // the failure, and have it discarded by the RAII destructor (#203).
        if (flush_file && file_ && file_.flush() < 0) fail("H5Fflush failed");
        buffer_datasets_.clear();
        step_numbers_   = {};
        timestamps_     = {};
        buffers_group_  = {};
        steps_group_    = {};
        metadata_group_ = {};
        // H5Fclose only really closes the file once nothing in it is open.
        if (flush_file && file_.close() < 0) fail("H5Fclose failed");
        file_ = {};
    });
}

network_topology buffer_logger::extract_topology()
{
    network_topology topology;

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

        // The shape of the buffer, guessed from its size (replay renders with it).
        // Config vis hints are keyed by config entry, which the logger cannot see.
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

        topology.buffers.push_back(std::move(info));
    }

    return topology;
}

void buffer_logger::capture_frame(std::uint64_t step)
{
    frame_data frame;
    frame.step = step;

    // Get timestamp
    // Seconds since this logging session's start() (was a process-lifetime
    // static, so a second session's timestamps did not start near 0).
    const auto now = config_.clock ? config_.clock() : std::chrono::steady_clock::now();
    frame.timestamp = std::chrono::duration<double>(now - start_time_).count();

    auto& io_buffers = env_.get_io_buffers();

    // Reserve space matching our dataset order
    frame.buffer_data.reserve(buffer_datasets_.size());

    // Capture in same order as buffer_datasets_
    for (const auto& bds : buffer_datasets_) {
        auto it = io_buffers.find(bds.buffer_id);
        if (it == io_buffers.end()) {
            // Buffer not found, fill with zeros
            frame.buffer_data.emplace_back(bds.buffer_size, 0.0f);
            continue;
        }

        auto& buffer = it->second;
        auto data = buffer->cur_read_buffer();

        if (data.empty() && bds.buffer_size > 0) {
            // GPU-resident buffer with no standing CPU mirror (io_buffer_gpu):
            // pull the current read page on demand. On failure, record a
            // correct-width zero row so the frame stays shape-consistent with
            // the dataset (an empty row would be a width mismatch).
            std::vector<float> tmp(bds.buffer_size);
            if (buffer->read_back_sync(tmp)) {
                frame.buffer_data.push_back(std::move(tmp));
            } else {
                frame.buffer_data.emplace_back(bds.buffer_size, 0.0f);
            }
            continue;
        }

        frame.buffer_data.emplace_back(data.begin(), data.end());
    }

    frame_buffer_.push_back(std::move(frame));
}

void buffer_logger::fail(std::string what)
{
    if (!write_ok()) return;   // keep the first cause
    COAST_ERROR("logging", "buffer_log {}: {}; no further frames are written",
                config_.output_path.string(), what);
    write_error_ = std::move(what);
    write_failed_.store(true, std::memory_order_relaxed);
}

void buffer_logger::write_frames()
{
    if (frame_buffer_.empty()) return;

    h5_exec_.submit_sync([this] {
    // After a failure the row offsets and the file's extents may disagree, so
    // later batches are dropped rather than written at the wrong rows (#203).
    if (!write_ok()) {
        frame_buffer_.clear();
        return;
    }

    std::size_t num_frames = frame_buffer_.size();
    std::size_t current_rows = stats_.frames_written;
    std::size_t new_rows = current_rows + num_frames;

    // start_step = the step of the first frame actually logged. start() cannot
    // know it — logging may start while the sim runs — so it is written with
    // the first batch. An empty log has no start_step (the reader defaults 0).
    if (current_rows == 0 &&
        hdf5::write_attr(metadata_group_.id(), "start_step",
                         static_cast<std::uint64_t>(frame_buffer_.front().step)) < 0) {
        fail("failed to write the start_step attribute");
        frame_buffer_.clear();
        return;
    }

    // This batch's rows [current_rows, new_rows) of `ds` (hdf5::append_rows).
    auto append = [&](hdf5::dataset& ds, std::span<const hsize_t> row_shape,
                      hid_t mem_type, const void* data) -> bool {
        return hdf5::append_rows(ds.id(), current_rows, num_frames, row_shape,
                                 mem_type, data) >= 0;
    };

    std::vector<std::uint64_t> steps;
    std::vector<double>        times;
    steps.reserve(num_frames);
    times.reserve(num_frames);
    for (const auto& f : frame_buffer_) {
        steps.push_back(f.step);
        times.push_back(f.timestamp);
    }
    if (!append(step_numbers_, {}, H5T_NATIVE_UINT64, steps.data())) {
        fail("failed to append to steps/step_numbers");
        frame_buffer_.clear();
        return;
    }
    if (!append(timestamps_, {}, H5T_NATIVE_DOUBLE, times.data())) {
        fail("failed to append to steps/timestamps");
        frame_buffer_.clear();
        return;
    }

    // Extend and write per-buffer data
    std::uint64_t batch_bytes = 0;
    for (std::size_t bi = 0; bi < buffer_datasets_.size(); ++bi) {
        auto& bds = buffer_datasets_[bi];
        std::size_t buf_size = bds.buffer_size;

        // Flatten all frames' data for this buffer
        std::vector<float> data_flat;
        data_flat.reserve(num_frames * buf_size);
        for (const auto& f : frame_buffer_) {
            const auto& bd = f.buffer_data[bi];
            data_flat.insert(data_flat.end(), bd.begin(), bd.end());
        }

        const hsize_t row_shape[1] = {buf_size};
        if (!append(bds.data, row_shape, H5T_NATIVE_FLOAT, data_flat.data())) {
            fail("failed to append to steps/buffers/" + std::to_string(bds.buffer_id) + "/data");
            frame_buffer_.clear();
            return;
        }
        batch_bytes += data_flat.size() * sizeof(float);
        bds.current_rows = new_rows;
    }

    // Count the batch only once all of it is in: stats_.frames_written is the
    // next batch's row offset and what stop() reports as written.
    stats_.bytes_written += batch_bytes;
    stats_.frames_written = new_rows;
    published_frames_.store(stats_.frames_written, std::memory_order_relaxed);
    published_bytes_.store(stats_.bytes_written, std::memory_order_relaxed);
    frame_buffer_.clear();
    });
}

} // namespace logging
