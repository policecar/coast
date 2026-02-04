//
// IO Buffer Logger - captures all io_buffer contents each simulation step
//

#ifndef COAST_BUFFER_LOGGER_H
#define COAST_BUFFER_LOGGER_H

#include "logging_common.h"
#include "hdf5_utils.h"
#include "gui_vis/core/vis_hints.h"

#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <optional>

namespace sim {
    class env;
    class io_buffer;
}

namespace logging {

/// Configuration for buffer logging
struct buffer_logger_config {
    std::filesystem::path output_path;
    std::size_t chunk_size = 1000;           // Frames per chunk
    compression_config compression;
    bool include_stats = true;               // Log buffer statistics
    std::string config_json;                 // Simulation config for metadata
    std::optional<coast::vis_hints_t> vis_hints;  // Visualization hints from sim config
};

/// Visualization hints for a buffer (stored in HDF5 for replay)
struct buffer_vis_hint {
    int width = 0;           // 0 = auto-detect from size
    int height = 0;
    int num_channels = 1;    // 1 = grayscale, 3 = RGB
};

/// Topology information for a single buffer
struct buffer_info {
    std::size_t id;
    std::string entity_name;
    std::size_t size;
    std::vector<std::size_t> input_ids;      // Which buffers feed into the entity
    buffer_vis_hint vis_hint;                // Visualization hints for replay
};

/// Network topology (connectivity graph)
struct network_topology {
    std::vector<buffer_info> buffers;
};

/// Captures all IO buffer contents at each simulation step
class buffer_logger {
public:
    explicit buffer_logger(sim::env& env, buffer_logger_config config);
    ~buffer_logger();

    buffer_logger(const buffer_logger&) = delete;
    buffer_logger& operator=(const buffer_logger&) = delete;

    /// Start logging (registers hooks, opens file)
    log_result start();

    /// Stop logging (unregisters hooks, closes file)
    log_result stop();

    /// Force flush buffered data to disk
    void flush();

    /// Check if currently logging
    [[nodiscard]] bool is_logging() const { return logging_active_; }

    /// Get statistics
    [[nodiscard]] const log_stats& stats() const { return stats_; }

    /// Get the output file path
    [[nodiscard]] const std::filesystem::path& output_path() const { return config_.output_path; }

private:
    sim::env& env_;
    buffer_logger_config config_;

    bool logging_active_ = false;
    std::size_t hook_id_ = 0;
    std::uint64_t start_step_ = 0;

    log_stats stats_;

    // HDF5 handles
    hdf5::file file_;
    hdf5::group metadata_group_;
    hdf5::group steps_group_;
    hdf5::group buffers_group_;

    // Per-buffer datasets
    struct buffer_datasets {
        std::size_t buffer_id;
        std::size_t buffer_size;
        hdf5::dataset data;
        hdf5::dataset stats;
        std::size_t current_rows = 0;
    };
    std::vector<buffer_datasets> buffer_datasets_;

    // Step tracking datasets
    hdf5::dataset step_numbers_;
    hdf5::dataset timestamps_;
    std::size_t step_count_ = 0;

    // Buffer for batching writes
    struct frame_data {
        std::uint64_t step;
        double timestamp;
        std::vector<std::vector<float>> buffer_data;
        std::vector<std::array<float, 5>> buffer_stats;  // sum, avg, min, max, nse
    };
    std::vector<frame_data> frame_buffer_;

    // Internal methods
    log_result create_file();
    log_result write_metadata();
    log_result create_datasets();
    network_topology extract_topology();

    void capture_frame(std::uint64_t step);
    void write_frames();
};

} // namespace logging

#endif // COAST_BUFFER_LOGGER_H
