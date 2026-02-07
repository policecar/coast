//
// HDF5 reader for buffer_log files
//

#ifndef COAST_BUFFER_LOG_READER_H
#define COAST_BUFFER_LOG_READER_H

#include "logging/hdf5_utils.h"

#include <string>
#include <vector>
#include <filesystem>
#include <optional>
#include <span>
#include <array>

namespace coast::replay {

/// Visualization hints from the log file
struct buffer_vis_hint {
    int width = 0;           // 0 = auto-detect from size
    int height = 0;
    int num_channels = 1;    // 1 = grayscale, 3 = RGB
};

/// Buffer information from the log file topology
struct buffer_info {
    std::size_t id;
    std::string name;
    std::size_t size;
    std::vector<std::size_t> input_ids;  // IDs of buffers feeding into this one
    buffer_vis_hint vis_hint;            // Visualization hints (may be default if not in file)
};

/// Network topology (connectivity graph)
struct replay_topology {
    std::vector<buffer_info> buffers;
};

/// Statistics for a single frame of a buffer
struct frame_stats {
    float sum = 0.0f;
    float avg = 0.0f;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float nse = 0.0f;
};

/// Metadata from the buffer log file
struct log_metadata {
    std::string schema_version;
    std::uint64_t start_step = 0;
    std::string config_json;
    std::string created;
    std::size_t total_frames = 0;
};

/// Reader for HDF5 buffer_log files produced by buffer_logger
class buffer_log_reader {
public:
    buffer_log_reader() = default;
    ~buffer_log_reader();

    buffer_log_reader(const buffer_log_reader&) = delete;
    buffer_log_reader& operator=(const buffer_log_reader&) = delete;
    buffer_log_reader(buffer_log_reader&&) noexcept;
    buffer_log_reader& operator=(buffer_log_reader&&) noexcept;

    /// Open a buffer_log HDF5 file
    /// @return true if successful
    bool open(const std::filesystem::path& path);

    /// Close the file
    void close();

    /// Check if a file is open
    [[nodiscard]] bool is_open() const { return static_cast<bool>(file_); }

    /// Get the file path
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

    /// Get file metadata
    [[nodiscard]] const log_metadata& metadata() const { return metadata_; }

    /// Get network topology
    [[nodiscard]] const replay_topology& topology() const { return topology_; }

    /// Get total number of frames in the file
    [[nodiscard]] std::size_t frame_count() const { return metadata_.total_frames; }

    /// Get number of buffers
    [[nodiscard]] std::size_t buffer_count() const { return topology_.buffers.size(); }

    /// Get step number for a frame index
    [[nodiscard]] std::uint64_t get_step(std::size_t frame_idx) const;

    /// Get timestamp for a frame index
    [[nodiscard]] double get_timestamp(std::size_t frame_idx) const;

    /// Read buffer data for a specific frame
    /// @param buffer_id The buffer ID
    /// @param frame_idx The frame index (0-based)
    /// @param out_data Output vector for buffer data
    /// @return true if successful
    bool read_frame_data(std::size_t buffer_id, std::size_t frame_idx,
                         std::vector<float>& out_data);

    /// Read buffer statistics for a specific frame
    /// @param buffer_id The buffer ID
    /// @param frame_idx The frame index (0-based)
    /// @return Statistics, or empty optional if not available
    [[nodiscard]] std::optional<frame_stats> read_frame_stats(std::size_t buffer_id,
                                                               std::size_t frame_idx);

    /// Read a range of frames for a buffer
    /// @param buffer_id The buffer ID
    /// @param start_frame Starting frame index
    /// @param count Number of frames to read
    /// @param out_data Output vector (size will be count * buffer_size)
    bool read_frame_range(std::size_t buffer_id, std::size_t start_frame,
                          std::size_t count, std::vector<float>& out_data);

    /// Get buffer info by ID
    [[nodiscard]] const buffer_info* get_buffer(std::size_t id) const;

    /// Get buffer index by ID
    [[nodiscard]] std::optional<std::size_t> get_buffer_index(std::size_t id) const;

private:
    std::filesystem::path path_;
    logging::hdf5::file file_;
    log_metadata metadata_;
    replay_topology topology_;

    // Cached step/timestamp data
    mutable std::vector<std::uint64_t> step_numbers_;
    mutable std::vector<double> timestamps_;
    mutable bool steps_loaded_ = false;

    // Internal methods
    bool read_metadata();
    bool read_topology();
    void load_step_data() const;
};

} // namespace coast::replay

#endif // COAST_BUFFER_LOG_READER_H
