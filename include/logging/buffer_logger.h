//
// IO Buffer Logger - captures all io_buffer contents each simulation step
//

#ifndef COAST_BUFFER_LOGGER_H
#define COAST_BUFFER_LOGGER_H

#include "logging/logging_common.h"
#include "logging/hdf5_utils.h"

#include <chrono>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
#include <array>
#include <atomic>
#include <cstdint>
#include <optional>

namespace sim {
    class env;
    class io_buffer;
}

namespace hdf5 { class executor; }

namespace logging {

/// Configuration for buffer logging
struct buffer_logger_config {
    std::filesystem::path output_path;
    std::size_t chunk_size = 1000;           // Frames per chunk
    compression_config compression;
    std::string config_json;                 // Simulation config for metadata
    /// Time source for the frame timestamps (seconds since start()). Empty =
    /// std::chrono::steady_clock::now. Injectable so a test can pin the time
    /// origin exactly instead of depending on how fast the machine runs. Called
    /// from start() and from the sim thread; must be thread-safe.
    std::function<std::chrono::steady_clock::time_point()> clock;
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

/// Captures all IO buffer contents at each simulation step.
///
/// All HDF5 work is submitted to the process-wide hdf5::executor; the
/// hooks called on the sim thread enqueue work but do not enter libhdf5
/// directly. See design/hdf5-executor.md.
class buffer_logger {
public:
    buffer_logger(sim::env& env, buffer_logger_config config, hdf5::executor& h5_exec);
    ~buffer_logger();

    buffer_logger(const buffer_logger&) = delete;
    buffer_logger& operator=(const buffer_logger&) = delete;

    /// Start logging (registers hooks, opens file)
    log_result start();

    /// Remove the post-swap hook. env hook removal is synchronous (#235): once
    /// this returns, the hook is not running and never runs again — on the sim
    /// thread between steps (inside a run_quiesced task, as
    /// logging_manager::stop_buffer_logging does) or on any other thread.
    /// Idempotent.
    void detach_hook();

    /// Stop logging: detach the hook (if still attached), write the remaining
    /// frames, close the file. Returns an error if any write of this session
    /// failed (write_ok() false), including a failure the final close surfaces:
    /// the file is then incomplete. Precondition: the hook cannot run concurrently —
    /// detach_hook() already ran inside run_quiesced, or the sim thread is not
    /// stepping (joined, or never started). Callers with a live sim thread go
    /// through logging_manager::stop_buffer_logging(ctrl).
    log_result stop();

    /// Force flush buffered data to disk. Same precondition as stop(): it
    /// touches the frame buffer the hook appends to, so call it inside
    /// run_quiesced (sim thread, between steps) or with the sim not stepping.
    void flush();

    /// Check if currently logging
    [[nodiscard]] bool is_logging() const { return logging_active_; }

    /// Snapshot of the counters published after each written batch. Safe from
    /// any thread (issue #214): stats_ itself is written by write_frames() on
    /// the HDF5 executor while the GUI / headless main thread polls this.
    [[nodiscard]] log_stats stats() const
    {
        log_stats s;
        s.frames_written = published_frames_.load(std::memory_order_relaxed);
        s.bytes_written  = published_bytes_.load(std::memory_order_relaxed);
        return s;
    }

    /// False once any HDF5 write of this session has failed (issue #203): disk
    /// full, file size limit, I/O error. Sticky until the next start(); after
    /// the failure no further frames are written. frames_written then counts
    /// the frames HDF5 accepted, NOT what is readable: HDF5 caches chunks and
    /// metadata, so a file that hit a full disk usually cannot be opened at
    /// all. Safe from any thread.
    [[nodiscard]] bool write_ok() const
    {
        return !write_failed_.load(std::memory_order_relaxed);
    }

    /// Get the output file path
    [[nodiscard]] const std::filesystem::path& output_path() const { return config_.output_path; }

private:
    sim::env&         env_;
    hdf5::executor& h5_exec_;
    buffer_logger_config config_;

    bool logging_active_ = false;
    // Empty = no hook attached. Not a 0 sentinel: env hook ids start at 0.
    std::optional<std::size_t> hook_id_;
    std::chrono::steady_clock::time_point start_time_{};  // timestamp origin

    // Writer-side counters: touched only inside write_frames() (on the executor,
    // ordered by submit_sync with the hook) and reset by start() before the hook
    // is attached. stats_.frames_written is also the next batch's row offset.
    log_stats stats_;
    // What stats() reports. Relaxed: each is a standalone counter, and no other
    // data is published through them.
    std::atomic<std::uint64_t> published_frames_{0};
    std::atomic<std::uint64_t> published_bytes_{0};

    // Sticky write failure (#203). Set on the executor by fail(); read by
    // write_ok() from any thread. The message is executor-side state like
    // stats_, read by stop() after its last submit_sync.
    std::atomic<bool> write_failed_{false};
    std::string       write_error_;

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
        std::size_t current_rows = 0;
    };
    std::vector<buffer_datasets> buffer_datasets_;

    // Step tracking datasets
    hdf5::dataset step_numbers_;
    hdf5::dataset timestamps_;

    // Buffer for batching writes
    struct frame_data {
        std::uint64_t step;
        double timestamp;
        std::vector<std::vector<float>> buffer_data;
    };
    std::vector<frame_data> frame_buffer_;

    // Internal methods
    log_result create_file();
    log_result write_metadata();
    log_result create_datasets();
    void close_hdf5_resources(bool flush_file);
    network_topology extract_topology();

    void capture_frame(std::uint64_t step);
    void write_frames();
    void fail(std::string what);   // executor only: record the first failure
};

} // namespace logging

#endif // COAST_BUFFER_LOGGER_H
