//
// Probe Logger - file-streaming observer for the probe system
//

#ifndef COAST_PROBE_LOGGER_H
#define COAST_PROBE_LOGGER_H

#include "logging_common.h"
#include "hdf5_utils.h"

#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>

namespace probe {
struct probe_instance;
}

namespace logging {

/// Output format for probe logging
enum class probe_log_format {
    hdf5,   // Structured, compressed, suitable for large datasets
    csv     // Human-readable, quick analysis
};

/// Configuration for probe logging
struct probe_logger_config {
    std::filesystem::path output_path;
    probe_log_format format = probe_log_format::hdf5;
    compression_config compression;
    std::size_t buffer_size = 10000;     // Samples to buffer before flush
};

/// Streams probe values to file
class probe_logger {
public:
    explicit probe_logger(probe_logger_config config);
    ~probe_logger();

    probe_logger(const probe_logger&) = delete;
    probe_logger& operator=(const probe_logger&) = delete;

    /// Start logging (opens file)
    log_result start();

    /// Stop logging (closes file)
    log_result stop();

    /// Force flush buffered data to disk
    void flush();

    /// Check if currently logging
    [[nodiscard]] bool is_logging() const { return logging_active_; }

    /// Add a probe to log (creates file_observer automatically)
    /// Returns false if probe is already being logged
    bool add_probe(probe::probe_instance* probe);

    /// Remove a probe from logging
    void remove_probe(probe::probe_instance* probe);

    /// Get all probes currently being logged
    [[nodiscard]] std::vector<probe::probe_instance*> active_probes() const;

    /// Get statistics
    [[nodiscard]] const log_stats& stats() const { return stats_; }

    /// Get the output file path
    [[nodiscard]] const std::filesystem::path& output_path() const { return config_.output_path; }

    // Internal: called by file_observer to record values
    void record_value(const std::string& probe_path, std::uint64_t step, double value);

private:
    probe_logger_config config_;

    bool logging_active_ = false;
    log_stats stats_;

    // Map from probe instance to its file observer (kept to remove later)
    std::unordered_map<probe::probe_instance*, std::size_t> probe_to_index_;

    // Per-probe data buffers
    struct probe_buffer {
        std::string path;
        std::vector<std::uint64_t> steps;
        std::vector<double> values;
        std::uint64_t total_samples = 0;
    };
    std::vector<probe_buffer> probe_buffers_;

    // HDF5 resources (when format == hdf5)
    hdf5::file hdf5_file_;
    hdf5::group probes_group_;
    struct hdf5_probe_dataset {
        std::string path;
        hdf5::dataset steps;   // Dataset for step numbers
        hdf5::dataset values;  // Dataset for values
        std::size_t current_rows = 0;
    };
    std::vector<hdf5_probe_dataset> hdf5_datasets_;

    // CSV resources (when format == csv)
    std::ofstream csv_file_;
    bool csv_header_written_ = false;

    // Internal methods
    log_result start_hdf5();
    log_result start_csv();
    void flush_hdf5();
    void flush_csv();
    void stop_hdf5();
    void stop_csv();

    void ensure_hdf5_dataset(std::size_t probe_index);
};

} // namespace logging

#endif // COAST_PROBE_LOGGER_H
