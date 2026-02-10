//
// Common types and utilities for the logging system
//

#ifndef COAST_LOGGING_COMMON_H
#define COAST_LOGGING_COMMON_H

#include <cstdint>
#include <string>

namespace logging {

/// Result of a logging operation
struct log_result {
    bool success = true;
    std::string error_message;

    static log_result ok() { return {true, {}}; }
    static log_result error(std::string msg) { return {false, std::move(msg)}; }

    explicit operator bool() const { return success; }
};

/// Statistics for logging sessions
struct log_stats {
    std::uint64_t frames_written = 0;
    std::uint64_t bytes_written = 0;
    std::uint64_t samples_written = 0;
    double elapsed_seconds = 0.0;
};

/// Configuration for compression
struct compression_config {
    enum class type { none, gzip, lzf };
    type algorithm = type::gzip;
    int level = 4;  // 1-9 for gzip, ignored for lzf/none
};

/// Configuration for chunking (HDF5)
struct chunk_config {
    std::size_t time_chunk = 1000;      // Chunk size along time dimension
    std::size_t spatial_chunk = 0;       // 0 = auto (use full spatial extent)
};

} // namespace logging

#endif // COAST_LOGGING_COMMON_H
