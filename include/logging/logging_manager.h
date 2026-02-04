//
// Logging Manager - coordinates all logging systems
//

#ifndef COAST_LOGGING_MANAGER_H
#define COAST_LOGGING_MANAGER_H

#include "logging_common.h"
#include "buffer_logger.h"
#include "probe_logger.h"
#include "state_logger.h"

#include <memory>
#include <filesystem>

namespace sim {
class env;
class sim_thread_controller;
}

namespace logging {

/// Central coordinator for all logging systems
class logging_manager {
public:
    explicit logging_manager(sim::env& env);
    ~logging_manager();

    logging_manager(const logging_manager&) = delete;
    logging_manager& operator=(const logging_manager&) = delete;

    // --- Buffer Logging ---

    /// Start buffer logging with given configuration
    log_result start_buffer_logging(buffer_logger_config config);

    /// Stop buffer logging
    log_result stop_buffer_logging();

    /// Check if buffer logging is active
    [[nodiscard]] bool is_buffer_logging() const;

    /// Get buffer logger (for status queries)
    [[nodiscard]] buffer_logger* get_buffer_logger() const { return buffer_logger_.get(); }

    // --- Probe Logging ---

    /// Start probe logging with given configuration
    log_result start_probe_logging(probe_logger_config config);

    /// Stop probe logging
    log_result stop_probe_logging();

    /// Check if probe logging is active
    [[nodiscard]] bool is_probe_logging() const;

    /// Get probe logger (for adding/removing probes)
    [[nodiscard]] probe_logger* get_probe_logger() const { return probe_logger_.get(); }

    // --- State Persistence ---

    /// Save current simulation state
    log_result save_state(std::uint64_t step_count, const state_save_config& config);

    /// Load simulation state from file
    state_load_result load_state(const std::filesystem::path& path);

    // --- Utility ---

    /// Flush all active loggers
    void flush_all();

    /// Stop all active logging
    void stop_all();

    /// Get combined statistics from all loggers
    [[nodiscard]] log_stats combined_stats() const;

private:
    sim::env& env_;

    std::unique_ptr<buffer_logger> buffer_logger_;
    std::unique_ptr<probe_logger> probe_logger_;
    state_logger state_logger_;
};

} // namespace logging

#endif // COAST_LOGGING_MANAGER_H
