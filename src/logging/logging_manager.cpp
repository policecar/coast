//
// Logging Manager implementation
//

#include "logging/logging_manager.h"
#include "sim_core/sim_env.h"

namespace logging {

logging_manager::logging_manager(sim::env& env)
    : env_(env)
{
}

logging_manager::~logging_manager()
{
    stop_all();
}

// --- Buffer Logging ---

log_result logging_manager::start_buffer_logging(buffer_logger_config config)
{
    if (buffer_logger_ && buffer_logger_->is_logging()) {
        return log_result::error("Buffer logging already active");
    }

    buffer_logger_ = std::make_unique<buffer_logger>(env_, std::move(config));
    return buffer_logger_->start();
}

log_result logging_manager::stop_buffer_logging()
{
    if (!buffer_logger_) {
        return log_result::error("Buffer logger not initialized");
    }
    auto result = buffer_logger_->stop();
    buffer_logger_.reset();
    return result;
}

bool logging_manager::is_buffer_logging() const
{
    return buffer_logger_ && buffer_logger_->is_logging();
}

// --- Probe Logging ---

log_result logging_manager::start_probe_logging(probe_logger_config config)
{
    if (probe_logger_ && probe_logger_->is_logging()) {
        return log_result::error("Probe logging already active");
    }

    probe_logger_ = std::make_unique<probe_logger>(std::move(config));
    return probe_logger_->start();
}

log_result logging_manager::stop_probe_logging()
{
    if (!probe_logger_) {
        return log_result::error("Probe logger not initialized");
    }
    auto result = probe_logger_->stop();
    probe_logger_.reset();
    return result;
}

bool logging_manager::is_probe_logging() const
{
    return probe_logger_ && probe_logger_->is_logging();
}

// --- State Persistence ---

log_result logging_manager::save_state(std::uint64_t step_count, const state_save_config& config)
{
    return state_logger_.save_state(env_, step_count, config);
}

state_load_result logging_manager::load_state(const std::filesystem::path& path)
{
    return state_logger_.load_state(env_, path);
}

// --- Utility ---

void logging_manager::flush_all()
{
    if (buffer_logger_ && buffer_logger_->is_logging()) {
        buffer_logger_->flush();
    }
    if (probe_logger_ && probe_logger_->is_logging()) {
        probe_logger_->flush();
    }
}

void logging_manager::stop_all()
{
    if (buffer_logger_ && buffer_logger_->is_logging()) {
        buffer_logger_->stop();
    }
    buffer_logger_.reset();

    if (probe_logger_ && probe_logger_->is_logging()) {
        probe_logger_->stop();
    }
    probe_logger_.reset();
}

log_stats logging_manager::combined_stats() const
{
    log_stats stats;

    if (buffer_logger_) {
        const auto& bs = buffer_logger_->stats();
        stats.frames_written += bs.frames_written;
        stats.bytes_written += bs.bytes_written;
    }

    if (probe_logger_) {
        const auto& ps = probe_logger_->stats();
        stats.samples_written += ps.samples_written;
        stats.bytes_written += ps.bytes_written;
    }

    return stats;
}

} // namespace logging
