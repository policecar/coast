//
// Logging Manager implementation
//

#include "logging/logging_manager.h"

#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"

#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace logging {

logging_manager::logging_manager(sim::env& env, hdf5::executor& h5_exec)
    : env_(env), h5_exec_(h5_exec)
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

    last_buffer_stats_ = {};
    buffer_logger_ = std::make_unique<buffer_logger>(env_, std::move(config), h5_exec_);
    return buffer_logger_->start();
}

log_result logging_manager::stop_buffer_logging(sim::sim_thread_controller& ctrl)
{
    if (!buffer_logger_) {
        return log_result::error("Buffer logger not initialized");
    }
    // Detach the hook ON the sim thread between steps (#197), so once
    // run_quiesced returns the hook never runs again and stop() below owns the
    // logger's state exclusively. Since #235 env hook removal is synchronous from
    // any thread, which makes the quiesce redundant here; it is kept so the #197
    // path (and its refusal while the controller stops) is unchanged.
    try {
        ctrl.run_quiesced([&] { buffer_logger_->detach_hook(); });
    } catch (const std::exception& e) {
        // Refused (controller stopping): the sim thread may still be finishing a
        // step, so the logger must not be finished or freed here.
        return log_result::error(std::string("Buffer logging not stopped: ") + e.what());
    }
    auto result = buffer_logger_->stop();
    last_buffer_stats_ = buffer_logger_->stats();   // after stop() flushed the tail
    buffer_logger_.reset();
    return result;
}

bool logging_manager::is_buffer_logging() const
{
    return buffer_logger_ && buffer_logger_->is_logging();
}

// --- State Persistence (quiesced-state seam) ---

logging_manager::extracted_state
logging_manager::extract_state(sim::sim_thread_controller& ctrl)
{
    std::uint64_t step = 0;
    // Extraction runs ON the sim thread at the post-swap boundary, where the read
    // pages and wgpu queue are quiescent and the step counter is frozen. Reading
    // the step inside the task guarantees it matches the extracted content.
    mem_group tree = ctrl.run_quiesced([&]() -> mem_group {
        step = ctrl.get_step_count();
        return state_logger::extract_state(env_, step);
    });
    return { std::move(tree), step };
}

std::expected<logging_manager::extracted_state, std::string>
logging_manager::try_extract_state(sim::sim_thread_controller& ctrl)
{
    try {
        std::uint64_t step = 0;
        std::optional<mem_group> tree =
            ctrl.run_quiesced([&]() -> std::optional<mem_group> {
                if (ctrl.is_halted()) return std::nullopt;
                step = ctrl.get_step_count();
                return state_logger::extract_state(env_, step);
            });
        if (!tree)
            return std::unexpected(std::string("the simulation is halted (fault or partial load)"));
        return extracted_state{ std::move(*tree), step };
    } catch (const std::exception& e) {
        return std::unexpected(std::string(e.what()));
    } catch (...) {
        return std::unexpected(std::string("unknown exception"));
    }
}

std::future<log_result>
logging_manager::serialize_state(mem_group tree, const state_save_config& config)
{
    return state_logger::serialize_state(std::move(tree), config, h5_exec_);
}

state_load_result
logging_manager::load_state(sim::sim_thread_controller& ctrl, const std::filesystem::path& path)
{
    // Parse on the executor worker (HDF5); apply on the sim thread (no HDF5).
    // parse_state reports a refused file as an error; the catch is a backstop,
    // since its callers (the GUI's Load state / Resume) catch nothing and an
    // escaping throw would terminate the process.
    state_logger::parse_result pr;
    try {
        pr = state_logger::parse_state(path, h5_exec_);
    } catch (const std::exception& e) {
        pr.result.error_message = "state file " + path.string() + ": " + e.what();
    } catch (...) {
        pr.result.error_message = "state file " + path.string() + ": unknown exception";
    }
    if (!pr.result.success) {
        return pr.result;   // file open / metadata error — nothing applied
    }

    const std::uint64_t step = pr.result.restored_step;
    auto applied = ctrl.run_quiesced([&]() -> state_load_result {
        state_load_result failed;
        try {
            auto r = state_logger::apply_state(env_, pr.tree);
            // Set the step counter atomically with apply, on the sim thread.
            ctrl.set_step_count(step);
            return r;
        } catch (const std::exception& e) {
            failed.error_message = e.what();
        } catch (...) {
            failed.error_message = "unknown exception";
        }
        // What was applied before the throw stays applied, so the env is partly
        // loaded. Halt here, on the sim thread inside the task, so no step runs
        // on it; the step counter keeps its value.
        ctrl.halt(core::step_error{ core::error_kind::partial_load,
                                    core::step_error::no_entity, COAST_SITE });
        return failed;
    });
    if (!applied) return applied;

    // The parse's metadata; the parse itself adds no warnings (parse_open_file).
    applied.config_json   = pr.result.config_json;
    applied.restored_step = step;
    return applied;
}

state_load_result logging_manager::load_state_startup(const std::filesystem::path& path)
{
    // Monolithic parse+apply inline on the executor — the build_env startup
    // exemption (design §5.2). Safe only at construction (sim thread parked,
    // never stepped); the caller sets the step counter.
    return state_logger_.load_state(env_, path, h5_exec_);
}

// --- Utility ---

void logging_manager::flush_all()
{
    if (buffer_logger_ && buffer_logger_->is_logging()) {
        buffer_logger_->flush();
    }
}

void logging_manager::stop_all()
{
    if (buffer_logger_ && buffer_logger_->is_logging()) {
        buffer_logger_->stop();
    }
    if (buffer_logger_) last_buffer_stats_ = buffer_logger_->stats();
    buffer_logger_.reset();
}

log_stats logging_manager::combined_stats() const
{
    // Atomic snapshot (#214) while a logger exists; afterwards its final counts.
    return buffer_logger_ ? buffer_logger_->stats() : last_buffer_stats_;
}

} // namespace logging
