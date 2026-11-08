//
// Logging Manager - coordinates all logging systems
//

#ifndef COAST_LOGGING_MANAGER_H
#define COAST_LOGGING_MANAGER_H

#include "logging/logging_common.h"
#include "logging/buffer_logger.h"
#include "logging/state_logger.h"

#include <expected>
#include <memory>
#include <string>
#include <filesystem>

namespace sim {
class env;
class sim_thread_controller;
}

namespace hdf5 { class executor; }

namespace logging {

/// Central coordinator for all logging systems
class logging_manager {
public:
    /// @param env      The simulation environment to log from.
    /// @param h5_exec  Process-wide HDF5 executor. Must outlive the manager.
    ///                 All HDF5 work (buffer logging, state save / load) is
    ///                 routed through this executor's worker thread.
    logging_manager(sim::env& env, hdf5::executor& h5_exec);
    ~logging_manager();

    logging_manager(const logging_manager&) = delete;
    logging_manager& operator=(const logging_manager&) = delete;

    // --- Buffer Logging ---

    /// Start buffer logging with given configuration
    log_result start_buffer_logging(buffer_logger_config config);

    /// Stop buffer logging and destroy the logger. The logger's post-swap hook
    /// is detached inside ctrl.run_quiesced, so it can no longer run when the
    /// file is finished and the logger freed — safe while the sim is stepping
    /// (issue #197). If the controller is shutting down the quiesce is refused:
    /// the logger is then kept (still active) and an error returned; stop_all()
    /// in ~logging_manager finishes it once the sim thread has been joined.
    log_result stop_buffer_logging(sim::sim_thread_controller& ctrl);

    /// Check if buffer logging is active
    [[nodiscard]] bool is_buffer_logging() const;

    /// Get buffer logger (for status queries)
    [[nodiscard]] buffer_logger* get_buffer_logger() const { return buffer_logger_.get(); }

    // --- State Persistence (quiesced-state seam; design §4-§5) ---

    /// Sim state extracted on the sim thread, plus the step it was captured at.
    struct extracted_state {
        mem_group     tree;
        std::uint64_t step = 0;
    };

    /// Extract full sim state on the SIM thread at a quiescent boundary
    /// (ctrl.run_quiesced): snapshot buffers (incl. GPU readback) + entity
    /// save_state into an in-memory tree, with the step counter read coherently.
    /// The sim thread blocks only for the (fast) extraction; the heavy HDF5 write
    /// is a separate serialize_state() step on the executor.
    extracted_state extract_state(sim::sim_thread_controller& ctrl);

    /// extract_state() for the GUI save paths, which have no caller above them
    /// to catch a throw. Refuses a halted controller (#229), checked inside the
    /// quiesced task, so a fault that lands while the task waits is seen and
    /// the faulted env is never read. An exception from the extraction (e.g. a
    /// failed GPU readback) is returned as the error instead of propagating.
    std::expected<extracted_state, std::string>
    try_extract_state(sim::sim_thread_controller& ctrl);

    /// Serialize an extracted tree to HDF5 on the executor worker. Returns the
    /// write future — await for durability (headless manifest needs this); the
    /// GUI may drop it (fire-and-forget durability via the executor).
    std::future<log_result> serialize_state(mem_group tree, const state_save_config& config);

    /// Runtime load: parse on the executor, apply on the SIM thread
    /// (ctrl.run_quiesced), and set the step counter — in one call. Returns the
    /// merged parse+apply result (schema + buffer warnings). If an entity's
    /// load_state throws (anything, not only std::exception), what was applied
    /// stays applied: the task then halts the controller (ctrl.halt, kind
    /// partial_load) before another step can run, leaves the step counter as it
    /// was, and the result is failed.
    /// Within one group, only ngm3 refuses as a whole (its load_state stages
    /// the payload); ngm2 and ngm25 can be left partly loaded.
    state_load_result load_state(sim::sim_thread_controller& ctrl,
                                 const std::filesystem::path& path);

    /// Startup load (build_env exemption, design §5.2): parse + apply inline on
    /// the executor, synchronously, with NO run_quiesced. Valid ONLY at
    /// construction, when the main thread is the sole user of the wgpu queue and
    /// the sole writer of sim state (sim thread parked, never stepped). Must NOT
    /// be used once stepping has begun — runtime loads use load_state() above.
    state_load_result load_state_startup(const std::filesystem::path& path);

    // --- Utility ---

    /// Flush all active loggers. Inherits buffer_logger::flush()'s
    /// precondition: inside run_quiesced, or with the sim not stepping.
    void flush_all();

    /// Stop all active logging. Does NOT quiesce: precondition is that the sim
    /// thread is not stepping (sim_instance destroys the controller first).
    void stop_all();

    /// Statistics of the buffer logger: the running session's, or — once it is
    /// stopped — the final counts of the last session (including the frames the
    /// stop itself flushed), until the next start_buffer_logging().
    [[nodiscard]] log_stats combined_stats() const;

private:
    sim::env&         env_;
    hdf5::executor& h5_exec_;

    std::unique_ptr<buffer_logger> buffer_logger_;
    log_stats last_buffer_stats_;   // final counts of the last stopped session
    state_logger state_logger_;
};

} // namespace logging

#endif // COAST_LOGGING_MANAGER_H
