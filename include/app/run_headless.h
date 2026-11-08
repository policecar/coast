#ifndef COAST_APP_RUN_HEADLESS_H
#define COAST_APP_RUN_HEADLESS_H

namespace coast { class sim_instance; }
namespace coast::cli { struct options; }

namespace coast::paths { struct app_config; }

namespace coast::app {


// Install the --save-on-term SIGTERM catcher.
//
// Call this BEFORE build_env when --save-on-term is active. run_headless calls it
// too (idempotently), but by then build_env has already loaded datasets, restored
// --load-state and initialised the GPU — a window of many seconds during which a
// SIGTERM hit the DEFAULT disposition. The job was then killed with no manifest,
// which coastd's read_outcome cannot distinguish from a crash: an operator cancel
// during startup counted toward the crash-loop guard. Arming first means the flag
// is latched, and run_headless takes its normal teardown path (manifest with
// exit_reason = user_signal) the moment it starts.
//
// Idempotent: a second call does NOT clear a signal that already arrived.
void arm_term_signal();

// Run the simulation in headless mode (no GUI). Drives the sim until either
// the step limit is reached or the controller pauses. Handles checkpointing,
// buffer logging, final state save, and progress reporting.
//
// app_cfg governs WHERE --checkpoint snapshots land: the config's snapshot_dir
// leaf is resolved under app_cfg.paths.config_base (same resolver as build_env's
// resume lookup, so write and resume agree). CLI paths (--save-state,
// --log-buffers, --emit-run-manifest) stay verbatim — never rebased under
// config_base (design §3.0).
//
// Pre: sim is non-null and fully constructed; opts.config_path is set.
// Returns 0 on success, non-zero on error. Does not throw.
int run_headless(sim_instance& sim, const cli::options& opts,
                 const paths::app_config& app_cfg);

} // namespace coast::app

#endif // COAST_APP_RUN_HEADLESS_H
