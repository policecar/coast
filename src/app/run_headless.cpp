#include "app/run_headless.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "app/headless_checkpoint.h"
#include "app/run_provenance.h"
#include "cli/options.h"
#include "core/app_config.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"
#include "logging/logging.h"
#include "logging/logging_manager.h"
#include "sim_core/run_manifest.h"

#include "core/step_error.h"
#include "diag/diag.h"
#include "diag/progress.h"

namespace coast::app {

namespace {
// Set by the SIGTERM handler when --save-on-term is active. A run loop watches it
// and stops early so a final checkpoint is written before exit.
//
// std::atomic<int>, not `volatile sig_atomic_t`: SIGTERM is delivered to an
// arbitrary thread while the sim jthread and the HDF5 executor are both running,
// so the plain-volatile form is formally a data race between the handler and the
// loop that reads it (and TSan reports it as one). std::atomic is guaranteed
// handler-safe since C++20 when it is lock-free, which int is on every target
// here — asserted below rather than assumed.
std::atomic<int> g_term_requested{0};
static_assert(std::atomic<int>::is_always_lock_free,
              "the SIGTERM flag must be lock-free to be async-signal-safe");
extern "C" void on_term_signal(int) { g_term_requested.store(1, std::memory_order_relaxed); }

// Armed once per process; a second call must not clear a signal already latched.
bool g_term_armed = false;
}  // namespace

void arm_term_signal() {
    if (g_term_armed) return;
    g_term_armed     = true;
    g_term_requested.store(0, std::memory_order_relaxed);
    std::signal(SIGTERM, on_term_signal);
}

namespace {

// --checkpoint N: resolve the snapshot directory and build the checkpointer, or
// nullptr when checkpointing is off or has nowhere to write. The config's
// snapshot_dir leaf is placed under app_cfg.paths.config_base (an absolute leaf
// still wins) — the same resolver as build_env's resume lookup, so a checkpoint
// written here is found by a later --resume.
std::shared_ptr<headless_checkpointer> make_checkpointer(sim_instance& sim,
                                                         const cli::options& opts,
                                                         const paths::app_config& app_cfg,
                                                         std::uint64_t start_step,
                                                         std::uint64_t target_step)
{
    if (opts.checkpoint_interval == 0) return nullptr;

    const std::filesystem::path dir =
        paths::resolve_snapshot_output_dir(opts.config_path, sim.get_config().snapshot_dir, app_cfg);

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
        COAST_WARN("checkpoint", "could not create snapshot directory: {}", dir.string());
    if (dir.empty()) {
        COAST_WARN("checkpoint", "--checkpoint set but snapshot_dir is empty; "
                                 "no checkpoints will be written");
        return nullptr;
    }
    auto* ctrl = sim.get_thread_controller();
    return std::make_shared<headless_checkpointer>(
        opts.checkpoint_interval, start_step, target_step, dir,
        std::filesystem::path(opts.config_path).stem().string(),
        sim.get_sim_env(), sim.get_logging_manager(), *ctrl);
}

// The supervisory poll loop. Returns once the target is reached, the controller
// paused itself (a fault, or a GUI-less pause), or a SIGTERM was latched. On the
// way it drives the probe observers, prints progress every 10000 steps and reaps
// finished checkpoint writes (so a failure is reported during the run).
void supervise(sim_instance& sim, sim::sim_thread_controller& ctrl,
               logging::logging_manager& lm, std::uint64_t start_step,
               std::uint64_t target_step, headless_checkpointer* cp,
               run_provenance& prov)
{
    const auto record = [&](const headless_checkpointer::written_t& w) {
        prov.record_savestate(w.path, w.step);
    };

    std::uint64_t last_reported_steps = start_step;
    while ((target_step == 0 || ctrl.get_step_count() < target_step) &&
           ctrl.get_mode() != sim::sim_mode::paused &&
           g_term_requested == 0) {

        auto steps = ctrl.get_step_count();
        std::chrono::milliseconds sleep_time{100};
        if (target_step > 0) {
            const auto remaining = (target_step > steps) ? (target_step - steps) : 0;
            if (remaining < 1000) {
                sleep_time = std::chrono::milliseconds{1};
            } else if (remaining < 10000) {
                sleep_time = std::chrono::milliseconds{10};
            }
        }
        std::this_thread::sleep_for(sleep_time);

        // Drive probe observers like the GUI does each frame.
        for (auto& entity : sim.get_sim_env().iterate_entities())
            entity.probes().update_observers();

        steps = ctrl.get_step_count();
        if (steps >= last_reported_steps + 10000) {
            const double  sps    = ctrl.get_actual_sps();
            std::uint64_t frames = 0;
            // Compose the whole status line first, then emit it with a single
            // insertion. The checkpoint hook writes stdout from the sim thread
            // concurrently; a multi-token `std::cout << a << b << …` sequence is
            // atomic only per token, so its @coast/progress line could land
            // mid-token and prefix ours (e.g. "  Steps: 5000@coast/progress …"),
            // which the line-oriented supervisor then fails to parse. One string =
            // one stdio write = no such tear.
            std::ostringstream line;
            line << "  Steps: " << steps << " | SPS: " << sps;
            if (lm.is_buffer_logging()) {
                frames = lm.combined_stats().frames_written;
                line << " | Logged: " << frames << " frames";
            }
            line << "\n";
            std::cout << line.str();
            diag::emit_progress({ diag::run_phase::step, steps, start_step,
                                  target_step, sps, frames });
            last_reported_steps = steps;
        }

        // Non-blocking: a failed write is reported now, not at the end of a
        // possibly days-long --unlimited run.
        if (cp) cp->collect(/*wait=*/false, record);
    }
}

} // namespace

int run_headless(sim_instance& sim, const cli::options& opts,
                 const paths::app_config& app_cfg) {
    // Flush stdout at every write. When stdout is a terminal it is line-buffered,
    // but when it is redirected to a FILE (as the console's step_async does) C
    // stdio switches to full buffering (~4-8 KB), so periodic progress lines sit
    // in the buffer until the process exits — a `tail -f` or a progress-bar
    // reader then sees nothing until the very end. unitbuf makes each `<<` reach
    // the file immediately (robust whether or not sync_with_stdio is on). Headless
    // output is sparse, so the extra flushes cost nothing. (The structured diag
    // progress lines flush themselves independently; see diag::emit_progress.)
    std::cout << std::unitbuf;

    std::cout << "COAST Headless Mode\n";
    std::cout << "Loaded configuration: " << opts.config_path << "\n";

    // --save-on-term's SIGTERM handler (installed below) is never restored. main
    // destroys sim_instance after this returns, which joins the observe writer and
    // closes its .observe.h5; coastd re-sends SIGTERM about once a second, and the
    // default disposition would kill that teardown and lose the file. A wedged
    // teardown is still bounded by coastd's SIGKILL at term_grace.

    // Run-manifest provenance (--emit-run-manifest); written when this returns.
    run_provenance prov(sim, opts);

    // The progress feed's bookkeeping, hoisted out of the try so the catch can
    // end the feed too (#233). Once phase=start is out, every exit writes exactly
    // one last line: done, or failed when the run exits 1 (a step line after
    // SIGTERM). close_feed() is the only writer of that line.
    struct feed_state {
        bool          started     = false;   // phase=start written
        bool          closed      = false;   // the run's last line written
        std::uint64_t start_step  = 0;
        std::uint64_t target_step = 0;
    } feed;
    const auto close_feed = [&feed](const diag::progress& p) {
        diag::emit_progress(p);
        feed.closed = true;
    };
    // An exception after phase=start ends the feed on failed. The sim thread is
    // stopped first: it may still be stepping (a throw from the run loop), and its
    // checkpoint hook would then write lines after this one. set_mode(paused) is
    // only a request; the empty quiesced task returns once the thread is parked at
    // a step boundary, after which a paused thread runs no step. Called from the
    // catch handlers, so it must not throw.
    const auto fail_feed = [&]() noexcept {
        if (!feed.started || feed.closed) return;
        try {
            auto* c = sim.get_thread_controller();   // non-null: start needed it
            c->set_mode(sim::sim_mode::paused);
            c->run_quiesced([] {});   // refuses only in the controller's destructor
            close_feed({ diag::run_phase::failed, c->get_step_count(), feed.start_step,
                         feed.target_step, c->get_actual_sps(),
                         sim.get_logging_manager().combined_stats().frames_written });
        } catch (...) {
            // Out of memory: no line, but the exit code still says failed.
        }
    };

    try {
        auto* controller = sim.get_thread_controller();
        if (!controller) {
            COAST_ERROR("headless", "no simulation thread controller available");
            return 1;
        }

        auto& logging_mgr = sim.get_logging_manager();

        // Start buffer logging if requested.
        if (!opts.buffer_log_path.empty()) {
            std::cout << "Starting buffer logging to: " << opts.buffer_log_path << "\n";
            logging::buffer_logger_config cfg;
            cfg.output_path = opts.buffer_log_path;

            auto result = logging_mgr.start_buffer_logging(cfg);
            if (!result) {
                COAST_ERROR("headless", "failed to start buffer logging: {}",
                            result.error_message);
                return 1;
            }
        }

        // Target step accounting (relative to current step, which may have
        // been restored from a checkpoint by build_env). The run's length comes
        // from opts.steps_mode, which cli::parse guarantees is never `unset` for
        // a headless run — so this is a genuine two-way branch, not a
        // zero-means-forever test.
        const bool          run_unlimited = (opts.steps_mode == cli::options::run_length::unlimited);
        const std::uint64_t start_step    = controller->get_step_count();
        const std::uint64_t target_step   = run_unlimited ? 0 : (start_step + opts.max_steps);

        std::cout << "Starting simulation";
        if (run_unlimited) {
            // "interrupt to stop" is graceful only for SIGTERM with
            // --save-on-term: the loop stops and the run takes the normal exit
            // path (exit_reason=user_signal). SIGINT is never caught, and SIGTERM
            // without --save-on-term hits the default disposition; such a run
            // loses its manifest and its --save-state, and leaves --log-buffers
            // unclosed (cli::parse refuses those outputs in that combination).
            std::cout << " with no step limit (--unlimited), from step "
                      << start_step << " — interrupt to stop";
        } else {
            std::cout << " for " << opts.max_steps
                      << " steps (from step " << start_step
                      << " to " << target_step << ")";
            controller->set_step_limit(target_step);
        }
        std::cout << "...\n";

        // Structured progress (contract in <diag/progress.h>) emitted alongside
        // the human text so external supervisors parse a stable schema, not the
        // free-form prints above.
        feed.start_step  = start_step;
        feed.target_step = target_step;
        diag::emit_progress({ diag::run_phase::start, start_step, start_step,
                              target_step, 0.0, 0 });
        feed.started = true;

        // --checkpoint N: snapshots at exact step boundaries, from a post_swap hook
        // on the sim thread (headless_checkpoint.h).
        std::shared_ptr<headless_checkpointer> cp =
            make_checkpointer(sim, opts, app_cfg, start_step, target_step);
        std::size_t cp_hook_id = 0;
        if (cp) {
            std::cout << "Checkpoints will be saved to: " << cp->snapshot_dir().string()
                      << " (every " << cp->interval() << " steps)\n";
            // Captures only the shared checkpointer (the hook outlives this frame
            // and is dropped with the env), so nothing dangles.
            cp_hook_id = sim.get_sim_env().set_post_swap_hook([cp] { cp->on_post_swap(); });
        }

        // --save-on-term: install a SIGTERM catcher so the loop can stop early and
        // write a final checkpoint. Arming is idempotent and deliberately does NOT
        // re-clear the latch on a second call (arm_term_signal): main may arm it
        // before build_env, and a SIGTERM arriving during a long build_env must
        // still be honoured here rather than being reset away.
        if (opts.save_on_term) {
            arm_term_signal();         // no-op if main already armed it pre-build_env
            // The final-on-SIGTERM save writes a CHECKPOINT (needs cp). With
            // checkpointing off (--checkpoint 0) there is nothing to write to, so a
            // graceful stop saves no resumable progress — warn rather than let the
            // operator believe a resume point will exist. Note the warning is about
            // the CHECKPOINT specifically: --save-state, a cleanly closed
            // --log-buffers and the manifest are all still delivered on this path.
            if (!cp)
                COAST_WARN("headless", "--save-on-term writes no CHECKPOINT without "
                                       "checkpointing (--checkpoint 0): a graceful stop still "
                                       "closes logs and writes --save-state/the manifest, but "
                                       "leaves no resume point");
        }

        controller->set_mode(sim::sim_mode::unlimited);

        supervise(sim, *controller, logging_mgr, start_step, target_step, cp.get(), prov);

        controller->set_mode(sim::sim_mode::paused);
        controller->set_step_limit(0);

        // A SIGTERM that arrives once the target is reached interrupted nothing
        // (coastd's shutdown signals every live child, finished or not); calling
        // that run signal-terminated would make coastd resume a complete job. The
        // check needs a quiesced count: set_mode(paused) is not a barrier, so the
        // sim may still be finishing its last step.
        std::uint64_t final_count = controller->get_step_count();
        if (g_term_requested != 0) {
            try {
                controller->run_quiesced([&] { final_count = controller->get_step_count(); });
            } catch (...) {
                // Costs only label precision; keep the count already read.
            }
        }
        const bool reached_target = (target_step != 0 && final_count >= target_step);
        const bool terminated_by_signal = (g_term_requested != 0) && !reached_target;
        // No checkpoint after a fault: the sim halted mid-step, so its state is
        // incoherent but would write as a valid-looking checkpoint, which coastd
        // would resume from. The last periodic checkpoint stays the resume point.
        const bool faulted = controller->get_fault().has_value();
        if (terminated_by_signal && cp && faulted)
            COAST_WARN("checkpoint", "skipping the save-on-term checkpoint: the sim halted on a "
                       "fault, so its current state is not a coherent step boundary — resume "
                       "falls back to the last periodic checkpoint");
        // The final checkpoint runs THROUGH run_quiesced on the sim thread
        // (headless_checkpointer::write_final).
        if (terminated_by_signal && cp && !faulted)
            cp->write_final();

        // Drain pending checkpoint writes. The sim is paused (no further hook
        // fires), so unregister the hook and await each async write — the manifest
        // must record a complete on-disk file (its size), per design §5.2.
        if (cp) {
            sim.get_sim_env().remove_post_swap_hook(cp_hook_id);
            cp->collect(/*wait=*/true, [&](const headless_checkpointer::written_t& w) {
                prov.record_savestate(w.path, w.step);
            });
        }

        // A hot-path fault halts the sim thread cleanly (by value, not by an
        // exception). Report it to stderr so it is captured in logs, and exit
        // non-zero; skip the normal completion path. The manifest exit_reason
        // stays fatal_error.
        if (auto fault = controller->get_fault()) {
            if (fault->error.entity_id != core::step_error::no_entity) {
                COAST_ERROR("headless", "simulation fault at step {}: {} at {} (entity {})",
                            fault->step, core::to_string(fault->error.kind),
                            fault->error.site, fault->error.entity_id);
            } else {
                COAST_ERROR("headless", "simulation fault at step {}: {} at {}",
                            fault->step, core::to_string(fault->error.kind),
                            fault->error.site);
            }
            close_feed({ diag::run_phase::failed, fault->step, start_step,
                         target_step, controller->get_actual_sps(),
                         logging_mgr.combined_stats().frames_written });
            return 1;
        }

        const std::uint64_t final_step = controller->get_step_count();
        // The terminal progress event is emitted only once the run's outcome is
        // known — after buffer logging stopped and --save-state was written — so
        // a run that exits 1 never ends its feed with phase=done (PR #228). The
        // logged-frame count is read after the stop: combined_stats() keeps the
        // stopped session's final counts. A SIGTERM-interrupted run is NOT
        // "done": reporting done/"completed" would contradict the manifest's
        // user_signal exit_reason and mislead a feed reader (and any human) into
        // thinking the target was reached; it ends on a step line instead.
        const auto emit_terminal = [&](bool failed) {
            const auto phase = failed               ? diag::run_phase::failed
                             : terminated_by_signal ? diag::run_phase::step
                                                    : diag::run_phase::done;
            close_feed({ phase, final_step, start_step, target_step,
                         controller->get_actual_sps(),
                         logging_mgr.combined_stats().frames_written });
        };
        if (terminated_by_signal) {
            std::cout << "Simulation interrupted (SIGTERM). Steps so far: " << final_step << "\n";
        } else {
            std::cout << "Simulation completed. Total steps: " << final_step << "\n";
        }

        // A failed buffer log fails the run (#203), but only after --save-state:
        // the sim state is intact and still worth writing.
        bool buffer_log_failed = false;
        if (logging_mgr.is_buffer_logging()) {
            if (auto r = logging_mgr.stop_buffer_logging(*controller); r) {
                std::cout << "Buffer logging stopped.\n";
            } else {
                COAST_ERROR("headless", "buffer logging failed: {}", r.error_message);
                buffer_log_failed = true;
            }
        }

        if (!opts.state_save_path.empty()) {
            std::cout << "Saving state to: " << opts.state_save_path << "\n";

            // Extract on the sim thread at a quiescent boundary (the loop set
            // paused mode, but the quiesced task runs regardless of mode), then
            // await the executor write so the manifest records the final file.
            auto extracted = logging_mgr.extract_state(*controller);

            logging::state_save_config cfg;
            cfg.output_path = opts.state_save_path;

            auto result = logging_mgr.serialize_state(std::move(extracted.tree), cfg).get();
            if (result) {
                std::cout << "State saved successfully.\n";
                prov.record_savestate(opts.state_save_path, extracted.step);
            } else {
                COAST_ERROR("headless", "failed to save state: {}", result.error_message);
                // Still a failed run — but label WHY it ended honestly. Returning
                // here leaves the manifest's exit_reason at its fatal_error default, and
                // coastd reads fatal_error as "do not resume, this job is poisoned"
                // and counts it toward the crash-loop guard. A cancelled job whose
                // final --save-state hit a full disk is neither: the run was ended
                // by the operator, and the save failure is an I/O problem to fix,
                // not a reason to give up on the job. Set user_signal when that is
                // what actually happened; a save failure on a normal run stays
                // fatal_error.
                if (terminated_by_signal) prov.set_exit_reason(sim::exit_reason_t::user_signal);
                emit_terminal(true);
                return 1;
            }
        }

        auto stats = logging_mgr.combined_stats();
        if (stats.frames_written > 0) {
            std::cout << "Logging summary: " << stats.frames_written << " frames, "
                      << (stats.bytes_written / (1024.0 * 1024.0)) << " MB\n";
        }

        if (buffer_log_failed) {
            // Same exit_reason rule as the --save-state failure above.
            if (terminated_by_signal) prov.set_exit_reason(sim::exit_reason_t::user_signal);
            emit_terminal(true);
            return 1;
        }
        emit_terminal(false);

        // A SIGTERM-stopped run is a clean early exit, not a full completion: record
        // user_signal so a supervisor sees it stopped short (and resumes) rather than
        // treating it as done. Completed steps are captured below regardless.
        prov.set_exit_reason(terminated_by_signal ? sim::exit_reason_t::user_signal
                                                  : sim::exit_reason_t::completed_n_steps);
        return 0;
    } catch (const std::exception& e) {
        COAST_ERROR("headless", "{}", e.what());
        fail_feed();
        return 1;  // the manifest's exit_reason stays fatal_error
    } catch (...) {
        // Nothing in COAST throws a non-std exception, but this function is
        // documented not to throw: one escaping it would reach std::terminate in
        // main (exit 134, no manifest).
        COAST_ERROR("headless", "unknown exception");
        fail_feed();
        return 1;
    }
}

} // namespace coast::app
