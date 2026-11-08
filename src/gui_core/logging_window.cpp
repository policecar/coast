#include "gui_core/logging_window.h"
#include "core/local_time.h"
#include "core/app_config.h"
#include "core/path_resolver.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"
#include "logging/state_logger.h"
#include "diag/diag.h"
#include "imgui.h"
#include "gui_core/imgui_scaling.h"

#include <atomic>
#include <filesystem>
#include <cstdio>
#include <chrono>
#include <future>
#include <iomanip>
#include <mutex>
#include <optional>
#include <vector>

namespace coast {

// ---------------------------------------------------------------------------
// Deterministic auto-checkpoint — shared state for the sim-thread hook.
//
// This mirrors the headless --checkpoint path (src/app/run_headless.cpp): the
// cadence is enforced by a post_swap hook that runs ONCE PER STEP on the sim
// thread, at the coherent post-swap boundary, testing the EXACT completed step
// count. That makes the interval a guarantee — a run that reaches step k*N
// (relative to when auto-checkpoint was enabled) checkpoints at exactly k*N,
// independent of GUI frame rate. The previous GUI implementation polled the step
// count each frame (`step >= last + interval`), which overshot by up to a frame
// and silently coarsened any interval finer than the frame period.
//
// Lifetime: held via shared_ptr and captured by the hook, so the ctx's lifetime
// does not depend on logging_window's (hook removal is synchronous since #235,
// so the hook does not run after teardown_checkpoint). The hook captures ONLY
// this ctx, never `this` — same discipline as gui_manager's status publisher.
//
// Threading: `enabled` / `interval` / `start_step` are written by the GUI thread
// (checkbox + input) and read by the sim-thread hook — atomics. `pending` (the
// async-write futures) is written by the hook and drained by the GUI thread —
// guarded by `mtx`. The immutable pointers/strings are resolved once when the
// hook is installed and never change for the sim's lifetime.
struct logging_checkpoint_ctx {
    // Immutable after setup_checkpoint().
    sim::env*                    env  = nullptr;
    logging::logging_manager*    lm   = nullptr;
    sim::sim_thread_controller*  ctrl = nullptr;
    std::string                  snapshot_dir;   // verbatim: relative → CWD, absolute → as-is
    std::string                  config_name;    // config stem for the filename

    // GUI -> sim-thread hook (live controls).
    std::atomic<bool>          enabled{false};
    std::atomic<std::uint64_t> interval{10000};
    std::atomic<std::uint64_t> start_step{0};    // modulo phase base

    // sim-thread hook -> GUI (results awaiting reaping).
    struct pending_t {
        std::future<logging::log_result> fut;
        std::filesystem::path            path;
        std::uint64_t                    step;
    };
    std::mutex                  mtx;
    std::vector<pending_t>      pending;          // in-flight async writes
    std::vector<std::string>    errors;           // extraction-time failures
};

logging_window::logging_window()
    : buffer_file_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , state_save_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , state_load_browser(ImGuiFileBrowserFlags_CloseOnEsc)
{
    buffer_file_browser.SetTitle("Select Buffer Log File");
    buffer_file_browser.SetTypeFilters({ ".h5", ".hdf5" });

    state_save_browser.SetTitle("Save State");
    state_save_browser.SetTypeFilters({ ".h5", ".hdf5" });

    state_load_browser.SetTitle("Load State");
    state_load_browser.SetTypeFilters({ ".h5", ".hdf5" });
}

void logging_window::set_sim_instance(sim_instance* instance)
{
    // Stop any active logging before changing sim
    if (sim && buffer_logging_active) {
        stop_buffer_logging();
        buffer_logging_active = false;
    }

    // Tear down auto-checkpoint on the OUTGOING sim while its env + executor are
    // still alive (run_gui swaps the pointer here before destroying the old
    // instance): remove the cadence hook, then block-drain any in-flight writes.
    teardown_checkpoint();

    sim = instance;
    last_error.clear();
    last_success.clear();

    // Reset auto-checkpoint UI state, then install the hook for the new sim.
    auto_checkpoint_enabled = false;
    last_checkpoint_path.clear();
    if (sim) {
        seed_log_defaults();
        setup_checkpoint();
    }
}

logging::log_result logging_window::stop_buffer_logging()
{
    // The stop must quiesce the sim thread (issue #197), so it needs the
    // controller; a sim_instance without one (moved-from) cannot be logging.
    auto* ctrl = sim ? sim->get_thread_controller() : nullptr;
    if (!ctrl) return logging::log_result::error("no simulation thread controller");
    return sim->get_logging_manager().stop_buffer_logging(*ctrl);
}

void logging_window::seed_log_defaults()
{
    // Seed the default log/state file paths under the new sim's config-relative
    // dirs. A field the user has already edited is left untouched (the *_edited_
    // latch). app_cfg_ is injected by gui_manager before any sim attaches; guard
    // defensively.
    if (!sim || !app_cfg_) return;
    const std::string cfg_path = sim->get_config_path();

    // Switching to a DIFFERENT config invalidates the user's per-config path
    // choices, so clear the latches and reseed under the new config's dirs.
    // Reloading the SAME (possibly edited) config keeps whatever the user chose.
    if (cfg_path != last_seeded_config_) {
        buffer_log_edited_  = false;
        state_path_edited_  = false;
        last_seeded_config_ = cfg_path;
    }

    // Logs (buffer) go under the config's <base>/logs (paths.log_dir under
    // config_base). The state save/load default is a full simulation snapshot, so
    // it goes under the config's snapshot dir — the SAME resolver as --checkpoint /
    // auto-save — rather than logs/.
    const std::filesystem::path logdir = coast::paths::resolve_config_output(
        cfg_path, app_cfg_->paths.log_dir, app_cfg_->paths.config_base);
    const std::filesystem::path snapdir = coast::paths::resolve_snapshot_output_dir(
        cfg_path, sim->get_config().snapshot_dir, *app_cfg_);

    if (!buffer_log_edited_) buffer_log_path = (logdir  / "buffer_log.h5").string();
    if (!state_path_edited_) state_path      = (snapdir / "state.h5").string();
}

void logging_window::render()
{
    // Render file browsers (always, even if window not visible)
    buffer_file_browser.Display();
    if (buffer_file_browser.HasSelected()) {
        buffer_log_path = buffer_file_browser.GetSelected().string();
        buffer_log_edited_ = true;
        buffer_file_browser.ClearSelected();
    }

    state_save_browser.Display();
    if (state_save_browser.HasSelected()) {
        state_path = state_save_browser.GetSelected().string();
        state_path_edited_ = true;
        state_save_browser.ClearSelected();

        // Perform the save
        if (auto* controller = sim ? sim->get_thread_controller() : nullptr) {
            logging::state_save_config cfg;
            cfg.output_path = state_path;

            // Extract on the sim thread at a quiescent boundary, then await the
            // executor write to report the outcome on the status line.
            auto& lm = sim->get_logging_manager();
            // A halted env may be partly loaded or faulted (#229); auto-save on
            // pause refuses it too. try_extract_state checks the halt inside
            // the quiesced task, before reading the env, and returns a throw
            // from the extraction (a failed GPU readback) instead of letting it
            // reach this GUI thread, where nothing would catch it.
            if (auto extracted = lm.try_extract_state(*controller); !extracted) {
                last_error = "Save failed: " + extracted.error();
                last_success.clear();
            } else if (auto result = lm.serialize_state(std::move(extracted->tree), cfg).get()) {
                last_success = "State saved to: " + state_path;
                last_error.clear();
            } else {
                last_error = "Save failed: " + result.error_message;
                last_success.clear();
            }
        }
    }

    state_load_browser.Display();
    if (state_load_browser.HasSelected()) {
        state_path = state_load_browser.GetSelected().string();
        state_path_edited_ = true;
        state_load_browser.ClearSelected();

        // Perform the load
        if (auto* controller = sim ? sim->get_thread_controller() : nullptr) {
            // Parse on the executor, apply on the sim thread, set the step
            // counter — load_state does all three (and merges warnings).
            auto result = sim->get_logging_manager().load_state(*controller, state_path);
            if (result) {
                loaded_step_count = result.restored_step;
                state_just_loaded = true;
                last_success = "State loaded from: " + state_path + " (step " + std::to_string(result.restored_step) + ")";
                last_error.clear();

                // Show warnings if any
                for (const auto& w : result.warnings) {
                    last_success += "\nWarning: " + w;
                }
            } else {
                last_error = "Load failed: " + result.error_message;
                if (const auto fault = controller->get_fault();
                    fault && fault->error.kind == core::error_kind::partial_load)
                    last_error += "\nThe state was only partly loaded, so the simulation is halted. "
                                  "Reload the configuration to run again.";
                last_success.clear();
            }
        }
    }

    if (!is_visible) return;

    ImGui::SetNextWindowSize(ImVec2(450, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Data Logging", &is_visible)) {
        if (!sim) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "No simulation loaded");
            ImGui::End();
            return;
        }

        render_buffer_logging_section();
        ImGui::Separator();
        render_state_section();
        ImGui::Separator();
        render_status_section();
    }
    ImGui::End();
}

void logging_window::render_buffer_logging_section()
{
    ImGui::Text("IO Buffer Logging");
    ImGui::Indent();

    // File path input
    char path_buf[512];
    snprintf(path_buf, sizeof(path_buf), "%s", buffer_log_path.c_str());

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - scaled(80));
    if (ImGui::InputText("##buffer_path", path_buf, sizeof(path_buf))) {
        buffer_log_path = path_buf;
        buffer_log_edited_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##buffer", ImVec2(scaled(70), 0))) {
        buffer_file_browser.Open();
    }

    // Helper lambda to start buffer logging
    auto start_buffer_logging = [this]() {
        // Remove existing file if it exists
        if (std::filesystem::exists(buffer_log_path)) {
            std::filesystem::remove(buffer_log_path);
        }

        logging::buffer_logger_config cfg;
        cfg.output_path = buffer_log_path;

        auto result = sim->get_logging_manager().start_buffer_logging(cfg);
        if (result) {
            buffer_logging_active = true;
            last_success = "Buffer logging started: " + buffer_log_path;
            last_error.clear();
        } else {
            last_error = result.error_message;
            last_success.clear();
        }
    };

    // Start/Stop button
    ImGui::BeginDisabled(buffer_log_path.empty());
    if (!buffer_logging_active) {
        if (ImGui::Button("Start Buffer Logging", ImVec2(scaled(150), 0))) {
            // Check if file exists
            if (std::filesystem::exists(buffer_log_path)) {
                buffer_confirm_overwrite = true;
                ImGui::OpenPopup("Overwrite Buffer Log?");
            } else {
                start_buffer_logging();
            }
        }
    } else {
        if (ImGui::Button("Stop Buffer Logging", ImVec2(scaled(150), 0))) {
            auto result = stop_buffer_logging();
            buffer_logging_active = false;
            if (result) {
                last_success = "Buffer logging stopped";
                last_error.clear();
            } else {
                last_error = result.error_message;
            }
        }
    }
    ImGui::EndDisabled();

    // Overwrite confirmation popup
    if (ImGui::BeginPopupModal("Overwrite Buffer Log?", &buffer_confirm_overwrite,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("File already exists:\n%s\n\nOverwrite?", buffer_log_path.c_str());
        ImGui::Separator();

        if (ImGui::Button("Yes", ImVec2(120, 0))) {
            start_buffer_logging();
            buffer_confirm_overwrite = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("No", ImVec2(120, 0))) {
            buffer_confirm_overwrite = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Status
    ImGui::SameLine();
    if (buffer_logging_active) {
        auto* logger = sim->get_logging_manager().get_buffer_logger();
        if (logger) {
            const auto stats = logger->stats();
            if (logger->write_ok()) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Recording: %llu frames, %.1f MB",
                                  static_cast<unsigned long long>(stats.frames_written),
                                  static_cast<double>(stats.bytes_written) / (1024.0 * 1024.0));
            } else {
                // Sticky (#203): nothing more reaches the file; Stop reports why.
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                                  "Write FAILED after %llu frames - log may be unreadable, press Stop",
                                  static_cast<unsigned long long>(stats.frames_written));
            }
        }
    } else {
        ImGui::TextDisabled("Stopped");
    }

    ImGui::Unindent();
}

void logging_window::render_state_section()
{
    ImGui::Text("State Persistence");
    ImGui::Indent();

    // File path input
    char path_buf[512];
    snprintf(path_buf, sizeof(path_buf), "%s", state_path.c_str());

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - scaled(170));
    if (ImGui::InputText("##state_path", path_buf, sizeof(path_buf))) {
        state_path = path_buf;
        state_path_edited_ = true;
    }

    ImGui::SameLine();
    if (ImGui::Button("Save State", ImVec2(scaled(80), 0))) {
        state_save_browser.SetInputName(std::filesystem::path(state_path).filename().string());
        state_save_browser.Open();
    }

    ImGui::SameLine();
    if (ImGui::Button("Load State", ImVec2(scaled(80), 0))) {
        state_load_browser.Open();
    }

    // Current step info
    if (auto* controller = sim->get_thread_controller()) {
        ImGui::Text("Current step: %llu",
                    static_cast<unsigned long long>(controller->get_step_count()));
    }

    // Auto-checkpoint settings. The cadence runs on the sim thread (the post_swap
    // hook installed in setup_checkpoint); the checkbox and input below only drive
    // that hook's atomics — they never extract or write state on the GUI thread.
    ImGui::Spacing();
    if (ImGui::Checkbox("Auto-checkpoint", &auto_checkpoint_enabled)) {
        if (auto* ctx = checkpoint_ctx_.get()) {
            if (auto_checkpoint_enabled) {
                // Create the snapshot dir lazily — only when actually enabling.
                std::error_code ec;
                std::filesystem::create_directories(ctx->snapshot_dir, ec);
                if (ec) {
                    last_error = "Could not create snapshot directory: " + ctx->snapshot_dir;
                    auto_checkpoint_enabled = false;
                } else {
                    // Phase the cadence to NOW: the first checkpoint lands exactly
                    // `interval` steps from the current step.
                    ctx->interval.store(static_cast<std::uint64_t>(auto_checkpoint_interval),
                                        std::memory_order_relaxed);
                    ctx->start_step.store(ctx->ctrl->get_step_count(),
                                          std::memory_order_relaxed);
                    ctx->enabled.store(true, std::memory_order_release);
                    last_error.clear();
                }
            } else {
                ctx->enabled.store(false, std::memory_order_release);
            }
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Automatically save checkpoints every N steps to the\n"
                          "snapshot directory (exact, sim-thread cadence)");
    }

    if (auto_checkpoint_enabled) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(scaled(100));
        if (ImGui::InputInt("steps##checkpoint", &auto_checkpoint_interval, 1000, 10000)) {
            if (auto_checkpoint_interval < 100) auto_checkpoint_interval = 100;
            // Live re-config: push the new interval and re-phase so the next
            // checkpoint lands `interval` steps from the current step.
            if (auto* ctx = checkpoint_ctx_.get()) {
                ctx->interval.store(static_cast<std::uint64_t>(auto_checkpoint_interval),
                                    std::memory_order_relaxed);
                ctx->start_step.store(ctx->ctrl->get_step_count(),
                                      std::memory_order_relaxed);
            }
        }

        // Show last checkpoint info
        if (!last_checkpoint_path.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Last: %s",
                std::filesystem::path(last_checkpoint_path).filename().string().c_str());
        }
    }

    ImGui::Unindent();
}

void logging_window::render_status_section()
{
    // Show last error or success message
    if (!last_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", last_error.c_str());
    }
    if (!last_success.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", last_success.c_str());
    }

    // Combined stats
    auto stats = sim->get_logging_manager().combined_stats();
    if (stats.frames_written > 0) {
        ImGui::Text("Total logged: %llu frames, %.2f MB",
                   static_cast<unsigned long long>(stats.frames_written),
                   static_cast<double>(stats.bytes_written) / (1024.0 * 1024.0));
    }
}

void logging_window::open_save_state_dialog()
{
    if (sim) {
        state_save_browser.SetInputName(std::filesystem::path(state_path).filename().string());
        state_save_browser.Open();
    }
}

void logging_window::open_load_state_dialog()
{
    if (sim) {
        state_load_browser.Open();
    }
}

void logging_window::update()
{
    // GUI thread, once per frame: surface completed async checkpoint writes. The
    // cadence test, state extraction and write submission all happen on the sim
    // thread inside the post_swap hook (setup_checkpoint); this only reaps the
    // results for the status line.
    reap_checkpoints();
}

void logging_window::setup_checkpoint()
{
    // Called from set_sim_instance with a fresh, non-null sim. Install the
    // deterministic cadence hook on the sim thread. The hook stays installed for
    // the sim's whole lifetime and is gated by ctx->enabled (one relaxed atomic
    // load per step when disabled), rather than installed/removed on every
    // checkbox toggle — that keeps the lifetime trivial (one install in
    // setup_checkpoint, one teardown in teardown_checkpoint) at negligible cost.
    auto* controller = sim->get_thread_controller();
    if (!controller) return;

    // The config's snapshot_dir leaf resolved under config_base (SAME resolver as
    // the headless path and gui_manager::get_snapshot_dir), so every snapshot site
    // agrees on the location. No fs side effect here — the directory is created
    // lazily when auto-checkpoint is enabled. app_cfg_ is injected by gui_manager's
    // ctor before any sim attaches, so this is unreachable; if a future path breaks
    // that invariant, fail loudly and skip installing the hook rather than build a
    // ctx pointing at an unresolved (possibly empty) directory — mirrors
    // gui_manager::get_snapshot_dir's no-sim branch.
    if (!app_cfg_) {
        COAST_ERROR("gui", "setup_checkpoint() with no app_config set — checkpoint hook not installed");
        return;
    }
    std::string dir = paths::resolve_snapshot_output_dir(sim->get_config_path(),
                                                       sim->get_config().snapshot_dir,
                                                       *app_cfg_).string();

    auto ctx = std::make_shared<logging_checkpoint_ctx>();
    ctx->env          = &sim->get_sim_env();
    ctx->lm           = &sim->get_logging_manager();
    ctx->ctrl         = controller;
    ctx->snapshot_dir = dir;
    ctx->config_name  = std::filesystem::path(sim->get_config_path()).stem().string();
    ctx->interval.store(static_cast<std::uint64_t>(auto_checkpoint_interval),
                        std::memory_order_relaxed);
    checkpoint_ctx_ = ctx;

    checkpoint_hook_id_ = sim->get_sim_env().set_post_swap_hook([ctx]() {
        if (!ctx->enabled.load(std::memory_order_acquire)) return;

        const std::uint64_t interval = ctx->interval.load(std::memory_order_relaxed);
        if (interval == 0) return;
        const std::uint64_t start_step = ctx->start_step.load(std::memory_order_relaxed);

        // do_step() increments step_count AFTER swap_io(), so at the post-swap
        // boundary the number of COMPLETED steps is count + 1 (same as headless).
        const std::uint64_t completed = ctx->ctrl->get_step_count() + 1;
        if (completed <= start_step) return;
        if ((completed - start_step) % interval != 0) return;

        // This hook runs on the sim thread inside swap_io(). A throw that left it
        // (extract_state, allocation) would reach do_step()'s backstop catch,
        // which halts the run as a fault (#198). A failed checkpoint should not
        // end the run, so it is caught here: recorded for the GUI and skipped
        // while stepping continues. (Mirrors the headless checkpoint hook.)
        try {
            // Exact boundary. Extract inline (sim thread); the HDF5 write is async
            // so stepping resumes without waiting on disk. The future is reaped on
            // the GUI thread (reap_checkpoints).
            logging::mem_group tree =
                logging::state_logger::extract_state(*ctx->env, completed);

            // Runs on the sim thread; snapshot_path stamps with localtime_r.
            const std::filesystem::path full_path =
                core::snapshot_path(ctx->snapshot_dir, ctx->config_name, completed);

            logging::state_save_config scfg;
            scfg.output_path           = full_path;

            auto fut = ctx->lm->serialize_state(std::move(tree), scfg);
            std::lock_guard lock(ctx->mtx);
            ctx->pending.push_back({ std::move(fut), full_path, completed });
        } catch (const std::exception& e) {
            std::lock_guard lock(ctx->mtx);
            ctx->errors.push_back("step " + std::to_string(completed) + ": " + e.what());
        } catch (...) {
            std::lock_guard lock(ctx->mtx);
            ctx->errors.push_back("step " + std::to_string(completed) + ": unknown error");
        }
    });
    checkpoint_hook_registered_ = true;
}

void logging_window::teardown_checkpoint()
{
    // Called from set_sim_instance while the OUTGOING `sim` (and its env +
    // executor) are still alive. Remove the hook, then block-drain pending writes
    // so no future outlives the executor that will satisfy it.
    if (!checkpoint_ctx_) return;
    if (sim && checkpoint_hook_registered_) {
        sim->get_sim_env().remove_post_swap_hook(checkpoint_hook_id_);
    }
    checkpoint_hook_registered_ = false;
    drain_checkpoints();
    checkpoint_ctx_.reset();
}

void logging_window::reap_checkpoints()
{
    auto ctx = checkpoint_ctx_;   // keep alive for the duration of this call
    if (!ctx) return;

    std::lock_guard lock(ctx->mtx);

    // Surface extraction-time failures (write failures arrive via the futures).
    for (const auto& e : ctx->errors) {
        last_error = "Checkpoint failed (" + e + ")";
    }
    ctx->errors.clear();

    // Reap completed async writes without blocking the GUI thread: a future that
    // is `ready` returns from get() immediately, so the lock is never held across
    // a blocking wait.
    for (auto it = ctx->pending.begin(); it != ctx->pending.end(); ) {
        if (it->fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++it;
            continue;
        }
        auto result = it->fut.get();
        if (result) {
            last_checkpoint_path = it->path.string();
            last_success = "Checkpoint saved: " + it->path.filename().string();
            last_error.clear();
        } else {
            last_error = "Checkpoint failed: " + result.error_message;
        }
        it = ctx->pending.erase(it);
    }
}

void logging_window::drain_checkpoints()
{
    auto ctx = checkpoint_ctx_;
    if (!ctx) return;

    // Move the queue out under the lock, then block on each write OUTSIDE it. The
    // hook is already removed (synchronously, #235), so this only keeps the lock
    // from being held across a blocking wait.
    std::vector<logging_checkpoint_ctx::pending_t> drained;
    {
        std::lock_guard lock(ctx->mtx);
        drained = std::move(ctx->pending);
        ctx->pending.clear();
    }
    for (auto& p : drained) {
        auto result = p.fut.get();   // block until the write completes
        if (result) {
            last_checkpoint_path = p.path.string();
        }
    }
}

} // namespace coast
