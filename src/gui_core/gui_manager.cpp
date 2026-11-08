#include "gui_core/gui_manager.h"
#include "core/local_time.h"
#include "core/app_config.h"
#include "sim_core/sim_thread_controller.h"
#include "gui_core/imgui_scaling.h"
#include "logging/logging.h"
#include "diag/diag.h"
#include "gui_plot/console_plot.h"
#include "sim_console/sim_console_api.h"

#include <cfloat>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <algorithm>

namespace coast {

void gui_manager::defer(std::function<void()> func)
{
    app.register_single_shot_func(std::move(func));
}

gui_manager::gui_manager(sdl_app& app, const paths::app_config& app_cfg, hdf5::executor& h5_exec)
    : app(app)
    , app_cfg_(app_cfg)
    , file_browser(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_ConfirmOnEnter)
    , buffer_replay_dialog_(h5_exec)
{
    file_browser.SetTitle("Open Configuration");
    file_browser.SetTypeFilters({ ".json" });

    // Give the logging window the same location policy (default-constructed
    // member; mirrors the set_sim_instance injection idiom) so its auto-checkpoint
    // path resolves snapshot_dir the same way the rest of the app does.
    logging_window_instance.set_app_config(app_cfg_);

    // The Metric Explorer is a sim-independent tool: it owns its datasets (imported
    // from config files) and persists across sim load/unload. Created once here and
    // freed at shutdown via free_resources() (see run_gui's shutdown hook).
    metric_explorer_window = std::make_unique<metric_explorer>(h5_exec);

    // The HDF5 Browser is likewise sim-independent (reads log files offline
    // through the shared executor) and persists across sim load/unload.
    h5_browser_window_ = std::make_unique<h5_browser_window>(h5_exec);

    // The campaign browser hands a selected run's observe file to the HDF5 browser,
    // so it needs the back-pointer. Both are owned here and outlive each other only
    // at shutdown, where the manager is destroyed as a whole.
    campaign_panel_.set_browser(h5_browser_window_.get());
}

void gui_manager::set_console(std::unique_ptr<console_iface> console)
{
    console_ = std::move(console);
    if (h5_browser_window_) h5_browser_window_->set_console(console_.get());
    campaign_panel_.set_console(console_.get());
}

void gui_manager::shutdown_console()
{
    // Drop the browser's back-pointer FIRST so it never observes a half-destroyed
    // console, then destroy the console — which joins its interpreter worker thread.
    if (h5_browser_window_) h5_browser_window_->set_console(nullptr);
    campaign_panel_.set_console(nullptr);
    console_.reset();
}

void gui_manager::free_resources()
{
    // Free GPU resources owned by persistent (sim-independent) GUI components.
    // Must run while the GL context is still current — call from the shutdown hook,
    // before the context is torn down.
    if (metric_explorer_window) {
        metric_explorer_window->free_resources();
    }
    // Release GL textures held by console image windows.
    coast::plot::free_resources();
}

void gui_manager::set_sim_instance(sim_instance* instance)
{
    // Free resources from old browsers before changing sim
    if (entity_browser_window) {
        entity_browser_window->free_resources();
        entity_browser_window.reset();
    }
    if (buffer_browser_window) {
        buffer_browser_window->free_resources();
        buffer_browser_window.reset();
    }
    if (dataset_browser_window) {
        dataset_browser_window->free_resources();
        dataset_browser_window.reset();
    }
    // NOTE: metric_explorer_window is intentionally NOT touched here — it is a
    // persistent, sim-independent tool (owns its own imported datasets) and must
    // survive sim load/unload. It is created once in the ctor and freed at
    // shutdown via free_resources().

    // Remove the status publish hook from the OUTGOING sim's env while it is still
    // alive (run_gui clears the sim pointer here before destroying the instance).
    remove_status_hook();

    sim = instance;

    // Update logging window with new sim instance
    logging_window_instance.set_sim_instance(sim);

    // Recreate browser windows and config editor if we have a simulation
    if (sim) {
        entity_browser_window = std::make_unique<entity_browser>(*sim);
        buffer_browser_window = std::make_unique<buffer_browser>(*sim);
        dataset_browser_window = std::make_unique<dataset_browser>(*sim);

        // Initialize probe system windows with sim environment
        probe_browser_window.set_env(&sim->get_sim_env());
        observer_panel_window.set_env(&sim->get_sim_env());

        cfg_editor = std::make_unique<config_editor>(sim->get_config(), sim->get_config_path());
        // Set reload callback to trigger loading of saved config
        cfg_editor->set_reload_callback([this](const std::string& path) {
            pending_load_path = path;
            loading_frames_shown = 0;
        });

        // Reset auto-save state
        previous_sim_mode = sim::sim_mode::paused;
        last_auto_save_path.clear();

        // Check for existing snapshots and offer to resume
        auto latest = find_latest_snapshot();
        if (latest) {
            pending_snapshot_path = latest->first;
            pending_snapshot_info = latest->second;
            show_snapshot_resume_dialog = true;
        }
    } else {
        cfg_editor.reset();
        show_config_editor = false;
        probe_browser_window.set_env(nullptr);
        observer_panel_window.set_env(nullptr);
    }
}

void gui_manager::open_file_browser()
{
    file_browser.Open();
}

bool gui_manager::has_selected_file() const
{
    return !selected_file_path.empty();
}

std::string gui_manager::get_selected_file()
{
    std::string result = std::move(selected_file_path);
    selected_file_path.clear();
    return result;
}

bool gui_manager::should_start_loading() const
{
    // Only return true if we have a pending load and the loading dialog
    // has been shown for at least 2 frames (to ensure it's visible)
    return !pending_load_path.empty() && loading_frames_shown >= 2;
}

void gui_manager::clear_pending_load()
{
    pending_load_path.clear();
    loading_frames_shown = 0;
}

void gui_manager::set_load_error(const std::string& error)
{
    load_error_message = error;
}

void gui_manager::render_menu_bar()
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("System")) {
            if (ImGui::MenuItem("Open Configuration...", "Ctrl+O")) {
                open_file_browser();
            }
            if (sim && ImGui::MenuItem("Configuration Editor...", nullptr, show_config_editor)) {
                show_config_editor = !show_config_editor;
            }
            ImGui::Separator();
            if (sim) {
                if (ImGui::MenuItem("Save State...", "Ctrl+S")) {
                    logging_window_instance.open_save_state_dialog();
                }
                if (ImGui::MenuItem("Load State...", "Ctrl+L")) {
                    logging_window_instance.open_load_state_dialog();
                }
                ImGui::Separator();
            }
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                app.quit();
            }
            ImGui::EndMenu();
        }

        // Simulation menu only visible when simulation is loaded
        if (sim) {
            if (ImGui::BeginMenu("Simulation")) {
                if (ImGui::MenuItem("Control...", nullptr, show_control)) {
                    show_control = !show_control;
                }
                if (ImGui::MenuItem("Status...", nullptr, show_status)) {
                    show_status = !show_status;
                }
                ImGui::EndMenu();
            }
        }

        // Introspection menu only visible when simulation is loaded
        if (sim) {
            if (ImGui::BeginMenu("Introspection")) {
                if (entity_browser_window) {
                    if (ImGui::MenuItem("Sim Entities...", nullptr, entity_browser_window->visible())) {
                        entity_browser_window->visible() = !entity_browser_window->visible();
                    }
                }
                if (buffer_browser_window) {
                    if (ImGui::MenuItem("IO Buffers...", nullptr, buffer_browser_window->visible())) {
                        buffer_browser_window->visible() = !buffer_browser_window->visible();
                    }
                }
                if (dataset_browser_window) {
                    if (ImGui::MenuItem("Dataset...", nullptr, dataset_browser_window->visible())) {
                        dataset_browser_window->visible() = !dataset_browser_window->visible();
                    }
                }
                if (ImGui::MenuItem("Logging...", nullptr, logging_window_instance.visible())) {
                    logging_window_instance.visible() = !logging_window_instance.visible();
                }
                if (ImGui::MenuItem("Performance...", nullptr, show_performance)) {
                    show_performance = !show_performance;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Probe Browser...", nullptr, probe_browser_window.is_visible())) {
                    probe_browser_window.toggle();
                }
                if (ImGui::MenuItem("Observer Capture...", nullptr, observe_panel_window.is_visible())) {
                    observe_panel_window.toggle();
                }
                ImGui::EndMenu();
            }
        }

        // Tools menu - always available (tools are independent of the loaded sim;
        // the Metric Explorer sources its datasets from imported config files).
        if (ImGui::BeginMenu("Tools")) {
            if (metric_explorer_window) {
                if (ImGui::MenuItem("Metric Explorer...", nullptr, metric_explorer_window->visible())) {
                    metric_explorer_window->visible() = !metric_explorer_window->visible();
                }
            }
            if (ImGui::MenuItem("coastd Daemon...", nullptr, show_coastd_)) {
                show_coastd_ = !show_coastd_;
            }
            ImGui::EndMenu();
        }

        // Analysis menu - always available (replay works without simulation)
        if (ImGui::BeginMenu("Analysis")) {
            if (ImGui::MenuItem("Buffer Replay...", nullptr, show_buffer_replay_)) {
                show_buffer_replay_ = !show_buffer_replay_;
            }
            if (ImGui::MenuItem("HDF5 Browser...", nullptr, show_h5_browser_)) {
                show_h5_browser_ = !show_h5_browser_;
            }
            if (ImGui::MenuItem("Campaign Browser...", nullptr, show_campaign_)) {
                show_campaign_ = !show_campaign_;
            }
            if (console_) {
                if (ImGui::MenuItem(console_->menu_label(), nullptr, console_->visible())) {
                    console_->set_visible(!console_->visible());
                }
            }
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

void gui_manager::render_dialogs()
{
    // File browser is always active
    file_browser.Display();
    if (file_browser.HasSelected()) {
        // Start the loading process - set pending path and reset frame counter
        pending_load_path = file_browser.GetSelected().string();
        loading_frames_shown = 0;
        load_error_message.clear();
        file_browser.ClearSelected();
    }

    // Show loading dialog when we have a pending load
    if (!pending_load_path.empty()) {
        // Center the loading dialog
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(400, 0), ImGuiCond_Always);

        ImGuiWindowFlags loading_flags = ImGuiWindowFlags_NoResize
                                       | ImGuiWindowFlags_NoMove
                                       | ImGuiWindowFlags_NoCollapse
                                       | ImGuiWindowFlags_AlwaysAutoResize;

        if (ImGui::Begin("Loading Configuration", nullptr, loading_flags)) {
            ImGui::Text("Loading configuration file...");
            ImGui::Spacing();
            ImGui::TextWrapped("%s", pending_load_path.c_str());
            ImGui::Spacing();

            // Show a simple progress indicator
            float progress = static_cast<float>(loading_frames_shown % 30) / 30.0f;
            ImGui::ProgressBar(progress, ImVec2(-1, 0), "");
        }
        ImGui::End();

        // Increment frame counter after rendering
        ++loading_frames_shown;
    }

    // Show error dialog if there was a loading error
    if (!load_error_message.empty()) {
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        bool show_error = true;
        if (ImGui::Begin("Load Error", &show_error, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Failed to load configuration:");
            ImGui::Spacing();
            ImGui::TextWrapped("%s", load_error_message.c_str());
            ImGui::Spacing();
            if (ImGui::Button("OK", ImVec2(scaled(120), 0))) {
                load_error_message.clear();
            }
        }
        ImGui::End();

        if (!show_error) {
            load_error_message.clear();
        }
    }

    // Only render simulation dialogs if we have a simulation
    if (sim) {
        render_control_dialog();
        render_status_dialog();
        if (show_config_editor && cfg_editor) {
            cfg_editor->render(&show_config_editor);
        }

        // Render introspection browser windows
        if (entity_browser_window) {
            entity_browser_window->render();
        }
        if (buffer_browser_window) {
            buffer_browser_window->render();
        }
        if (dataset_browser_window) {
            dataset_browser_window->render();
        }
        render_performance_window();

        // Probe system - browser handles both update and observer windows
        probe_browser_window.update();  // Drain probe values to observers
        probe_browser_window.render();  // Render browser and observer windows

        // Observe (offline capture) system panel
        observe_panel_window.render();
    }

    // Metric Explorer is sim-independent (owns its imported datasets); render it
    // regardless of whether a simulation is loaded.
    if (metric_explorer_window) {
        metric_explorer_window->render();
    }

    // Logging window doesn't require a simulation
    logging_window_instance.update();  // Check for auto-checkpoints
    logging_window_instance.render();

    // Buffer replay dialog (available even without sim)
    if (show_buffer_replay_) {
        buffer_replay_dialog_.render(&show_buffer_replay_);
    }

    // Another panel (the campaign browser) can hand the HDF5 browser a file to open.
    // The show flag lives here, so the request is consumed here — otherwise the file
    // loads into a window the user cannot see.
    if (h5_browser_window_ && h5_browser_window_->take_visible_request()) {
        show_h5_browser_ = true;
    }

    // HDF5 browser (available even without sim)
    if (show_h5_browser_ && h5_browser_window_) {
        h5_browser_window_->render(&show_h5_browser_);
    }

    if (show_campaign_) {
        campaign_panel_.render(&show_campaign_);
    }

    if (show_coastd_) {
        coastd_panel_.render(&show_coastd_);
    } else {
        // A closed panel must stop TALKING to the daemon. Without this the poller
        // thread survives the window and keeps issuing a `list` RPC every 2 s, on a
        // connection nobody can see, for the rest of the process lifetime.
        coastd_panel_.on_hidden();
    }

    // Injected C++ console (owns its own visibility)
    if (console_) console_->render();

    // Console-created plots (drawn on the GUI thread regardless of console
    // visibility; a no-op when no macro has made one).
    coast::plot::render_all();

    // Async sim runs launched from the console (step_async): poll their child
    // processes and, when the status dialog is open, draw the runs table. No-op
    // when nothing has been launched.
    coast::simc::render_runs_window();

    // Render snapshot resume dialog
    render_snapshot_resume_dialog();

    // Check for auto-save (mode transition detection)
    check_auto_save();
}

void gui_manager::render_control_dialog()
{
    if (!show_control || !sim) return;

    auto* controller = sim->get_thread_controller();
    if (!controller) return;

    ImGui::SetNextWindowSize(ImVec2(350, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(250, 0), ImVec2(FLT_MAX, FLT_MAX));
    ImGui::SetNextWindowPos(ImVec2(20, 40), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Simulation Control", &show_control)) {
        sim::sim_mode current_mode = controller->get_mode();
        bool is_running = (current_mode != sim::sim_mode::paused);
        // A halted run (a per-step fault or a failed state load) cannot step
        // again; the controller refuses, so the controls say so.
        const bool halted = controller->is_halted();

        const float button_width = scaled(30);
        const float label_width = ImGui::CalcTextSize("unlimited").x + scaled(8);
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float available_width = ImGui::GetContentRegionAvail().x;

        // Play/Pause button (pinned left)
        ImGui::BeginDisabled(halted);
        if (is_running) {
            if (ImGui::Button("||##playpause", ImVec2(button_width, 0))) {
                controller->set_mode(sim::sim_mode::paused);
            }
        } else {
            if (ImGui::Button(">##playpause", ImVec2(button_width, 0))) {
                if (sim_speed_value > 250) {
                    controller->set_mode(sim::sim_mode::unlimited);
                } else {
                    controller->set_steps_per_second(static_cast<double>(sim_speed_value));
                    controller->set_mode(sim::sim_mode::throttled);
                }
            }
        }

        ImGui::EndDisabled();

        // Single step button (pinned left, next to play)
        ImGui::SameLine();
        ImGui::BeginDisabled(is_running || halted);
        if (ImGui::Button(">|##step", ImVec2(button_width, 0))) {
            controller->single_step();
        }
        ImGui::EndDisabled();

        // Speed slider (stretches to fill space)
        ImGui::SameLine();
        float slider_width = available_width - (button_width * 2) - label_width - (spacing * 3);
        if (slider_width < 50.0f) slider_width = 50.0f;
        ImGui::SetNextItemWidth(slider_width);
        if (ImGui::SliderInt("##speed", &sim_speed_value, 1, 251, "")) {
            if (is_running) {
                if (sim_speed_value > 250) {
                    controller->set_mode(sim::sim_mode::unlimited);
                } else {
                    controller->set_steps_per_second(static_cast<double>(sim_speed_value));
                    controller->set_mode(sim::sim_mode::throttled);
                }
            }
        }

        // Speed label (pinned right)
        ImGui::SameLine();
        const char* label_text = (sim_speed_value > 250) ? "unlimited" : nullptr;
        char label_buf[16];
        if (!label_text) {
            snprintf(label_buf, sizeof(label_buf), "%d/s", sim_speed_value);
            label_text = label_buf;
        }
        float text_width = ImGui::CalcTextSize(label_text).x;
        float label_offset = label_width - text_width;
        if (label_offset > 0) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + label_offset);
        }
        ImGui::TextUnformatted(label_text);

        if (halted) {
            if (const auto fault = controller->get_fault()) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Halted at step %llu: %s (%s)",
                                   static_cast<unsigned long long>(fault->step),
                                   core::to_string(fault->error.kind), fault->error.site);
            }
            ImGui::TextWrapped("Reload the configuration to run again.");
        }

        // Auto-save checkbox
        ImGui::Checkbox("Auto-save on pause", &auto_save_enabled);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Automatically save simulation state when paused.\nSnapshots stored in: %s",
                             get_snapshot_dir().c_str());
        }

        // Show last auto-save info
        if (!last_auto_save_path.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Last: %s", std::filesystem::path(last_auto_save_path).filename().string().c_str());
        }
    }
    ImGui::End();
}

void gui_manager::remove_status_hook()
{
    if (status_hook_registered_ && sim) {
        sim->get_sim_env().remove_post_swap_hook(status_hook_id_);
    }
    status_hook_registered_ = false;
}

void gui_manager::sync_status_hook()
{
    const bool want = show_status && sim != nullptr;

    if (want && !status_hook_registered_) {
        // Publish each entity's status_str() from the sim thread, where the live
        // state it reads (dataset cursors, neuron-group mismatch aggregates) is
        // coherent. The hook captures the shared slot + the env pointer, never
        // `this` — its lifetime is decoupled from gui_manager.
        auto* env = &sim->get_sim_env();
        auto  pub = status_pub_;
        status_hook_id_ = env->set_post_swap_hook([pub, env]() {
            pub->produce_if_wanted([env]() {
                std::vector<std::string> lines;
                for (auto& io_ent : env->iterate_entities()) {
                    lines.push_back(io_ent.status_str());
                }
                return lines;
            });
        });
        status_hook_registered_ = true;
    } else if (!want && status_hook_registered_) {
        remove_status_hook();
    }
}

void gui_manager::render_status_dialog()
{
    sync_status_hook();
    if (!show_status || !sim) return;

    ImGui::SetNextWindowSize(ImVec2(300, 250), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20, 250), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Status", &show_status)) {
        // Display the strings the sim thread published; never touch live entity
        // state from the GUI thread.
        for (const auto& line : status_pub_->consume()) {
            ImGui::TextWrapped("%s", line.c_str());
        }
    }
    ImGui::End();
}

void gui_manager::render_status_bar()
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float status_bar_height = ImGui::GetFrameHeight();

    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - status_bar_height));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, status_bar_height));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration
                           | ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoResize
                           | ImGuiWindowFlags_NoSavedSettings
                           | ImGuiWindowFlags_NoBringToFrontOnFocus
                           | ImGuiWindowFlags_NoFocusOnAppearing;

    if (ImGui::Begin("##StatusBar", nullptr, flags)) {
        ImGui::Text("FPS: %6.1f", ImGui::GetIO().Framerate);

        if (sim) {
            // Simulation status
            if (auto* controller = sim->get_thread_controller()) {
                ImGui::SameLine();
                sim::sim_mode mode = controller->get_mode();
                if (controller->is_halted()) {
                    ImGui::Text(" | Sim: Halted         ");
                } else if (mode == sim::sim_mode::paused) {
                    ImGui::Text(" | Sim: Paused         ");
                } else {
                    double actual_sps = controller->get_actual_sps();
                    if (mode == sim::sim_mode::unlimited) {
                        ImGui::Text(" | Sim: %7.0f/s (max)", actual_sps);
                    } else {
                        ImGui::Text(" | Sim: %7.0f/s     ", actual_sps);
                    }
                }
            }

            ImGui::SameLine();
            ImGui::Text(" | Config: %s", sim->get_config_path().c_str());
        }
    }
    ImGui::End();
}

void gui_manager::render_performance_window()
{
    if (!show_performance || !sim) return;

    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(400, 40), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Performance", &show_performance)) {
        // Placeholder
    }
    ImGui::End();
}

void gui_manager::render_snapshot_resume_dialog()
{
    if (!show_snapshot_resume_dialog || pending_snapshot_path.empty()) return;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(500, 0), ImGuiCond_Appearing);

    if (ImGui::Begin("Resume from Snapshot?", &show_snapshot_resume_dialog,
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
        ImGui::TextWrapped("A previous simulation snapshot was found:");
        ImGui::Spacing();

        ImGui::Indent();
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "%s",
                          std::filesystem::path(pending_snapshot_path).filename().string().c_str());
        if (!pending_snapshot_info.empty()) {
            ImGui::TextDisabled("%s", pending_snapshot_info.c_str());
        }
        ImGui::Unindent();

        ImGui::Spacing();
        ImGui::TextWrapped("Would you like to resume from this snapshot?");
        ImGui::Spacing();

        float button_width = scaled(120);
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float total_width = button_width * 2 + spacing;
        float start_x = (ImGui::GetContentRegionAvail().x - total_width) * 0.5f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + start_x);

        if (ImGui::Button("Resume", ImVec2(button_width, 0))) {
            if (auto* controller = sim ? sim->get_thread_controller() : nullptr) {
                sim->prepare_resume();   // as headless --resume (SPEC.md S.5.14)
                // load_state parses on the executor, applies on the sim thread,
                // and sets the step counter — all via the quiesced seam.
                auto result = sim->get_logging_manager().load_state(*controller, pending_snapshot_path);
                if (result) {
                    last_auto_save_path = pending_snapshot_path;
                    COAST_INFO("gui", "resumed from snapshot at step {}", result.restored_step);
                } else {
                    COAST_ERROR("gui", "failed to load snapshot: {}", result.error_message);
                }
            }
            show_snapshot_resume_dialog = false;
            pending_snapshot_path.clear();
            pending_snapshot_info.clear();
        }

        ImGui::SameLine();

        if (ImGui::Button("Start Fresh", ImVec2(button_width, 0))) {
            show_snapshot_resume_dialog = false;
            pending_snapshot_path.clear();
            pending_snapshot_info.clear();
        }
    }
    ImGui::End();
}

void gui_manager::check_auto_save()
{
    if (!sim || !auto_save_enabled) return;

    auto* controller = sim->get_thread_controller();
    if (!controller) return;

    sim::sim_mode current_mode = controller->get_mode();

    // Detect transition from running to paused. A halt forces paused too, but
    // a halted env may be partly loaded or faulted: saving it would put a
    // snapshot of that state where the resume dialog offers it next. Read the
    // mode first: the halt stores halted before paused, so a paused mode that
    // came from a halt is always seen together with is_halted().
    if (previous_sim_mode != sim::sim_mode::paused &&
        current_mode == sim::sim_mode::paused &&
        !controller->is_halted()) {
        perform_auto_save();
    }

    previous_sim_mode = current_mode;
}

void gui_manager::perform_auto_save()
{
    if (!sim) return;

    auto* controller = sim->get_thread_controller();
    if (!controller) return;

    std::string snapshot_dir = get_snapshot_dir();

    // Create directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(snapshot_dir, ec);
    if (ec) {
        return;  // Failed to create directory
    }

    // Extract on the sim thread at a quiescent boundary; the captured step names
    // the file coherently with its content. try_extract_state also refuses a
    // halt that lands after the check in check_auto_save, and returns a throw
    // from the extraction instead of letting it reach this GUI thread. A failed
    // auto-save is skipped silently, like a failed write below.
    auto& lm = sim->get_logging_manager();
    auto try_extracted = lm.try_extract_state(*controller);
    if (!try_extracted) return;
    auto& extracted = *try_extracted;

    const std::filesystem::path full_path = core::snapshot_path(
        snapshot_dir, std::filesystem::path(sim->get_config_path()).stem().string(), extracted.step);

    // Perform the save (await the executor write).
    logging::state_save_config cfg;
    cfg.output_path = full_path;

    auto result = lm.serialize_state(std::move(extracted.tree), cfg).get();
    if (result) {
        last_auto_save_path = full_path.string();
    }
}

std::optional<std::pair<std::string, std::string>> gui_manager::find_latest_snapshot()
{
    if (!sim) return std::nullopt;

    std::string snapshot_dir = get_snapshot_dir();

    // Check if directory exists
    if (!std::filesystem::exists(snapshot_dir) ||
        !std::filesystem::is_directory(snapshot_dir)) {
        return std::nullopt;
    }

    // Get the config name to filter relevant snapshots
    std::string config_name = std::filesystem::path(sim->get_config_path()).stem().string();

    // Find all matching HDF5 files
    std::filesystem::path latest_path;
    std::filesystem::file_time_type latest_time;
    bool found = false;

    for (const auto& entry : std::filesystem::directory_iterator(snapshot_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();

        // Check if it's an HDF5 file starting with our config name
        if (!filename.starts_with(config_name + "_")) continue;
        if (!filename.ends_with(".h5") && !filename.ends_with(".hdf5")) continue;

        auto write_time = entry.last_write_time();
        if (!found || write_time > latest_time) {
            latest_path = entry.path();
            latest_time = write_time;
            found = true;
        }
    }

    if (!found) return std::nullopt;

    // Extract info from filename (format: config_YYYYMMDD_HHMMSS_stepcount.h5)
    std::string info;
    std::string filename = latest_path.stem().string();

    // Try to parse step count from end
    auto last_underscore = filename.rfind('_');
    if (last_underscore != std::string::npos) {
        try {
            std::uint64_t steps = std::stoull(filename.substr(last_underscore + 1));
            info = "Step: " + std::to_string(steps);
        } catch (...) {
            // Ignore parsing errors
        }
    }

    // Add file modification time
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        latest_time - std::filesystem::file_time_type::clock::now() +
        std::chrono::system_clock::now());
    if (!info.empty()) info += " | ";
    info += "Saved: " + core::format_local_time(sctp, "%Y-%m-%d %H:%M:%S");

    return std::make_pair(latest_path.string(), info);
}

std::string gui_manager::get_snapshot_dir() const
{
    // Precondition: callers (perform_auto_save / find_latest_snapshot) guard on
    // sim first, so this is unreachable. If a future caller forgets the guard,
    // fail loudly (log) and return an empty path — which finds/creates nothing —
    // rather than a plausible-but-wrong bare leaf that would silently point
    // auto-save at the wrong directory.
    if (!sim) {
        COAST_ERROR("gui", "get_snapshot_dir() called with no simulation loaded");
        return {};
    }

    // The config's snapshot_dir leaf resolved under config_base (an absolute leaf
    // wins). SAME resolver as the headless resume/checkpoint path, so GUI
    // auto-save/resume and headless always agree on the location.
    return paths::resolve_snapshot_output_dir(sim->get_config_path(),
                                            sim->get_config().snapshot_dir,
                                            app_cfg_).string();
}

} // namespace coast
