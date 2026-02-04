#include "gui_manager.h"
#include "sim_thread_controller.h"
#include "imgui_scaling.h"
#include "probe/probe_instance.h"
#include "logging/logging.h"

#include <cfloat>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace coast {

gui_manager::gui_manager(sdl_app& app)
    : app(app)
    , file_browser(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_ConfirmOnEnter)
{
    file_browser.SetTitle("Open Configuration");
    file_browser.SetTypeFilters({ ".json" });
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

        // Set up callback for logging probes to file
        probe_browser_window.set_log_probe_callback([this](probe::probe_instance* probe) -> bool {
            if (!sim) return false;
            auto* logger = sim->get_logging_manager().get_probe_logger();
            if (!logger) {
                // Probe logging not started - show the logging window
                logging_window_instance.visible() = true;
                return false;
            }
            return logger->add_probe(probe);
        });

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
                ImGui::EndMenu();
            }
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
    }

    // Logging window doesn't require a simulation
    logging_window_instance.render();

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

        const float button_width = scaled(30);
        const float label_width = ImGui::CalcTextSize("unlimited").x + scaled(8);
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float available_width = ImGui::GetContentRegionAvail().x;

        // Play/Pause button (pinned left)
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

        // Single step button (pinned left, next to play)
        ImGui::SameLine();
        ImGui::BeginDisabled(is_running);
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

void gui_manager::render_status_dialog()
{
    if (!show_status || !sim) return;

    auto& sim_env = sim->get_sim_env();

    ImGui::SetNextWindowSize(ImVec2(300, 250), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20, 250), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Status", &show_status)) {
        for (auto& io_ent : sim_env.iterate_entities()) {
            ImGui::TextWrapped("%s", io_ent.status_str().c_str());
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
                if (mode == sim::sim_mode::paused) {
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
            if (sim) {
                auto result = sim->get_logging_manager().load_state(pending_snapshot_path);
                if (result) {
                    // Successfully loaded - restore step counter
                    if (auto* controller = sim->get_thread_controller()) {
                        controller->set_step_count(result.restored_step);
                    }
                    last_auto_save_path = pending_snapshot_path;
                    std::cout << "Resumed from snapshot at step " << result.restored_step << "\n";
                } else {
                    std::cerr << "Failed to load snapshot: " << result.error_message << "\n";
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

    // Detect transition from running to paused
    if (previous_sim_mode != sim::sim_mode::paused &&
        current_mode == sim::sim_mode::paused) {
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

    // Generate filename: config_name_YYYYMMDD_HHMMSS_stepcount.h5
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);

    std::string config_name = std::filesystem::path(sim->get_config_path()).stem().string();
    std::uint64_t step_count = controller->get_step_count();

    std::ostringstream filename;
    filename << config_name << "_"
             << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_"
             << step_count << ".h5";

    std::filesystem::path full_path = std::filesystem::path(snapshot_dir) / filename.str();

    // Perform the save
    logging::state_save_config cfg;
    cfg.output_path = full_path;
    cfg.compression.algorithm = logging::compression_config::type::gzip;
    cfg.compression.level = 4;

    auto result = sim->get_logging_manager().save_state(step_count, cfg);
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
    auto time_t = std::chrono::system_clock::to_time_t(sctp);
    std::tm tm = *std::localtime(&time_t);

    std::ostringstream time_str;
    time_str << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (!info.empty()) info += " | ";
    info += "Saved: " + time_str.str();

    return std::make_pair(latest_path.string(), info);
}

std::string gui_manager::get_snapshot_dir() const
{
    if (!sim) return "snapshots";

    // Get snapshot directory from config, or use default
    const auto& config = sim->get_config();
    std::string snapshot_dir = config.snapshot_dir;

    // If relative path, make it relative to the config file's directory
    std::filesystem::path dir_path(snapshot_dir);
    if (dir_path.is_relative()) {
        std::filesystem::path config_path(sim->get_config_path());
        dir_path = config_path.parent_path() / dir_path;
    }

    return dir_path.string();
}

} // namespace coast
