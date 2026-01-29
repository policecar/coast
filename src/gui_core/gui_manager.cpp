#include "gui_manager.h"

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
    // Stop simulation when changing instances
    simulation_running = false;

    // Free resources from old browsers before changing sim
    if (entity_browser_window) {
        entity_browser_window->free_resources();
        entity_browser_window.reset();
    }
    if (buffer_browser_window) {
        buffer_browser_window->free_resources();
        buffer_browser_window.reset();
    }

    sim = instance;

    // Recreate browser windows and config editor if we have a simulation
    if (sim) {
        entity_browser_window = std::make_unique<entity_browser>(*sim);
        buffer_browser_window = std::make_unique<buffer_browser>(*sim);

        cfg_editor = std::make_unique<config_editor>(sim->get_config(), sim->get_config_path());
        // Set reload callback to trigger loading of saved config
        cfg_editor->set_reload_callback([this](const std::string& path) {
            pending_load_path = path;
            loading_frames_shown = 0;
        });
    } else {
        cfg_editor.reset();
        show_config_editor = false;
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
                if (ImGui::MenuItem("Logging...", nullptr, logging_window_instance.visible())) {
                    logging_window_instance.visible() = !logging_window_instance.visible();
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
            if (ImGui::Button("OK", ImVec2(120, 0))) {
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
    }

    // Logging window doesn't require a simulation
    logging_window_instance.render();
}

void gui_manager::render_control_dialog()
{
    if (!show_control || !sim) return;

    auto& groups = sim->get_groups();

    ImGui::SetNextWindowSize(ImVec2(350, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20, 40), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Simulation Control", &show_control)) {
        // Start/Stop buttons
        if (simulation_running) {
            if (ImGui::Button("Stop", ImVec2(100, 30))) {
                simulation_running = false;
            }
        } else {
            if (ImGui::Button("Start", ImVec2(100, 30))) {
                simulation_running = true;
            }
        }
        ImGui::SameLine();
        ImGui::Text(simulation_running ? "Running" : "Paused");

        ImGui::Separator();
        ImGui::SliderInt("Steps/frame", &steps_per_frame, 0, 2000);
        ImGui::SameLine();
        ImGui::Checkbox("Auto", &auto_steps);

        // Auto-adjust steps/frame based on FPS using P controller
        if (auto_steps && simulation_running) {
            if (auto_steps_cooldown > 0) {
                --auto_steps_cooldown;
            } else {
                float fps = ImGui::GetIO().Framerate;
                float error = fps - AUTO_STEPS_TARGET_FPS;
                int adjustment = static_cast<int>(error * AUTO_STEPS_P_COEFFICIENT);

                if (adjustment != 0) {
                    steps_per_frame = std::clamp(steps_per_frame + adjustment, 1, 2000);
                    auto_steps_cooldown = AUTO_STEPS_COOLDOWN_FRAMES;
                }
            }
        }

        ImGui::Separator();
        ImGui::Text("Group Parameters:");
        for (std::size_t i = 0; i < groups.size(); ++i) {
            std::string label = "Group " + std::to_string(i + 1) + " inhibition";
            ImGui::SliderFloat(label.c_str(), &groups[i].get_local_inhibition_strength(), 0.1f, 20.0f);
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
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

        if (sim) {
            ImGui::SameLine();
            ImGui::Text(" | Config: %s", sim->get_config_path().c_str());
        }
    }
    ImGui::End();
}

} // namespace coast
