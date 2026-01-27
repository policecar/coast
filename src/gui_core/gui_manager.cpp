#include "gui_manager.h"

#include <algorithm>
#include <cstdio>
#include <string>

#include "implot.h"

namespace coast {

gui_manager::gui_manager(sdl_app& app)
    : app(app)
    , file_browser(ImGuiFileBrowserFlags_CloseOnEsc | ImGuiFileBrowserFlags_ConfirmOnEnter)
{
    file_browser.SetTitle("Open Configuration");
    file_browser.SetTypeFilters({ ".json" });

    // Initialize time axis for activity plot
    for (std::size_t i = 0; i < ACTIVITY_HISTORY_SIZE; ++i) {
        time_axis.push_back(static_cast<float>(i));
    }
}

void gui_manager::set_sim_instance(sim_instance* instance)
{
    sim = instance;
    reset_per_group_state();

    // Recreate config editor if we have a simulation
    if (sim) {
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

void gui_manager::reset_per_group_state()
{
    // Stop simulation when changing instances
    simulation_running = false;

    // Clear activity history
    activity_history.clear();
    activity_sample_counter = 0;

    // Reset visibility flags
    show_ring_buffer.clear();
    show_neurons.clear();

    if (sim) {
        std::size_t num_groups = sim->get_groups().size();
        show_ring_buffer.resize(num_groups, false);
        show_neurons.resize(num_groups, false);
        activity_history.resize(num_groups);
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

        // Visualization menu only visible when simulation is loaded
        if (sim) {
            if (ImGui::BeginMenu("Visualization")) {
                if (ImGui::MenuItem("Activity Plot...", nullptr, show_activity_plot)) {
                    show_activity_plot = !show_activity_plot;
                }

                auto& groups = sim->get_groups();
                if (!groups.empty()) {
                    ImGui::Separator();
                    for (std::size_t i = 0; i < groups.size(); ++i) {
                        char label[64];
                        snprintf(label, sizeof(label), "Group %zu Ring Buffer", i + 1);
                        if (ImGui::MenuItem(label, nullptr, show_ring_buffer[i])) {
                            show_ring_buffer[i] = !show_ring_buffer[i];
                        }
                    }
                    ImGui::Separator();
                    for (std::size_t i = 0; i < groups.size(); ++i) {
                        char label[64];
                        snprintf(label, sizeof(label), "Group %zu Neurons", i + 1);
                        if (ImGui::MenuItem(label, nullptr, show_neurons[i])) {
                            show_neurons[i] = !show_neurons[i];
                        }
                    }
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
        render_activity_plot();
        if (show_config_editor && cfg_editor) {
            cfg_editor->render(&show_config_editor);
        }
        render_ring_buffer_windows();
        render_neuron_windows();
    }
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

void gui_manager::render_ring_buffer_windows()
{
    if (!sim) return;

    auto& ring_buffers = sim->get_ring_buffers();

    for (std::size_t i = 0; i < ring_buffers.size(); ++i) {
        if (show_ring_buffer[i]) {
            char label[64];
            snprintf(label, sizeof(label), "Ring Buffer %zu", i + 1);
            ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(
                ImVec2(380.0f, 40.0f + static_cast<float>(i) * 220.0f),
                ImGuiCond_FirstUseEver
            );
            show_ring_buffer[i] = ring_buffers[i].render_imgui(label, vis_config.ring_buffer_scale);
        }
    }
}

void gui_manager::render_neuron_windows()
{
    if (!sim) return;

    auto& visualizations = sim->get_visualizations();

    for (std::size_t i = 0; i < visualizations.size(); ++i) {
        // Update visibility state for conditional updates in main loop
        visualizations[i].set_visible(show_neurons[i]);

        if (show_neurons[i]) {
            char label[64];
            snprintf(label, sizeof(label), "Neurons Group %zu", i + 1);
            float scale = (i == 0) ? vis_config.neurons_scale_first : vis_config.neurons_scale_other;
            ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(
                ImVec2(800.0f, 40.0f + static_cast<float>(i) * 420.0f),
                ImGuiCond_FirstUseEver
            );
            show_neurons[i] = visualizations[i].render_imgui(label, scale);
        }
    }
}

void gui_manager::update_activity_history()
{
    if (!sim) return;

    auto& groups = sim->get_groups();
    auto& sim_env = sim->get_sim_env();

    // Sample activity every N steps to avoid overwhelming the plot
    ++activity_sample_counter;
    if (activity_sample_counter < 10) return;
    activity_sample_counter = 0;

    for (std::size_t g = 0; g < groups.size(); ++g) {
        // Get the IO buffer for this group and compute sum of activity
        auto buf_opt = sim_env.get_io_buffer(groups[g].get_outp_id());
        if (!buf_opt) continue;

        auto* buf = buf_opt.value();
        float sum = 0.0f;
        auto read_buf = buf->cur_read_buffer();
        for (std::size_t i = 0; i < read_buf.size(); ++i) {
            sum += read_buf[i];
        }

        // Add to rolling buffer
        activity_history[g].push_back(sum);
        if (activity_history[g].size() > ACTIVITY_HISTORY_SIZE) {
            activity_history[g].pop_front();
        }
    }
}

void gui_manager::render_activity_plot()
{
    if (!show_activity_plot || !sim) return;

    auto& groups = sim->get_groups();

    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(400, 40), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Activity Plot", &show_activity_plot)) {
        if (ImPlot::BeginPlot("Group Activity", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("Time", "Activity Sum");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, ACTIVITY_HISTORY_SIZE, ImGuiCond_Always);

            for (std::size_t g = 0; g < groups.size(); ++g) {
                if (activity_history[g].empty()) continue;

                // Convert deque to contiguous vector for plotting
                std::vector<float> data(activity_history[g].begin(), activity_history[g].end());

                char label[32];
                snprintf(label, sizeof(label), "Group %zu", g + 1);
                ImPlot::PlotLine(label, data.data(), static_cast<int>(data.size()));
            }
            ImPlot::EndPlot();
        }
    }
    ImGui::End();
}

} // namespace coast
