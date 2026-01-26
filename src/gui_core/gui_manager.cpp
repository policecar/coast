#include "gui_manager.h"

#include <cstdio>
#include <string>

#include "imgui.h"

namespace coast {

gui_manager::gui_manager(
    sdl_app& app,
    sim::env& sim_env,
    std::vector<ngm2::neuron_group_t>& groups,
    std::vector<vec_ring_buffer>& ring_buffers,
    std::vector<ngm_flat_vis>& visualizations
)
    : app(app)
    , sim_env(sim_env)
    , groups(groups)
    , ring_buffers(ring_buffers)
    , visualizations(visualizations)
{
    show_ring_buffer.resize(groups.size(), false);
    show_neurons.resize(groups.size(), false);
}

void gui_manager::render_menu_bar()
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("System")) {
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                app.quit();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Simulation")) {
            if (ImGui::MenuItem("Control...", nullptr, show_control)) {
                show_control = !show_control;
            }
            if (ImGui::MenuItem("Status...", nullptr, show_status)) {
                show_status = !show_status;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Visualization")) {
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
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

void gui_manager::render_dialogs()
{
    render_control_dialog();
    render_status_dialog();
    render_ring_buffer_windows();
    render_neuron_windows();
}

void gui_manager::render_control_dialog()
{
    if (!show_control) return;

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
    if (!show_status) return;

    ImGui::SetNextWindowSize(ImVec2(300, 250), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20, 250), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Status", &show_status)) {
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Separator();
        for (auto& io_ent : sim_env.iterate_entities()) {
            ImGui::TextWrapped("%s", io_ent.status_str().c_str());
        }
    }
    ImGui::End();
}

void gui_manager::render_ring_buffer_windows()
{
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
    for (std::size_t i = 0; i < visualizations.size(); ++i) {
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

} // namespace coast
