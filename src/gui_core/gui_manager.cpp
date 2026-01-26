#include "gui_manager.h"

#include <cstdio>
#include <string>
#include <numeric>

#include "imgui.h"
#include "implot.h"
#include "imnodes.h"

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

    // Initialize activity history for each group
    activity_history.resize(groups.size());
    for (std::size_t i = 0; i < ACTIVITY_HISTORY_SIZE; ++i) {
        time_axis.push_back(static_cast<float>(i));
    }
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
            if (ImGui::MenuItem("Activity Plot...", nullptr, show_activity_plot)) {
                show_activity_plot = !show_activity_plot;
            }
            if (ImGui::MenuItem("Network Editor...", nullptr, show_network_editor)) {
                show_network_editor = !show_network_editor;
            }
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
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Configuration")) {
            if (ImGui::MenuItem("Editor...", nullptr, show_config_editor)) {
                show_config_editor = !show_config_editor;
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
    render_activity_plot();
    render_network_editor();
    render_config_editor();
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

void gui_manager::update_activity_history()
{
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
    if (!show_activity_plot) return;

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

void gui_manager::render_network_editor()
{
    if (!show_network_editor) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Network Editor", &show_network_editor)) {
        ImNodes::BeginNodeEditor();

        // Create nodes for each entity in the simulation
        int node_id = 0;
        int attr_id = 0;

        for (auto& entity : sim_env.iterate_entities()) {
            ImNodes::BeginNode(node_id);

            ImNodes::BeginNodeTitleBar();
            ImGui::TextUnformatted(entity.status_str().substr(0, 30).c_str());
            ImNodes::EndNodeTitleBar();

            // Input attribute
            ImNodes::BeginInputAttribute(attr_id++);
            ImGui::Text("input");
            ImNodes::EndInputAttribute();

            // Output attribute
            ImNodes::BeginOutputAttribute(attr_id++);
            ImGui::Text("output");
            ImNodes::EndOutputAttribute();

            ImNodes::EndNode();
            ++node_id;
        }

        ImNodes::EndNodeEditor();
    }
    ImGui::End();
}

void gui_manager::render_config_editor()
{
    if (!show_config_editor) return;

    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(50, 100), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Configuration Editor", &show_config_editor)) {
        ImGui::Text("Simulation Parameters");
        ImGui::Separator();

        // Display and edit group parameters
        for (std::size_t i = 0; i < groups.size(); ++i) {
            if (ImGui::CollapsingHeader(("Group " + std::to_string(i + 1)).c_str())) {
                ImGui::PushID(static_cast<int>(i));

                float inhibition = groups[i].get_local_inhibition_strength();
                if (ImGui::SliderFloat("Local Inhibition", &inhibition, 0.1f, 20.0f)) {
                    groups[i].get_local_inhibition_strength() = inhibition;
                }

                ImGui::Text("Neuron Count: %zu", groups[i].get_neuron_count());
                ImGui::Text("Output ID: %lu", groups[i].get_outp_id());

                ImGui::PopID();
            }
        }

        ImGui::Separator();
        ImGui::Text("Visualization Settings");

        ImGui::SliderFloat("Ring Buffer Scale", &vis_config.ring_buffer_scale, 0.5f, 4.0f);
        ImGui::SliderFloat("Neurons Scale (First)", &vis_config.neurons_scale_first, 0.25f, 2.0f);
        ImGui::SliderFloat("Neurons Scale (Other)", &vis_config.neurons_scale_other, 0.25f, 2.0f);
    }
    ImGui::End();
}

} // namespace coast
