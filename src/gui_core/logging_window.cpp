#include "logging_window.h"
#include "sim_instance.h"
#include "sim_thread_controller.h"
#include "imgui.h"
#include "imgui_scaling.h"

#include <filesystem>
#include <cstdio>

namespace coast {

logging_window::logging_window()
    : buffer_file_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , probe_file_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , state_save_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , state_load_browser(ImGuiFileBrowserFlags_CloseOnEsc)
{
    buffer_file_browser.SetTitle("Select Buffer Log File");
    buffer_file_browser.SetTypeFilters({ ".h5", ".hdf5" });

    probe_file_browser.SetTitle("Select Probe Log File");
    probe_file_browser.SetTypeFilters({ ".h5", ".hdf5", ".csv" });

    state_save_browser.SetTitle("Save State");
    state_save_browser.SetTypeFilters({ ".h5", ".hdf5" });

    state_load_browser.SetTitle("Load State");
    state_load_browser.SetTypeFilters({ ".h5", ".hdf5" });
}

void logging_window::set_sim_instance(sim_instance* instance)
{
    // Stop any active logging before changing sim
    if (sim && buffer_logging_active) {
        sim->get_logging_manager().stop_buffer_logging();
        buffer_logging_active = false;
    }
    if (sim && probe_logging_active) {
        sim->get_logging_manager().stop_probe_logging();
        probe_logging_active = false;
    }

    sim = instance;
    last_error.clear();
    last_success.clear();
}

void logging_window::render()
{
    // Render file browsers (always, even if window not visible)
    buffer_file_browser.Display();
    if (buffer_file_browser.HasSelected()) {
        buffer_log_path = buffer_file_browser.GetSelected().string();
        buffer_file_browser.ClearSelected();
    }

    probe_file_browser.Display();
    if (probe_file_browser.HasSelected()) {
        probe_log_path = probe_file_browser.GetSelected().string();
        // Update format based on extension
        if (probe_log_path.ends_with(".csv")) {
            probe_format_idx = 1;
        } else {
            probe_format_idx = 0;
        }
        probe_file_browser.ClearSelected();
    }

    state_save_browser.Display();
    if (state_save_browser.HasSelected()) {
        state_path = state_save_browser.GetSelected().string();
        state_save_browser.ClearSelected();

        // Perform the save
        if (sim) {
            auto* controller = sim->get_thread_controller();
            std::uint64_t step = controller ? controller->get_step_count() : 0;

            logging::state_save_config cfg;
            cfg.output_path = state_path;
            cfg.compression.algorithm = logging::compression_config::type::gzip;
            cfg.compression.level = 4;

            auto result = sim->get_logging_manager().save_state(step, cfg);
            if (result) {
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
        state_load_browser.ClearSelected();

        // Perform the load
        if (sim) {
            auto result = sim->get_logging_manager().load_state(state_path);
            if (result) {
                loaded_step_count = result.restored_step;
                state_just_loaded = true;
                last_success = "State loaded from: " + state_path + " (step " + std::to_string(result.restored_step) + ")";
                last_error.clear();

                // Restore the step counter
                if (auto* controller = sim->get_thread_controller()) {
                    controller->set_step_count(result.restored_step);
                }

                // Show warnings if any
                for (const auto& w : result.warnings) {
                    last_success += "\nWarning: " + w;
                }
            } else {
                last_error = "Load failed: " + result.error_message;
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
        render_probe_logging_section();
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
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##buffer", ImVec2(scaled(70), 0))) {
        buffer_file_browser.Open();
    }

    // Start/Stop button
    ImGui::BeginDisabled(buffer_log_path.empty());
    if (!buffer_logging_active) {
        if (ImGui::Button("Start Buffer Logging", ImVec2(scaled(150), 0))) {
            logging::buffer_logger_config cfg;
            cfg.output_path = buffer_log_path;
            cfg.chunk_size = 1000;
            cfg.compression.algorithm = logging::compression_config::type::gzip;
            cfg.compression.level = 4;
            cfg.include_stats = true;

            auto result = sim->get_logging_manager().start_buffer_logging(cfg);
            if (result) {
                buffer_logging_active = true;
                last_success = "Buffer logging started: " + buffer_log_path;
                last_error.clear();
            } else {
                last_error = result.error_message;
                last_success.clear();
            }
        }
    } else {
        if (ImGui::Button("Stop Buffer Logging", ImVec2(scaled(150), 0))) {
            auto result = sim->get_logging_manager().stop_buffer_logging();
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

    // Status
    ImGui::SameLine();
    if (buffer_logging_active) {
        auto* logger = sim->get_logging_manager().get_buffer_logger();
        if (logger) {
            const auto& stats = logger->stats();
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Recording: %lu frames, %.1f MB",
                              stats.frames_written,
                              static_cast<double>(stats.bytes_written) / (1024.0 * 1024.0));
        }
    } else {
        ImGui::TextDisabled("Stopped");
    }

    ImGui::Unindent();
}

void logging_window::render_probe_logging_section()
{
    ImGui::Text("Probe Logging");
    ImGui::Indent();

    // File path input
    char path_buf[512];
    snprintf(path_buf, sizeof(path_buf), "%s", probe_log_path.c_str());

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - scaled(80));
    if (ImGui::InputText("##probe_path", path_buf, sizeof(path_buf))) {
        probe_log_path = path_buf;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##probe", ImVec2(scaled(70), 0))) {
        probe_file_browser.Open();
    }

    // Format selector
    const char* formats[] = { "HDF5", "CSV" };
    ImGui::SetNextItemWidth(scaled(80));
    ImGui::Combo("Format", &probe_format_idx, formats, 2);

    // Start/Stop button
    ImGui::SameLine();
    ImGui::BeginDisabled(probe_log_path.empty());
    if (!probe_logging_active) {
        if (ImGui::Button("Start Probe Logging", ImVec2(scaled(150), 0))) {
            logging::probe_logger_config cfg;
            cfg.output_path = probe_log_path;
            cfg.format = (probe_format_idx == 0) ?
                         logging::probe_log_format::hdf5 :
                         logging::probe_log_format::csv;
            cfg.buffer_size = 10000;
            cfg.compression.algorithm = logging::compression_config::type::gzip;
            cfg.compression.level = 4;

            auto result = sim->get_logging_manager().start_probe_logging(cfg);
            if (result) {
                probe_logging_active = true;
                last_success = "Probe logging started: " + probe_log_path;
                last_error.clear();
            } else {
                last_error = result.error_message;
                last_success.clear();
            }
        }
    } else {
        if (ImGui::Button("Stop Probe Logging", ImVec2(scaled(150), 0))) {
            auto result = sim->get_logging_manager().stop_probe_logging();
            probe_logging_active = false;
            if (result) {
                last_success = "Probe logging stopped";
                last_error.clear();
            } else {
                last_error = result.error_message;
            }
        }
    }
    ImGui::EndDisabled();

    // Status and instructions
    if (probe_logging_active) {
        auto* logger = sim->get_logging_manager().get_probe_logger();
        if (logger) {
            const auto& stats = logger->stats();
            auto probes = logger->active_probes();
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Recording: %zu probes, %lu samples",
                              probes.size(), stats.samples_written);
        }
    } else {
        ImGui::TextDisabled("Use Probe Browser to add probes for logging");
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
        ImGui::Text("Current step: %lu", controller->get_step_count());
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
    if (stats.frames_written > 0 || stats.samples_written > 0) {
        ImGui::Text("Total logged: %lu frames, %lu samples, %.2f MB",
                   stats.frames_written, stats.samples_written,
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

} // namespace coast
