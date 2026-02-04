#ifndef COAST_LOGGING_WINDOW_H
#define COAST_LOGGING_WINDOW_H

#include <string>
#include <cstdint>
#include "imgui.h"
#include "imfilebrowser.h"

namespace coast {

class sim_instance;

/**
 * Window for controlling data logging functionality.
 * Provides controls for:
 * - IO Buffer logging (HDF5)
 * - Probe file logging (HDF5/CSV)
 * - State save/load
 */
class logging_window {
public:
    logging_window();

    /**
     * Set the simulation instance to control logging for.
     */
    void set_sim_instance(sim_instance* instance);

    /**
     * Render the logging control window.
     */
    void render();

    /**
     * Access visibility state for menu toggle.
     */
    bool& visible() { return is_visible; }

    /**
     * Trigger state save dialog (can be called from menu).
     */
    void open_save_state_dialog();

    /**
     * Trigger state load dialog (can be called from menu).
     */
    void open_load_state_dialog();

    /**
     * Check if state was just loaded (for external handling).
     */
    bool has_loaded_state() const { return state_just_loaded; }

    /**
     * Get the step count from loaded state.
     */
    std::uint64_t get_loaded_step_count() const { return loaded_step_count; }

    /**
     * Clear the loaded state flag.
     */
    void clear_loaded_state() { state_just_loaded = false; }

private:
    sim_instance* sim = nullptr;
    bool is_visible = false;

    // Buffer logging state
    std::string buffer_log_path = "buffer_log.h5";
    bool buffer_logging_active = false;

    // Probe logging state
    std::string probe_log_path = "probe_log.h5";
    int probe_format_idx = 0;  // 0 = HDF5, 1 = CSV
    bool probe_logging_active = false;

    // State save/load
    std::string state_path = "state.h5";
    bool state_just_loaded = false;
    std::uint64_t loaded_step_count = 0;
    std::string last_error;
    std::string last_success;

    // File browsers
    ImGui::FileBrowser buffer_file_browser;
    ImGui::FileBrowser probe_file_browser;
    ImGui::FileBrowser state_save_browser;
    ImGui::FileBrowser state_load_browser;

    // Internal rendering helpers
    void render_buffer_logging_section();
    void render_probe_logging_section();
    void render_state_section();
    void render_status_section();
};

} // namespace coast

#endif // COAST_LOGGING_WINDOW_H
