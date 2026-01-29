#ifndef COAST_GUI_MANAGER_H
#define COAST_GUI_MANAGER_H

#include <string>
#include <memory>

#include "imgui.h"
#include "imfilebrowser.h"

#include "sdl_app.h"
#include "sim_instance.h"
#include "config_editor.h"
#include "entity_browser.h"
#include "buffer_browser.h"
#include "logging_window.h"

namespace coast {

/**
 * Manages the main GUI elements including menu bar and dialogs.
 * Coordinates visibility state and rendering of simulation controls
 * and introspection windows.
 *
 * Can operate with or without a loaded simulation. When no simulation
 * is loaded, only the System menu is visible.
 */
class gui_manager {
public:
    /**
     * Constructor - creates GUI manager without simulation.
     * Use set_sim_instance() to attach a simulation later.
     */
    explicit gui_manager(sdl_app& app);

    /**
     * Set the simulation instance to manage.
     * Pass nullptr to clear the current simulation.
     * Resets visibility state and recreates internal structures.
     */
    void set_sim_instance(sim_instance* instance);

    /**
     * Get the current simulation instance (may be nullptr).
     */
    [[nodiscard]] sim_instance* get_sim_instance() { return sim; }

    /**
     * Check if a simulation is currently loaded.
     */
    [[nodiscard]] bool has_simulation() const { return sim != nullptr; }

    /**
     * Render the main menu bar.
     */
    void render_menu_bar();

    /**
     * Render all visible dialogs.
     */
    void render_dialogs();

    /**
     * Render the status bar at the bottom of the main window.
     */
    void render_status_bar();

    /**
     * Check if simulation should be running.
     */
    [[nodiscard]] bool is_simulation_running() const { return simulation_running && has_simulation(); }

    /**
     * Get the number of simulation steps per frame.
     */
    [[nodiscard]] int get_steps_per_frame() const { return steps_per_frame; }

    /**
     * Open file browser to select a configuration file.
     */
    void open_file_browser();

    /**
     * Check if a file has been selected in the file browser.
     * @deprecated Use should_start_loading() instead for proper loading dialog support.
     */
    [[nodiscard]] bool has_selected_file() const;

    /**
     * Get the selected file path and clear the selection.
     * @deprecated Use get_pending_load_path() instead.
     */
    [[nodiscard]] std::string get_selected_file();

    /**
     * Check if it's time to start loading a configuration.
     * Returns true only after the loading dialog has been displayed for at least one frame,
     * ensuring the user sees the loading message before the blocking load operation.
     */
    [[nodiscard]] bool should_start_loading() const;

    /**
     * Get the path of the configuration file pending load.
     */
    [[nodiscard]] const std::string& get_pending_load_path() const { return pending_load_path; }

    /**
     * Clear the pending load state after loading completes (success or failure).
     */
    void clear_pending_load();

    /**
     * Set loading error message to display to user.
     */
    void set_load_error(const std::string& error);

private:
    // Reference to application
    sdl_app& app;

    // Simulation instance (may be nullptr)
    sim_instance* sim = nullptr;

    // File browser for opening configurations
    ImGui::FileBrowser file_browser;
    std::string selected_file_path;

    // Loading state for showing loading dialog before blocking load
    std::string pending_load_path;
    int loading_frames_shown = 0;
    std::string load_error_message;

    // Configuration editor (recreated when sim changes)
    std::unique_ptr<config_editor> cfg_editor;

    // Dialog visibility state
    bool show_control = false;
    bool show_status = true;
    bool show_config_editor = false;

    // Introspection browser windows (created when sim is set)
    std::unique_ptr<entity_browser> entity_browser_window;
    std::unique_ptr<buffer_browser> buffer_browser_window;
    logging_window logging_window_instance;

    // Simulation control state
    bool simulation_running = false;
    int steps_per_frame = 100;
    bool auto_steps = false;
    int auto_steps_cooldown = 0;
    static constexpr int AUTO_STEPS_COOLDOWN_FRAMES = 10;
    static constexpr float AUTO_STEPS_TARGET_FPS = 35.0f;
    static constexpr float AUTO_STEPS_P_COEFFICIENT = 0.4f;

    // Render individual dialogs
    void render_control_dialog();
    void render_status_dialog();
};

} // namespace coast

#endif // COAST_GUI_MANAGER_H
