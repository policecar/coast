#ifndef COAST_GUI_MANAGER_H
#define COAST_GUI_MANAGER_H

#include <string>
#include <memory>
#include <optional>
#include <vector>

#include "core/demand_published.h"
#include "imgui.h"
#include "sim_core/sim_thread_controller.h"
#include "imfilebrowser.h"

#include "gui_core/sdl_app.h"
#include "sim_core/sim_instance.h"
#include "gui_core/config_editor.h"
#include "gui_core/entity_browser.h"
#include "gui_core/buffer_browser.h"
#include "gui_core/dataset_browser.h"
#include "gui_core/logging_window.h"
#include "gui_probe/probe_browser.h"
#include "gui_probe/observer_panel.h"
#include "gui_observe/observe_panel.h"
#include "gui_replay/buffer_replay_dialog.h"
#include "gui_tools/campaign_panel.h"
#include "gui_tools/h5_browser_window.h"
#include "gui_tools/coastd_panel.h"
#include "gui_tools/metric_explorer.h"
#include "gui_core/console_iface.h"

namespace hdf5 { class executor; }

namespace coast::paths { struct app_config; }

namespace coast {

/**
 * Manages the main GUI elements including menu bar and dialogs.
 * Coordinates visibility state and rendering of simulation controls
 * and introspection windows.
 *
 * Can operate with or without a loaded simulation. The System, Tools, and
 * Analysis menus are always available; the Simulation and Introspection menus
 * appear only when a simulation is loaded.
 */
class gui_manager {
public:
    /**
     * Constructor - creates GUI manager without simulation.
     * Use set_sim_instance() to attach a simulation later.
     *
     * @param app      The SDL/GL application that owns the main window.
     * @param app_cfg  Application location config. Must outlive the gui_manager
     *                 (owned by main, forwarded via run_gui). Governs WHERE the
     *                 GUI resolves per-config outputs (snapshot dir for auto-save
     *                 and resume) via coast::paths::resolve_snapshot_output_dir.
     * @param h5_exec  Process-wide HDF5 executor. Must outlive the gui_manager;
     *                 forwarded to the replay dialog so that any HDF5 file the
     *                 user opens through the GUI is read on the executor's
     *                 worker thread.
     */
    gui_manager(sdl_app& app, const paths::app_config& app_cfg, hdf5::executor& h5_exec);

    /**
     * Inject the optional C++ console (takes ownership). Adds an Analysis-menu
     * item, renders it, and wires it into the HDF5 browser so the browser can
     * open the browsed file in it and push variable bindings. Call before the
     * render loop starts.
     */
    void set_console(std::unique_ptr<console_iface> console);

    /**
     * Tear down the console, joining its interpreter worker thread. Call at
     * shutdown BEFORE the plot subsystem is deinit()'d: once the console thread is
     * joined, no macro can still be creating figures or freeing image textures, so
     * plot_manager::deinit() runs against a quiescent subsystem. Clears the HDF5
     * browser's back-pointer to the console first so it never dangles. Idempotent.
     */
    void shutdown_console();

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
     * Free GPU resources owned by persistent, sim-independent GUI components
     * (the Metric Explorer). Call from the application shutdown hook, while the
     * GL context is still current, before it is destroyed.
     */
    void free_resources();

    /**
     * Register a closure to run once on the GUI thread, in the next state phase
     * (outside the ImGui frame), then discarded. The registration point GUI
     * components use for deferred GUI-thread work — e.g. freeing a GL resource
     * whose owner was released on another thread. Thread-safe. Forwards to
     * sdl_app::register_single_shot_func.
     */
    void defer(std::function<void()> func);


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

    // Application location config (owned by main, outlives this). Source of the
    // config_base / leaf policy get_snapshot_dir resolves against.
    const paths::app_config& app_cfg_;

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
    bool show_performance = false;

    // Simulation speed slider value (1-250 = steps/sec, 251 = unlimited)
    int sim_speed_value = 100;

    // Auto-save state
    bool auto_save_enabled = false;
    sim::sim_mode previous_sim_mode = sim::sim_mode::paused;
    std::string last_auto_save_path;

    // Snapshot resume dialog state
    bool show_snapshot_resume_dialog = false;
    std::string pending_snapshot_path;
    std::string pending_snapshot_info;

    // Introspection browser windows (created when sim is set)
    std::unique_ptr<entity_browser> entity_browser_window;
    std::unique_ptr<buffer_browser> buffer_browser_window;
    std::unique_ptr<dataset_browser> dataset_browser_window;
    std::unique_ptr<metric_explorer> metric_explorer_window;
    logging_window logging_window_instance;

    // Probe system windows
    gui_probe::probe_browser probe_browser_window;
    gui_probe::observer_panel observer_panel_window;

    // Observe (offline capture) system window
    gui_observe::observe_panel observe_panel_window;

    // Analysis windows
    replay::buffer_replay_dialog buffer_replay_dialog_;
    bool show_buffer_replay_ = false;
    std::unique_ptr<h5_browser_window> h5_browser_window_;
    bool show_h5_browser_ = false;
    // Campaign browser: scans a directory of runs and filters them by the swept
    // parameters. Sim-independent, like the two above.
    campaign_panel campaign_panel_;
    bool           show_campaign_ = false;

    // coastd daemon-control panel (Tools menu). Independent of any loaded sim.
    coastd_panel coastd_panel_;
    bool         show_coastd_ = false;

    // Optional injected C++ console (concrete impl in coast_console). Contributed
    // by main.cpp via run_gui so coast_lib need not depend on LLVM. Owns its own
    // visibility; the HDF5 browser holds a non-owning pointer to drive it.
    std::unique_ptr<console_iface> console_;

    // Status dialog: per-entity status strings are computed by status_str() ON
    // THE SIM THREAD inside a gated post_swap hook (where cur_idx/cur_epoch and
    // the neuron-group mismatch SoA are coherent — process()'s parallel_for is
    // done) and published here; the GUI only displays the published strings. The
    // shared_ptr decouples the slot's lifetime from gui_manager's. The hook
    // is registered while the Status window is open and torn down on sim swap.
    std::shared_ptr<demand_published<std::vector<std::string>>> status_pub_ =
        std::make_shared<demand_published<std::vector<std::string>>>();
    std::size_t status_hook_id_ = 0;
    bool        status_hook_registered_ = false;
    void sync_status_hook();
    void remove_status_hook();   // remove from the current sim's env (if any)

    // Render individual dialogs
    void render_control_dialog();
    void render_status_dialog();
    void render_performance_window();
    void render_snapshot_resume_dialog();

    // Auto-save helpers
    void check_auto_save();
    void perform_auto_save();
    std::optional<std::pair<std::string, std::string>> find_latest_snapshot();
    std::string get_snapshot_dir() const;
};

} // namespace coast

#endif // COAST_GUI_MANAGER_H
