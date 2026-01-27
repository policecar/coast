#ifndef COAST_GUI_MANAGER_H
#define COAST_GUI_MANAGER_H

#include <vector>
#include <deque>
#include <string>

#include "sdl_app.h"
#include "sim_env.h"
#include "hd_ngm2_neuron_group.h"
#include "vec_ring_buffer.h"
#include "ngm_flat_vis.h"
#include "json_loader.h"
#include "config_editor.h"

namespace coast {

/**
 * Manages the main GUI elements including menu bar and dialogs.
 * Coordinates visibility state and rendering of simulation controls
 * and visualization windows.
 */
class gui_manager {
public:
    /**
     * Configuration for visualization windows.
     */
    struct vis_config_t {
        float ring_buffer_scale = 2.0f;
        float neurons_scale_first = 0.75f;
        float neurons_scale_other = 1.0f;
    };

    gui_manager(
        sdl_app& app,
        sim::env& sim_env,
        std::vector<ngm2::neuron_group_t>& groups,
        std::vector<vec_ring_buffer>& ring_buffers,
        std::vector<ngm_flat_vis>& visualizations,
        loader::global_cfg_t& config
    );

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
    [[nodiscard]] bool is_simulation_running() const { return simulation_running; }

    /**
     * Get the number of simulation steps per frame.
     */
    [[nodiscard]] int get_steps_per_frame() const { return steps_per_frame; }

    /**
     * Set visualization configuration.
     */
    void set_vis_config(const vis_config_t& config) { vis_config = config; }

    /**
     * Update activity history for plotting.
     * Should be called periodically during simulation.
     */
    void update_activity_history();

private:
    // References to external components
    sdl_app& app;
    sim::env& sim_env;
    std::vector<ngm2::neuron_group_t>& groups;
    std::vector<vec_ring_buffer>& ring_buffers;
    std::vector<ngm_flat_vis>& visualizations;

    // Configuration editor
    config_editor cfg_editor;

    // Dialog visibility state
    bool show_control = false;
    bool show_status = true;
    bool show_activity_plot = false;
    bool show_network_editor = false;
    bool show_config_editor = false;
    std::vector<bool> show_ring_buffer;
    std::vector<bool> show_neurons;

    // Simulation control state
    bool simulation_running = false;
    int steps_per_frame = 100;
    bool auto_steps = false;
    int auto_steps_cooldown = 0;
    static constexpr int AUTO_STEPS_COOLDOWN_FRAMES = 10;
    static constexpr float AUTO_STEPS_TARGET_FPS = 35.0f;
    static constexpr float AUTO_STEPS_P_COEFFICIENT = 0.4f;

    // Visualization configuration
    vis_config_t vis_config;

    // Activity plot data (rolling buffer per group)
    static constexpr std::size_t ACTIVITY_HISTORY_SIZE = 500;
    std::vector<std::deque<float>> activity_history;
    std::vector<float> time_axis;
    int activity_sample_counter = 0;

    // Render individual dialogs
    void render_control_dialog();
    void render_status_dialog();
    void render_ring_buffer_windows();
    void render_neuron_windows();
    void render_activity_plot();
    void render_network_editor();
};

} // namespace coast

#endif // COAST_GUI_MANAGER_H
