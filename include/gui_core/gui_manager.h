#ifndef COAST_GUI_MANAGER_H
#define COAST_GUI_MANAGER_H

#include <vector>

#include "sdl_app.h"
#include "sim_env.h"
#include "hd_ngm2_neuron_group.h"
#include "vec_ring_buffer.h"
#include "ngm_flat_vis.h"

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
        std::vector<ngm_flat_vis>& visualizations
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

private:
    // References to external components
    sdl_app& app;
    sim::env& sim_env;
    std::vector<ngm2::neuron_group_t>& groups;
    std::vector<vec_ring_buffer>& ring_buffers;
    std::vector<ngm_flat_vis>& visualizations;

    // Dialog visibility state
    bool show_control = false;
    bool show_status = true;
    std::vector<bool> show_ring_buffer;
    std::vector<bool> show_neurons;

    // Simulation control state
    bool simulation_running = false;
    int steps_per_frame = 100;

    // Visualization configuration
    vis_config_t vis_config;

    // Render individual dialogs
    void render_control_dialog();
    void render_status_dialog();
    void render_ring_buffer_windows();
    void render_neuron_windows();
};

} // namespace coast

#endif // COAST_GUI_MANAGER_H
