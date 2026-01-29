#ifndef COAST_BUFFER_BROWSER_H
#define COAST_BUFFER_BROWSER_H

namespace coast {

class sim_instance;

/**
 * Browser window for viewing IO buffers.
 * Provides a list of all IO buffers in the simulation with options
 * to view their state and visualizations.
 *
 * Currently a placeholder - future implementation will include:
 * - Buffer listing with "View" buttons
 * - On-demand visualization creation (vec_ring_buffer)
 * - Post-process hook management (activated when buffers are visualized)
 */
class buffer_browser {
public:
    explicit buffer_browser(sim_instance& sim);
    ~buffer_browser();

    /**
     * Render the browser window and any open visualizations.
     */
    void render();

    /**
     * Access visibility state for menu toggle.
     */
    bool& visible() { return is_visible; }

    /**
     * Free GPU/OpenGL resources.
     */
    void free_resources();

private:
    [[maybe_unused]] sim_instance& sim;
    bool is_visible = false;

    void render_browser_window();
};

} // namespace coast

#endif // COAST_BUFFER_BROWSER_H
