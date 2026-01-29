#ifndef COAST_ENTITY_BROWSER_H
#define COAST_ENTITY_BROWSER_H

namespace coast {

class sim_instance;

/**
 * Browser window for viewing simulation entities.
 * Provides a list of all entities in the simulation with options
 * to view their state and visualizations.
 *
 * Currently a placeholder - future implementation will include:
 * - Entity listing with "View" buttons
 * - On-demand visualization creation (ngm_flat_vis)
 * - Activity plot integration
 */
class entity_browser {
public:
    explicit entity_browser(sim_instance& sim);
    ~entity_browser();

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

#endif // COAST_ENTITY_BROWSER_H
