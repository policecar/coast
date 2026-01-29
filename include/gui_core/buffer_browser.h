#ifndef COAST_BUFFER_BROWSER_H
#define COAST_BUFFER_BROWSER_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace coast {

class sim_instance;

namespace vis {
class vis_instance;
struct vis_info;
}

/**
 * Browser window for viewing IO buffers.
 * Provides a list of all IO buffers in the simulation with options
 * to view their state and visualizations.
 *
 * Features:
 * - Buffer listing with ID, Size, and statistics
 * - Row selection with detail panel
 * - On-demand visualization creation via registry
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
    sim_instance& sim;
    bool is_visible = false;

    // Cached buffer data for display
    struct buffer_row {
        std::size_t id;
        std::size_t size;
        float sum;
        float avg;
        float min_val;
        float max_val;
    };
    std::vector<buffer_row> cached_rows;
    bool cache_dirty = true;

    // Selection state
    std::size_t selected_buffer_id = SIZE_MAX;  // SIZE_MAX = no selection
    bool detail_panel_open = true;

    // Active visualization instances (keyed by "vis_id:buffer_id")
    std::unordered_map<std::string, std::unique_ptr<vis::vis_instance>> active_visualizations;

    void rebuild_cache();
    void render_browser_window();
    void render_detail_panel(const buffer_row& selected);
    void update_visualizations();

    static std::string make_vis_key(const std::string& vis_id, std::size_t buffer_id);
};

} // namespace coast

#endif // COAST_BUFFER_BROWSER_H
