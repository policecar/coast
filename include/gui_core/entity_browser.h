#ifndef COAST_ENTITY_BROWSER_H
#define COAST_ENTITY_BROWSER_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "tools/sim_builder_cfg.h"

namespace coast {

class sim_instance;

namespace vis {
class vis_instance;
struct vis_info;
}

/**
 * Browser window for viewing simulation entities.
 * Displays entities in a tabular format with columns for:
 * - ID, Name, Type, Configuration, Output Size, Input Count
 *
 * Features:
 * - Filter/search functionality
 * - Sortable columns (click header to sort)
 *
 * Future features:
 * - On-demand visualization creation
 * - Selection and detail view
 */
class entity_browser {
public:
    explicit entity_browser(sim_instance& sim);
    ~entity_browser();

    /**
     * Render the browser window.
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

    // Filter state
    char filter_text[256] = "";

    // Cached entity data for sorting
    struct entity_row {
        std::size_t id;
        std::string name;
        sim::io_entity_type_t type;
        std::string config;
        std::size_t output_size;
        std::size_t input_count;
    };
    std::vector<entity_row> cached_rows;
    bool cache_dirty = true;

    // Selection state
    std::size_t selected_entity_id = SIZE_MAX;  // SIZE_MAX = no selection
    bool detail_panel_open = true;

    // Splitter state (ratio of table height to total available height)
    float splitter_ratio = 0.6f;
    static constexpr float SPLITTER_MIN_RATIO = 0.2f;
    static constexpr float SPLITTER_MAX_RATIO = 0.9f;

    // Active visualization instances (keyed by "vis_id:entity_id")
    std::unordered_map<std::string, std::unique_ptr<vis::vis_instance>> active_visualizations;

    void rebuild_cache();
    void render_browser_window();
    void render_visualization_windows();
    void render_detail_panel(const entity_row& selected);
    void update_visualizations();

    // Helper to generate unique key for a visualization instance
    static std::string make_vis_key(const std::string& vis_id, std::size_t entity_id);
};

} // namespace coast

#endif // COAST_ENTITY_BROWSER_H
