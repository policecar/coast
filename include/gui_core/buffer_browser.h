#ifndef COAST_BUFFER_BROWSER_H
#define COAST_BUFFER_BROWSER_H

#include "core/demand_published.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sim { class env; }

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

    // Per-buffer statistics computed ON THE SIM THREAD at the post_swap boundary
    // (where the read pages are coherent and GPU readback is safe) and published
    // to the GUI. No entity name here — that is immutable structure the GUI looks
    // up itself.
    struct buffer_stat {
        std::size_t id;
        std::size_t size;
        float sum;
        float avg;
        float min_val;
        float max_val;
    };

    // Demand-driven publish slot (shared_ptr captured by the post_swap hook, so
    // the slot's lifetime does not depend on this browser's). Recomputed at GUI
    // consumption rate, not the step rate.
    std::shared_ptr<demand_published<std::vector<buffer_stat>>> stats_pub_ =
        std::make_shared<demand_published<std::vector<buffer_stat>>>();
    std::size_t stats_hook_id_ = 0;
    bool        stats_hook_registered_ = false;

    // Cached buffer data for display (built on the GUI thread from the published
    // stats + immutable entity names).
    struct buffer_row {
        std::size_t id;
        std::string output_of;  // Entity name that outputs to this buffer
        std::size_t size;
        float sum;
        float avg;
        float min_val;
        float max_val;
    };
    std::vector<buffer_row> cached_rows;
    bool cache_dirty = true;

    // Filter state
    char filter_text[256] = {};

    // Selection state
    std::size_t selected_buffer_id = SIZE_MAX;  // SIZE_MAX = no selection
    bool detail_panel_open = true;

    // Active visualization instances (keyed by "vis_id:buffer_id")
    std::unordered_map<std::string, std::unique_ptr<vis::vis_instance>> active_visualizations;

    void rebuild_cache();
    void render_browser_window();
    void render_visualization_windows();
    void render_detail_panel(const buffer_row& selected);
    void update_visualizations();

    // Register/remove the post_swap stats hook to track window visibility.
    void sync_stats_hook();
    // Sim-thread stats sweep over every io_buffer's read page (read_back_sync for
    // GPU-only buffers). Static so the hook never captures `this`.
    static std::vector<buffer_stat> compute_buffer_stats(sim::env& env);

    static std::string make_vis_key(const std::string& vis_id, std::size_t buffer_id);
};

} // namespace coast

#endif // COAST_BUFFER_BROWSER_H
