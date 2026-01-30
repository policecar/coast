#ifndef COAST_DATASET_BROWSER_H
#define COAST_DATASET_BROWSER_H

#include <string>
#include <vector>
#include <memory>
#include <span>
#include <functional>

#include "core/gl_texture.h"
#include "core/color_types.h"

namespace coast {

class sim_instance;

/**
 * Browser window for visually inspecting dataset samples.
 * Displays samples in a grid layout with virtualized scrolling
 * to handle large datasets efficiently.
 */
class dataset_browser {
public:
    explicit dataset_browser(sim_instance& sim);
    ~dataset_browser();

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

    // Dataset selection
    int selected_dataset_idx = -1;
    std::vector<std::string> dataset_names;
    std::vector<std::size_t> dataset_entity_ids;

    // Image info for selected dataset
    int img_width = 0;
    int img_height = 0;
    int img_channels = 0;
    std::size_t sample_count = 0;

    // Thumbnail display settings
    static constexpr int THUMB_SIZE = 64;      // Display size of each thumbnail
    static constexpr int THUMB_PADDING = 4;    // Padding between thumbnails
    static constexpr int VISIBLE_ROWS_BUFFER = 2;  // Extra rows to render above/below visible area

    // Texture cache for visible thumbnails
    struct cached_thumb {
        gl_texture texture;
        std::size_t sample_idx = SIZE_MAX;
        bool valid = false;
    };
    std::vector<cached_thumb> thumb_cache;
    std::vector<color_t> thumb_pixels;  // Reusable buffer for texture upload

    // Function to get sample data from selected dataset
    std::function<std::span<const float>(std::size_t)> get_sample_func;

    void rebuild_dataset_list();
    void select_dataset(int idx);
    void render_thumbnail_grid();
    void update_thumbnail(std::size_t cache_idx, std::size_t sample_idx);
    void convert_sample_to_rgba(std::span<const float> sample, std::vector<color_t>& out);
};

} // namespace coast

#endif // COAST_DATASET_BROWSER_H
