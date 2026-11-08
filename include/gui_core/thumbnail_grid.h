#ifndef COAST_THUMBNAIL_GRID_H
#define COAST_THUMBNAIL_GRID_H

#include <cstddef>
#include <set>
#include <vector>

#include "gui_vis/core/color_types.h"
#include "gui_vis/core/gl_texture.h"
#include "gui_core/dataset_access.h"

namespace coast {

/**
 * Reusable virtualized thumbnail grid over a dataset_handle's samples.
 *
 * Only the rows currently scrolled into view are textured (texture cache sized to
 * the visible window), so it scales to datasets with tens of thousands of samples.
 *
 * Selection is optional:
 *   - sel == nullptr : view-only (the Dataset Browser). Plain images, no clicks.
 *   - sel != nullptr : clicking a thumbnail toggles its index in *sel, and
 *                      selected thumbnails are drawn with a highlight border.
 *
 * Must be rendered inside an existing ImGui window/child; it creates its own
 * scrolling child region internally.
 */
class thumbnail_grid {
public:
    void render(const dataset_handle& ds, std::set<std::size_t>* sel);
    void free_resources();

private:
    static constexpr int THUMB_SIZE = 64;
    static constexpr int THUMB_PADDING = 4;
    static constexpr int VISIBLE_ROWS_BUFFER = 2;

    struct cached_thumb {
        gl_texture texture;
        std::size_t sample_idx = SIZE_MAX;
        bool valid = false;
    };
    std::vector<cached_thumb> thumb_cache;
    std::vector<color_t> thumb_pixels;  // reusable upload buffer

    void update_thumbnail(const dataset_handle& ds, std::size_t cache_idx, std::size_t sample_idx);
};

} // namespace coast

#endif // COAST_THUMBNAIL_GRID_H
