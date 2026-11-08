#ifndef COAST_MATRIX_CANVAS_H
#define COAST_MATRIX_CANVAS_H

#include <span>

#include "gui_vis/core/colormap.h"
#include "gui_vis/core/gl_texture.h"

namespace coast {

/**
 * Zoomable, color-coded matrix renderer.
 *
 * Stateless w.r.t. the data (the caller owns the value buffer and the header
 * thumbnail textures) but stateful w.r.t. the view: zoom and pan persist across
 * frames. The transform mirrors graph_canvas: screen = origin + (world + pan)*zoom,
 * with cursor-centered scroll-wheel zoom. Row/column header thumbnails are pinned
 * to the left/top edge and scale in lockstep with the body cells. Cell numbers
 * scale with zoom and are hidden once a cell shrinks below a legibility threshold.
 *
 * Must be called inside an existing ImGui window; it consumes the remaining
 * content region as its canvas.
 */
class matrix_canvas {
public:
    struct inputs {
        int rows = 0;
        int cols = 0;
        const float* values = nullptr;            // row-major, rows*cols, may be null/empty
        float vmin = 0.0f;
        float vmax = 1.0f;
        colormap_t palette = colormap_t::viridis;
        std::span<const gl_texture> row_thumbs;   // size == rows (optional; empty = no gutter)
        std::span<const gl_texture> col_thumbs;   // size == cols (optional; empty = no gutter)

        // Optional display->underlying permutations (null = identity). When set,
        // display index d reads underlying index order[d] for both cells and the
        // matching header thumbnail. At most one is expected to be non-identity.
        const int* row_order = nullptr;
        const int* col_order = nullptr;

        // Highlighted header to outline, in DISPLAY indices (-1 = none), plus the
        // sort-direction glyph to draw on it: +1 ascending (▲), -1 descending (▼),
        // 0 none.
        int highlight_row = -1;
        int highlight_col = -1;
        int sort_dir = 0;
    };

    // Result of one render: which header (if any) was clicked this frame, in
    // DISPLAY indices (-1 = none). A click is a press+release without a pan drag.
    struct interaction {
        int clicked_row = -1;
        int clicked_col = -1;
    };

    interaction render(const inputs& in);

    /// Forget zoom/pan so the next render re-fits the matrix to the gutter corner.
    void reset_view() { view_initialized_ = false; }

private:
    float zoom_ = 1.0f;
    float pan_x_ = 0.0f;   // world-space pan offset (base px units)
    float pan_y_ = 0.0f;
    bool  view_initialized_ = false;
};

} // namespace coast

#endif // COAST_MATRIX_CANVAS_H
