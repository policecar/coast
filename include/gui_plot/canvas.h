#ifndef COAST_GUI_PLOT_CANVAS_H
#define COAST_GUI_PLOT_CANVAS_H

#include "gui_plot/figure.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace coast::plot {

struct canvas_save_data; // defined in plot_detail.h (internal); param of snapshot_canvas

// The canvas's pan/zoom transform: screen = origin + (world + pan) * zoom, with
// the viewport's origin added by the caller. Pan is in WORLD units (pixels at
// zoom 1), which is what makes a pixel drag divide by zoom to track the cursor
// 1:1, and makes the view region-relative — an export at 2× the on-screen size
// frames identically (as long as the on-screen grid fits; once it overflows and
// scrolls, the export skips the min-cell floor and frames the whole grid instead,
// see canvas::paint_body).
//
// Deliberately ImGui-free and pure: this is the whole of the interesting math, so
// keeping it out of the painter lets it be tested without a GUI context
// (tests/unit/gui_plot/test_canvas_view.cpp).
struct canvas_view {
    float zoom  = 1.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;

    // Upper bound matches the export path's scale clamp (detail::export_scale);
    // below the lower bound a fitted grid has nothing legible left.
    static constexpr float min_zoom = 0.2f;
    static constexpr float max_zoom = 8.0f;

    void reset() { *this = canvas_view{}; }

    [[nodiscard]] float to_screen_x(float wx) const { return (wx + pan_x) * zoom; }
    [[nodiscard]] float to_screen_y(float wy) const { return (wy + pan_y) * zoom; }
    [[nodiscard]] float to_world_x(float sx) const { return sx / zoom - pan_x; }
    [[nodiscard]] float to_world_y(float sy) const { return sy / zoom - pan_y; }

    // Multiply zoom by `factor` (clamped to [min_zoom, max_zoom]), keeping the
    // world point currently under the viewport-local point (mx, my) exactly there.
    // Anchoring is the part graph_canvas gets wrong (it computes the anchor then
    // discards it unless smoothing is off — graph_canvas.cpp:220-226), so it is
    // tested directly.
    void zoom_by(float factor, float mx, float my);

    // Clamp pan to the scroll range of a grid with world-extent (gw,gh) inside a
    // viewport (vw,vh): pan ∈ [min(0, v/zoom − g), 0] per axis, collapsing to {0}
    // when the content fits. Keeps the grid from scrolling into empty space.
    void clamp_pan(float gw, float gh, float vw, float vh);

    // Scroll offset (screen px, ≥ 0) ↔ pan (world units, ≤ 0): scroll = −pan·zoom.
    // The scrollbars speak scroll; the transform speaks pan.
    [[nodiscard]] float scroll_x() const { return -pan_x * zoom; }
    [[nodiscard]] float scroll_y() const { return -pan_y * zoom; }
    void                set_scroll_x(float s) { pan_x = -s / zoom; }
    void                set_scroll_y(float s) { pan_y = -s / zoom; }
};

// Which scrollbars a grid of world-extent (gw,gh) at `zoom` needs inside a viewport
// (vw,vh), and the content area left after reserving `bar` px per shown bar. A shown
// bar shrinks the cross axis, which can pull the other bar in — resolved in two
// passes. avail is floored at 1 (a negative child size means "content-relative" to
// ImGui, not a tiny box). Pure (no ImGui) so it is unit-tested.
struct scroll_reserve {
    bool  need_h  = false;
    bool  need_v  = false;
    float avail_w = 0.0f;
    float avail_h = 0.0f;
};
[[nodiscard]] scroll_reserve compute_reserve(float gw, float gh, float zoom, float vw,
                                             float vh, float bar);

// A canvas subdivides ONE window into a uniform rows × cols grid of cells (like
// ROOT's TPad), each hosting a figure instead of it having its own window.
// Placed figures keep receiving update() normally; the canvas draws them in
// their cell. Cells are 0-indexed, row-major.
//
//   auto c = std::make_shared<coast::plot::canvas>("dash", 2, 3);
//   c->show();
//   auto h = std::make_shared<coast::plot::histogram>("h");  // not shown
//   c->place(h, 0, 0);
//
// The grid is scroll/zoomable: the wheel zooms about the cursor, middle-drag pans
// (clamped so the grid can't be dragged off-screen). Cells shrink to fit the
// window but never below a preferred minimum size — past that the grid overflows
// the viewport and scrolls, rather than cramming every cell onto one sheet. As long
// as every cell stays above that minimum the layout is exactly the classic
// fit-to-region grid; once the window is small enough (or the cells many enough)
// that a cell would shrink past the floor, it clamps to the minimum and the grid
// scrolls — so the "identical to the old grid" behaviour holds above the floor, not
// at every size. (An off-screen export skips the floor entirely and always frames
// the whole grid.) Zoom scales cell geometry AND cell contents (ImGui/ImPlot
// metrics + text), so a magnified cell looks like a bigger plot, not a stretched one.
class canvas final : public figure {
public:
    canvas(std::string title, int rows, int cols);
    ~canvas() override; // un-hosts its children so they are not left stuck invisible

    // Place a figure into cell (row, col). It stops showing as its own window and
    // renders in the cell, replacing any current occupant. A null handle or
    // out-of-range indices are ignored.
    void place(std::shared_ptr<figure> fig, int row, int col);
    void clear_cell(int row, int col); // empty the cell (occupant returns to its window)

    void save_h5(std::string path) override; // saves the whole grid
    void free_gl() override;
    [[nodiscard]] int rows() const { return rows_; }
    [[nodiscard]] int cols() const { return cols_; }

    // -- view --
    // Back to the fitted, unzoomed grid (zoom 1, no pan). Any thread.
    void reset_view();
    [[nodiscard]] float zoom() const; // current magnification; 1.0 == fit-to-region

    // Fill a canvas snapshot (grid dims + each occupied, saveable cell's DTO).
    // GUI thread, under the canvas lock. Used by flush_captures().
    bool snapshot_canvas(canvas_save_data& out);

protected:
    void paint_body(region r) override;
    void paint_controls() override; // zoom readout + reset
    [[nodiscard]] bool is_saveable() const override { return true; }

private:
    [[nodiscard]] bool        valid(int r, int c) const { return r >= 0 && r < rows_ && c >= 0 && c < cols_; }
    [[nodiscard]] std::size_t idx(int r, int c) const
    {
        return static_cast<std::size_t>(r) * static_cast<std::size_t>(cols_) + static_cast<std::size_t>(c);
    }

    // Wheel-zoom + middle-drag-pan against the viewport child. origin_x/y are the
    // viewport's screen-space origin (world (0,0) at zoom 1, pan 0). GUI thread,
    // m_ held. Called from paint_body only.
    void handle_view_input(float origin_x, float origin_y);
    // Draw the H/V scrollbars into the strips reserved beside the viewport and,
    // while one is dragged, write the new pan. Called in the PARENT window (after
    // the viewport child ends) so no cell child-window covers and steals the click.
    // (vp_x, vp_y) is the viewport top-left in screen space; aw/ah the viewport
    // (reserve-shrunk) size; gw/gh the grid's world extent; `bar` the bar thickness.
    // GUI thread, m_ held, interactive frames only.
    void draw_scrollbars(float vp_x, float vp_y, float aw, float ah, float gw, float gh,
                         bool need_h, bool need_v, float bar);
    void reset_view_locked(); // m_ already held (paint_controls); reset_view() locks

    const int                            rows_, cols_;
    std::vector<std::shared_ptr<figure>> cells_;

    // -- view state (m_; mutated on the GUI thread, read by reset_view/zoom) --
    canvas_view view_;
    // Anchor-based drag: latch the view's pan and the cursor at click, then track
    // absolutely. Accumulating per-frame deltas would drift against the zoom divisor.
    bool  panning_       = false;
    float pan_anchor_x_  = 0.0f, pan_anchor_y_ = 0.0f;  // pan when the drag started
    float drag_anchor_x_ = 0.0f, drag_anchor_y_ = 0.0f; // cursor when the drag started
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_CANVAS_H
