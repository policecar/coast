#include "gui_plot/canvas.h"

#include "gui_plot/plot_detail.h"  // canvas_save_data, detail::scale_implot_style
#include "gui_plot/plot_manager.h" // enqueue_canvas_h5
#include "imgui.h"
#include "imgui_internal.h" // ScrollbarEx, ImRect, ImGuiAxis for the manual scrollbars
#include "implot.h"         // ImPlot::GetStyle for the zoom's ImPlot half

#include <algorithm>
#include <cmath> // std::lround for the pan↔scroll conversion
#include <cstdio>
#include <optional>

namespace coast::plot {

namespace {

constexpr float k_zoom_speed = 0.1f; // per wheel tick, multiplicative

// Below this a cell has no room for a plot: ImPlot's chrome costs ~46×38 px at
// scale 1, and BeginPlot happily returns true with a NEGATIVE interior rather than
// refusing (measured — plot_zoom_toy/FINDINGS.md, E4). Skipping the body here is
// both the guard against that and the reason zooming out stays cheap.
constexpr float k_min_cell_w = 48.0f;
constexpr float k_min_cell_h = 40.0f;

// Preferred minimum cell size: the smallest a cell shrinks to before the grid
// stops fitting-to-window and starts SCROLLING instead. Distinct from
// k_min_cell_w/h (the hard "no room for any plot at all" floor, ~ImPlot's chrome):
// this is still-a-legible-plot. Once cols·k_pref_cell_w exceeds the viewport the
// grid overflows and the scrollbars/pan take over.
constexpr float k_pref_cell_w = 200.0f;
constexpr float k_pref_cell_h = 160.0f;

} // namespace

void canvas_view::zoom_by(float factor, float mx, float my)
{
    const float z1 = std::clamp(zoom * factor, min_zoom, max_zoom);
    if (z1 == zoom) return; // already railed, or a no-op factor
    // The world point under the cursor must not move:
    //   to_world(m) before == to_world(m) after
    //   mx/z0 - pan0 == mx/z1 - pan1   =>   pan1 = mx/z1 - (mx/z0 - pan0)
    const float world_x = to_world_x(mx);
    const float world_y = to_world_y(my);
    zoom                = z1;
    pan_x               = mx / z1 - world_x;
    pan_y               = my / z1 - world_y;
}

canvas::canvas(std::string title, int rows, int cols)
    : figure(std::move(title)), rows_(std::max(rows, 1)), cols_(std::max(cols, 1)),
      cells_(static_cast<std::size_t>(std::max(rows, 1)) * static_cast<std::size_t>(std::max(cols, 1)))
{
}
canvas::~canvas()
{
    std::scoped_lock lk(m_);
    for (auto& c : cells_)
        if (c) c->set_hosted(false);
}

void canvas::place(std::shared_ptr<figure> fig, int row, int col)
{
    if (!fig) return;
    std::scoped_lock lk(m_);
    if (!valid(row, col)) return;
    auto& slot = cells_[idx(row, col)];
    if (slot && slot != fig) slot->set_hosted(false);
    fig->set_hosted(true);
    slot = std::move(fig);
}
void canvas::clear_cell(int row, int col)
{
    std::scoped_lock lk(m_);
    if (!valid(row, col)) return;
    auto& slot = cells_[idx(row, col)];
    if (slot) {
        slot->set_hosted(false); // returns to its own window
        slot.reset();
    }
}

void canvas::free_gl()
{
    std::scoped_lock lk(m_);
    for (auto& c : cells_)
        if (c) c->free_gl();
}

void canvas::reset_view()
{
    std::scoped_lock lk(m_);
    reset_view_locked();
}
void canvas::reset_view_locked()
{
    view_.reset();
    panning_ = false;
}
float canvas::zoom() const
{
    std::scoped_lock lk(m_);
    return view_.zoom;
}

void canvas::handle_view_input(float origin_x, float origin_y)
{
    ImGuiIO& io = ImGui::GetIO();
    // ChildWindows: a cell is a child window, so hovering a CELL must still count
    // as hovering the viewport — otherwise the wheel would only work over gutters.
    const bool hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);

    if (hovered && io.MouseWheel != 0.0f) {
        view_.zoom_by(1.0f + io.MouseWheel * k_zoom_speed, io.MousePos.x - origin_x,
                      io.MousePos.y - origin_y);
        // Consume it. The cells are drawn after this, and ImPlot zooms its axes on
        // the wheel too — line/histogram get ImPlot's default persistent zoom (no
        // SetupAxes call), so without this a wheel tick over one of them would zoom
        // the canvas AND the plot's axes at once. The canvas owns the wheel; ImPlot
        // keeps left-drag inside a cell, and middle-drag pans the grid.
        // Zeroed unconditionally, not only when the zoom changed, so ownership does
        // not silently transfer to the cell once the zoom hits a rail.
        io.MouseWheel = 0.0f;
    }

    // Middle-drag pans. Left is left alone: it belongs to ImPlot inside a cell.
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
        panning_       = true;
        drag_anchor_x_ = io.MousePos.x;
        drag_anchor_y_ = io.MousePos.y;
        pan_anchor_x_  = view_.pan_x;
        pan_anchor_y_  = view_.pan_y;
    }
    if (panning_) {
        // Deliberately not gated on `hovered`: a drag that leaves the viewport must
        // keep tracking, or the grid sticks whenever the cursor overshoots an edge.
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
            // /zoom turns a pixel drag into world units, so the grid tracks the
            // cursor 1:1 on screen at any magnification.
            view_.pan_x = pan_anchor_x_ + (io.MousePos.x - drag_anchor_x_) / view_.zoom;
            view_.pan_y = pan_anchor_y_ + (io.MousePos.y - drag_anchor_y_) / view_.zoom;
        } else {
            panning_ = false;
        }
    }
}

void canvas_view::clamp_pan(float gw, float gh, float vw, float vh)
{
    // Pan is world-space and the grid occupies world [0,gw]×[0,gh]. The scroll
    // range runs from showing the left/top edge (pan 0) to showing the right/bottom
    // edge (pan = viewport/zoom − grid, negative). When the content fits
    // (grid·zoom ≤ viewport) that lower bound is ≥ 0, so the range collapses to {0}
    // — a small/unscrolled grid stays pinned exactly where it fits.
    const float min_x = std::min(0.0f, vw / zoom - gw);
    const float min_y = std::min(0.0f, vh / zoom - gh);
    pan_x             = std::clamp(pan_x, min_x, 0.0f);
    pan_y             = std::clamp(pan_y, min_y, 0.0f);
}

scroll_reserve compute_reserve(float gw, float gh, float zoom, float vw, float vh, float bar)
{
    // A shown bar reserves `bar` px on the cross axis, which can pull the other bar
    // in — two passes reach a self-consistent pair (need_h is resolved LAST, against
    // the refined need_v, so the only way it newly turns on is when need_v already
    // did). > 0.5 px so sub-pixel float slop doesn't flicker a bar on.
    const float cw   = gw * zoom;
    const float ch   = gh * zoom;
    const auto  over = [](float content, float avail) { return content - avail > 0.5f; };
    bool        need_v = over(ch, vh);
    bool        need_h = over(cw, vw);
    need_v             = over(ch, vh - (need_h ? bar : 0.0f));
    need_h             = over(cw, vw - (need_v ? bar : 0.0f));
    // max(…,1): a negative child size means "content-relative" to ImGui, not a tiny
    // box, which would blow the layout up on a viewport thinner than a scrollbar.
    return {need_h, need_v, std::max(vw - (need_v ? bar : 0.0f), 1.0f),
            std::max(vh - (need_h ? bar : 0.0f), 1.0f)};
}

void canvas::draw_scrollbars(float vp_x, float vp_y, float aw, float ah, float gw, float gh,
                             bool need_h, bool need_v, float bar)
{
    // Scroll is screen-space and non-negative; pan is world-space and ≤ 0:
    //   scroll = -pan * zoom,  pan = -scroll / zoom.
    // ScrollbarEx sizes the thumb from visible/contents and, while held, writes the
    // clamped scroll back into `s` — mapped straight back onto pan (next frame).
    const float z = view_.zoom;
    if (need_v) {
        ImS64        s = static_cast<ImS64>(std::lround(view_.scroll_y()));
        const ImRect bb(ImVec2(vp_x + aw, vp_y), ImVec2(vp_x + aw + bar, vp_y + ah));
        if (ImGui::ScrollbarEx(bb, ImGui::GetID("##canvas_vscroll"), ImGuiAxis_Y, &s,
                               static_cast<ImS64>(std::lround(ah)),
                               static_cast<ImS64>(std::lround(gh * z))))
            view_.set_scroll_y(static_cast<float>(s));
    }
    if (need_h) {
        ImS64        s = static_cast<ImS64>(std::lround(view_.scroll_x()));
        const ImRect bb(ImVec2(vp_x, vp_y + ah), ImVec2(vp_x + aw, vp_y + ah + bar));
        if (ImGui::ScrollbarEx(bb, ImGui::GetID("##canvas_hscroll"), ImGuiAxis_X, &s,
                               static_cast<ImS64>(std::lround(aw)),
                               static_cast<ImS64>(std::lround(gw * z))))
            view_.set_scroll_x(static_cast<float>(s));
    }
}

void canvas::paint_body(region rg)
{
    // "Chrome" — the min-cell floor plus the scrollbars/pan — only applies to a
    // live, TOP-LEVEL canvas. An off-screen export has no scrollbars (flooring there
    // would just clip the grid to the top-left, so it keeps pure fit-to-region and
    // frames the whole grid); a canvas HOSTED in another canvas's cell would
    // otherwise floor against the tiny cell and draw its own bars inside it. Both
    // fall back to the classic fit-to-region grid.
    const bool chrome = rg.interactive && !is_hosted();

    // Base (zoom-1) cell size: shrink-to-fit, floored at the preferred minimum where
    // chrome applies (past the floor the grid overflows and scrolls); pure
    // fit-to-region otherwise. Read the gutter BEFORE scaling the style — the
    // transform applies the zoom, so a scaled ItemSpacing here would apply it twice.
    const float pad   = ImGui::GetStyle().ItemSpacing.x;
    const float fit_w = (rg.w - pad * static_cast<float>(cols_ - 1)) / static_cast<float>(cols_);
    const float fit_h = (rg.h - pad * static_cast<float>(rows_ - 1)) / static_cast<float>(rows_);
    const float bw    = chrome ? std::max(fit_w, k_pref_cell_w) : std::max(fit_w, 1.0f);
    const float bh    = chrome ? std::max(fit_h, k_pref_cell_h) : std::max(fit_h, 1.0f);
    // Grid extent in world (zoom-1) px — the scroll content size, used below.
    const float grid_w = static_cast<float>(cols_) * bw + static_cast<float>(cols_ - 1) * pad;
    const float grid_h = static_cast<float>(rows_) * bh + static_cast<float>(rows_ - 1) * pad;

    // Which scrollbars are needed and the content area left after reserving them.
    // Read the bar thickness now, before the style is zoom-scaled below (chrome
    // stays fixed size). One-frame lag by design: this uses the PREVIOUS frame's
    // zoom — handle_view_input applies this frame's wheel a few lines down — so on
    // the frame you cross the overflow threshold the reserve trails by one frame.
    // Self-correcting and imperceptible; sizing the child up front is what lets the
    // reserved strips exist before we read input from inside the child.
    const float          bar = ImGui::GetStyle().ScrollbarSize;
    const scroll_reserve res = chrome ? compute_reserve(grid_w, grid_h, view_.zoom, rg.w, rg.h, bar)
                                      : scroll_reserve{false, false, rg.w, rg.h};

    const ImVec2 vp = ImGui::GetCursorScreenPos(); // viewport/child top-left (screen)

    // The viewport clips the panned grid. NoScrollWithMouse keeps ImGui's own
    // scrolling off the wheel so the zoom can have it. Sized to the reserved area so
    // no cell reaches under a scrollbar strip.
    ImGui::BeginChild("##viewport", ImVec2(res.avail_w, res.avail_h), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    const ImVec2 org_local  = ImGui::GetCursorPos();       // for SetCursorPos
    const ImVec2 org_screen = ImGui::GetCursorScreenPos(); // for the input math
    // Non-chrome frames (export capture, or a hosted canvas) must not touch view
    // state: a capture's parked cursor would turn an in-progress middle-drag into
    // pan = -inf (the parked -FLT_MAX enters the anchor math), and a release during
    // the capture would make that permanent. Skipping entirely also leaves
    // panning_/anchors intact, so a live drag resumes on the next interactive frame.
    if (chrome) {
        handle_view_input(org_screen.x, org_screen.y);
        view_.clamp_pan(grid_w, grid_h, res.avail_w, res.avail_h); // wheel-anchor + drag land here
    }

    // Scale the cell CONTENTS to match the cell geometry, so a magnified cell reads
    // as a bigger plot and not a stretched one. Same recipe the off-screen exporter
    // uses (plot_capture.cpp), and non-idempotent — hence scaling saved pristine
    // copies and restoring them below rather than compounding per frame.
    const float zoom   = view_.zoom;
    const bool  scaled = (zoom != 1.0f);
    // Only touch the styles when actually zoomed: it keeps an unzoomed canvas off
    // ImPlot entirely (an empty one then needs no ImPlot context at all) and saves
    // copying both style structs every frame at rest.
    std::optional<ImGuiStyle>  saved_g;
    std::optional<ImPlotStyle> saved_p;
    if (scaled) {
        saved_g = ImGui::GetStyle();
        saved_p = ImPlot::GetStyle();
        ImGui::GetStyle().ScaleAllSizes(zoom);
        detail::scale_implot_style(ImPlot::GetStyle(), zoom);
        ImGui::GetStyle().FontScaleMain *= zoom; // ScaleAllSizes is geometry-only
    }

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            const float wx = static_cast<float>(c) * (bw + pad); // world (zoom-1 px)
            const float wy = static_cast<float>(r) * (bh + pad);
            ImGui::SetCursorPos(ImVec2(org_local.x + view_.to_screen_x(wx),
                                       org_local.y + view_.to_screen_y(wy)));
            const ImVec2 sz(bw * zoom, bh * zoom);
            char         cid[32];
            std::snprintf(cid, sizeof cid, "##cell_%d_%d", r, c);
            // false == ImGui culled the cell (fully outside the viewport). Its
            // contents cost nothing then; partially visible cells are clipped to the
            // intersection automatically. EndChild is required either way.
            const bool visible = ImGui::BeginChild(cid, sz, ImGuiChildFlags_Borders,
                                                   ImGuiWindowFlags_NoScrollbar |
                                                       ImGuiWindowFlags_NoScrollWithMouse);
            if (visible) {
                auto& child = cells_[idx(r, c)];
                if (!child) {
                    ImGui::TextDisabled("(empty)");
                } else if (sz.x >= k_min_cell_w && sz.y >= k_min_cell_h) {
                    const ImVec2 avail = ImGui::GetContentRegionAvail();
                    // Propagate interactivity: a hosted canvas inside a captured
                    // canvas must skip its own input handling too.
                    child->body_painter()(region{avail.x, avail.y, rg.interactive});
                }
                // else: too small to host a plot — the cell's border still shows
                // where it is. See k_min_cell_w.
            }
            ImGui::EndChild();
        }
    }

    if (scaled) {
        ImGui::GetStyle()  = *saved_g;
        ImPlot::GetStyle() = *saved_p;
    }
    ImGui::EndChild();

    // Scrollbars live in the reserved strips outside the child, in the PARENT
    // window (drawn after EndChild), so no cell child-window sits in front to steal
    // the click. A drag here sets pan for the NEXT frame; that 1-frame catch-up is
    // imperceptible for a drag — unlike wheel-zoom, which must anchor same-frame,
    // the reason pan stays a hand-rolled transform and not ImGui's own (one-frame
    // late) scroll.
    if (chrome)
        draw_scrollbars(vp.x, vp.y, res.avail_w, res.avail_h, grid_w, grid_h, res.need_h,
                        res.need_v, bar);
}

void canvas::paint_controls()
{
    ImGui::Text("zoom %.2fx", static_cast<double>(view_.zoom));
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset view")) reset_view_locked(); // m_ already held
    ImGui::SameLine();
    ImGui::TextDisabled("(wheel: zoom, middle-drag/scrollbars: pan)");
    ImGui::SameLine(); // let the base's Save row continue this line
}

void canvas::save_h5(std::string path)
{
    float rw = 0.0f, rh = 0.0f;
    {
        std::scoped_lock lk(m_);
        rw = last_w_;
        rh = last_h_;
    }
    plot_manager::instance().enqueue_canvas_h5(weak_from_this(), std::move(path), rw, rh);
}

bool canvas::snapshot_canvas(canvas_save_data& out)
{
    std::scoped_lock lk(m_);
    out.title = title_;
    out.rows  = rows_;
    out.cols  = cols_;
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < cols_; ++c) {
            auto& child = cells_[idx(r, c)];
            if (!child) continue;
            h5_save_data d;
            if (child->snapshot_h5(d)) out.cells.push_back({r, c, std::move(d)});
        }
    return true;
}

} // namespace coast::plot
