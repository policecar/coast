#include "gui_tools/matrix_canvas.h"

#include "imgui.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace coast {

namespace {

constexpr float CELL       = 56.0f;   // base cell size (px) at zoom == 1
constexpr float MIN_ZOOM   = 0.05f;
constexpr float MAX_ZOOM   = 25.0f;
constexpr float ZOOM_SPEED = 0.15f;
constexpr float NUM_HIDE_PX = 30.0f;  // hide cell numbers once a cell is smaller than this

ImU32 to_u32(color_t c) { return IM_COL32(c.r, c.g, c.b, c.a); }

// Black or white text, whichever contrasts better with the cell fill.
ImU32 text_color_for(color_t bg)
{
    const float lum = 0.299f * bg.r + 0.587f * bg.g + 0.114f * bg.b;
    return lum > 140.0f ? IM_COL32(0, 0, 0, 255) : IM_COL32(255, 255, 255, 255);
}

} // namespace

matrix_canvas::interaction matrix_canvas::render(const inputs& in)
{
    interaction act;
    if (in.rows <= 0 || in.cols <= 0 || in.values == nullptr) {
        ImGui::TextDisabled("No matrix to display — select samples and Compute.");
        return act;
    }

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.x = std::max(avail.x, 64.0f);
    avail.y = std::max(avail.y, 64.0f);
    const ImVec2 p1(p0.x + avail.x, p0.y + avail.y);

    // Reserve the region and make it a hover/drag target.
    ImGui::InvisibleButton("##matrix_canvas", avail,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool hovered = ImGui::IsItemHovered();

    // First frame for this matrix: place cell (0,0) just inside the gutter corner.
    // gutter_px == CELL*zoom and screen(0) == p0 + pan*zoom, so pan == CELL lands
    // the body exactly against the gutter at any zoom.
    if (!view_initialized_) {
        zoom_  = 1.0f;
        pan_x_ = CELL;
        pan_y_ = CELL;
        view_initialized_ = true;
    }

    ImGuiIO& io = ImGui::GetIO();

    // Cursor-centered scroll-wheel zoom (keep the world point under the cursor fixed).
    if (hovered && io.MouseWheel != 0.0f) {
        const float wx = (io.MousePos.x - p0.x) / zoom_ - pan_x_;
        const float wy = (io.MousePos.y - p0.y) / zoom_ - pan_y_;
        zoom_ = std::clamp(zoom_ * (1.0f + io.MouseWheel * ZOOM_SPEED), MIN_ZOOM, MAX_ZOOM);
        pan_x_ = (io.MousePos.x - p0.x) / zoom_ - wx;
        pan_y_ = (io.MousePos.y - p0.y) / zoom_ - wy;
    }

    // Drag to pan (left or right button).
    if (hovered && (ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
                    ImGui::IsMouseDragging(ImGuiMouseButton_Right))) {
        pan_x_ += io.MouseDelta.x / zoom_;
        pan_y_ += io.MouseDelta.y / zoom_;
    }

    const float cell_px   = CELL * zoom_;
    const float gutter_px = cell_px;            // header size == cell size (lockstep scaling)
    const float body_left = p0.x + gutter_px;
    const float body_top  = p0.y + gutter_px;

    // world (base px) -> screen
    auto sx = [&](float wx) { return p0.x + (wx + pan_x_) * zoom_; };
    auto sy = [&](float wy) { return p0.y + (wy + pan_y_) * zoom_; };
    // screen -> fractional cell index
    auto col_at = [&](float screen_x) { return ((screen_x - p0.x) / zoom_ - pan_x_) / CELL; };
    auto row_at = [&](float screen_y) { return ((screen_y - p0.y) / zoom_ - pan_y_) / CELL; };
    // display index -> underlying index (identity when no order array is given)
    auto urow = [&](int d) { return in.row_order ? in.row_order[d] : d; };
    auto ucol = [&](int d) { return in.col_order ? in.col_order[d] : d; };

    const int c0 = std::max(0, static_cast<int>(std::floor(col_at(body_left))));
    const int c1 = std::min(in.cols - 1, static_cast<int>(std::floor(col_at(p1.x))));
    const int r0 = std::max(0, static_cast<int>(std::floor(row_at(body_top))));
    const int r1 = std::min(in.rows - 1, static_cast<int>(std::floor(row_at(p1.y))));

    const float range = (in.vmax > in.vmin) ? (in.vmax - in.vmin) : 1.0f;
    ImFont* font = ImGui::GetFont();
    const bool  show_numbers = cell_px >= NUM_HIDE_PX;
    const float font_size = std::clamp(cell_px * 0.26f, 6.0f, 72.0f);

    // ---- body cells ----
    dl->PushClipRect(ImVec2(body_left, body_top), p1, true);
    for (int r = r0; r <= r1; ++r) {
        for (int c = c0; c <= c1; ++c) {
            const float v = in.values[static_cast<std::size_t>(urow(r)) * in.cols + ucol(c)];
            const float t = std::clamp((v - in.vmin) / range, 0.0f, 1.0f);
            const color_t fill = apply_colormap(in.palette, t);

            const ImVec2 a(sx(static_cast<float>(c) * CELL), sy(static_cast<float>(r) * CELL));
            const ImVec2 b(a.x + cell_px, a.y + cell_px);
            dl->AddRectFilled(a, b, to_u32(fill));
            dl->AddRect(a, b, IM_COL32(0, 0, 0, 40));

            if (show_numbers) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.2f", v);
                const ImVec2 ts = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, buf);
                const ImVec2 tp(a.x + (cell_px - ts.x) * 0.5f, a.y + (cell_px - ts.y) * 0.5f);
                dl->AddText(font, font_size, tp, text_color_for(fill), buf);
            }
        }
    }
    dl->PopClipRect();

    // ---- column headers (top gutter): body X-transform, pinned Y ----
    if (!in.col_thumbs.empty()) {
        dl->PushClipRect(ImVec2(body_left, p0.y), ImVec2(p1.x, body_top), true);
        for (int c = c0; c <= c1; ++c) {
            const int uc = ucol(c);
            if (static_cast<std::size_t>(uc) >= in.col_thumbs.size() ||
                !in.col_thumbs[uc].is_valid()) continue;
            const ImVec2 a(sx(static_cast<float>(c) * CELL), p0.y);
            const ImVec2 b(a.x + cell_px, p0.y + gutter_px);
            dl->AddImage(in.col_thumbs[uc].as_imgui_texture(), a, b);
            dl->AddRect(a, b, IM_COL32(0, 0, 0, 60));
        }
        dl->PopClipRect();
    }

    // ---- row headers (left gutter): body Y-transform, pinned X ----
    if (!in.row_thumbs.empty()) {
        dl->PushClipRect(ImVec2(p0.x, body_top), ImVec2(body_left, p1.y), true);
        for (int r = r0; r <= r1; ++r) {
            const int ur = urow(r);
            if (static_cast<std::size_t>(ur) >= in.row_thumbs.size() ||
                !in.row_thumbs[ur].is_valid()) continue;
            const ImVec2 a(p0.x, sy(static_cast<float>(r) * CELL));
            const ImVec2 b(p0.x + gutter_px, a.y + cell_px);
            dl->AddImage(in.row_thumbs[ur].as_imgui_texture(), a, b);
            dl->AddRect(a, b, IM_COL32(0, 0, 0, 60));
        }
        dl->PopClipRect();
    }

    // ---- corner box (covers where the two gutters meet) ----
    dl->AddRectFilled(p0, ImVec2(body_left, body_top), IM_COL32(30, 30, 30, 255));

    // ---- selection highlight + sort-direction glyph on the focused header ----
    // sort_dir: +1 ascending (▲), -1 descending (▼), 0 none. The glyph sits in the
    // header's top-right corner over a dark backing so it reads over any thumbnail.
    constexpr ImU32 SEL = IM_COL32(255, 230, 0, 255);
    auto draw_glyph = [&](ImVec2 a, ImVec2 b) {
        if (in.sort_dir == 0) return;
        const float s  = std::min(b.x - a.x, b.y - a.y) * 0.16f;
        const float cx = b.x - s - 3.0f;
        const float cy = a.y + s + 3.0f;
        dl->AddCircleFilled(ImVec2(cx, cy), s + 2.0f, IM_COL32(0, 0, 0, 180));
        if (in.sort_dir < 0)   // descending: down-pointing triangle ▼
            dl->AddTriangleFilled(ImVec2(cx - s, cy - s), ImVec2(cx + s, cy - s),
                                  ImVec2(cx, cy + s), SEL);
        else                   // ascending: up-pointing triangle ▲
            dl->AddTriangleFilled(ImVec2(cx - s, cy + s), ImVec2(cx + s, cy + s),
                                  ImVec2(cx, cy - s), SEL);
    };
    if (in.highlight_col >= 0 && in.highlight_col < in.cols) {
        dl->PushClipRect(ImVec2(body_left, p0.y), ImVec2(p1.x, body_top), true);
        const ImVec2 a(sx(static_cast<float>(in.highlight_col) * CELL), p0.y);
        const ImVec2 b(a.x + cell_px, body_top);
        dl->AddRect(a, b, SEL, 0.0f, 0, 3.0f);
        draw_glyph(a, b);
        dl->PopClipRect();
    }
    if (in.highlight_row >= 0 && in.highlight_row < in.rows) {
        dl->PushClipRect(ImVec2(p0.x, body_top), ImVec2(body_left, p1.y), true);
        const ImVec2 a(p0.x, sy(static_cast<float>(in.highlight_row) * CELL));
        const ImVec2 b(body_left, a.y + cell_px);
        dl->AddRect(a, b, SEL, 0.0f, 0, 3.0f);
        draw_glyph(a, b);
        dl->PopClipRect();
    }

    // ---- legend: colorbar with numeric min/max (screen-fixed, bottom-left) ----
    {
        const float bar_w = 150.0f, bar_h = 14.0f, pad = 8.0f;
        const ImVec2 bp0(p0.x + pad, p1.y - bar_h - pad);
        constexpr int steps = 64;
        for (int i = 0; i < steps; ++i) {
            const float t = static_cast<float>(i) / steps;
            const color_t c = apply_colormap(in.palette, t);
            const ImVec2 q0(bp0.x + t * bar_w, bp0.y);
            const ImVec2 q1(bp0.x + static_cast<float>(i + 1) / steps * bar_w, bp0.y + bar_h);
            dl->AddRectFilled(q0, q1, to_u32(c));
        }
        dl->AddRect(bp0, ImVec2(bp0.x + bar_w, bp0.y + bar_h), IM_COL32(255, 255, 255, 180));

        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "%.3g", in.vmin);
        dl->AddText(ImVec2(bp0.x, bp0.y - 16.0f), IM_COL32(255, 255, 255, 255), lbl);
        std::snprintf(lbl, sizeof(lbl), "%.3g", in.vmax);
        const ImVec2 ts = ImGui::CalcTextSize(lbl);
        dl->AddText(ImVec2(bp0.x + bar_w - ts.x, bp0.y - 16.0f), IM_COL32(255, 255, 255, 255), lbl);
    }

    dl->AddRect(p0, p1, IM_COL32(255, 255, 255, 80));

    // ---- header click (select) vs. drag (pan) ----
    // A left release with no past-threshold drag is a click. GetMouseDragDelta with
    // the default threshold returns (0,0) when the press never exceeded
    // io.MouseDragThreshold, so a non-zero delta means the user was panning.
    if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        const ImVec2 dd = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        const bool was_pan = (dd.x != 0.0f || dd.y != 0.0f);
        if (!was_pan) {
            const ImVec2 m = io.MousePos;
            const bool in_top  = m.x >= body_left && m.x < p1.x &&
                                 m.y >= p0.y && m.y < body_top;
            const bool in_left = m.x >= p0.x && m.x < body_left &&
                                 m.y >= body_top && m.y < p1.y;
            if (in_top) {
                const int c = static_cast<int>(std::floor(col_at(m.x)));
                if (c >= 0 && c < in.cols) act.clicked_col = c;
            } else if (in_left) {
                const int r = static_cast<int>(std::floor(row_at(m.y)));
                if (r >= 0 && r < in.rows) act.clicked_row = r;
            }
        }
    }

    return act;
}

} // namespace coast
