#include "gui_plot/pixels.h"

#include "gui_plot/plot_tools.h" // reshape_field/draw_colormap_grid + ImPlot

#include <algorithm>

namespace coast::plot {

pixels::pixels(std::string title, int width, int height)
    : figure(std::move(title)), width_(std::max(width, 1)), height_(std::max(height, 1))
{
}

void pixels::update(std::span<const double> y)
{
    std::scoped_lock lk(m_);
    y_.assign(y.begin(), y.end());
    has_grid_ = false;
    reshaped_.reset(); // the cached field describes the previous values
}
void pixels::update(std::span<const float> y)
{
    std::vector<double> yd(y.begin(), y.end());
    update(std::span<const double>(yd));
}
std::size_t pixels::point_count() const
{
    std::scoped_lock lk(m_);
    return has_grid_ ? loaded_.grid.size() : y_.size();
}

void pixels::set_axis_visible(plot_axis axis, bool visible)
{
    std::scoped_lock lk(m_);
    switch (axis) {
    case plot_axis::x: show_x_axis_ = visible; break;
    case plot_axis::y: show_y_axis_ = visible; break;
    case plot_axis::z: show_z_axis_ = visible; break;
    }
}
bool pixels::axis_visible(plot_axis axis) const
{
    std::scoped_lock lk(m_);
    switch (axis) {
    case plot_axis::x: return show_x_axis_;
    case plot_axis::y: return show_y_axis_;
    case plot_axis::z: return show_z_axis_;
    }
    return true;
}
void pixels::set_palette(colormap_t palette)
{
    std::scoped_lock lk(m_);
    colormap_ = palette;
}

void pixels::adopt_grid(grid_result g)
{
    loaded_   = std::move(g);
    has_grid_ = true;
    reshaped_.reset();
}

const grid_result& pixels::current_grid() const
{
    // A loaded figure holds its grid directly — it IS the product, so there is
    // nothing to cache. A live one reshapes its values, once per update().
    if (has_grid_) return loaded_;
    if (!reshaped_) reshaped_ = reshape_field(y_, width_, height_);
    return *reshaped_;
}

void pixels::paint_body(region rg)
{
    if (has_grid_ ? loaded_.grid.empty() : y_.empty()) {
        ImGui::TextDisabled("no data yet — call update()");
        return;
    }
    const grid_result& gr = current_grid();
    // Equal axes keep pixels square; color range spans the field's min…max.
    draw_colormap_grid(gr, gr.min_val, gr.max_count, ImVec2(rg.w, rg.h), colormap_,
                       /*equal_axes=*/true, "value###pxscale", show_x_axis_, show_y_axis_,
                       show_z_axis_);
}

void pixels::paint_controls()
{
    draw_theme_row();
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::BeginCombo("palette", colormap_name(colormap_))) {
        for (int i = 0; i < static_cast<int>(colormap_t::COUNT); ++i) {
            const auto cm  = static_cast<colormap_t>(i);
            const bool sel = (colormap_ == cm);
            if (ImGui::Selectable(colormap_name(cm), sel)) colormap_ = cm;
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("axes:");
    ImGui::SameLine();
    ImGui::Checkbox("x##axvis", &show_x_axis_);
    ImGui::SameLine();
    ImGui::Checkbox("y##ayvis", &show_y_axis_);
    ImGui::SameLine();
    ImGui::Checkbox("z##azvis", &show_z_axis_);
    ImGui::SameLine();
}

bool pixels::fill_snapshot(h5_save_data& out) const
{
    if (has_grid_ ? loaded_.grid.empty() : y_.empty()) return false;
    out.kind        = "pixels";
    out.colormap    = colormap_name(colormap_); // stored on disk as the palette name
    out.show_x_axis = show_x_axis_;
    out.show_y_axis = show_y_axis_;
    out.show_z_axis = show_z_axis_;
    // Copy, not move: gr is the live cache, not a temporary — moving its grid out
    // would leave the next frame painting an empty figure.
    const grid_result& gr = current_grid();
    out.grid              = gr.grid;
    out.nbx               = gr.nbx;
    out.nby               = gr.nby;
    out.x_min             = gr.x0;
    out.x_max             = gr.x1;
    out.y_min             = gr.y0;
    out.y_max             = gr.y1;
    return true;
}

} // namespace coast::plot
