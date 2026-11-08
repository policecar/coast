#include "gui_plot/histogram2d.h"

#include "gui_plot/plot_tools.h" // bin_points/draw_colormap_grid/draw_candles + ImPlot

#include <algorithm>
#include <cmath>

namespace coast::plot {

histogram2d::histogram2d(std::string title, int x_bins, int y_bins)
    : figure(std::move(title)),
      // <= 0 => ImPlotBin_Sturges: let ImPlot choose the bin count automatically.
      bins_x_(x_bins > 0 ? x_bins : ImPlotBin_Sturges),
      bins_y_(y_bins > 0 ? y_bins : ImPlotBin_Sturges)
{
}

histogram2d::histogram2d(std::string title, int x_bins, int y_bins, std::optional<range> x_domain,
                         std::optional<range> y_domain)
    : figure(std::move(title)), bins_x_(x_bins > 0 ? x_bins : ImPlotBin_Sturges),
      bins_y_(y_bins > 0 ? y_bins : ImPlotBin_Sturges),
      // An invalid domain would collapse that axis to zero width; auto-range it
      // instead of binning into nothing.
      x_domain_(x_domain && x_domain->valid() ? x_domain : std::nullopt),
      y_domain_(y_domain && y_domain->valid() ? y_domain : std::nullopt)
{
}

void histogram2d::update(std::span<const double> x, std::span<const double> y)
{
    std::scoped_lock lk(m_);
    x_.assign(x.begin(), x.end());
    y_.assign(y.begin(), y.end());
    has_grid_ = false; // switch a loaded (pre-binned) figure back to live data
    invalidate_cache();
}
void histogram2d::update(std::span<const float> x, std::span<const float> y)
{
    std::vector<double> xd(x.begin(), x.end());
    std::vector<double> yd(y.begin(), y.end());
    update(std::span<const double>(xd), std::span<const double>(yd));
}
std::size_t histogram2d::point_count() const
{
    std::scoped_lock lk(m_);
    return has_grid_ ? loaded_.grid.size() : y_.size();
}

void histogram2d::set_axis_visible(plot_axis axis, bool visible)
{
    std::scoped_lock lk(m_);
    switch (axis) {
    case plot_axis::x: show_x_axis_ = visible; break;
    case plot_axis::y: show_y_axis_ = visible; break;
    case plot_axis::z: show_z_axis_ = visible; break;
    }
}
bool histogram2d::axis_visible(plot_axis axis) const
{
    std::scoped_lock lk(m_);
    switch (axis) {
    case plot_axis::x: return show_x_axis_;
    case plot_axis::y: return show_y_axis_;
    case plot_axis::z: return show_z_axis_;
    }
    return true;
}
void histogram2d::set_palette(colormap_t palette)
{
    std::scoped_lock lk(m_);
    colormap_ = palette;
}
void histogram2d::set_log_density(bool on)
{
    std::scoped_lock lk(m_);
    log_density_ = on;
}
void histogram2d::set_candle(bool on)
{
    std::scoped_lock lk(m_);
    candle_ = on;
}

std::optional<range> histogram2d::x_domain() const
{
    std::scoped_lock lk(m_);
    return x_domain_;
}
std::optional<range> histogram2d::y_domain() const
{
    std::scoped_lock lk(m_);
    return y_domain_;
}
outer_ring histogram2d::outer() const
{
    std::scoped_lock lk(m_);
    // With no points there is nothing to bin (and bin_points has no empty case:
    // it takes the data's min/max), so short-circuit to an empty ring.
    if (!has_grid_ && std::min(x_.size(), y_.size()) == 0) return {};
    return current_grid().outer;
}

void histogram2d::invalidate_cache()
{
    cache_ = {};
}

void histogram2d::adopt_grid(grid_result g)
{
    loaded_   = std::move(g);
    has_grid_ = true;
    invalidate_cache();
}

const grid_result& histogram2d::current_grid() const
{
    // A loaded figure holds its grid directly — no cache entry needed, it IS the
    // product. A live one bins its raw points, once per update().
    if (has_grid_) return loaded_;
    if (!cache_.binned) cache_.binned = bin_points(x_, y_, bins_x_, bins_y_, x_domain_, y_domain_);
    return *cache_.binned;
}

const grid_result& histogram2d::log_grid() const
{
    if (!cache_.log) {
        grid_result lg = current_grid(); // same frame, transformed values
        // log₁₀(n+1): +1 keeps empty cells at 0 and avoids log(0).
        for (double& v : lg.grid) v = std::log10(v + 1.0);
        lg.max_count = std::log10(current_grid().max_count + 1.0);
        cache_.log   = std::move(lg);
    }
    return *cache_.log;
}

const std::vector<candle_box>& histogram2d::candle_boxes() const
{
    if (!cache_.boxes) cache_.boxes = compute_candles(current_grid());
    return *cache_.boxes;
}

void histogram2d::paint_body(region rg)
{
    if (has_grid_) {
        if (loaded_.grid.empty()) {
            ImGui::TextDisabled("empty histogram");
            return;
        }
    } else if (std::min(x_.size(), y_.size()) == 0) {
        ImGui::TextDisabled("histogram2d needs update(x, y)");
        return;
    }

    const ImVec2       reg = ImVec2(rg.w, rg.h);
    const grid_result& gr  = current_grid();
    last_outer_            = gr.outer; // for paint_controls, which must not re-bin
    if (candle_) {
        draw_candles(gr, candle_boxes(), reg, show_x_axis_, show_y_axis_);
    } else {
        // Scale reads in log₁₀(n+1) units when log is on (ImPlot's ColormapScale
        // maps linearly and has no log mode).
        const grid_result& shown = log_density_ ? log_grid() : gr;
        draw_colormap_grid(shown, 0.0, shown.max_count, reg, colormap_, /*equal_axes=*/false,
                           log_density_ ? "log10(n+1)###h2scale" : "count###h2scale",
                           show_x_axis_, show_y_axis_, show_z_axis_);
    }
}

void histogram2d::paint_controls()
{
    draw_theme_row();
    // A histogram2d can be shown as a density heatmap or as candle/box plots (both
    // derived from the same binned grid).
    int m = candle_ ? 1 : 0;
    ImGui::TextUnformatted("view:");
    ImGui::SameLine();
    ImGui::RadioButton("heatmap", &m, 0);
    ImGui::SameLine();
    ImGui::RadioButton("candle", &m, 1);
    ImGui::SameLine();
    candle_ = (m == 1);
    // Palette + log-density + colorbar apply to the colormapped (heatmap) view.
    const bool colormapped = !candle_;
    if (colormapped) {
        // Viridis is perceptually uniform; the qualitative palettes (Deep, Paired,
        // …) render a density map near-flat, so the sequential ones (Plasma, Hot,
        // Jet, …) are the sensible picks.
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
        ImGui::Checkbox("log density", &log_density_);
        ImGui::SameLine();
    }
    // Individually toggle the x/y axes and the z (colorbar). Hiding z frees its
    // gutter for the heatmap. z is only meaningful when a colorbar is drawn.
    ImGui::TextUnformatted("axes:");
    ImGui::SameLine();
    ImGui::Checkbox("x##axvis", &show_x_axis_);
    ImGui::SameLine();
    ImGui::Checkbox("y##ayvis", &show_y_axis_);
    ImGui::SameLine();
    if (has_colorbar()) {
        ImGui::Checkbox("z##azvis", &show_z_axis_);
        ImGui::SameLine();
    }
    // Surface the outliers: a fixed domain is the only way points can fall
    // outside the frame, and letting them vanish silently is exactly the failure
    // this accounting exists to prevent.
    if (const double n_out = last_outer_.total(); n_out > 0.0) {
        ImGui::TextDisabled("%g outside the fixed domain", n_out);
        ImGui::SameLine();
    }
}

bool histogram2d::fill_snapshot(h5_save_data& out) const
{
    if (!has_grid_ && std::min(x_.size(), y_.size()) == 0) return false;
    if (has_grid_ && loaded_.grid.empty()) return false;
    out.kind = "histogram2d";
    out.colormap    = colormap_name(colormap_); // stored on disk as the palette name
    out.log_density = log_density_;
    out.mode        = candle_ ? "candle" : "heatmap";
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
    // Persist the domains separately from the x_min/x_max frame above: for an
    // auto-ranged figure that frame is data-derived and must NOT come back as a
    // fixed domain, which would freeze one sample's extent into every reload.
    if (x_domain_) {
        out.has_x_range = true;
        out.x_range_lo  = x_domain_->lo;
        out.x_range_hi  = x_domain_->hi;
    }
    if (y_domain_) {
        out.has_y_range = true;
        out.y_range_lo  = y_domain_->lo;
        out.y_range_hi  = y_domain_->hi;
    }
    // Unlike the 1D histogram (which saves raw ys and recomputes), a saved
    // histogram2d holds only its binned grid — so the ring must travel with it.
    // An all-zero ring is left empty: outer_ring::empty() is defined as
    // total()==0, so "no ring" and "zero ring" are the same reading, and an
    // auto-ranged figure must not persist a ring it can never populate.
    if (gr.outer.total() > 0.0)
        out.outer.assign(&gr.outer.cell[0][0], &gr.outer.cell[0][0] + 9);
    return true;
}

} // namespace coast::plot
