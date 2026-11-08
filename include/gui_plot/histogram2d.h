#ifndef COAST_GUI_PLOT_HISTOGRAM2D_H
#define COAST_GUI_PLOT_HISTOGRAM2D_H

#include "gui_plot/figure.h"     // figure + region + plot_axis + grid_result (via plot_data.h)
#include "gui_vis/core/colormap.h" // colormap_t (framework palette enum)

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace coast::plot {

// 2D histogram (density heatmap) with a selectable palette, a log-density toggle,
// per-axis visibility (incl. the z colorbar), and a heatmap<->candle view. Both
// spans of update(x, y) are binned into an x_bins × y_bins grid; pass <= 0 for
// either count to auto-pick via Sturges' rule.
//
// By default each axis's frame is the data's own min/max, so two figures of
// different samples are NOT comparable. Construct with a domain to fix an axis
// (see the domain overload).
class histogram2d final : public figure {
public:
    explicit histogram2d(std::string title, int x_bins = 0, int y_bins = 0);

    // Fixed-domain form. Each axis is independently fixed or auto: pass nullopt
    // to leave that axis auto-ranged. A fixed axis bins over [lo, hi] regardless
    // of the data, so figures of different samples share bin edges.
    //
    // Points outside a fixed domain are tallied in outer() — one ring cell each,
    // ROOT-style — instead of being dropped or clamped into the edge bins. An
    // invalid domain (hi <= lo) falls back to auto for that axis. Domains are
    // fixed at construction and immutable.
    histogram2d(std::string title, int x_bins, int y_bins, std::optional<range> x_domain,
                std::optional<range> y_domain);

    void update(std::span<const double> x, std::span<const double> y);
    void update(std::span<const float> x, std::span<const float> y);
    [[nodiscard]] std::size_t point_count() const override;

    // The fixed domains, or nullopt where auto-ranged.
    [[nodiscard]] std::optional<range> x_domain() const;
    [[nodiscard]] std::optional<range> y_domain() const;
    // Points outside the fixed domains, by zone. Empty when fully auto-ranged.
    // Reads the cached binning (O(n) only on the first call after an update()).
    // For a live figure, sum(grid) + outer().total() == point_count(). Not so for
    // a loaded one: it has no points left, and point_count() then reports its
    // grid's cell count (see the override).
    [[nodiscard]] outer_ring outer() const;

    // -- typed config (the reason this hierarchy exists: reachable directly) --
    void               set_axis_visible(plot_axis axis, bool visible);
    [[nodiscard]] bool axis_visible(plot_axis axis) const;
    void               set_palette(colormap_t palette); // e.g. colormap_t::plasma
    void               set_log_density(bool on);
    // Show as candle/box plots (one box-and-whisker per x-column, from that
    // column's binned y-distribution) instead of a density heatmap. Same data.
    void               set_candle(bool on);

protected:
    void paint_body(region r) override;
    void paint_controls() override;
    bool fill_snapshot(h5_save_data& out) const override;
    [[nodiscard]] bool is_saveable() const override { return true; }

private:
    friend std::shared_ptr<figure> make_figure_from_h5(const h5_save_data&);

    // -- render products, all cached (see cache_ below) --
    // The loaded grid, or the live points binned. Reference into the cache: valid
    // until the next update()/adopt_grid().
    [[nodiscard]] const grid_result& current_grid() const;
    // current_grid() in log10(n+1) units — the heatmap's log-density view.
    [[nodiscard]] const grid_result& log_grid() const;
    // Per-column box stats for the candle view, from current_grid().
    [[nodiscard]] const std::vector<candle_box>& candle_boxes() const;
    // Drop every cached product. The ONE place that must be called whenever the
    // data behind them changes, or a frame would render the previous sample.
    void invalidate_cache();
    // Become a loaded (pre-binned) figure holding `g`. Used by the h5 loader —
    // routed through here rather than assigning loaded_ so it cannot forget the
    // invalidation.
    void adopt_grid(grid_result g);

    [[nodiscard]] bool has_colorbar() const { return !candle_; } // z meaningful?

    std::vector<double>  x_, y_;
    int                  bins_x_, bins_y_;
    std::optional<range> x_domain_, y_domain_; // nullopt => auto-range that axis
    // Ring from the last paint, so paint_controls can report outliers without
    // re-binning. Painted-thread only; outer() recomputes rather than read this.
    outer_ring           last_outer_;
    colormap_t           colormap_ = colormap_t::viridis;
    bool                log_density_ = false;
    bool                candle_      = false;
    bool                show_x_axis_ = true;
    bool                show_y_axis_ = true;
    bool                show_z_axis_ = true;
    bool                has_grid_    = false; // a loaded figure holds its grid
    grid_result         loaded_;

    // Everything the painters need is a pure function of the data (the raw points
    // or the loaded grid) — none of it depends on the frame, the region, or the
    // view toggles. Without this the GUI re-binned every point, re-took the log of
    // every cell, and recomputed every column's quartiles at 60 Hz, for data that
    // changes only when the console calls update().
    //
    // Each product is computed on first use and kept until invalidate_cache().
    // Mutable + filled from the const painters/accessors, all of which hold m_.
    struct cache {
        std::optional<grid_result>             binned; // live points binned (unused when loaded_)
        std::optional<grid_result>             log;    // current_grid() in log10(n+1) units
        std::optional<std::vector<candle_box>> boxes;  // per-column candle stats
    };
    mutable cache cache_;
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_HISTOGRAM2D_H
