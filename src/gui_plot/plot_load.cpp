// The ONE place that maps a saved "kind" string back to a concrete figure type.
// Adding a loadable figure kind means adding one case here (and its .{h,cpp}
// pair + CMake entry) — nothing else in the module needs to change.
//
// Runs on the calling (console) thread — touches no HDF5/GL, only plain data; the
// stored colormap name maps straight to the palette enum. Friend of each kind so
// it can restore private state.

#include "gui_plot/plot_detail.h" // make_figure_from_h5 declaration

#include "gui_vis/core/colormap.h" // colormap_from_name

#include "gui_plot/histogram.h"
#include "gui_plot/histogram2d.h"
#include "gui_plot/line.h"
#include "gui_plot/pixels.h"

#include "diag/diag.h"

#include <algorithm>
#include <memory>

namespace coast::plot {

namespace {

// The saved domains, as the figure constructors take them. Absent => nullopt =>
// that axis auto-ranges, which is what a figure saved without a domain was.
std::optional<range> saved_x_domain(const h5_save_data& d)
{
    if (!d.has_x_range) return std::nullopt;
    return range{d.x_range_lo, d.x_range_hi};
}
std::optional<range> saved_y_domain(const h5_save_data& d)
{
    if (!d.has_y_range) return std::nullopt;
    return range{d.y_range_lo, d.y_range_hi};
}

// The persisted 9-cell ring, or an empty one if the file predates it / the
// figure was auto-ranged.
outer_ring saved_outer(const h5_save_data& d)
{
    outer_ring r;
    if (d.outer.size() != 9) return r;
    for (int zy = 0; zy < 3; ++zy) {
        for (int zx = 0; zx < 3; ++zx) r.cell[zy][zx] = d.outer[static_cast<std::size_t>(zy) * 3 + zx];
    }
    return r;
}

} // namespace

std::shared_ptr<figure> make_figure_from_h5(const h5_save_data& d)
{
    if (d.kind == "histogram2d") {
        auto k            = std::make_shared<histogram2d>(d.title, d.nbx, d.nby, saved_x_domain(d),
                                                          saved_y_domain(d));
        k->theme_         = d.theme;
        k->candle_        = (d.mode == "candle");
        k->colormap_      = colormap_from_name(d.colormap);
        k->log_density_   = d.log_density;
        k->show_x_axis_   = d.show_x_axis;
        k->show_y_axis_   = d.show_y_axis;
        k->show_z_axis_   = d.show_z_axis;
        double max_count  = 0.0;
        for (double v : d.grid)
            if (v > max_count) max_count = v;
        k->adopt_grid(grid_result{d.grid,  d.nbx,   d.nby,           d.x_min,   d.x_max,
                                  d.y_min, d.y_max, /*min_val=*/0.0, max_count, saved_outer(d)});
        return k;
    }
    if (d.kind == "pixels") {
        auto k            = std::make_shared<pixels>(d.title, d.nbx, d.nby);
        k->theme_         = d.theme;
        k->colormap_      = colormap_from_name(d.colormap);
        k->show_x_axis_   = d.show_x_axis;
        k->show_y_axis_   = d.show_y_axis;
        k->show_z_axis_   = d.show_z_axis;
        double mn = 0.0, mx = 0.0;
        if (!d.grid.empty()) {
            mn = mx = d.grid.front();
            for (double v : d.grid) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
        // No ring: pixels is a reshaped field, not a binned one — nothing can
        // fall outside it.
        k->adopt_grid(grid_result{d.grid,  d.nbx,   d.nby, d.x_min, d.x_max,
                                  d.y_min, d.y_max, mn,    mx,      outer_ring{}});
        return k;
    }
    if (d.kind == "histogram") {
        const auto dom = saved_x_domain(d);
        auto       k   = dom ? std::make_shared<histogram>(d.title, d.bins > 0 ? d.bins : 1, *dom)
                             : std::make_shared<histogram>(d.title, d.bins > 0 ? d.bins : 1);
        k->theme_      = d.theme;
        k->y_          = d.ys;
        // y_ is assigned directly rather than through update(), so the tallies
        // must be brought up to date by hand.
        k->recount_outliers();
        return k;
    }
    if (d.kind == "line") {
        auto k    = std::make_shared<line>(d.title);
        k->theme_ = d.theme;
        k->x_     = d.xs;
        k->y_     = d.ys;
        return k;
    }
    COAST_ERROR("plot", "load: unknown plot kind '{}'", d.kind);
    return nullptr;
}

} // namespace coast::plot
