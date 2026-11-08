#ifndef COAST_GUI_PLOT_PIXELS_H
#define COAST_GUI_PLOT_PIXELS_H

#include "gui_plot/figure.h"     // figure + region + plot_axis + grid_result (via plot_data.h)
#include "gui_vis/core/colormap.h" // colormap_t (framework palette enum)

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace coast::plot {

// Pixel/matrix view: interprets update(y) as a width × height row-major field
// (row 0 = top) and draws it as a colormapped image, one cell per value, with a
// selectable palette and per-axis visibility. Pixels are kept square.
class pixels final : public figure {
public:
    pixels(std::string title, int width, int height);

    void update(std::span<const double> y);
    void update(std::span<const float> y);
    [[nodiscard]] std::size_t point_count() const override;

    void               set_axis_visible(plot_axis axis, bool visible);
    [[nodiscard]] bool axis_visible(plot_axis axis) const;
    void               set_palette(colormap_t palette);

protected:
    void paint_body(region r) override;
    void paint_controls() override;
    bool fill_snapshot(h5_save_data& out) const override;
    [[nodiscard]] bool is_saveable() const override { return true; }

private:
    friend std::shared_ptr<figure> make_figure_from_h5(const h5_save_data&);
    // The loaded grid, or the live values reshaped. Reference into reshaped_
    // below: valid until the next update()/adopt_grid().
    [[nodiscard]] const grid_result& current_grid() const;
    // Become a loaded (pre-gridded) figure holding `g`. Used by the h5 loader —
    // routed through here rather than assigning loaded_ so it cannot forget to
    // drop the cache.
    void adopt_grid(grid_result g);

    std::vector<double> y_;
    int                 width_, height_;
    colormap_t          colormap_ = colormap_t::viridis;
    bool                show_x_axis_ = true;
    bool                show_y_axis_ = true;
    bool                show_z_axis_ = true;
    bool                has_grid_    = false;
    grid_result         loaded_;
    // The reshaped field: a pure function of y_, which changes only on update().
    // Computed on first use and dropped there, so the GUI does not rebuild the
    // whole field every frame. Mutable + filled from the const painters, which
    // hold m_. Unused when has_grid_ (loaded_ is already the product).
    mutable std::optional<grid_result> reshaped_;
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_PIXELS_H
