#ifndef COAST_GUI_PLOT_PLOT_DETAIL_H
#define COAST_GUI_PLOT_PLOT_DETAIL_H

// Module-internal glue shared across the console-plot .cpp files. NOT part of the
// macro-facing surface (console_plot.h) — only the framework/kind translation
// units include it. Holds the canvas save DTO, the reconstruct-from-HDF5 loader,
// and a couple of stateless helpers. The stateful services (registry, task hook,
// save queues) live on plot_manager (plot_manager.h), not here.

#include "gui_plot/figure.h"
#include "gui_plot/plot_data.h"

#include <memory>
#include <string>
#include <vector>

struct ImPlotStyle; // scale_implot_style takes it by reference; no implot.h needed here

namespace coast::plot {

// Plain-data hand-off for the canvas HDF5 writer (built on the GUI thread,
// written on the worker). One entry per occupied, saveable cell.
struct canvas_save_data {
    struct cell {
        int          row, col;
        h5_save_data data;
    };
    std::string                path;
    std::string                title;
    int                        rows = 0, cols = 0;
    std::vector<cell>          cells;
    std::vector<unsigned char> thumb_png;
    int                        thumb_w = 0, thumb_h = 0;
};

// The one name→type map: reconstruct a fully-configured figure from a loaded DTO
// (nullptr on an unknown kind). Defined in plot_load.cpp; a friend of each kind
// so it can restore private state. Adding a kind adds one case there.
std::shared_ptr<figure> make_figure_from_h5(const h5_save_data& d);

namespace detail {

// Stateless export-sizing helpers (defined in plot_capture.cpp).
float       export_scale(int w, int h, float rw, float rh); // uniform DPI scale for an enlarged export
std::string default_png_name(const std::string& title);     // export_dir / (slug + ".png")
std::string default_h5_name(const std::string& title);      // export_dir / (slug + ".h5")

// Scale the pixel-size fields of an ImPlotStyle by f. The ImPlot half of a
// uniform zoom; the ImGui half is ImGuiStyle::ScaleAllSizes and the text half is
// ImGuiStyle::FontScaleMain (ScaleAllSizes deliberately touches neither font
// field). Shared by the off-screen exporter and the canvas's zoom.
// NOT idempotent — always scale a pristine copy of the style, never compound.
void scale_implot_style(::ImPlotStyle& s, float f);

} // namespace detail
} // namespace coast::plot

#endif // COAST_GUI_PLOT_PLOT_DETAIL_H
