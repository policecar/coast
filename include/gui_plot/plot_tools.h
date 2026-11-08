#ifndef COAST_PLOT_TOOLS_H
#define COAST_PLOT_TOOLS_H

// Reusable, stateless rendering helpers for the console-plot bridge, plus the
// plot-theme layer. GUI-internal: this header pulls in ImGui/ImPlot and is only
// ever included by the .cpp side of gui_plot (never by console_plot.h, which is
// kept ImGui-free so a clang-repl interpreter can parse it cheaply).
//
// All functions here are free (take their inputs as parameters, hold no plot
// state) and run on the GUI thread.

#include "gui_plot/plot_data.h"    // grid_result (plain, ImGui-free) lives here
#include "gui_vis/core/colormap.h" // colormap_t (framework palette enum)

#include "imgui.h"
#include "implot.h"
#include "implot_internal.h" // BeginItem/FitPoint/GetPlotDrawList for the candle renderer

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace coast::plot {

// grid_result (the binned/reshaped 2D field) is declared in console_plot.h so it
// can be a data member of the ImGui-free public figure classes.

// Bin (x, y) points into a raw-count grid (row 0 = top/max-y). bins_x/bins_y <= 0
// auto-pick via Sturges' rule (matching ImPlot). Caller must ensure at least one
// point. min_val is 0 (count grids). See draw_body's histogram2d path.
//
// A supplied x_domain/y_domain fixes that axis's frame instead of deriving it
// from the data's min/max. Points outside a fixed domain are tallied into
// result.outer (ROOT-style) rather than clamped into the edge bins — so with a
// fixed domain, sum(grid) may be < point count and the remainder is in .outer.
// With no domain (or an invalid one) the axis auto-ranges exactly as before and
// .outer stays empty.
[[nodiscard]] grid_result bin_points(std::span<const double> x, std::span<const double> y,
                                     int bins_x, int bins_y,
                                     std::optional<range> x_domain = std::nullopt,
                                     std::optional<range> y_domain = std::nullopt);

// Reshape raw values into a width x height row-major field with its value range.
// Row 0 = the first values (drawn at top). Extra values ignored; missing stay 0.
[[nodiscard]] grid_result reshape_field(std::span<const double> values, int width, int height);

// Draw a colormapped grid (heatmap) + optional colorbar into `region`, honoring
// the per-axis visibility toggles. `grid` values are already in display units
// (any log transform is applied by the caller). Shared by histogram2d + pixels.
void draw_colormap_grid(const grid_result& gr, double scale_min, double scale_max, ImVec2 region,
                        colormap_t colormap, bool equal_axes, const char* scale_label,
                        bool show_x_axis, bool show_y_axis, bool show_z_axis);

// ROOT-style candle stats: one box-and-whisker per populated x-column, its
// quartiles/median/whiskers computed from that column's binned y-distribution
// (approximate at the grid's y-resolution). Empty columns yield no box, so the
// result may be shorter than gr.nbx (or empty).
//
// Pure computation, split out from the draw so a figure can cache it: the stats
// depend only on the grid, which changes only when new data arrives.
[[nodiscard]] std::vector<candle_box> compute_candles(const grid_result& gr);

// Draw candle/box plots from stats compute_candles produced for `gr` — which
// also supplies the frame and bin width the boxes are drawn at, so it must be
// the same grid the stats came from.
void draw_candles(const grid_result& gr, std::span<const candle_box> boxes, ImVec2 region,
                  bool show_x_axis, bool show_y_axis);

// ---------------------------------------------------------------- plot themes
// A theme overrides ImPlot's structural colors so the same figures can render
// dark for the screen or light for a publication. Plots refer to a theme by
// name; the registry is GUI-thread-only (built lazily). See plot_tools.cpp.

// Names of the currently-registered themes (ensures the registry is built).
[[nodiscard]] std::vector<std::string> theme_names();

// RAII: push a theme's structural colors for the enclosing BeginPlot/EndPlot,
// restore on scope exit. Unknown name -> no-op. GUI thread.
struct theme_scope {
    int pushed = 0;
    explicit theme_scope(const std::string& name);
    ~theme_scope();
    theme_scope(const theme_scope&)            = delete;
    theme_scope& operator=(const theme_scope&) = delete;
};

// Override the file plot themes persist to / load from (default
// "coast_plot_themes.json" in the CWD). Set once at startup from the app config
// (state.plot_themes) BEFORE any theme is accessed; a non-empty value wins, an
// empty one keeps the default. GUI thread.
void set_theme_file(std::string path);

// Override the default directory the save-PNG / save-HDF5 dialogs seed into
// (default "." = CWD). Set once at startup from the app config (output.plot_dir);
// a non-empty value wins. The dialogs still let the user edit the full path.
void set_export_dir(std::string dir);

// Request the live Plot Theme Editor be shown (sets an internal GUI-thread flag).
void request_theme_editor();

// Draw the Plot Theme Editor if it was requested; closing its window clears the
// flag. No-op when not requested. GUI thread. Call once per frame in render_all.
void draw_theme_editor();

} // namespace coast::plot

#endif // COAST_PLOT_TOOLS_H
