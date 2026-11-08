#ifndef COAST_GUI_PLOT_PLOT_DATA_H
#define COAST_GUI_PLOT_PLOT_DATA_H

// Plain, ImGui-free data types shared across the console-plot module: the
// binned/reshaped render field (grid_result) and the HDF5 serialization DTO
// (h5_save_data). No behaviour — just data — so both the parse-light figure
// headers and the plot_tools render helpers can depend on them cheaply.

#include <cstdint>
#include <string>
#include <vector>

namespace coast::plot {

// A fixed, closed value domain [lo, hi] for one histogram axis. Supplying one
// suppresses the min/max auto-ranging, so histograms of different samples share
// bin edges and are directly comparable.
//
// Plot-local and double on purpose: the figures store and bin in double, and
// tools/metric.h's value_range is float — sharing it would round the bin edges.
// Two range types in the tree is the accepted cost; see the fixed-range PR.
struct range {
    double lo = 0.0;
    double hi = 0.0;

    // A domain is usable only if it spans something. {0,0} (the default) reads
    // as "no domain" — it is also ImPlot's own auto-range sentinel.
    [[nodiscard]] constexpr bool valid() const { return hi > lo; }
    [[nodiscard]] constexpr bool contains(double v) const { return v >= lo && v <= hi; }

    // Exact equality — a domain is a pair of endpoints the caller stated, not a
    // computed quantity, so there is nothing to tolerance. Lets an optional<range>
    // be compared against a range directly (`h->domain() == range{0, 1}`), which
    // checks engagement and value in one step.
    [[nodiscard]] friend constexpr bool operator==(const range&, const range&) = default;
};

// Where one coordinate fell relative to a fixed domain. Used to address
// outer_ring; `inside` is the ordinary in-grid case.
enum class zone : std::uint8_t { under = 0, inside = 1, over = 2 };

// Out-of-range accounting for a fixed-domain histogram2d, in ROOT's TH2 spirit:
// every out-of-range point lands in exactly ONE cell, so nothing is silently
// dropped and nothing is clamped into an edge bin (which would fabricate a spike
// at a value that was never observed). sum(grid) + outer.total() == point count.
//
// Unlike ROOT, this does NOT tax the inner grid's indexing: there is no bin 0 =
// underflow convention to remember: the grid keeps its natural 0..nbx-1 /
// 0..nby-1 addressing and the ring sits beside it.
//
//                under-x     in-x      over-x
//      over-y  [ 0 ][ 2 ]  [ 0 ][ 1 ]  [ 0 ][ 2 ]      cell[zone_y][zone_x]
//       in-y   [ 1 ][ 0 ]  <   GRID >  [ 1 ][ 2 ]      [inside][inside] is
//      under-y [ 2 ][ 0 ]  [ 2 ][ 1 ]  [ 2 ][ 2 ]      always 0 — that's the grid
struct outer_ring {
    double cell[3][3] = {};

    [[nodiscard]] constexpr double& at(zone zy, zone zx)
    {
        return cell[static_cast<int>(zy)][static_cast<int>(zx)];
    }
    [[nodiscard]] constexpr double at(zone zy, zone zx) const
    {
        return cell[static_cast<int>(zy)][static_cast<int>(zx)];
    }
    // Everything outside the grid. Skips [inside][inside], which is the grid.
    [[nodiscard]] constexpr double total() const
    {
        double s = 0.0;
        for (int zy = 0; zy < 3; ++zy) {
            for (int zx = 0; zx < 3; ++zx) {
                if (zy != 1 || zx != 1) s += cell[zy][zx];
            }
        }
        return s;
    }
    [[nodiscard]] constexpr bool empty() const { return total() == 0.0; }
};

// A binned/reshaped 2D field ready for colormapped or candle rendering. Grid is
// row-major with row 0 = top (max-y); [min_val, max_count] is the color range.
// Produced by the plot_tools helpers (bin_points / reshape_field).
struct grid_result {
    std::vector<double> grid;
    int                 nbx = 0, nby = 0;
    double              x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    double              min_val = 0;   // color-range floor (0 for count grids)
    double              max_count = 0; // color-range ceiling
    // Points outside a fixed domain. Always empty when auto-ranged: the frame is
    // then derived from the data, so nothing can fall outside it.
    outer_ring          outer;
};

// One box-and-whisker in plot units, derived from a single grid column's binned
// y-distribution: x is the column's center, [q1, q3] the box, med the median
// line, [lo, hi] the whisker ends. Produced by plot_tools' compute_candles.
//
// Plain data (and here rather than in plot_tools.h) so the ImGui-free
// histogram2d can hold a cached vector of them.
struct candle_box {
    double x   = 0;
    double q1  = 0;
    double med = 0;
    double q3  = 0;
    double lo  = 0;
    double hi  = 0;
};

// Plain-data snapshot of a figure for the HDF5 writer. Filled on the GUI thread
// (under the figure lock) by figure::fill_snapshot, then handed by value to the
// executor worker — it holds NO GL/ImGui state, so the worker can write it off
// the GUI thread.
struct h5_save_data {
    std::string path;
    std::string kind;     // "histogram" | "line" | "histogram2d" | "pixels"
    std::string title;
    std::string theme = "dark"; // structural-color theme name (all kinds)
    std::string colormap;       // histogram2d/pixels palette name
    // histogram2d: binned grid (raw counts, row-major, row 0 = top/max-y) + frame
    int                 nbx = 0, nby = 0;
    double              x_min = 0, x_max = 0, y_min = 0, y_max = 0;
    bool                log_density = false;
    std::string         mode;         // histogram2d view: "heatmap" | "candle"
    // Per-axis visibility. z = the colorbar/scale; hidden means it is not drawn
    // and its gutter is reclaimed by the plot.
    bool                show_x_axis = true, show_y_axis = true, show_z_axis = true;
    std::vector<double> grid;
    // Fixed axis domains (histogram uses x_* only; absent => auto-ranged). Kept
    // apart from x_min/x_max above: those are the frame a histogram2d grid was
    // binned with, which for an auto-ranged figure is derived from the data and
    // must NOT resurrect as a fixed domain on load.
    bool                has_x_range = false, has_y_range = false;
    double              x_range_lo = 0, x_range_hi = 0;
    double              y_range_lo = 0, y_range_hi = 0;
    // histogram2d outer-ring counts, row-major [zone_y][zone_x] (9 values, or
    // empty). Persisted because a saved histogram2d holds its binned grid rather
    // than its raw points, so the ring cannot be recomputed on load. A 1D
    // histogram saves raw ys, so its under/overflow ARE recomputed and need no
    // field here.
    std::vector<double> outer;
    // histogram (1D) / line: raw series (+ bin count for 1D histograms)
    int                 bins = 0;
    std::vector<double> xs, ys;
    // embedded preview
    std::vector<unsigned char> thumb_png;
    int                        thumb_w = 0, thumb_h = 0;
};

} // namespace coast::plot

#endif // COAST_GUI_PLOT_PLOT_DATA_H
