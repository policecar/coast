#ifndef COAST_CONSOLE_PLOT_H
#define COAST_CONSOLE_PLOT_H

// Console plot bridge (design/console_api_design.md) — umbrella header.
//
// One flat hierarchy of plain, named plot objects. Each concrete kind lives in
// its own header/source pair; this umbrella just pulls them together and declares
// the free functions that drive the render loop and load saved plots. A macro can
// include this to get everything, or include only the kind header(s) it uses.
//
//   figure.h         the base + the cross-thread painter seam (ImGui-free)
//   histogram.h      1D histogram          line.h          xy / line
//   histogram2d.h    2D density + candle   pixels.h        colormapped field
//   image.h          picture channel       canvas.h        grid of figures
//
// There is no creation factory and no type erasure: construct a concrete kind
// with std::make_shared and reach its typed config directly, e.g.
//   auto h = std::make_shared<coast::plot::histogram2d>("act", 64, 64);
//   h->show();
//   h->update(xs, ys);
//   h->set_axis_visible(coast::plot::plot_axis::x, false);
//
// To add a new figure kind: add a <kind>.{h,cpp} pair deriving figure, add a case
// to make_figure_from_h5 (plot_load.cpp) if it is loadable, and list both files
// in CMake. Nothing in this umbrella or the base needs to change beyond one
// include line below.
//
// This header (and every kind header) is ImGui-free so a clang-repl interpreter
// can parse it cheaply — all ImPlot/GL lives in the .cpp painters.

#include "gui_plot/canvas.h"
#include "gui_plot/figure.h"
#include "gui_plot/histogram.h"
#include "gui_plot/histogram2d.h"
#include "gui_plot/image.h"
#include "gui_plot/line.h"
#include "gui_plot/pixels.h"

#include <memory>
#include <string>

namespace hdf5 {
class executor; // HDF5 writes are funnelled through its single worker thread
}

namespace coast::plot {

// -- load saved plots ------------------------------------------------------

// Reconstruct a figure (with data + config) from a loaded HDF5 file. Returns
// nullptr on failure (missing executor, bad file). The reconstructed figure is
// shown (registered). Safe from the console thread — the HDF5 read is funnelled
// through the executor set via set_executor().
std::shared_ptr<figure> load(std::string path);
// Load a canvas previously written by canvas::save_h5, reconstructing every
// stored cell, and show it. Returns nullptr on failure.
std::shared_ptr<canvas> load_canvas(std::string path);

// Wire the HDF5 executor used by load()/save. Call once at startup before the
// console can run. Stores the address; the executor must outlive all plot
// activity.
void set_executor(hdf5::executor& exec);

// -- GUI thread only --
void                      render_all();     // draw every visible, non-hosted figure; reap dead ones
void flush_captures(hdf5::executor& exec);  // service queued save_png()/save_h5() requests;
                                            // call OUTSIDE an active ImGui frame (state phase),
                                            // as it runs its own off-screen render pass; HDF5
                                            // writes go via `exec`
void                      free_resources(); // release GL textures + clear the registry; call
                                            // before the GL context is destroyed
[[nodiscard]] std::size_t plot_count();     // number of registered (live) figures

} // namespace coast::plot

#endif // COAST_CONSOLE_PLOT_H
