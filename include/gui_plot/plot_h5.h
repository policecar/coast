#ifndef COAST_GUI_PLOT_PLOT_H5_H
#define COAST_GUI_PLOT_PLOT_H5_H

// Internal: HDF5 (de)serialization of the plain plot/canvas DTOs. All libhdf5
// calls run on the hdf5 executor worker thread. Split from plot_detail.h so only
// the persistence translation units pull in <hdf5>.

#include "gui_plot/plot_data.h"
#include "gui_plot/plot_detail.h" // canvas_save_data
#include "logging/hdf5_utils.h"   // hid_t + the hdf5:: wrappers

#include <string>
#include <vector>

namespace coast::plot {

// Write/read a single plot's attrs + datasets into/from an already-open group.
// read_plot_from does not throw: a missing or malformed (a shape or type the
// writer never produces) plot is logged with `ctx` and returns false.
void write_plot_into(hid_t gid, const h5_save_data& d);
bool read_plot_from(hid_t gid, h5_save_data& out, const std::string& ctx);
// Write a `thumbnail` PNG dataset into an open group, if present.
void write_thumbnail_into(hid_t gid, const std::vector<unsigned char>& png, int w, int h);

// Whole-file variants (one plot / one canvas per file).
void write_plot_h5(const h5_save_data& d);
bool read_plot_h5(const std::string& path, h5_save_data& out);
void write_canvas_h5(const canvas_save_data& d);
bool read_canvas_h5(const std::string& path, canvas_save_data& out);

} // namespace coast::plot

#endif // COAST_GUI_PLOT_PLOT_H5_H
