#include "gui_plot/plot_h5.h"

// Diagnostics in the gui_plot module all use the single "plot" category on
// purpose — the old ad-hoc [console_plot] / [plot_theme] prefixes are
// deliberately collapsed into one category rather than kept as sub-categories.
// The message text still distinguishes save/load/theme paths where it matters.
#include "diag/diag.h"

#include <cstdint>
#include <optional>
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

// HDF5 (de)serialization of the plain plot/canvas DTOs. All libhdf5 calls run on
// the hdf5 executor worker thread (the H5T_NATIVE_* macros used by write_attr
// included). Pure plain-data in/out — no dependency on the figure classes.

namespace coast::plot {

namespace {

// Fixed axis domains, written only when present. Absence is the signal for
// "auto-ranged" — the reader's has_*_range defaults to false — so a file written
// before fixed domains existed loads as auto-ranged, which is what it was.
// Returns a negative herr_t if any attribute write failed (folded by the caller,
// which warns once for the whole group), 0 on success.
herr_t write_fixed_domains(hid_t gid, const h5_save_data& d)
{
    using namespace hdf5;
    herr_t rc = 0;
    if (d.has_x_range) {
        rc |= write_attr(gid, "x_range_lo", d.x_range_lo);
        rc |= write_attr(gid, "x_range_hi", d.x_range_hi);
    }
    if (d.has_y_range) {
        rc |= write_attr(gid, "y_range_lo", d.y_range_lo);
        rc |= write_attr(gid, "y_range_hi", d.y_range_hi);
    }
    return rc;
}

// Mirror of the above: a domain exists iff both of its attrs are present.
void read_fixed_domains(hid_t gid, h5_save_data& out)
{
    using namespace hdf5;
    // Absence is the ordinary "auto-ranged" signal: nullopt, no error output.
    const auto xlo = read_attr_double(gid, "x_range_lo");
    const auto xhi = read_attr_double(gid, "x_range_hi");
    if (xlo && xhi) {
        out.has_x_range = true;
        out.x_range_lo  = *xlo;
        out.x_range_hi  = *xhi;
    }
    const auto ylo = read_attr_double(gid, "y_range_lo");
    const auto yhi = read_attr_double(gid, "y_range_hi");
    if (ylo && yhi) {
        out.has_y_range = true;
        out.y_range_lo  = *ylo;
        out.y_range_hi  = *yhi;
    }
}

} // namespace

void write_plot_into(hid_t gid, const h5_save_data& d)
{
    using namespace hdf5;
    herr_t rc = 0;
    rc |= write_attr(gid, "coast_plot_version", static_cast<std::int64_t>(1));
    rc |= write_attr(gid, "kind", d.kind);
    rc |= write_attr(gid, "title", d.title);
    rc |= write_attr(gid, "theme", d.theme);

    // A failed create leaves ds invalid, so its write fails too: one check.
    auto write_1d = [&](const char* name, const std::vector<double>& v) -> herr_t {
        const hsize_t dims[1] = {v.size()};
        dataspace     sp      = dataspace::simple(dims);
        dataset       ds      = dataset::create(gid, name, H5T_NATIVE_DOUBLE, sp.id());
        return ds.write(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, v.data());
    };

    if (d.kind == "histogram2d" || d.kind == "pixels") {
        rc |= write_attr(gid, "nbx", static_cast<std::int64_t>(d.nbx));
        rc |= write_attr(gid, "nby", static_cast<std::int64_t>(d.nby));
        rc |= write_attr(gid, "x_min", d.x_min);
        rc |= write_attr(gid, "x_max", d.x_max);
        rc |= write_attr(gid, "y_min", d.y_min);
        rc |= write_attr(gid, "y_max", d.y_max);
        rc |= write_attr(gid, "log_density", static_cast<std::int64_t>(d.log_density ? 1 : 0));
        rc |= write_attr(gid, "show_x_axis", static_cast<std::int64_t>(d.show_x_axis ? 1 : 0));
        rc |= write_attr(gid, "show_y_axis", static_cast<std::int64_t>(d.show_y_axis ? 1 : 0));
        rc |= write_attr(gid, "show_z_axis", static_cast<std::int64_t>(d.show_z_axis ? 1 : 0));
        if (d.kind == "histogram2d") {
            rc |= write_attr(gid, "mode", d.mode);
            rc |= write_fixed_domains(gid, d);
        }
        rc |= write_attr(gid, "colormap", d.colormap);
        // The outer ring travels with the grid: a saved histogram2d has no raw
        // points left to recompute it from. Producers leave it empty when there
        // is nothing outside the grid (see histogram2d::fill_snapshot).
        if (!d.outer.empty()) rc |= write_1d("outer", d.outer);
        const hsize_t dims[2] = {static_cast<hsize_t>(d.nby), static_cast<hsize_t>(d.nbx)};
        dataspace     sp      = dataspace::simple(dims);
        dataset       ds      = dataset::create(gid, "grid", H5T_NATIVE_DOUBLE, sp.id());
        rc |= ds.write(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, d.grid.data());
    } else {
        if (d.kind == "histogram") {
            rc |= write_attr(gid, "bins", static_cast<std::int64_t>(d.bins));
            rc |= write_fixed_domains(gid, d);
        }
        if (!d.xs.empty()) rc |= write_1d("x", d.xs);
        rc |= write_1d("y", d.ys);
    }
    if (rc < 0) COAST_WARN("plot", "one or more plot attributes or datasets failed to write to '{}'", d.title);
}

void write_thumbnail_into(hid_t gid, const std::vector<unsigned char>& png, int w, int h)
{
    using namespace hdf5;
    if (png.empty()) return;
    const hsize_t dims[1] = {png.size()};
    dataspace     sp      = dataspace::simple(dims);
    dataset       ds      = dataset::create(gid, "thumbnail", H5T_NATIVE_UINT8, sp.id());
    herr_t rc = ds.write(H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, png.data());
    rc |= write_attr(ds.id(), "width", static_cast<std::int64_t>(w));
    rc |= write_attr(ds.id(), "height", static_cast<std::int64_t>(h));
    rc |= write_attr(ds.id(), "format", std::string_view("png"));
    if (rc < 0) COAST_WARN("plot", "thumbnail or its attributes failed to write");
}

void write_plot_h5(const h5_save_data& d)
{
    using namespace hdf5;
    file f = file::create(d.path);
    if (!f) {
        COAST_ERROR("plot", "save_h5: cannot create {}", d.path);
        return;
    }
    group g = group::create(f.id(), "plot");
    if (!g) {
        COAST_ERROR("plot", "save_h5: cannot create /plot in {}", d.path);
        return;
    }
    write_plot_into(g.id(), d);
    write_thumbnail_into(g.id(), d.thumb_png, d.thumb_w, d.thumb_h);
    // On a full disk the writes above usually succeed into HDF5's cache and the
    // close is what fails (#203). Release the group first so the close is real.
    g = {};
    if (f.close() < 0) COAST_ERROR("plot", "save_h5: writing {} failed; the file is incomplete", d.path);
}

// The attribute and dataset readers throw on a shape or type the writer never
// produces; read_plot_from turns that into a failed load.
static bool read_plot_from_unchecked(hid_t gid, h5_save_data& out, const std::string& ctx)
{
    using namespace hdf5;
    auto kind = read_attr_string(gid, "kind");
    if (!kind) {
        COAST_ERROR("plot", "load: missing 'kind' attr in {}", ctx);
        return false;
    }
    out.kind  = *kind;
    out.title = read_attr_string(gid, "title").value_or("loaded");
    out.theme = read_attr_string(gid, "theme").value_or("dark");

    // grid is written [nby, nbx], the others 1-D.
    auto read_doubles = [&](const char* name, int rank, std::vector<double>& v) -> bool {
        dataset ds = dataset::open(gid, name);
        if (!ds) return false;
        v = read_array<double>(ds.id(), H5T_NATIVE_DOUBLE, rank).values;
        return true;
    };

    if (out.kind == "histogram2d" || out.kind == "pixels") {
        out.nbx         = static_cast<int>(read_attr_int64(gid, "nbx").value_or(0));
        out.nby         = static_cast<int>(read_attr_int64(gid, "nby").value_or(0));
        out.x_min       = read_attr_double(gid, "x_min").value_or(0.0);
        out.x_max       = read_attr_double(gid, "x_max").value_or(1.0);
        out.y_min       = read_attr_double(gid, "y_min").value_or(0.0);
        out.y_max       = read_attr_double(gid, "y_max").value_or(1.0);
        out.log_density = read_attr_int64(gid, "log_density").value_or(0) != 0;
        // Backward-compatible: files written before axis toggles existed default
        // to all axes visible.
        out.show_x_axis = read_attr_int64(gid, "show_x_axis").value_or(1) != 0;
        out.show_y_axis = read_attr_int64(gid, "show_y_axis").value_or(1) != 0;
        out.show_z_axis = read_attr_int64(gid, "show_z_axis").value_or(1) != 0;
        out.mode        = read_attr_string(gid, "mode").value_or("heatmap");
        out.colormap    = read_attr_string(gid, "colormap").value_or("Viridis");
        read_fixed_domains(gid, out);
        // Absent in pre-fixed-domain files, in auto-ranged ones, and in fixed-
        // domain ones with nothing outside the grid: an empty ring is the correct
        // reading of all three.
        if (H5Lexists(gid, "outer", H5P_DEFAULT) > 0) read_doubles("outer", 1, out.outer);
        if (!read_doubles("grid", 2, out.grid)) {
            COAST_ERROR("plot", "load: cannot read grid in {}", ctx);
            return false;
        }
        if (out.grid.size() != static_cast<std::size_t>(out.nbx) * out.nby) {
            COAST_ERROR("plot", "load: grid size {} != nbx*nby ({}*{})", out.grid.size(), out.nbx,
                        out.nby);
            return false;
        }
        return true;
    }

    if (out.kind == "histogram" || out.kind == "line") {
        out.bins = static_cast<int>(read_attr_int64(gid, "bins").value_or(0));
        read_fixed_domains(gid, out);
        if (!read_doubles("y", 1, out.ys)) {
            COAST_ERROR("plot", "load: cannot read y in {}", ctx);
            return false;
        }
        if (H5Lexists(gid, "x", H5P_DEFAULT) > 0) read_doubles("x", 1, out.xs);
        return true;
    }

    COAST_ERROR("plot", "load: unknown plot kind '{}' in {}", out.kind, ctx);
    return false;
}

bool read_plot_from(hid_t gid, h5_save_data& out, const std::string& ctx)
{
    try {
        return read_plot_from_unchecked(gid, out, ctx);
    } catch (const std::exception& e) {
        COAST_ERROR("plot", "load: malformed {}: {}", ctx, e.what());
        return false;
    }
}

bool read_plot_h5(const std::string& path, h5_save_data& out)
{
    using namespace hdf5;
    file f = file::open_read(path);
    if (!f) {
        COAST_ERROR("plot", "load: cannot open {}", path);
        return false;
    }
    group g = group::open(f.id(), "plot");
    if (!g) {
        COAST_ERROR("plot", "load: no /plot group in {}", path);
        return false;
    }
    return read_plot_from(g.id(), out, path);
}

void write_canvas_h5(const canvas_save_data& d)
{
    using namespace hdf5;
    file f = file::create(d.path);
    if (!f) {
        COAST_ERROR("plot", "save_h5: cannot create {}", d.path);
        return;
    }
    group g = group::create(f.id(), "canvas");
    if (!g) {
        COAST_ERROR("plot", "save_h5: cannot create /canvas in {}", d.path);
        return;
    }
    herr_t crc = 0;
    crc |= write_attr(g.id(), "coast_canvas_version", static_cast<std::int64_t>(1));
    crc |= write_attr(g.id(), "title", d.title);
    crc |= write_attr(g.id(), "rows", static_cast<std::int64_t>(d.rows));
    crc |= write_attr(g.id(), "cols", static_cast<std::int64_t>(d.cols));
    if (crc < 0) COAST_WARN("plot", "canvas attributes failed to write to {}", d.path);
    for (const auto& cell : d.cells) {
        char name[32];
        std::snprintf(name, sizeof name, "cell_%d_%d", cell.row, cell.col);
        group cg = group::create(g.id(), name);
        if (!cg) continue;
        write_plot_into(cg.id(), cell.data);
    }
    write_thumbnail_into(g.id(), d.thumb_png, d.thumb_w, d.thumb_h);
    // As in write_plot_h5: the close is where a full disk usually fails (#230).
    g = {};
    if (f.close() < 0) COAST_ERROR("plot", "save_h5: writing {} failed; the file is incomplete", d.path);
}

static bool read_canvas_h5_unchecked(const std::string& path, canvas_save_data& out)
{
    using namespace hdf5;
    file f = file::open_read(path);
    if (!f) {
        COAST_ERROR("plot", "load: cannot open {}", path);
        return false;
    }
    group g = group::open(f.id(), "canvas");
    if (!g) {
        COAST_ERROR("plot", "load: no /canvas group in {}", path);
        return false;
    }
    out.rows  = static_cast<int>(read_attr_int64(g.id(), "rows").value_or(0));
    out.cols  = static_cast<int>(read_attr_int64(g.id(), "cols").value_or(0));
    out.title = read_attr_string(g.id(), "title").value_or("canvas");
    if (out.rows < 1 || out.cols < 1) {
        COAST_ERROR("plot", "load: bad canvas dims {}x{} in {}", out.rows, out.cols, path);
        return false;
    }
    for (int r = 0; r < out.rows; ++r)
        for (int c = 0; c < out.cols; ++c) {
            char name[32];
            std::snprintf(name, sizeof name, "cell_%d_%d", r, c);
            if (H5Lexists(g.id(), name, H5P_DEFAULT) <= 0) continue;
            group cg = group::open(g.id(), name);
            if (!cg) continue;
            h5_save_data d;
            if (read_plot_from(cg.id(), d, name)) out.cells.push_back({r, c, std::move(d)});
        }
    return true;
}

// The canvas attributes throw on a malformed file; a malformed cell is only
// skipped (read_plot_from does not throw).
bool read_canvas_h5(const std::string& path, canvas_save_data& out)
{
    try {
        return read_canvas_h5_unchecked(path, out);
    } catch (const std::exception& e) {
        COAST_ERROR("plot", "load: malformed {}: {}", path, e.what());
        return false;
    }
}

} // namespace coast::plot
