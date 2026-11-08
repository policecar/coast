// console_plot — the data/registry layer of the ImPlot console bridge. Rendering
// needs a live GUI context (verified by hand); this exercises figure creation,
// thread-safe data update, point_count, and clear_all headlessly.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "gui_plot/console_plot.h"

#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace p = coast::plot;

namespace {

// Write a minimal saved-figure HDF5 file (the /plot schema written by
// console_plot's own writer) directly via the hdf5 layer, so the load() /
// load_canvas() read+reconstruct path can be exercised headlessly (the real
// save path renders a GL thumbnail and needs a GUI context). All libhdf5 calls
// run on the executor worker thread, per the executor contract.

// A single "line" plot group under an already-open parent `loc`.
void write_line_group(hid_t loc, const char* name, const std::vector<double>& ys)
{
    using namespace hdf5;
    group   g  = group::create(loc, name);
    herr_t  rc = 0;
    rc |= write_attr(g.id(), "coast_plot_version", static_cast<std::int64_t>(1));
    rc |= write_attr(g.id(), "kind", std::string_view("line"));
    rc |= write_attr(g.id(), "title", std::string_view("rt-line"));
    rc |= write_attr(g.id(), "theme", std::string_view("dark"));
    REQUIRE(rc >= 0);
    const hsize_t dims[1] = {ys.size()};
    dataspace     sp      = dataspace::simple(dims);
    dataset       ds      = dataset::create(g.id(), "y", H5T_NATIVE_DOUBLE, sp.id());
    REQUIRE(ds.write(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, ys.data()) >= 0);
}

// A single "histogram2d" plot group carrying a named colormap — exercises the
// loaded-kind path whose colormap name maps to the palette enum on load.
void write_hist2d_group(hid_t loc, const char* name, int nbx, int nby)
{
    using namespace hdf5;
    group  g  = group::create(loc, name);
    herr_t rc = 0;
    rc |= write_attr(g.id(), "coast_plot_version", static_cast<std::int64_t>(1));
    rc |= write_attr(g.id(), "kind", std::string_view("histogram2d"));
    rc |= write_attr(g.id(), "title", std::string_view("rt-h2d"));
    rc |= write_attr(g.id(), "theme", std::string_view("dark"));
    rc |= write_attr(g.id(), "nbx", static_cast<std::int64_t>(nbx));
    rc |= write_attr(g.id(), "nby", static_cast<std::int64_t>(nby));
    rc |= write_attr(g.id(), "x_min", 0.0);
    rc |= write_attr(g.id(), "x_max", 1.0);
    rc |= write_attr(g.id(), "y_min", 0.0);
    rc |= write_attr(g.id(), "y_max", 1.0);
    rc |= write_attr(g.id(), "mode", std::string_view("heatmap"));
    rc |= write_attr(g.id(), "colormap", std::string_view("Plasma"));
    REQUIRE(rc >= 0);
    std::vector<double> grid(static_cast<std::size_t>(nbx) * nby, 1.0);
    const hsize_t       dims[2] = {static_cast<hsize_t>(nby), static_cast<hsize_t>(nbx)};
    dataspace           sp      = dataspace::simple(dims);
    dataset             ds      = dataset::create(g.id(), "grid", H5T_NATIVE_DOUBLE, sp.id());
    REQUIRE(ds.write(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, grid.data()) >= 0);
}

std::string tmp_path(const char* name)
{
    return (std::filesystem::temp_directory_path() / name).string();
}

} // namespace

// load()/load_canvas() must fail cleanly (nullptr) when no executor is wired.
// This must run BEFORE any test calls set_executor, since the executor is a
// process-wide global with no un-set path.
TEST_CASE("console_plot: load returns nullptr when no executor is set")
{
    CHECK(p::load("/nonexistent.h5") == nullptr);
    CHECK(p::load_canvas("/nonexistent.h5") == nullptr);
}

TEST_CASE("console_plot: figure save schema round-trips through load()")
{
    hdf5::executor exec;
    p::set_executor(exec);
    p::free_resources();

    // -- line --
    const std::string   line_path = tmp_path("coast_rt_line.h5");
    std::vector<double> ys(42);
    for (int i = 0; i < 42; ++i) ys[i] = static_cast<double>(i);
    exec.submit_sync([&] {
        hdf5::file f = hdf5::file::create(line_path);
        write_line_group(f.id(), "plot", ys);
        return true;
    });

    auto f = p::load(line_path);
    REQUIRE(f != nullptr);
    CHECK(f->point_count() == ys.size());
    CHECK(p::plot_count() == 1); // the loaded figure is registered

    // -- histogram2d (loaded colormap name maps to the palette enum) --
    const std::string h2d_path = tmp_path("coast_rt_h2d.h5");
    exec.submit_sync([&] {
        hdf5::file fh = hdf5::file::create(h2d_path);
        write_hist2d_group(fh.id(), "plot", 8, 5);
        return true;
    });
    auto g = p::load(h2d_path);
    REQUIRE(g != nullptr);
    CHECK(g->point_count() == static_cast<std::size_t>(8 * 5)); // grid size, no draw() needed

    // Bad path → nullptr, no throw.
    CHECK(p::load(tmp_path("coast_rt_missing.h5")) == nullptr);

    p::free_resources();
    std::filesystem::remove(line_path);
    std::filesystem::remove(h2d_path);
}

TEST_CASE("console_plot: canvas save schema round-trips through load_canvas()")
{
    hdf5::executor exec;
    p::set_executor(exec);
    p::free_resources();

    const std::string   path = tmp_path("coast_rt_canvas.h5");
    std::vector<double> ys(16, 3.0);
    exec.submit_sync([&] {
        hdf5::file  f = hdf5::file::create(path);
        hdf5::group g = hdf5::group::create(f.id(), "canvas");
        herr_t rc = 0;
        rc |= hdf5::write_attr(g.id(), "coast_canvas_version", static_cast<std::int64_t>(1));
        rc |= hdf5::write_attr(g.id(), "title", std::string_view("rt-canvas"));
        rc |= hdf5::write_attr(g.id(), "rows", static_cast<std::int64_t>(2));
        rc |= hdf5::write_attr(g.id(), "cols", static_cast<std::int64_t>(2));
        REQUIRE(rc >= 0);
        write_line_group(g.id(), "cell_0_0", ys);
        write_hist2d_group(g.id(), "cell_1_1", 4, 4);
        return true;
    });

    auto c = p::load_canvas(path);
    REQUIRE(c != nullptr);
    // The canvas is registered; its reconstructed children are canvas-owned and
    // not separately registered.
    CHECK(p::plot_count() == 1);

    CHECK(p::load_canvas(tmp_path("coast_rt_missing_canvas.h5")) == nullptr);

    p::free_resources();
    std::filesystem::remove(path);
}

TEST_CASE("console_plot: create, update, count, clear")
{
    p::free_resources();
    CHECK(p::plot_count() == 0);

    // Construction alone does not register — show() does.
    auto h = std::make_shared<p::histogram>("hist", 32);
    auto l = std::make_shared<p::line>("line");
    CHECK(p::plot_count() == 0);
    h->show();
    l->show();
    CHECK(p::plot_count() == 2);

    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) data[i] = static_cast<double>(i);
    h->update(data);
    CHECK(h->point_count() == 100);

    std::vector<double> x(50, 0.0);
    std::vector<double> y(50, 1.0);
    l->update(x, y);
    CHECK(l->point_count() == 50);

    // float overload (matches row_view::as<float>()).
    std::vector<float> fdata(10, 2.0f);
    h->update(std::span<const float>(fdata));
    CHECK(h->point_count() == 10);

    p::free_resources();
    CHECK(p::plot_count() == 0);

    // The macro's handle keeps the figure object alive after clear; updating it
    // is safe (just no longer rendered).
    h->update(std::vector<double>(5, 0.0));
    CHECK(h->point_count() == 5);
}

TEST_CASE("console_plot: image kind registers on show() (source stored; render needs GUI)")
{
    p::free_resources();
    CHECK(p::plot_count() == 0);

    auto img = std::make_shared<p::image>("picture");
    img->show();
    CHECK(p::plot_count() == 1);
    // set_png only records the path on this (non-GUI) thread; decode + GL upload
    // happen in render() on the GUI thread.
    img->set_png("/tmp/does_not_matter_here.png");
    img->set_title("renamed");

    std::vector<std::uint8_t> px(4 * 2 * 4, 200); // 4x2 RGBA
    img->set_rgba(px, 4, 2);

    p::free_resources();
    CHECK(p::plot_count() == 0);
}

TEST_CASE("console_plot: canvas registers and hosts panels")
{
    p::free_resources();
    CHECK(p::plot_count() == 0);

    auto c = std::make_shared<p::canvas>("dash", 2, 2);
    c->show();
    CHECK(p::plot_count() == 1); // the canvas is itself a figure

    auto h  = std::make_shared<p::histogram>("h");
    auto px = std::make_shared<p::pixels>("in", 4, 4);
    h->show();
    px->show();
    CHECK(p::plot_count() == 3); // canvas + 2 shown figures

    // Placement is safe; null and out-of-range indices are ignored (no crash, no
    // registry change — a placed figure stays registered, just hosted). Rendering
    // itself needs a GUI context (verified by hand).
    c->place(h, 0, 0);
    c->place(px, 1, 1);
    c->place(std::shared_ptr<p::figure>{}, 0, 1); // null → ignored
    c->place(h, 99, 99);                          // out of range → ignored
    c->clear_cell(0, 0);
    c->clear_cell(5, 5);                          // out of range → ignored
    CHECK(p::plot_count() == 3);

    c->close();
    p::free_resources();
    CHECK(p::plot_count() == 0);
}

TEST_CASE("console_plot: concurrent updates with a concurrent reader are safe")
{
    p::free_resources();
    auto f = std::make_shared<p::line>("concurrent");

    // Writers hammer update() (mimicking the console thread) while a reader
    // hammers point_count() (mimicking the GUI thread's render read). This is the
    // real clash the console guards against: the reader must never see a torn
    // size, so point_count() must always be one of the sizes a writer sets.
    // Sizes are 100..103 (100 + thread index); the reader also tolerates 0 in
    // case it runs before the first update lands.
    constexpr int            n_writers = 4;
    std::atomic<bool>        stop_reader{false};
    std::atomic<bool>        reader_ok{true};
    std::vector<std::thread> ts;
    for (int t = 0; t < n_writers; ++t) {
        ts.emplace_back([f, t] {
            std::vector<double> d(static_cast<std::size_t>(100 + t), static_cast<double>(t));
            for (int i = 0; i < 2000; ++i) f->update(d);
        });
    }
    std::thread reader([&] {
        while (!stop_reader.load(std::memory_order_relaxed)) {
            const std::size_t n = f->point_count();
            if (n != 0 && (n < 100 || n >= 100 + static_cast<std::size_t>(n_writers)))
                reader_ok.store(false, std::memory_order_relaxed);
        }
    });

    for (auto& t : ts) t.join();
    stop_reader.store(true, std::memory_order_relaxed);
    reader.join();

    CHECK(reader_ok.load());        // never observed a torn/out-of-range size
    CHECK(f->point_count() >= 100); // whichever update landed last
    p::free_resources();
}

// The redesign's headline: kind-specific config is reachable directly on the
// concrete type — no GUI, no down-cast. Axis visibility is pure state, so this
// runs headlessly (only draw() needs a GL context).
TEST_CASE("console_plot: histogram2d/pixels axis visibility is settable programmatically")
{
    p::free_resources();

    auto h = std::make_shared<p::histogram2d>("axtest", 8, 8);
    // Default: every axis visible.
    CHECK(h->axis_visible(p::plot_axis::x));
    CHECK(h->axis_visible(p::plot_axis::y));
    CHECK(h->axis_visible(p::plot_axis::z));

    h->set_axis_visible(p::plot_axis::x, false);
    h->set_axis_visible(p::plot_axis::z, false);
    CHECK_FALSE(h->axis_visible(p::plot_axis::x));
    CHECK(h->axis_visible(p::plot_axis::y)); // untouched
    CHECK_FALSE(h->axis_visible(p::plot_axis::z));

    // Idempotent + reversible.
    h->set_axis_visible(p::plot_axis::x, true);
    CHECK(h->axis_visible(p::plot_axis::x));

    // pixels exposes the same typed surface.
    auto px = std::make_shared<p::pixels>("pxax", 4, 4);
    px->set_axis_visible(p::plot_axis::y, false);
    CHECK_FALSE(px->axis_visible(p::plot_axis::y));
    CHECK(px->axis_visible(p::plot_axis::x));

    p::free_resources();
}
