// Fixed histogram domains — the binning semantics, the ROOT-style outer-ring
// accounting, and the save/load round-trip of the domains themselves.
//
// bin_points is the unit under test for the 2D path: it is pure computation (no
// ImGui calls), so it runs headlessly even though plot_tools.h pulls the ImPlot
// headers in. The 1D path delegates its binning to ImPlot at paint time, so what
// is observable headlessly there is the domain + the under/overflow tallies —
// the drawn bars are verified by hand against a live GUI.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "gui_plot/console_plot.h"
#include "gui_plot/plot_h5.h"    // write_plot_into / read_plot_from (the writer half)
#include "gui_plot/plot_tools.h" // bin_points (the 2D binner itself)

#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <cstdint>
#include <filesystem>
#include <numeric>
#include <optional>
#include <string_view>
#include <vector>

namespace p  = coast::plot;
namespace fs = std::filesystem;

namespace {

double grid_total(const p::grid_result& g)
{
    return std::accumulate(g.grid.begin(), g.grid.end(), 0.0);
}

} // namespace

// ---------------------------------------------------------------- bin_points

TEST_CASE("bin_points: no domain auto-ranges from the data, exactly as before")
{
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> y{0.0, 1.0, 2.0, 3.0};

    const p::grid_result g = p::bin_points(x, y, 4, 4);

    // Frame derived from the data's own min/max.
    CHECK(g.x0 == doctest::Approx(0.0));
    CHECK(g.x1 == doctest::Approx(3.0));
    CHECK(g.y0 == doctest::Approx(0.0));
    CHECK(g.y1 == doctest::Approx(3.0));
    // Nothing can fall outside a frame that came from the data.
    CHECK(g.outer.empty());
    CHECK(grid_total(g) == doctest::Approx(4.0));
}

TEST_CASE("bin_points: a fixed domain replaces the data-derived frame")
{
    // Data spans [1,2] but the domain says [0,10]: the frame must be the domain,
    // which is the entire point — two samples binned this way share bin edges.
    const std::vector<double> x{1.0, 2.0};
    const std::vector<double> y{1.0, 2.0};

    const p::grid_result g = p::bin_points(x, y, 10, 10, p::range{0.0, 10.0}, p::range{0.0, 10.0});

    CHECK(g.x0 == doctest::Approx(0.0));
    CHECK(g.x1 == doctest::Approx(10.0));
    CHECK(g.y0 == doctest::Approx(0.0));
    CHECK(g.y1 == doctest::Approx(10.0));
    CHECK(g.outer.empty()); // both points are inside
    CHECK(grid_total(g) == doctest::Approx(2.0));
}

TEST_CASE("bin_points: two different samples share bin edges under one domain")
{
    // The property the whole feature exists for. Auto-ranged, these two samples
    // get different frames and their bins mean different things, so the figures
    // are not comparable. Under one domain the frames are identical and bin k of
    // one is bin k of the other.
    const std::vector<double> xa{0.10, 0.20}, ya{0.10, 0.20}; // tight sample
    const std::vector<double> xb{0.00, 0.95}, yb{0.00, 0.95}; // wide sample

    const p::grid_result auto_a = p::bin_points(xa, ya, 4, 4);
    const p::grid_result auto_b = p::bin_points(xb, yb, 4, 4);
    CHECK(auto_a.x0 != doctest::Approx(auto_b.x0)); // different frames => not comparable
    CHECK(auto_a.x1 != doctest::Approx(auto_b.x1));

    const p::range       dom{0.0, 1.0};
    const p::grid_result fix_a = p::bin_points(xa, ya, 4, 4, dom, dom);
    const p::grid_result fix_b = p::bin_points(xb, yb, 4, 4, dom, dom);
    CHECK(fix_a.x0 == doctest::Approx(fix_b.x0));
    CHECK(fix_a.x1 == doctest::Approx(fix_b.x1));
    CHECK(fix_a.y0 == doctest::Approx(fix_b.y0));
    CHECK(fix_a.y1 == doctest::Approx(fix_b.y1));
    CHECK(fix_a.grid.size() == fix_b.grid.size());
    // ...and no phantom counts: the augment-the-data workaround this replaces
    // (pushing 0.0 and 1.0 onto both axes to pin the frame) added two points
    // that were never measured.
    CHECK(grid_total(fix_a) == doctest::Approx(2.0));
    CHECK(grid_total(fix_b) == doctest::Approx(2.0));
}

TEST_CASE("bin_points: axes are independently fixable")
{
    const std::vector<double> x{1.0, 2.0};
    const std::vector<double> y{5.0, 7.0};

    // x fixed, y auto.
    const p::grid_result g = p::bin_points(x, y, 4, 4, p::range{0.0, 10.0}, std::nullopt);
    CHECK(g.x0 == doctest::Approx(0.0));  // fixed
    CHECK(g.x1 == doctest::Approx(10.0));
    CHECK(g.y0 == doctest::Approx(5.0));  // auto, from the data
    CHECK(g.y1 == doctest::Approx(7.0));

    // y fixed, x auto.
    const p::grid_result h = p::bin_points(x, y, 4, 4, std::nullopt, p::range{0.0, 10.0});
    CHECK(h.x0 == doctest::Approx(1.0));
    CHECK(h.x1 == doctest::Approx(2.0));
    CHECK(h.y0 == doctest::Approx(0.0));
    CHECK(h.y1 == doctest::Approx(10.0));
}

TEST_CASE("bin_points: out-of-range points go to the ring, NOT into the edge bins")
{
    // The distinguishing test. Clamping would put the outlier in the top-right
    // grid cell and inflate it; dropping would lose it entirely. ROOT semantics:
    // it lands in exactly one ring cell and the grid is untouched.
    const std::vector<double> x{0.5, 99.0};
    const std::vector<double> y{0.5, 99.0};

    const p::grid_result g = p::bin_points(x, y, 4, 4, p::range{0.0, 1.0}, p::range{0.0, 1.0});

    CHECK(grid_total(g) == doctest::Approx(1.0)); // only the in-range point
    CHECK(g.outer.total() == doctest::Approx(1.0));
    CHECK(g.outer.at(p::zone::over, p::zone::over) == doctest::Approx(1.0));
    // The edge bin was NOT inflated: max_count reflects one point per cell.
    CHECK(g.max_count == doctest::Approx(1.0));
    // Nothing lost, nothing double counted.
    CHECK(grid_total(g) + g.outer.total() == doctest::Approx(2.0));
}

TEST_CASE("bin_points: a corner outlier is counted once, in one cell")
{
    // under-x AND over-y simultaneously: the case that per-axis counters would
    // double count. It must land in exactly one ring cell.
    const std::vector<double> x{-5.0};
    const std::vector<double> y{99.0};

    const p::grid_result g = p::bin_points(x, y, 4, 4, p::range{0.0, 1.0}, p::range{0.0, 1.0});

    CHECK(grid_total(g) == doctest::Approx(0.0));
    CHECK(g.outer.total() == doctest::Approx(1.0)); // once, not twice
    CHECK(g.outer.at(p::zone::over, p::zone::under) == doctest::Approx(1.0));
}

TEST_CASE("bin_points: every zone is reachable and the ring conserves points")
{
    // One point per zone (9 total, 8 outside + 1 inside), so a mis-signed zone
    // test or a transposed [zy][zx] index shows up as a misplaced count.
    const std::vector<double> x{-1, 0.5, 9,  -1, 0.5, 9, -1, 0.5, 9};
    const std::vector<double> y{-1, -1,  -1, 0.5, 0.5, 0.5, 9, 9, 9};

    const p::grid_result g = p::bin_points(x, y, 4, 4, p::range{0.0, 1.0}, p::range{0.0, 1.0});

    CHECK(grid_total(g) == doctest::Approx(1.0)); // the (0.5, 0.5) point
    CHECK(g.outer.total() == doctest::Approx(8.0));
    CHECK(grid_total(g) + g.outer.total() == doctest::Approx(9.0));

    CHECK(g.outer.at(p::zone::under, p::zone::under) == doctest::Approx(1.0));
    CHECK(g.outer.at(p::zone::under, p::zone::inside) == doctest::Approx(1.0));
    CHECK(g.outer.at(p::zone::under, p::zone::over) == doctest::Approx(1.0));
    CHECK(g.outer.at(p::zone::inside, p::zone::under) == doctest::Approx(1.0));
    CHECK(g.outer.at(p::zone::inside, p::zone::over) == doctest::Approx(1.0));
    CHECK(g.outer.at(p::zone::over, p::zone::under) == doctest::Approx(1.0));
    CHECK(g.outer.at(p::zone::over, p::zone::inside) == doctest::Approx(1.0));
    CHECK(g.outer.at(p::zone::over, p::zone::over) == doctest::Approx(1.0));
    // [inside][inside] is the grid, never the ring.
    CHECK(g.outer.at(p::zone::inside, p::zone::inside) == doctest::Approx(0.0));
}

TEST_CASE("bin_points: the closed upper edge is inside, in the last bin")
{
    // A value exactly at hi indexes to nbx and must clamp back to nbx-1 rather
    // than be treated as an overflow — range is closed [lo, hi].
    const std::vector<double> x{1.0};
    const std::vector<double> y{1.0};

    const p::grid_result g = p::bin_points(x, y, 4, 4, p::range{0.0, 1.0}, p::range{0.0, 1.0});

    CHECK(g.outer.empty());
    CHECK(grid_total(g) == doctest::Approx(1.0));
    CHECK(g.grid[3] == doctest::Approx(1.0)); // row 0 (top = max y), last column
}

TEST_CASE("bin_points: an invalid domain falls back to auto-ranging")
{
    const std::vector<double> x{2.0, 4.0};
    const std::vector<double> y{2.0, 4.0};

    // hi <= lo would otherwise collapse the frame to zero width and divide by 0.
    const p::grid_result g = p::bin_points(x, y, 4, 4, p::range{5.0, 5.0}, p::range{9.0, 1.0});

    CHECK(g.x0 == doctest::Approx(2.0)); // auto, from the data
    CHECK(g.x1 == doctest::Approx(4.0));
    CHECK(g.y0 == doctest::Approx(2.0));
    CHECK(g.y1 == doctest::Approx(4.0));
    CHECK(g.outer.empty());
    CHECK(grid_total(g) == doctest::Approx(2.0));
}

// ------------------------------------------------------- figure-level surface

TEST_CASE("histogram: auto-ranged has no domain and never any outliers")
{
    auto                h = std::make_shared<p::histogram>("h", 8);
    std::vector<double> v{-100.0, 0.5, 100.0};
    h->update(v);

    CHECK(!h->domain().has_value());
    CHECK(h->underflow() == 0);
    CHECK(h->overflow() == 0);
    CHECK(h->point_count() == 3);
}

TEST_CASE("histogram: a fixed domain tallies under/overflow instead of dropping them")
{
    auto h = std::make_shared<p::histogram>("h", 8, p::range{0.0, 1.0});
    // optional<range> == range: engagement and value in one check.
    CHECK(h->domain() == p::range{0.0, 1.0});

    std::vector<double> v{-3.0, -1.0, 0.25, 0.75, 1.0, 4.0};
    h->update(v);

    CHECK(h->underflow() == 2);                    // -3, -1
    CHECK(h->overflow() == 1);                     // 4  (1.0 is inside: closed)
    CHECK(h->point_count() == 6);                  // nothing dropped from the data
    CHECK(h->point_count() - h->underflow() - h->overflow() == 3); // in-range
}

TEST_CASE("histogram: update() re-tallies, it does not accumulate")
{
    auto h = std::make_shared<p::histogram>("h", 8, p::range{0.0, 1.0});

    std::vector<double> a{-1.0, -2.0};
    h->update(a);
    CHECK(h->underflow() == 2);

    std::vector<double> b{0.5};
    h->update(b); // stale tallies would still read 2
    CHECK(h->underflow() == 0);
    CHECK(h->overflow() == 0);
}

TEST_CASE("histogram: an invalid domain falls back to auto-ranging")
{
    auto h = std::make_shared<p::histogram>("h", 8, p::range{1.0, 1.0}); // hi <= lo
    CHECK(!h->domain().has_value());

    std::vector<double> v{-5.0, 5.0};
    h->update(v);
    CHECK(h->underflow() == 0); // auto-ranged: nothing is "outside"
    CHECK(h->overflow() == 0);
}

TEST_CASE("histogram2d: domains are readable and the ring is reachable")
{
    auto h = std::make_shared<p::histogram2d>("h2", 4, 4, p::range{0.0, 1.0}, std::nullopt);

    CHECK(h->x_domain() == p::range{0.0, 1.0});
    CHECK(!h->y_domain().has_value()); // independently auto

    std::vector<double> x{0.5, 9.0};
    std::vector<double> y{0.5, 0.6};
    h->update(x, y);

    const p::outer_ring r = h->outer();
    CHECK(r.total() == doctest::Approx(1.0)); // the x=9 point
    CHECK(r.at(p::zone::inside, p::zone::over) == doctest::Approx(1.0));
}

TEST_CASE("histogram2d: auto-ranged figure has an empty ring")
{
    auto                h = std::make_shared<p::histogram2d>("h2", 4, 4);
    std::vector<double> x{0.0, 1.0, 500.0};
    std::vector<double> y{0.0, 1.0, 500.0};
    h->update(x, y);

    CHECK(!h->x_domain().has_value());
    CHECK(!h->y_domain().has_value());
    CHECK(h->outer().empty());
}

// -------------------------------------------------------------- save / load
// The real save path renders a GL thumbnail and needs a GUI context, so the
// round-trip is exercised by writing the /plot schema directly and loading it —
// the same approach test_console_plot.cpp uses.

namespace {

void write_hist_group(hid_t loc, const char* name, const std::vector<double>& ys, int bins,
                      std::optional<p::range> domain)
{
    using namespace hdf5;
    group  g  = group::create(loc, name);
    herr_t rc = 0;
    rc |= write_attr(g.id(), "coast_plot_version", static_cast<std::int64_t>(1));
    rc |= write_attr(g.id(), "kind", std::string_view("histogram"));
    rc |= write_attr(g.id(), "title", std::string_view("rt-hist"));
    rc |= write_attr(g.id(), "theme", std::string_view("dark"));
    rc |= write_attr(g.id(), "bins", static_cast<std::int64_t>(bins));
    if (domain) {
        rc |= write_attr(g.id(), "x_range_lo", domain->lo);
        rc |= write_attr(g.id(), "x_range_hi", domain->hi);
    }
    REQUIRE(rc >= 0);
    const hsize_t dims[1] = {ys.size()};
    dataspace     sp      = dataspace::simple(dims);
    dataset       ds      = dataset::create(g.id(), "y", H5T_NATIVE_DOUBLE, sp.id());
    REQUIRE(ds.write(H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, ys.data()) >= 0);
}

fs::path make_hist_file(hdf5::executor& exec, const std::vector<double>& ys,
                        std::optional<p::range> domain, const char* fname)
{
    const fs::path  path = fs::temp_directory_path() / fname;
    std::error_code ec;
    fs::remove(path, ec);
    exec.submit_sync([&] {
        const hid_t fid =
            H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        write_hist_group(fid, "plot", ys, 8, domain);
        H5Fclose(fid);
    });
    return path;
}

} // namespace

TEST_CASE("histogram: a saved fixed domain survives the round-trip")
{
    hdf5::executor exec;
    p::set_executor(exec); // process-wide global; there is no un-set path
    p::free_resources();

    const std::vector<double> ys{-3.0, 0.25, 0.75, 4.0};
    const fs::path path = make_hist_file(exec, ys, p::range{0.0, 1.0}, "coast_hist_range_rt.h5");

    auto f = p::load(path);
    REQUIRE(f != nullptr);
    auto h = std::dynamic_pointer_cast<p::histogram>(f);
    REQUIRE(h != nullptr);

    // Without persistence this reloads auto-ranged and the frame is silently
    // lost — the regression this field exists to prevent.
    CHECK(h->domain() == p::range{0.0, 1.0});
    // The tallies are recomputed from the raw ys, not persisted.
    CHECK(h->underflow() == 1);
    CHECK(h->overflow() == 1);

    std::error_code ec;
    fs::remove(path, ec);
}

// The tests above hand-write the /plot schema, so they only exercise the READ
// half. These drive the real writer (write_plot_into) into the real reader
// (read_plot_from), which is what catches the two halves disagreeing — a field
// name typo'd on one side reads back as "absent", i.e. silently auto-ranged.
// write_plot_into is pure HDF5; only the thumbnail needs GL, so this runs
// headlessly. The whole-file save path (figure::save) remains hand-verified.

namespace {

// One write -> read cycle through the real writer/reader pair.
p::h5_save_data write_then_read(hdf5::executor& exec, const p::h5_save_data& in, const char* fname)
{
    const fs::path  path = fs::temp_directory_path() / fname;
    std::error_code ec;
    fs::remove(path, ec);

    p::h5_save_data out;
    exec.submit_sync([&] {
        {
            const hid_t fid =
                H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            REQUIRE(fid >= 0);
            const hid_t g = H5Gcreate2(fid, "plot", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            REQUIRE(g >= 0);
            p::write_plot_into(g, in); // the half nothing else covers
            H5Gclose(g);
            H5Fclose(fid);
        }
        const hid_t fid = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        const hid_t g = H5Gopen2(fid, "plot", H5P_DEFAULT);
        REQUIRE(g >= 0);
        REQUIRE(p::read_plot_from(g, out, "test"));
        H5Gclose(g);
        H5Fclose(fid);
    });
    fs::remove(path, ec);
    return out;
}

} // namespace

TEST_CASE("plot_h5: writer and reader agree on a 1D histogram's domain")
{
    hdf5::executor exec;

    p::h5_save_data in;
    in.kind        = "histogram";
    in.title       = "w";
    in.bins        = 8;
    in.ys          = {0.1, 0.2, 0.3};
    in.has_x_range = true;
    in.x_range_lo  = -2.5;
    in.x_range_hi  = 7.5;

    const p::h5_save_data out = write_then_read(exec, in, "coast_h5_wr_hist.h5");

    REQUIRE(out.has_x_range); // false here would mean the writer/reader disagree
    CHECK(out.x_range_lo == doctest::Approx(-2.5));
    CHECK(out.x_range_hi == doctest::Approx(7.5));
    CHECK(out.bins == 8);
    REQUIRE(out.ys.size() == 3);
}

TEST_CASE("plot_h5: an auto-ranged 1D histogram writes no domain attrs")
{
    hdf5::executor exec;

    p::h5_save_data in;
    in.kind  = "histogram";
    in.title = "w";
    in.bins  = 8;
    in.ys    = {0.1, 0.2};
    // has_x_range stays false: nothing should be written, and the reader must not
    // invent a domain from whatever else is in the group.

    const p::h5_save_data out = write_then_read(exec, in, "coast_h5_wr_hist_auto.h5");

    CHECK(!out.has_x_range);
    CHECK(!out.has_y_range);
}

TEST_CASE("plot_h5: writer and reader agree on a histogram2d's domains and ring")
{
    hdf5::executor exec;

    p::h5_save_data in;
    in.kind        = "histogram2d";
    in.title       = "w2";
    in.nbx         = 2;
    in.nby         = 2;
    in.grid        = {1.0, 2.0, 3.0, 4.0};
    in.x_min = 0.0, in.x_max = 1.0, in.y_min = 0.0, in.y_max = 1.0;
    in.mode        = "heatmap";
    in.colormap    = "Viridis";
    in.has_x_range = true;
    in.x_range_lo  = 0.0;
    in.x_range_hi  = 1.0;
    in.has_y_range = true;
    in.y_range_lo  = -1.0;
    in.y_range_hi  = 1.0;
    // A distinct value per cell, so a transposed or mis-ordered write shows up.
    in.outer = {10, 11, 12, 13, 0, 15, 16, 17, 18};

    const p::h5_save_data out = write_then_read(exec, in, "coast_h5_wr_h2d.h5");

    REQUIRE(out.has_x_range);
    CHECK(out.x_range_lo == doctest::Approx(0.0));
    CHECK(out.x_range_hi == doctest::Approx(1.0));
    REQUIRE(out.has_y_range);
    CHECK(out.y_range_lo == doctest::Approx(-1.0));
    CHECK(out.y_range_hi == doctest::Approx(1.0));
    // The ring survives, in order: a saved histogram2d has no raw points left to
    // recompute it from, so a lost ring is unrecoverable.
    REQUIRE(out.outer.size() == 9);
    for (std::size_t i = 0; i < 9; ++i) CHECK(out.outer[i] == doctest::Approx(in.outer[i]));
}

TEST_CASE("plot_h5: an auto-ranged histogram2d writes no domains and no ring")
{
    // Driven from a real figure rather than a hand-built h5_save_data: the ring
    // is fill_snapshot's to populate, so hand-building one cannot catch it
    // persisting a ring an auto-ranged figure can never fill.
    hdf5::executor exec;

    auto src = std::make_shared<p::histogram2d>("w2", 2, 2);
    const std::vector<double> xs{0.1, 0.9}, ys{0.1, 0.9};
    src->update(std::span<const double>(xs), std::span<const double>(ys));

    p::h5_save_data in;
    REQUIRE(src->snapshot_h5(in));
    CHECK(!in.has_x_range);
    CHECK(!in.has_y_range);
    CHECK(in.outer.empty()); // not a zeroed ring

    const p::h5_save_data out = write_then_read(exec, in, "coast_h5_wr_h2d_auto.h5");

    CHECK(!out.has_x_range);
    CHECK(!out.has_y_range);
    CHECK(out.outer.empty());
    // The frame still round-trips — it is the grid's extent, not a domain.
    REQUIRE(out.grid.size() == 4);
}

TEST_CASE("plot_h5: a histogram2d's own snapshot survives the writer/reader pair")
{
    // The 2D twin of the 1D end-to-end case: figure -> fill_snapshot ->
    // write_plot_into -> read_plot_from -> make_figure_from_h5. A histogram2d
    // saves only its binned grid, so its ring has no second source — if any seam
    // drops it, the outlier count is gone for good.
    hdf5::executor exec;

    auto src = std::make_shared<p::histogram2d>("src2", 2, 2, p::range{0.0, 1.0},
                                                p::range{0.0, 1.0});
    // Two inside, one under-x/over-y corner, one over-x: a ring that is neither
    // empty nor symmetric, so a transposed or dropped ring shows up.
    const std::vector<double> xs{0.25, 0.75, -5.0, 3.0};
    const std::vector<double> ys{0.25, 0.75, 7.0, 0.5};
    src->update(std::span<const double>(xs), std::span<const double>(ys));

    p::h5_save_data snap;
    REQUIRE(src->snapshot_h5(snap));
    REQUIRE(snap.has_x_range);
    REQUIRE(snap.has_y_range);
    REQUIRE(snap.outer.size() == 9); // real outliers => a ring is written

    const p::h5_save_data out = write_then_read(exec, snap, "coast_h5_wr_h2d_e2e.h5");

    auto f = p::make_figure_from_h5(out);
    REQUIRE(f != nullptr);
    auto h = std::dynamic_pointer_cast<p::histogram2d>(f);
    REQUIRE(h != nullptr);
    CHECK(h->x_domain() == p::range{0.0, 1.0});
    CHECK(h->y_domain() == p::range{0.0, 1.0});

    const p::outer_ring r = h->outer();
    CHECK(r.total() == doctest::Approx(2.0));
    CHECK(r.at(p::zone::over, p::zone::under) == doctest::Approx(1.0));  // (-5, 7)
    CHECK(r.at(p::zone::inside, p::zone::over) == doctest::Approx(1.0)); // (3, 0.5)
}

TEST_CASE("plot_h5: a figure's own snapshot survives the writer/reader pair")
{
    // End to end across every seam a live save touches except the GL thumbnail:
    // figure -> fill_snapshot -> write_plot_into -> read_plot_from ->
    // make_figure_from_h5. A domain lost at any one of them shows up here.
    hdf5::executor exec;

    auto src = std::make_shared<p::histogram>("src", 8, p::range{0.0, 1.0});
    std::vector<double> v{-3.0, 0.25, 0.75, 4.0};
    src->update(v);

    p::h5_save_data snap;
    REQUIRE(src->snapshot_h5(snap));
    CHECK(snap.has_x_range);

    const p::h5_save_data out = write_then_read(exec, snap, "coast_h5_wr_e2e.h5");

    auto f = p::make_figure_from_h5(out);
    REQUIRE(f != nullptr);
    auto h = std::dynamic_pointer_cast<p::histogram>(f);
    REQUIRE(h != nullptr);
    CHECK(h->domain() == p::range{0.0, 1.0});
    CHECK(h->underflow() == 1);
    CHECK(h->overflow() == 1);
}

TEST_CASE("histogram: a file with no domain attrs loads as auto-ranged")
{
    // Backward compatibility: every histogram saved before fixed domains existed
    // has no x_range_* attrs, and was auto-ranged.
    hdf5::executor exec;
    p::set_executor(exec);
    p::free_resources();

    const std::vector<double> ys{-3.0, 0.25, 4.0};
    const fs::path path = make_hist_file(exec, ys, std::nullopt, "coast_hist_norange_rt.h5");

    auto f = p::load(path);
    REQUIRE(f != nullptr);
    auto h = std::dynamic_pointer_cast<p::histogram>(f);
    REQUIRE(h != nullptr);

    CHECK(!h->domain().has_value());
    CHECK(h->underflow() == 0);
    CHECK(h->overflow() == 0);
    CHECK(h->point_count() == 3);

    std::error_code ec;
    fs::remove(path, ec);
}
