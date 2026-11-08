// The console-plot render cache: histogram2d/pixels compute their render
// products (the binned grid, its log view, the candle box stats, the reshaped
// field) once per update() instead of once per frame.
//
// What is observable headlessly is the CONTRACT the cache must not break: a
// product must never outlive the data it was derived from, and it must not be
// consumed by a reader. Painting needs a GUI context, so the cached grid is
// reached through outer() (which reads the same cached binning) and through
// snapshot_h5(). compute_candles is pure computation and is tested directly.
//
// The cache is a performance change; every case here fails if the invalidation
// or the ownership is wrong, and none of them care how fast it is.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "gui_plot/console_plot.h"
#include "gui_plot/plot_detail.h" // make_figure_from_h5 (the loader's adopt path)
#include "gui_plot/plot_tools.h"  // bin_points + compute_candles

#include <memory>
#include <numeric>
#include <span>
#include <vector>

namespace p = coast::plot;

namespace {

double grid_total(const p::grid_result& g)
{
    return std::accumulate(g.grid.begin(), g.grid.end(), 0.0);
}

} // namespace

// ------------------------------------------------------------- invalidation

TEST_CASE("histogram2d: a second update() re-bins rather than serving the first")
{
    // The regression the cache introduces if update() forgets to drop it: every
    // sample after the first renders as the one before it. outer() reads the same
    // cached binning the painters do, so a stale grid shows up here.
    auto h = std::make_shared<p::histogram2d>("cache", 4, 4, p::range{0.0, 1.0},
                                              p::range{0.0, 1.0});

    const std::vector<double> x1{0.25, 0.75}, y1{0.25, 0.75}; // both inside
    h->update(std::span<const double>(x1), std::span<const double>(y1));
    REQUIRE(h->outer().total() == doctest::Approx(0.0));

    // A sample with two outliers. A stale cache still reads 0.
    const std::vector<double> x2{0.5, 9.0, -9.0}, y2{0.5, 0.5, 0.5};
    h->update(std::span<const double>(x2), std::span<const double>(y2));
    CHECK(h->outer().total() == doctest::Approx(2.0));
    CHECK(h->outer().at(p::zone::inside, p::zone::over) == doctest::Approx(1.0));
    CHECK(h->outer().at(p::zone::inside, p::zone::under) == doctest::Approx(1.0));

    // ...and back the other way: a sample with no outliers must not inherit the
    // previous ring either.
    h->update(std::span<const double>(x1), std::span<const double>(y1));
    CHECK(h->outer().total() == doctest::Approx(0.0));
}

TEST_CASE("histogram2d: repeated reads of unchanged data agree")
{
    // The other half of the contract: caching must be invisible. Reading twice
    // without an update in between must give the same answer, and the first read
    // must not consume the product.
    auto h = std::make_shared<p::histogram2d>("stable", 4, 4, p::range{0.0, 1.0},
                                              p::range{0.0, 1.0});
    const std::vector<double> x{0.5, 5.0}, y{0.5, 0.5};
    h->update(std::span<const double>(x), std::span<const double>(y));

    const p::outer_ring a = h->outer();
    const p::outer_ring b = h->outer();
    CHECK(a.total() == doctest::Approx(b.total()));
    CHECK(b.total() == doctest::Approx(1.0));
    CHECK(h->point_count() == 2);
}

TEST_CASE("histogram2d: a loaded figure's grid is not re-derived from stale points")
{
    // update() then load-style adoption and back: the two data sources (raw
    // points vs a pre-binned grid) must not cross-contaminate through the cache.
    // Reached via the real loader, which is the only adopter.
    p::h5_save_data d;
    d.kind  = "histogram2d";
    d.title = "loaded";
    d.nbx = 2, d.nby = 2;
    d.grid   = {1.0, 2.0, 3.0, 4.0};
    d.x_min = 0.0, d.x_max = 1.0, d.y_min = 0.0, d.y_max = 1.0;
    d.mode     = "heatmap";
    d.colormap = "Viridis";
    d.has_x_range = true, d.x_range_lo = 0.0, d.x_range_hi = 1.0;
    d.has_y_range = true, d.y_range_lo = 0.0, d.y_range_hi = 1.0;
    d.outer = {0, 0, 0, 0, 0, 7, 0, 0, 0}; // one populated ring cell (inside/over)

    auto f = p::make_figure_from_h5(d);
    REQUIRE(f != nullptr);
    auto h = std::dynamic_pointer_cast<p::histogram2d>(f);
    REQUIRE(h != nullptr);
    // The loaded figure carries the saved ring; nothing is re-binned.
    CHECK(h->outer().total() == doctest::Approx(7.0));
    CHECK(h->point_count() == 4); // grid cells

    // update() switches it back to live points: the loaded ring must not survive.
    const std::vector<double> x{0.5}, y{0.5};
    h->update(std::span<const double>(x), std::span<const double>(y));
    CHECK(h->outer().total() == doctest::Approx(0.0));
    CHECK(h->point_count() == 1);
}

// -------------------------------------------------------------- ownership

TEST_CASE("histogram2d: snapshot_h5 copies the cache instead of consuming it")
{
    // fill_snapshot used to move the grid out of a freshly computed temporary.
    // Against a cache that move guts the figure, and the next frame paints an
    // empty plot. Two snapshots in a row must agree.
    auto                      h = std::make_shared<p::histogram2d>("snap", 2, 2);
    const std::vector<double> x{0.1, 0.9, 0.5}, y{0.1, 0.9, 0.5};
    h->update(std::span<const double>(x), std::span<const double>(y));

    p::h5_save_data a, b;
    REQUIRE(h->snapshot_h5(a));
    REQUIRE(h->snapshot_h5(b));
    REQUIRE(a.grid.size() == 4);
    CHECK(b.grid == a.grid); // the second snapshot is not empty/zeroed
    // The figure itself is still intact after being snapshotted.
    CHECK(h->point_count() == 3);
}

TEST_CASE("pixels: snapshot_h5 copies the cache instead of consuming it")
{
    auto                      px = std::make_shared<p::pixels>("snap", 2, 2);
    const std::vector<double> v{1.0, 2.0, 3.0, 4.0};
    px->update(std::span<const double>(v));

    p::h5_save_data a, b;
    REQUIRE(px->snapshot_h5(a));
    REQUIRE(px->snapshot_h5(b));
    REQUIRE(a.grid.size() == 4);
    CHECK(b.grid == a.grid);
    CHECK(px->point_count() == 4);
}

TEST_CASE("pixels: a second update() reshapes rather than serving the first")
{
    auto px = std::make_shared<p::pixels>("cache", 2, 2);

    const std::vector<double> a{1.0, 1.0, 1.0, 1.0};
    px->update(std::span<const double>(a));
    p::h5_save_data s1;
    REQUIRE(px->snapshot_h5(s1));
    CHECK(s1.grid == std::vector<double>{1.0, 1.0, 1.0, 1.0});

    const std::vector<double> b{5.0, 6.0, 7.0, 8.0};
    px->update(std::span<const double>(b));
    p::h5_save_data s2;
    REQUIRE(px->snapshot_h5(s2));
    CHECK(s2.grid == std::vector<double>{5.0, 6.0, 7.0, 8.0}); // stale cache: all 1.0
}

// -------------------------------------------------------- compute_candles

TEST_CASE("compute_candles: quartiles come from the column's binned y-distribution")
{
    // Four points in one column, all in the bottom y-bin [0, 0.25). The stats are
    // interpolated WITHIN that bin (weighted quantile), so they must land inside
    // it rather than snapping to a bin edge.
    const std::vector<double> x{0.25, 0.25, 0.25, 0.25};
    const std::vector<double> y{0.1, 0.1, 0.1, 0.1};
    const p::grid_result      g =
        p::bin_points(x, y, 2, 4, p::range{0.0, 1.0}, p::range{0.0, 1.0});
    REQUIRE(grid_total(g) == doctest::Approx(4.0));

    const std::vector<p::candle_box> boxes = p::compute_candles(g);

    // Column 1 is empty and yields no box — a candle there would assert a
    // distribution that was never measured.
    REQUIRE(boxes.size() == 1);
    const p::candle_box& b = boxes.front();
    CHECK(b.x == doctest::Approx(0.25));    // column 0's center, binw = 0.5
    CHECK(b.q1 == doctest::Approx(0.0625)); // target 1/4 of the way into the bin
    CHECK(b.med == doctest::Approx(0.125)); // half way
    CHECK(b.q3 == doctest::Approx(0.1875));
    CHECK(b.q1 <= b.med);
    CHECK(b.med <= b.q3);
    CHECK(b.lo <= b.q1);
    CHECK(b.hi >= b.q3);
}

TEST_CASE("compute_candles: one box per populated column, in column order")
{
    // Two populated columns of different heights: a transposed grid index or an
    // off-by-one in the row flip puts the boxes at the wrong x or the wrong y.
    const std::vector<double> x{0.1, 0.1, 0.9};
    const std::vector<double> y{0.1, 0.2, 0.9};
    const p::grid_result      g =
        p::bin_points(x, y, 2, 4, p::range{0.0, 1.0}, p::range{0.0, 1.0});

    const std::vector<p::candle_box> boxes = p::compute_candles(g);

    REQUIRE(boxes.size() == 2);
    CHECK(boxes[0].x == doctest::Approx(0.25)); // left column, low ys
    CHECK(boxes[1].x == doctest::Approx(0.75)); // right column, high y
    CHECK(boxes[0].med < 0.5);
    CHECK(boxes[1].med > 0.5);
}

TEST_CASE("compute_candles: an empty grid yields no boxes rather than a crash")
{
    CHECK(p::compute_candles(p::grid_result{}).empty());
}
