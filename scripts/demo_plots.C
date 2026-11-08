// demo_plots.C — a tour of the COAST console plot types (coast::plot).
//
// Load + run from the console:
//     .x scripts/demo_plots.C
// This runs demo_plots(), which just prints the menu below. The file's
// declarations stay loaded, so you can then call any single demo directly, e.g.
//     demo_histogram();
//     demo_canvas();
//     demo_all();
//
// Every demo builds its own synthetic data (fixed seed → reproducible) and opens
// a live ImPlot window: construct the figure with std::make_shared, then show()
// it (that registers it with the render loop — nothing appears until show()).
// Handles are intentionally not returned: once shown, a plot persists as long as
// its window is open (the registry keeps it alive); closing the window hides it.
// Each figure carries Save PNG… / Save H5… and a theme picker.

#include "gui_plot/console_plot.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace cp = coast::plot;

// ---- small synthetic-data helpers -----------------------------------------

// n Gaussian samples (fixed seed so the plots look the same every run).
static std::vector<double> gaussian(int n, double mean, double sigma, std::uint32_t seed)
{
    std::mt19937                     rng(seed);
    std::normal_distribution<double> nd(mean, sigma);
    std::vector<double>              v(static_cast<std::size_t>(n));
    for (double& x : v) x = nd(rng);
    return v;
}

// ---- individual demos ------------------------------------------------------

// 1D histogram: the distribution of a Gaussian sample.
void demo_histogram()
{
    auto h = std::make_shared<cp::histogram>("demo: gaussian histogram", 64);
    h->show();
    h->update(gaussian(20000, 0.0, 1.0, 1));
}

// Line plot: a decaying sine, y-vs-index; plus a second series with explicit x.
void demo_line()
{
    const int           n = 400;
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * 0.05;
        y[static_cast<std::size_t>(i)] = std::sin(t) * std::exp(-t * 0.05);
    }
    auto l = std::make_shared<cp::line>("demo: damped sine");
    l->show();
    l->update(y);
}

// 2D histogram (density heatmap) of a correlated Gaussian cloud. Try the
// palette / log-density / axis toggles, and the heatmap↔candle "view" switch.
void demo_histogram2d()
{
    const int           n = 40000;
    std::vector<double> xs(static_cast<std::size_t>(n)), ys(static_cast<std::size_t>(n));
    std::mt19937        rng(2);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        const double a = nd(rng), b = nd(rng);
        xs[static_cast<std::size_t>(i)] = a;
        ys[static_cast<std::size_t>(i)] = 0.6 * a + 0.8 * b; // correlated
    }
    auto h2 = std::make_shared<cp::histogram2d>("demo: 2D gaussian");
    h2->show();
    h2->update(xs, ys);
}

// Candle/box plot (ROOT CANDLE sense): one box-and-whisker per x-column, from
// that column's y-distribution. Here each of 8 groups has a shifting mean and
// spread. Same data model as histogram2d — this is its candle "view".
void demo_candle()
{
    const int                        groups = 8, per = 4000;
    std::vector<double>              xs, ys;
    xs.reserve(static_cast<std::size_t>(groups) * per);
    ys.reserve(static_cast<std::size_t>(groups) * per);
    std::mt19937 rng(3);
    for (int g = 0; g < groups; ++g) {
        const double mean  = static_cast<double>(g);
        const double sigma = 0.4 + 0.12 * static_cast<double>(g);
        std::normal_distribution<double> nd(mean, sigma);
        for (int i = 0; i < per; ++i) {
            xs.push_back(static_cast<double>(g));
            ys.push_back(nd(rng));
        }
    }
    // A candle plot is a histogram2d in candle view: default the y-resolution
    // finer (50) than Sturges so the per-column quartiles are not overly coarse.
    auto c = std::make_shared<cp::histogram2d>("demo: candles", 0, 50);
    c->set_candle(true);
    c->show();
    c->update(xs, ys);
}

// Pixel/matrix view: a value field drawn as a colormapped image with square
// pixels — here a 28×28 Gaussian bump (an MNIST-sized input). Feed a raw vector
// via pixels(title, width, height).
void demo_pixels()
{
    const int           w = 28, h = 28;
    std::vector<double> field(static_cast<std::size_t>(w) * h);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            const double dr = r - 13.5, dc = c - 13.5;
            field[static_cast<std::size_t>(r) * w + c] = std::exp(-(dr * dr + dc * dc) / 40.0);
        }
    auto p = std::make_shared<cp::pixels>("demo: 28x28 bump", w, h);
    p->show();
    p->update(field); // row 0 = top of the image
}

// Image view: a raw RGBA gradient uploaded to a texture (the generic "something
// produced a picture" channel).
void demo_image()
{
    const int                  w = 160, h = 120;
    std::vector<std::uint8_t>  px(static_cast<std::size_t>(w) * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            std::uint8_t* p = &px[(static_cast<std::size_t>(y) * w + x) * 4];
            p[0] = static_cast<std::uint8_t>(255 * x / (w - 1)); // R ramps across
            p[1] = static_cast<std::uint8_t>(255 * y / (h - 1)); // G ramps down
            p[2] = 128;
            p[3] = 255;
        }
    auto im = std::make_shared<cp::image>("demo: RGBA gradient");
    im->show();
    im->set_rgba(px, w, h);
}

// Theming: the same histogram under the built-in "bright" theme (white plot
// background, dark axes). Use each plot's theme picker to switch, and the
// "edit…" button to open the live Plot Theme Editor (themes persist to disk).
void demo_theme()
{
    auto h = std::make_shared<cp::histogram>("demo: bright theme", 64);
    h->set_theme("bright");
    h->show();
    h->update(gaussian(20000, 0.0, 1.0, 4));
    std::printf("demo_theme: set theme 'bright'. Use the plot's theme combo\n"
                "            + 'edit…' button to open the Plot Theme Editor.\n");
}

// Canvas (TPad-style): one window subdivided into a 2×3 grid, each cell hosting
// a different vis-item instead of its own window.
//
// Only the canvas is show()n. Each cell item is constructed but NOT shown — it is
// place()d into a cell, so it only ever appears inside the canvas, never as its
// own window. update() changes data only, so the create/place/update order here
// is a matter of taste.
void demo_canvas()
{
    auto cv = std::make_shared<cp::canvas>("demo: dashboard", 2, 3);
    cv->show();

    auto h = std::make_shared<cp::histogram>("hist", 48);
    cv->place(h, 0, 0);
    h->update(gaussian(15000, 0.0, 1.0, 5));

    auto l = std::make_shared<cp::line>("sine");
    cv->place(l, 0, 1);
    const int           n = 300;
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) y[static_cast<std::size_t>(i)] = std::sin(i * 0.06);
    l->update(y);

    std::vector<double> xs(20000), ys(20000);
    std::mt19937        rng(6);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double a = nd(rng), b = nd(rng);
        xs[i] = a;
        ys[i] = 0.6 * a + 0.8 * b;
    }
    auto h2 = std::make_shared<cp::histogram2d>("heat", 0, 0);
    cv->place(h2, 0, 2);
    h2->update(xs, ys);

    auto cd = std::make_shared<cp::histogram2d>("candles", 0, 50);
    cd->set_candle(true);
    cv->place(cd, 1, 0);
    cd->update(xs, ys);

    const int           pw = 28, ph = 28;
    std::vector<double> field(static_cast<std::size_t>(pw) * ph);
    for (int r = 0; r < ph; ++r)
        for (int c = 0; c < pw; ++c) {
            const double dr = r - 13.5, dc = c - 13.5;
            field[static_cast<std::size_t>(r) * pw + c] = std::exp(-(dr * dr + dc * dc) / 40.0);
        }
    auto px = std::make_shared<cp::pixels>("bump", pw, ph);
    cv->place(px, 1, 1);
    px->update(field);

    const int                 iw = 96, ih = 96;
    std::vector<std::uint8_t> img(static_cast<std::size_t>(iw) * ih * 4);
    for (int y2 = 0; y2 < ih; ++y2)
        for (int x2 = 0; x2 < iw; ++x2) {
            std::uint8_t* p = &img[(static_cast<std::size_t>(y2) * iw + x2) * 4];
            p[0] = static_cast<std::uint8_t>(255 * x2 / (iw - 1));
            p[1] = static_cast<std::uint8_t>(255 * y2 / (ih - 1));
            p[2] = 160;
            p[3] = 255;
        }
    auto im = std::make_shared<cp::image>("img");
    cv->place(im, 1, 2);
    im->set_rgba(img, iw, ih);
}

// Open every demo at once.
void demo_all()
{
    demo_histogram();
    demo_line();
    demo_histogram2d();
    demo_candle();
    demo_pixels();
    demo_image();
    demo_theme();
    demo_canvas();
}

// Entry point run by `.x scripts/demo_plots.C` (the console calls the function
// named after the file). Prints the menu; call any listed function directly.
void demo_plots()
{
    std::printf("demo_plots — COAST console plot demos\n");
    std::printf("  the file is now loaded; call any of these in the console:\n");
    std::printf("    demo_histogram()    1D histogram of Gaussian samples\n");
    std::printf("    demo_line()         line plot (damped sine)\n");
    std::printf("    demo_histogram2d()  2D density heatmap (correlated cloud)\n");
    std::printf("    demo_candle()       candle/box plot per x-column\n");
    std::printf("    demo_pixels()       28x28 value field as a pixel image\n");
    std::printf("    demo_image()        raw RGBA image via the image kind\n");
    std::printf("    demo_theme()        histogram under the 'bright' theme\n");
    std::printf("    demo_canvas()       2x3 canvas hosting all of the above\n");
    std::printf("    demo_all()          open every demo at once\n");
}
