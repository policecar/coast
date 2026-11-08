// draw_candles — axis framing against a FIXED domain.
//
// A histogram2d built with explicit domains (cp::range(0,1)) promises a fixed
// frame: that is the whole point of supplying one, and it is what makes a grid of
// them comparable cell-to-cell. The heatmap path keeps that promise by handing
// gr.x0..gr.x1 to PlotHeatmap as bounds. This pins the candle path to the same
// contract, since it draws its boxes by hand and fits the axes itself.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "gui_plot/plot_tools.h"

#include "imgui.h"
#include "implot.h"
#include "implot_internal.h" // GetPlot: read back the axis range the fit produced

#include <vector>

using coast::plot::bin_points;
using coast::plot::draw_candles;
using coast::plot::grid_result;
using coast::plot::range;

namespace {

struct gui_fixture {
    gui_fixture()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO& io    = ImGui::GetIO();
        io.DisplaySize = ImVec2(1280.0f, 720.0f);
        io.DeltaTime   = 1.0f / 60.0f;
        io.IniFilename = nullptr;
        unsigned char* px = nullptr;
        int            w = 0, h = 0;
        io.Fonts->GetTexDataAsRGBA32(&px, &w, &h);
        io.Fonts->SetTexID(static_cast<ImTextureID>(1));
    }
    ~gui_fixture()
    {
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
};
gui_fixture& gui()
{
    static gui_fixture f;
    return f;
}

// Points filling the band [lo,hi] on both axes, inside the fixed domain [0,1].
grid_result grid_over_band(double lo, double hi)
{
    std::vector<double> xs, ys;
    for (int i = 0; i <= 40; ++i) {
        const double t = lo + (hi - lo) * (static_cast<double>(i) / 40.0);
        xs.push_back(t);
        ys.push_back(t);
    }
    return bin_points(xs, ys, 64, 64, range(0.0, 1.0), range(0.0, 1.0));
}

struct axis_range {
    double x_min, x_max, y_min, y_max;
};

// Draw the candles and read back the range ImPlot settled on. GetPlot resolves its
// id against the CURRENT window, so it must be called with `host` still current.
// Two frames: ImPlot applies an AutoFit range the frame after the data is fitted.
axis_range candle_axes(const grid_result& gr)
{
    axis_range out{};
    const std::vector<coast::plot::candle_box> boxes = coast::plot::compute_candles(gr);
    for (int frame = 0; frame < 3; ++frame) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(800.0f, 600.0f));
        ImGui::Begin("host", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar);
        draw_candles(gr, boxes, ImVec2(400.0f, 300.0f), true, true);
        if (ImPlotPlot* p = ImPlot::GetPlot("##candle")) {
            out.x_min = p->Axes[ImAxis_X1].Range.Min;
            out.x_max = p->Axes[ImAxis_X1].Range.Max;
            out.y_min = p->Axes[ImAxis_Y1].Range.Min;
            out.y_max = p->Axes[ImAxis_Y1].Range.Max;
        }
        ImGui::End();
        ImGui::Render();
    }
    return out;
}

} // namespace

TEST_CASE("draw_candles: a fixed domain frames the x axis, whatever the data covers")
{
    gui();
    // Same fixed domain, same 64 bins, different populated spans -- exactly a grid
    // of per-epoch histograms whose data drifts.
    const axis_range narrow = candle_axes(grid_over_band(0.40, 0.60));
    const axis_range wide   = candle_axes(grid_over_band(0.10, 0.90));

    CAPTURE(narrow.x_min);
    CAPTURE(narrow.x_max);
    CAPTURE(wide.x_min);
    CAPTURE(wide.x_max);

    // The frame must be the DECLARED domain, not the populated columns.
    CHECK(narrow.x_min == doctest::Approx(0.0).epsilon(0.02));
    CHECK(narrow.x_max == doctest::Approx(1.0).epsilon(0.02));
    CHECK(wide.x_min == doctest::Approx(0.0).epsilon(0.02));
    CHECK(wide.x_max == doctest::Approx(1.0).epsilon(0.02));

    // ... which is what makes two cells of a canvas comparable: same domain in,
    // same axis out, so a bin is the same width in both.
    CHECK(narrow.x_min == doctest::Approx(wide.x_min).epsilon(0.02));
    CHECK(narrow.x_max == doctest::Approx(wide.x_max).epsilon(0.02));
}

TEST_CASE("draw_candles: a fixed domain frames the y axis too")
{
    gui();
    const axis_range narrow = candle_axes(grid_over_band(0.40, 0.60));
    const axis_range wide   = candle_axes(grid_over_band(0.10, 0.90));

    CAPTURE(narrow.y_min);
    CAPTURE(narrow.y_max);
    CHECK(narrow.y_min == doctest::Approx(0.0).epsilon(0.02));
    CHECK(narrow.y_max == doctest::Approx(1.0).epsilon(0.02));
    CHECK(narrow.y_min == doctest::Approx(wide.y_min).epsilon(0.02));
    CHECK(narrow.y_max == doctest::Approx(wide.y_max).epsilon(0.02));
}

TEST_CASE("draw_candles: with no fixed domain the frame follows the data")
{
    gui();
    // bin_points with no domain derives the frame from the data's min/max, and the
    // candle axes must follow THAT -- the fix must not hard-code [0,1] thinking.
    std::vector<double> xs, ys;
    for (int i = 0; i <= 20; ++i) {
        xs.push_back(2.0 + 0.1 * static_cast<double>(i)); // 2.0 .. 4.0
        ys.push_back(5.0 + 0.2 * static_cast<double>(i)); // 5.0 .. 9.0
    }
    const grid_result gr = bin_points(xs, ys, 16, 16);
    const axis_range  a  = candle_axes(gr);
    CAPTURE(a.x_min);
    CAPTURE(a.x_max);
    CHECK(a.x_min == doctest::Approx(gr.x0).epsilon(0.02));
    CHECK(a.x_max == doctest::Approx(gr.x1).epsilon(0.02));
    CHECK(a.y_min == doctest::Approx(gr.y0).epsilon(0.02));
    CHECK(a.y_max == doctest::Approx(gr.y1).epsilon(0.02));
}
