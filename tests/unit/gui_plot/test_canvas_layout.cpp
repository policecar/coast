// canvas::paint_body — the real painter, driven headlessly.
//
// canvas_view's math is covered by test_canvas_view.cpp; this covers the WIRING,
// which pure math cannot: that the transform is actually applied to the cells (and
// not, say, with x and y swapped), that zoom 1 still produces the classic grid,
// that off-viewport cells get culled, and that the wheel reaches the view.
//
// No SDL, no GL, no display: ImGui computes every rect during the frame, long
// before anything rasterizes, so a null backend is enough — and deterministic.
// (This is why the gui_plot CMakeLists no longer says layout is hand-verified only.)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "gui_plot/canvas.h"

#include "imgui.h"
#include "imgui_internal.h" // ImGuiWindow: Rect()/ClipRect, g.Windows
#include "implot.h"

#include <cstring>
#include <memory>
#include <string>

using coast::plot::canvas;
using coast::plot::region;

namespace {

// One ImGui+ImPlot context for the whole binary; doctest runs cases in one process.
struct gui_fixture {
    gui_fixture()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        // A real canvas hosts ImPlot figures, so it always has an ImPlot context;
        // omitting it here would test a configuration that never ships.
        ImPlot::CreateContext();
        ImGuiIO& io    = ImGui::GetIO();
        io.DisplaySize = ImVec2(1280.0f, 720.0f);
        io.DeltaTime   = 1.0f / 60.0f;
        io.IniFilename = nullptr;
        // Null backend still needs real font metrics for CalcTextSize.
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

// A hosted figure that records what its painter was actually handed. This is the
// cell-side view of the seam contract: the target dimension it receives, and what
// input state was left for it.
struct probe_figure final : public coast::plot::figure {
    explicit probe_figure(std::string t) : figure(std::move(t)) {}

    int   paints     = 0;
    float last_w     = 0.0f;
    float last_h     = 0.0f;
    float seen_wheel = -1.0f;

protected:
    void paint_body(region r) override
    {
        ++paints;
        last_w     = r.w;
        last_h     = r.h;
        seen_wheel = ImGui::GetIO().MouseWheel;
    }
};

constexpr float k_region_w = 600.0f;
constexpr float k_region_h = 400.0f;

// Input must go through the event queue, not by poking io.MouseDown/MousePos:
// NewFrame() rebuilds those from queued events, so direct writes get overwritten.
// Queue BEFORE the frame that should observe them.
void mouse_to(float x, float y) { ImGui::GetIO().AddMousePosEvent(x, y); }
void mouse_button(int b, bool down) { ImGui::GetIO().AddMouseButtonEvent(b, down); }
void mouse_wheel(float w) { ImGui::GetIO().AddMouseWheelEvent(0.0f, w); }

// Paint the canvas body into a region for one frame. interactive=false is the
// exporter's shape of the call (plot_capture.cpp). The region defaults to
// k_region_w/h; tests that probe the min-cell floor pass their own so they can put
// cells above or below it. The host window is large enough that a region up to
// ~1000 px is not clipped by it (culling must come from the canvas viewport, not
// the host).
void paint_frame(const std::shared_ptr<canvas>& cv, bool interactive = true,
                 float rw = k_region_w, float rh = k_region_h)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(1200.0f, 800.0f));
    ImGui::Begin("host", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    cv->body_painter()(region{rw, rh, interactive});
    ImGui::End();
    ImGui::Render();
}


// Cells are child windows named ".../##cell_<r>_<c>_<id>"; find one by substring.
ImGuiWindow* find_cell(int r, int c)
{
    char key[32];
    std::snprintf(key, sizeof key, "##cell_%d_%d", r, c);
    ImGuiContext& g     = *ImGui::GetCurrentContext();
    ImGuiWindow*  found = nullptr;
    for (ImGuiWindow* w : g.Windows)
        if (std::strstr(w->Name, key) != nullptr) {
            REQUIRE(found == nullptr); // the id must be unambiguous, or the test lies
            found = w;
        }
    return found;
}

// SkipItems is the culling signal, and the one that matters: it is what BeginChild
// returns (inverted), which is what paint_body branches on to skip a cell's body.
// NOT the clip rect — for a child entirely outside the viewport ImGui early-outs
// and leaves ClipRect as the parent's, rather than computing an empty intersection.
bool is_culled(const ImGuiWindow* w) { return w->SkipItems; }

} // namespace

TEST_CASE("canvas layout: with cells above the min-cell floor, zoom 1 is the classic grid")
{
    gui();
    // 900x400 over 2x3 keeps every cell (~295x196) above the 200x160 floor, so the
    // floor does not bind and the layout must be pixel-identical to the old
    // shrink-to-fit grid.
    constexpr float rw = 900.0f, rh = 400.0f;
    auto            cv = std::make_shared<canvas>("t", 2, 3);
    paint_frame(cv, true, rw, rh);
    paint_frame(cv, true, rw, rh); // child rects settle after the first frame

    const float pad = ImGui::GetStyle().ItemSpacing.x;
    const float bw  = (rw - pad * 2.0f) / 3.0f; // cols-1 gutters
    const float bh  = (rh - pad * 1.0f) / 2.0f; // rows-1 gutters

    ImGuiWindow* c00 = find_cell(0, 0);
    ImGuiWindow* c01 = find_cell(0, 1);
    ImGuiWindow* c10 = find_cell(1, 0);
    REQUIRE(c00 != nullptr);
    REQUIRE(c01 != nullptr);
    REQUIRE(c10 != nullptr);

    // Size: the untouched grid must be pixel-identical to the pre-zoom behaviour.
    CHECK(c00->Rect().GetWidth() == doctest::Approx(bw).epsilon(0.01));
    CHECK(c00->Rect().GetHeight() == doctest::Approx(bh).epsilon(0.01));

    // Stride: +1 col steps by (bw+pad) in x ONLY, +1 row by (bh+pad) in y ONLY.
    // This is what catches a transposed transform.
    CHECK(c01->Rect().Min.x - c00->Rect().Min.x == doctest::Approx(bw + pad).epsilon(0.01));
    CHECK(c01->Rect().Min.y - c00->Rect().Min.y == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(c10->Rect().Min.y - c00->Rect().Min.y == doctest::Approx(bh + pad).epsilon(0.01));
    CHECK(c10->Rect().Min.x - c00->Rect().Min.x == doctest::Approx(0.0f).epsilon(0.01));
}

TEST_CASE("canvas layout: past the floor, cells clamp to the minimum and the grid overflows")
{
    gui();
    // 600x400 over 2x3 would give ~195-wide cells; the 200x160 floor (canvas.cpp
    // k_pref_cell_w/h) clamps the width, so the grid outgrows the viewport. The
    // height is pure-fit (==400), so reserving the H-bar tips it over too and BOTH
    // bars show; we assert the vertical one, which shrinks the ##viewport width.
    constexpr float rw = 600.0f, rh = 400.0f;
    auto            cv = std::make_shared<canvas>("tfloor", 2, 3);
    paint_frame(cv, true, rw, rh);
    paint_frame(cv, true, rw, rh);

    // The cell clamps to the floor width, not the ~195 fit width.
    ImGuiWindow* c00 = find_cell(0, 0);
    REQUIRE(c00 != nullptr);
    CHECK(c00->Rect().GetWidth() == doctest::Approx(200.0f).epsilon(0.01)); // k_pref_cell_w

    // Overflow reserved a scrollbar: the viewport child (a cell's parent) is one bar
    // narrower than the region it was handed.
    ImGuiWindow* vp = c00->ParentWindow;
    REQUIRE(vp != nullptr);
    const float bar = ImGui::GetStyle().ScrollbarSize;
    CHECK(vp->Rect().GetWidth() == doctest::Approx(rw - bar).epsilon(0.01));
}

TEST_CASE("canvas layout: the wheel zooms, and zoom scales cells about the cursor")
{
    gui();
    auto cv = std::make_shared<canvas>("t2", 2, 2);
    paint_frame(cv);
    paint_frame(cv);
    REQUIRE(cv->zoom() == doctest::Approx(1.0f));

    ImGuiWindow* before = find_cell(0, 0);
    REQUIRE(before != nullptr);
    const float base_w = before->Rect().GetWidth();

    // Park the cursor inside the grid, let hover resolve (it is computed from the
    // previous frame's rects), then spin the wheel.
    mouse_to(100.0f, 100.0f);
    paint_frame(cv);
    mouse_wheel(3.0f);
    paint_frame(cv);
    paint_frame(cv);

    CHECK(cv->zoom() > 1.0f); // the wheel reached the view

    ImGuiWindow* after = find_cell(0, 0);
    REQUIRE(after != nullptr);
    // Cell geometry must track the zoom.
    CHECK(after->Rect().GetWidth() == doctest::Approx(base_w * cv->zoom()).epsilon(0.02));

    // ... and reset must put it back exactly.
    cv->reset_view();
    paint_frame(cv);
    paint_frame(cv);
    CHECK(cv->zoom() == doctest::Approx(1.0f));
    CHECK(find_cell(0, 0)->Rect().GetWidth() == doctest::Approx(base_w).epsilon(0.01));

    mouse_to(-FLT_MAX, -FLT_MAX); // leave the shared context clean for the next case
    paint_frame(cv);
}

TEST_CASE("canvas cells: hosted figure is handed the ZOOMED target dimension")
{
    gui();
    auto cv = std::make_shared<canvas>("t4", 1, 1);
    auto p  = std::make_shared<probe_figure>("p");
    cv->place(p, 0, 0);
    paint_frame(cv);
    paint_frame(cv);

    REQUIRE(p->paints > 0); // the canvas actually drives its occupant
    const float w_at_1 = p->last_w;
    const float h_at_1 = p->last_h;
    CHECK(w_at_1 > 0.0f);

    mouse_to(100.0f, 100.0f);
    paint_frame(cv);
    for (int i = 0; i < 8; ++i) { // zoom in a few ticks
        mouse_wheel(1.0f);
        paint_frame(cv);
    }
    REQUIRE(cv->zoom() > 1.0f);

    // The region handed to the cell must grow with the zoom -- this is the whole
    // "can a subplot receive a target dimension" question, answered against the
    // real painter rather than by inspection.
    CHECK(p->last_w > w_at_1);
    CHECK(p->last_h > h_at_1);
    CHECK(p->last_w == doctest::Approx(w_at_1 * cv->zoom()).epsilon(0.05));

    cv->reset_view();
    mouse_to(-FLT_MAX, -FLT_MAX);
    paint_frame(cv);
}

TEST_CASE("canvas cells: the canvas consumes the wheel so ImPlot cannot also zoom")
{
    gui();
    auto cv = std::make_shared<canvas>("t5", 1, 1);
    auto p  = std::make_shared<probe_figure>("p5");
    cv->place(p, 0, 0);
    mouse_to(100.0f, 100.0f);
    paint_frame(cv);
    paint_frame(cv);

    // Spin the wheel; the cell must observe a wheel of 0 -- otherwise a hosted
    // line/histogram (which take ImPlot's default persistent wheel-zoom, having no
    // SetupAxes call) would zoom its axes at the same time as the canvas.
    p->seen_wheel = -1.0f;
    mouse_wheel(2.0f);
    paint_frame(cv);
    REQUIRE(cv->zoom() > 1.0f); // the canvas did take the wheel...
    CHECK(p->seen_wheel == doctest::Approx(0.0f)); // ...and left none for the cell

    cv->reset_view();
    mouse_to(-FLT_MAX, -FLT_MAX);
    paint_frame(cv);
}

TEST_CASE("canvas cells: a cell too small for a plot is not painted at all")
{
    gui();
    // 8x8 over a 600x400 region: base cells clamp to the 200x160 min-cell floor;
    // zooming out to 0.2 takes them to 40x32, well under the ~48x40 mark where
    // ImPlot's chrome leaves no interior.
    auto cv = std::make_shared<canvas>("t6", 8, 8);
    auto p  = std::make_shared<probe_figure>("p6");
    cv->place(p, 0, 0);
    paint_frame(cv);
    paint_frame(cv);
    REQUIRE(p->paints > 0); // painted while big enough -- vacuous otherwise

    mouse_to(100.0f, 100.0f);
    paint_frame(cv);
    for (int i = 0; i < 40; ++i) { // zoom out to the floor
        mouse_wheel(-1.0f);
        paint_frame(cv);
    }
    REQUIRE(cv->zoom() < 0.5f);

    const int before = p->paints;
    paint_frame(cv);
    paint_frame(cv);
    CHECK(p->paints == before); // body skipped: no room for a plot

    cv->reset_view();
    mouse_to(-FLT_MAX, -FLT_MAX);
    paint_frame(cv);
}

TEST_CASE("canvas view: a capture during a middle-drag must not corrupt the pan")
{
    gui();
    // 3x3 over 600x400 floors cells to 200x160, so the grid overflows and pan has
    // room to move. (A fitting grid is clamped to pan 0 and cannot be dragged at
    // all — that is the point of clamp_pan.)
    constexpr float rw = 600.0f, rh = 400.0f;
    auto            cv = std::make_shared<canvas>("t7", 3, 3);
    paint_frame(cv, true, rw, rh);
    paint_frame(cv, true, rw, rh);
    ImGuiWindow* cell = find_cell(0, 0);
    REQUIRE(cell != nullptr);
    const float rest_x = cell->Rect().Min.x;

    // Latch a middle-drag and pull LEFT (toward showing the right-hand cells) — the
    // only direction an overflowing grid can pan, since pan is clamped to [min, 0].
    mouse_to(300.0f, 200.0f);
    paint_frame(cv, true, rw, rh);
    mouse_button(ImGuiMouseButton_Middle, true);
    paint_frame(cv, true, rw, rh);
    mouse_to(275.0f, 200.0f); // -25 px, within the clamp range
    paint_frame(cv, true, rw, rh);
    cell = find_cell(0, 0);
    REQUIRE(cell != nullptr);
    const float dragged_x = cell->Rect().Min.x;
    REQUIRE(dragged_x < rest_x - 15.0f); // the drag moved the grid left — vacuous otherwise

    // A capture fires mid-drag: the exporter parks the cursor but the button
    // stays down, and — the nasty half of the edge — the RELEASE lands while the
    // frames are non-interactive, so no interactive frame ever sees the button
    // again. Without the gate, pan absorbs the parked -FLT_MAX and stays -inf.
    mouse_to(-FLT_MAX, -FLT_MAX);
    paint_frame(cv, /*interactive=*/false, rw, rh);
    mouse_button(ImGuiMouseButton_Middle, false);
    paint_frame(cv, /*interactive=*/false, rw, rh);

    // Back live: the grid must sit exactly where the drag left it.
    mouse_to(300.0f, 200.0f);
    paint_frame(cv, true, rw, rh);
    paint_frame(cv, true, rw, rh);
    cell = find_cell(0, 0);
    REQUIRE(cell != nullptr);
    CHECK(cell->Rect().Min.x == doctest::Approx(dragged_x).epsilon(0.01));

    mouse_to(-FLT_MAX, -FLT_MAX);
    paint_frame(cv, true, rw, rh);
}

TEST_CASE("canvas layout: scrolling an overflowing grid culls the cells past the edge")
{
    gui();
    // 6x6 floored to 200x160 is a 1240x1000 grid in a ~586x386 viewport, so
    // scrolling to the far corner pushes the top-left cell fully out of view.
    // ('Drag the whole grid off' is impossible now — clamp_pan keeps content in
    // view — so culling instead happens for cells scrolled past the viewport edge.)
    constexpr float rw = 600.0f, rh = 400.0f;
    auto            cv = std::make_shared<canvas>("t3", 6, 6);
    paint_frame(cv, true, rw, rh);
    paint_frame(cv, true, rw, rh);
    REQUIRE(!is_culled(find_cell(0, 0))); // top-left visible at rest — vacuous otherwise
    REQUIRE(is_culled(find_cell(5, 5)));  // bottom-right already off-screen at rest

    // Middle-drag hard toward the upper-left; clamp pins pan at the far (max-scroll)
    // corner, which shows the bottom-right of the grid.
    mouse_to(300.0f, 200.0f);
    paint_frame(cv, true, rw, rh); // let hover resolve over the viewport
    mouse_button(ImGuiMouseButton_Middle, true);
    paint_frame(cv, true, rw, rh); // click latches the anchor
    mouse_to(-5000.0f, -5000.0f);
    paint_frame(cv, true, rw, rh); // drag past the clamp; pan pins at max scroll
    mouse_button(ImGuiMouseButton_Middle, false);
    paint_frame(cv, true, rw, rh);

    // Top-left cell is now scrolled fully off the left edge and culled...
    CHECK(find_cell(0, 0)->Rect().Max.x < 0.0f);
    CHECK(is_culled(find_cell(0, 0)));
    // ...while the bottom-right cell has scrolled into view.
    CHECK(!is_culled(find_cell(5, 5)));

    mouse_to(-FLT_MAX, -FLT_MAX);
    paint_frame(cv, true, rw, rh);
}

TEST_CASE("canvas layout: a non-interactive (export) frame skips the floor and frames the whole grid")
{
    gui();
    // interactive=false is the exporter's shape (plot_capture.cpp). Even at 600x400
    // over 2x3 — where an interactive frame floors cells to 200 and scrolls — an
    // export must stay pure fit-to-region (~195-wide cells, whole grid framed), or
    // thumbnails clip to the top-left. This pins that the floor is chrome-gated.
    constexpr float rw = 600.0f, rh = 400.0f;
    auto            cv = std::make_shared<canvas>("texport", 2, 3);
    paint_frame(cv, /*interactive=*/false, rw, rh);
    paint_frame(cv, /*interactive=*/false, rw, rh);

    const float  pad   = ImGui::GetStyle().ItemSpacing.x;
    const float  fit_w = (rw - pad * 2.0f) / 3.0f; // ~194.7, below the 200 floor
    ImGuiWindow* c00   = find_cell(0, 0);
    REQUIRE(c00 != nullptr);
    CHECK(c00->Rect().GetWidth() == doctest::Approx(fit_w).epsilon(0.01)); // fit, not floored
    CHECK(c00->Rect().GetWidth() < 200.0f);                               // explicitly below the floor
    // No reserve on export: the viewport child fills the whole region width.
    CHECK(c00->ParentWindow->Rect().GetWidth() == doctest::Approx(rw).epsilon(0.01));
}

TEST_CASE("canvas scrollbars: dragging the vertical bar scrolls the grid")
{
    gui();
    // 3x3 @ 600x400 floors to 200x160 and overflows both axes, so both bars show.
    // Grab the vertical bar near the top of its strip (the thumb sits there at
    // scroll 0) and drag down; the grid must scroll up under it. This is the only
    // end-to-end exercise of draw_scrollbars — a wrong bb/axis/sign would ship green.
    constexpr float rw = 600.0f, rh = 400.0f;
    auto            cv = std::make_shared<canvas>("tbar", 3, 3);
    paint_frame(cv, true, rw, rh);
    paint_frame(cv, true, rw, rh);
    ImGuiWindow* c00 = find_cell(0, 0);
    REQUIRE(c00 != nullptr);
    const float rest_y = c00->Rect().Min.y;

    // The vertical strip sits just right of the viewport child (its parent).
    const float  bar     = ImGui::GetStyle().ScrollbarSize;
    ImGuiWindow* vp      = c00->ParentWindow;
    const float  strip_x = vp->Rect().Max.x + bar * 0.5f;
    const float  top_y   = vp->Rect().Min.y + 4.0f;

    mouse_to(strip_x, top_y);
    paint_frame(cv, true, rw, rh); // hover resolves over the bar
    mouse_button(ImGuiMouseButton_Left, true);
    paint_frame(cv, true, rw, rh); // press on the thumb
    mouse_to(strip_x, top_y + 150.0f);
    paint_frame(cv, true, rw, rh); // drag down
    mouse_button(ImGuiMouseButton_Left, false);
    paint_frame(cv, true, rw, rh);

    c00 = find_cell(0, 0);
    REQUIRE(c00 != nullptr);
    CHECK(c00->Rect().Min.y < rest_y - 15.0f); // grid scrolled up under the bar

    mouse_to(-FLT_MAX, -FLT_MAX);
    paint_frame(cv, true, rw, rh);
}
