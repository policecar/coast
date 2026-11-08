// canvas_view — the canvas's pan/zoom transform.
//
// The transform is deliberately pure and ImGui-free, so it is tested here
// directly rather than "by hand in the GUI". The property that matters is the one
// graph_canvas gets wrong (graph_canvas.cpp:220-226 computes the cursor anchor and
// then discards it): zooming must not move the world point under the cursor.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "gui_plot/canvas.h"

#include <cmath>

using coast::plot::canvas_view;
using coast::plot::compute_reserve;

namespace {
// Screen-space tolerance. The invariant is about pixels, so judge it in pixels:
// a drift under a hundredth of a pixel is invisible and is float noise, not error.
constexpr float k_eps = 0.01f;
} // namespace

TEST_CASE("canvas_view: default is the identity view")
{
    const canvas_view v;
    CHECK(v.zoom == doctest::Approx(1.0f));
    CHECK(v.pan_x == doctest::Approx(0.0f));
    CHECK(v.pan_y == doctest::Approx(0.0f));
    // At zoom 1 / pan 0 the transform must be exactly identity, or every existing
    // canvas silently shifts.
    CHECK(v.to_screen_x(123.5f) == doctest::Approx(123.5f));
    CHECK(v.to_screen_y(-7.25f) == doctest::Approx(-7.25f));
}

TEST_CASE("canvas_view: world/screen round-trip")
{
    for (float zoom : {0.25f, 0.5f, 1.0f, 2.0f, 7.5f}) {
        for (float pan : {-320.0f, -1.5f, 0.0f, 42.0f}) {
            canvas_view v;
            v.zoom  = zoom;
            v.pan_x = pan;
            v.pan_y = -pan;
            for (float w : {-500.0f, 0.0f, 13.0f, 1024.0f}) {
                CAPTURE(zoom);
                CAPTURE(pan);
                CAPTURE(w);
                CHECK(v.to_world_x(v.to_screen_x(w)) == doctest::Approx(w).epsilon(1e-4));
                CHECK(v.to_world_y(v.to_screen_y(w)) == doctest::Approx(w).epsilon(1e-4));
            }
        }
    }
}

TEST_CASE("canvas_view: zoom_by keeps the point under the cursor fixed")
{
    // The whole point of anchored zoom. Sweep cursor positions (including the
    // origin and negative/off-viewport ones) and zoom factors in both directions.
    for (float mx : {0.0f, 1.0f, 200.0f, 799.0f, -50.0f}) {
        for (float my : {0.0f, 350.0f, -20.0f}) {
            for (float factor : {1.1f, 0.9f, 2.0f, 0.5f, 1.001f}) {
                for (float start_zoom : {0.5f, 1.0f, 3.0f}) {
                    canvas_view v;
                    v.zoom  = start_zoom;
                    v.pan_x = 17.0f;
                    v.pan_y = -9.0f;

                    const float world_x_before = v.to_world_x(mx);
                    const float world_y_before = v.to_world_y(my);
                    v.zoom_by(factor, mx, my);
                    const float world_x_after = v.to_world_x(mx);
                    const float world_y_after = v.to_world_y(my);

                    CAPTURE(mx);
                    CAPTURE(my);
                    CAPTURE(factor);
                    CAPTURE(start_zoom);
                    // Compare in SCREEN space: a world-space epsilon would be
                    // unfairly lenient at high zoom and unfairly strict at low.
                    CHECK(std::fabs(world_x_after - world_x_before) * v.zoom < k_eps);
                    CHECK(std::fabs(world_y_after - world_y_before) * v.zoom < k_eps);
                }
            }
        }
    }
}

TEST_CASE("canvas_view: zoom_by clamps to the band and stays anchored at the rail")
{
    canvas_view v;
    for (int i = 0; i < 200; ++i) v.zoom_by(1.5f, 100.0f, 100.0f); // hammer the top
    CHECK(v.zoom == doctest::Approx(canvas_view::max_zoom));

    // At the rail the anchor must still hold — a clamped zoom that shifted pan
    // would drift the view every wheel tick against the stop.
    const float world_x = v.to_world_x(100.0f);
    v.zoom_by(1.5f, 100.0f, 100.0f);
    CHECK(v.zoom == doctest::Approx(canvas_view::max_zoom));
    CHECK(std::fabs(v.to_world_x(100.0f) - world_x) * v.zoom < k_eps);

    for (int i = 0; i < 200; ++i) v.zoom_by(0.5f, 100.0f, 100.0f); // hammer the bottom
    CHECK(v.zoom == doctest::Approx(canvas_view::min_zoom));
}

TEST_CASE("canvas_view: a no-op zoom does not move the view")
{
    canvas_view v;
    v.zoom  = 2.0f;
    v.pan_x = 33.0f;
    v.pan_y = 44.0f;
    v.zoom_by(1.0f, 400.0f, 300.0f);
    CHECK(v.zoom == doctest::Approx(2.0f));
    CHECK(v.pan_x == doctest::Approx(33.0f));
    CHECK(v.pan_y == doctest::Approx(44.0f));
}

TEST_CASE("canvas_view: reset returns to the identity view")
{
    canvas_view v;
    v.zoom_by(2.0f, 500.0f, 500.0f);
    v.pan_x += 123.0f;
    REQUIRE(v.zoom != doctest::Approx(1.0f)); // the test would be vacuous otherwise
    v.reset();
    CHECK(v.zoom == doctest::Approx(1.0f));
    CHECK(v.pan_x == doctest::Approx(0.0f));
    CHECK(v.pan_y == doctest::Approx(0.0f));
}

TEST_CASE("canvas_view: panning is in world units, so a drag tracks the cursor 1:1")
{
    // What handle_view_input does: pan += pixel_delta / zoom. The resulting
    // on-screen motion of a fixed world point must equal the pixel delta at any
    // zoom -- that is the reason for the divisor.
    for (float zoom : {0.5f, 1.0f, 4.0f}) {
        canvas_view v;
        v.zoom                   = zoom;
        constexpr float world_pt = 250.0f;
        const float     before   = v.to_screen_x(world_pt);
        constexpr float drag_px  = 60.0f;
        v.pan_x += drag_px / zoom;
        CAPTURE(zoom);
        CHECK(v.to_screen_x(world_pt) - before == doctest::Approx(drag_px).epsilon(1e-4));
    }
}

TEST_CASE("canvas_view: clamp_pan collapses the range to {0} when the content fits")
{
    canvas_view v; // zoom 1
    v.pan_x = 42.0f;
    v.pan_y = -17.0f;
    // grid 400x300 inside a 600x400 viewport fits both axes: only pan 0 is legal, so
    // a small/unscrolled grid stays pinned exactly where it fits.
    v.clamp_pan(400.0f, 300.0f, 600.0f, 400.0f);
    CHECK(v.pan_x == doctest::Approx(0.0f));
    CHECK(v.pan_y == doctest::Approx(0.0f));
}

TEST_CASE("canvas_view: clamp_pan bounds pan to the scroll range when content overflows")
{
    canvas_view v; // zoom 1; grid 1000x1000 in a 600x400 viewport
    v.pan_x = 50.0f;   // past the left edge (positive pan shows empty space)
    v.clamp_pan(1000.0f, 1000.0f, 600.0f, 400.0f);
    CHECK(v.pan_x == doctest::Approx(0.0f)); // clamped up to 0

    v.pan_x = -999.0f; // past the right edge
    v.pan_y = -999.0f;
    v.clamp_pan(1000.0f, 1000.0f, 600.0f, 400.0f);
    CHECK(v.pan_x == doctest::Approx(-400.0f)); // viewport - grid = 600 - 1000
    CHECK(v.pan_y == doctest::Approx(-600.0f)); // 400 - 1000
    // At the far edge the scroll offset equals contents - visible.
    CHECK(v.scroll_x() == doctest::Approx(400.0f));
}

TEST_CASE("canvas_view: clamp_pan range scales with zoom")
{
    canvas_view v;
    v.zoom = 2.0f;
    // content = grid*zoom = 1000*2 = 2000 in a 600 viewport → pan_x ∈ [600/2 - 1000,
    // 0] = [-700, 0]; scroll at the edge = -pan*zoom = 1400 = 2000 - 600.
    v.pan_x = -5000.0f;
    v.clamp_pan(1000.0f, 500.0f, 600.0f, 400.0f);
    CHECK(v.pan_x == doctest::Approx(-700.0f));
    CHECK(v.scroll_x() == doctest::Approx(1400.0f));
}

TEST_CASE("canvas_view: scroll/pan is an exact round-trip")
{
    // scroll = -pan*zoom; the scrollbars speak scroll, the transform speaks pan.
    for (float zoom : {0.25f, 1.0f, 3.5f}) {
        canvas_view v;
        v.zoom = zoom;
        for (float s : {0.0f, 13.0f, 640.0f}) {
            v.set_scroll_x(s);
            v.set_scroll_y(s * 0.5f);
            CAPTURE(zoom);
            CAPTURE(s);
            CHECK(v.scroll_x() == doctest::Approx(s));
            CHECK(v.scroll_y() == doctest::Approx(s * 0.5f));
            CHECK(v.pan_x <= 0.0f); // a non-negative scroll maps to a non-positive pan
        }
    }
}

TEST_CASE("compute_reserve: no bars when the grid fits the viewport")
{
    const auto r = compute_reserve(100.0f, 100.0f, 1.0f, 200.0f, 200.0f, 14.0f);
    CHECK_FALSE(r.need_h);
    CHECK_FALSE(r.need_v);
    CHECK(r.avail_w == doctest::Approx(200.0f));
    CHECK(r.avail_h == doctest::Approx(200.0f));
}

TEST_CASE("compute_reserve: one overflowing axis yields one bar, reserved on the cross axis")
{
    const auto h = compute_reserve(300.0f, 50.0f, 1.0f, 200.0f, 200.0f, 14.0f);
    CHECK(h.need_h);
    CHECK_FALSE(h.need_v);
    CHECK(h.avail_w == doctest::Approx(200.0f)); // full width
    CHECK(h.avail_h == doctest::Approx(186.0f)); // minus the horizontal bar

    const auto v = compute_reserve(50.0f, 300.0f, 1.0f, 200.0f, 200.0f, 14.0f);
    CHECK(v.need_v);
    CHECK_FALSE(v.need_h);
    CHECK(v.avail_w == doctest::Approx(186.0f));
    CHECK(v.avail_h == doctest::Approx(200.0f));
}

TEST_CASE("compute_reserve: a bar on one axis can pull in the other (the two-step)")
{
    // Overflows horizontally; fits vertically by only 10 px < bar(14). The h-bar eats
    // 14 px of height, which then overflows vertically → the v-bar must appear too.
    // A single pass would miss this and clip the bottom row.
    const auto r = compute_reserve(300.0f, 190.0f, 1.0f, 200.0f, 200.0f, 14.0f);
    CHECK(r.need_h);
    CHECK(r.need_v);
    CHECK(r.avail_w == doctest::Approx(186.0f));
    CHECK(r.avail_h == doctest::Approx(186.0f));
}

TEST_CASE("compute_reserve: avail is floored at 1 for a sub-scrollbar viewport")
{
    const auto r = compute_reserve(100.0f, 100.0f, 1.0f, 10.0f, 10.0f, 14.0f);
    CHECK(r.need_h);
    CHECK(r.need_v);
    CHECK(r.avail_w == doctest::Approx(1.0f)); // 10 - 14 floored to 1, not negative
    CHECK(r.avail_h == doctest::Approx(1.0f));
}
