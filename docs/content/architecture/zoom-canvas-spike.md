+++
title = "Spike: zoom/scroll canvas"
weight = 30
+++

# ZoomScrollCanvas feasibility spike — results

This is a design record: the measured findings that shaped the canvas pan/zoom
implementation (`src/gui_plot/canvas.cpp`, PR #110). The spike program that
produced these numbers (`plot_zoom_toy/`, headless, null backend — no
SDL/GL/display) served its purpose and was removed; it lives in the git history
of PR #110. Run against the repo's pinned imgui 1.92.4 WIP (19231) /
implot 1.1 WIP.

Question: can a scroll/zoom grid canvas be built on `gui_plot`'s existing seams,
i.e. can hosted subplots receive a target dimension + target position and be cropped?

**Answer: yes. No blocker found. All four risks resolved in favour of the design.**

## E1 — cropping + culling (the load-bearing assumption): CONFIRMED

5×5 grid of 100×100 cells in a 400×300 viewport, pan (-150,-150).

| outcome | count |
|---|---|
| culled (empty clip, `SkipItems`) | 9 |
| partial (cropped at a viewport edge) | 10 |
| full | 6 |

`ImGui`'s computed clip rect equalled the predicted `intersect(cell_inset_by_border, viewport_clip)`
for **25/25** cells. Cells straddling an edge render their visible part; cells fully
outside get an empty clip and `BeginChild` returns false (still call `EndChild`).

Cropping and free-form positioning are therefore *already* what ImGui does — the
canvas needs no new mechanism, only a pan offset added to the cell origin.

## E2 — mid-frame scale reaches a later child: CONFIRMED (all paths)

Font size ratio vs baseline (13.0px), measured inside a child begun *after* the scale was set:

| path | ratio | FramePadding.x |
|---|---|---|
| `style.FontScaleMain = 2` (modern) | 2.00× | 4.0 |
| `io.FontGlobalScale = 2` (legacy; what the exporter uses) | 2.00× | 4.0 |
| `SetWindowFontScale(2)` inside child | 2.00× | 4.0 |
| exporter triple (`ScaleAllSizes` + `FontGlobalScale`) | 2.00× | 8.0 |

Only the exporter's triple also scales ImGui metrics (padding 4→8), which is what
makes an enlarged cell look like a real zoom rather than a stretched one.

Two notes:
- `io.FontGlobalScale` lives under `#ifndef IMGUI_DISABLE_OBSOLETE_FUNCTIONS`
  (imgui.h:2586 — "Moved to style.FontScaleMain in 1.92"). The exporter rode it
  at the time. Works, but new code should use `style.FontScaleMain`.
  *Since resolved:* the exporter now uses `style.FontScaleMain` too.
- The real GL backend sets `ImGuiBackendFlags_RendererHasTextures`
  (imgui_impl_opengl3.cpp:370) → 1.92 dynamic font atlas → zoomed text
  **re-rasterizes crisply** instead of being bitmap-stretched.

## E3 — ImPlot AutoFit across culled frames: NO FLASH

Data spans x∈[0,10]. Plot submitted, settled, then *not submitted* for `gap` frames
(what culling does), then resubmitted.

| gap (frames) | virgin xmax (control) | settled | 1st frame back | 2nd frame back |
|---|---|---|---|---|
| 1 | 1.000 | 10.000 | 10.000 | 10.000 |
| 5 | 1.000 | 10.000 | 10.000 | 10.000 |
| 60 | 1.000 | 10.000 | 10.000 | 10.000 |
| 300 | 1.000 | 10.000 | 10.000 | 10.000 |

The **control matters**: a virgin plot's first frame reads 1.000 (un-fitted), so the
test can distinguish retained state from in-frame refitting. It does — ImPlot keeps
per-plot fit state across at least 300 unsubmitted frames.

Culling off-screen cells is therefore free of the scroll-in refit flash that
`plot_capture.cpp:146-153` documents for one-shot off-screen captures.

## E4 — cell-size floor: PlotMinSize hypothesis REFUTED; real floor ≈ 46×38 px

`BeginPlot` succeeds at **every** size tested, 400×300 down to 8×6.
`PlotMinSize` (default 200×150) is **never consulted** for a canvas cell:

```cpp
// implot.cpp:2493-2496
if (frame_size.x < gp.Style.PlotMinSize.x && (size.x < 0.0f || gp.CurrentSubplot != nullptr))
    frame_size.x = gp.Style.PlotMinSize.x;
```

It clamps only for *auto* sizes (`size.x < 0`) or subplots. Cells pass an explicit
positive size (`plot_tools.cpp:158`), so they bypass it. Confirmed empirically:
scaling `PlotMinSize` to 10×7.5 (E4b) produced results **byte-identical** to the
default (E4a).

The real constraint is chrome overhead — plot interior = requested − ~46×38 px:

| requested | plot interior |
|---|---|
| 400×300 | 354×262 |
| 100×80 | 54×42 |
| 50×40 | 4×2 |
| 20×16 | **-5×-22** |
| 8×6 | **-17×-32** |

Below ~46×38 the interior goes **negative** and `BeginPlot` still returns true.
That overhead scales with the style scale, so the floor moves with zoom.

**Design consequence:** the canvas must skip `body_painter()` for cells below a
zoom-scaled threshold and draw a placeholder instead. This is a correctness guard
*and* the main perf win at zoom-out — it's the same branch either way.

## Caveats / not covered

- Headless: layout state only. Nothing here rasterizes, so this says nothing about
  visual quality — only about geometry, culling, fit state and size math.
- Input handling untested. `canvas::paint_body` is called from the live window
  **and** the off-screen exporter (`plot_capture.cpp` thumbnails canvases
  through the same seam), and the exporter does not reset `io.MousePos`. Pan/zoom
  reading the mouse inside `paint_body` needs an "interactive" gate.
  *Since resolved:* `region.interactive` threads that gate through the painter
  seam — the exporter paints with `interactive=false` and `paint_body` skips
  `handle_view_input` entirely (covered by
  `test_canvas_layout.cpp`, "a capture during a middle-drag must not corrupt
  the pan").
- Export semantics of a zoomed canvas (whole grid vs. current view): a decision,
  not a discovery.
- Cache interaction: none. The a2f2747 products key on data identity; pan/zoom
  touches neither `update()` nor `adopt_grid()`, and domains stay immutable.
