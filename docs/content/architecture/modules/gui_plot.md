+++
title = "gui_plot"
weight = 69
+++

# gui_plot — Live ImPlot Figures for Interpreted Code

The `gui_plot` module provides the plot kinds a [console]({{% relref "console" %}})
macro creates: `histogram`, `line`, `histogram2d`, `pixels`, `image`, and
`canvas`. A macro constructs one, calls `show()`, and a live ImPlot window
appears — updated from the console thread, drawn on the GUI thread, exportable
to PNG or HDF5.

The dependency runs **one way**: `gui_plot` lives in `coast_gui` and knows
nothing about the console, LLVM, or clang. `COAST_WITH_CONSOLE=OFF` yields an
LLVM-free build with `gui_plot` fully intact. (A headless build,
`COAST_WITH_GUI=OFF`, has neither — `gui_plot` is ImPlot-backed, so it goes with
the rest of the GUI layer.) The coupling is at *runtime* — an
interpreted macro `#include`s these headers — and that fact drives the module's
single hardest design constraint.

## The Constraint That Shapes Everything

> Every header a macro may include must be **ImGui-free**, because clang-repl
> parses it at runtime and the user pays that cost.

This is why `figure.h` defines `region` — a plain `{w, h}` POD — instead of
using `ImVec2`, and why all ImPlot and OpenGL code lives in the `.cpp` painters.
`plot_tools.h` is the **only** header in the module that includes ImGui/ImPlot,
and it is deliberately *not* reachable from `console_plot.h`.

It also explains the flat shape: **no factory, no type erasure, no plugin
registry** — unlike [gui_vis]({{% relref "gui_vis" %}}). A macro names a concrete
type and reaches its typed configuration directly. Indirection here would buy
nothing and cost parse time.

## Directory Layout

```
include/gui_plot/
├── plot_data.h        # Plain DTOs: range, zone, outer_ring, grid_result, h5_save_data
├── figure.h           # Base class, region, painter, plot_axis (ImGui-free)
├── plot_manager.h     # The singleton: registry, save queues, task hook, uid
├── plot_detail.h      # Module-internal glue: canvas_save_data, make_figure_from_h5
├── plot_h5.h          # HDF5 (de)serialization decls (the only TU pulling hdf5)
├── plot_tools.h       # Render helpers + theme layer — the ONLY ImGui-including header
├── histogram.h        # ─┐
├── line.h             #  │
├── histogram2d.h      #  ├─ one kind each
├── pixels.h           #  │
├── image.h            #  │
├── canvas.h           # ─┘
└── console_plot.h     # Umbrella: all kinds + free functions (what a macro includes)

src/gui_plot/          # matching implementations, plus:
├── plot_load.cpp      # the one kind-string → concrete-type map
└── plot_capture.cpp   # off-screen FBO export + the manager's capture methods
```

## figure — the Base and the Painter Seam

`coast::plot::figure` is non-copyable and inherits `enable_shared_from_this`.
Its API splits by thread:

```cpp
// any thread (internally locked)
set_title  set_theme  show  close  visible  point_count  save_png  save_h5
// GUI thread only
render  body_painter  free_gl  snapshot_h5
```

Four protected seams define a kind. Only the first is mandatory:

| Seam | Default | Role |
|---|---|---|
| `paint_body(region)` | **pure** | draw the plot; called with the mutex held and the theme pushed |
| `paint_controls()` | `{}` | extra chrome, live window only |
| `fill_snapshot(h5_save_data&)` | `false` | populate the plain-data snapshot; the base already set title/theme |
| `is_saveable()` | `false` | gates *both* `save_h5()` and the "Save H5…" button |

`body_painter()` returns a `painter` = `std::function<void(region)>` that
captures `shared_from_this()` — so **the closure pins the figure alive** — then
locks the mutex and pushes the theme before calling `paint_body`. That makes a
painter a self-contained drawing action that is safe to run later: the console
thread can be mid-`update()` while the GUI thread paints, and the painter still
sees consistent state. It is the module's whole cross-thread story.

## The Kinds

| Kind | Constructor | Distinctive API | Saveable |
|---|---|---|---|
| `histogram` | `(title, bins = 64)` or `(title, bins, range domain)` | `update(span)`, `domain()`, `underflow()`, `overflow()` | yes |
| `line` | `(title)` | `update(y)`, `update(x, y)` | yes |
| `histogram2d` | `(title, x_bins = 0, y_bins = 0[, x_dom, y_dom])` | `update(x, y)`, `outer()`, `set_palette`, `set_axis_visible`, `set_log_density`, `set_candle` | yes |
| `pixels` | `(title, width, height)` | `update(y)` row-major, row 0 = top; `set_palette`, `set_axis_visible` | yes |
| `image` | `(title)` | `set_png(path)`, `set_rgba(span, w, h)` | **no** |
| `canvas` | `(title, rows, cols)` | `place(fig, row, col)`, `clear_cell(row, col)` | yes (own path) |

Bins `<= 0` on `histogram2d` auto-pick via Sturges' rule. `update()` **re-tallies
rather than accumulates**. A figure placed into a `canvas` is *hosted* — do not
also `show()` it.

{{% callout info %}}
**Domains are immutable, fixed at construction.** Under/overflow tallies are
computed inside `update()`, so a settable domain would silently stale them. If
you want a different range, build a different figure.
{{% /callout %}}

## plot_manager — Registry, Reaping, Tombstone

The singleton owns the `hdf5::executor*`, the `gui_manager*` host, the figure
registry, three save queues, and the uid counter.

**The registry owns shown figures** (`vector<shared_ptr<figure>>`), which is why
a macro can `show()` a figure and drop the handle without it vanishing. The
reaping rule in `render_all` is exact:

```cpp
snap.clear();                                   // so use_count() == registry + macro only
erase_if(figs_, [](auto& f){ return !f->open() && f.use_count() == 1; });
```

Closed **and** nobody else holds it. A closed figure a macro still holds stays
hidden and is revived by the next `show()` — `update()` does not bring it back.

**`instance_` is a heap pointer, not a Meyers static**, so teardown happens at a
controlled point. `deinit()` deletes it and sets a process-wide **tombstone**;
`instance()` then throws `std::logic_error` forever after. The tombstone is not
defensive pedantry: a silent re-create would register a second `atexit` handler
and hand back a manager with `gui_host_ == nullptr`, whose deferred GL frees
would then run at atexit **with no GL context**. Failing loudly beats that.

`free_resources()` calls `free_gl()` on every figure *while the context is still
valid*. Dropping the registry's `shared_ptr`s is not enough — a macro may still
hold an `image`, and its `~gl_texture` would otherwise run on the console thread
after the context is gone.

The teardown order in `run_gui.cpp` is load-bearing: **console down first**
(joins the interpreter thread, so no macro can be mid-flight creating figures),
*then* `plot_manager::deinit()` while the GL context is still current and before
`gui`/`main_app` — which the manager references — leave scope.

## Persistence

`save_png()` and `save_h5()` only **enqueue**. `flush_captures(exec)` does the
work on the GUI thread, and `run_gui` registers it as a state func because:

{{% callout warning %}}
`flush_captures` must run **outside** an active ImGui frame. Each export runs its
own `NewFrame … Render` pass, and ImGui has a single global frame. Calling it
inside one corrupts that frame.
{{% /callout %}}

**PNG.** The exporter lazily loads GL 3.0 FBO entry points, saves and scales the
ImGui/ImPlot styles, builds a borderless window around `body_painter()`, renders
into a `GL_RGBA8` texture bound to an FBO at exactly the requested size, reads
the pixels back, forces alpha to 255, and flips vertically. **Exports may exceed
the on-screen window** because the FBO is sized from the *request*, not the
swapchain; the on-screen size only feeds `detail::export_scale` (clamped to
`[0.5, 8.0]`) so text doesn't come out pixel-native-tiny.

One non-obvious detail: the capture calls `build_frame()` **twice** — the first
is a throwaway priming frame, because ImPlot applies AutoFit the frame *after*
data submission. A single-frame capture rasterized the pre-fit 0..1 plot; that
was the "first save has the wrong scale, second one is fine" bug. Do not
optimize the extra frame away.

**HDF5.** `snapshot_h5()` fills an `h5_save_data` — a plain-data struct holding
**no GL or ImGui state** (kind, title, theme, colormap, grid + frame, axis
toggles, fixed domains, `outer`, raw `xs`/`ys`, and a thumbnail PNG). That purity
is what lets it be moved by value to an executor worker and written off the GUI
thread. Layout: one plot per file under `/plot` (`coast_plot_version = 1`); a
canvas under `/canvas` with a `cell_<r>_<c>` subgroup per occupied saveable cell.

Saveable: `histogram`, `line`, `histogram2d`, `pixels`, `canvas`. Not `image`.

**Why `histogram2d` persists its `outer` ring but `histogram` does not** — a
saved `histogram2d` holds only its **binned grid**, not its raw points, so the
ring cannot be recomputed on load and must travel with it. A 1D `histogram`
saves raw `ys` plus its domain, so `recount_outliers()` regenerates the tallies
on load. An all-zero ring is written as *empty*, because `outer_ring::empty()`
is `total() == 0`: "no ring" and "zero ring" read identically, and an
auto-ranged figure must not persist a ring it can never populate.

Loading is backward-compatible by construction: absent domain attrs → auto-ranged,
absent `outer` → empty ring, absent axis toggles → all visible. The reader probes
with `H5Aexists`/`H5Lexists` first, specifically so libhdf5 does not dump its
error stack on every load of an auto-ranged figure.

## Themes and Colormaps

Themes override ImPlot's **structural** colors only — plot/frame/legend
backgrounds, axis text, grid, ticks. Item colors live in `ImPlotSpec` in this
ImPlot version and are not themed. Two built-ins (`dark`, `bright`) cannot be
renamed or deleted; users add more via the in-GUI **Plot Theme Editor**, which
persists to `coast_plot_themes.json` in the working directory (next to the ImGui
`.ini`) and merges over the built-ins at load. `theme_scope` is the RAII
push/pop; an unknown name is a no-op.

Figures store a theme **by name**, so editing a theme retroactively restyles
every plot using it.

`colormap_t` is *not* defined here — it comes from `gui_vis/core/colormap.h` and
mirrors ImPlot's twelve **continuous** built-ins 1:1 and in ImPlot's order.
ImPlot's qualitative palettes (Deep/Dark/Pastel/Paired) are intentionally
excluded. Persisted by display name; unknown names fall back to `viridis`.

## Boundaries (read these before extending)

- **No ImGui in a macro-reachable header.** Adding one taxes every console
  session's parse time. `plot_tools.h` is the sanctioned exception and stays out
  of `console_plot.h`.
- **There is no "Plots" menu and no plot browser.** Figures are free-floating
  windows created by macros; `render_all()` is called unconditionally from
  `gui_manager::render()` and is a no-op when no macro has made one. The only
  in-GUI chrome is the per-figure theme/save row plus the shared theme editor.
- **Loading a saved plot has no GUI path.** `load()` / `load_canvas()` exist as
  API but nothing in `gui_core/`, `gui_tools/`, or `app/` calls them — loading is
  console-only (`coast::plot::load("f.h5")`). A saved plot is not reachable from
  the HDF5 Browser.
- **Deferred GUI tasks are not drained in `render_all()`** — that would run
  `glDeleteTextures` inside the ImGui frame. They run in the state phase via
  `gui_manager::defer`.
- **`set_executor` / `set_gui_host` are write-once at startup.** The lazy
  singleton init is lock-free and assumes a single-threaded first access.
- **Headless has no GUI host**, so `post_gui_task` falls back to a local queue
  drained by `free_resources()`.

## Tests

Executables under `tests/unit/gui_plot/`, all linking `coast_gui` — no LLVM. They
cover the **data and registry layer**; rendering needs a live GUI context and is
verified by hand. The whole suite is skipped in a headless build (`COAST_WITH_GUI=OFF`),
where `coast_gui` does not exist.

| Test | Covers |
|---|---|
| `test_console_plot.cpp` | load-without-executor → nullptr; figure and canvas H5 round-trips; create/update/count/clear; `image` registers on `show()`; canvas hosting; concurrent updates against a concurrent reader; axis visibility |
| `test_histogram_ranges.cpp` | `bin_points` auto-vs-fixed ranging, closed upper edge, invalid domain → auto; the outer-ring contract (out-of-range never lands in edge bins, corner counted once, ring conserves points); `update()` re-tallies; writer/reader agreement incl. auto-ranged writing no domain attrs and pre-domain files loading as auto-ranged |
| `test_plot_manager_deinit.cpp` | **Its own executable** — `deinit()` sets a *process-wide* tombstone, so sharing a process with cases that call `instance()` would poison them. Pins: live before deinit; `instance()` throws after; still throws on repeat; second `deinit()` is idempotent |

## See Also

- [console]({{% relref "console" %}}) — the interpreter whose macros drive these figures
- [gui_vis]({{% relref "gui_vis" %}}) — the *other* visualization system: registry-driven, for live sim entities
- [Threading]({{% relref "/threading" %}}) — the console thread, and the GUI-thread flush rule
