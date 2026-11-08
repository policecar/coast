+++
title = "gui_core"
weight = 70
+++

# gui_core — Application Framework

The `gui_core` module is the SDL2/ImGui application framework: the main window, event loop, menu system, browser windows, and editors.

## Directory Layout

```
include/gui_core/
├── sdl_app.h                  # SDL2 + OpenGL3 + ImGui lifecycle
├── gui_manager.h              # Main GUI orchestration (menus, dialogs)
├── entity_browser.h           # Entity table + visualization panel
├── buffer_browser.h           # IO buffer table + live stats
├── dataset_browser.h          # Thumbnail grid for dataset samples
├── dataset_access.h           # Shared dataset enumeration + sample→RGBA conversion
├── thumbnail_grid.h           # Reusable virtualized (selectable) thumbnail grid
├── config_editor.h            # JSON config editing GUI
├── network_editor.h           # Node-graph editor for simulation topology
├── logging_window.h           # Logging controls (buffer, probe, state)
└── imgui_scaling.h            # HiDPI scaling helper: scaled(base_value)

# The single-slot demand-driven sim→GUI publisher this module consumes,
# `demand_published<T>`, is GUI-free and lives in include/core/demand_published.h.

src/gui_core/
├── sdl_app.cpp                # SDL init, ImGui init, main loop, shutdown
├── gui_manager.cpp            # Menu bar, control dialog, status bar
├── entity_browser.cpp         # Sortable/filterable table, vis management
├── buffer_browser.cpp         # Buffer table with live statistics
├── dataset_browser.cpp        # Virtualized thumbnail rendering
├── dataset_access.cpp         # enumerate_datasets(), convert_sample_to_rgba()
├── thumbnail_grid.cpp         # Virtualized grid render + lazy texture cache
├── config_editor.cpp          # Full config editing UI (2300+ lines)
├── network_editor.cpp         # ImNodeFlow integration
└── logging_window.cpp         # Buffer/probe/state logging controls
```

## sdl_app

Manages the SDL2 + OpenGL lifecycle:

1. **Construction**: Init SDL2, create OpenGL context (3.0/3.2), create window (1280x800, maximized), init ImGui + ImPlot, load font with HiDPI scaling
2. **run()**: Main loop — poll events, state update callbacks, ImGui frame, GUI callbacks, render, swap
3. **Destruction**: Cleanup ImGui, destroy GL context, destroy window, SDL_Quit

Provides callback registration: `register_state_func()`, `register_gui_func()`, `register_shutdown_func()` — each returns an ID for later deregistration.

## gui_manager

Orchestrates all GUI windows and dialogs. Holds:

**Browsers**: `entity_browser`, `buffer_browser`, `dataset_browser` (heap-allocated, recreated on sim load)

**Panels**: `logging_window`, `probe_browser`, `observer_panel`, `buffer_replay_dialog` (value members)

**Editors**: `config_editor` (heap-allocated)

**Tools**: `metric_explorer` (heap-allocated, **sim-independent**) — created **once** in the constructor and **not** recreated on sim load. It owns its own imported datasets (see [gui_tools]({{% relref "gui_tools" %}})). Its GPU resources are released at shutdown via `free_resources()`.

**Analysis**: `h5_browser_window_` (heap-allocated, **sim-independent**) — the [HDF5 Browser]({{% relref "/architecture/modules/h5browse" %}}) panel, following the same lifecycle pattern as `metric_explorer`: created once in the constructor (takes the shared `hdf5::executor&`), not recreated on sim load. Unlike `metric_explorer` it owns no GPU textures, so it is not part of `free_resources()`.

### Menu Structure

| Menu | Items | Visible |
|------|-------|---------|
| **System** | Open Configuration (Ctrl+O), Config Editor, Save/Load State (Ctrl+S/L), Exit | always |
| **Simulation** | Control, Status | sim loaded |
| **Introspection** | Sim Entities, IO Buffers, Dataset, Logging, Performance, Probe Browser, Observer Capture | sim loaded |
| **Tools** | Metric Explorer | always |
| **Analysis** | Buffer Replay, HDF5 Browser | always |

The **Tools** and **Analysis** menus are independent of the loaded simulation and
are always present; **Simulation** and **Introspection** appear only while a sim is
loaded.

### set_sim_instance()

Called when a new config is loaded. Frees old browser GL resources, creates new browsers, initializes probe system, creates config editor with reload callback, checks for existing snapshots and offers resume. It deliberately **does not** touch the persistent, sim-independent tools (`metric_explorer`).

### free_resources()

Frees GPU resources owned by persistent (sim-independent) components — currently the `metric_explorer`. Called from `run_gui`'s shutdown hook, while the GL context is still current, before it is torn down (browsers are instead freed through `set_sim_instance(nullptr)`).

### Auto-save

Detects running→paused transitions. Saves state to `<snapshot_dir>/<config>_<timestamp>_<step>.h5`. Snapshot directory is configurable per-config JSON. Like every runtime state save, it runs through the quiesced split: `try_extract_state` via `run_quiesced` (it refuses a halted run and returns an extraction throw as an error, so a failed auto-save is skipped rather than ending the GUI), then `serialize_state` on the executor (see [Logging Pipeline]({{% relref "/data-flow/logging-pipeline" %}})).

### Status dialog

The per-entity status lines are computed by `io_entity::status_str()` **on the sim thread** inside a gated `post_swap` hook (the values it formats — dataset cursors, mismatch aggregates — are live sim state) and published through a `demand_published` slot the dialog consumes each frame. The hook exists only while the dialog is open.

## Entity Browser

Sortable, filterable table of all simulation entities. Features:
- Column sorting (ID, Name, Type, Config, Output Size, Input Count)
- Substring filter bar
- **Detail panel** with draggable splitter (ratio 0.2–0.9)
- Visualization checkboxes from `vis_registry` — toggle creates/destroys `vis_instance`
- Active visualizations update every frame, even when browser is hidden

## Buffer Browser

Same pattern as entity browser but for IO buffers. Shows live statistics (sum, avg, min, max) and a detail panel with buffer visualization checkboxes.

The stats are **not** read from the live buffers on the GUI thread — that would race `swap_buffer()`. While the window is open, a gated `post_swap` hook computes them on the sim thread (using `read_back_sync()` for GPU-only buffers, which therefore show real numbers) and publishes them through a `demand_published` slot the GUI consumes each frame; the hook is removed when the window closes. See [Threading]({{% relref "/threading" %}}) §4b for the mechanism.

## Dataset Browser

A thin view-only consumer of the shared dataset helpers below: a dataset selector
combo plus a `thumbnail_grid`. Features:
- Dataset selector combo (over `enumerate_datasets`)
- 64x64 thumbnails via nearest-neighbor resampling
- Lazy texture cache, virtualized scrolling (only visible rows + 2-row buffer)
- Hover tooltip with sample index

## Shared Dataset Helpers

Two helpers, extracted so both the Dataset Browser and the Metric Explorer share a
single code path instead of repeating the dataset switch and grid logic:

- **`dataset_access`** — `enumerate_datasets(sim::env&, const sim::sim_builder&)` is
  the one true `io_entity_type` → resolved-handle switch (geometry + a type-erased
  `get_sample` accessor; includes the `color_mnist` special case). The returned
  `dataset_handle` is valid for the lifetime of the `sim::env` it came from (a
  `sim_instance` or a `dataset_loader`). `convert_sample_to_rgba()` resamples one
  normalized sample into a thumbnail (magenta-fills malformed pixels so errors are
  visible).
- **`thumbnail_grid`** — a reusable virtualized thumbnail grid with a lazy texture
  cache. `render(handle, sel)` is view-only when `sel == nullptr`, or
  click-to-toggle with highlight when given a selection set (the Metric Explorer's
  sample picker uses the latter).

## Config Editor

Full editing UI for `global_cfg_t` with sections for datasets, NGM2, simulation graph, logging, and vis hints. Features dirty tracking, save/restore, and a reload callback.

## Network Editor

Graphical node-graph editor built on ImNodeFlow. Config editor is the single source of truth — user edits are validated through callbacks before being applied.

## Logging Window

See [logging module]({{% relref "logging" %}}) for the underlying system. The window provides:
- Buffer logging controls (path, start/stop, status)
- State save/load with file browsers
- Auto-checkpoint with configurable interval (deterministic sim-thread cadence, like headless `--checkpoint`)

## imgui_scaling

`scaled(base_value)` = `base_value * (ImGui::GetFontSize() / 16.0f)`. Ensures all pixel sizes scale with font size for HiDPI support.
